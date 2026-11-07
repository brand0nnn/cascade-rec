from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from cascade.models.graph_latent_ranker import GraphLatentRanker
from cascade.pipelines.ranking_objective_study import (
    cold_mask, gate_decision, gradient_summary, inner_metrics, objective_loss,
    probe_rows, probe_statistics, train_epoch,
)


def test_pairwise_loss_uses_positive_column_and_cancels_query_offsets():
    logits=torch.tensor([[2.,1.,0.],[0.,3.,1.]],dtype=torch.float64,requires_grad=True)
    expected=F.softplus(logits[:,1:]-logits[:,:1]).mean()
    assert torch.equal(objective_loss(logits,'pairwise'),expected)
    shifted=logits+torch.tensor([[12.],[-8.]],dtype=torch.float64)
    assert objective_loss(shifted,'pairwise').item()==pytest.approx(expected.item(),abs=1e-14)
    expected.backward()
    assert torch.allclose(logits.grad.sum(1),torch.zeros(2,dtype=torch.float64),atol=1e-15)
    better=logits.detach().clone();better[:,0]+=1
    assert objective_loss(better,'pairwise')<expected


def test_bce_control_matches_original_and_nonfinite_candidates_fail_closed():
    logits=torch.tensor([[1.,0.,-1.]])
    expected=F.binary_cross_entropy_with_logits(logits,torch.tensor([[1.,0.,0.]]))
    assert torch.equal(objective_loss(logits,'bce'),expected)
    assert objective_loss(logits+4,'bce')!=expected
    with pytest.raises(ValueError,match='finite'):objective_loss(torch.tensor([[1.,-torch.inf]]),'pairwise')
    with pytest.raises(ValueError,match='unknown'):objective_loss(logits,'unregistered')


def test_probe_is_fixed_chronologically_before_outcomes_and_not_only_tuning():
    instances=pd.DataFrame({'inst':[5,4,3,2,1,0],'ts':[3,1,1,8,7,7]})
    rows,stage=probe_rows(instances,np.array([True,True,True,False,False,False]),np.array([True,False,True,True,True,True]),2)
    assert rows.tolist()==[2,0,5,4]
    assert stage.tolist()==['prefix','prefix','tune','tune']


def test_probe_separates_harmless_offsets_from_candidate_margin_changes():
    # Use 12 alternatives for an actual top-ten boundary, retaining padding.
    candidates=np.tile(np.r_[np.arange(1,14),0],(2,1))
    base=np.tile(np.r_[np.arange(13,0,-1,dtype=float),-np.inf],(2,1))
    shifted=base+np.array([[-4.],[2.]])
    targets=np.array([1,2]);stages=np.array(['prefix','tune'])
    rows,means,centered=probe_statistics(shifted,base,candidates,targets,stages)
    assert means.tolist()==[-4.,2.] and not centered.any()
    original=probe_statistics(base,base,candidates,targets,stages)[0]
    assert [r['positive_margin_mean'] for r in rows]==pytest.approx([r['positive_margin_mean'] for r in original])
    assert [r['top10_boundary_margin_mean'] for r in rows]==[r['top10_boundary_margin_mean'] for r in original]
    changed=shifted.copy();changed[:,0]+=2
    assert probe_statistics(changed,base,candidates,targets,stages)[0][0]['centered_residual_std']>0
    with pytest.raises(ValueError,match='natural target'):probe_statistics(base,base,candidates,np.array([99,2]),stages)


def test_gate_requires_three_seeds_cold_guard_and_two_individual_improvements():
    assert gate_decision(.01,[.011,.012,.01],.02,[.021,.02,.02])
    assert not gate_decision(.01,[.011,.01,.01],.02,[.021,.02,.02])
    assert not gate_decision(.01,[.011,.012,.01],.02,[.019,.019,.019])
    assert not gate_decision(.01,[.01,.01,.01],.02,[.02,.02,.02])
    with pytest.raises(ValueError):gate_decision(.01,[.011,.012],.02,[.02,.02])
    with pytest.raises(ValueError):gate_decision(.01,[.011,.012,.01],np.nan,[.02,.02,.02])


def test_cold_gate_metrics_keep_retrieval_misses_and_use_user_macro():
    data=SimpleNamespace(items=pd.DataFrame({'item':[1,2,3],'first_ts':[5,10,15]}),split={'t_ranker_train':10})
    instances=pd.DataFrame({'item':[1,2,3]})
    assert cold_mask(data,instances).tolist()==[False,True,True]
    metrics=inner_metrics(np.array([1.,np.inf,1.]),np.array([1,1,2]),np.array([False,True,True]),np.array([True,False,True]))
    assert metrics['end_to_end_ndcg10']==pytest.approx(.75)
    assert metrics['unseen_before_T0_ndcg10']==pytest.approx(.5)
    assert metrics['conditional_ndcg10']==1


def model_and_inputs():
    rng=np.random.default_rng(8);n=52
    text=rng.normal(size=(n,384)).astype(np.float32);text[0]=0
    bank=np.zeros((n,1154),np.float32);bank[:,:384]=text
    factors=np.zeros((n,64),np.float32);known=np.zeros(n,bool)
    torch.manual_seed(4)
    model=GraphLatentRanker(text,np.zeros(n,int),np.zeros(n,int),bank,factors,known,9,6,'R1')
    candidates=np.arange(1,52)[None,:]
    archive={'cand':candidates,'score':np.linspace(.9,.6,51,dtype=np.float32)[None,:],'target':np.array([1])}
    feature={'common':np.zeros((1,51,9),np.float32),'graph':np.zeros((1,51,6),np.float32)}
    return model,archive,feature,np.array([[1,2]])


def test_real_train_step_records_active_gradient_flow_and_inactive_branches():
    model,archive,feature,history=model_and_inputs()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    for epoch in [1,2]:
        loss,logs=train_epoch(model,optimizer,archive,feature,history,np.array([0]),np.arange(51)[None,:],
                              'cpu','pairwise',epoch,'inner')
        assert np.isfinite(loss) and logs[0]['head_grad_l2']>0
        assert not logs[0]['latent_active'] and not logs[0]['graph_active']
        assert logs[0]['final_bias_grad_l2']<1e-6
    assert gradient_summary(model)['text_grad_l2']>0
    assert model.head[-1].weight.abs().sum()>0


def test_train_step_rejects_wrong_positive_or_duplicate_alternatives():
    model,archive,feature,history=model_and_inputs();optimizer=torch.optim.Adam(model.parameters())
    wrong=np.arange(51)[None,:].copy();wrong[0,[0,1]]=wrong[0,[1,0]]
    with pytest.raises(ValueError,match='alignment'):
        train_epoch(model,optimizer,archive,feature,history,np.array([0]),wrong,'cpu','bce',1,'inner')
    duplicate=np.arange(51)[None,:].copy();duplicate[0,2]=1
    with pytest.raises(ValueError,match='duplicate'):
        train_epoch(model,optimizer,archive,feature,history,np.array([0]),duplicate,'cpu','bce',1,'inner')
