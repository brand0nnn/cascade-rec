import numpy as np
import pytest
import torch

from cascade.models.graph_latent_ranker import GraphLatentRanker
from cascade.pipelines.graph_family_study import (
    ARMS, COREVIEW_FEATURES, META_FEATURES, compare_control, enabled_names, mask_features,
)

NAMES=['log_other_reviewers','log_witness_paths','normalised_path_strength',
       'same_leaf_history_fraction','same_store_history_fraction','supported_history_fraction']


def test_masks_apply_to_named_standardised_columns_without_mutating_common_or_source():
    graph=np.arange(24,dtype=np.float32).reshape(2,2,6)-10
    features={'common':np.ones((2,2,9),np.float32),'graph':graph.copy()}
    for arm in ARMS:
        masked=mask_features(features,arm,NAMES)
        assert masked['common'] is features['common']
        for column,name in enumerate(NAMES):
            expected=graph[...,column] if name in enabled_names(arm,NAMES) else np.zeros((2,2))
            assert np.array_equal(masked['graph'][...,column],expected)
    assert np.array_equal(features['graph'],graph)
    assert enabled_names('GM',NAMES)==list(META_FEATURES)
    assert set(enabled_names('GC',NAMES))==set(COREVIEW_FEATURES)


def test_manifest_order_is_respected_and_changed_feature_contract_fails_closed():
    reversed_names=NAMES[::-1]
    f={'common':np.ones((1,1,9)), 'graph':np.arange(1,7)[None,None,:]}
    actual=mask_features(f,'GM',reversed_names)['graph'][0,0]
    assert actual.tolist()==[0,2,3,0,0,0]
    with pytest.raises(ValueError,match='manifest'):enabled_names('GM',NAMES[:-1]+['new_feature'])
    with pytest.raises(ValueError,match='unregistered'):enabled_names('GX',NAMES)
    with pytest.raises(ValueError,match='width'):mask_features({**f,'graph':f['graph'][...,:5]},'GM',NAMES)


def make_model(arm):
    rng=np.random.default_rng(5)
    text=rng.normal(size=(6,384)).astype(np.float32);text[0]=0
    bank=rng.normal(size=(6,1154)).astype(np.float32);bank[0]=0
    factors=np.zeros((6,64),np.float32);known=np.zeros(6,bool)
    torch.manual_seed(23)
    result=GraphLatentRanker(text,np.array([0,1,2,1,2,3]),np.array([0,1,1,2,3,2]),
        bank,factors,known,9,6,arm).eval()
    with torch.no_grad(): result.head[-1].weight.fill_(.05)
    return result


def test_same_r2_model_reproduces_prior_r1_and_r2_functions_beyond_epoch_zero():
    r1=make_model('R1');r2=make_model('R2');fresh=make_model('R2')
    candidates=torch.tensor([[1,3],[2,4]])
    history=torch.tensor([[0,2,5],[0,1,3]])
    scores=torch.tensor([[.8,.7],[.6,.5]])
    f={'common':np.ones((2,2,9),np.float32),'graph':np.arange(24,dtype=np.float32).reshape(2,2,6)-8}
    for arm,reference in [('G0',r1),('GMC',r2)]:
        masked=mask_features(f,arm,NAMES)
        actual=fresh(candidates,history,scores,torch.as_tensor(masked['common']),torch.as_tensor(masked['graph']))
        expected=reference(candidates,history,scores,torch.as_tensor(f['common']),torch.as_tensor(f['graph']))
        assert torch.equal(actual,expected)
        assert all(torch.equal(value,reference.state_dict()[name]) for name,value in fresh.state_dict().items())


def test_ablated_family_cannot_change_model_inputs():
    f={'common':np.ones((1,2,9),np.float32),'graph':np.ones((1,2,6),np.float32)}
    for arm,disabled in [('GM',COREVIEW_FEATURES),('GC',META_FEATURES)]:
        changed={'common':f['common'],'graph':f['graph'].copy()}
        for i,name in enumerate(NAMES):
            if name in disabled:changed['graph'][...,i]+=1000
        assert np.array_equal(mask_features(f,arm,NAMES)['graph'],mask_features(changed,arm,NAMES)['graph'])


def curves():
    return [{'epoch':0,'loss':None,'end_to_end_ndcg10':.01,'unseen_before_T0_ndcg10':.008},
            {'epoch':1,'loss':.63,'end_to_end_ndcg10':.02,'unseen_before_T0_ndcg10':.03}]


def test_control_reproduction_obeys_loss_tolerance_and_requires_exact_ranking_selection():
    reference=curves();actual=curves()
    actual[1]['loss']+=1e-7
    assert compare_control(actual,1,reference,1)
    actual[1]['loss']+=1e-3
    assert not compare_control(actual,1,reference,1)
    actual=curves();actual[1]['end_to_end_ndcg10']+=1e-12
    assert not compare_control(actual,1,reference,1)
    assert not compare_control(reference,0,reference,1)
    assert not compare_control(reference[:1],0,reference,1)
