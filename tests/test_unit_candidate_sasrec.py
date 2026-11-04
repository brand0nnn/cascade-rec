from types import SimpleNamespace

import numpy as np
import torch
from recbole.config import Config
from torch.nn import functional as F

from cascade.models.local_sasrec import ContentSASRec, UnitCandidateSASRec
from cascade.models.local_variants import LocalSASRecScorer
from cascade.models.recbole_baselines import SASRecBCE,recbole_config_dict


def model_pair(features):
    cfg=recbole_config_dict('sequential','.','local_test')
    cfg.update(n_layers=2,n_heads=1,hidden_size=64,inner_size=64,hidden_dropout_prob=.2,
        attn_dropout_prob=.2,hidden_act='relu',layer_norm_eps=1e-8,initializer_range=.02,loss_type='BPR',seed=0)
    cfg=Config(model=SASRecBCE,dataset='local_test',config_dict=cfg);cfg['device']=torch.device('cpu')
    dataset=SimpleNamespace(num=lambda field:7,token2id=lambda field,tokens:np.asarray(tokens,dtype=np.int64))
    torch.manual_seed(0);a=ContentSASRec(cfg,dataset,features).eval()
    torch.manual_seed(0);b=UnitCandidateSASRec(cfg,dataset,features).eval()
    return a,b,dataset


def batch(model):
    return {model.ITEM_SEQ:torch.tensor([[1,2,0],[2,0,0]]),model.ITEM_SEQ_LEN:torch.tensor([2,1]),
        model.ITEM_ID:torch.tensor([3,1]),model.POS_ITEM_ID:torch.tensor([3,1]),model.NEG_ITEM_ID:torch.tensor([2,3])}


def test_candidate_normalisation_keeps_initial_parameters_and_history_forward_identical():
    a,b,_=model_pair(np.random.default_rng(0).normal(size=(7,384)).astype(np.float32))
    assert a.state_dict().keys()==b.state_dict().keys()
    assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
    x=batch(b)
    assert torch.equal(a.item_embedding(x[b.ITEM_SEQ]),b.item_embedding(x[b.ITEM_SEQ]))
    assert torch.equal(a.forward(x[b.ITEM_SEQ],x[b.ITEM_SEQ_LEN]),b.forward(x[b.ITEM_SEQ],x[b.ITEM_SEQ_LEN]))


def test_loss_native_prediction_and_shared_scorer_use_identical_unit_candidates():
    _,b,dataset=model_pair(np.random.default_rng(0).normal(size=(7,384)).astype(np.float32));x=batch(b)
    query=b.forward(x[b.ITEM_SEQ],x[b.ITEM_SEQ_LEN])
    candidates=F.normalize(b.item_embedding.weight,dim=-1)
    full=query@candidates.T
    assert torch.allclose(b.full_sort_predict(x),full)
    assert torch.allclose(b.predict(x),full[torch.arange(2),x[b.ITEM_ID]])
    expected=F.binary_cross_entropy_with_logits(full[torch.arange(2),x[b.POS_ITEM_ID]],torch.ones(2))
    expected+=F.binary_cross_entropy_with_logits(full[torch.arange(2),x[b.NEG_ITEM_ID]],torch.zeros(2))
    assert torch.allclose(b.calculate_loss(x),expected)
    scorer=LocalSASRecScorer(b,dataset,6)
    with torch.no_grad():
        actual=scorer.score_all(SimpleNamespace(hist=torch.tensor([[0,1,2],[0,0,2]])))
    assert torch.allclose(actual,full,atol=1e-6)
    assert torch.allclose(candidates[1:].norm(dim=-1),torch.ones(6)) and not candidates[0].any()


def test_future_features_do_not_change_normalised_training_loss_or_gradients():
    features=np.random.default_rng(0).normal(size=(7,384)).astype(np.float32)
    altered=features.copy();altered[4:]*=100
    _,a,_=model_pair(features);_,b,_=model_pair(altered)
    la,lb=a.calculate_loss(batch(a)),b.calculate_loss(batch(b))
    assert torch.equal(la,lb)
    la.backward();lb.backward()
    assert all(torch.equal(pa.grad,pb.grad) for pa,pb in zip(a.parameters(),b.parameters()))
