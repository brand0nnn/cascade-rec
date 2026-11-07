from types import SimpleNamespace

import json
import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from cascade.pipelines.graph_latent_factors import BPRFactors,PreT0NegativeSampler,prepare_pca,train_bpr


def tiny_data():
    observations=pd.DataFrame({'user':[1,1,2,2,3,4], 'item':[1,2,2,3,4,5], 'ts':[1,2,1,3,1,20]})
    calls=[]
    def interactions(periods):
        assert periods==['retriever_train'],'validation or Test was requested'
        calls.append(tuple(periods))
        return observations[observations.ts<10].copy()
    return SimpleNamespace(n_items=7,n_users=4,split={'t_ranker_train':10},
        interactions=interactions,calls=calls,
        items=pd.DataFrame({'item':np.arange(1,8),'first_ts':[1,1,3,1,20,30,40],'text':['x']*7}))


def test_pca_fits_only_pre_T0_and_transforms_new_products_without_refit(tmp_path):
    rng=np.random.default_rng(0);text=rng.normal(size=(71,72)).astype(np.float32);text[0]=0;text[70]=0
    data=SimpleNamespace(n_items=70,split={'t_ranker_train':10},
        items=pd.DataFrame({'item':np.arange(1,71),'first_ts':[1]*66+[20]*4,'text':['x']*68+['','x']}))
    a=tmp_path/'a';b=tmp_path/'b';prepare_pca(data,text,a)
    changed=text.copy();changed[67:]*=100
    prepare_pca(data,changed,b)
    with np.load(a/'pca_state.npz') as x,np.load(b/'pca_state.npz') as y:
        assert np.array_equal(x['mean'],y['mean']) and np.array_equal(x['components'],y['components'])
        assert x['fit_item_ids'].tolist()==list(range(1,67))
    va,vb=np.load(a/'pca_vectors.npy'),np.load(b/'pca_vectors.npy')
    known=np.load(a/'pca_known.npy')
    assert np.array_equal(va[:67],vb[:67]) and not np.array_equal(va[67:69],vb[67:69])
    assert known[67:69].all() and not known[[0,69,70]].any()
    assert not va[[0,69,70]].any() and np.allclose(np.linalg.norm(va[known],axis=1),1,atol=1e-6)
    proof=json.loads((a/'provenance.json').read_text())
    assert proof['status']=='complete' and proof['validation_calls']==0 and proof['reload_vectors_exact']
    with pytest.raises(FileExistsError):prepare_pca(data,text,a)


def test_pca_rejects_insufficient_pre_T0_support_before_creating_output(tmp_path):
    data=tiny_data()
    with pytest.raises(ValueError,match='at least 64'):prepare_pca(data,np.ones((8,72),np.float32),tmp_path/'missing')
    assert not (tmp_path/'missing').exists()


def test_negative_sampler_excludes_all_user_positives_and_future_items():
    users=np.array([1,1,2,2,3]);items=np.array([1,2,2,3,4]);pool=np.arange(1,5)
    a=PreT0NegativeSampler(users,items,pool,17);b=PreT0NegativeSampler(users,items,pool,17)
    query=np.tile([1,2,3],1000);x,y=a.sample(query),b.sample(query)
    assert np.array_equal(x,y) and np.isin(x,pool).all()
    for u,i in zip(query,x):assert int(i) not in a.used[int(u)]
    assert a.returned==3000 and len(a.first_negatives)==1024
    with pytest.raises(ValueError,match='exhausts'):PreT0NegativeSampler([1,1],[1,2],[1,2],0)
    with pytest.raises(ValueError,match='unknown training'):a.sample([99])


def test_bpr_loss_is_pairwise_and_unknown_rows_get_no_gradient():
    known_items=np.array([False,True,True,True,False]);known_users=np.array([False,True,False])
    model=BPRFactors(2,4,dim=2,seed=0,known_users=known_users,known_items=known_items)
    users=torch.tensor([1]);positive=torch.tensor([1]);negative=torch.tensor([2])
    expected=F.softplus(model(users,negative)-model(users,positive)).mean()
    actual=model.loss(users,positive,negative)
    assert torch.equal(actual,expected)
    actual.backward()
    assert not model.item_embedding.weight.grad[4].any() and not model.user_embedding.weight.grad[2].any()
    assert not model.item_embedding.weight[[0,4]].any() and not model.user_embedding.weight[[0,2]].any()


def test_bpr_initialisation_is_reproducible_and_ignores_unknown_vocabulary_size():
    state=torch.get_rng_state()
    a=BPRFactors(2,4,seed=23,known_users=[0,1,0],known_items=[0,1,1,1,0])
    assert torch.equal(state,torch.get_rng_state())
    b=BPRFactors(4,7,seed=23,known_users=[0,1,0,0,0],known_items=[0,1,1,1,0,0,0,0])
    assert torch.equal(a.user_embedding.weight[1],b.user_embedding.weight[1])
    assert torch.equal(a.item_embedding.weight[1:4],b.item_embedding.weight[1:4])


def test_bpr_smoke_exports_fixed_state_and_never_requests_validation(tmp_path):
    data=tiny_data();a=tmp_path/'a';b=tmp_path/'b'
    train_bpr(data,a,seed=7,device='cpu',smoke=True)
    train_bpr(data,b,seed=7,device='cpu',smoke=True)
    assert data.calls==[('retriever_train',),('retriever_train',)]
    for name in ('bpr_vectors.npy','bpr_known.npy','bpr_known_users.npy'):
        assert np.array_equal(np.load(a/name),np.load(b/name))
    vectors=np.load(a/'bpr_vectors.npy');known=np.load(a/'bpr_known.npy')
    assert known.tolist()==[False,True,True,True,True,False,False,False]
    assert not vectors[~known].any() and np.allclose(np.linalg.norm(vectors[known],axis=1),1,atol=1e-6)
    metrics=json.loads((a/'metrics.json').read_text());audit=json.loads((a/'sampler_audit.json').read_text())
    assert metrics['status']=='complete' and metrics['epochs']==1 and metrics['interaction_count']==5
    assert metrics['includes_first_user_interactions'] and metrics['validation_calls']==0 and not metrics['validation_used']
    assert metrics['reload_max_weight_difference']==0 and metrics['reload_scores_exact'] and metrics['reload_vectors_exact']
    assert audit['negatives_returned']==5 and audit['future_negatives']==audit['user_positive_negatives']==0
    sa=torch.load(a/'model.pth',weights_only=True);sb=torch.load(b/'model.pth',weights_only=True)
    assert all(torch.equal(sa[k],sb[k]) for k in sa)
    assert not list(a.glob('*val*')) and not list(a.glob('*test*'))
    with pytest.raises(FileExistsError):train_bpr(data,a,seed=99,device='cpu',smoke=True)
    assert len(data.calls)==2
