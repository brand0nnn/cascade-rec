import numpy as np
import pytest
import torch

from cascade.models.graph_latent_ranker import GraphLatentRanker
from cascade.pipelines.steps_final import expansion, rank_top
from cascade.pipelines.steps_final_analysis import composition, funnel, groups


def test_ranks_use_pessimistic_ties_but_display_ids_and_ignore_padding():
    cand=np.array([[9,3,0],[8,2,0]])
    ranks,top=rank_top(np.array([[1.,1.,999.],[5.,3.,999.]]),cand,np.array([9,7]),True)
    np.testing.assert_array_equal(ranks,[2,np.inf])
    np.testing.assert_array_equal(top[:,:3],[[3,9,0],[8,2,0]])


def test_k_misses_receive_no_end_to_end_credit():
    cand=np.array([[1,2,3]])
    r,_=rank_top(np.array([[.1,.2]]),cand[:,:2],np.array([3]))
    assert np.isinf(r[0])
    r,_=rank_top(np.array([[.1,.2,.3]]),cand,np.array([3]))
    assert r[0]==1


def test_expansion_separates_new_entry_and_competition_with_zeroes():
    a=np.array([0.,1.,0.]); b=np.array([.5,.2,0.])
    entry,competition=expansion(a,b,np.array([False,True,False]),np.array([True,True,False]))
    np.testing.assert_allclose(entry,[.5,0,0])
    np.testing.assert_allclose(competition,[0,-.8,0])
    np.testing.assert_allclose(entry+competition,b-a)
    assert entry.mean()==pytest.approx(.5/3)  # full denominator, not just entrants


def test_expansion_rejects_non_nested_candidate_pools():
    with pytest.raises(ValueError,match='nested'):
        expansion(np.zeros(1),np.zeros(1),np.array([True]),np.array([False]))


def test_pointwise_ranker_logits_are_invariant_to_candidate_budget():
    torch.manual_seed(4)
    text=np.random.default_rng(0).normal(size=(12,384)).astype(np.float32); text[0]=0
    labels=np.arange(12)
    model=GraphLatentRanker(text,labels,labels,np.zeros((12,1154),np.float32),
        np.zeros((12,64),np.float32),np.zeros(12,bool),9,6,'R2').eval()
    torch.nn.init.normal_(model.head[-1].weight)
    c=torch.tensor([[4,5,6,7]]); h=torch.tensor([[0,1,2,3]])
    score=torch.randn(1,4); common=torch.randn(1,4,9); graph=torch.randn(1,4,6)
    with torch.no_grad():
        full=model(c,h,score,common,graph)
        half=model(c[:,:2],h,score[:,:2],common[:,:2],graph[:,:2])
    torch.testing.assert_close(full[:,:2],half,atol=2e-6,rtol=2e-6)


def test_funnel_partitions_hits_and_misses():
    f=funnel(np.array([np.inf,20,20,3,3]),np.array([np.inf,30,2,30,1]),np.ones(5,bool))
    assert all(f[name]==1 for name in ('retrieval_miss','unrescued','rescued','demoted','retained'))
    assert f['top10_hits']==2


def test_composition_excludes_unknown_metadata_from_hhi():
    top=np.array([[1,2,3,0],[4,0,0,0]])
    counts=np.array([0,5,2,0,0]); head=np.array([False,True,False,False,False])
    first=np.array([999,0,0,10,10]); leaf=np.array([0,1,1,0,0]); store=np.array([0,1,2,0,0])
    r=composition(top,np.array([1,2]),counts,head,first,leaf,store,5)
    assert r['leaf_hhi']==1 and r['store_hhi']==.5
    assert r['leaf_supported_lists']==1 and r['leaf_known_slots']==2
    assert r['head_slot_share']==.25 and r['unseen_preT0_slot_share']==.5
    assert r['catalogue_products']==4


def test_new_item_definition_moves_to_test_start():
    from types import SimpleNamespace
    import pandas as pd
    data=SimpleNamespace(items=pd.DataFrame({'item':[1,2,3],'first_ts':[5,15,25]}),split={'t1':10,'t2':20})
    inst=pd.DataFrame({'item':[1,2,3]})
    np.testing.assert_array_equal(groups(data,inst,'val')['new'],[False,True,True])
    np.testing.assert_array_equal(groups(data,inst,'test')['new'],[False,False,True])


def test_adaptive_fetch_filters_future_and_full_history_without_target_information():
    from types import SimpleNamespace
    from cascade.pipelines.steps_final_benchmark import filtered
    class Index:
        ntotal=800
        hnsw=SimpleNamespace(efSearch=0)
        def search(self,q,k):
            return np.arange(k,0,-1,dtype=np.float32)[None,:],np.arange(k)[None,:]
    index=Index();first=np.zeros(801,np.int64);first[1:501]=20
    batch=SimpleNamespace(ts=np.array([10]),mask_items=np.array([501,502]))
    c,s,fetch,rounds=filtered(index,np.ones((1,4),np.float32),None,batch,first,'hnsw')
    assert fetch==800 and rounds==2 and index.hnsw.efSearch==800
    np.testing.assert_array_equal(c[0],np.arange(503,703))


def test_benchmark_query_reproduces_existing_recency_scorer():
    from pathlib import Path
    import pandas as pd
    from cascade.data.instances import ProcessedData
    from cascade.eval.harness import make_batch
    from cascade.pipelines.local_followups import TextMeanScorer
    from cascade.pipelines.steps_final_benchmark import query
    data=ProcessedData(Path('.'),1,4,np.array([0,0,3]),np.array([1,2,3]),np.array([1,20,40]),
        pd.DataFrame(),pd.DataFrame(),{},50)
    frame=pd.DataFrame({'inst':[0],'user':[1],'item':[4],'ts':[50],'hist_len':[3]})
    text=np.random.default_rng(3).normal(size=(5,384)).astype(np.float32)
    text/=np.linalg.norm(text,axis=1,keepdims=True);text[0]=0
    _,q=query(data,torch.from_numpy(text),frame)
    expected=TextMeanScorer(data,text,'cpu',90).score_all(make_batch(data,frame,50)).numpy()
    np.testing.assert_allclose(q@text.T,expected,atol=2e-7,rtol=2e-6)
