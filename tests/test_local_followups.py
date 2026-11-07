import numpy as np

from cascade.pipelines.local_followups import rrf_merge


def test_fusion_uses_actual_lists_unique_budget_and_zero_padding():
    ids,scores=rrf_merge(np.array([1,2,3,0]),np.array([4,2,1,0]),k=3)
    assert ids.tolist()==[1,2,4]
    assert 0 not in scores
    assert scores[2]==1/62
    ids,_=rrf_merge(np.arange(1,201),np.arange(101,301),k=200)
    assert len(ids)==len(set(ids))==200
    assert all(1<=i<=300 for i in ids)


def test_fusion_tie_order_is_deterministic():
    ids,_=rrf_merge(np.array([2]),np.array([1]),k=3)
    assert ids.tolist()==[1,2,0]


def test_frozen_recency_uses_strict_prior_ages_and_ignores_padding():
    from types import SimpleNamespace
    import torch
    from cascade.pipelines.local_followups import TextMeanScorer
    day=86400000
    data=SimpleNamespace(user_ptr=np.array([0,0,2]),seq_ts=np.array([0,90*day]))
    text=np.array([[0.,0.],[1.,0.],[0.,1.]],np.float32)
    batch=SimpleNamespace(hist=torch.tensor([[0,1,2]]),users=np.array([1]),hist_len=np.array([2]),ts=np.array([180*day]))
    uniform=TextMeanScorer(data,text,'cpu').score_all(batch)
    recent=TextMeanScorer(data,text,'cpu',90).score_all(batch)
    assert torch.allclose(uniform[0,1:],torch.tensor([1.,1.])/np.sqrt(2))
    assert torch.allclose(recent[0,1:],torch.tensor([1.,2.])/np.sqrt(5))
    batch.hist=torch.tensor([[1,2]])
    assert torch.equal(recent,TextMeanScorer(data,text,'cpu',90).score_all(batch))


def test_popularity_reads_only_pre_T0_interactions():
    from types import SimpleNamespace
    import pandas as pd
    import torch
    from cascade.pipelines.local_followups import PopularityScorer
    seen=[]
    def interactions(periods):
        seen.extend(periods)
        return pd.DataFrame({'item':[1,1,2]})
    model=PopularityScorer(SimpleNamespace(n_items=3,interactions=interactions))
    assert seen==['retriever_train']
    assert torch.equal(model.score_all(SimpleNamespace(users=[1]))[0],torch.tensor([0.,2.,1.,0.]))
