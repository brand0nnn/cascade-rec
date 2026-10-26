"""Hand-computed group definitions, metric averaging and paired uncertainty."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.ranker_meso import aligned_archive, group_labels, summarize_group
from cascade.eval import metrics as M


def test_chunked_bootstrap_preserves_dense_estimator():
    users=np.array([1,1,2,3,3,3]); a=np.array([0.,1.,0.,1.,0.,0.]); b=np.array([1.,0.,1.,1.,1.,0.])
    means=pd.Series(b-a).groupby(users).mean().to_numpy()
    rng=np.random.default_rng(4)
    boots=means[rng.integers(0,len(means),size=(103,len(means)))].mean(axis=1)
    expected=np.percentile(boots,[2.5,97.5])
    result=paired_bootstrap(users,a,b,n_resamples=103,seed=4,batch_size=7)
    assert result['diff']==pytest.approx(means.mean())
    np.testing.assert_array_equal([result['ci_low'],result['ci_high']],expected)


def test_item_and_history_groups_do_not_use_target_or_future_history():
    t1=100000; day=86400000
    items=pd.DataFrame({'item':[1,2,3,4], 'leaf_cat':['A','B','','C'],
                        'first_ts':[0,0,0,t1], 'segment':['head','tail','tail','head']})
    histories=np.zeros((4,50),dtype=np.int32)
    histories[0,-1:]=[1]; histories[1,-3:]=[1,1,2]
    histories[2,-3:]=[1,2,4]; histories[3,-2:]=[3,3]
    data=SimpleNamespace(items=items,n_items=4,max_hist_len=50,split={'t1':t1,'t2':t1+400*day},
                         histories=lambda u,h,l:histories)
    cases=pd.DataFrame({'inst':[0,1,2,3], 'user':[1,2,3,4], 'item':[4,2,1,2],
                        'hist_len':[1,11,20,12], 'ts':[t1,t1+90*day,t1+180*day,t1+270*day]})
    result=group_labels(data,cases)
    assert result.item_type.tolist()==['new-in-period','warm-tail','warm-head','warm-tail']
    assert result.history_diversity.tolist()==['1-2','11+ / homogeneous','11+ / mixed','11+ / unknown categories']
    assert result.validation_window.tolist()==['days 0-89','days 90-179','days 180-269','days 270+']


def test_seed_metric_average_is_not_average_rank_or_ensemble():
    users=np.array([1,1,2,3]); base=np.array([2,20,np.inf,5])
    ranks=np.array([[15,3,np.inf,5],[2,20,np.inf,1]])
    out=summarize_group(users,base,ranks,np.ones(4,bool),n_resamples=100)
    expected=np.mean([M.macro_micro(M.ndcg_at(r,10),users)['macro'] for r in ranks])
    assert out['ndcg10_macro_mean']==pytest.approx(expected)
    assert out['n_users']==3 and out['n_retrieved_users']==2
    assert out['rescued_mean']==0.5 and out['demoted_mean']==0.5
    assert out['delta_vs_r1_user_bootstrap']['diff']==pytest.approx(expected-M.macro_micro(M.ndcg_at(base,10),users)['macro'])


def test_saved_ranks_must_align_by_case_and_user(tmp_path):
    path=tmp_path/'ranks.npz'; np.savez(path,inst=[1,2],user=[10,20],ranks=[1,2],retrieval_ranks=[1,2])
    with pytest.raises(ValueError,match='Instance alignment'):
        aligned_archive(path,np.array([2,1]),np.array([10,20]))
    with pytest.raises(ValueError,match='User alignment'):
        aligned_archive(path,np.array([1,2]),np.array([20,10]))
