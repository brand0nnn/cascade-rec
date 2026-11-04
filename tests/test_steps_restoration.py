from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cascade.data.batched_graph_features import BatchedGraphFeatures
from cascade.data.instances import ProcessedData,build_sequences
from cascade.data.local_graph_features import FrozenGraphFeatures
from cascade.data.snapshot_graph_features import SnapshotGraphFeatures
from cascade.pipelines.steps_restoration import grid


def example():
    edges=pd.DataFrame({'user':[1,1,1,2,2,2,3,3,3,4,4],
        'item':[1,2,4,1,3,5,2,3,6,1,7],
        'timestamp':[1,2,11,1,10,20,1,12,21,1,19]})
    ptr,item,ts=build_sequences(edges,4)
    items=pd.DataFrame({'item':np.arange(1,9),'leaf_cat':['X','Y','X','Y','X','','X','Y'],
        'store':['A','B','A','B','','C','B','A'],'first_ts':[1,1,10,11,20,21,19,25]})
    data=ProcessedData(Path('.'),4,8,ptr,item,ts,pd.DataFrame(),items,{'t_ranker_train':10,'t1':20,'t2':30},50)
    queries=pd.DataFrame({'user':[1,2,3,4],'hist_len':[3,3,3,2],'ts':[40]*4,'item':[3,2,1,8]})
    candidates=np.array([[3,5,8,0],[2,4,8,0],[1,4,8,0],[2,3,8,0]])
    return data,queries,candidates


def test_explicit_T0_reproduces_frozen_builder_without_changing_split():
    data,queries,candidates=example();split=dict(data.split)
    a=SnapshotGraphFeatures(data,10);b=BatchedGraphFeatures(data)
    np.testing.assert_array_equal(a.degree,b.degree)
    for x,y in zip(a.build(queries,candidates),b.build(queries,candidates)):
        np.testing.assert_array_equal(x,y)
    assert data.split==split


def test_T1_includes_B_edges_but_excludes_exact_cutoff_and_later_edges():
    data,queries,candidates=example();builder=SnapshotGraphFeatures(data,20)
    assert builder.degree[3]==2 and builder.degree[7]==1
    assert builder.degree[5]==0 and builder.degree[6]==0 and builder.degree[8]==0
    assert builder.audit['max_review_timestamp']==19
    assert builder.audit['review_edges']==9
    for a,b in zip(builder.build(queries,candidates),FrozenGraphFeatures.build(builder,queries,candidates)):
        np.testing.assert_allclose(a,b,atol=2e-6,rtol=2e-6)


def test_refreshed_builder_excludes_query_user_and_never_reads_targets():
    data,queries,candidates=example();builder=SnapshotGraphFeatures(data,20)
    a=builder.build(queries,candidates)
    changed=queries.copy();changed['item']=8
    for x,y in zip(a,builder.build(changed,candidates)):np.testing.assert_array_equal(x,y)
    # Shared witnesses for candidate3 do not count user1's own historical edges.
    slow=FrozenGraphFeatures.build(builder,queries,candidates)
    np.testing.assert_allclose(a[1],slow[1],atol=2e-6,rtol=2e-6)


def test_T2_graph_and_pre_snapshot_queries_are_rejected():
    data,queries,candidates=example()
    with pytest.raises(ValueError,match='T0/T1'):SnapshotGraphFeatures(data,30)
    queries.loc[0,'ts']=19
    with pytest.raises(ValueError,match='precedes'):SnapshotGraphFeatures(data,20).build(queries,candidates)


def test_grid_restores_all_original_variants_and_retains_strong_control():
    settings=grid();assert len(settings)==14 and len({s['setting'] for s in settings})==14
    assert {s['k'] for s in settings if s['kind']=='flat'}=={50,100,200,500}
    assert {(s['training'],s['nprobe']) for s in settings if s['kind']=='ivf'}=={(t,p) for t in ('preT0','catalogue') for p in (4,16,32)}
    assert {s['efSearch'] for s in settings if s['kind']=='hnsw'}=={32,64,128,'adaptive'}


def test_prefix_membership_must_not_be_inferred_from_full_pool_pessimistic_rank():
    from cascade.pipelines.steps_final import rank_top
    c=np.array([[7,8,9]]);s=np.ones((1,3));target=np.array([7])
    assert rank_top(s,c,target)[0][0]==3
    assert rank_top(s[:,:2],c[:,:2],target)[0][0]==2


def test_native_HNSW_low_effort_is_not_silently_raised_during_refill():
    from types import SimpleNamespace
    from cascade.pipelines.steps_restoration_benchmark import filtered_native
    class Index:
        ntotal=1000
        hnsw=SimpleNamespace(efSearch=0)
        def search(self,q,k):return np.arange(k,0,-1,dtype=np.float32)[None,:],np.arange(k)[None,:]
    first=np.zeros(1001,np.int64);first[1:451]=100
    batch=SimpleNamespace(ts=np.array([50]),mask_items=np.array([451]))
    index=Index();c,s,info=filtered_native(index,np.ones((1,8),np.float32),batch,first,200,32)
    assert info['fetch']==800 and info['effective_ef']==32 and info['search_rounds']==2
    np.testing.assert_array_equal(c[0],np.arange(452,652))


def test_exact_equivalence_allows_only_numerical_boundary_ties():
    from cascade.pipelines.steps_restoration_benchmark import equivalent_sets
    dense=np.array([-np.inf,1.,.5,.5000001,.1])
    assert equivalent_sets(np.array([1,3]),np.array([1,2]),dense)==1
    with pytest.raises(ValueError,match='outside numerical'):
        equivalent_sets(np.array([1,4]),np.array([1,2]),dense)


def test_native_faiss_accepts_original_effort_values_without_wrapper_guard():
    import faiss
    faiss.omp_set_num_threads(1)
    x=np.random.default_rng(1).normal(size=(1000,16)).astype(np.float32)
    x/=np.linalg.norm(x,axis=1,keepdims=True)
    index=faiss.IndexHNSWFlat(16,16,faiss.METRIC_INNER_PRODUCT);index.add(x)
    for effort in (32,64,128):
        index.hnsw.efSearch=effort
        _,ids=index.search(x[:2],200)
        assert (ids>=0).sum()==400 and index.hnsw.efSearch==effort
        assert all(len(np.unique(row))==200 for row in ids)
