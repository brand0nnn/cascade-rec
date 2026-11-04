from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cascade.data.local_graph_features import FrozenGraphFeatures


def data(extra=None):
    reviews=pd.DataFrame({'user':[1,1,1,1,2,2,3,3,4,4,4],
        'item':[1,2,3,4,1,3,2,3,1,2,3],'ts':[1,2,3,4,1,2,1,2,1,2,3]})
    if extra is not None:reviews=pd.concat([reviews,extra],ignore_index=True)
    items=pd.DataFrame({'item':np.arange(1,7),'leaf_cat':['X','Y','X','','X',' '],
        'store':['A','B','A','','A',''],'first_ts':[1,1,1,4,11,12]})
    def histories(users,lengths,max_len):
        out=np.zeros((len(users),max_len),np.int64)
        for row,length in enumerate(lengths):
            history=np.array([1,2])[:length];out[row,-len(history):]=history
        return out
    return SimpleNamespace(n_users=4,n_items=6,split={'t_ranker_train':10},items=items,
        interactions=lambda periods:reviews.copy(),histories=histories)


def queries():
    return pd.DataFrame({'user':[1],'hist_len':[2],'ts':[12],'item':[3]})


def test_graph_counts_normalisation_and_candidate_specific_history_support():
    features=FrozenGraphFeatures(data())
    common,graph=features.build(queries(),np.array([[3,4,5,0]]))
    assert common.dtype==graph.dtype==np.float32
    assert common.shape==(1,4,9) and graph.shape==(1,4,6)
    assert graph[0,0,0]==pytest.approx(np.log1p(3))
    assert graph[0,0,1]==pytest.approx(np.log1p(4))
    assert graph[0,0,2]==pytest.approx(2/np.sqrt(6))
    assert graph[0,0,3:5].tolist()==[.5,.5]
    assert graph[0,0,5]==1
    assert common[0,1,2]==1 and graph[0,1,5]==0
    assert common[0,2,7:9].tolist()==[1,1]
    assert not common[0,3].any() and not graph[0,3].any()


def test_own_user_paths_and_self_history_paths_are_removed():
    features=FrozenGraphFeatures(data())
    common,graph=features.build(queries(),np.array([[1,4]]))
    # Candidate 1 cannot use its own history occurrence; only history item 2 remains.
    assert graph[0,0,0]==pytest.approx(np.log1p(1))
    assert graph[0,0,1]==pytest.approx(np.log1p(1))
    assert graph[0,0,2]==pytest.approx(.5)
    assert graph[0,0,3:5].tolist()==[0,0]
    # Candidate 4 is reviewed only by the query user.
    assert common[0,1,0:2].tolist()==[0,0]
    assert not graph[0,1].any()


def test_future_edges_and_target_labels_cannot_change_features():
    first=FrozenGraphFeatures(data())
    later=pd.DataFrame({'user':[2,1],'item':[5,1],'ts':[11,30]})
    second=FrozenGraphFeatures(data(later))
    candidates=np.array([[3,4,5,6,0]])
    a=first.build(queries(),candidates)
    relabelled=queries();relabelled['item']=6
    b=second.build(relabelled,candidates)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))
    assert first.audit['strict_before_ms']==10 and first.audit['max_review_timestamp']==4


def test_unknown_metadata_and_before_cutoff_queries_are_safe():
    features=FrozenGraphFeatures(data())
    common,graph=features.build(queries(),np.array([[6]]))
    assert not common[0,0,[3,5,7,8]].any()
    assert not graph[0,0,3:5].any()
    earlier=queries();earlier['ts']=9
    with pytest.raises(ValueError,match='cutoff'):features.build(earlier,np.array([[3]]))


def test_own_only_history_is_globally_known_without_relational_support():
    d=data()
    def own_only(users,lengths,max_len):
        history=np.zeros((len(users),max_len),np.int64);history[:,-1]=4
        return history
    d.histories=own_only
    features=FrozenGraphFeatures(d)
    common,graph=features.build(queries(),np.array([[3,5]]))
    assert common[0,:,2].tolist()==[1,1]
    assert not graph.any()
