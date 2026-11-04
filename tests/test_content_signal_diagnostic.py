from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cascade.pipelines.content_signal_diagnostic import FrozenPool, neighbourhoods, vector_statistics


def test_pooling_separates_candidate_norms_and_history_norms():
    day=86400000
    data=SimpleNamespace(user_ptr=np.array([0,0,2]),seq_ts=np.array([0,90*day]))
    vectors=np.array([[0,0],[2,0],[0,1],[1,1]],np.float32)
    batch=SimpleNamespace(hist=torch.tensor([[0,1,2]]),users=np.array([1]),hist_len=np.array([2]),ts=np.array([180*day]))
    dot=FrozenPool(data,vectors,'cpu',90,cosine=False).score_all(batch)
    cos=FrozenPool(data,vectors,'cpu',90,cosine=True).score_all(batch)
    unit=FrozenPool(data,vectors,'cpu',90,cosine=True,unit_history=True).score_all(batch)
    assert torch.allclose(dot[0],torch.tensor([0,4/3,2/3,4/3]))
    assert torch.allclose(cos[0],torch.tensor([0,1/np.sqrt(2),1/np.sqrt(2),1.],dtype=torch.float32))
    assert unit[0,2]>unit[0,1] and cos[0,2]==cos[0,1]
    batch.hist=torch.tensor([[1,2]])
    assert torch.equal(dot,FrozenPool(data,vectors,'cpu',90,cosine=False).score_all(batch))


def test_geometry_identifies_collinearity_without_using_raw_dimension_as_rank():
    vectors=np.array([[0,0],[1,0],[2,0],[3,0]],np.float32)
    result=vector_statistics(vectors,np.array([1,2,3]),np.array([[0,1],[1,2]]),np.array([0,1,2,4]))
    assert result['random_pair_cosine_mean']==pytest.approx(1.)
    assert result['centred_effective_rank']==pytest.approx(1.)
    assert result['centred_effective_rank_over_dimension']==pytest.approx(.5)
    assert result['norm_log_preT0_count_spearman']==pytest.approx(1.)


def test_neighbours_only_use_permitted_reference_ids_and_exclude_self():
    vectors=np.random.default_rng(0).normal(size=(25,8)).astype(np.float32)
    probes=np.array([1,22,23]);prior=np.arange(1,23)
    result=neighbourhoods(vectors,probes,prior,'cpu')
    assert result.shape==(3,20)
    assert np.isin(result,prior).all()
    assert not (result==probes[:,None]).any()
