"""Exercise isolated structure recovery and missing/original artifact checks."""
import numpy as np
import pandas as pd
import pytest

from cascade.data.instances import build_instances, build_sequences
from cascade.pipelines.study_preflight import inspect_inputs, stage_structure
from cascade.eval.saved_results import digest


def test_structure_recovery_preserves_mapping_and_sources(tmp_path):
    source, target = tmp_path / 'original', tmp_path / 'staged'
    source.mkdir()
    inter = pd.DataFrame(dict(user=[1, 1, 1, 2, 2, 2], item=[1, 2, 3, 1, 3, 2],
                              timestamp=[1, 4, 8, 2, 6, 10], rating=[5.] * 6))
    items = pd.DataFrame(dict(item=[1, 2, 3], parent_asin=['a', 'b', 'c']))
    users = pd.DataFrame(dict(user=[1, 2], user_id=['u', 'v']))
    for name, frame in [('interactions', inter), ('items', items), ('users', users)]:
        frame.to_parquet(source / f'{name}.parquet', index=False)
    original_hash = digest(source / 'interactions.parquet')
    result = stage_structure(source, target, dict(split=dict(t1=7, t2=9, ranker_train_fraction=.5)))
    assert result['models_executed'] is False
    assert digest(source / 'interactions.parquet') == original_hash
    pd.testing.assert_frame_equal(pd.read_parquet(target / 'items.parquet'), items)
    staged = pd.read_parquet(target / 'interactions.parquet')
    ptr, _, _ = build_sequences(staged, 2)
    expected, _ = build_instances(staged, ptr)
    pd.testing.assert_frame_equal(pd.read_parquet(target / 'instances.parquet'), expected)
    with pytest.raises(FileExistsError):
        stage_structure(source, target, {})


def test_preflight_keeps_missing_and_wrong_identity_explicit(tmp_path):
    report = inspect_inputs(tmp_path / 'data', tmp_path / 'candidates')
    assert report['status'] == 'not_ready'
    assert report['missing']
    candidate = tmp_path / 'candidates'
    candidate.mkdir()
    (candidate / 'val.npz').write_bytes(b'not original')
    report = inspect_inputs(tmp_path / 'data', candidate,
                            dict(val_sha256='original', meta_sha256='original'))
    assert report['historical_candidate_identity'] == {'val.npz': False, 'meta.json': None}
    assert report['status'] == 'not_ready'
