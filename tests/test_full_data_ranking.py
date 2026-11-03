import numpy as np
import pytest

from cascade.pipelines.full_data_ranking import batch_args, enabled_columns


def test_full_data_feature_masks_do_not_change_shared_inputs_or_paired_alternatives():
    archive = {'cand': np.arange(1, 13).reshape(2, 6), 'score': np.ones((2, 6), np.float32),
               'history': np.array([[0, 20], [0, 21]])}
    features = {'common': np.arange(108, dtype=np.float32).reshape(2, 6, 9),
                'graph': np.arange(72, dtype=np.float32).reshape(2, 6, 6)-30}
    before = features['graph'].copy()
    columns = np.array([[3, 1, 5], [2, 0, 4]])
    for method, enabled in [('bce_text', []), ('pairwise_text', []),
                            ('pairwise_coreview', [0,1,2,5]), ('pairwise_combined', list(range(6)))]:
        args = [value.numpy() for value in batch_args(archive, features, np.arange(2), method, 'cpu', columns)]
        np.testing.assert_array_equal(args[0], np.take_along_axis(archive['cand'], columns, axis=1))
        np.testing.assert_array_equal(args[3], np.take_along_axis(features['common'], columns[:, :, None], axis=1))
        raw = np.take_along_axis(before, columns[:, :, None], axis=1)
        for i in range(6):
            np.testing.assert_array_equal(args[4][..., i], raw[..., i] if i in enabled else np.zeros((2,3)))
        np.testing.assert_array_equal(features['graph'], before)
    with pytest.raises(ValueError, match='unregistered'): enabled_columns('new_arm')
