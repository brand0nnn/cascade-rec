import numpy as np

from cascade.analysis.title_leak_audit import max_proxy, qualifying_keys, release_proxy, visible


def test_release_proxy_ignores_multi_model_and_post_t2():
    keys = [frozenset(), frozenset({'iphone|13'}), frozenset({'iphone|13', 'iphone|12', 'iphone|11', 'iphone|x'}),
            frozenset({'iphone|13', 'iphone|12'}), frozenset({'iphone|14'})]
    first = np.array([0, 300, 100, 200, 900])
    p = release_proxy(keys, first, t2=800)
    assert p == {'iphone|13': 200.0, 'iphone|12': 200.0}            # 4-key item (100) and post-T2 item ignored
    assert release_proxy(keys, first, t2=800, q=2)['iphone|13'] == 300.0
    assert release_proxy(keys, first, t2=800, q=10)['iphone|12'] == 200.0   # fewer than q -> latest available


def test_visible_future_and_qualifying():
    p = {'iphone|13': 200.0, 'iphone|14': 500.0}
    k = frozenset({'iphone|13', 'iphone|14'})
    assert visible(k, p, 300) == {'iphone|13'} and visible(frozenset({'pixel|6'}), p, 10**9) == frozenset()
    assert max_proxy(k, p) == 500.0 and max_proxy(frozenset(), p) == -np.inf
    assert qualifying_keys(k, {'iphone': 13.0}) == {'iphone|14'}
    assert qualifying_keys(frozenset({'iphone|14', 'iphone|13', 'iphone|12', 'iphone|11'}), {'iphone': 11.0}) == frozenset()
