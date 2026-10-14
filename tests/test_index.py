import numpy as np
import pytest

from cascade.index.faiss_index import build_index, filtered_search, recall_vs_exact, search, set_search_params


@pytest.fixture(scope="module")
def vecs():
    rng = np.random.default_rng(0)
    v = rng.standard_normal((2001, 16)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v[0] = 0
    q = rng.standard_normal((20, 16)).astype(np.float32)
    return v, q / np.linalg.norm(q, axis=1, keepdims=True)


def brute_force(v, q, k):
    s = q @ v[1:].T
    return np.argsort(-s, axis=1, kind="stable")[:, :k] + 1


def test_exact_equals_bruteforce(vecs):
    v, q = vecs
    _, ids = search(build_index(v, "exact"), q, 50)
    assert (ids == brute_force(v, q, 50)).all()


def test_ivf_full_probe_equals_exact(vecs):
    v, q = vecs
    ivf = build_index(v, "ivf", nlist=16)
    set_search_params(ivf, nprobe=16)
    _, a = search(ivf, q, 50)
    _, e = search(build_index(v, "exact"), q, 50)
    assert (a == e).all()
    set_search_params(ivf, nprobe=1)
    _, a1 = search(ivf, q, 50)
    assert recall_vs_exact(a1, e) < 1.0  # approximate with 1 of 16 cells


def test_hnsw_requires_ef_ge_k(vecs):
    v, q = vecs
    h = build_index(v, "hnsw")
    set_search_params(h, ef_search=20)
    with pytest.raises(AssertionError):
        search(h, q, 50)
    set_search_params(h, ef_search=200)
    _, a = search(h, q, 50)
    _, e = search(build_index(v, "exact"), q, 50)
    assert recall_vs_exact(a, e) > 0.9


def test_filtered_search_drops_blocked_and_unreleased(vecs):
    v, q = vecs
    idx = build_index(v, "exact")
    _, top = search(idx, q[:2], 5)
    first_ts = np.zeros(len(v), dtype=np.int64)
    first_ts[top[0, 1]] = 100  # released after t
    blocked = [{int(top[0, 0])}, set()]
    s, ids, short = filtered_search(idx, q[:2], 10, blocked, ts=np.array([50, 50]), first_ts=first_ts, buffer=20)
    assert top[0, 0] not in ids[0] and top[0, 1] not in ids[0]
    assert (ids[1][:5] == top[1]).all() and short == 0
