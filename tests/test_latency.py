import copy
import time

import numpy as np
import pytest

from cascade.eval.latency import (COMPONENTS, PHASES, filter_candidates, percentile_summary, sample_instances,
                                  stamps_to_ms, summarise_times, timed_chain)
from cascade.index.faiss_index import build_index, filtered_search, search
from cascade.pipelines.latency import build_jobs, job_order, validate_config
from cascade.utils.config import REPO_ROOT, load_config


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(REPO_ROOT / "configs/cascade/latency.yaml")


def test_sample_instances_deterministic_and_disjoint():
    w1, m1 = sample_instances(1000, 100, 10, seed=0)
    w2, m2 = sample_instances(1000, 100, 10, seed=0)
    assert np.array_equal(w1, w2) and np.array_equal(m1, m2)
    assert len(w1) == 10 and len(m1) == 100 and not set(w1) & set(m1) and len(set(m1)) == 100
    assert not np.array_equal(sample_instances(1000, 100, 10, seed=1)[1], m1)
    with pytest.raises(ValueError):
        sample_instances(50, 45, 10, seed=0)


def test_percentile_summary():
    s = percentile_summary(np.arange(1, 101, dtype=float))
    assert s["median"] == pytest.approx(50.5) and s["p95"] == pytest.approx(95.05)
    assert s["min"] == 1 and s["max"] == 100 and s["n"] == 100


def test_timed_chain_total_covers_every_stage():
    def slow(x):
        time.sleep(0.002)
        return x + 1

    out, stamps = timed_chain([slow, slow, lambda x: x * 10], 0)
    assert out == 20 and len(stamps) == 4
    ms = stamps_to_ms(stamps[None])[0]
    assert (ms >= 0).all() and ms[0] >= 2 and ms[1] >= 2
    assert (stamps[-1] - stamps[0]) / 1e6 == pytest.approx(ms.sum())  # total = end-to-end = sum of stages


def test_summarise_times_components_are_per_query_sums():
    rng = np.random.default_rng(0)
    times = rng.exponential(1.0, size=(200, len(PHASES)))
    out = summarise_times(times)
    assert set(out["phases"]) == set(PHASES) and set(out["components"]) == set(COMPONENTS)
    assert set(COMPONENTS["total"]) == set(PHASES)
    assert out["components"]["total"]["median"] == pytest.approx(np.median(times.sum(1)))
    # every phase belongs to exactly one non-total component
    parts = [p for c, ps in COMPONENTS.items() if c != "total" for p in ps]
    assert sorted(parts) == sorted(PHASES)
    i = [PHASES.index(p) for p in COMPONENTS["retrieval"]]
    assert out["components"]["retrieval"]["p95"] == pytest.approx(np.percentile(times[:, i].sum(1), 95))


def test_filter_candidates_matches_filtered_search():
    rng = np.random.default_rng(0)
    n_items, d, k1, buffer = 300, 8, 20, 30
    V = np.zeros((n_items + 1, d), dtype=np.float32)
    V[1:] = rng.normal(size=(n_items, d))
    V[1:] /= np.linalg.norm(V[1:], axis=1, keepdims=True)
    Q = rng.normal(size=(10, d)).astype(np.float32)
    first_ts = np.concatenate([[np.iinfo(np.int64).max], rng.integers(0, 100, n_items)])
    ts = rng.integers(0, 100, 10)
    hist = [rng.choice(np.arange(1, n_items + 1), 15, replace=False) for _ in range(10)]
    index = build_index(V, "exact")
    ref_s, ref_i, _ = filtered_search(index, Q, k1, [set(h.tolist()) for h in hist], ts, first_ts, buffer)
    raw_s, raw_i = search(index, Q, k1 + buffer)
    for r in range(10):
        ids, sc = filter_candidates(raw_i[r], raw_s[r], hist[r], ts[r], first_ts, k1)
        n = len(ids)
        assert n <= k1 and np.array_equal(ids, ref_i[r, :n]) and (ref_i[r, n:] == 0).all()
        assert np.allclose(sc, ref_s[r, :n])
        assert not np.isin(ids, hist[r]).any() and (first_ts[ids] <= ts[r]).all()


def test_config_parses_and_matches_index_sweep(cfg):
    validate_config(cfg, load_config(REPO_ROOT / cfg["index_sweep_config"]))
    assert cfg["split"] == "val" and cfg["queries"] == 1000 and cfg["warmup_queries"] == 50


def test_validate_config_rejects_bad_settings(cfg):
    for key, value in [("split", "test"), ("threads", 4)]:
        bad = copy.deepcopy(cfg)
        bad[key] = value
        with pytest.raises(ValueError):
            validate_config(bad)
    bad = copy.deepcopy(cfg)
    bad["indexes"][1]["ef_search"] = 64
    with pytest.raises(ValueError):
        validate_config(bad, load_config(REPO_ROOT / cfg["index_sweep_config"]))
    bad = copy.deepcopy(cfg)
    bad["indexes"].append(dict(bad["indexes"][0]))
    with pytest.raises(ValueError):
        validate_config(bad)


def test_build_jobs_and_order(cfg):
    jobs = build_jobs(cfg)
    n_runs = len(cfg["ranker_runs"])
    assert len(jobs) == (len(cfg["k1"]["exact"]) + 4 * len(cfg["k1"]["approximate"])) * n_runs
    exact = sorted(j["k1"] for j in jobs if j["kind"] == "exact")
    assert exact == sorted(cfg["k1"]["exact"] * n_runs)
    hnsw = {j["index"]: j for j in jobs if j["kind"] == "hnsw"}
    assert hnsw["hnsw_ef200"]["ef_search_effective"] == 450 and hnsw["hnsw_ef800"]["ef_search_effective"] == 800
    assert all(j["n_requested"] == j["k1"] + cfg["buffer"] for j in jobs)
    assert len({j["key"] for j in jobs}) == len(jobs)
    order = job_order(len(jobs), cfg["random_seed"])
    assert order == job_order(len(jobs), cfg["random_seed"]) and sorted(order) == list(range(len(jobs)))
