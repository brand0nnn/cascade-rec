"""End-to-end CPU latency of the cascade on Val (CLAUDE.md §5 "Latency", §8, §10).

    python -m cascade.pipelines.latency --config configs/cascade/latency.yaml --run-id <id> [--sample]

For every (index config, K₁, ranker checkpoint) job, ``queries`` fixed random Val instances are passed one at a time
through user tower → FAISS search → §4.6 filtering → ranker features + scoring → top 10, single-threaded on CPU, after
``warmup_queries`` warm-up queries (disjoint instances, same for every job). Job order is a seeded permutation. Index
build time is recorded separately; disk I/O, checkpoint loading, index building and feature-table creation are not
timed. Run on an idle machine: the load average before and after is recorded.

Writes ``results/latency/<run_id>[_sample]/`` (refuses to overwrite): metrics.json, config.yaml, git_hash.txt,
report.md and ``timings.npz`` (per-query ms per phase for every job, each job's final top-10 lists as
``top10__<job>``, measured / warm-up instance ids).
``--sample`` uses a fixed user subset of the *full* Val data and the development query counts; never report it.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import torch  # noqa: I001 — must be imported before faiss on macOS (libomp clash)
import faiss
import numpy as np
import yaml
from threadpoolctl import threadpool_info, threadpool_limits

from cascade.data.instances import ProcessedData
from cascade.data.ranker_features import RankerFeatureBuilder
from cascade.eval.latency import COMPONENTS, FINAL_LIST, PHASES, CascadeQuery, measure, sample_instances, summarise_times
from cascade.index.faiss_index import build_index, set_search_params
from cascade.models.ranker_mlp import RankerFeatureSpec
from cascade.pipelines.build_candidates import load_candidates
from cascade.pipelines.index_sweep import effective_ef_search, overlap_fraction
from cascade.pipelines.ranker_training import create_ranker
from cascade.pipelines.train_retriever import load_retriever
from cascade.utils.config import REPO_ROOT, load_config
from cascade.utils.io import cpu_model, make_run_dir, save_run
from cascade.utils.logging import get_logger

log = get_logger(__name__)

INDEX_KINDS = ("exact", "hnsw", "ivf")
# Fields that must agree with the index sweep so latency and accuracy describe the same index configs.
SHARED_WITH_INDEX_SWEEP = ("retriever_run", "buffer", "ef_search_policy", "hnsw", "ivf", "indexes")


def validate_config(cfg: dict, index_sweep_cfg: dict | None = None) -> None:
    """Reject configs that break the §5 protocol or disagree with the index sweep."""
    if cfg["split"] != "val":
        raise ValueError("latency is measured on Val only (Test is final-evaluation only)")
    if cfg["threads"] != 1:
        raise ValueError("§5 latency is single-threaded")
    names = [s["name"] for s in cfg["indexes"]]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate index config names: {names}")
    for s in cfg["indexes"]:
        if s["kind"] not in INDEX_KINDS:
            raise ValueError(f"unknown index kind {s['kind']!r}")
        if s["kind"] == "hnsw" and "ef_search" not in s or s["kind"] == "ivf" and "nprobe" not in s:
            raise ValueError(f"index config {s['name']} is missing its search setting")
    for group in ("exact", "approximate"):
        if not cfg["k1"][group] or any(int(k) < 1 for k in cfg["k1"][group]):
            raise ValueError(f"k1.{group} must be a non-empty list of positive ints")
    if not cfg["ranker_runs"]:
        raise ValueError("at least one ranker run is required")
    if cfg["queries"] < 1 or cfg["warmup_queries"] < 0:
        raise ValueError("queries must be >= 1 and warmup_queries >= 0")
    effective_ef_search(0, 0, cfg["ef_search_policy"])  # validates the policy name
    if index_sweep_cfg is not None:
        diff = [k for k in SHARED_WITH_INDEX_SWEEP if cfg[k] != index_sweep_cfg[k]]
        if diff:
            raise ValueError(f"latency config disagrees with the index sweep on {diff}")


def build_jobs(cfg: dict) -> list[dict]:
    """One job per (index config, K₁, ranker run); exact uses ``k1.exact``, approximate indexes ``k1.approximate``."""
    jobs = []
    for spec in cfg["indexes"]:
        for k1 in cfg["k1"]["exact" if spec["kind"] == "exact" else "approximate"]:
            k1, n_req = int(k1), int(k1) + cfg["buffer"]
            job = {"index": spec["name"], "kind": spec["kind"], "k1": k1, "n_requested": n_req}
            if spec["kind"] == "hnsw":
                job.update(ef_search_configured=spec["ef_search"], ef_search_effective=effective_ef_search(spec["ef_search"], n_req, cfg["ef_search_policy"]))
            elif spec["kind"] == "ivf":
                job.update(nlist=cfg["ivf"]["nlist"], nprobe=spec["nprobe"])
            for run in cfg["ranker_runs"]:
                jobs.append({**job, "ranker_run": run, "key": f"{spec['name']}_k{k1}__{Path(run).name}"})
    return jobs


def job_order(n_jobs: int, seed: int) -> list[int]:
    """Deterministic shuffled job order (so no config is always timed first)."""
    return [int(i) for i in np.random.default_rng(seed).permutation(n_jobs)]


def load_ranker(run: Path, builder: RankerFeatureBuilder, candidate_dir: str) -> tuple[torch.nn.Module, dict]:
    """Frozen ranker checkpoint, checked against the current feature builder (same contract as the K₁ sweep)."""
    rcfg = yaml.safe_load((run / "config.yaml").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    if metrics.get("status") == "running" or metrics.get("development_sample") or rcfg["model"] not in ("mlp", "din"):
        raise ValueError(f"{run}: latency requires a completed full-data MLP/DIN checkpoint")
    if Path(rcfg["run"]["candidate_dir"]) != Path(candidate_dir):
        raise ValueError(f"{run}: trained on {rcfg['run']['candidate_dir']}, not {candidate_dir}")
    fs = rcfg["run"]["feature_spec"]
    spec = RankerFeatureSpec(**{**fs, "feature_groups": tuple(fs["feature_groups"])})
    if dict(spec.cardinalities) != builder.item_cardinalities or not np.allclose(
            [rcfg["run"]["score_mean"], rcfg["run"]["score_std"]], [builder.score_mean, builder.score_std], rtol=0, atol=1e-8):
        raise ValueError(f"{run}: frozen feature contract differs from the current builder")
    model = create_ranker(spec, rcfg)
    ckpt = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    if ckpt["epoch"] != metrics["best_epoch"]:
        raise ValueError(f"{run}: checkpoint does not match the selected epoch")
    info = {"model": rcfg["model"], "seed": int(rcfg["seed"]), "feature_groups": list(fs["feature_groups"]),
            "best_epoch": metrics["best_epoch"], "n_parameters": int(sum(p.numel() for p in model.parameters()))}
    return model.eval(), info


def write_report(path: Path, res: dict) -> None:
    rt, pr = res["runtime"], res["protocol"]
    dev = " — DEVELOPMENT SAMPLE, timings invalid, do not report" if res["development_sample"] else ""
    lines = [f"# Cascade latency (Val){dev}", "",
             f"{pr['n_queries']} queries one at a time after {pr['n_warmup']} warm-up queries (seed {pr['random_seed']}); "
             f"{rt['cpu']}, {rt['platform']}; torch {rt['torch_version']} ({rt['torch_threads']} thread), "
             f"FAISS {rt['faiss_version']} ({rt['faiss_threads']} thread). Load average before / after: "
             f"{rt['loadavg_before']} / {rt['loadavg_after']}.", "",
             "Median / p95 in ms. User tower = history lookup + forward; retrieval = FAISS search (K1 + buffer) + §4.6 filtering; "
             "ranker = features + forward + top-10 ordering; total is timed end to end (sum of phases per query, so the "
             "median total is not the sum of component medians).", "",
             "| job | efSearch used | user fwd | user tower | FAISS search | filter | retrieval | rank features | rank forward | ranker | total | overlap w/ exact ref |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def mp(b: dict) -> str:
        return f"{b['median']:.2f} / {b['p95']:.2f}"

    for key in res["job_order"]:
        j = res["jobs"][key]
        ph, co = j["timing"]["phases"], j["timing"]["components"]
        lines.append(f"| {key} | {j.get('ef_search_effective', '—')} | {mp(ph['user_forward'])} | {mp(co['user_tower'])} | {mp(ph['faiss_search'])} | "
                     f"{mp(ph['filter'])} | {mp(co['retrieval'])} | {mp(ph['rank_prepare'])} | {mp(ph['rank_score'])} | {mp(co['ranker'])} | "
                     f"{mp(co['total'])} | {j['sanity']['mean_overlap_with_reference']:.4f} |")
    lines += ["", "Index build (single thread, not part of query latency): "
              + ", ".join(f"{k} {v['build_seconds']:.1f}s" for k, v in res["index_builds"].items()) + ".",
              "Rows are listed in the (shuffled) order they were timed. Sanity fields in metrics.json are on the timed queries only; "
              "they are not accuracy results."]
    path.write_text("\n".join(lines) + "\n")


def main(config_path: str, run_id: str | None, sample: bool) -> Path:
    cfg = load_config(REPO_ROOT / config_path)
    sweep_cfg = load_config(REPO_ROOT / cfg["index_sweep_config"]) if cfg.get("index_sweep_config") else None
    validate_config(cfg, sweep_cfg)
    jobs = build_jobs(cfg)
    n_queries, n_warmup = (cfg["development"]["queries"], cfg["development"]["warmup_queries"]) if sample else (cfg["queries"], cfg["warmup_queries"])
    run_id = (run_id or time.strftime("%Y%m%d-%H%M%S")) + ("_sample" if sample else "")
    run_dir = make_run_dir("latency", run_id, exist_ok=False)

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    faiss.omp_set_num_threads(1)

    data = ProcessedData.load(str(REPO_ROOT / cfg["data_config"]))
    cand_dir = REPO_ROOT / cfg["candidate_dir"]
    meta = json.loads((cand_dir / "meta.json").read_text())
    if Path(meta["source_run"]) != Path(cfg["retriever_run"]):
        raise ValueError(f"candidate lists come from {meta['source_run']}, not {cfg['retriever_run']}")
    retriever, rcfg = load_retriever(REPO_ROOT / cfg["retriever_run"], data, "cpu")
    V = retriever.all_item_vectors().float().numpy()
    inst = data.split_instances("val")
    if sample:  # same user-subset rule as the index sweep's development mode
        dev = cfg["development"]
        users = np.random.default_rng(dev["random_seed"]).choice(np.arange(1, data.n_users + 1), max(1, round(data.n_users * dev["user_fraction"])), replace=False)
        inst = inst[np.isin(inst["user"].values, users)].reset_index(drop=True)
    assert ((inst["ts"] >= data.split["t1"]) & (inst["ts"] < data.split["t2"])).all(), "latency instances must be Val"
    warm_rows, rows = sample_instances(len(inst), n_queries, n_warmup, cfg["random_seed"])
    first_ts = np.concatenate([[np.iinfo(np.int64).max], data.items["first_ts"].values])
    log.info("val: %d instances; timing %d queries after %d warm-ups; %d jobs", len(inst), len(rows), len(warm_rows), len(jobs))

    builder = RankerFeatureBuilder.create(data, load_candidates(cand_dir, "ranker_train"))
    rankers, ranker_info = {}, {}
    for run in cfg["ranker_runs"]:
        rankers[run], ranker_info[run] = load_ranker(REPO_ROOT / run, builder, cfg["candidate_dir"])

    ref = np.load(cand_dir / "val.npz")  # exact torch top-500 Val lists from the same retriever (cross-check only)
    ref_rows = np.searchsorted(ref["inst"], inst["inst"].values[rows])
    assert (ref["inst"][ref_rows] == inst["inst"].values[rows]).all(), "reference lists do not cover the timed instances"
    ref_cand = ref["cand"][ref_rows]
    targets = inst["item"].values[rows]

    indexes, builds = {}, {}
    for kind in dict.fromkeys(s["kind"] for s in cfg["indexes"]):
        t0 = time.perf_counter()
        if kind == "hnsw":
            indexes[kind] = build_index(V, kind, hnsw_m=cfg["hnsw"]["m"], ef_construction=cfg["hnsw"]["ef_construction"])
        elif kind == "ivf":
            indexes[kind] = build_index(V, kind, nlist=cfg["ivf"]["nlist"], seed=cfg["ivf"]["seed"])
        else:
            indexes[kind] = build_index(V, kind)
        builds[kind] = {"build_seconds": time.perf_counter() - t0, "n_items": int(indexes[kind].ntotal)}
        log.info("built %s index in %.1fs", kind, builds[kind]["build_seconds"])

    order = job_order(len(jobs), cfg["random_seed"])
    res = {"scope": "validation only", "split": "val", "development_sample": sample, "retriever_run": cfg["retriever_run"],
           "retriever_seed": rcfg["seed"], "candidate_dir": cfg["candidate_dir"], "rankers": ranker_info, "n_val_instances": len(inst),
           "protocol": {"n_queries": len(rows), "n_warmup": len(warm_rows), "random_seed": cfg["random_seed"], "buffer": cfg["buffer"],
                        "query_unit": "one Val instance through the full cascade, issued one at a time", "final_list": FINAL_LIST,
                        "phases": list(PHASES), "components": {k: list(v) for k, v in COMPONENTS.items()},
                        "excludes": ["disk I/O", "checkpoint loading", "index building (see index_builds)", "one-time feature-table creation"]},
           "index_builds": builds, "job_order": [jobs[i]["key"] for i in order], "jobs": {}}
    raw = {"inst": inst["inst"].values[rows], "user": inst["user"].values[rows], "warmup_inst": inst["inst"].values[warm_rows],
           "phases": np.array(PHASES), "job_keys": np.array(res["job_order"])}
    loadavg_before = os.getloadavg()
    with threadpool_limits(limits=1), torch.inference_mode():
        threadpools = threadpool_info()
        for i in order:
            job = jobs[i]
            index = indexes[job["kind"]]
            if job["kind"] == "hnsw":
                set_search_params(index, ef_search=job["ef_search_effective"])
            elif job["kind"] == "ivf":
                set_search_params(index, nprobe=job["nprobe"])
            q = CascadeQuery(data, retriever, index, rankers[job["ranker_run"]], builder, inst, first_ts, job["k1"], job["n_requested"])
            times, out = measure(q, warm_rows, rows)
            raw[job["key"]] = times
            raw[f"top10__{job['key']}"] = out["top"]
            n_cand = (out["cand"] > 0).sum(1)
            job_res = {k: v for k, v in job.items() if k != "key"}
            job_res["timing"] = summarise_times(times)
            job_res["sanity"] = {"mean_candidates": float(n_cand.mean()), "short_of_k1": int((n_cand < job["k1"]).sum()),
                                 "mean_overlap_with_reference": float(overlap_fraction(out["cand"], ref_cand[:, : job["k1"]]).mean()),
                                 "target_in_candidates": float((out["cand"] == targets[:, None]).any(1).mean()),
                                 "target_in_top10": float((out["top"] == targets[:, None]).any(1).mean())}
            res["jobs"][job["key"]] = job_res
            log.info("%s: total median %.2f ms (p95 %.2f), overlap with reference %.4f", job["key"], job_res["timing"]["components"]["total"]["median"],
                     job_res["timing"]["components"]["total"]["p95"], job_res["sanity"]["mean_overlap_with_reference"])
    loadavg_after = os.getloadavg()
    res["runtime"] = {"cpu": cpu_model(), "platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
                      "torch_version": torch.__version__, "faiss_version": faiss.__version__, "numpy_version": np.__version__,
                      "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                      "faiss_threads": faiss.omp_get_max_threads(), "threadpools": threadpools, "cpu_count": os.cpu_count(),
                      "loadavg_before": [round(x, 2) for x in loadavg_before], "loadavg_after": [round(x, 2) for x in loadavg_after]}
    if not sample and max(loadavg_before[0], loadavg_after[0]) > 1.5:
        log.warning("load average %.2f / %.2f: the machine was not idle, timings may be inflated", loadavg_before[0], loadavg_after[0])

    np.savez_compressed(run_dir / "timings.npz", **raw)
    save_run(run_dir, res, {**cfg, "run_id": run_id, "sample": sample, "config_path": config_path})
    write_report(run_dir / "report.md", res)
    log.info("saved %s", run_dir)
    return run_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cascade/latency.yaml")
    ap.add_argument("--run-id", default=None, help="results/latency/<run-id>[_sample]; must not exist")
    ap.add_argument("--sample", action="store_true", help="fixed Val user subset + development query counts; timings invalid")
    a = ap.parse_args()
    main(a.config, a.run_id, a.sample)
