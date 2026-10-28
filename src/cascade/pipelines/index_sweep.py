"""RQ2 sweep 2, retrieval half (CLAUDE.md §8, §10): FAISS index settings at fixed K₁ on Val.

    python -m cascade.pipelines.index_sweep --config configs/cascade/index_sweep.yaml --run-id <id> [--sample] [--device cpu]

For every index config (exact, HNSW, IVF) the frozen retriever's user vectors query the index, results are §4.6-filtered
(``filtered_search``: ask K₁ + buffer, drop history / not-yet-released items, keep K₁), and we report recall relative to
exact search, retrieval Recall@K (target in the filtered top-K₁, retrieval order) per-user macro and micro, per segment
(§4.7), the number of queries left with fewer than K₁ candidates, and index build time. Search wall time is batch
throughput only, not the §5 latency protocol.

Writes ``results/index_sweep/<run_id>[_sample]/`` (refuses to overwrite): metrics.json, config.yaml, git_hash.txt,
report.md, ``val_target_pos.npz`` (per-instance target column per config, -1 = retrieval miss, so nDCG@10 with a
ranker can be added later) and, if ``save_candidates``, ``<candidates_out>/<run_id>/<config>/val.npz`` lists in the
``load_candidates`` format. ``--sample`` uses a fixed user subset of the *full* data (the frozen retriever's item ids
belong to the full catalogue); development only, never report it.

HNSW: FAISS does not raise efSearch to the number of requested results, and §8 requires efSearch ≥ that number, so
under ``ef_search_policy: at_least_requested`` the search runs with max(efSearch, K₁ + buffer); both are reported.
"""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import torch  # noqa: I001 — must be imported before faiss on macOS (libomp clash: faiss-first segfaults in torch ops)
import faiss
import numpy as np

from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.eval.segments import new_in_period, segment_masks
from cascade.index.faiss_index import build_index, filtered_search, set_search_params
from cascade.pipelines.train_retriever import load_retriever
from cascade.utils.config import REPO_ROOT, load_config
from cascade.utils.io import cpu_model, make_run_dir, save_run
from cascade.utils.logging import get_logger

log = get_logger(__name__)

EF_POLICIES = ("at_least_requested",)


def effective_ef_search(ef_search: int, n_requested: int, policy: str) -> int:
    """efSearch actually used: §8 requires efSearch ≥ #results requested (FAISS does not enforce it)."""
    if policy not in EF_POLICIES:
        raise ValueError(f"unknown ef_search_policy {policy!r}")
    return max(ef_search, n_requested)


def overlap_fraction(approx_ids: np.ndarray, exact_ids: np.ndarray) -> np.ndarray:
    """Per query, fraction of the exact list's ids (0 = padding, ignored) that ``approx_ids`` also contains.

    Vectorised form of ``faiss_index.recall_vs_exact`` (its mean); ids are unique within each row.
    """
    c = np.sort(np.concatenate([approx_ids, exact_ids], axis=1), axis=1)
    both = ((c[:, 1:] == c[:, :-1]) & (c[:, 1:] > 0)).sum(1)
    return both / np.maximum(1, (exact_ids > 0).sum(1))


def target_positions(cand: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """0-based column of each row's target in ``cand``, -1 if absent."""
    hit = cand == targets[:, None]
    return np.where(hit.any(1), hit.argmax(1), -1)


def summarise(target_pos: np.ndarray, users: np.ndarray, cutoffs: list[int], masks: dict[str, np.ndarray], rve: np.ndarray, short: np.ndarray) -> dict:
    """Overall + per-segment block: Recall@c (macro/micro), mean recall vs exact, #queries short of K₁."""
    ranks = np.where(target_pos >= 0, target_pos + 1, np.inf).astype(np.float64)

    def block(m: np.ndarray) -> dict:
        out = {"n_instances": int(m.sum()), "n_users": int(len(np.unique(users[m])))}
        for c in cutoffs:
            out[f"recall@{c}"] = M.macro_micro(M.recall_at(ranks[m], c), users[m])
        out["recall_vs_exact"] = float(rve[m].mean()) if m.any() else float("nan")
        out["short_of_k1"] = int(short[m].sum())
        return out

    return {**block(np.ones(len(users), dtype=bool)), "segments": {name: block(m) for name, m in masks.items()}}


def compare_lists(ours: np.ndarray, ref: np.ndarray, ours_pos: np.ndarray, ref_pos: np.ndarray) -> dict:
    """Agreement of two [n, K₁] candidate-list arrays (same rows) and their target positions."""
    ov = overlap_fraction(ours, ref)
    return {
        "mean_overlap": float(ov.mean()),
        "rows_same_set": float((ov == 1).mean()),
        "rows_same_order": float((ours == ref).all(1).mean()),
        "target_found_both": int(((ours_pos >= 0) & (ref_pos >= 0)).sum()),
        "target_found_ours_only": int(((ours_pos >= 0) & (ref_pos < 0)).sum()),
        "target_found_ref_only": int(((ours_pos < 0) & (ref_pos >= 0)).sum()),
        "target_same_position": float((ours_pos == ref_pos).mean()),
    }


def mismatch_score_gaps(ours: np.ndarray, ours_score: np.ndarray, ref: np.ndarray, Q: np.ndarray, V: np.ndarray) -> np.ndarray:
    """For every item in exactly one of the two lists, |its exact score − our K₁-th score|; near-ties give ~0.

    Rows where ``ours`` is short of K₁ (padding) are skipped: there the reference simply has more items.
    """
    gaps = []
    for r in np.flatnonzero((overlap_fraction(ours, ref) < 1) & (ours > 0).all(1)):
        diff = np.setxor1d(ours[r][ours[r] > 0], ref[r][ref[r] > 0])
        gaps.append(np.abs(V[diff] @ Q[r] - ours_score[r, -1]))
    return np.concatenate(gaps) if gaps else np.zeros(0)


def user_vectors(model, data: ProcessedData, inst, batch_size: int) -> np.ndarray:
    """Normalised user vectors u for each instance (last L history items)."""
    dev = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for s in range(0, len(inst), batch_size):
            b = inst.iloc[s : s + batch_size]
            hist = torch.from_numpy(data.histories(b["user"].values, b["hist_len"].values, data.max_hist_len)).to(dev)
            out.append(model.user_vectors(hist).float().cpu().numpy())
    return np.ascontiguousarray(np.concatenate(out), dtype=np.float32)


def history_sets(data: ProcessedData, inst) -> list[set[int]]:
    """Each instance's full history before t (items blocked by §4.6)."""
    starts = data.user_ptr[inst["user"].values]
    return [set(data.seq_item[s : s + h].tolist()) for s, h in zip(starts, inst["hist_len"].values)]


def run_config(spec: dict, cfg: dict, V: np.ndarray, Q: np.ndarray, blocked: list[set[int]], ts: np.ndarray, first_ts: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Build one index, run the filtered search; returns (info, cand [n, K₁], score [n, K₁], short [n] bool)."""
    k1, buffer, bs = cfg["k1"], cfg["buffer"], cfg["query_batch_size"]
    kind = spec["kind"]
    info: dict = {"kind": kind, "n_requested": k1 + buffer}
    t0 = time.perf_counter()
    if kind == "hnsw":
        index = build_index(V, kind, hnsw_m=cfg["hnsw"]["m"], ef_construction=cfg["hnsw"]["ef_construction"])
        eff = effective_ef_search(spec["ef_search"], k1 + buffer, cfg["ef_search_policy"])
        set_search_params(index, ef_search=eff)
        info.update(m=cfg["hnsw"]["m"], ef_construction=cfg["hnsw"]["ef_construction"], ef_search_configured=spec["ef_search"], ef_search_effective=eff)
    elif kind == "ivf":
        index = build_index(V, kind, nlist=cfg["ivf"]["nlist"], seed=cfg["ivf"]["seed"])
        set_search_params(index, nprobe=spec["nprobe"])
        info.update(nlist=cfg["ivf"]["nlist"], nprobe=spec["nprobe"])
    else:
        index = build_index(V, kind)
    info["build_seconds"] = time.perf_counter() - t0
    n = len(Q)
    cand = np.zeros((n, k1), dtype=np.int32)
    score = np.zeros((n, k1), dtype=np.float32)
    short = np.zeros(n, dtype=bool)
    t0 = time.perf_counter()
    for s in range(0, n, bs):
        sc, ids, _ = filtered_search(index, Q[s : s + bs], k1, blocked[s : s + bs], ts[s : s + bs], first_ts, buffer)
        cand[s : s + len(ids)] = ids
        score[s : s + len(ids)] = np.where(np.isfinite(sc), sc, 0)
        short[s : s + len(ids)] = (ids == 0).any(1)
    info["search_seconds_batch"] = time.perf_counter() - t0
    info["mean_filtered_candidates"] = float((cand > 0).sum(1).mean())
    info["search_ms_per_query_batch"] = 1000 * info["search_seconds_batch"] / max(1, n)
    return info, cand, score, short


def write_report(path: Path, res: dict) -> None:
    k1 = res["k1"]
    lines = [f"# Index sweep ({res['split']}, K1 = {k1}){' — DEVELOPMENT SAMPLE, do not report' if res['development_sample'] else ''}", "",
             f"Retriever `{res['retriever_run']}`; {res['n_instances']} instances, {res['n_users']} users; {res['n_items']} items indexed; "
             f"FAISS {res['faiss_version']} ({res['faiss_threads']} thread(s)), {res['cpu']}. Search asks for K1 + buffer = {k1 + res['buffer']} results, then §4.6 filtering.", "",
             f"| config | efSearch (configured → used) | build s | recall vs exact | Recall@{k1} macro | Recall@{k1} micro | new-in-period Recall@{k1} macro | short of K1 | mean #candidates | search ms/query (batch) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, c in res["configs"].items():
        m = c["metrics"]
        ef = f"{c['ef_search_configured']} → {c['ef_search_effective']}" if "ef_search_configured" in c else "—"
        new = m["segments"]["item_new_in_period"][f"recall@{k1}"]["macro"]
        lines.append(f"| {name} | {ef} | {c['build_seconds']:.1f} | {m['recall_vs_exact']:.4f} | {m[f'recall@{k1}']['macro']:.4f} | "
                     f"{m[f'recall@{k1}']['micro']:.4f} | {new:.4f} | {m['short_of_k1']} | {c['mean_filtered_candidates']:.1f} | {c['search_ms_per_query_batch']:.3f} |")
    lines += ["", "Recall vs exact by segment:", "", "| config | " + " | ".join(next(iter(res["configs"].values()))["metrics"]["segments"]) + " |",
              "|---|" + "---|" * len(next(iter(res["configs"].values()))["metrics"]["segments"])]
    for name, c in res["configs"].items():
        lines.append(f"| {name} | " + " | ".join(f"{b['recall_vs_exact']:.4f}" for b in c["metrics"]["segments"].values()) + " |")
    cc = res.get("cross_check_exact_vs_reference")
    if cc:
        lines += ["", f"Cross-check: FAISS exact filtered top-{k1} vs `{cc['reference']}` (first {k1} columns, torch exact search): "
                  f"mean overlap {cc['mean_overlap']:.5f}, rows with identical set {cc['rows_same_set']:.4f}, identical order {cc['rows_same_order']:.4f}, "
                  f"same target position {cc['target_same_position']:.5f}; target found by ours only / reference only: {cc['target_found_ours_only']} / {cc['target_found_ref_only']}; "
                  f"{cc['rows_short_ours']} rows are short of K1 here (buffer exhausted; the reference ranks the full catalogue); "
                  f"{cc['rows_different_set_excluding_short']} other rows differ in set, max |score − K1-th score| over their mismatched items "
                  f"{cc['mismatch_max_score_gap']:.2e} ({cc['mismatch_items']} items, i.e. boundary ties). Order differences within a set are tied scores."]
    lines += ["", "Search ms/query is batch throughput (4096-query batches, single-threaded FAISS on macOS), not the §5 latency protocol."]
    path.write_text("\n".join(lines) + "\n")


def main(config_path: str, run_id: str | None, sample: bool, device: str) -> Path:
    cfg = load_config(REPO_ROOT / config_path)
    assert cfg["split"] == "val", "the index sweep runs on Val only (Test is final-evaluation only)"
    assert cfg["indexes"][0]["kind"] == "exact", "the first config must be exact search (reference)"
    effective_ef_search(0, 0, cfg["ef_search_policy"])  # validate policy before any work
    run_id = (run_id or time.strftime("%Y%m%d-%H%M%S")) + ("_sample" if sample else "")
    cand_root = REPO_ROOT / cfg["candidates_out"] / run_id
    if cfg["save_candidates"] and cand_root.exists():
        raise FileExistsError(cand_root)
    run_dir = make_run_dir("index_sweep", run_id, exist_ok=False)
    if cfg["save_candidates"]:
        cand_root.mkdir(parents=True, exist_ok=False)

    data = ProcessedData.load(str(REPO_ROOT / cfg["data_config"]))
    model, rcfg = load_retriever(REPO_ROOT / cfg["retriever_run"], data, device)
    V = model.all_item_vectors().float().cpu().numpy()
    inst = data.split_instances("val")
    if sample:  # same user subset rule as the K₁ sweep's development mode
        dev = cfg["development"]
        users = np.random.default_rng(dev["random_seed"]).choice(np.arange(1, data.n_users + 1), max(1, round(data.n_users * dev["user_fraction"])), replace=False)
        inst = inst[np.isin(inst["user"].values, users)].reset_index(drop=True)
    log.info("val: %d instances, %d items", len(inst), data.n_items)
    t0 = time.perf_counter()
    Q = user_vectors(model, data, inst, 1024)
    uv_seconds = time.perf_counter() - t0
    first_ts = np.concatenate([[np.iinfo(np.int64).max], data.items["first_ts"].values])
    blocked = history_sets(data, inst)
    ts, users, targets = inst["ts"].values, inst["user"].values, inst["item"].values.astype(np.int32)
    seg_map = np.concatenate([["pad"], data.items["segment"].values])
    masks = segment_masks(seg_map[targets], new_in_period(first_ts[targets], "val", data.split), inst["hist_len"].values)
    cutoffs = [c for c in cfg["recall_cutoffs"] if c <= cfg["k1"]]

    res = {"scope": "validation only", "split": "val", "development_sample": sample, "retriever_run": cfg["retriever_run"],
           "retriever_seed": rcfg["seed"], "k1": cfg["k1"], "buffer": cfg["buffer"], "n_instances": len(inst),
           "n_users": int(len(np.unique(users))), "n_items": data.n_items, "faiss_version": faiss.__version__,
           "faiss_threads": faiss.omp_get_max_threads(), "cpu": cpu_model(), "machine": platform.machine(),
           "user_vector_seconds": uv_seconds, "configs": {}}
    exact = None
    all_pos = []
    for spec in cfg["indexes"]:
        log.info("index %s", spec)
        info, cand, score, short = run_config(spec, cfg, V, Q, blocked, ts, first_ts)
        if exact is None:
            exact, exact_score = cand, score
        pos = target_positions(cand, targets)
        info["metrics"] = summarise(pos, users, cutoffs, masks, overlap_fraction(cand, exact), short)
        res["configs"][spec["name"]] = info
        all_pos.append(pos)
        log.info("%s: build %.1fs, recall vs exact %.4f, Recall@%d macro %.4f, short %d", spec["name"], info["build_seconds"],
                 info["metrics"]["recall_vs_exact"], cfg["k1"], info["metrics"][f"recall@{cfg['k1']}"]["macro"], info["metrics"]["short_of_k1"])
        if cfg["save_candidates"]:
            (cand_root / spec["name"]).mkdir()
            np.savez(cand_root / spec["name"] / "val.npz", inst=inst["inst"].values, user=users, cand=cand,
                     score=score.astype(np.float16), target_pos=pos)

    ref_path = REPO_ROOT / cfg["reference_candidates"] / "val.npz" if cfg.get("reference_candidates") else None
    if ref_path is not None and ref_path.exists():
        z = np.load(ref_path)
        rows = np.searchsorted(z["inst"], inst["inst"].values)
        assert (z["inst"][rows] == inst["inst"].values).all(), "reference lists do not cover these instances"
        ref = z["cand"][rows, : cfg["k1"]]
        ref_pos = np.where(z["target_pos"][rows] < cfg["k1"], z["target_pos"][rows], -1)
        cc = compare_lists(exact, ref, all_pos[0], ref_pos)
        gaps = mismatch_score_gaps(exact, exact_score, ref, Q, V)
        full = (exact > 0).all(1)
        cc.update(reference=str(ref_path.relative_to(REPO_ROOT)), rows_short_ours=int((~full).sum()),
                  rows_different_set_excluding_short=int(((overlap_fraction(exact, ref) < 1) & full).sum()),
                  mismatch_items=int(len(gaps)), mismatch_max_score_gap=float(gaps.max()) if len(gaps) else 0.0)
        res["cross_check_exact_vs_reference"] = cc
        log.info("cross-check vs %s: %s", ref_path, cc)

    np.savez_compressed(run_dir / "val_target_pos.npz", inst=inst["inst"].values, user=users, configs=np.array([s["name"] for s in cfg["indexes"]]),
                        target_pos=np.stack(all_pos).astype(np.int16))
    save_run(run_dir, res, {**cfg, "run_id": run_id, "sample": sample, "device": device, "config_path": config_path})
    write_report(run_dir / "report.md", res)
    log.info("saved %s", run_dir)
    return run_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cascade/index_sweep.yaml")
    ap.add_argument("--run-id", default=None, help="results/index_sweep/<run-id>[_sample]; must not exist")
    ap.add_argument("--sample", action="store_true", help="fixed user subset for development only")
    ap.add_argument("--device", default="cpu", help="device for the user tower (FAISS always runs on CPU)")
    a = ap.parse_args()
    main(a.config, a.run_id, a.sample, a.device)
