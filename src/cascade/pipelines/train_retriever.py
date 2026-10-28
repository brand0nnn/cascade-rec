"""Train a two-tower retriever variant on the retriever-training period and evaluate it with the harness (CLAUDE.md §7).

    python -m cascade.pipelines.train_retriever --config configs/retriever/two_tower_text.yaml --seed 0 [--sample] [--device mps]

Writes results/retriever/<name>[_sample]/seed<k>/{metrics.json, config.yaml, git_hash.txt, model.pth, *_ranks.npz, train_log.json}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from cascade.data.instances import ProcessedData
from cascade.eval.harness import evaluate
from cascade.models.retriever import RetrieverScorer, TwoTowerRetriever, training_loss
from cascade.utils.config import REPO_ROOT, config_hash, load_config
from cascade.utils.io import cpu_model, make_run_dir, save_run
from cascade.utils.logging import get_logger
from cascade.utils.seed import set_seed

log = get_logger(__name__)

EVAL_CUTOFFS = (10, 50, 100, 200, 500)


def file_sha256(path: str | Path) -> str:
    """Hash an artifact without loading the complete file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(model: TwoTowerRetriever) -> str:
    """Stable parameter hash for checking matched-arm initialization."""
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_splits(splits: tuple[str, ...]) -> tuple[str, ...]:
    """Test is evaluated only when explicitly requested by a caller."""
    if not splits or len(set(splits)) != len(splits) or any(s not in ("val", "test") for s in splits):
        raise ValueError("splits must be a nonempty unique selection of val/test")
    return tuple(splits)


def safe_run_id(run_id: str) -> str:
    path = Path(run_id)
    if path.is_absolute() or ".." in path.parts or not path.parts or str(path) == ".":
        raise ValueError("run_id must be a relative path without '..'")
    return str(path)


def build_model(cfg: dict, data: ProcessedData, text_emb: np.ndarray | None, seen: np.ndarray) -> TwoTowerRetriever:
    """One constructor path for fresh training and old/new checkpoint loading."""
    return TwoTowerRetriever(
        cfg["item_tower"], cfg["user_tower"], text_emb, data.n_items, seen,
        max_len=data.max_hist_len, temperature=cfg["temperature"],
        id_dropout=cfg.get("id_dropout", 0.0),
        item=cfg.get("item", {}), user=cfg.get("user", {}),
    )


def q_target_audit(data: ProcessedData, seq_tgt: np.ndarray) -> tuple[dict, dict]:
    """Describe the historical static-q approximation without changing it."""
    interactions = data.interactions(["retriever_train"])["item"].values
    counts = np.bincount(interactions, minlength=data.n_items + 1)
    targets = np.bincount(seq_tgt[seq_tgt > 0], minlength=data.n_items + 1)
    q, target_q = counts / counts.sum(), targets / targets.sum()
    summary = {
        "q_definition": "all retriever-training interactions; unchanged",
        "target_definition": "sequence targets excluding each user's first item",
        "interaction_count": int(counts.sum()), "target_count": int(targets.sum()),
        "excluded_first_events": int(counts.sum() - targets.sum()),
        "marginal_l1_difference": float(np.abs(q - target_q).sum()),
        "items_in_q_without_training_target": int(((counts > 0) & (targets == 0)).sum()),
        "limit": "Sequence-window batches and position subsampling further approximate the negative marginal.",
    }
    return summary, {"interaction_counts": counts, "target_counts": targets}


def training_windows(data: ProcessedData, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Period-A sequences as left-padded (input, target) windows [W, L].

    Each user's retriever-training sequence x_1..x_n gives targets x_2..x_n with input = the previous item. Sequences longer than
    L+1 are cut into consecutive windows from the end (stride L), so every target is predicted exactly once;
    a window's context does not reach back into the previous window.
    """
    a = data.interactions(["retriever_train"])
    users, items = a["user"].values, a["item"].values
    starts = np.flatnonzero(np.r_[True, users[1:] != users[:-1]])
    ends = np.r_[starts[1:], len(users)]
    ins, tgts = [], []
    for s, e in zip(starts, ends):
        seq = items[s:e]
        hi = len(seq)
        while hi > 1:
            lo = max(0, hi - 1 - max_len)
            x, y = seq[lo : hi - 1], seq[lo + 1 : hi]
            row_in = np.zeros(max_len, dtype=np.int64)
            row_tg = np.zeros(max_len, dtype=np.int64)
            row_in[max_len - len(x) :], row_tg[max_len - len(y) :] = x, y
            ins.append(row_in)
            tgts.append(row_tg)
            hi = lo + 1
    return np.stack(ins), np.stack(tgts)


def retriever_train_log_q(data: ProcessedData) -> tuple[np.ndarray, np.ndarray]:
    """log q_j = log(count(j) / total) over the retriever-training period (static counts, §7.3) and the
    'seen in retriever-training' mask. Unseen items get log q = 0
    (never used: training targets are always retriever-training items)."""
    cnt = np.bincount(data.interactions(["retriever_train"])["item"].values, minlength=data.n_items + 1).astype(np.float64)
    q = cnt / cnt.sum()
    log_q = np.where(cnt > 0, np.log(np.maximum(q, 1e-30)), 0.0)
    return log_q.astype(np.float32), cnt > 0


def pick_device(want: str) -> torch.device:
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(want)


def main(config_path: str, seed: int, sample: bool, device: str, *, splits: tuple[str, ...] = ("val",), run_id: str | None = None, data_config: str | None = None, technical_smoke: bool = False) -> dict:
    splits = validate_splits(splits)
    cfg = load_config(config_path)
    if technical_smoke:
        if not sample:
            raise ValueError("technical_smoke requires separately mapped sample data")
        cfg["epochs"] = min(int(cfg["epochs"]), 1)
    if cfg["epochs"] < 1:
        raise ValueError("epochs must be positive")
    if data_config is not None:
        cfg["data_config"] = str(data_config)
    selected_run_id = safe_run_id(run_id or f"{cfg['name']}{'_sample' if sample else ''}/seed{seed}")
    if (REPO_ROOT / "results" / "retriever" / selected_run_id).exists():
        raise FileExistsError(f"Run already exists: {selected_run_id}")
    t_start = time.time()
    set_seed(seed)
    dev = pick_device(device)
    data = ProcessedData.load(cfg.get("data_config", "configs/data.yaml"), sample)
    L = data.max_hist_len
    text_emb = np.load(data.dir / "item_text_emb.npy") if cfg["item_tower"] != "id_only" else None
    log_q_np, seen_in_retriever_train = retriever_train_log_q(data)
    model = build_model(cfg, data, text_emb, seen_in_retriever_train).to(dev)
    initial_state_hash = state_sha256(model)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    log_q = torch.from_numpy(log_q_np).to(dev) if cfg["logq"] else None
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    seq_in, seq_tgt = training_windows(data, L)
    if len(seq_in) == 0 or not np.any(seq_tgt > 0):
        raise ValueError("No retriever training targets")
    log.info("%s | device=%s | %d training windows, %d target positions", cfg["name"], dev, len(seq_in), int((seq_tgt > 0).sum()))
    audit, audit_arrays = q_target_audit(data, seq_tgt)
    run_dir = make_run_dir("retriever", selected_run_id, exist_ok=False)
    np.savez_compressed(run_dir / "q_target_counts.npz", **audit_arrays)
    (run_dir / "q_target_audit.json").write_text(json.dumps(audit, indent=2))
    input_hashes = {}
    for filename in ("sequences.npz", "instances.parquet", "items.parquet", "item_segments.parquet", "split.json", "item_text_emb.npy"):
        path = data.dir / filename
        if path.is_file():
            input_hashes[filename] = file_sha256(path)
    subset_count = cfg["early_stop_val_subset"]
    selector_inst = data.split_instances("val")
    if subset_count is not None and subset_count < len(selector_inst):
        idx = np.sort(np.random.default_rng(0).choice(len(selector_inst), subset_count, replace=False))
        selector_inst = selector_inst.iloc[idx]
    np.savez_compressed(run_dir / "selector_instances.npz", inst=selector_inst["inst"].values)
    saved_cfg = {**cfg, "seed": seed, "sample": sample, "technical_smoke": technical_smoke, "run_id": selected_run_id, "evaluation_splits": list(splits)}
    save_run(run_dir, {"status": "running", "test_predictions_loaded": False}, saved_cfg)
    g = torch.Generator().manual_seed(seed)
    id_g = torch.Generator(device="cpu").manual_seed(seed + 104729)
    rng = np.random.default_rng(seed)

    best, best_epoch, bad, history = -1.0, -1, 0, []
    t0 = time.time()
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = rng.permutation(len(seq_in))
        losses = []
        te = time.time()
        for s in range(0, len(perm), cfg["batch_size"]):
            idx = perm[s : s + cfg["batch_size"]]
            loss = training_loss(model, torch.from_numpy(seq_in[idx]).to(dev), torch.from_numpy(seq_tgt[idx]).to(dev), log_q, cfg["max_inbatch"], g, id_g)
            if not torch.isfinite(loss):
                save_run(run_dir, {"status": "failed", "error": "Nonfinite training loss", "epoch": epoch,
                                   "batch_start": s, "test_predictions_loaded": False}, saved_cfg)
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}, batch {s}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        train_s = time.time() - te
        val = evaluate(RetrieverScorer(model), data, "val", cutoffs=(100, 200), max_instances=cfg["early_stop_val_subset"], show_progress=False, batch_size=cfg["eval_batch_size"])
        score = val["recall@100"]["macro"]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_subset_recall@100": score, "val_subset_recall@200": val["recall@200"]["macro"], "train_s": train_s})
        log.info("epoch %d loss %.4f | val-subset R@100 %.4f R@200 %.4f | %.0fs", epoch, np.mean(losses), score, val["recall@200"]["macro"], train_s)
        if score > best:
            best, best_epoch, bad = score, epoch, 0
            torch.save(model.state_dict(), run_dir / "model.pth")
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    train_time = time.time() - t0
    (run_dir / "train_log.json").write_text(json.dumps(history, indent=2))

    model.load_state_dict(torch.load(run_dir / "model.pth", map_location=dev, weights_only=True))
    scorer = RetrieverScorer(model)
    metrics = {"name": cfg["name"], "seed": seed, "sample": sample, "device": str(dev), "cpu": cpu_model(),
               "best_epoch": best_epoch, "best_val_subset_recall@100": best, "train_time_s": train_time,
               "status": "complete", "id_dropout": model.id_dropout,
               "initial_state_sha256": initial_state_hash,
               "config_sha256": config_hash(saved_cfg, n=64),
               "checkpoint_sha256": file_sha256(run_dir / "model.pth"),
               "input_sha256": input_hashes, "q_target_audit": audit,
               "selector": {"metric": "macro Recall@100", "subset_seed": 0, "instances": len(selector_inst),
                            "instance_file_sha256": file_sha256(run_dir / "selector_instances.npz")},
               "rng": {"window_seed": seed, "position_subsampling_seed": seed, "id_dropout_seed": seed + 104729},
               "evaluation_splits": list(splits), "test_predictions_loaded": "test" in splits}
    source_files = (Path(__file__), Path(__file__).parents[1] / "models" / "retriever.py")
    metrics["source_sha256"] = {str(path.relative_to(Path(__file__).parents[3])): file_sha256(path) for path in source_files}
    data_config_path = Path(cfg.get("data_config", "configs/data.yaml"))
    metrics["data_config_sha256"] = file_sha256(data_config_path) if data_config_path.is_file() else None
    for split in splits:
        metrics[split] = evaluate(scorer, data, split, cutoffs=EVAL_CUTOFFS, save_dir=run_dir, batch_size=cfg["eval_batch_size"])
        log.info("%s %s: R@50 %.4f R@100 %.4f R@200 %.4f R@500 %.4f nDCG@10 %.4f (macro)", cfg["name"], split,
                 *[metrics[split][f"recall@{c}"]["macro"] for c in (50, 100, 200, 500)], metrics[split]["ndcg@10"]["macro"])
    metrics["total_time_s"] = time.time() - t_start
    metrics["peak_rss_native"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metrics["peak_rss_unit"] = "bytes on macOS, KiB on Linux"
    metrics["peak_rss_scope"] = "process high-water mark; may include earlier arms in a shared runner process"
    metrics["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None
    metrics["run_dir"] = str(run_dir)
    save_run(run_dir, metrics, saved_cfg)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--splits", nargs="+", choices=("val", "test"), default=("val",))
    ap.add_argument("--run-id")
    ap.add_argument("--data-config")
    ap.add_argument("--technical-smoke", action="store_true", help="one-epoch technical check, requires --sample")
    a = ap.parse_args()
    main(a.config, a.seed, a.sample, a.device, splits=tuple(a.splits), run_id=a.run_id, data_config=a.data_config, technical_smoke=a.technical_smoke)


def load_retriever(run_dir, data: ProcessedData | None = None, device: str = "cpu", *, data_config: str | None = None, sample: bool = False) -> tuple[TwoTowerRetriever, dict]:
    """Rebuild a trained retriever from ``results/retriever/<name>/seed<k>`` (config.yaml + model.pth)."""
    import yaml
    from pathlib import Path

    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    if data_config is not None:
        cfg["data_config"] = str(data_config)
    if data is None:
        data = ProcessedData.load(cfg.get("data_config", "configs/data.yaml"), sample)
    text_emb = np.load(data.dir / "item_text_emb.npy") if cfg["item_tower"] != "id_only" else None
    _, seen_in_retriever_train = retriever_train_log_q(data)
    model = build_model(cfg, data, text_emb, seen_in_retriever_train)
    model.load_state_dict(torch.load(run_dir / "model.pth", map_location="cpu", weights_only=True))
    return model.to(pick_device(device)).eval(), cfg
