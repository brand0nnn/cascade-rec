"""Train a two-tower retriever variant on period A and evaluate it with the harness (CLAUDE.md §7).

    python -m cascade.pipelines.train_retriever --config configs/retriever/v2_transformer.yaml --seed 0 [--toy] [--device mps]

Writes results/retriever/<name>[_toy]/seed<k>/{metrics.json, config.yaml, git_hash.txt, model.pth, *_ranks.npz, train_log.json}.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from cascade.data.instances import ProcessedData
from cascade.eval.harness import evaluate
from cascade.models.retriever import RetrieverScorer, TwoTowerRetriever, training_loss
from cascade.utils.config import load_config
from cascade.utils.io import cpu_model, make_run_dir, save_run
from cascade.utils.logging import get_logger
from cascade.utils.seed import set_seed

log = get_logger(__name__)

EVAL_CUTOFFS = (10, 50, 100, 200, 500)


def training_windows(data: ProcessedData, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Period-A sequences as left-padded (input, target) windows [W, L].

    Each user's A-sequence x_1..x_n gives targets x_2..x_n with input = the previous item. Sequences longer than
    L+1 are cut into consecutive windows from the end (stride L), so every target is predicted exactly once;
    a window's context does not reach back into the previous window.
    """
    a = data.interactions(["A"])
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


def period_a_log_q(data: ProcessedData) -> tuple[np.ndarray, np.ndarray]:
    """log q_j = log(count_A(j) / total_A) (static counts, §7.3) and the 'seen in A' mask. Unseen items get log q = 0
    (never used: training targets are always period-A items)."""
    cnt = np.bincount(data.interactions(["A"])["item"].values, minlength=data.n_items + 1).astype(np.float64)
    q = cnt / cnt.sum()
    log_q = np.where(cnt > 0, np.log(np.maximum(q, 1e-30)), 0.0)
    return log_q.astype(np.float32), cnt > 0


def pick_device(want: str) -> torch.device:
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(want)


def main(config_path: str, seed: int, toy: bool, device: str) -> dict:
    cfg = load_config(config_path)
    set_seed(seed)
    dev = pick_device(device)
    data = ProcessedData.load(cfg.get("data_config", "configs/data.yaml"), toy)
    L = data.max_hist_len
    text_emb = np.load(data.dir / "item_text_emb.npy") if cfg["item_tower"] != "id_only" else None
    log_q_np, seen_in_a = period_a_log_q(data)
    model = TwoTowerRetriever(cfg["item_tower"], cfg["user_tower"], text_emb, data.n_items, seen_in_a, max_len=L, temperature=cfg["temperature"]).to(dev)
    log_q = torch.from_numpy(log_q_np).to(dev) if cfg["logq"] else None
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    seq_in, seq_tgt = training_windows(data, L)
    log.info("%s | device=%s | %d training windows, %d target positions", cfg["name"], dev, len(seq_in), int((seq_tgt > 0).sum()))
    run_dir = make_run_dir("retriever", f"{cfg['name']}{'_toy' if toy else ''}/seed{seed}")
    g = torch.Generator().manual_seed(seed)
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
            loss = training_loss(model, torch.from_numpy(seq_in[idx]).to(dev), torch.from_numpy(seq_tgt[idx]).to(dev), log_q, cfg["max_inbatch"], g)
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

    model.load_state_dict(torch.load(run_dir / "model.pth", map_location=dev))
    scorer = RetrieverScorer(model)
    metrics = {"name": cfg["name"], "seed": seed, "toy": toy, "device": str(dev), "cpu": cpu_model(),
               "best_epoch": best_epoch, "best_val_subset_recall@100": best, "train_time_s": train_time}
    for split in ("val", "test"):
        metrics[split] = evaluate(scorer, data, split, cutoffs=EVAL_CUTOFFS, save_dir=run_dir, batch_size=cfg["eval_batch_size"])
        log.info("%s %s: R@50 %.4f R@100 %.4f R@200 %.4f R@500 %.4f nDCG@10 %.4f (macro)", cfg["name"], split,
                 *[metrics[split][f"recall@{c}"]["macro"] for c in (50, 100, 200, 500)], metrics[split]["ndcg@10"]["macro"])
    save_run(run_dir, metrics, {**cfg, "seed": seed, "toy": toy})
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    main(a.config, a.seed, a.toy, a.device)
