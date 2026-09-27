"""Train a RecBole baseline on our split and evaluate it with our harness (CLAUDE.md §6).

    python -m cascade.pipelines.train_baseline --config configs/baselines/sasrec.yaml --seed 0 [--train-on A] [--toy]

Writes results/baselines/<model>_<train_on>[_toy]/seed<k>/{metrics.json, config.yaml, git_hash.txt, *_ranks.npz}.
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.utils import get_trainer, init_logger, init_seed

from cascade.data.instances import ProcessedData
from cascade.eval.harness import evaluate
from cascade.models import recbole_baselines as RB
from cascade.utils.config import REPO_ROOT, load_config
from cascade.utils.io import cpu_model, make_run_dir, save_run
from cascade.utils.logging import get_logger
from cascade.utils.seed import set_seed

log = get_logger(__name__)

MODELS = {"Pop": (RB.PopExact, "general"), "ItemKNN": (RB.ItemKNNLite, "general"), "BPR": (BPR, "general"), "SASRec": (RB.SASRecBCE, "sequential")}


def pick_device(want: str) -> torch.device:
    if want == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(want)


def main(config_path: str, seed: int, train_on: str | None, toy: bool, device: str) -> dict:
    cfg = load_config(config_path)
    train_on = train_on or cfg.get("train_on", "AB")
    model_cls, kind = MODELS[cfg["model"]]
    set_seed(seed)
    data = ProcessedData.load(cfg.get("data_config", "configs/data.yaml"), toy)
    rb_root = data.dir / "recbole"
    ds_name = RB.export_atomic(data, rb_root, kind, train_on, cfg.get("early_stop_val_subset", 20000))

    run_dir = make_run_dir("baselines", f"{cfg['model']}_{train_on}{'_toy' if toy else ''}/seed{seed}")
    rb_cfg = RB.recbole_config_dict(kind, rb_root, ds_name)
    rb_cfg.update(cfg.get("recbole", {}))
    rb_cfg.update(seed=seed, checkpoint_dir=str(run_dir / "checkpoint"), use_gpu=False)
    config = Config(model=model_cls, dataset=ds_name, config_dict=rb_cfg)
    config["device"] = pick_device(device)
    init_logger(config)
    fh = logging.FileHandler(run_dir / "recbole.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    init_seed(seed, True)
    torch.autograd.set_detect_anomaly(False)  # see note in recbole_baselines.py
    dataset = create_dataset(config)
    train_data, valid_data, _ = data_preparation(config, dataset)  # 3rd loader is the valid copy; never used
    model = model_cls(config, train_data._dataset).to(config["device"])
    log.info("%s on %s | device=%s | RecBole n_users=%d n_items=%d", cfg["model"], train_on, config["device"], dataset.user_num, dataset.item_num)

    t0 = time.time()
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid = None
    if cfg["model"] in ("Pop", "ItemKNN"):
        pass  # statistics computed in __init__; no training / RecBole validation
    else:
        best_valid_score, best_valid = trainer.fit(train_data, valid_data, saved=True, show_progress=config["show_progress"])
        if cfg["model"] in ("BPR", "SASRec"):
            ckpt = torch.load(trainer.saved_model_file, map_location=config["device"], weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
    train_time = time.time() - t0
    model.eval()

    if cfg["model"] == "Pop":
        scorer = RB.PopScorer(model, dataset, data.n_items)
    elif cfg["model"] == "ItemKNN":
        scorer = RB.ItemKNNScorer(model, dataset, data.n_items)
    elif cfg["model"] == "BPR":
        train_users = np.unique(data.interactions(RB.TRAIN_PERIODS[train_on])["user"].values)
        scorer = RB.BPRScorer(model, dataset, data.n_items, train_users)
    else:
        scorer = RB.SASRecScorer(model, dataset, data.n_items)

    metrics = {
        "model": cfg["model"], "train_on": train_on, "seed": seed, "toy": toy,
        "train_time_s": train_time, "device": str(config["device"]), "cpu": cpu_model(),
        "recbole_best_valid_subset": {k: float(v) for k, v in (best_valid or {}).items()},
    }
    for split in ("val", "test"):
        metrics[split] = evaluate(scorer, data, split, batch_size=cfg.get("eval_batch_size", 512), save_dir=run_dir)
        log.info("%s %s: recall@10=%.4f ndcg@10=%.4f recall@100=%.4f (macro)", cfg["model"], split,
                 metrics[split]["recall@10"]["macro"], metrics[split]["ndcg@10"]["macro"], metrics[split]["recall@100"]["macro"])
    save_run(run_dir, metrics, {**cfg, "train_on": train_on, "seed": seed, "toy": toy, "recbole": {k: v for k, v in rb_cfg.items()}})
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-on", choices=["AB", "A"], default=None)
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    main(a.config, a.seed, a.train_on, a.toy, a.device)
