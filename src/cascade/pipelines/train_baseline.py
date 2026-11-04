"""Train a RecBole baseline on our split and evaluate it with our harness (CLAUDE.md §6).

    python -m cascade.pipelines.train_baseline --config configs/baselines/sasrec.yaml --seed 0 [--train-on retriever_train] [--sample]

Validation-only by default; Test requires --splits val test explicitly.
Fresh output directories include selected weights, ID mappings and provenance.
"""
from __future__ import annotations

import argparse
import logging
import json
import time
from pathlib import Path

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
from cascade.eval.saved_results import digest

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


def main(config_path: str, seed: int, train_on: str | None, sample: bool, device: str,
         *, run_id: str | None = None, splits: tuple[str, ...] = ("val",),
         data_config: str | None = None, num_threads: int | None = None,
         technical_smoke: bool = False) -> dict:
    if not splits or len(set(splits)) != len(splits) or any(s not in ("val", "test") for s in splits):
        raise ValueError("splits must be a nonempty unique selection of val/test")
    if technical_smoke and not sample:
        raise ValueError("technical smoke requires sample data")
    if num_threads is not None:
        if num_threads < 1:
            raise ValueError("num_threads must be positive")
        torch.set_num_threads(num_threads)
    cfg = load_config(config_path)
    if data_config is not None:
        cfg["data_config"] = data_config
    if technical_smoke:
        cfg.setdefault("recbole", {})["epochs"] = 1
    train_on = train_on or cfg.get("train_on", "all_train")
    if train_on not in RB.TRAIN_PERIODS:
        raise ValueError("unknown training period")
    selected_id = run_id or f"{Path(config_path).stem}/{train_on}{'_sample' if sample else ''}/seed{seed}"
    if Path(selected_id).is_absolute() or ".." in Path(selected_id).parts or selected_id in ("", "."):
        raise ValueError("run_id must be a nonempty relative path without '..'")
    run_dir = make_run_dir("baselines", selected_id, exist_ok=False)
    model_cls, kind = MODELS[cfg["model"]]
    set_seed(seed)
    data = ProcessedData.load(cfg.get("data_config", "configs/data.yaml"), sample)
    # A fresh export prevents stale mappings/splits from an earlier cached export.
    rb_root = run_dir / "atomic"
    ds_name = RB.export_atomic(data, rb_root, kind, train_on, cfg.get("early_stop_val_subset", 20000))

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
    saved_cfg = {**cfg, "train_on": train_on, "seed": seed, "sample": sample,
                 "technical_smoke": technical_smoke, "evaluation_splits": list(splits),
                 "num_threads": torch.get_num_threads(), "recbole": rb_cfg}
    save_run(run_dir, {"status": "running", "test_predictions_loaded": False}, saved_cfg)
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
    # Portable inference state, separate from RecBole's optimizer checkpoint.
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, run_dir / "model.pth")
    np.savez_compressed(run_dir / "id_mappings.npz",
                        item_tokens=np.asarray(dataset.field2id_token["item_id"], dtype=str),
                        user_tokens=np.asarray(dataset.field2id_token["user_id"], dtype=str))
    paths = [data.dir / name for name in ("items.parquet", "users.parquet", "instances.parquet",
                                         "interactions.parquet", "sequences.npz", "split.json")]
    source_paths = [Path(__file__), REPO_ROOT / "src/cascade/models/recbole_baselines.py",
                    REPO_ROOT / "src/cascade/eval/harness.py", REPO_ROOT / config_path]
    def relative(path):
        return str(path.resolve().relative_to(REPO_ROOT))
    provenance = {"input_sha256": {relative(p): digest(p) for p in paths},
                  "source_sha256": {relative(p): digest(p) for p in source_paths},
                  "weights_sha256": digest(run_dir / "model.pth"),
                  "mapping_sha256": digest(run_dir / "id_mappings.npz"),
                  "split": data.split, "sample": sample, "technical_smoke": technical_smoke,
                  "checkpoint_epoch": int(ckpt["epoch"]) if cfg["model"] in ("BPR", "SASRec") else None,
                  "evaluation_splits": list(splits), "test_predictions_loaded": "test" in splits}
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

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
        "status": "complete", "model": cfg["model"], "train_on": train_on, "seed": seed, "sample": sample,
        "technical_smoke": technical_smoke, "test_predictions_loaded": "test" in splits,
        "checkpoint_epoch": provenance["checkpoint_epoch"],
        "train_time_s": train_time, "device": str(config["device"]), "cpu": cpu_model(),
        "recbole_best_valid_subset": {k: float(v) for k, v in (best_valid or {}).items()},
    }
    for split in splits:
        metrics[split] = evaluate(scorer, data, split, batch_size=cfg.get("eval_batch_size", 512), save_dir=run_dir)
        log.info("%s %s: recall@10=%.4f ndcg@10=%.4f recall@100=%.4f (macro)", cfg["model"], split,
                 metrics[split]["recall@10"]["macro"], metrics[split]["ndcg@10"]["macro"], metrics[split]["recall@100"]["macro"])
    save_run(run_dir, metrics, saved_cfg)
    log.info("Selected weights saved: %s", run_dir / "model.pth")
    fh.close()
    logging.getLogger().removeHandler(fh)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-on", choices=list(RB.TRAIN_PERIODS), default=None)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--run-id")
    ap.add_argument("--data-config")
    ap.add_argument("--splits", nargs="+", choices=("val", "test"), default=("val",))
    ap.add_argument("--num-threads", type=int)
    ap.add_argument("--technical-smoke", action="store_true")
    a = ap.parse_args()
    main(a.config, a.seed, a.train_on, a.sample, a.device, run_id=a.run_id,
         data_config=a.data_config, splits=tuple(a.splits), num_threads=a.num_threads,
         technical_smoke=a.technical_smoke)
