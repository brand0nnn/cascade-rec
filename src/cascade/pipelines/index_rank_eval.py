"""End-to-end nDCG@10 of frozen MLP rankers on the index sweep's candidate lists (Val only, exploratory).

Completes RQ2 sweep 2: for every index config saved by ``index_sweep.py`` (filtered top-K₁ lists), score the
lists with each frozen MLP checkpoint and compare each config with exact search on the same Val instances
(paired user bootstrap on seed-mean per-instance nDCG@10; seed SD reported separately).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from cascade.data.instances import ProcessedData
from cascade.data.ranker_features import RankerFeatureBuilder
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.ranker_meso import group_labels
from cascade.models.ranker_mlp import RankerFeatureSpec
from cascade.pipelines.build_candidates import load_candidates
from cascade.pipelines.ranker_k1_sweep import score_prefixes
from cascade.pipelines.ranker_training import create_ranker
from cascade.utils.io import REPO_ROOT, git_hash, save_run


def load_ranker(run: Path, builder: RankerFeatureBuilder):
    """Frozen MLP checkpoint, with the same feature-contract checks as the K₁ sweep."""
    rcfg = yaml.safe_load((run / "config.yaml").read_text())
    spec = RankerFeatureSpec(**{**rcfg["run"]["feature_spec"], "feature_groups": tuple(rcfg["run"]["feature_spec"]["feature_groups"])})
    if dict(spec.cardinalities) != builder.item_cardinalities or not np.allclose(
            [rcfg["run"]["score_mean"], rcfg["run"]["score_std"]], [builder.score_mean, builder.score_std], rtol=0, atol=1e-8):
        raise ValueError(f"frozen feature contract differs from the current builder: {run}")
    model = create_ranker(spec, rcfg)
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu", weights_only=True)["state_dict"], strict=True)
    return model, int(rcfg["seed"])


def summarize(users: np.ndarray, ranks: np.ndarray, found: np.ndarray) -> dict:
    """Seed-level macro metrics for one config ([seeds, n] ranks)."""
    ndcg = np.stack([M.ndcg_at(r, 10) for r in ranks])
    macro = np.array([M.macro_micro(v, users)["macro"] for v in ndcg])
    return {
        "retrieval_recall_macro": M.macro_micro(found.astype(float), users)["macro"],
        "ndcg10_macro_mean": float(macro.mean()), "ndcg10_macro_seed_sd": float(macro.std(ddof=1)) if len(macro) > 1 else None,
        "recall10_macro_mean": float(np.mean([M.macro_micro(M.recall_at(r, 10), users)["macro"] for r in ranks])),
        "conditional_ndcg10_macro_mean": float(np.mean([M.macro_micro(v[found], users[found])["macro"] for v in ndcg])),
        "per_seed_ndcg10_macro": macro.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = REPO_ROOT / "results" / "index_rank_eval" / args.run_id
    root.mkdir(parents=True, exist_ok=False)
    data = ProcessedData.load(cfg.get("data_config", "configs/data.yaml"))
    builder = RankerFeatureBuilder.create(data, load_candidates(REPO_ROOT / cfg["candidate_dir"], "ranker_train"))
    models = [load_ranker(REPO_ROOT / r, builder) for r in cfg["ranker_runs"]]
    lists_root = REPO_ROOT / cfg["index_lists_dir"]
    ranks, ref = {}, None
    for name in cfg["index_configs"]:
        lists = load_candidates(lists_root / name, "val")
        if ref is None:
            ref = lists
            labels = group_labels(data, data.split_instances("val").set_index("inst").loc[lists["inst"]].reset_index())
        elif not (np.array_equal(lists["inst"], ref["inst"]) and np.array_equal(lists["user"], ref["user"])):
            raise ValueError(f"index config {name} is not aligned with {cfg['index_configs'][0]}")
        # Optional nested K₁ prefixes (valid for these pointwise rankers); default = the full list width.
        counts = list(cfg.get("k1_counts") or [lists["cand"].shape[1]])
        per_model = np.stack([score_prefixes(m, builder, lists, counts, device=args.device,
                                             batch_pairs=cfg.get("score_batch_pairs", 8000))[0] for m, _ in models])
        for ki, k1 in enumerate(counts):
            ranks[name if len(counts) == 1 else f"{name}@{k1}"] = per_model[:, ki]
        print(f"scored {name} at K1 {counts}", flush=True)
    users = ref["user"].astype(np.int64)
    base = cfg.get("reference") or next(iter(ranks))
    nd = {n: np.stack([M.ndcg_at(r, 10) for r in rk]).mean(axis=0) for n, rk in ranks.items()}
    masks = {"all": np.ones(len(users), dtype=bool)}
    masks.update({f"item_type={v}": labels["item_type"].to_numpy() == v for v in sorted(labels["item_type"].unique())})
    out = {"scope": "validation only, exploratory", "git_hash": git_hash(), "reference": base,
           "ranker_seeds": [s for _, s in models], "by_config": {}}
    for name, rk in ranks.items():
        out["by_config"][name] = {}
        for g, m in masks.items():
            found = np.isfinite(rk[0][m])
            block = summarize(users[m], rk[:, m], found)
            if name != base:
                block[f"minus_{base}"] = paired_bootstrap(users[m], nd[base][m], nd[name][m], n_resamples=cfg.get("bootstrap_resamples", 1000), seed=0)
            out["by_config"][name][g] = block
    np.savez_compressed(root / "ranks.npz", inst=ref["inst"], user=users, configs=np.array(list(ranks)), ranks=np.stack(list(ranks.values())))
    (root / "results.json").write_text(json.dumps(out, indent=2))
    save_run(root, {k: v for k, v in out.items() if k != "by_config"}, cfg)
    for name, block in out["by_config"].items():
        a = block["all"]; d = a.get(f"minus_{base}")
        print(f"{name:14s} R@K1 {a['retrieval_recall_macro']:.4f} nDCG@10 {a['ndcg10_macro_mean']:.6f} ± {a['ndcg10_macro_seed_sd']:.6f}"
              + (f"  vs {base} {d['diff']:+.6f} [{d['ci_low']:+.6f}, {d['ci_high']:+.6f}]" if d else ""))


if __name__ == "__main__":
    main()
