import json

import numpy as np
import pytest
import torch

from cascade.models.retriever import RetrieverScorer, TwoTowerRetriever
from cascade.pipelines.build_candidates import force_insert, load_candidates, retrieve, target_scores


@pytest.fixture(scope="module")
def scorer(toy_data):
    torch.manual_seed(0)
    emb = np.load(toy_data.dir / "item_text_emb.npy") if (toy_data.dir / "item_text_emb.npy").exists() else None
    if emb is None:
        pytest.skip("toy text embeddings not built")
    m = TwoTowerRetriever("text", "transformer", emb, toy_data.n_items, np.ones(toy_data.n_items + 1, bool), max_len=toy_data.max_hist_len)
    return RetrieverScorer(m)


@pytest.fixture(scope="module")
def lists(toy_data, scorer):
    out = {}
    for split in ("B", "val"):
        inst = toy_data.split_instances(split).head(400)
        out[split] = (inst, retrieve(scorer, toy_data, inst, 50, show_progress=False))
    return out


def test_lists_sorted_and_filtered(toy_data, lists):
    first_ts = np.concatenate([[np.iinfo(np.int64).max], toy_data.items["first_ts"].values])
    inst, z = lists["val"]
    assert (np.diff(z["score"].astype(np.float32), axis=1) <= 1e-3).all(), "lists must be sorted by score"
    for r, (u, hl, t) in enumerate(zip(inst["user"], inst["hist_len"], inst["ts"])):
        s = toy_data.user_ptr[u]
        hist = set(toy_data.seq_item[s : s + hl].tolist())
        c = z["cand"][r]
        assert not (set(c.tolist()) & hist), "history items must be filtered out"
        assert (first_ts[c] <= t).all(), "not-yet-released items must be filtered out"


def test_eval_lists_never_force_target(lists):
    inst, z = lists["val"]
    assert ((z["target_pos"] >= 0) == (z["target_rank"] <= 50)).all()
    hits = z["target_pos"] >= 0
    assert (z["cand"][hits, z["target_pos"][hits]] == inst["item"].values[hits]).all()


def test_train_lists_always_contain_target(toy_data, scorer, lists):
    inst, z = lists["B"]
    f = force_insert(z, inst["item"].values.astype(np.int32), target_scores(scorer, toy_data, inst))
    assert (f["target_pos"] >= 0).all()
    assert (f["cand"][np.arange(len(inst)), f["target_pos"]] == inst["item"].values).all()
    assert f["forced"].sum() == (z["target_pos"] < 0).sum()


def test_load_candidates_ranker_val_forced(tmp_path, toy_data, scorer, lists):
    inst, z = lists["val"]
    np.savez(tmp_path / "val.npz", **z)
    np.savez(tmp_path / "val_target_score.npz", item=inst["item"].values.astype(np.int32), score=target_scores(scorer, toy_data, inst))
    (tmp_path / "meta.json").write_text(json.dumps({"source_run": "test"}))
    e2e = load_candidates(tmp_path, "val", k1=20)
    assert e2e["cand"].shape[1] == 20 and e2e["label"].sum() == (z["target_rank"] <= 20).sum()
    rv = load_candidates(tmp_path, "val", k1=20, force=True)
    assert (rv["label"].sum(1) == 1).all()
