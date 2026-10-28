"""Focused controls for the ID-residual study; no external/full-data fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from cascade.data.instances import ProcessedData, build_instances, build_sequences
from cascade.models.retriever import ItemTower, RetrieverScorer, TwoTowerRetriever, training_loss
from cascade.pipelines import build_candidates, retriever_ablation, train_retriever
from cascade.utils.seed import set_seed


def small_model(p=0.0):
    set_seed(7)
    text = np.random.default_rng(5).normal(size=(7, 8)).astype(np.float32)
    return TwoTowerRetriever("id_text", "transformer", text, 6, np.array([0, 1, 1, 1, 0, 0, 1], bool),
                             max_len=3, id_dropout=p,
                             item={"dim": 8, "hidden": 16, "dropout": 0.1},
                             user={"dim": 8, "ffn_dim": 16, "n_blocks": 1, "dropout": 0.2})


def test_dropout_keeps_parameter_initialization_and_eval_identical():
    control, treatment = small_model(0), small_model(0.5)
    assert train_retriever.state_sha256(control) == train_retriever.state_sha256(treatment)
    control.eval()
    treatment.eval()
    hist = torch.tensor([[0, 1, 2], [1, 2, 4]])
    torch.testing.assert_close(control.user_vectors(hist), treatment.user_vectors(hist), rtol=0, atol=0)
    torch.testing.assert_close(control.all_item_vectors(), treatment.all_item_vectors(), rtol=0, atol=0)


def test_one_mask_shared_across_history_targets_and_separate_rng():
    model = small_model(0.5).train()
    x, y = torch.tensor([[0, 1, 2], [1, 2, 3]]), torch.tensor([[0, 2, 3], [2, 3, 6]])
    received = []
    hook = model.item_tower.register_forward_pre_hook(lambda mod, args, kwargs: received.append((args[0].clone(), kwargs["id_keep_mask"])), with_kwargs=True)
    global_rng = torch.get_rng_state().clone()
    model.batch_id_keep_mask(x, y, torch.Generator().manual_seed(9))
    assert torch.equal(global_rng, torch.get_rng_state())
    positions = torch.Generator().manual_seed(12)
    expected_positions = torch.Generator().manual_seed(12)
    training_loss(model, x, y, None, 3, positions, torch.Generator().manual_seed(9)).backward()
    torch.randperm(5, generator=expected_positions)
    assert torch.equal(positions.get_state(), expected_positions.get_state())
    assert len(received) == 2 and received[0][1] is received[1][1]
    assert set(received[0][1][torch.unique(torch.cat((x.ravel(), y.ravel())))].tolist()) <= {0.0, 2.0}
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    hook.remove()


def test_dropped_and_unseen_id_gradients_are_zero():
    tower = ItemTower("id_text", np.ones((5, 3), np.float32), 4,
                      np.array([0, 1, 1, 0, 1], bool), dim=2, hidden=4, dropout=0).eval()
    mask = torch.tensor([0., 0., 2., 2., 1.])
    tower(torch.tensor([0, 1, 2, 3]), id_keep_mask=mask).sum().backward()
    torch.testing.assert_close(tower.id_emb.weight.grad[0:2], torch.zeros(2, 2))
    torch.testing.assert_close(tower.id_emb.weight.grad[2], torch.full((2,), 2.))
    torch.testing.assert_close(tower.id_emb.weight.grad[3], torch.zeros(2))
    unseen = torch.tensor([3])
    torch.testing.assert_close(tower(unseen), tower(unseen, use_id=False), rtol=0, atol=0)


def test_default_forward_matches_historical_item_algebra():
    model = small_model().eval()
    ids = torch.tensor([[0, 1, 2], [3, 4, 6]])
    tower = model.item_tower
    expected = (tower.mlp(tower.text_emb[ids]) + tower.id_emb(ids) * tower.id_mask[ids]) * (ids > 0).unsqueeze(-1)
    torch.testing.assert_close(tower(ids), expected, rtol=0, atol=0)


def test_missing_dedicated_generator_is_rejected():
    model = small_model(0.5).train()
    with pytest.raises(ValueError, match="dedicated"):
        training_loss(model, torch.tensor([[0, 1, 2]]), torch.tensor([[0, 2, 3]]), None, 20)


@pytest.mark.parametrize("p", [-0.1, 1.0])
def test_invalid_dropout_rejected(p):
    with pytest.raises(ValueError, match="id_dropout"):
        small_model(p)


@pytest.fixture
def tiny_data(tmp_path):
    rows = [(1, 1, 1, 0), (1, 2, 2, 0), (1, 3, 110, 1), (1, 4, 210, 2), (1, 5, 220, 2), (1, 6, 310, 3),
            (2, 2, 3, 0), (2, 3, 4, 0), (2, 4, 120, 1), (2, 5, 230, 2), (2, 6, 320, 3)]
    inter = pd.DataFrame(rows, columns=["user", "item", "timestamp", "period"])
    ptr, seq, ts = build_sequences(inter, 2)
    instances, _ = build_instances(inter, ptr)
    items = pd.DataFrame({"item": np.arange(1, 13), "first_ts": [1, 2, 4, 120, 220, 310] + [0] * 6,
                          "segment": ["head"] * 6 + ["tail"] * 6, "top1pct": [True] + [False] * 11})
    directory = tmp_path / "data"
    directory.mkdir()
    np.save(directory / "item_text_emb.npy", np.random.default_rng(4).normal(size=(13, 8)).astype(np.float32))
    np.savez(directory / "sequences.npz", user_ptr=ptr, seq_item=seq, seq_ts=ts)
    return ProcessedData(directory, 2, 12, ptr, seq, ts, instances, items,
                         {"t_ranker_train": 100, "t1": 200, "t2": 300}, 3)


def tiny_config(tmp_path, p):
    cfg = {"name": "tiny", "user_tower": "transformer", "item_tower": "id_text", "id_dropout": p,
           "temperature": 0.05, "lr": 0.001, "batch_size": 2, "max_inbatch": 4, "epochs": 1,
           "patience": 5, "early_stop_val_subset": 20_000, "eval_batch_size": 4, "logq": True,
           "item": {"dim": 8, "hidden": 16, "dropout": 0.1},
           "user": {"dim": 8, "ffn_dim": 16, "n_blocks": 1, "dropout": 0.2}}
    path = tmp_path / f"config{p}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg


def test_real_cpu_paired_fit_load_probe_and_no_automatic_test(tmp_path, monkeypatch, tiny_data):
    monkeypatch.setattr(ProcessedData, "load", classmethod(lambda cls, *args, **kwargs: tiny_data))
    def fresh(experiment, run_id, *, exist_ok=True):
        out = tmp_path / "results" / experiment / run_id
        out.mkdir(parents=True, exist_ok=exist_ok)
        return out
    monkeypatch.setattr(train_retriever, "make_run_dir", fresh)
    original = train_retriever.evaluate
    splits = []
    def spy(model, data, split, **kwargs):
        splits.append(split)
        return original(model, data, split, show_progress=False, **{k: v for k, v in kwargs.items() if k != "show_progress"})
    monkeypatch.setattr(train_retriever, "evaluate", spy)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        control_cfg, _ = tiny_config(tmp_path, 0)
        treatment_cfg, _ = tiny_config(tmp_path, 0.5)
        c = train_retriever.main(str(control_cfg), 0, False, "cpu", run_id="study/control/seed0", data_config="isolated.yaml")
        t = train_retriever.main(str(treatment_cfg), 0, False, "cpu", run_id="study/dropout/seed0", data_config="isolated.yaml")
        assert splits and set(splits) == {"val"}
        assert c["initial_state_sha256"] == t["initial_state_sha256"]
        assert c["selector"] == t["selector"] and c["input_sha256"] == t["input_sha256"]
        assert t["q_target_audit"]["excluded_first_events"] == 2
        assert t["test_predictions_loaded"] is False and not (Path(t["run_dir"]) / "test_ranks.npz").exists()
        loaded, cfg = train_retriever.load_retriever(t["run_dir"], tiny_data)
        assert loaded.id_dropout == 0.5 and cfg["data_config"] == "isolated.yaml"
        probe = retriever_ablation.run_probe(t["run_dir"], tmp_path / "probe", "isolated.yaml", "cpu", n_resamples=10)
        assert len(probe["metrics"]) == 4
        assert probe["id_on_reproduction"]["rank_differences"] == 0
        assert probe["replay_validated"] is True
        with pytest.raises(FileExistsError):
            train_retriever.main(str(treatment_cfg), 0, False, "cpu", run_id="study/dropout/seed0")
        monkeypatch.setattr(retriever_ablation, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(retriever_ablation, "fit_path", lambda run_id, arm, seed, sample=False: Path(c["run_dir"] if arm == "control" else t["run_dir"]))
        summary = retriever_ablation.summarize_pairs("study", "pilot", "isolated.yaml", n_resamples=10)
        assert summary["paired_provenance"][0]["matched"]
        assert summary["text_reference_guard"]["status"] == "not_checked"
        assert "new_in_period" in summary["dropout_minus_control"]
    finally:
        torch.set_num_threads(old_threads)


def test_historical_checkpoint_defaults_to_no_dropout(tmp_path, monkeypatch, tiny_data):
    path, cfg = tiny_config(tmp_path, 0)
    cfg.pop("id_dropout")
    path.write_text(yaml.safe_dump(cfg))
    _, seen = train_retriever.retriever_train_log_q(tiny_data)
    model = train_retriever.build_model(cfg, tiny_data, np.load(tiny_data.dir / "item_text_emb.npy"), seen)
    torch.save(model.state_dict(), tmp_path / "model.pth")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    loaded, _ = train_retriever.load_retriever(tmp_path, tiny_data)
    assert loaded.id_dropout == 0
    assert train_retriever.state_sha256(loaded) == train_retriever.state_sha256(model)


def test_phase_and_no_test_defaults(monkeypatch):
    assert train_retriever.main.__kwdefaults__["splits"] == ("val",)
    assert build_candidates.main.__defaults__[-1] == ("ranker_train", "val")
    with pytest.raises(ValueError, match="replication"):
        retriever_ablation.main("study", phase="pilot", seed=1)
    with pytest.raises(ValueError, match="relative"):
        train_retriever.safe_run_id("../escape")
    with pytest.raises(ValueError, match="splits"):
        train_retriever.validate_splits(("val", "val"))


def test_one_arm_pilot_never_promotes_or_evaluates_test(tmp_path, monkeypatch):
    monkeypatch.setattr(retriever_ablation, "REPO_ROOT", tmp_path)
    calls = []
    def fit(config, seed, sample, device, **kwargs):
        calls.append((seed, kwargs))
        return {"run_dir": "unused", "initial_state_sha256": "same"}
    monkeypatch.setattr(train_retriever, "main", fit)
    result = retriever_ablation.main("study", phase="pilot", arm="dropout", data_config="isolated.yaml")
    assert [c[0] for c in calls] == [0]
    assert calls[0][1]["splits"] == ("val",) and calls[0][1]["data_config"] == "isolated.yaml"
    assert result["test_predictions_loaded"] is False


@pytest.mark.parametrize("bad", [np.nan, 1.5, 0., -1., -np.inf])
def test_aligned_ranks_rejects_invalid_values_even_on_unscorable_events(tmp_path, tiny_data, bad):
    inst = tiny_data.split_instances("val")
    ranks = np.ones(len(inst))
    ranks[0] = bad
    np.savez(tmp_path / "val_ranks.npz", inst=inst["inst"].values, user=inst["user"].values,
             rank=ranks, scorable=np.zeros(len(inst), bool))
    with pytest.raises(ValueError, match="ranks"):
        retriever_ablation.aligned_ranks(tmp_path, tiny_data)


@pytest.mark.parametrize("shape", [(3, 1), (2,)])
def test_aligned_ranks_rejects_nonvector_or_wrong_length(tmp_path, tiny_data, shape):
    inst = tiny_data.split_instances("val")
    np.savez(tmp_path / "val_ranks.npz", inst=inst["inst"].values, user=inst["user"].values, rank=np.ones(shape))
    with pytest.raises(ValueError, match="one-dimensional shape"):
        retriever_ablation.aligned_ranks(tmp_path, tiny_data)


@pytest.mark.parametrize("source_ranks,expected_validated", [(None, False), ([1., 2., 3.], False), ([11., 12., 13.], True)])
def test_probe_marks_missing_or_failed_replay_unvalidated(tmp_path, monkeypatch, tiny_data, source_ranks, expected_validated):
    inst = tiny_data.split_instances("val")
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.pth").write_bytes(b"frozen-checkpoint")
    if source_ranks is not None:
        np.savez(source / "val_ranks.npz", inst=inst["inst"].values, user=inst["user"].values, rank=source_ranks)
    monkeypatch.setattr(ProcessedData, "load", classmethod(lambda cls, *args, **kwargs: tiny_data))
    monkeypatch.setattr(train_retriever, "load_retriever", lambda *args, **kwargs: (object(), {"item_tower": "id_text"}))
    monkeypatch.setattr(retriever_ablation, "RetrieverScorer", lambda *args, **kwargs: object())
    measured = np.array([12., 13., 14.])
    def evaluate_probe(model, data, split, *, save_dir, **kwargs):
        assert split == "val"
        save_dir.mkdir()
        np.savez(save_dir / "val_ranks.npz", inst=inst["inst"].values, user=inst["user"].values, rank=measured)
        return {"ndcg@10": retriever_ablation.macro_micro(retriever_ablation.ndcg_at(measured, 10), inst["user"].values)}
    monkeypatch.setattr(retriever_ablation, "evaluate", evaluate_probe)
    result = retriever_ablation.run_probe(str(source), tmp_path / "probe", n_resamples=5)
    assert result["replay_validated"] is expected_validated
    assert result["id_on_reproduction"]["status"] == ("validated" if expected_validated else "unvalidated")
    assert len(result["differences_from_ids_on"]) == 3
    if not expected_validated:
        assert "do not interpret" in result["interpretation"]
    if source_ranks is not None:
        assert result["id_on_reproduction"]["rank_differences"] == 3
