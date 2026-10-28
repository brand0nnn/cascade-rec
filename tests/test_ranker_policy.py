"""Controlled intervention, common sampling, frozen preprocessing and no-Test guards."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from cascade.models.ranker_mlp import MLPRanker, RankerFeatureSpec
from cascade.pipelines import ranker_policy_study as study
from cascade.pipelines.ranker_training import (
    create_ranker, fit_candidate_lists, policy_training_columns, policy_training_order,
)
from cascade.pipelines.ranker_k1_sweep import score_prefixes
from cascade.utils.io import REPO_ROOT
from cascade.utils.seed import set_seed


def spec():
    return RankerFeatureSpec({name: 8 for name in MLPRanker.categorical_names},
                            item_text_dim=3, feature_groups=("retrieval", "matching"))


def batch():
    return {"numeric": torch.tensor([[3., 0., 1., .2], [-4., 1., 0., .1]]),
            "history_length_bucket": torch.tensor([2, 3]), "recency_bucket": torch.tensor([2, 2]),
            "history_item_ids": torch.tensor([[0, 2, 3], [4, 5, 6]]),
            "history_leaf_ids": torch.tensor([[0, 2, 2], [3, 3, 4]]),
            "item_text": torch.tensor([[.1, .3, .2], [.2, .5, .7]])}


def lists():
    cand = np.tile(np.arange(1, 67), (4, 1))
    labels = np.zeros(cand.shape, dtype=np.int8)
    labels[np.arange(4), [0, 65, 4, 65]] = 1
    return {"inst": np.array([10, 20, 30, 40]), "user": np.array([1, 1, 2, 3]),
            "cand": cand, "score": np.tile(np.linspace(1, -1, 66), (4, 1)),
            "label": labels, "target_pos": np.array([0, 65, 4, 65]),
            "forced": np.array([False, True, False, True])}


@pytest.fixture
def single_thread_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_masks_preserve_initial_parameters_and_checkpoint_controls_all_inference(tmp_path):
    models = []
    for enabled in ("enabled", "masked"):
        set_seed(0)
        models.append(create_ranker(spec(), {"model": "mlp", "dropout": 0., "retrieval_score_input": enabled}).eval())
    for a, b in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    features = batch()
    changed = {**features, "numeric": features["numeric"].clone()}
    changed["numeric"][:, 0] = torch.tensor([-100., 200.])
    assert not torch.equal(models[0](features), models[0](changed))
    torch.testing.assert_close(models[1](features), models[1](changed), rtol=0, atol=0)
    # The intervention belongs to the saved model, not only the creating config.
    path = tmp_path / "masked.pt"
    torch.save(models[1].state_dict(), path)
    restored = create_ranker(spec(), {"model": "mlp", "dropout": 0.}).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    torch.testing.assert_close(restored(features), restored(changed), rtol=0, atol=0)
    assert not restored._retrieval_score_enabled.item()


def test_historical_checkpoint_strict_loading_keeps_historical_score_input():
    model = create_ranker(spec(), {"model": "mlp"})
    state = model.state_dict()
    del state["_retrieval_score_enabled"]
    restored = create_ranker(spec(), {"model": "mlp", "retrieval_score_input": "masked"})
    restored.load_state_dict(state, strict=True)
    assert restored._retrieval_score_enabled.item()


def test_score_off_has_zero_input_gradient_for_every_candidate():
    model = create_ranker(spec(), {"model": "mlp", "dropout": 0., "retrieval_score_input": "masked"})
    features = batch()
    features["numeric"].requires_grad_(True)
    model(features).sum().backward()
    torch.testing.assert_close(features["numeric"].grad[:, 0], torch.zeros(2), rtol=0, atol=0)


def test_common_negatives_and_relative_order_survive_filtering_and_reordering():
    full = lists()
    subset = {key: value[[2, 0]] for key, value in full.items()}
    all_cols = policy_training_columns(full, 50, 2, 1)
    sub_cols = policy_training_columns(subset, 50, 2, 1)
    np.testing.assert_array_equal(sub_cols, all_cols[[2, 0]])
    assert not np.array_equal(all_cols, policy_training_columns(full, 50, 2, 2))
    for row, cols in enumerate(all_cols):
        assert cols[0] == full["target_pos"][row]
        assert len(np.unique(cols)) == 51
        assert not full["label"][row, cols[1:]].any()
    order = policy_training_order(full["inst"], full["inst"], 2, 1)
    sub_order = policy_training_order(subset["inst"], full["inst"], 2, 1)
    expected = full["inst"][order]
    np.testing.assert_array_equal(subset["inst"][sub_order], expected[np.isin(expected, subset["inst"])])


def test_natural_selection_removes_whole_forced_lists_and_rejects_empty_support():
    full = lists()
    retained = study.select_training_lists(full, "natural_only", np.array([2, 3]))
    np.testing.assert_array_equal(retained["inst"], [30])
    np.testing.assert_array_equal(retained["score"][0], full["score"][2])
    assert (retained["label"].sum(axis=1) == 1).all()
    with pytest.raises(ValueError, match="empty"):
        study.select_training_lists(full, "natural_only", np.array([3]))


def test_normalization_is_fitted_on_full_unfiltered_population(tmp_path, monkeypatch):
    full = lists()
    expected = SimpleNamespace(item_cardinalities={"item_id": 8}, score_mean=-.2, score_std=.08)
    calls = []
    def build(data, inputs):
        calls.append(inputs)
        return expected
    monkeypatch.setattr(study.RankerFeatureBuilder, "create", build)
    path = tmp_path / "reference.yaml"
    path.write_text(yaml.safe_dump({"run": {"feature_spec": {"cardinalities": {"item_id": 8}},
                                            "score_mean": -.2, "score_std": .08}}))
    builder = study.frozen_feature_builder(object(), full, path)
    retained = study.select_training_lists(full, "natural_only")
    assert calls == [full]
    assert len(retained["inst"]) < len(calls[0]["inst"])
    assert builder is expected
    expected.score_mean = -.1
    with pytest.raises(ValueError, match="contract"):
        study.frozen_feature_builder(object(), full, path)


def test_no_test_prediction_inputs_and_no_outputs_on_missing_prerequisites(tmp_path, monkeypatch):
    cfg = study.load_config(REPO_ROOT / "configs/ranker/policy_study.yaml")
    data_cfg = tmp_path / "data.yaml"
    data_cfg.write_text(yaml.safe_dump({"processed_dir": str(tmp_path / "missing_processed")}))
    cfg.update(data_config=str(data_cfg), ranker_config=str(REPO_ROOT / cfg["ranker_config"]),
               candidate_dir=str(tmp_path / "missing_candidates"))
    config_path = tmp_path / "study.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(study, "REPO_ROOT", tmp_path)
    assert all("test" not in path.name.lower() for path in study.required_input_paths(cfg))
    with pytest.raises(FileNotFoundError, match="Recover original"):
        study.run(config_path, "missing", stage="pilot")
    assert not (tmp_path / "results").exists()


def test_alignment_rejects_test_instances():
    data = SimpleNamespace(instances=pd.DataFrame({"inst": [10], "user": [1], "period": [3], "ts": [100]}))
    with pytest.raises(ValueError, match="Test is prohibited"):
        study._validate_alignment(data, {"inst": np.array([10]), "user": np.array([1])}, 2)


def test_stage_selection_and_existing_directory_refuse_before_training(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="seeds"):
        study.run(tmp_path / "unused.yaml", "bad", stage="pilot", seeds=[1])
    cfg = study.load_config(REPO_ROOT / "configs/ranker/policy_study.yaml")
    cfg["ranker_config"] = str(REPO_ROOT / cfg["ranker_config"])
    config_path = tmp_path / "study.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(study, "REPO_ROOT", tmp_path)
    root = tmp_path / "results/ranker_policy/existing"
    root.mkdir(parents=True)
    marker = root / "keep.txt"
    marker.write_text("saved")
    with pytest.raises(FileExistsError):
        study.run(config_path, "existing", stage="pilot")
    assert marker.read_text() == "saved"


def test_invalid_sampling_and_empty_grouped_training_fail_early():
    full = lists()
    with pytest.raises(ValueError, match="unknown training sampling"):
        fit_candidate_lists(object(), object(), full, full, sampling_strategy="other")
    empty = {key: value[:0] for key, value in full.items()}
    with pytest.raises(ValueError, match="empty"):
        fit_candidate_lists(object(), object(), empty, full)


def test_synthetic_four_arm_training_checkpoint_and_prefix_scoring(tmp_path, single_thread_torch):
    """Technical smoke of the real fitting/scoring path, never research evidence."""
    class Builder:
        def build(self, inputs, rows, cols, **kwargs):
            n = len(rows)
            return {"numeric": np.column_stack((inputs["score"][rows, cols],
                        np.ones(n), np.zeros(n), np.full(n, .2))).astype(np.float32),
                    "history_length_bucket": np.full(n, 2), "recency_bucket": np.full(n, 2),
                    "history_item_ids": np.tile([0, 2, 3], (n, 1)),
                    "history_leaf_ids": np.tile([0, 2, 2], (n, 1)),
                    "item_text": np.tile([.1, .3, .2], (n, 1)).astype(np.float32)}
    full = lists()
    val = {key: value.copy() for key, value in full.items()}
    val["forced"][:] = False
    initialization = None
    for arm, policy in study.ARMS.items():
        train = study.select_training_lists(full, policy["training_population"])
        set_seed(0)
        model = create_ranker(spec(), {"model": "mlp", **policy})
        current = study._parameter_digest(model)
        assert initialization is None or current == initialization
        initialization = current
        path = tmp_path / arm / "model.pt"
        path.parent.mkdir()
        fitted = fit_candidate_lists(model, Builder(), train, val, max_epochs=1, patience=3,
            seed=0, checkpoint_path=path, sampling_strategy="instance_keyed",
            training_instance_universe=full["inst"])
        checkpoint = torch.load(path, weights_only=True)
        assert fitted.best_epoch == 1
        assert checkpoint["optimizer_steps"] == 1
        assert checkpoint["retrieval_score_input"] == policy["retrieval_score_input"]
        ranks, aucs = score_prefixes(model, Builder(), val, [10, 20, 50, 66],
                                    device="cpu", batch_pairs=132)
        assert ranks.shape == aucs.shape == (4, 4)
        with np.load(path.with_name("val_ranks.npz")) as archived:
            np.testing.assert_array_equal(ranks[-1], archived["ranks"])
        assert np.isfinite(ranks[-1]).all()
        assert np.isnan(aucs[0, [1, 3]]).all()
        assert np.all((aucs[-1] >= 0) & (aucs[-1] <= 1))
