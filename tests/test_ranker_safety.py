"""Protect shared ranker evidence from accidental overwrite or partial resumption."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from cascade.models.ranker_mlp import MLPRanker
from cascade.pipelines import ranker_training
from cascade.utils import io


def test_make_run_dir_exclusive_creation_preserves_existing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "REPO_ROOT", tmp_path)
    run = io.make_run_dir("ranker_mlp", "saved", exist_ok=False)
    checkpoint = run / "model.pt"
    checkpoint.write_bytes(b"saved checkpoint")

    with pytest.raises(FileExistsError):
        io.make_run_dir("ranker_mlp", "saved", exist_ok=False)

    assert checkpoint.read_bytes() == b"saved checkpoint"
    # Other pipelines retain the helper's previous default behaviour.
    assert io.make_run_dir("ranker_mlp", "saved") == run


@pytest.mark.parametrize("interface", ["grouped", "toy"])
def test_ranker_cli_refuses_existing_run_directory(tmp_path, monkeypatch, interface):
    monkeypatch.setattr(io, "REPO_ROOT", tmp_path)
    run = io.make_run_dir("ranker_mlp", "saved")
    checkpoint = run / "model.pt"
    checkpoint.write_bytes(b"saved checkpoint")
    config = {"model": "mlp", "item_text_dim": 0}
    cardinalities = {name: 8 for name in MLPRanker.categorical_names}
    monkeypatch.setattr(ranker_training, "load_config", lambda _: config)
    monkeypatch.setattr(ranker_training, "set_seed", lambda _: None)
    monkeypatch.setattr(ranker_training, "create_ranker", lambda *_: object())

    def unexpected_fit(*args, **kwargs):
        pytest.fail("training must not start when the output run exists")

    monkeypatch.setattr(ranker_training, "fit_candidate_lists", unexpected_fit)
    monkeypatch.setattr(ranker_training, "fit_ranker", unexpected_fit)
    argv = ["ranker_training", "--config", "unused.yaml", "--run-id", "saved"]
    if interface == "grouped":
        monkeypatch.setattr(ranker_training.ProcessedData, "load", lambda *_: object())
        monkeypatch.setattr(ranker_training, "load_candidates", lambda *a, **kw: {})
        monkeypatch.setattr(
            ranker_training.RankerFeatureBuilder, "create",
            lambda *a, **kw: SimpleNamespace(item_cardinalities=cardinalities),
        )
        argv += ["--candidate-dir", "unused-candidates"]
    else:
        archive = ({"numeric": np.zeros((1, 4))}, np.ones(1), np.ones(1), np.ones(1))
        monkeypatch.setattr(ranker_training, "_load_npz", lambda _: archive)
        monkeypatch.setattr(ranker_training, "_cardinalities", lambda _: cardinalities)
        argv += ["--train", "unused-train.npz", "--val", "unused-val.npz"]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(FileExistsError):
        ranker_training.main()

    assert list(run.iterdir()) == [checkpoint]
    assert checkpoint.read_bytes() == b"saved checkpoint"


@pytest.mark.parametrize("interface", ["grouped", "toy"])
def test_resume_is_rejected_before_loading_config_or_data(monkeypatch, capsys, interface):
    def unexpected_load(*args, **kwargs):
        pytest.fail("unsupported --resume must fail before loading config or data")

    monkeypatch.setattr(ranker_training, "load_config", unexpected_load)
    monkeypatch.setattr(ranker_training.ProcessedData, "load", unexpected_load)
    monkeypatch.setattr(ranker_training, "load_candidates", unexpected_load)
    monkeypatch.setattr(ranker_training, "_load_npz", unexpected_load)
    argv = ["ranker_training", "--config", "unused.yaml", "--resume", "saved.pt"]
    argv += (["--candidate-dir", "unused-candidates"] if interface == "grouped" else
             ["--train", "unused-train.npz", "--val", "unused-val.npz"])
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as error:
        ranker_training.main()

    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "--resume is unsupported" in message
    assert "optimizer/RNG state" in message
    assert "selected-artifact carryover" in message
