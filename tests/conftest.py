"""Always exercise the core pipeline on synthetic data; also check prepared samples when available."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from cascade.data.instances import ProcessedData, build_instances, build_sequences
from cascade.data.preprocess import output_dir
from cascade.data.split import assign_periods, compute_t_ranker_train, item_segments
from cascade.utils.config import REPO_ROOT, load_config


@pytest.fixture(scope="session")
def synthetic_data(tmp_path_factory) -> ProcessedData:
    """Isolated reviews with tied times, histories >50, and products first seen in Val/Test.

    Text vectors are seeded random test inputs, not MiniLM embeddings or research data.
    The production split/instance/container code writes and reads the temporary tables.
    """
    directory = tmp_path_factory.mktemp("synthetic-reviews")
    n_users, n_warm = 48, 1536
    n_items = n_warm + 4 * n_users
    records = []
    for user in range(1, n_users + 1):
        warm = (np.arange(76) + (user - 1) * 37) % n_warm + 1
        times = 1000 + np.arange(64) * 10
        times[33] = times[32]  # two distinct items at one timestamp
        events = list(zip(warm[:64], times))
        events += list(zip(warm[64:72], 2000 + np.arange(8) * 10))
        for period, start in enumerate((3000, 4000)):
            new = n_warm + period * 2 * n_users + (user - 1) * 2 + 1
            events += [(new, start), (int(warm[72 + period * 2]), start + 10),
                       (new + 1, start + 20), (int(warm[73 + period * 2]), start + 30)]
        records.extend((user, int(item), int(ts), float(1 + item % 5)) for item, ts in events)
    inter = pd.DataFrame(records, columns=["user", "item", "timestamp", "rating"])
    inter = inter.sort_values(["user", "timestamp", "item"], kind="stable").reset_index(drop=True)
    split = {"t1": 3000, "t2": 4000, "ranker_train_fraction": 0.15}
    split["t_ranker_train"] = compute_t_ranker_train(inter.timestamp.to_numpy(), split["t1"], 0.15)
    inter["period"] = assign_periods(inter.timestamp.to_numpy(), split["t_ranker_train"], split["t1"], split["t2"])
    item_ids = np.arange(1, n_items + 1)
    items = pd.DataFrame({"item": item_ids, "parent_asin": [f"test-item-{i}" for i in item_ids],
                          "first_ts": inter.groupby("item").timestamp.min().reindex(item_ids).to_numpy()})
    assert items.first_ts.notna().all(), "every synthetic product must have an observed first review"
    items.first_ts = items.first_ts.astype(np.int64)
    inter.to_parquet(directory / "interactions.parquet", index=False)
    items.to_parquet(directory / "items.parquet", index=False)
    item_segments(inter, n_items, split["t2"]).to_parquet(directory / "item_segments.parquet", index=False)
    ptr, seq_item, seq_ts = build_sequences(inter, n_users)
    np.savez(directory / "sequences.npz", user_ptr=ptr, seq_item=seq_item, seq_ts=seq_ts,
             n_users=n_users, n_items=n_items)
    instances, _ = build_instances(inter, ptr)
    instances.to_parquet(directory / "instances.parquet", index=False)
    (directory / "split.json").write_text(json.dumps(split))
    vectors = np.random.default_rng(0).normal(size=(n_items + 1, 384)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors[0] = 0
    np.save(directory / "item_text_emb.npy", vectors)
    config = directory / "data.yaml"
    config.write_text(yaml.safe_dump({"processed_dir": str(directory), "max_hist_len": 50}))
    return ProcessedData.load(str(config))


@pytest.fixture(scope="session", params=("synthetic", "prepared"))
def sample_data(request) -> ProcessedData:
    """Run each existing core check on synthetic data and the optional prepared 2% sample."""
    if request.param == "synthetic":
        return request.getfixturevalue("synthetic_data")
    cfg = load_config(REPO_ROOT / "configs/data.yaml")
    if not (output_dir(cfg, True) / "instances.parquet").exists():
        pytest.skip("sample data not built; run scripts/prepare_data.sh --sample")
    return ProcessedData.load(str(REPO_ROOT / "configs/data.yaml"), sample=True)
