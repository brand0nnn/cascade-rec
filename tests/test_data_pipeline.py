"""The downloaded-data pipeline must remain runnable from a fresh clone."""
import gzip
import json

import pandas as pd
import yaml

from cascade.data import instances, preprocess, split


def test_small_chronological_pipeline(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [
        ("u1", "a", 5, 1), ("u1", "b", 4, 2), ("u1", "c", 5, 3),
        ("u2", "b", 4, 1), ("u2", "c", 5, 3), ("u2", "d", 3, 4),
        ("u3", "a", 5, 1), ("u3", "e", 5, 5),
    ]
    pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating", "timestamp"]).to_csv(
        raw / "reviews.csv.gz", index=False, compression="gzip"
    )
    with gzip.open(raw / "meta.jsonl.gz", "wt") as stream:
        for item in "abcde":
            stream.write(json.dumps({"parent_asin": item, "title": item,
                                     "categories": ["Phones"], "store": "Store", "price": "$10.00"}) + "\n")
    config = {
        "raw_dir": str(raw), "processed_dir": str(tmp_path / "processed"),
        "sources": {"interactions": {"file": "reviews.csv.gz"},
                    "metadata": {"file": "meta.jsonl.gz"}},
        "split": {"t1": 4, "t2": 5, "period_b_fraction": 0.25},
        "max_hist_len": 50, "toy": {"seed": 1, "user_fraction": 1.0},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))

    preprocess.main(str(path))
    split.main(str(path))
    instances.main(str(path))
    data = instances.ProcessedData.load(str(path))

    assert len(data.interaction_table) == 8
    assert len(data.instances) == 5  # first event of each user has no history
    assert data.split_instances("test")["item"].tolist() == [5]
    assert data.items["has_meta"].all()
    assert data.items["price"].eq(10.0).all()
    assert data.histories([3], [1], 3).tolist() == [[0, 0, 1]]
