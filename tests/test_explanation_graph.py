import json

import pandas as pd
import pytest

from cascade.analysis.explanation_graph import (
    assess_device_audit, co_review_witnesses, device_relation, known_equal,
    make_audit, metadata_ego_html, parse_devices,
)


def test_explicit_device_variants_and_source_spans():
    title = "Case for Apple iPhone 13 Pro Max and Samsung Galaxy S20 Ultra 5G"
    parsed = parse_devices(title)
    assert parsed["positive"] == ["galaxy s20 ultra 5g", "iphone 13 pro max"]
    for mention in parsed["mentions"]:
        assert title[mention["start"]:mention["end"]] == mention["text"]
    assert device_relation(parse_devices("iPhone 13 Pro"), parse_devices("iPhone 13 Pro Max")) == "mismatch"
    assert parse_devices("Galaxy S20+")["positive"] == ["galaxy s20 plus"]
    assert parse_devices("iPhone13ProMax")["positive"] == ["iphone 13 pro max"]


def test_negative_universal_shorthand_and_missing_are_not_positive_matches():
    negative = parse_devices("Not for iPhone 13 Pro Max")
    assert negative["positive"] == []
    assert negative["negative"] == ["iphone 13 pro max"]
    assert parse_devices("Universal holder for all smartphones")['status'] == "universal"
    assert parse_devices("iPhone 12/13 case")['status'] == "ambiguous"
    assert device_relation(negative, parse_devices("iPhone 13 Pro Max")) == "unknown"
    assert device_relation(parse_devices(""), parse_devices("")) == "unknown"
    assert not known_equal("", "")
    assert not known_equal(None, None)
    assert known_equal("Cases > Basic", "Cases > Basic")


def test_coreview_requires_other_user_and_strict_cutoff():
    reviews = pd.DataFrame({"user": [1, 1, 2, 2, 3, 3], "item": [10, 20, 10, 20, 10, 20],
                            "timestamp": [1, 2, 3, 4, 3, 5]})
    paths = co_review_witnesses(reviews, [10], 20, current_user=1, cutoff=5)
    assert paths["user"].tolist() == [2]
    assert (paths[["history_ts", "candidate_ts"]] < 5).all().all()
    assert co_review_witnesses(reviews, [20], 20, 1, 5).empty


def test_manual_audit_is_deterministic_disjoint_and_blank():
    items = pd.DataFrame({"item": range(1, 241), "parent_asin": [str(i) for i in range(240)],
                          "title": ["iPhone 13" if i % 2 else "Universal holder" for i in range(240)],
                          "first_ts": [i for i in range(240)]})
    parsed = [parse_devices(t) for t in items["title"]]
    a = make_audit(items, parsed, 120)
    pd.testing.assert_frame_equal(a, make_audit(items, parsed, 120))
    assert len(a) == a["item"].nunique() == 200
    assert a["audit_split"].value_counts().to_dict() == {"development": 100, "locked": 100}
    assert (a["expected_positive_devices"] == "").all()
    assert assess_device_audit(a)["status"] == "blocked"


def test_audit_precision_gate_needs_labels_and_positive_support():
    audit = pd.DataFrame({"audit_split": ["locked"] * 100, "reviewed": ["yes"] * 100,
                          "parsed_positive_devices": [json.dumps(["iphone 13"])] * 100,
                          "expected_positive_devices": [json.dumps(["iphone 13"])] * 100})
    result = assess_device_audit(audit)
    assert result["status"] == "passed"
    assert result["positive_edges"] == 100
    assert result["precision_wilson95"][0] < 1
    audit.loc[:9, "expected_positive_devices"] = "[]"
    assert assess_device_audit(audit)["status"] == "blocked"
    audit["parsed_positive_devices"] = "[]"
    assert assess_device_audit(audit)["status"] == "blocked"


def test_ego_html_escapes_titles_and_disclaims_model_explanation():
    item = pd.Series({"item": 1, "parent_asin": "A", "title": "<script> iPhone 13", "leaf_cat": "Cases", "store": "Shop"})
    output = metadata_ego_html(item, parse_devices(item["title"]))
    assert "<script>" not in output
    assert "&lt;script&gt;" in output
    assert "no model outcome or causal attribution" in output


def test_base_build_filters_reviews_and_refuses_overwrite(tmp_path):
    import yaml
    from cascade.pipelines.explain_graph import build_base

    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame({"item": [1, 2], "parent_asin": ["A", "B"], "title": ["iPhone 13", "Universal holder"],
                  "leaf_cat": ["Cases", "Holders"], "store": ["Shop", ""], "first_ts": [1, 2]}).to_parquet(source / "items.parquet")
    pd.DataFrame({"user": [1, 2, 3], "item": [1, 2, 2], "timestamp": [1, 2, 3]}).to_parquet(source / "interactions.parquet")
    data_config = tmp_path / "data.yaml"
    data_config.write_text(yaml.safe_dump({"processed_dir": str(source), "split": {"t1": 3, "ranker_train_fraction": 0.15}}))
    graph_config = tmp_path / "graph.yaml"
    graph_config.write_text(yaml.safe_dump({"data_config": str(data_config), "seed": 0, "device_minimum_precision": 0.95}))
    output = tmp_path / "output"
    manifest = build_base(str(graph_config), str(output))
    assert manifest["counts"]["reviews_before_t1"] == 2
    assert pd.read_parquet(output / "review_edges.parquet")["timestamp"].max() < 3
    assert manifest["candidate_overlay"]["status"] == "deferred"
    assert manifest["device_claim_gate"]["status"] == "blocked"
    assert "Test predictions" in " ".join(manifest["limits"])
    with pytest.raises(FileExistsError):
        build_base(str(graph_config), str(output))


def test_indexed_paths_use_strict_snapshot_and_exclude_current_user():
    import numpy as np
    from cascade.analysis.graph_target_profiles import ReviewIndex
    reviews = pd.DataFrame({"user": [1, 1, 2, 2, 3, 3], "item": [10, 20, 10, 20, 10, 20], "timestamp": [1, 2, 3, 4, 3, 5]})
    index = ReviewIndex(reviews)
    path = index.target_paths(np.array([0, 10]), 20, 1, 5)
    assert path["other_reviewers"] == path["witness_paths"] == 1
    assert path["witness_user"] == 2
    assert path["witness_history_ts"] < 5 and path["witness_target_ts"] < 5
    assert index.target_paths(np.array([10]), 20, 1, 4)["other_reviewers"] == 0


def test_target_profile_alignment_rejects_population_or_user_changes():
    import numpy as np
    from cascade.analysis.graph_target_profiles import align_validation
    instances = pd.DataFrame({"inst": [5, 8], "user": [1, 2], "period": [2, 2], "ts": [10, 11]})
    archive = {"inst": np.array([5, 8]), "user": np.array([1, 2])}
    align_validation(instances, archive, 10, 20)
    with pytest.raises(ValueError, match="IDs/order"):
        align_validation(instances.iloc[::-1], archive, 10, 20)
    with pytest.raises(ValueError, match="users"):
        align_validation(instances, dict(archive, user=np.array([1, 3])), 10, 20)
    with pytest.raises(ValueError, match="validation-only"):
        align_validation(instances, archive, 10, 11)


def test_target_profiles_preserve_metadata_unknowns_and_training_snapshots():
    import numpy as np
    from types import SimpleNamespace
    from cascade.analysis.graph_target_profiles import target_profiles
    items = pd.DataFrame({"item": [1, 2, 3, 4], "leaf_cat": ["", "Cases", "", "Chargers"],
                          "store": ["", "Store A", "Store A", ""], "first_ts": [1, 1, 15, 25],
                          "segment": ["head", "head", "tail", "head"]})
    data = SimpleNamespace(items=items, n_items=4, max_hist_len=50,
                           split={"t_ranker_train": 10, "t1": 20},
                           histories=lambda users, lengths, max_len: np.array([[1, 2], [1, 2]]))
    instances = pd.DataFrame({"inst": [9, 10], "user": [1, 2], "item": [3, 4], "hist_len": [2, 2]})
    reviews = pd.DataFrame({"user": [7, 7], "item": [2, 3], "timestamp": [1, 15]})
    profiles = target_profiles(data, instances, reviews)
    assert profiles["same_leaf_last"].tolist() == ["unknown", "false"]
    assert profiles["same_store_history"].tolist() == ["true", "unknown"]
    assert profiles["co_review_retriever_train"].tolist() == ["false", "false"]
    assert profiles["co_review_all_train"].tolist() == ["true", "false"]
    assert profiles["item_type"].tolist() == ["warm-tail", "new-in-period"]
