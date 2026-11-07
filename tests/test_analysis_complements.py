"""Unit test for the similar/complement target classifier (exploratory analysis helper)."""
import numpy as np

from cascade.analysis.complements import classify


def test_classify_last_item_and_history_views():
    # items: 1..5 ; leaf / store lookups indexed by item id (0 = padding)
    leaf = np.array(["", "A", "A", "B", "", "B"], dtype=object)
    store = np.array(["", "s", "t", "s", "s", ""], dtype=object)
    histories = np.array([[0, 3, 1], [0, 0, 1], [0, 0, 1], [0, 3, 2], [0, 0, 4], [0, 0, 1]])
    target = np.array([2, 2, 3, 1, 1, 5])
    out = classify(target, histories, leaf, store)
    assert out["last_class"].tolist() == [
        "same leaf, diff store", "same leaf, diff store", "diff leaf, same store",
        "same leaf, diff store", "leaf unknown", "diff leaf / store unknown"]
    assert out["leaf_in_history"].tolist() == [
        "leaf in history", "leaf in history", "leaf not in history",
        "leaf in history", "unknown", "leaf not in history"]
    assert out["same_leaf_last"].tolist() == ["true", "true", "false", "true", "unknown", "false"]
