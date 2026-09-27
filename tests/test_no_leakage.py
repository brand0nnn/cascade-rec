import numpy as np
import pandas as pd

from cascade.data.instances import build_instances, build_sequences, history_lengths


def test_history_lengths_ties():
    users = np.array([1, 1, 1, 1, 2, 2])
    ts = np.array([5, 7, 7, 9, 1, 2])
    user_ptr = np.array([0, 0, 4, 6])
    # same-timestamp events are not in each other's history
    assert history_lengths(users, ts, user_ptr).tolist() == [0, 1, 1, 3, 0, 1]


def test_histories_strictly_before_target(toy_data):
    inst = toy_data.instances.sample(n=min(3000, len(toy_data.instances)), random_state=0)
    for u, target, t, hl in zip(inst["user"], inst["item"], inst["ts"], inst["hist_len"]):
        s = toy_data.user_ptr[u]
        h_items = toy_data.seq_item[s : s + hl]
        h_ts = toy_data.seq_ts[s : s + hl]
        assert (h_ts < t).all()
        assert target not in h_items
        # every earlier event of the user is in the history (no gaps)
        all_ts = toy_data.seq_ts[s : toy_data.user_ptr[u + 1]]
        assert (all_ts < t).sum() == hl
    assert (toy_data.instances["hist_len"] > 0).all()


def test_padded_histories_are_suffix(toy_data):
    inst = toy_data.split_instances("test").head(200)
    H = toy_data.histories(inst["user"].values, inst["hist_len"].values, 50)
    for row, (u, hl) in zip(H, zip(inst["user"], inst["hist_len"])):
        s = toy_data.user_ptr[u]
        full = toy_data.seq_item[s : s + hl]
        nz = row[row > 0]
        assert nz.tolist() == full[-50:].tolist()
