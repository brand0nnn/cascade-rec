import numpy as np

from cascade.pipelines.train_retriever import retriever_train_log_q, training_windows


def test_windows_cover_each_period_a_target_once(sample_data):
    L = 50
    seq_in, seq_tgt = training_windows(sample_data, L)
    a = sample_data.interactions(["retriever_train"])
    n_users_a = a["user"].nunique()
    assert (seq_tgt > 0).sum() == len(a) - n_users_a  # every A item except each user's first is a target once
    # input at each position is the item right before its target in the user's sequence
    pairs_win = set(zip(seq_in[seq_tgt > 0].tolist(), seq_tgt[seq_tgt > 0].tolist()))
    same_user = a["user"].values[1:] == a["user"].values[:-1]
    pairs_true = set(zip(a["item"].values[:-1][same_user].tolist(), a["item"].values[1:][same_user].tolist()))
    assert pairs_win == pairs_true
    assert ((seq_in > 0) == (seq_tgt > 0)).all()


def test_log_q_only_period_a(sample_data):
    log_q, seen = retriever_train_log_q(sample_data)
    a = sample_data.interactions(["retriever_train"])
    cnt = np.bincount(a["item"].values, minlength=sample_data.n_items + 1)
    assert (seen == (cnt > 0)).all()
    assert np.isclose(np.exp(log_q[seen]).sum(), 1.0)
