import numpy as np
import torch

from cascade.eval.harness import evaluate, filtered_ranks, make_batch


class RandomModel:
    def __init__(self, n_items, seed=0):
        self.g = torch.Generator().manual_seed(seed)
        self.n = n_items

    def score_all(self, batch):
        return torch.rand(len(batch.users), self.n + 1, generator=self.g)


class CheatModel:
    """Scores history and unreleased items highest; filtering must remove them."""

    def __init__(self, data):
        self.data = data
        self.first_ts = np.concatenate([[0], data.items["first_ts"].values])

    def score_all(self, batch):
        s = torch.zeros(len(batch.users), self.data.n_items + 1)
        s[torch.from_numpy(batch.mask_rows), torch.from_numpy(batch.mask_items.astype(np.int64))] = 100.0
        s[torch.from_numpy(self.first_ts[None, :] > batch.ts[:, None])] = 50.0
        s[torch.arange(len(batch.users)), torch.from_numpy(batch.targets.astype(np.int64))] = 1.0
        return s


def test_candidates_exclude_history_and_unreleased(sample_data):
    inst = sample_data.split_instances("test").head(300)
    b = make_batch(sample_data, inst, 50)
    first_ts = torch.from_numpy(np.concatenate([[np.iinfo(np.int64).max], sample_data.items["first_ts"].values]))
    ranks, top = filtered_ranks(CheatModel(sample_data).score_all(b), b, first_ts)
    assert (ranks == 1).all(), "history / unreleased items were not filtered out"
    ft = first_ts.numpy()
    for r in range(len(inst)):
        hist = set(b.mask_items[b.mask_rows == r].tolist())
        for i in top[r]:
            assert i not in hist and i != 0
            assert ft[i] <= b.ts[r] or i == b.targets[r]


def test_target_released_at_its_own_event(sample_data):
    inst = sample_data.split_instances("test")
    ft = sample_data.items.set_index("item")["first_ts"]
    assert (ft.loc[inst["item"]].values <= inst["ts"].values).all()


def test_random_model_near_chance(sample_data):
    r = evaluate(RandomModel(sample_data.n_items), sample_data, "val", show_progress=False)
    n_avail = sample_data.n_items  # upper bound; filtered catalogue is smaller
    assert r["recall@100"]["micro"] < 5 * 100 / (n_avail * 0.5)
