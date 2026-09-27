import torch

from cascade.models.recbole_baselines import left_to_right_pad


def test_left_to_right_pad():
    H = torch.tensor([[0, 0, 5, 6], [1, 2, 3, 4], [0, 0, 0, 9]])
    R, lens = left_to_right_pad(H)
    assert R.tolist() == [[5, 6, 0, 0], [1, 2, 3, 4], [9, 0, 0, 0]]
    assert lens.tolist() == [2, 4, 1]
