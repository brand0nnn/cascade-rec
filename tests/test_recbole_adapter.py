import torch
import pytest
from types import SimpleNamespace

from cascade.models.recbole_baselines import left_to_right_pad
from cascade.models.recbole_baselines import SASRecBCE


def test_left_to_right_pad():
    H = torch.tensor([[0, 0, 5, 6], [1, 2, 3, 4], [0, 0, 0, 9]])
    R, lens = left_to_right_pad(H)
    assert R.tolist() == [[5, 6, 0, 0], [1, 2, 3, 4], [9, 0, 0, 0]]
    assert lens.tolist() == [2, 4, 1]


def test_sasrec_override_uses_bce_despite_bpr_sampler_configuration():
    embedding = torch.nn.Embedding.from_pretrained(torch.tensor([[0., 0.], [2., 0.], [-2., 0.]]))
    model = SimpleNamespace(ITEM_SEQ='seq', ITEM_SEQ_LEN='length', POS_ITEM_ID='pos', NEG_ITEM_ID='neg',
                            item_embedding=embedding, forward=lambda seq, length: torch.tensor([[1., 0.]]))
    batch = {'seq': torch.tensor([[1]]), 'length': torch.tensor([1]),
             'pos': torch.tensor([1]), 'neg': torch.tensor([2])}
    loss = SASRecBCE.calculate_loss(model, batch)
    assert loss.item() == pytest.approx(2 * torch.nn.functional.softplus(torch.tensor(-2.)).item())


def test_baseline_rejects_invalid_split_before_creating_outputs():
    from cascade.pipelines.train_baseline import main
    with pytest.raises(ValueError, match='splits'):
        main('unused.yaml', 0, None, True, 'cpu', splits=('val', 'val'))


def test_baseline_existing_run_is_preserved(tmp_path, monkeypatch):
    from cascade.pipelines.train_baseline import main
    import cascade.utils.io as io
    monkeypatch.setattr(io, 'REPO_ROOT', tmp_path)
    original = tmp_path / 'results/baselines/previous'
    original.mkdir(parents=True)
    weights = original / 'model.pth'
    weights.write_bytes(b'original weights')
    cfg = tmp_path / 'config.yaml'
    cfg.write_text('model: SASRec\n')
    with pytest.raises(FileExistsError):
        main(str(cfg), 0, None, True, 'cpu', run_id='previous')
    assert weights.read_bytes() == b'original weights'
