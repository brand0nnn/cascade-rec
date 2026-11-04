import numpy as np
import torch

from cascade.pipelines.local_rankers import SemanticRanker, score_lists


def test_ranker_ignores_left_padding_and_preserves_shared_initialisation():
    text=np.random.default_rng(0).normal(size=(8,384)).astype(np.float32)
    text[0]=0
    leaf=np.array([0,1,1,2,0,2,1,0])
    store=np.array([0,1,2,2,0,1,1,0])
    torch.manual_seed(0)
    mean=SemanticRanker(text,leaf,store,False).eval()
    torch.manual_seed(0)
    attention=SemanticRanker(text,leaf,store,True).eval()
    for key,value in mean.state_dict().items():
        assert torch.equal(value,attention.state_dict()[key])
    candidates=torch.tensor([[3,4]])
    for model in (mean,attention):
        with torch.no_grad():
            a=model(candidates,torch.tensor([[0,0,1,2]]))
            b=model(candidates,torch.tensor([[1,2]]))
        assert torch.allclose(a,b,atol=1e-6)


def test_ranker_ties_are_pessimistic_and_missing_targets_remain_misses():
    class Zero(torch.nn.Module):
        def forward(self,candidates,history):
            return torch.zeros_like(candidates,dtype=torch.float32)
    archive={'cand':np.array([[1,2,0],[1,2,0]]),'target':np.array([2,3])}
    ranks,top=score_lists(Zero(),archive,np.array([[0,1],[0,1]]),'cpu')
    assert ranks[0]==2 and np.isinf(ranks[1])
    assert top.shape==(2,3)
