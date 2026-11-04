import numpy as np
import torch

from cascade.models.graph_latent_ranker import GraphLatentRanker


def inputs():
    rng=np.random.default_rng(2)
    text=rng.normal(size=(6,384)).astype(np.float32);text[0]=0
    graph=rng.normal(size=(6,1154)).astype(np.float32);graph[0]=0
    latent={key:rng.normal(size=(6,64)).astype(np.float32) for key in ('bpr','pca')}
    known={'bpr':np.array([0,1,1,0,0,1]),'pca':np.array([0,1,1,1,1,1])}
    return text,np.array([0,1,2,1,0,0]),np.array([0,1,2,1,0,0]),graph,latent,known


def model(arm,bank=None):
    values=list(inputs())
    if bank is not None:values[3]=bank
    torch.manual_seed(19)
    return GraphLatentRanker(*values,9,6,arm).eval()


def batch():
    return (torch.tensor([[1,3,0],[2,4,5]]),torch.tensor([[0,1,2],[0,2,3]]),
        torch.tensor([[.3,.2,-torch.inf],[.4,.2,.1]]),torch.ones(2,3,9),torch.ones(2,3,6))


def test_all_arms_have_identical_initial_states_and_epoch_zero_scores():
    reference=model('R1').state_dict()
    for arm in GraphLatentRanker.ARMS:
        m=model(arm)
        assert reference.keys()==m.state_dict().keys()
        assert all(torch.equal(reference[k],m.state_dict()[k]) for k in reference)
        assert torch.equal(m(*batch()),batch()[2])
        assert not any(k in m.state_dict() for k in ('text','leaf','store','graph_bank','bpr_factors','pca_factors'))


def test_unavailable_latents_are_masked_after_adapter_bias():
    m=model('R3')
    with torch.no_grad():m.latent_adapter.bias.fill_(7)
    value=m._factor_block(torch.tensor([[3,4,0]]),torch.tensor([[0,3,4]]),'bpr')
    assert not value.any()


def test_self_only_encoder_cannot_read_neighbours_but_real_encoder_can():
    graph=inputs()[3];changed=graph.copy();changed[:,384:]+=10
    a,b=model('R6',graph),model('R6',changed)
    c,d=model('R7',graph),model('R7',changed)
    with torch.no_grad():
        for m in (a,b,c,d):m.head[-1].weight.fill_(.1)
    assert torch.equal(a(*batch()),b(*batch()))
    x,y=c(*batch()),d(*batch())
    assert not torch.allclose(x[torch.isfinite(x)],y[torch.isfinite(y)])


def test_padding_is_masked_after_text_and_graph_adapters():
    m=model('R7')
    with torch.no_grad():m.head[-1].weight.fill_(.1)
    c,h,s,common,graph=batch()
    assert torch.allclose(m(c,h,s,common,graph),m(c,h[:,1:],s,common,graph),atol=1e-6)
