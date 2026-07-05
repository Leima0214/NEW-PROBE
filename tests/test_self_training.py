import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probe.engine.self_training import DomainAlignmentHead, SimSiamHeads, probe_pretrain_step
from probe.models.prompts import PrototypeState


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def encode(self, images, prototype_state):
        features = self.projection(images)
        prompts = features[:, None, :].repeat(1, 2, 1)
        return features, features[:, None, :], prompts


def test_pretraining_uses_both_source_views():
    torch.manual_seed(0)
    model = Encoder()
    heads = SimSiamHeads(embed_dim=4, hidden_dim=8, out_dim=8)
    alignment = DomainAlignmentHead(embed_dim=4, projection_dim=4)
    alignment.requires_grad_(False)
    state = PrototypeState(torch.zeros(4), torch.eye(4), torch.zeros(2, 4))
    source1 = torch.randn(3, 4, requires_grad=True)
    source2 = torch.randn(3, 4, requires_grad=True)
    target1 = torch.randn(3, 4, requires_grad=True)
    target2 = torch.randn(3, 4, requires_grad=True)

    probe_pretrain_step(
        model,
        heads,
        alignment,
        source1,
        source2,
        target1,
        target2,
        state,
        optimizer=None,
    )

    assert source1.grad is not None
    assert source2.grad is not None
