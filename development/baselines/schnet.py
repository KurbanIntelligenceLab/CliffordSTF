"""SchNet baseline (PyG-native), adapted to the ``data_wrapper`` interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import nn
from torch_geometric.nn.models import SchNet

if TYPE_CHECKING:
    import torch
    from omegaconf import DictConfig
    from torch_geometric.data import Data


class SchNetDataWrapper(nn.Module):
    """Adapt :class:`torch_geometric.nn.models.SchNet` to ``data_wrapper``."""

    def __init__(self, schnet: SchNet) -> None:
        super().__init__()
        self.schnet = schnet

    def forward(self, data: Data) -> torch.Tensor:
        energy = self.schnet(data.z, data.pos, data.batch)
        return cast("torch.Tensor", energy).view(-1)


def build_schnet(cfg: DictConfig) -> SchNetDataWrapper:
    """Build the SchNet baseline from a resolved config."""
    m = cfg.model
    schnet = SchNet(
        hidden_channels=m.hidden_channels,
        num_filters=m.num_filters,
        num_interactions=m.num_interactions,
        num_gaussians=m.num_gaussians,
        cutoff=m.cutoff,
    )
    return SchNetDataWrapper(schnet)
