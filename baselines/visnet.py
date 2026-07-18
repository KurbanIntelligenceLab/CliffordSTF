"""ViSNet baseline (PyG-native), adapted to the ``data_wrapper`` interface.

PyG's :class:`torch_geometric.nn.models.ViSNet` consumes ``(z, pos, batch)``
and returns ``(energy, forces_or_none)``. When ``derivative=True`` it
internally enables autograd on ``pos`` and returns the gradient-based
forces; when ``derivative=False`` it returns ``(energy, None)``.

:class:`ViSNetDataWrapper` is a thin adapter that:

- pulls ``data.z / data.pos / data.batch`` off the PyG batch,
- threads the ``derivative`` flag through (derived from
  ``cfg.dataset.task_type`` at factory time),
- returns either ``(energy, forces)`` (force tasks) or bare ``energy``
  (scalar tasks) per the cliffordstf ``data_wrapper`` contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torch_geometric.nn.models import ViSNet

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch_geometric.data import Data


class ViSNetDataWrapper(nn.Module):
    """Adapt :class:`ViSNet` to the cliffordstf ``data_wrapper`` interface."""

    def __init__(self, visnet: ViSNet, *, derivative: bool) -> None:
        super().__init__()
        self.visnet = visnet
        self.derivative = derivative

    def forward(self, data: Data) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        z = data.z
        pos = data.pos
        batch = data.batch
        if self.derivative:
            pos = pos.clone().requires_grad_(True)
            with torch.enable_grad():
                energy, forces = self.visnet(z, pos, batch)
            return cast("torch.Tensor", energy).view(-1), cast("torch.Tensor", forces)
        energy, _ = self.visnet(z, pos, batch)
        return cast("torch.Tensor", energy).view(-1)


_FORCE_TASK_TYPES = frozenset({"energy_forces", "s2ef"})


def build_visnet(cfg: DictConfig) -> ViSNetDataWrapper:
    """Build the ViSNet baseline from a resolved config."""
    m = cfg.model
    task_type = cfg.dataset.get("task_type", "scalar") if "dataset" in cfg else "scalar"
    derivative = task_type in _FORCE_TASK_TYPES
    visnet = ViSNet(
        lmax=m.get("lmax", 1),
        vecnorm_type=m.get("vecnorm_type", None),
        trainable_vecnorm=m.get("trainable_vecnorm", False),
        num_heads=m.get("num_heads", 8),
        num_layers=m.get("num_layers", 6),
        hidden_channels=m.get("hidden_channels", 96),
        num_rbf=m.get("num_rbf", 32),
        trainable_rbf=m.get("trainable_rbf", False),
        max_z=m.get("max_z", 100),
        cutoff=m.get("cutoff", 5.0),
        max_num_neighbors=m.get("max_num_neighbors", 32),
        vertex=m.get("vertex", False),
        reduce_op=m.get("reduce_op", "sum"),
        derivative=derivative,
    )
    return ViSNetDataWrapper(visnet, derivative=derivative)
