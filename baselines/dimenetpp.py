"""DimeNet++ baseline (PyG-native), adapted to the ``data_wrapper`` interface.

PyG's :class:`torch_geometric.nn.models.DimeNetPlusPlus` consumes
``(z, pos, batch)`` and returns a per-graph energy tensor.
:class:`DimeNetPPDataWrapper` is the thin adapter that lets the
cliffordstf trainer call it as ``model(data) -> energy`` (forces are
derived via autograd by :func:`cliffordstf.training.losses.compute_loss`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import nn
from torch_geometric.nn.models import DimeNetPlusPlus

if TYPE_CHECKING:
    import torch
    from omegaconf import DictConfig
    from torch_geometric.data import Data


class DimeNetPPDataWrapper(nn.Module):
    """Adapt :class:`DimeNetPlusPlus` to the ``data_wrapper`` interface."""

    def __init__(self, dimenetpp: DimeNetPlusPlus) -> None:
        super().__init__()
        self.dimenetpp = dimenetpp

    def forward(self, data: Data) -> torch.Tensor:
        energy = self.dimenetpp(data.z, data.pos, data.batch)
        return cast("torch.Tensor", energy).view(-1)


def build_dimenetpp(cfg: DictConfig) -> DimeNetPPDataWrapper:
    """Build the DimeNet++ baseline from a resolved config."""
    m = cfg.model
    dimenetpp = DimeNetPlusPlus(
        hidden_channels=m.hidden_channels,
        out_channels=m.get("out_channels", 1),
        num_blocks=m.num_blocks,
        int_emb_size=m.get("int_emb_size", 64),
        basis_emb_size=m.basis_emb_size,
        out_emb_channels=m.get("out_emb_channels", 256),
        num_spherical=m.num_spherical,
        num_radial=m.num_radial,
        cutoff=m.cutoff,
        envelope_exponent=m.get("envelope_exponent", 5),
    )
    return DimeNetPPDataWrapper(dimenetpp)
