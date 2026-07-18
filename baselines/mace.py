"""MACE baseline at L=1 / L=2 / L=3 (e3nn-backed).

Wraps :class:`mace.modules.MACE` so the cliffordstf trainer can call it
with a PyG ``Data`` object. ``mace`` and ``e3nn`` are imported lazily
inside the constructor so this module is safe to import without the
``cliffordstf[baselines]`` extra installed.

All four variants (``mace_l1``, ``mace_l2``, ``mace_l3``,
``mace_l2_10m``) share a single ``_build_mace`` factory; they differ
only by ``max_ell`` / ``hidden_irreps`` in the packaged YAMLs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import radius_graph

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch_geometric.data import Data


_FORCE_TASK_TYPES = frozenset({"energy_forces", "s2ef"})


class MACEDataWrapper(nn.Module):
    """Adapt ``mace.modules.MACE`` to ``data_wrapper``."""

    def __init__(
        self,
        *,
        max_ell: int,
        hidden_irreps: str,
        num_interactions: int,
        correlation: int,
        num_bessel: int,
        num_polynomial_cutoff: int,
        cutoff: float,
        max_neighbors: int,
        num_elements: int,
        mlp_irreps: str,
        avg_num_neighbors: float,
        compute_forces: bool,
    ) -> None:
        super().__init__()
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([slice])
        from e3nn import o3
        from mace.modules import MACE, interaction_classes

        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.num_elements = num_elements
        self.compute_forces = compute_forces

        interaction_cls = interaction_classes["RealAgnosticResidualInteractionBlock"]
        self._backbone = MACE(
            r_max=cutoff,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            max_ell=max_ell,
            interaction_cls=interaction_cls,
            interaction_cls_first=interaction_cls,
            num_interactions=num_interactions,
            num_elements=num_elements,
            hidden_irreps=o3.Irreps(hidden_irreps),
            MLP_irreps=o3.Irreps(mlp_irreps),
            atomic_energies=np.zeros(num_elements),
            avg_num_neighbors=avg_num_neighbors,
            atomic_numbers=list(range(1, num_elements + 1)),
            correlation=correlation,
            gate=torch.nn.functional.silu,
        )

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        from mace.tools.torch_tools import to_one_hot

        pos = data.pos
        if not pos.requires_grad:
            pos = pos.clone().requires_grad_(True)
        batch = data.batch
        z = data.z.long()

        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_neighbors
        )
        node_attrs = to_one_hot(
            torch.clamp(z - 1, min=0).unsqueeze(-1),
            num_classes=self.num_elements,
        )
        num_graphs = int(batch.max().item()) + 1
        mace_data = {
            "positions": pos,
            "node_attrs": node_attrs,
            "edge_index": edge_index,
            "batch": batch,
            "ptr": torch.cat(
                [
                    torch.zeros(1, dtype=torch.long, device=batch.device),
                    torch.bincount(batch).cumsum(0),
                ]
            ),
            "cell": torch.zeros(num_graphs, 3, 3, dtype=pos.dtype, device=pos.device),
            "shifts": torch.zeros(edge_index.shape[1], 3, dtype=pos.dtype, device=pos.device),
            "unit_shifts": torch.zeros(edge_index.shape[1], 3, dtype=pos.dtype, device=pos.device),
        }
        out = self._backbone(mace_data, training=self.training, compute_force=self.compute_forces)
        energy = out["energy"].view(-1)
        forces = out.get("forces")
        if forces is None:
            forces = torch.zeros_like(pos)
        return energy, forces


def _build_mace(cfg: DictConfig) -> MACEDataWrapper:
    """Shared builder for every MACE L-variant."""
    m = cfg.model
    task_type = cfg.dataset.get("task_type", "scalar") if "dataset" in cfg else "scalar"
    return MACEDataWrapper(
        max_ell=m.max_ell,
        hidden_irreps=m.hidden_irreps,
        num_interactions=m.get("num_interactions", 2),
        correlation=m.get("correlation", 3),
        num_bessel=m.get("num_bessel", 8),
        num_polynomial_cutoff=m.get("num_polynomial_cutoff", 5),
        cutoff=m.get("cutoff", 6.0),
        max_neighbors=m.get("max_neighbors", 50),
        num_elements=m.get("num_elements", 100),
        mlp_irreps=m.get("MLP_irreps", "16x0e"),
        avg_num_neighbors=m.get("avg_num_neighbors", 50.0),
        compute_forces=task_type in _FORCE_TASK_TYPES,
    )


def build_mace_l1(cfg: DictConfig) -> MACEDataWrapper:
    return _build_mace(cfg)


def build_mace_l2(cfg: DictConfig) -> MACEDataWrapper:
    return _build_mace(cfg)


def build_mace_l3(cfg: DictConfig) -> MACEDataWrapper:
    return _build_mace(cfg)


def build_mace_l2_10m(cfg: DictConfig) -> MACEDataWrapper:
    return _build_mace(cfg)
