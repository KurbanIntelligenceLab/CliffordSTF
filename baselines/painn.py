"""PaiNN baseline (SchNetPack-backed).

Wraps :class:`schnetpack.representation.PaiNN` so the cliffordstf trainer
can call it with a PyG ``Data`` object. ``schnetpack`` is only imported
inside the constructor: importing this module without the
``cliffordstf[baselines]`` extra installed is therefore safe; the import
error fires only when :func:`build_painn` is actually invoked.

If ``direct_forces`` is ``True``, the wrapper learns a 1-channel head on
top of the equivariant vector representation and returns
``(energy, forces)``. Otherwise it returns bare energy and lets the
trainer compute forces from ``-dE/dR`` via autograd.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
import torch.utils.data.dataloader as _dl
from torch import nn

from baselines._schnetpack_utils import pyg_to_schnetpack

# PyTorch >=2.7 renamed ``T_co`` to ``_T_co``. Older SchNetPack imports the
# old name, so patch it before any SchNetPack-side import fires.
if not hasattr(_dl, "T_co") and hasattr(_dl, "_T_co"):
    _dl.T_co = _dl._T_co  # type: ignore[attr-defined]


if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch_geometric.data import Data


class PaiNNDataWrapper(nn.Module):
    """Adapt SchNetPack's PaiNN representation to ``data_wrapper``."""

    def __init__(
        self,
        *,
        hidden_channels: int,
        num_layers: int,
        num_rbf: int,
        cutoff: float,
        direct_forces: bool,
    ) -> None:
        super().__init__()
        from schnetpack.nn import CosineCutoff, GaussianRBF
        from schnetpack.representation import PaiNN

        self.cutoff = cutoff
        self.direct_forces = direct_forces
        self._backbone = PaiNN(
            n_atom_basis=hidden_channels,
            n_interactions=num_layers,
            radial_basis=GaussianRBF(n_rbf=num_rbf, cutoff=cutoff),
            cutoff_fn=CosineCutoff(cutoff),
        )
        if self.direct_forces:
            self.force_head: nn.Linear | None = nn.Linear(hidden_channels, 1, bias=False)
        else:
            self.force_head = None

    def forward(self, data: Data) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from schnetpack import properties as spk_props

        inputs = pyg_to_schnetpack(data, self.cutoff)
        output = self._backbone(inputs)

        energy = _extract_energy(output)
        batch = inputs[spk_props.idx_m]
        energy = _pool_energy(energy, batch)
        energy = energy.view(-1)

        if self.direct_forces and isinstance(output, dict):
            vectors = output.get("vector_representation")
            if vectors is not None and self.force_head is not None:
                forces = self.force_head(vectors).squeeze(-1)
                return energy, forces
        return energy


def _extract_energy(output: Any) -> torch.Tensor:
    """Pull a scalar / per-atom energy tensor out of SchNetPack's output dict."""
    if not isinstance(output, dict):
        return cast("torch.Tensor", output)
    candidate = output.get("scalar_representation") or output.get("energy") or output.get("y")
    if candidate is not None:
        return cast("torch.Tensor", candidate)
    for value in output.values():
        if isinstance(value, torch.Tensor) and value.dim() in (1, 2):
            return value
    raise RuntimeError(f"Could not find energy in PaiNN output. Keys: {list(output.keys())}")


def _pool_energy(energy: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Sum per-atom or mean-of-features energies to per-graph energies."""
    n_graphs = int(batch.max().item()) + 1
    if energy.dim() == 2:
        per_atom = energy.mean(dim=-1)
        per_graph = torch.zeros(n_graphs, device=energy.device, dtype=energy.dtype)
        per_graph.index_add_(0, batch, per_atom)
        return per_graph
    if energy.dim() == 1:
        per_graph = torch.zeros(n_graphs, device=energy.device, dtype=energy.dtype)
        per_graph.index_add_(0, batch, energy)
        return per_graph
    return energy


def build_painn(cfg: DictConfig) -> PaiNNDataWrapper:
    """Build the PaiNN baseline from a resolved config."""
    m = cfg.model
    return PaiNNDataWrapper(
        hidden_channels=m.get("hidden_channels", 128),
        num_layers=m.get("num_layers", 4),
        num_rbf=m.get("num_rbf", 64),
        cutoff=m.get("cutoff", 6.0),
        direct_forces=m.get("direct_forces", True),
    )
