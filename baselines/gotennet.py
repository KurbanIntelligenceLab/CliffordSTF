"""GotenNet baseline (Geometric Tensor Network).

Wraps the upstream ``gotennet`` package's encoder with a graph-level
pool + small MLP head + optional direct-forces head. The ``gotennet``
package is imported lazily inside the constructor so this module is
safe to import without the ``cliffordstf[baselines]`` extra installed.

If ``direct_forces`` is ``True``, per-atom forces are read directly off
the L=1 channel of the encoder's vector features and the wrapper
returns ``(energy, forces)``. Otherwise it returns bare energy and the
trainer computes forces from ``-dE/dR`` via autograd.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn.pool import radius_graph

if TYPE_CHECKING:
    import torch
    from omegaconf import DictConfig
    from torch_geometric.data import Data


_POOLING_MAP: dict[str, object] = {
    "mean": global_mean_pool,
    "add": global_add_pool,
}

_TASK_POOLING: dict[str, str] = {
    "scalar": "mean",
    "energy_forces": "add",
    "s2ef": "add",
    "is2re": "add",
}


class GotenNetDataWrapper(nn.Module):
    """GotenNet encoder + pool + MLP head, ``data_wrapper`` interface."""

    def __init__(
        self,
        *,
        n_atom_basis: int,
        n_interactions: int,
        cutoff: float,
        max_num_neighbors: int,
        cutoff_fn_name: str,
        pooling: str,
        task_type: str,
        direct_forces: bool,
    ) -> None:
        super().__init__()
        self.cutoff_val = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.direct_forces = direct_forces

        resolved_pool = _TASK_POOLING.get(task_type, "mean") if pooling == "auto" else pooling
        if resolved_pool not in _POOLING_MAP:
            raise ValueError(f"Unknown pooling '{resolved_pool}'. Use 'mean' or 'add'.")
        self.pool = _POOLING_MAP[resolved_pool]

        from gotennet.models.components.layers import CosineCutoff, PolynomialCutoff

        if cutoff_fn_name.lower() in ("cosine", "cosinecutoff"):
            cutoff_fn: nn.Module = CosineCutoff(cutoff)
        elif cutoff_fn_name.lower() in ("polynomial", "polynomialcutoff"):
            cutoff_fn = PolynomialCutoff(cutoff)
        else:
            raise ValueError(
                f"Unknown cutoff_fn_name={cutoff_fn_name!r}. Use 'cosine' or 'polynomial'."
            )

        self.mode: str
        try:
            from gotennet import GotenNetWrapper

            self.encoder: nn.Module = GotenNetWrapper(
                n_atom_basis=n_atom_basis,
                n_interactions=n_interactions,
                cutoff_fn=cutoff_fn,
                max_num_neighbors=max_num_neighbors,
            )
            self.mode = "wrapper"
        except ImportError:
            from gotennet import GotenNet

            self.encoder = GotenNet(
                n_atom_basis=n_atom_basis,
                n_interactions=n_interactions,
                cutoff_fn=cutoff_fn,
            )
            self.mode = "base"

        self.head = nn.Sequential(
            nn.Linear(n_atom_basis, n_atom_basis),
            nn.SiLU(),
            nn.Linear(n_atom_basis, 1),
        )
        self.force_head = nn.Linear(n_atom_basis, 1, bias=False) if self.direct_forces else None

    def forward(self, data: Data) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "wrapper":
            h, x_vec = self.encoder(data)
        else:
            edge_index = radius_graph(
                data.pos,
                r=self.cutoff_val,
                batch=data.batch,
                max_num_neighbors=self.max_num_neighbors,
            )
            row, col = edge_index
            edge_vec = data.pos[col] - data.pos[row]
            edge_diff = edge_vec.norm(dim=-1)
            h, x_vec = self.encoder(data.z, edge_index, edge_diff, edge_vec)

        g = self.pool(h, data.batch)
        energy = self.head(g).view(-1)

        if self.direct_forces and self.force_head is not None:
            forces = self.force_head(x_vec[:, :3, :]).squeeze(-1)
            return energy, forces
        return energy


def build_gotennet(cfg: DictConfig) -> GotenNetDataWrapper:
    """Build the GotenNet baseline from a resolved config."""
    m = cfg.model
    task_type = cfg.dataset.get("task_type", "scalar") if "dataset" in cfg else "scalar"
    return GotenNetDataWrapper(
        n_atom_basis=m.n_atom_basis,
        n_interactions=m.n_interactions,
        cutoff=m.cutoff,
        max_num_neighbors=m.max_num_neighbors,
        cutoff_fn_name=m.get("cutoff_fn_name", "cosine"),
        pooling=m.get("pooling", "auto"),
        task_type=task_type,
        direct_forces=m.get("direct_forces", True),
    )
