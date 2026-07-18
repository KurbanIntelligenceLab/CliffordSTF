"""PyG ``Data`` -> NequIP ``AtomicDataDict`` conversion.

Shared utility for the NequIP baseline (:mod:`baselines.nequip`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch_geometric.data import Data


def pyg_to_atomic_data(
    data: Data, r_max: float, type_names: list[str]
) -> dict[str, torch.Tensor]:
    """Convert a PyG ``Data`` / ``Batch`` into a NequIP-compatible dict."""
    from torch_cluster import radius_graph

    pos = data.pos
    z = data.z if hasattr(data, "z") else data.atomic_numbers
    batch = (
        data.batch
        if hasattr(data, "batch") and data.batch is not None
        else torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
    )

    device = pos.device
    dtype = pos.dtype
    n_graphs = int(batch.max().item()) + 1

    edge_index = radius_graph(pos, r=r_max, batch=batch)

    num_types = len(type_names)
    atom_types = (z.long().flatten() - 1).clamp(0, num_types - 1)

    cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(n_graphs, -1, -1) * 100.0
    pbc = torch.zeros(n_graphs, 3, dtype=torch.bool, device=device)
    edge_cell_shift = torch.zeros(edge_index.size(1), 3, dtype=dtype, device=device)

    sender, receiver = edge_index[0], edge_index[1]
    edge_vectors = pos[receiver] - pos[sender]
    edge_lengths = torch.linalg.norm(edge_vectors, dim=-1, keepdim=True)

    num_atoms = torch.bincount(batch, minlength=n_graphs)
    ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=device), num_atoms.cumsum(0)])

    return {
        "pos": pos,
        "atomic_numbers": z.long().view(-1, 1),
        "atom_types": atom_types.view(-1, 1),
        "edge_index": edge_index,
        "edge_cell_shift": edge_cell_shift,
        "edge_vectors": edge_vectors,
        "edge_lengths": edge_lengths.squeeze(-1),
        "cell": cell,
        "pbc": pbc,
        "batch": batch,
        "ptr": ptr,
        "num_atoms": num_atoms,
    }
