"""PyG ``Data`` -> SchNetPack input-dict conversion.

Shared utility for SchNetPack-backed baselines (currently :mod:`baselines.painn`).
Uses :mod:`schnetpack.properties` for version-safe key names; falls back to
the canonical underscore-prefixed strings when the module is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

if TYPE_CHECKING:
    from torch_geometric.data import Data


def pyg_to_schnetpack(data: Data, cutoff: float) -> dict[Any, torch.Tensor]:
    """Convert a PyG ``Data`` / ``Batch`` into a SchNetPack-compatible dict.

    Args:
        data: PyG batch with ``z`` (or ``atomic_numbers``), ``pos``, and
            (optionally) ``edge_index``, ``cell``, ``pbc``.
        cutoff: Radius for ``radius_graph`` when ``edge_index`` is missing.

    Returns:
        Dict keyed by ``schnetpack.properties.*`` symbols, suitable for
        SchNetPack representation modules.
    """
    try:
        from schnetpack import properties as spk_props
    except ImportError:
        spk_props = None

    pos = data.pos
    z = data.z if hasattr(data, "z") else data.atomic_numbers

    batch = (
        data.batch
        if hasattr(data, "batch") and data.batch is not None
        else torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
    )

    if not hasattr(data, "edge_index") or data.edge_index is None:
        from torch_cluster import radius_graph

        edge_index = radius_graph(pos, r=cutoff, batch=batch)
    else:
        edge_index = data.edge_index

    n_atoms = torch.bincount(batch)
    r_ij = pos[edge_index[1]] - pos[edge_index[0]]

    if spk_props is not None:
        inputs: dict[Any, torch.Tensor] = {
            spk_props.Z: cast("torch.Tensor", z).long().view(-1),
            spk_props.R: pos,
            spk_props.Rij: r_ij,
            spk_props.idx_i: edge_index[0],
            spk_props.idx_j: edge_index[1],
            spk_props.n_atoms: n_atoms,
            spk_props.idx_m: batch,
        }
        if hasattr(data, "cell") and data.cell is not None:
            inputs[spk_props.cell] = data.cell
        if hasattr(data, "pbc") and data.pbc is not None:
            inputs[spk_props.pbc] = data.pbc
    else:
        inputs = {
            "_atomic_numbers": cast("torch.Tensor", z).long().view(-1),
            "_positions": pos,
            "_Rij": r_ij,
            "_idx_i": edge_index[0],
            "_idx_j": edge_index[1],
            "_n_atoms": n_atoms,
            "_idx_m": batch,
        }
        if hasattr(data, "cell") and data.cell is not None:
            inputs["_cell"] = data.cell
        if hasattr(data, "pbc") and data.pbc is not None:
            inputs["_pbc"] = data.pbc

    return inputs
