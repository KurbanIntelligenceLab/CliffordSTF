"""ICTP baseline at L=1 / L=2 / L=3.

Wraps the upstream ``ictp`` package's CartesianMACE representation +
linear readouts so the cliffordstf trainer can call it with a PyG
``Data`` object. ``ictp`` is imported lazily inside the constructor
so this module is safe to import without the upstream package
installed; the ``ImportError`` only fires when one of the
``build_ictp_l*`` factories is actually invoked.

Uses the "full" CartesianMACE variant
(``coupled_product_feats=True``, ``symmetric_product=False``). The
readout pattern mirrors ``ictp.build_model``: one ``LinearLayer`` per
interaction except the last, which gets a small MLP
(``LinearLayer -> RescaledSiLULayer -> LinearLayer``) with the
trailing activation stripped. Forces are computed via autograd from
``-dE/dR`` on every call when ``compute_forces`` is True.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch_geometric.nn import radius_graph
from torch_scatter import scatter

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch_geometric.data import Data


_FORCE_TASK_TYPES = frozenset({"energy_forces", "s2ef"})


class ICTPDataWrapper(nn.Module):
    """Adapt ``ictp.CartesianMACE`` + linear readouts to ``data_wrapper``."""

    def __init__(
        self,
        *,
        l_max_hidden_feats: int,
        l_max_edge_attrs: int,
        n_hidden_feats: int,
        n_product_feats: int,
        n_interactions: int,
        correlation: int,
        n_basis: int,
        n_polynomial_cutoff: int,
        r_cutoff: float,
        n_species: int,
        avg_n_neighbors: float,
        radial_mlp: list[int],
        readout_mlp: list[int],
        max_neighbors: int,
        compute_forces: bool,
    ) -> None:
        super().__init__()
        from ictp.nn.layers import LinearLayer, RescaledSiLULayer
        from ictp.nn.representations import CartesianMACE

        self.r_cutoff = r_cutoff
        self.max_neighbors = max_neighbors
        self.n_species = n_species
        self.n_hidden_feats = n_hidden_feats
        self.compute_forces = compute_forces

        self._representation = CartesianMACE(
            r_cutoff=r_cutoff,
            n_basis=n_basis,
            n_polynomial_cutoff=n_polynomial_cutoff,
            n_species=n_species,
            n_hidden_feats=n_hidden_feats,
            n_product_feats=n_product_feats,
            coupled_product_feats=True,
            symmetric_product=False,
            l_max_hidden_feats=l_max_hidden_feats,
            l_max_edge_attrs=l_max_edge_attrs,
            avg_n_neighbors=avg_n_neighbors,
            correlation=correlation,
            n_interactions=n_interactions,
            radial_MLP=radial_mlp,
            use_charge_embedding=False,
        )

        readouts: list[nn.Module] = []
        for i in range(n_interactions):
            if i == n_interactions - 1:
                layers: list[nn.Module] = []
                in_sizes = [n_hidden_feats, *readout_mlp]
                out_sizes = [*readout_mlp, 1]
                for in_s, out_s in zip(in_sizes, out_sizes, strict=False):
                    layers.append(LinearLayer(in_s, out_s))
                    layers.append(RescaledSiLULayer())
                readouts.append(nn.Sequential(*layers[:-1]))
            else:
                readouts.append(LinearLayer(n_hidden_feats, 1))
        self._readouts = nn.ModuleList(readouts)

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        pos = data.pos
        if not pos.requires_grad:
            pos = pos.clone().requires_grad_(True)
        batch = data.batch
        z = data.z.long()

        edge_index = radius_graph(
            pos, r=self.r_cutoff, batch=batch, max_num_neighbors=self.max_neighbors
        )
        node_attrs = torch.zeros(z.size(0), self.n_species, device=z.device, dtype=pos.dtype)
        node_attrs.scatter_(1, torch.clamp(z - 1, min=0).unsqueeze(-1), 1.0)

        vectors = pos[edge_index[0]] - pos[edge_index[1]]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        graph = {
            "node_attrs": node_attrs,
            "edge_index": edge_index,
            "vectors": vectors,
            "lengths": lengths,
        }

        node_feats_list = self._representation(graph)
        node_energies = torch.zeros(pos.size(0), 1, device=pos.device, dtype=pos.dtype)
        for feats, readout in zip(node_feats_list, self._readouts, strict=False):
            node_energies = node_energies + readout(feats)

        energy = scatter(node_energies.squeeze(-1), batch, dim=0, reduce="sum")

        if self.compute_forces:
            grad = torch.autograd.grad(
                energy.sum(),
                pos,
                create_graph=self.training,
                retain_graph=self.training,
            )
            forces = -grad[0]
        else:
            forces = torch.zeros_like(pos)
        return energy, forces


def _build_ictp(cfg: DictConfig) -> ICTPDataWrapper:
    """Shared builder for every ICTP L-variant."""
    m = cfg.model
    task_type = cfg.dataset.get("task_type", "scalar") if "dataset" in cfg else "scalar"
    return ICTPDataWrapper(
        l_max_hidden_feats=m.l_max_hidden_feats,
        l_max_edge_attrs=m.get("l_max_edge_attrs", 3),
        n_hidden_feats=m.n_hidden_feats,
        n_product_feats=m.get("n_product_feats", m.n_hidden_feats),
        n_interactions=m.get("n_interactions", 2),
        correlation=m.get("correlation", 3),
        n_basis=m.get("n_basis", 8),
        n_polynomial_cutoff=m.get("n_polynomial_cutoff", 5),
        r_cutoff=m.get("cutoff", 6.0),
        n_species=m.get("n_species", 100),
        avg_n_neighbors=m.get("avg_n_neighbors", 50.0),
        radial_mlp=list(m.get("radial_MLP", [64, 64, 64])),
        readout_mlp=list(m.get("readout_MLP", [16])),
        max_neighbors=m.get("max_neighbors", 50),
        compute_forces=task_type in _FORCE_TASK_TYPES,
    )


def build_ictp_l1(cfg: DictConfig) -> ICTPDataWrapper:
    return _build_ictp(cfg)


def build_ictp_l2(cfg: DictConfig) -> ICTPDataWrapper:
    return _build_ictp(cfg)


def build_ictp_l3(cfg: DictConfig) -> ICTPDataWrapper:
    return _build_ictp(cfg)
