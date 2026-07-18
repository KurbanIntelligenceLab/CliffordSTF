"""TorchMD-Net baseline (equivariant-transformer / tensornet).

Wraps :func:`torchmdnet.models.model.create_model` so the cliffordstf
trainer can call it with a PyG ``Data`` object. ``torchmdnet`` is
imported lazily inside the wrapper's constructor so importing this
module without the ``cliffordstf[baselines]`` extra installed remains
safe; the ``ImportError`` fires only when :func:`build_torchmdnet` is
actually invoked.

TorchMD-Net trains with ``derivative=True``: forces are computed inside
the backbone via autograd on ``pos`` and returned as ``-dE/dR``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import nn

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch_geometric.data import Data


class TorchMDNetDataWrapper(nn.Module):
    """Adapt :mod:`torchmdnet`'s ``create_model`` output to ``data_wrapper``."""

    def __init__(
        self,
        *,
        model_arch: str,
        embedding_dimension: int,
        num_layers: int,
        num_rbf: int,
        rbf_type: str,
        trainable_rbf: bool,
        activation: str,
        cutoff_lower: float,
        cutoff_upper: float,
        max_z: int,
        max_num_neighbors: int,
        num_heads: int,
        distance_influence: str,
        neighbor_embedding: bool,
        attn_activation: str,
        output_model: str,
        reduce_op: str,
        vector_cutoff: bool,
        equivariance_invariance_group: str,
        static_shapes: bool,
    ) -> None:
        super().__init__()
        from torchmdnet.models.model import create_model

        args: dict[str, object] = {
            "model": model_arch,
            "embedding_dimension": embedding_dimension,
            "num_layers": num_layers,
            "num_rbf": num_rbf,
            "rbf_type": rbf_type,
            "trainable_rbf": trainable_rbf,
            "activation": activation,
            "cutoff_lower": float(cutoff_lower),
            "cutoff_upper": float(cutoff_upper),
            "max_z": max_z,
            "max_num_neighbors": max_num_neighbors,
            "num_heads": num_heads,
            "distance_influence": distance_influence,
            "neighbor_embedding": neighbor_embedding,
            "attn_activation": attn_activation,
            "output_model": output_model,
            "reduce_op": reduce_op,
            "derivative": True,
            "prior_model": None,
            "atom_filter": -1,
            "aggr": "add",
            "precision": 32,
            "vector_cutoff": vector_cutoff,
            "equivariance_invariance_group": equivariance_invariance_group,
            "static_shapes": static_shapes,
        }
        self._backbone = create_model(
            args,
            prior_model=None,
            mean=torch.tensor(0.0),
            std=torch.tensor(1.0),
        )

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        z = data.z if hasattr(data, "z") else data.atomic_numbers
        pos = data.pos
        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(cast("torch.Tensor", z).size(0), dtype=torch.long, device=pos.device)
        )
        pos = pos.clone().requires_grad_(True)
        with torch.enable_grad():
            energy, neg_dy = self._backbone(z, pos, batch, box=None)
        forces = neg_dy if neg_dy is not None else torch.zeros_like(pos)
        energy = cast("torch.Tensor", energy).view(-1)
        return energy, forces


def build_torchmdnet(cfg: DictConfig) -> TorchMDNetDataWrapper:
    """Build the TorchMD-Net baseline from a resolved config."""
    m = cfg.model
    return TorchMDNetDataWrapper(
        model_arch=m.get("model_arch", "equivariant-transformer"),
        embedding_dimension=m.get("embedding_dimension", 128),
        num_layers=m.get("num_layers", 6),
        num_rbf=m.get("num_rbf", 64),
        rbf_type=m.get("rbf_type", "expnorm"),
        trainable_rbf=m.get("trainable_rbf", False),
        activation=m.get("activation", "silu"),
        cutoff_lower=m.get("cutoff_lower", 0.0),
        cutoff_upper=m.get("cutoff_upper", 6.0),
        max_z=m.get("max_z", 100),
        max_num_neighbors=m.get("max_num_neighbors", 50),
        num_heads=m.get("num_heads", 8),
        distance_influence=m.get("distance_influence", "both"),
        neighbor_embedding=m.get("neighbor_embedding", True),
        attn_activation=m.get("attn_activation", "silu"),
        output_model=m.get("output_model", "Scalar"),
        reduce_op=m.get("reduce_op", "add"),
        vector_cutoff=m.get("vector_cutoff", False),
        equivariance_invariance_group=m.get("equivariance_invariance_group", "O(3)"),
        static_shapes=m.get("static_shapes", False),
    )
