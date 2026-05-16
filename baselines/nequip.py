"""NequIP baseline (e3nn-backed).

Wraps :class:`nequip.model.NequIPGNNModel` so the cliffordstf trainer
can call it with a PyG ``Data`` object. ``nequip`` and its ``e3nn`` /
``torch.serialization`` dependencies are imported lazily inside the
constructor; importing this module without the
``cliffordstf[baselines]`` extra installed is therefore safe.

The wrapper always returns ``(energy, forces)``. When the resolved
``cfg.dataset.task_type`` is scalar, ``do_derivatives`` is left False
and the forces tensor is a zero placeholder of the right shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from baselines._nequip_utils import pyg_to_atomic_data

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch_geometric.data import Data


_FORCE_TASK_TYPES = frozenset({"energy_forces", "s2ef"})


class NequIPDataWrapper(nn.Module):
    """Adapt NequIP's ``NequIPGNNModel`` to ``data_wrapper``."""

    def __init__(
        self,
        *,
        r_max: float,
        num_layers: int,
        l_max: int,
        parity: bool,
        num_features: int,
        type_embed_num_features: int | None,
        radial_mlp_depth: int,
        radial_mlp_width: int,
        num_bessels: int,
        polynomial_cutoff_p: int,
        max_num_elements: int,
        do_derivatives: bool,
    ) -> None:
        super().__init__()
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([slice])
        from nequip.model import NequIPGNNModel
        from nequip.utils.global_state import set_global_state

        set_global_state(allow_tf32=False)
        self.r_max = r_max
        self.do_derivatives = do_derivatives
        self.type_names: list[str] = [str(i) for i in range(1, max_num_elements + 1)]
        self._backbone = NequIPGNNModel(
            r_max=r_max,
            type_names=self.type_names,
            num_layers=num_layers,
            l_max=l_max,
            parity=parity,
            num_features=num_features,
            type_embed_num_features=type_embed_num_features or num_features,
            radial_mlp_depth=radial_mlp_depth,
            radial_mlp_width=radial_mlp_width,
            num_bessels=num_bessels,
            polynomial_cutoff_p=polynomial_cutoff_p,
            avg_num_neighbors=50.0,
            seed=42,
            model_dtype="float32",
            do_derivatives=do_derivatives,
        )

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        atomic = pyg_to_atomic_data(data, self.r_max, self.type_names)
        backbone_dtype = next(self._backbone.parameters()).dtype
        for k in ("pos", "edge_vectors", "edge_lengths", "edge_cell_shift", "cell"):
            if k in atomic and atomic[k].is_floating_point():
                atomic[k] = atomic[k].to(backbone_dtype)
        if not atomic["pos"].requires_grad:
            atomic["pos"] = atomic["pos"].clone().requires_grad_(True)
        # Force the position-branch of ForceStressOutput so training graphs
        # are retained; the pre-computed edge_vectors branch is inference-only.
        if self.do_derivatives:
            atomic.pop("edge_vectors", None)
            atomic.pop("edge_lengths", None)
        out = self._backbone(atomic)
        energy = out["total_energy"].view(-1)
        forces = out.get("forces")
        if forces is None:
            forces = torch.zeros_like(atomic["pos"], device=energy.device, dtype=energy.dtype)
        return energy, forces


def build_nequip(cfg: DictConfig) -> NequIPDataWrapper:
    """Build the NequIP baseline from a resolved config."""
    m = cfg.model
    task_type = cfg.dataset.get("task_type", "scalar") if "dataset" in cfg else "scalar"
    return NequIPDataWrapper(
        r_max=m.get("r_max", 6.0),
        num_layers=m.get("num_layers", 4),
        l_max=m.get("l_max", 1),
        parity=m.get("parity", True),
        num_features=m.get("num_features", 32),
        type_embed_num_features=m.get("type_embed_num_features", None),
        radial_mlp_depth=m.get("radial_mlp_depth", 2),
        radial_mlp_width=m.get("radial_mlp_width", 64),
        num_bessels=m.get("num_bessels", 8),
        polynomial_cutoff_p=m.get("polynomial_cutoff_p", 6),
        max_num_elements=m.get("max_num_elements", 90),
        do_derivatives=task_type in _FORCE_TASK_TYPES,
    )
