"""Per-batch metric accumulators for the Lightning validation/test loops.

Mirrors the aggregation logic of
:func:`cliffordstf.training.evaluate.evaluate_epoch` so the Lightning
path reports the same scientific metrics:

* Scalar tasks (``cfg.dataset.task_type in SCALAR_TASK_TYPES``):
  ``energy_mae`` (rescaled by ``runtime_stats["std"]`` when provided).
* Energy-forces tasks: ``energy_mae``, ``force_mae``, ``force_cos``,
  ``efwt`` (percentage of structures with energy err < 0.02 and per-atom
  max force err < 0.03).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch_scatter import scatter_max

from cliffordstf.training.evaluate import (
    EFWT_ENERGY_THRESHOLD,
    EFWT_FORCE_THRESHOLD,
    _rescale_energy_mae,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


class ScalarMetricAccumulator:
    """Track running ``energy_mae`` for scalar tasks."""

    def __init__(self) -> None:
        self.total_energy_ae = torch.zeros((), dtype=torch.float64)
        self.n_structures = 0

    def update(self, energy_pred: torch.Tensor, energy_target: torch.Tensor) -> None:
        self.total_energy_ae = self.total_energy_ae.to(energy_pred.device) + (
            energy_pred - energy_target
        ).abs().sum().double()
        self.n_structures += energy_pred.size(0)

    def finalize(self, runtime_stats: Mapping[str, object] | None) -> dict[str, float]:
        n = max(1, self.n_structures)
        return {
            "energy_mae": _rescale_energy_mae(self.total_energy_ae.item() / n, runtime_stats),
        }


class EnergyForcesMetricAccumulator:
    """Track running ``energy_mae`` / ``force_mae`` / ``force_cos`` / ``efwt``."""

    def __init__(self) -> None:
        self.total_energy_ae = torch.zeros((), dtype=torch.float64)
        self.total_force_ae = torch.zeros((), dtype=torch.float64)
        self.total_cos_sim = torch.zeros((), dtype=torch.float64)
        self.total_efwt = torch.zeros((), dtype=torch.float64)
        self.n_structures = 0
        self.n_force_components = 0
        self.n_force_atoms = 0

    def update(
        self,
        energy_pred: torch.Tensor,
        energy_target: torch.Tensor,
        forces_pred: torch.Tensor,
        forces_target: torch.Tensor,
        batch: torch.Tensor,
    ) -> None:
        device = energy_pred.device
        self.total_energy_ae = self.total_energy_ae.to(device)
        self.total_force_ae = self.total_force_ae.to(device)
        self.total_cos_sim = self.total_cos_sim.to(device)
        self.total_efwt = self.total_efwt.to(device)

        energy_ae = (energy_pred - energy_target).abs()
        self.total_energy_ae += energy_ae.sum().double()

        self.total_force_ae += (forces_pred - forces_target).abs().sum().double()
        self.n_force_components += forces_target.numel()

        cos = torch.cosine_similarity(forces_pred, forces_target, dim=-1)
        self.total_cos_sim += cos.sum().double()
        self.n_force_atoms += cos.numel()

        per_atom_ferr = (forces_pred - forces_target).norm(dim=-1)
        n_graphs = energy_pred.size(0)
        max_ferr_per_graph = cast(
            torch.Tensor,
            scatter_max(per_atom_ferr, batch, dim=0, dim_size=n_graphs)[0],
        )
        self.total_efwt += (
            ((energy_ae < EFWT_ENERGY_THRESHOLD) & (max_ferr_per_graph < EFWT_FORCE_THRESHOLD))
            .sum()
            .double()
        )
        self.n_structures += n_graphs

    def finalize(self, runtime_stats: Mapping[str, object] | None) -> dict[str, float]:
        n = max(1, self.n_structures)
        nf = max(1, self.n_force_components)
        na = max(1, self.n_force_atoms)
        return {
            "energy_mae": _rescale_energy_mae(self.total_energy_ae.item() / n, runtime_stats),
            "force_mae": self.total_force_ae.item() / nf,
            "force_cos": self.total_cos_sim.item() / na,
            "efwt": self.total_efwt.item() / n * 100.0,
        }


__all__ = ["EnergyForcesMetricAccumulator", "ScalarMetricAccumulator"]
