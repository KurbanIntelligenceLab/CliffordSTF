"""Validation / test loop dispatched by ``cfg.dataset.task_type``.

Energy-forces returns four metrics:
    ``energy_mae``: per-structure energy mean absolute error.
    ``force_mae``: per-component force mean absolute error.
    ``force_cos``: per-atom mean cosine similarity between predicted and
        target force vectors.
    ``efwt``: percentage of structures with energy error < 0.02 and per-atom
        max force error < 0.03 (the FairChem "energy-and-forces-within-
        threshold" metric).

Scalar returns one metric:
    ``energy_mae``: per-structure absolute error on the scalar target.

When ``runtime_stats`` contains a ``"std"`` (and optional ``"mean"``), the
reported ``energy_mae`` is multiplied by ``std`` so that QM9 / Molecule3D
metrics are in the dataset's native (physical) units.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torch_scatter import scatter_max
from tqdm import tqdm

from cliffordstf.training.losses import (
    SCALAR_TASK_TYPES,
    compute_forces,
    forward_model,
    get_free_atom_mask,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

EFWT_ENERGY_THRESHOLD = 0.02
EFWT_FORCE_THRESHOLD = 0.03


def evaluate_epoch(
    model: nn.Module,
    loader: object,
    cfg: DictConfig,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    *,
    runtime_stats: Mapping[str, object] | None = None,
) -> dict[str, float]:
    """Evaluate ``model`` on ``loader`` and return task-appropriate metrics."""
    task_type = cfg.dataset.get("task_type", "energy_forces")
    if task_type in SCALAR_TASK_TYPES:
        return _evaluate_scalar(model, loader, device, amp_dtype, runtime_stats)
    return _evaluate_energy_forces(model, loader, cfg, device, amp_dtype, runtime_stats)


def _rescale_energy_mae(mae: float, runtime_stats: Mapping[str, object] | None) -> float:
    if runtime_stats is None:
        return mae
    std_raw = runtime_stats.get("std", None)
    if std_raw is None:
        return mae
    if isinstance(std_raw, torch.Tensor):
        std = float(std_raw.item())
    else:
        std = float(cast(float, std_raw))
    return mae * std


def _evaluate_scalar(
    model: nn.Module,
    loader: object,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    runtime_stats: Mapping[str, object] | None,
) -> dict[str, float]:
    model.eval()
    use_amp = amp_dtype is not None and device.type == "cuda"

    total_energy_ae = torch.zeros((), device=device, dtype=torch.float64)
    n_structures = 0

    with torch.no_grad():
        for data in tqdm(loader, desc="  Val", unit="batch", leave=False):
            data = data.to(device)
            with torch.autocast(
                device_type="cuda",
                enabled=use_amp,
                dtype=amp_dtype or torch.float32,
            ):
                out = forward_model(model, data)
            energy_pred = out[0] if isinstance(out, tuple) else out
            energy_target = (
                data.energy.view(-1) if hasattr(data, "energy") else data.y.view(-1)
            ).to(energy_pred.dtype)
            total_energy_ae += (energy_pred - energy_target).abs().sum().double()
            n_structures += energy_pred.size(0)

    n = max(1, n_structures)
    energy_mae = total_energy_ae.item() / n
    return {"energy_mae": _rescale_energy_mae(energy_mae, runtime_stats)}


def _evaluate_energy_forces(
    model: nn.Module,
    loader: object,
    cfg: DictConfig,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    runtime_stats: Mapping[str, object] | None,
) -> dict[str, float]:
    model.eval()
    eval_on_free = cfg.dataset.get("eval_on_free_atoms", False)
    use_amp = amp_dtype is not None and device.type == "cuda"

    total_energy_ae = torch.zeros((), device=device, dtype=torch.float64)
    total_force_ae = torch.zeros((), device=device, dtype=torch.float64)
    total_cos_sim = torch.zeros((), device=device, dtype=torch.float64)
    total_efwt = torch.zeros((), device=device, dtype=torch.float64)
    n_structures = 0
    n_force_components = 0
    n_force_atoms = 0

    for data in tqdm(loader, desc="  Val", unit="batch", leave=False):
        data = data.to(device)
        data.pos.requires_grad_(True)
        with torch.autocast(
            device_type="cuda",
            enabled=use_amp,
            dtype=amp_dtype or torch.float32,
        ):
            out = forward_model(model, data)

        if isinstance(out, tuple):
            energy_pred, forces_pred = out[0], out[1]
        else:
            energy_pred = out
            forces_pred = compute_forces(energy_pred, data.pos, create_graph=False)

        energy_target = data.energy.view(-1) if hasattr(data, "energy") else data.y.view(-1)
        forces_target = data.force.to(forces_pred.dtype)

        free_mask = get_free_atom_mask(data) if eval_on_free else None
        if free_mask is not None:
            fp = forces_pred[free_mask]
            ft = forces_target[free_mask]
            batch_free = data.batch[free_mask]
        else:
            fp = forces_pred
            ft = forces_target
            batch_free = data.batch

        energy_ae = (energy_pred - energy_target).abs()
        total_energy_ae += energy_ae.sum().double()

        total_force_ae += (fp - ft).abs().sum().double()
        n_force_components += ft.numel()

        cos = torch.cosine_similarity(fp, ft, dim=-1)
        total_cos_sim += cos.sum().double()
        n_force_atoms += cos.numel()

        per_atom_ferr = (fp - ft).norm(dim=-1)
        n_graphs = energy_pred.size(0)
        max_ferr_per_graph = cast(
            torch.Tensor,
            scatter_max(per_atom_ferr, batch_free, dim=0, dim_size=n_graphs)[0],
        )
        total_efwt += (
            ((energy_ae < EFWT_ENERGY_THRESHOLD) & (max_ferr_per_graph < EFWT_FORCE_THRESHOLD))
            .sum()
            .double()
        )

        n_structures += n_graphs

    n = max(1, n_structures)
    nf = max(1, n_force_components)
    na = max(1, n_force_atoms)
    energy_mae = total_energy_ae.item() / n
    return {
        "energy_mae": _rescale_energy_mae(energy_mae, runtime_stats),
        "force_mae": total_force_ae.item() / nf,
        "force_cos": total_cos_sim.item() / na,
        "efwt": total_efwt.item() / n * 100.0,
    }
