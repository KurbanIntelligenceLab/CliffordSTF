"""Validation / test loop for energy + forces models.

Returns four metrics:
    ``energy_mae``: per-structure energy mean absolute error.
    ``force_mae``: per-component force mean absolute error.
    ``force_cos``: per-atom mean cosine similarity between predicted and
        target force vectors.
    ``efwt``: percentage of structures with energy error < 0.02 and per-atom
        max force error < 0.03 (the FairChem "energy-and-forces-within-
        threshold" metric).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torch_scatter import scatter_max
from tqdm import tqdm

from cliffordstf.training.losses import (
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
) -> dict[str, float]:
    """Evaluate ``model`` on ``loader`` and return energy/force metrics.

    Energy_forces tasks need gradients for autograd force computation, so the
    inner loop does not wrap forward in ``torch.no_grad``. AMP is enabled
    only when ``amp_dtype`` is set and the device is CUDA.
    """
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
    return {
        "energy_mae": total_energy_ae.item() / n,
        "force_mae": total_force_ae.item() / nf,
        "force_cos": total_cos_sim.item() / na,
        "efwt": total_efwt.item() / n * 100.0,
    }
