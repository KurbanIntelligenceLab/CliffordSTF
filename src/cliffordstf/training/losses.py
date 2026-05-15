"""Loss kernels and the energy/forces loss dispatcher.

The trainer is energy-forces-only: there is no ``scalar`` / ``is2re`` branch.
``compute_loss`` reads ``cfg.dataset.*`` for loss-type selection and weighting,
and ``cfg.training.loss`` (``"mse"`` or ``"l1"``) for the default kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torch.nn.functional import l1_loss, mse_loss

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch_geometric.data import Data


ForwardResult = (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | torch.Tensor
)


def forward_model(model: nn.Module, data: Data) -> ForwardResult:
    """Call the wrapper model on a batch using the ``data_wrapper`` interface.

    The model returns either ``(energy, forces)`` (direct forces) or just
    ``energy`` (autograd forces). Some models also return a third tensor
    (``dens_loss`` from DeNS auxiliary training). All three shapes are
    forwarded as-is; the caller decides how to consume them.
    """
    out = model(data)
    if isinstance(out, tuple):
        return cast(ForwardResult, out)
    return cast(torch.Tensor, out).view(-1)


def compute_forces(
    energy: torch.Tensor,
    pos: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    """Compute forces as ``-dE/dR`` via autograd.

    Forced to fp32 inside an ``autocast(enabled=False)`` block: half-precision
    gradients are numerically unstable for force computation.
    """
    with torch.autocast(device_type="cuda", enabled=False):
        energy_f32 = energy.float()
        grad_outputs = torch.autograd.grad(
            energy_f32.sum(),
            pos,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=True,
        )[0]
    if grad_outputs is None:
        return torch.zeros_like(pos)  # type: ignore[unreachable]
    return -grad_outputs


def per_atom_mae_loss(
    pred: torch.Tensor, target: torch.Tensor, natoms_per_graph: torch.Tensor
) -> torch.Tensor:
    """``L1(pred / natoms, target / natoms)`` (matches FairChem ``per_atom_mae``)."""
    return l1_loss(pred / natoms_per_graph, target / natoms_per_graph)


def l2norm_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean of per-atom L2 norms of the error vector (FairChem ``l2norm``)."""
    return cast(torch.Tensor, torch.linalg.vector_norm(pred - target, ord=2, dim=-1).mean())


def get_free_atom_mask(data: Data) -> torch.Tensor | None:
    """Boolean mask of free (non-fixed) atoms, or ``None`` if unavailable.

    Convention:
        - ``data.fixed`` (bool tensor): ``True`` = fixed atom.
        - ``data.tags`` (int tensor): ``0`` = sub-surface (fixed), ``1`` =
          surface, ``2`` = adsorbate.
    """
    if hasattr(data, "fixed") and data.fixed is not None:
        return cast(torch.Tensor, ~data.fixed.bool())
    if hasattr(data, "tags") and data.tags is not None:
        return cast(torch.Tensor, data.tags > 0)
    return None


def compute_loss(
    model: nn.Module,
    data: Data,
    cfg: DictConfig,
    training: bool = True,
) -> torch.Tensor:
    """Compute the energy + forces loss for one batch.

    Reads:
        ``cfg.training.loss``: ``"mse"`` (default) or ``"l1"``.
        ``cfg.dataset.energy_loss``: ``"per_atom_mae"`` or unset (default loss kernel).
        ``cfg.dataset.force_loss``: ``"l2norm"`` or unset (default loss kernel).
        ``cfg.dataset.energy_weight`` (default 1.0).
        ``cfg.dataset.force_weight`` (default 1.0).
        ``cfg.dataset.train_on_free_atoms`` (default ``False``): drop fixed
            atoms before computing the force loss.
    """
    loss_type = cfg.training.get("loss", "mse")
    loss_fn = mse_loss if loss_type == "mse" else l1_loss

    data.pos.requires_grad_(True)
    out = forward_model(model, data)

    aux_loss: torch.Tensor | None = None
    if isinstance(out, tuple) and len(out) == 3:
        energy_pred, forces_pred, aux_loss = out
    elif isinstance(out, tuple):
        energy_pred, forces_pred = out
    else:
        energy_pred = out
        forces_pred = compute_forces(energy_pred, data.pos, create_graph=training)

    energy_target = data.energy.view(-1) if hasattr(data, "energy") else data.y.view(-1)
    energy_target = energy_target.to(energy_pred.dtype)
    forces_target = data.force.to(forces_pred.dtype)

    if cfg.dataset.get("train_on_free_atoms", False):
        free_mask = get_free_atom_mask(data)
        if free_mask is not None:
            forces_pred = forces_pred[free_mask]
            forces_target = forces_target[free_mask]

    e_loss_type = cfg.dataset.get("energy_loss", None)
    if e_loss_type == "per_atom_mae":
        natoms_per_graph = torch.bincount(data.batch).to(energy_pred.dtype)
        loss_e = per_atom_mae_loss(energy_pred, energy_target, natoms_per_graph)
    else:
        loss_e = loss_fn(energy_pred, energy_target)

    f_loss_type = cfg.dataset.get("force_loss", None)
    if f_loss_type == "l2norm":
        loss_f = l2norm_loss(forces_pred, forces_target)
    else:
        loss_f = loss_fn(forces_pred, forces_target)

    energy_weight: float = cfg.dataset.get("energy_weight", 1.0)
    force_weight: float = cfg.dataset.get("force_weight", 1.0)
    total: torch.Tensor = energy_weight * loss_e + force_weight * loss_f
    if aux_loss is not None:
        total = total + aux_loss
    return total
