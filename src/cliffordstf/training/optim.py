"""Optimizer and learning-rate scheduler builders driven by config.

Both builders read ``cfg.training.*`` and return the corresponding torch
objects. The optimizer is restricted to ``adam`` / ``adamw``; the scheduler
supports ``step`` / ``cosine`` / ``plateau`` (or ``None`` when
``cfg.training.lr_scheduler`` is not set).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn, optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    ReduceLROnPlateau,
    StepLR,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


def build_optimizer(cfg: DictConfig, model: nn.Module) -> optim.Optimizer:
    """Build an ``Adam`` or ``AdamW`` optimizer from ``cfg.training``.

    Reads:
        ``cfg.training.lr`` (required).
        ``cfg.training.weight_decay`` (default ``0.0``).
        ``cfg.training.optimizer`` (``"adam"`` or ``"adamw"``, default ``"adam"``).
    """
    lr = cfg.training.lr
    weight_decay = cfg.training.get("weight_decay", 0.0)
    opt_name = cfg.training.get("optimizer", "adam")

    param_groups = [
        {
            "params": list(model.parameters()),
            "lr": lr,
            "weight_decay": weight_decay,
        }
    ]

    if opt_name == "adam":
        return optim.Adam(param_groups)
    if opt_name == "adamw":
        return optim.AdamW(param_groups)
    raise ValueError(f"Unknown optimizer '{opt_name}'. Use 'adam' or 'adamw'.")


def build_scheduler(
    cfg: DictConfig,
    optimizer: optim.Optimizer,
) -> LRScheduler | ReduceLROnPlateau | None:
    """Build a scheduler from ``cfg.training.lr_scheduler``, or ``None``.

    Supported types:
        ``step`` -> :class:`torch.optim.lr_scheduler.StepLR` with
            ``step_size`` (default 10) and ``gamma`` (default 0.8).
        ``cosine`` -> :class:`torch.optim.lr_scheduler.CosineAnnealingLR`
            with ``T_max`` (default ``cfg.training.epochs``) and
            ``eta_min`` (default 0.0).
        ``plateau`` -> :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`
            with ``factor`` (default 0.5), ``patience`` (default 10), and
            ``min_lr`` (default 1e-7); the trainer is responsible for
            calling ``scheduler.step(val_metric)`` instead of ``step()``.
    """
    sched_cfg = cfg.training.get("lr_scheduler", None)
    if not sched_cfg:
        return None

    sched_type = sched_cfg.get("type", None)
    if sched_type == "step":
        return StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 10),
            gamma=sched_cfg.get("gamma", 0.8),
        )
    if sched_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get("T_max", cfg.training.epochs),
            eta_min=sched_cfg.get("eta_min", 0.0),
        )
    if sched_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=sched_cfg.get("factor", 0.5),
            patience=sched_cfg.get("patience", 10),
            min_lr=sched_cfg.get("min_lr", 1e-7),
        )
    raise ValueError(f"Unknown scheduler type '{sched_type}'. Use 'step', 'cosine', or 'plateau'.")
