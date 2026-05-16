"""Training utilities: optimizer / scheduler builders, losses, EMA.

The ``trainer`` module imports :mod:`cliffordstf.models` (which in turn imports
this package via the wrapper's EMA dependency), so it is NOT re-exported here -
users construct ``train`` / ``train_one_run`` via
``from cliffordstf.training.trainer import train``.
"""

from __future__ import annotations

from cliffordstf.training.checkpoint_callback import LegacyCheckpointCallback
from cliffordstf.training.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    save_logs,
    wait_for_checkpoint,
)
from cliffordstf.training.ema import ExponentialMovingAverage
from cliffordstf.training.ema_callback import EMACallback
from cliffordstf.training.evaluate import evaluate_epoch
from cliffordstf.training.lightning import CliffordSTFLightningModule
from cliffordstf.training.lightning_trainer import train_lightning
from cliffordstf.training.losses import (
    compute_forces,
    compute_loss,
    forward_model,
    get_free_atom_mask,
    l2norm_loss,
    per_atom_mae_loss,
)
from cliffordstf.training.optim import build_optimizer, build_scheduler

__all__ = [
    "CliffordSTFLightningModule",
    "EMACallback",
    "ExponentialMovingAverage",
    "LegacyCheckpointCallback",
    "build_optimizer",
    "build_scheduler",
    "compute_forces",
    "compute_loss",
    "evaluate_epoch",
    "forward_model",
    "get_free_atom_mask",
    "l2norm_loss",
    "load_checkpoint",
    "per_atom_mae_loss",
    "save_checkpoint",
    "save_logs",
    "train_lightning",
    "wait_for_checkpoint",
]
