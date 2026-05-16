"""Lightning :class:`Callback` writing the legacy checkpoint format.

Phase 2 Step 21. The legacy trainer writes two files per fold under
``<output_root>/.../models/``:

* ``ckpt_last.pth``     — most recent epoch, for resume.
* ``ckpt_best_val.pth`` — snapshot at the lowest ``val_loss`` so far.

Both files share the schema documented in
:func:`cliffordstf.training.checkpointing.save_checkpoint`. This
callback writes the same files when running under the Lightning
trainer so checkpoints stay interchangeable across the two paths.

Writes are dispatched to the shared background-thread executor in
:mod:`cliffordstf.training.checkpointing`, and the callback drains
the queue via :func:`wait_for_checkpoint` from ``on_fit_end`` and
``on_exception``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import pytorch_lightning as pl

from cliffordstf.training.checkpointing import (
    save_checkpoint,
    wait_for_checkpoint,
)

if TYPE_CHECKING:
    from pathlib import Path

    from omegaconf import DictConfig
    from torch import nn


class LegacyCheckpointCallback(pl.Callback):
    """Write ``ckpt_last.pth`` / ``ckpt_best_val.pth`` in the legacy schema."""

    def __init__(self, model_dir: Path, monitor: str = "val_loss") -> None:
        self.model_dir = model_dir
        self.monitor = monitor
        self.best_val: float = math.inf
        self.best_epoch: int = -1

    def _save(
        self,
        path: Path,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        epoch: int,
        metrics: dict[str, float] | None,
    ) -> None:
        optimizer = trainer.optimizers[0] if trainer.optimizers else None
        scheduler = (
            trainer.lr_scheduler_configs[0].scheduler if trainer.lr_scheduler_configs else None
        )
        save_checkpoint(
            path=path,
            model=cast("nn.Module", pl_module.model),
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            cfg=cast("DictConfig", pl_module.cfg),
            scheduler=scheduler,
            training_state={"best_val": self.best_val, "best_epoch": self.best_epoch},
        )

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._save(
            self.model_dir / "ckpt_last.pth",
            pl_module,
            trainer,
            epoch=trainer.current_epoch,
            metrics=_collect_callback_metrics(trainer),
        )

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        callback_metrics = _collect_callback_metrics(trainer)
        val = callback_metrics.get(self.monitor)
        if val is None or not math.isfinite(val):
            return
        if val < self.best_val:
            self.best_val = val
            self.best_epoch = trainer.current_epoch
            self._save(
                self.model_dir / "ckpt_best_val.pth",
                pl_module,
                trainer,
                epoch=trainer.current_epoch,
                metrics=callback_metrics,
            )

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        wait_for_checkpoint()

    def on_exception(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException
    ) -> None:
        wait_for_checkpoint()


def _collect_callback_metrics(trainer: pl.Trainer) -> dict[str, float]:
    """Extract finite scalar metrics from ``trainer.callback_metrics``."""
    metrics: dict[str, float] = {}
    for key, value in trainer.callback_metrics.items():
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(scalar):
            metrics[key] = scalar
    return metrics


__all__ = ["LegacyCheckpointCallback"]
