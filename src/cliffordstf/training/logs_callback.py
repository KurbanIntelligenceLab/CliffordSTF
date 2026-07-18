"""Lightning :class:`Callback` writing ``logs.json`` in the legacy schema.

Phase 2 Step 25. The legacy trainer dumps a per-fold ``logs.json`` via
:func:`cliffordstf.training.checkpointing.save_logs`. This callback
mirrors that contract for the Lightning path: it snapshots
``trainer.callback_metrics`` at the end of every validation epoch,
appends one record per epoch to an in-memory ``history`` list, and
writes the whole thing to ``logs_dir / "logs.json"`` at fit-end via
the same :func:`save_logs` helper.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import pytorch_lightning as pl

from cliffordstf.training.checkpointing import save_logs

if TYPE_CHECKING:
    from pathlib import Path

    from omegaconf import DictConfig


class LegacyLogsCallback(pl.Callback):
    """Write per-epoch ``trainer.callback_metrics`` to ``logs.json`` on fit-end."""

    def __init__(self, logs_dir: Path) -> None:
        self.logs_dir = logs_dir
        self.history: list[dict[str, float | int]] = []

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        snapshot: dict[str, float | int] = {"epoch": trainer.current_epoch}
        for key, value in trainer.callback_metrics.items():
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(scalar):
                snapshot[key] = scalar
        self.history.append(snapshot)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        path = self.logs_dir / "logs.json"
        payload: dict[str, Any] = {"history": self.history}
        save_logs(path, payload, cfg=cast("DictConfig", pl_module.cfg))


__all__ = ["LegacyLogsCallback"]
