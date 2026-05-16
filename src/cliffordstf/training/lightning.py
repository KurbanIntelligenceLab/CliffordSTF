"""PyTorch Lightning wrapper around any cliffordstf-compatible model.

Step 18 of the Phase-2 Lightning migration. Provides
:class:`CliffordSTFLightningModule`, a thin :class:`pytorch_lightning.LightningModule`
that wraps an existing ``nn.Module`` (in-tree :mod:`cliffordstf.models` or
plug-in :mod:`baselines`) and delegates loss / optimizer / scheduler
construction to the existing builders so the same YAML drives both the
legacy :func:`cliffordstf.training.trainer.train` loop and a Lightning
:class:`pytorch_lightning.Trainer`.

The wrapped model must satisfy the ``data_wrapper`` interface:
``model(data) -> energy`` or ``(energy, forces)``. Force tasks rely on
autograd, so the Lightning :class:`Trainer` configuration must pass
``inference_mode=False`` (Lightning's default
``torch.inference_mode()`` would otherwise disable the autograd graph
in :meth:`validation_step`).

Step 23 added rich validation / test metrics mirroring the legacy
:func:`cliffordstf.training.evaluate.evaluate_epoch`. The Module
accumulates per-batch predictions and, at the end of each
validation / test epoch, logs ``val_energy_mae`` / ``val_force_mae``
/ ``val_force_cos`` / ``val_efwt`` (plus the corresponding ``test_*``
keys). ``runtime_stats`` is threaded through so scalar tasks rescale
energy MAE to physical units the same way the legacy trainer does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl
import torch
from torch.nn.functional import l1_loss, mse_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from cliffordstf.training._val_metrics import (
    EnergyForcesMetricAccumulator,
    ScalarMetricAccumulator,
)
from cliffordstf.training.losses import (
    SCALAR_TASK_TYPES,
    compute_forces,
    compute_loss,
    forward_model,
    get_free_atom_mask,
)
from cliffordstf.training.optim import build_optimizer, build_scheduler

if TYPE_CHECKING:
    from collections.abc import Mapping

    from omegaconf import DictConfig
    from torch import nn
    from torch_geometric.data import Data


_MetricAccumulator = ScalarMetricAccumulator | EnergyForcesMetricAccumulator


class CliffordSTFLightningModule(pl.LightningModule):
    """LightningModule that wraps any cliffordstf-compatible ``nn.Module``."""

    def __init__(
        self,
        model: nn.Module,
        cfg: DictConfig,
        *,
        runtime_stats: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.runtime_stats = runtime_stats
        self._val_acc: _MetricAccumulator | None = None
        self._test_acc: _MetricAccumulator | None = None

    def forward(self, data: Data) -> Any:  # noqa: ANN401
        return self.model(data)

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        loss = compute_loss(self.model, batch, self.cfg, training=True)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=int(batch.num_graphs),
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_acc = _new_accumulator(self.cfg)

    def validation_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        with torch.enable_grad():
            loss, payload = _forward_loss_and_predictions(self.model, batch, self.cfg)
        if self._val_acc is not None:
            _apply_update(self._val_acc, payload)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=int(batch.num_graphs),
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        if self._val_acc is None:
            return
        for name, value in self._val_acc.finalize(self.runtime_stats).items():
            self.log(f"val_{name}", value, on_epoch=True)

    def on_test_epoch_start(self) -> None:
        self._test_acc = _new_accumulator(self.cfg)

    def test_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        with torch.enable_grad():
            loss, payload = _forward_loss_and_predictions(self.model, batch, self.cfg)
        if self._test_acc is not None:
            _apply_update(self._test_acc, payload)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=int(batch.num_graphs),
        )
        return loss

    def on_test_epoch_end(self) -> None:
        if self._test_acc is None:
            return
        for name, value in self._test_acc.finalize(self.runtime_stats).items():
            self.log(f"test_{name}", value, on_epoch=True)

    def configure_optimizers(self) -> Any:  # noqa: ANN401 - Lightning's union is overly specific
        optimizer = build_optimizer(self.cfg, self.model)
        scheduler = build_scheduler(self.cfg, optimizer)
        if scheduler is None:
            return {"optimizer": optimizer}
        if isinstance(scheduler, ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
            }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def _new_accumulator(cfg: DictConfig) -> _MetricAccumulator:
    task_type = (
        cfg.dataset.get("task_type", "energy_forces") if "dataset" in cfg else "energy_forces"
    )
    if task_type in SCALAR_TASK_TYPES:
        return ScalarMetricAccumulator()
    return EnergyForcesMetricAccumulator()


def _apply_update(accumulator: _MetricAccumulator, payload: dict[str, torch.Tensor]) -> None:
    if isinstance(accumulator, ScalarMetricAccumulator):
        accumulator.update(payload["energy_pred"], payload["energy_target"])
    else:
        accumulator.update(
            payload["energy_pred"],
            payload["energy_target"],
            payload["forces_pred"],
            payload["forces_target"],
            payload["batch"],
        )


def _forward_loss_and_predictions(
    model: nn.Module, batch: Data, cfg: DictConfig
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one forward pass and return ``(loss, predictions_for_metrics)``."""
    loss_type = cfg.training.get("loss", "mse")
    loss_fn = mse_loss if loss_type == "mse" else l1_loss
    task_type = (
        cfg.dataset.get("task_type", "energy_forces") if "dataset" in cfg else "energy_forces"
    )

    if task_type in SCALAR_TASK_TYPES:
        out = forward_model(model, batch)
        energy_pred = out[0] if isinstance(out, tuple) else out
        energy_target = (
            batch.energy.view(-1) if hasattr(batch, "energy") else batch.y.view(-1)
        ).to(energy_pred.dtype)
        loss = loss_fn(energy_pred, energy_target)
        return loss, {"energy_pred": energy_pred, "energy_target": energy_target}

    batch.pos.requires_grad_(True)
    out = forward_model(model, batch)
    if isinstance(out, tuple) and len(out) >= 2:
        energy_pred = out[0]
        forces_pred = out[1]
    else:
        energy_pred = out
        forces_pred = compute_forces(energy_pred, batch.pos, create_graph=False)

    energy_target = (
        batch.energy.view(-1) if hasattr(batch, "energy") else batch.y.view(-1)
    ).to(energy_pred.dtype)
    forces_target = batch.force.to(forces_pred.dtype)

    fp, ft = forces_pred, forces_target
    if cfg.dataset.get("eval_on_free_atoms", False):
        free_mask = get_free_atom_mask(batch)
        if free_mask is not None:
            fp = fp[free_mask]
            ft = ft[free_mask]

    energy_weight = float(cfg.dataset.get("energy_weight", 1.0))
    force_weight = float(cfg.dataset.get("force_weight", 1.0))
    loss = energy_weight * loss_fn(energy_pred, energy_target) + force_weight * loss_fn(fp, ft)

    return loss, {
        "energy_pred": energy_pred,
        "energy_target": energy_target,
        "forces_pred": fp,
        "forces_target": ft,
        "batch": batch.batch[get_free_atom_mask(batch)]
        if cfg.dataset.get("eval_on_free_atoms", False) and get_free_atom_mask(batch) is not None
        else batch.batch,
    }


__all__ = ["CliffordSTFLightningModule"]
