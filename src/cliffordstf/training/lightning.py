"""PyTorch Lightning wrapper around any cliffordstf-compatible model.

Step 18 of the Phase-2 Lightning migration. Provides
:class:`CliffordSTFLightningModule`, a thin :class:`pytorch_lightning.LightningModule`
that wraps an existing ``nn.Module`` (in-tree :mod:`cliffordstf.models` or
plug-in :mod:`baselines`) and delegates loss / optimizer / scheduler
construction to the existing builders so the same YAML drives both the
legacy :func:`cliffordstf.training.trainer.train` loop and a Lightning
:class:`pytorch_lightning.Trainer` once Step 19 wires it.

The wrapped model must satisfy the ``data_wrapper`` interface:
``model(data) -> energy`` or ``(energy, forces)``. Force tasks rely on
autograd, so a future Lightning :class:`Trainer` configuration must
pass ``inference_mode=False`` (Lightning's default
``torch.inference_mode()`` would otherwise disable the autograd graph
in :meth:`validation_step`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from cliffordstf.training.losses import compute_loss
from cliffordstf.training.optim import build_optimizer, build_scheduler

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch import nn
    from torch_geometric.data import Data


class CliffordSTFLightningModule(pl.LightningModule):
    """LightningModule that wraps any cliffordstf-compatible ``nn.Module``."""

    def __init__(self, model: nn.Module, cfg: DictConfig) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg

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

    def validation_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        with torch.enable_grad():
            loss = compute_loss(self.model, batch, self.cfg, training=False)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=int(batch.num_graphs),
        )
        return loss

    def test_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        with torch.enable_grad():
            loss = compute_loss(self.model, batch, self.cfg, training=False)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=int(batch.num_graphs),
        )
        return loss

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


__all__ = ["CliffordSTFLightningModule"]
