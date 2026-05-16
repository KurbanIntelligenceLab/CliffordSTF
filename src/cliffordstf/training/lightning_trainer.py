"""Lightning-backed training entrypoint (Phase 2 Step 19).

Mirrors :func:`cliffordstf.training.trainer.train`'s contract so the
same ``(cfg, loaders)`` pair can drive either the legacy loop or this
Lightning :class:`pytorch_lightning.Trainer`-based path. The legacy
entry stays usable; Step 22 wires the CLI to pick between the two.

The Lightning trainer is configured with ``inference_mode=False`` so
the autograd graph survives :meth:`validation_step` /
:meth:`test_step` (force tasks read forces from ``-dE/dR``).

EMA and the project's existing checkpoint format are handled in
follow-up steps:

* Step 20 — EMA :class:`Callback` translating
  :class:`cliffordstf.training.ema.ExponentialMovingAverage`.
* Step 21 — Checkpoint :class:`Callback` that writes the legacy
  ``ckpt_last.pth`` / ``ckpt_best_val.pth`` format so Lightning and
  legacy checkpoints stay interchangeable.

Step 19 keeps the trainer minimal: an optional
:class:`EarlyStopping` callback and Lightning's stock progress
bar / logging. No custom callbacks yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping

from cliffordstf.models import build_model
from cliffordstf.reproducibility import SeedManager
from cliffordstf.training.checkpoint_callback import LegacyCheckpointCallback
from cliffordstf.training.ema_callback import EMACallback
from cliffordstf.training.lightning import CliffordSTFLightningModule
from cliffordstf.training.trainer import build_output_dirs

if TYPE_CHECKING:
    from collections.abc import Mapping

    from omegaconf import DictConfig

    from cliffordstf.domain import ModelFactory


def _resolve_accelerator(cfg: DictConfig) -> str:
    """Map ``cfg.device`` ('auto'|'cpu'|'cuda') to a Lightning ``accelerator``."""
    device = cfg.get("device", "auto")
    if device == "auto":
        return "gpu" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        return "gpu"
    return "cpu"


_PrecisionLiteral = Literal["32-true", "bf16-mixed", "16-mixed"]


def _resolve_precision(cfg: DictConfig, accelerator: str) -> _PrecisionLiteral:
    """Resolve a Lightning ``precision`` string from ``cfg.training.amp``."""
    if accelerator != "gpu" or not cfg.training.get("amp", False):
        return "32-true"
    dtype = cfg.training.get("amp_dtype", "bfloat16")
    return "bf16-mixed" if dtype == "bfloat16" else "16-mixed"


def _build_callbacks(cfg: DictConfig, model_dir: Path) -> list[pl.Callback]:
    """Build the default Lightning callback set.

    Always installs :class:`EMACallback` (no-op when the wrapped model
    does not expose the EMA contract) and :class:`LegacyCheckpointCallback`
    (writes ``ckpt_last.pth`` / ``ckpt_best_val.pth`` in the legacy
    schema). Adds :class:`EarlyStopping` when
    ``cfg.training.early_stopping.patience`` is positive.
    """
    callbacks: list[pl.Callback] = [
        EMACallback(),
        LegacyCheckpointCallback(model_dir=model_dir),
    ]
    es_cfg = cfg.training.get("early_stopping", None)
    patience = int(es_cfg.get("patience", 0)) if es_cfg else 0
    if patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                mode="min",
                check_finite=True,
            )
        )
    return callbacks


def train_lightning(
    cfg: DictConfig,
    loaders: dict[str, object],
    *,
    runtime_stats: Mapping[str, object] | None = None,
    extra_parts: tuple[str, ...] = (),
    model_extras: Mapping[str, ModelFactory] | None = None,
) -> dict[str, Any]:
    """Lightning-backed training entrypoint with the legacy ``train()`` contract.

    Args:
        cfg: Resolved config; reads ``cfg.training.{epochs,grad_clip,
            grad_accum_steps,val_every,amp,amp_dtype,early_stopping}`` and
            ``cfg.dataset.task_type`` (the latter via the wrapped
            :class:`CliffordSTFLightningModule`).
        loaders: Dict with required ``"train"`` / ``"val"`` keys and
            optional ``"test"``.
        runtime_stats: Forwarded for parity with the legacy trainer.
            Reserved for Step 21's metric-aware callbacks; currently
            unused by Step 19.
        extra_parts: Path segments appended to
            :func:`build_output_dirs` so per-fold runs do not collide.
        model_extras: Optional plug-in catalog passed through to
            :func:`cliffordstf.models.build_model` (mirrors the legacy
            trainer's ``model_extras`` kwarg).

    Returns:
        Dict containing the resolved output directory and any tracked
        metrics. Shape will grow as Steps 20-22 land.
    """
    SeedManager.set_global_seed(cfg.seed)
    base_dir, model_dir, logs_dir = build_output_dirs(cfg, extra_parts)

    raw_model = build_model(cfg.model.name, cfg, extras=model_extras)
    module = CliffordSTFLightningModule(raw_model, cfg)

    accelerator = _resolve_accelerator(cfg)
    precision = _resolve_precision(cfg, accelerator)

    grad_clip = cfg.training.get("grad_clip", 0.0) or 0.0
    grad_accum = int(cfg.training.get("grad_accum_steps", 1))
    val_every = int(cfg.training.get("val_every", 1))

    trainer = pl.Trainer(
        max_epochs=int(cfg.training.epochs),
        check_val_every_n_epoch=val_every,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        gradient_clip_val=float(grad_clip),
        accumulate_grad_batches=grad_accum,
        inference_mode=False,
        callbacks=_build_callbacks(cfg, model_dir),
        default_root_dir=str(logs_dir),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(
        module,
        train_dataloaders=loaders["train"],
        val_dataloaders=loaders.get("val"),
    )

    test_metrics: list[dict[str, float]] = []
    if "test" in loaders and loaders["test"] is not None:
        raw = trainer.test(module, dataloaders=loaders["test"], verbose=False)
        test_metrics = [dict(m) for m in cast("list[Any]", raw)]

    return {
        "output_dir": str(base_dir),
        "test_metrics": test_metrics,
        "runtime_stats": dict(runtime_stats) if runtime_stats is not None else None,
    }


__all__ = ["train_lightning"]
