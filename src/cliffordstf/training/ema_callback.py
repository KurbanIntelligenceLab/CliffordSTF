"""Lightning :class:`Callback` translating the legacy EMA contract.

Phase 2 Step 20. The legacy trainer drives EMA via four methods that
:class:`cliffordstf.models.wrapper.CliffordSTFWrapper` exposes:

* ``init_ema()``         — called once before training starts.
* ``update_ema()``       — called after every train batch.
* ``apply_ema()``        — swap live weights for EMA shadow before
  validation / test.
* ``restore_from_ema()`` — swap back when validation / test ends.

:class:`EMACallback` wires the same lifecycle through Lightning hooks.
Models that do not expose these methods (most baselines) are a no-op:
the callback simply skips its body when the four attributes are
absent, so it is safe to install for every variant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytorch_lightning as pl

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch


class _EMACapable:
    def init_ema(self) -> None: ...

    def update_ema(self) -> None: ...

    def apply_ema(self) -> None: ...

    def restore_from_ema(self) -> None: ...


def _model_has_ema(pl_module: pl.LightningModule) -> bool:
    """Return ``True`` if the wrapped model exposes the EMA contract."""
    model = getattr(pl_module, "model", None)
    if model is None:
        return False
    return all(
        callable(getattr(model, name, None))
        for name in ("init_ema", "update_ema", "apply_ema", "restore_from_ema")
    )


class EMACallback(pl.Callback):
    """Wire :class:`ExponentialMovingAverage` hooks into a Lightning ``Trainer``."""

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if _model_has_ema(pl_module):
            cast("_EMACapable", pl_module.model).init_ema()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,  # noqa: ANN401 - Lightning passes through whatever batch was loaded
        batch_idx: int,
    ) -> None:
        if _model_has_ema(pl_module):
            cast("_EMACapable", pl_module.model).update_ema()

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if _model_has_ema(pl_module):
            cast("_EMACapable", pl_module.model).apply_ema()

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if _model_has_ema(pl_module):
            cast("_EMACapable", pl_module.model).restore_from_ema()

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if _model_has_ema(pl_module):
            cast("_EMACapable", pl_module.model).apply_ema()

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if _model_has_ema(pl_module):
            cast("_EMACapable", pl_module.model).restore_from_ema()


__all__ = ["EMACallback"]
