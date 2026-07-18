"""Tests for :class:`EMACallback` (Phase 2 Step 20)."""

from __future__ import annotations

from typing import Any

from torch import nn

from cliffordstf.training.ema_callback import EMACallback, _model_has_ema


class _NotALightningModule:
    def __init__(self, model: nn.Module | None) -> None:
        self.model = model


class _DummyEMAModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.calls: list[str] = []

    def init_ema(self) -> None:
        self.calls.append("init_ema")

    def update_ema(self) -> None:
        self.calls.append("update_ema")

    def apply_ema(self) -> None:
        self.calls.append("apply_ema")

    def restore_from_ema(self) -> None:
        self.calls.append("restore_from_ema")


def test_model_has_ema_returns_true_when_methods_present() -> None:
    assert _model_has_ema(_NotALightningModule(_DummyEMAModel())) is True  # type: ignore[arg-type]


def test_model_has_ema_returns_false_for_plain_nn_module() -> None:
    assert _model_has_ema(_NotALightningModule(nn.Linear(2, 2))) is False  # type: ignore[arg-type]


def test_model_has_ema_returns_false_when_model_missing() -> None:
    assert _model_has_ema(_NotALightningModule(None)) is False  # type: ignore[arg-type]


def _drive_full_lifecycle(model: nn.Module) -> list[str]:
    cb = EMACallback()
    holder = _NotALightningModule(model)
    cb.on_fit_start(None, holder)  # type: ignore[arg-type]
    cb.on_train_batch_end(None, holder, None, None, 0)  # type: ignore[arg-type]
    cb.on_validation_start(None, holder)  # type: ignore[arg-type]
    cb.on_validation_end(None, holder)  # type: ignore[arg-type]
    cb.on_test_start(None, holder)  # type: ignore[arg-type]
    cb.on_test_end(None, holder)  # type: ignore[arg-type]
    return list(getattr(model, "calls", []))


def test_ema_callback_calls_each_hook_in_order() -> None:
    model = _DummyEMAModel()
    calls = _drive_full_lifecycle(model)
    assert calls == [
        "init_ema",
        "update_ema",
        "apply_ema",
        "restore_from_ema",
        "apply_ema",
        "restore_from_ema",
    ]


def test_ema_callback_is_noop_for_plain_module() -> None:
    plain = nn.Linear(2, 2)
    cb = EMACallback()
    holder: Any = _NotALightningModule(plain)
    cb.on_fit_start(None, holder)
    cb.on_train_batch_end(None, holder, None, None, 0)
    cb.on_validation_start(None, holder)
    cb.on_validation_end(None, holder)
