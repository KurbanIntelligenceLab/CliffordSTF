"""Tests for the ``cliffordstf.models.build_model`` resolver."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cliffordstf.models import AVAILABLE_MODELS, build_model


def test_available_models_holds_the_three_step4_variants() -> None:
    assert set(AVAILABLE_MODELS) == {
        "clifford_stf",
        "clifford_stf_full",
        "clifford_stf_full_10m",
    }


def test_build_model_unknown_name_raises_keyerror() -> None:
    cfg = MagicMock()
    with pytest.raises(KeyError, match="Unknown model 'no_such_model'"):
        build_model("no_such_model", cfg)


def test_build_model_uses_extras() -> None:
    cfg = MagicMock()
    sentinel = object()
    extras = {"plugged_in": lambda _cfg: sentinel}
    assert build_model("plugged_in", cfg, extras=extras) is sentinel


def test_build_model_extras_can_override_builtin() -> None:
    cfg = MagicMock()
    extras_only = object()
    extras = {"override_me": lambda _cfg: extras_only}
    assert build_model("override_me", cfg, extras=extras) is extras_only


def test_available_models_mapping_is_read_only() -> None:
    with pytest.raises(TypeError):
        AVAILABLE_MODELS["sneak_in"] = lambda _cfg: None  # type: ignore[index]
