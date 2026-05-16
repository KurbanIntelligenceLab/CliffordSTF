"""Confirms the baselines catalog is empty until a baseline lands."""

from __future__ import annotations

from baselines import AVAILABLE_MODELS


def test_available_models_starts_empty() -> None:
    assert dict(AVAILABLE_MODELS) == {}
