"""Tests for the ``nequip`` baseline (requires the [baselines] extra)."""

from __future__ import annotations

import pytest

pytest.importorskip("nequip")

from baselines import AVAILABLE_MODELS, CONFIGS_DIR  # noqa: E402
from baselines.nequip import NequIPDataWrapper, build_nequip  # noqa: E402
from cliffordstf.io.config import load_config  # noqa: E402


def test_nequip_is_registered_in_catalog() -> None:
    assert "nequip" in AVAILABLE_MODELS


def test_nequip_yaml_resolves_with_extra_search_paths() -> None:
    cfg = load_config(
        argv=["model.name=nequip"],
        extra_search_paths=[CONFIGS_DIR],
    )
    assert cfg.model.name == "nequip"
    assert cfg.model.r_max == 6.0
    assert cfg.model.num_layers == 4
    assert cfg.model.l_max == 1
    assert cfg.model.num_features == 44


def test_build_nequip_returns_data_wrapper() -> None:
    cfg = load_config(
        argv=[
            "model.name=nequip",
            "model.r_max=4.0",
            "model.num_layers=1",
            "model.num_features=8",
            "model.num_bessels=4",
            "model.max_num_elements=20",
            "dataset.task_type=energy_forces",
        ],
        extra_search_paths=[CONFIGS_DIR],
    )
    model = build_nequip(cfg)
    assert isinstance(model, NequIPDataWrapper)
