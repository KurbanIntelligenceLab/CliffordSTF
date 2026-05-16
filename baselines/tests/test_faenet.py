"""Tests for the ``faenet`` baseline (requires the [baselines] extra)."""

from __future__ import annotations

import pytest

pytest.importorskip("faenet")

from baselines import AVAILABLE_MODELS, CONFIGS_DIR  # noqa: E402
from baselines.faenet import FAENetDataWrapper, build_faenet  # noqa: E402
from cliffordstf.io.config import load_config  # noqa: E402


def test_faenet_is_registered_in_catalog() -> None:
    assert "faenet" in AVAILABLE_MODELS


def test_faenet_yaml_resolves_with_extra_search_paths() -> None:
    cfg = load_config(
        argv=["model.name=faenet"],
        extra_search_paths=[CONFIGS_DIR],
    )
    assert cfg.model.name == "faenet"
    assert cfg.model.hidden_channels == 320
    assert cfg.model.num_interactions == 4
    assert cfg.model.frame_averaging == "3D"
    assert cfg.model.fa_method == "stochastic"


def test_build_faenet_returns_data_wrapper() -> None:
    cfg = load_config(
        argv=[
            "model.name=faenet",
            "model.hidden_channels=16",
            "model.num_interactions=1",
            "model.num_gaussians=8",
            "model.num_filters=16",
            "model.cutoff=4.0",
            "model.max_num_neighbors=10",
        ],
        extra_search_paths=[CONFIGS_DIR],
    )
    model = build_faenet(cfg)
    assert isinstance(model, FAENetDataWrapper)
