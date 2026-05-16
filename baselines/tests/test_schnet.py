"""Tests for the ``schnet`` baseline."""

from __future__ import annotations

from baselines import AVAILABLE_MODELS, CONFIGS_DIR
from baselines.schnet import SchNetDataWrapper, build_schnet
from cliffordstf.io.config import load_config


def test_schnet_is_registered_in_catalog() -> None:
    assert "schnet" in AVAILABLE_MODELS


def test_schnet_yaml_resolves_with_extra_search_paths() -> None:
    cfg = load_config(
        argv=["model.name=schnet"],
        extra_search_paths=[CONFIGS_DIR],
    )
    assert cfg.model.name == "schnet"
    assert cfg.model.hidden_channels == 192
    assert cfg.model.num_filters == 192
    assert cfg.model.num_interactions == 6
    assert cfg.model.num_gaussians == 50
    assert cfg.model.cutoff == 6.0


def test_build_schnet_returns_data_wrapper() -> None:
    cfg = load_config(
        argv=[
            "model.name=schnet",
            "model.hidden_channels=8",
            "model.num_filters=8",
            "model.num_interactions=1",
            "model.num_gaussians=8",
            "model.cutoff=4.0",
        ],
        extra_search_paths=[CONFIGS_DIR],
    )
    model = build_schnet(cfg)
    assert isinstance(model, SchNetDataWrapper)
