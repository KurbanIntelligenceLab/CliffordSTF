"""Tests for the ``visnet`` baseline."""

from __future__ import annotations

from baselines import AVAILABLE_MODELS, CONFIGS_DIR
from baselines.visnet import ViSNetDataWrapper, build_visnet
from cliffordstf.io.config import load_config


def test_visnet_is_registered_in_catalog() -> None:
    assert "visnet" in AVAILABLE_MODELS


def test_visnet_yaml_resolves_with_extra_search_paths() -> None:
    cfg = load_config(
        argv=["model.name=visnet"],
        extra_search_paths=[CONFIGS_DIR],
    )
    assert cfg.model.name == "visnet"
    assert cfg.model.num_layers == 6
    assert cfg.model.hidden_channels == 96
    assert cfg.model.lmax == 1
    assert cfg.model.cutoff == 5.0


def test_build_visnet_returns_data_wrapper() -> None:
    cfg = load_config(
        argv=[
            "model.name=visnet",
            "model.num_heads=2",
            "model.num_layers=1",
            "model.hidden_channels=8",
            "model.num_rbf=8",
            "model.cutoff=4.0",
            "model.max_num_neighbors=10",
            "dataset.task_type=energy_forces",
        ],
        extra_search_paths=[CONFIGS_DIR],
    )
    model = build_visnet(cfg)
    assert isinstance(model, ViSNetDataWrapper)
    assert model.derivative is True
