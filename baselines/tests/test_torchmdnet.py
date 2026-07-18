"""Tests for the ``torchmdnet`` baseline (requires the [baselines] extra)."""

from __future__ import annotations

import pytest

pytest.importorskip("torchmdnet")

from baselines import AVAILABLE_MODELS, CONFIGS_DIR  # noqa: E402
from baselines.torchmdnet import TorchMDNetDataWrapper, build_torchmdnet  # noqa: E402
from cliffordstf.io.config import load_config  # noqa: E402


def test_torchmdnet_is_registered_in_catalog() -> None:
    assert "torchmdnet" in AVAILABLE_MODELS


def test_torchmdnet_yaml_resolves_with_extra_search_paths() -> None:
    cfg = load_config(
        argv=["model.name=torchmdnet"],
        extra_search_paths=[CONFIGS_DIR],
    )
    assert cfg.model.name == "torchmdnet"
    assert cfg.model.model_arch == "equivariant-transformer"
    assert cfg.model.embedding_dimension == 104
    assert cfg.model.num_layers == 6
    assert cfg.model.cutoff_upper == 6.0


def test_build_torchmdnet_returns_data_wrapper() -> None:
    cfg = load_config(
        argv=[
            "model.name=torchmdnet",
            "model.embedding_dimension=8",
            "model.num_layers=1",
            "model.num_rbf=8",
            "model.cutoff_upper=4.0",
            "model.max_num_neighbors=10",
            "model.num_heads=2",
        ],
        extra_search_paths=[CONFIGS_DIR],
    )
    model = build_torchmdnet(cfg)
    assert isinstance(model, TorchMDNetDataWrapper)
