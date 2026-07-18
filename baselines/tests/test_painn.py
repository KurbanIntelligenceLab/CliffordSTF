"""Tests for the ``painn`` baseline.

Requires the ``[baselines]`` extra (``schnetpack``). Tests skip cleanly
when the dep is missing so the base test suite stays green.
"""

from __future__ import annotations

import pytest

pytest.importorskip("schnetpack")

from baselines import AVAILABLE_MODELS, CONFIGS_DIR  # noqa: E402
from baselines.painn import PaiNNDataWrapper, build_painn  # noqa: E402
from cliffordstf.io.config import load_config  # noqa: E402


def test_painn_is_registered_in_catalog() -> None:
    assert "painn" in AVAILABLE_MODELS


def test_painn_yaml_resolves_with_extra_search_paths() -> None:
    cfg = load_config(
        argv=["model.name=painn"],
        extra_search_paths=[CONFIGS_DIR],
    )
    assert cfg.model.name == "painn"
    assert cfg.model.hidden_channels == 144
    assert cfg.model.num_layers == 4
    assert cfg.model.num_rbf == 64
    assert cfg.model.cutoff == 6.0


def test_build_painn_returns_data_wrapper() -> None:
    cfg = load_config(
        argv=[
            "model.name=painn",
            "model.hidden_channels=8",
            "model.num_layers=1",
            "model.num_rbf=8",
            "model.cutoff=4.0",
        ],
        extra_search_paths=[CONFIGS_DIR],
    )
    model = build_painn(cfg)
    assert isinstance(model, PaiNNDataWrapper)
