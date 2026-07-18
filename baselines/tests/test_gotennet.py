"""Tests for the ``gotennet`` baseline (requires the [baselines] extra)."""

from __future__ import annotations

import pytest

pytest.importorskip("gotennet")

from baselines import AVAILABLE_MODELS, CONFIGS_DIR  # noqa: E402
from baselines.gotennet import GotenNetDataWrapper, build_gotennet  # noqa: E402
from cliffordstf.io.config import load_config  # noqa: E402


def test_gotennet_is_registered_in_catalog() -> None:
    assert "gotennet" in AVAILABLE_MODELS


def test_gotennet_yaml_resolves_with_extra_search_paths() -> None:
    cfg = load_config(
        argv=["model.name=gotennet"],
        extra_search_paths=[CONFIGS_DIR],
    )
    assert cfg.model.name == "gotennet"
    assert cfg.model.n_atom_basis == 96
    assert cfg.model.n_interactions == 5
    assert cfg.model.cutoff == 6.0


def test_build_gotennet_returns_data_wrapper() -> None:
    cfg = load_config(
        argv=[
            "model.name=gotennet",
            "model.n_atom_basis=8",
            "model.n_interactions=1",
            "model.cutoff=4.0",
            "model.max_num_neighbors=10",
            "dataset.task_type=energy_forces",
        ],
        extra_search_paths=[CONFIGS_DIR],
    )
    model = build_gotennet(cfg)
    assert isinstance(model, GotenNetDataWrapper)
