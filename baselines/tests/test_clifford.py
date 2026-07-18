"""Tests for the ``clifford`` baseline registered in ``baselines.AVAILABLE_MODELS``."""

from __future__ import annotations

from baselines import AVAILABLE_MODELS, CONFIGS_DIR
from cliffordstf.io.config import load_config


def test_clifford_is_registered_in_catalog() -> None:
    assert "clifford" in AVAILABLE_MODELS


def test_configs_dir_points_at_packaged_yamls() -> None:
    assert (CONFIGS_DIR / "model" / "clifford.yaml").is_file()


def test_clifford_yaml_resolves_with_extra_search_paths() -> None:
    cfg = load_config(
        argv=["model.name=clifford"],
        extra_search_paths=[CONFIGS_DIR],
    )
    assert cfg.model.name == "clifford"
    assert cfg.model.n_channels == 80
    assert cfg.model.n_interactions == 5
    assert cfg.model.cutoff == 6.0
    assert cfg.model.stf_mode == "none"
    assert cfg.model.use_hodge_forces is False
    assert cfg.model.use_adaptive_routing is False
    assert cfg.model.use_cross_track is False
    assert cfg.model.use_l2 is False
