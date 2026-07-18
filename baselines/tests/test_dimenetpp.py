"""Tests for the ``dimenetpp`` baseline."""

from __future__ import annotations

from baselines import AVAILABLE_MODELS, CONFIGS_DIR
from baselines.dimenetpp import DimeNetPPDataWrapper, build_dimenetpp
from cliffordstf.io.config import load_config


def test_dimenetpp_is_registered_in_catalog() -> None:
    assert "dimenetpp" in AVAILABLE_MODELS


def test_dimenetpp_yaml_resolves_with_extra_search_paths() -> None:
    cfg = load_config(
        argv=["model.name=dimenetpp"],
        extra_search_paths=[CONFIGS_DIR],
    )
    assert cfg.model.name == "dimenetpp"
    assert cfg.model.hidden_channels == 128
    assert cfg.model.num_blocks == 4
    assert cfg.model.basis_emb_size == 8
    assert cfg.model.num_spherical == 7
    assert cfg.model.num_radial == 6
    assert cfg.model.cutoff == 6.0


def test_build_dimenetpp_returns_data_wrapper() -> None:
    cfg = load_config(
        argv=[
            "model.name=dimenetpp",
            "model.hidden_channels=8",
            "model.num_blocks=1",
            "model.basis_emb_size=4",
            "model.out_emb_channels=8",
            "model.num_spherical=3",
            "model.num_radial=3",
            "model.cutoff=4.0",
        ],
        extra_search_paths=[CONFIGS_DIR],
    )
    model = build_dimenetpp(cfg)
    assert isinstance(model, DimeNetPPDataWrapper)
