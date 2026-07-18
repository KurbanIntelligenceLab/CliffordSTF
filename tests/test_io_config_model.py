"""Tests for the packaged CliffordSTF model config YAMLs.

These tests load model configs from the real ``DEFAULT_CONFIGS_DIR`` (no
``extra_search_paths``) so they verify the packaged location is reachable
and the merge produces a fully-populated ``cfg.model`` block.
"""

from __future__ import annotations

from cliffordstf.io.config import load_config


def test_clifford_stf_resolves_from_package() -> None:
    cfg = load_config(argv=["model.name=clifford_stf"])
    assert cfg.model.name == "clifford_stf"
    assert cfg.model.interface == "data_wrapper"
    assert cfg.model.n_atom_types == 100
    assert cfg.model.n_channels == 60
    assert cfg.model.n_interactions == 5
    assert cfg.model.n_rbf == 50
    assert cfg.model.cutoff == 6.0
    assert cfg.model.n_hidden_output == 60
    assert cfg.model.max_neighbors == 50
    assert cfg.model.stf_mode == "stf2+stf3"
    assert cfg.model.use_hodge_forces is True
    assert cfg.model.use_cross_track is True
    assert cfg.model.use_self_interaction is False
    assert cfg.model.use_gp_readout is False
    assert cfg.model.use_compile is False


def test_clifford_stf_full_resolves_from_package() -> None:
    cfg = load_config(argv=["model.name=clifford_stf_full"])
    assert cfg.model.name == "clifford_stf_full"
    assert cfg.model.interface == "data_wrapper"
    assert cfg.model.n_channels == 60
    assert cfg.model.n_hidden_output == 60
    assert cfg.model.stf_mode == "stf2+stf3"
    assert cfg.model.use_hodge_forces is True
    assert cfg.model.use_adaptive_routing is True
    assert cfg.model.routing_mode == "learned"
    assert cfg.model.use_cross_track is True
    assert cfg.model.use_self_interaction is False
    assert cfg.model.use_gp_readout is False
    assert cfg.model.use_compile is False


def test_clifford_stf_full_10m_resolves_from_package() -> None:
    cfg = load_config(argv=["model.name=clifford_stf_full_10m"])
    assert cfg.model.name == "clifford_stf_full_10m"
    assert cfg.model.interface == "data_wrapper"
    assert cfg.model.n_channels == 188
    assert cfg.model.n_hidden_output == 256
    assert cfg.model.stf_mode == "stf2+stf3"
    assert cfg.model.use_hodge_forces is True
    assert cfg.model.use_adaptive_routing is True
    assert cfg.model.routing_mode == "learned"
    assert cfg.model.use_cross_track is True
    assert cfg.model.use_self_interaction is False
    assert cfg.model.use_gp_readout is False
    # ``use_compile`` is intentionally absent from the verbatim source;
    # the wrapper's constructor default (False) is the effective value.
    assert cfg.model.get("use_compile", False) is False
