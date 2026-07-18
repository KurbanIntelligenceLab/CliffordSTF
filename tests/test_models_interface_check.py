"""Tests for the ``cfg.model.interface`` runtime guard in cliffordstf.models."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from cliffordstf.models import (
    SUPPORTED_MODEL_INTERFACES,
    build_clifford_stf,
)


def _tiny_cfg(**model_overrides: object) -> object:
    base = {
        "model": {
            "name": "clifford_stf",
            "interface": "data_wrapper",
            "n_atom_types": 100,
            "n_channels": 8,
            "n_interactions": 1,
            "n_rbf": 8,
            "cutoff": 4.0,
            "n_hidden_output": 8,
            "max_neighbors": 10,
            "stf_mode": "stf2+stf3",
            "use_hodge_forces": True,
            "use_cross_track": True,
            "use_self_interaction": False,
            "use_gp_readout": False,
            "use_compile": False,
        }
    }
    base["model"].update(model_overrides)
    return OmegaConf.create(base)


def test_data_wrapper_is_a_supported_interface() -> None:
    assert "data_wrapper" in SUPPORTED_MODEL_INTERFACES


def test_build_accepts_data_wrapper_interface() -> None:
    cfg = _tiny_cfg(interface="data_wrapper")
    build_clifford_stf(cfg)


def test_build_accepts_missing_interface_field() -> None:
    cfg = _tiny_cfg()
    del cfg.model.interface
    build_clifford_stf(cfg)


def test_build_rejects_unknown_interface() -> None:
    cfg = _tiny_cfg(interface="something_else")
    with pytest.raises(ValueError, match=r"Unsupported cfg\.model\.interface"):
        build_clifford_stf(cfg)
