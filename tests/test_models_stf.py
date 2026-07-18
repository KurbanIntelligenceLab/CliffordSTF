"""Smoke tests for the public ``CliffordSTF`` model and its three factories."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data

from cliffordstf.models import (
    AVAILABLE_MODELS,
    build_clifford_stf,
    build_clifford_stf_full,
    build_clifford_stf_full_10m,
    build_model,
)
from cliffordstf.models.wrapper import (
    ABLATION_CONFIGS,
    CliffordSTFWrapper,
    build_from_ablation,
)


def _tiny_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "model": {
                "n_channels": 8,
                "n_interactions": 1,
                "cutoff": 4.0,
                "n_rbf": 8,
                "n_hidden_output": 16,
                "max_neighbors": 10,
                "n_heads": 2,
                "use_self_interaction": False,
                "use_gp_readout": False,
            }
        }
    )


def _tiny_data() -> Data:
    torch.manual_seed(0)
    return Data(
        z=torch.randint(1, 30, (5,)),
        pos=torch.randn(5, 3) * 2.0,
        batch=torch.zeros(5, dtype=torch.long),
    )


def test_available_models_keys_are_exactly_three_variants():
    assert set(AVAILABLE_MODELS) == {
        "clifford_stf",
        "clifford_stf_full",
        "clifford_stf_full_10m",
    }


def test_available_models_is_immutable_mapping():
    with pytest.raises(TypeError):
        AVAILABLE_MODELS["new_variant"] = lambda cfg: None  # type: ignore[index]


@pytest.mark.parametrize(
    "factory",
    [build_clifford_stf, build_clifford_stf_full, build_clifford_stf_full_10m],
)
def test_each_factory_returns_a_clifford_stf_wrapper(
    factory: Callable[[DictConfig], CliffordSTFWrapper],
):
    cfg = _tiny_cfg()
    model = factory(cfg)
    assert isinstance(model, CliffordSTFWrapper)


@pytest.mark.parametrize("name", ["clifford_stf", "clifford_stf_full", "clifford_stf_full_10m"])
def test_build_model_resolves_each_variant(name: str):
    cfg = _tiny_cfg()
    model = build_model(name, cfg)
    assert isinstance(model, CliffordSTFWrapper)


def test_build_model_raises_for_unknown_name():
    with pytest.raises(KeyError, match="Unknown model"):
        build_model("not_a_real_model", _tiny_cfg())


@pytest.mark.parametrize("name", ["clifford_stf", "clifford_stf_full", "clifford_stf_full_10m"])
def test_factory_forward_pass_returns_finite_energy_and_forces(name: str):
    cfg = _tiny_cfg()
    model = build_model(name, cfg).eval()
    data = _tiny_data()
    out = model(data)
    assert isinstance(out, tuple)
    energy, forces = out
    assert energy.shape == (1,)
    assert forces.shape == (5, 3)
    assert torch.isfinite(energy).all()
    assert torch.isfinite(forces).all()


def test_ablation_configs_has_twelve_named_variants():
    assert len(ABLATION_CONFIGS) == 12
    assert "baseline" in ABLATION_CONFIGS
    assert "stf2" in ABLATION_CONFIGS
    assert "full" in ABLATION_CONFIGS


def test_build_from_ablation_baseline_returns_wrapper():
    model = build_from_ablation(
        "baseline",
        n_channels=8,
        n_interactions=1,
        cutoff=4.0,
        n_rbf=8,
        n_hidden_output=16,
        max_neighbors=10,
        n_heads=2,
        use_self_interaction=False,
        use_gp_readout=False,
        use_ema=False,
    )
    assert isinstance(model, CliffordSTFWrapper)


def test_build_from_ablation_unknown_raises_value_error():
    with pytest.raises(ValueError, match="Unknown config"):
        build_from_ablation("not_a_real_ablation")
