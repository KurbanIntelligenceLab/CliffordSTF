"""Tests for the ``mace_l*`` baselines (requires the [baselines] extra)."""

from __future__ import annotations

import pytest

pytest.importorskip("mace")

from baselines import AVAILABLE_MODELS, CONFIGS_DIR  # noqa: E402
from baselines.mace import (  # noqa: E402
    MACEDataWrapper,
    build_mace_l1,
    build_mace_l2,
    build_mace_l3,
)
from cliffordstf.io.config import load_config  # noqa: E402


def test_mace_variants_are_registered_in_catalog() -> None:
    for name in ("mace_l1", "mace_l2", "mace_l3", "mace_l2_10m"):
        assert name in AVAILABLE_MODELS


def test_mace_l1_yaml_resolves() -> None:
    cfg = load_config(argv=["model.name=mace_l1"], extra_search_paths=[CONFIGS_DIR])
    assert cfg.model.max_ell == 1
    assert cfg.model.hidden_irreps == "62x0e + 62x1o"


def test_mace_l2_yaml_resolves() -> None:
    cfg = load_config(argv=["model.name=mace_l2"], extra_search_paths=[CONFIGS_DIR])
    assert cfg.model.max_ell == 2
    assert cfg.model.hidden_irreps == "49x0e + 49x1o + 49x2e"


def test_mace_l3_yaml_resolves() -> None:
    cfg = load_config(argv=["model.name=mace_l3"], extra_search_paths=[CONFIGS_DIR])
    assert cfg.model.max_ell == 3
    assert cfg.model.hidden_irreps == "28x0e + 28x1o + 28x2e + 28x3o"


def test_build_each_mace_variant_returns_wrapper() -> None:
    base_argv = [
        "model.hidden_irreps=4x0e",
        "model.num_elements=10",
        "model.cutoff=4.0",
        "model.num_bessel=4",
        "model.max_neighbors=10",
        "dataset.task_type=energy_forces",
    ]
    for name, builder in (
        ("mace_l1", build_mace_l1),
        ("mace_l2", build_mace_l2),
        ("mace_l3", build_mace_l3),
    ):
        cfg = load_config(
            argv=[f"model.name={name}", "model.max_ell=1", *base_argv],
            extra_search_paths=[CONFIGS_DIR],
        )
        assert isinstance(builder(cfg), MACEDataWrapper)
