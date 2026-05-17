"""Tests for the ``ictp_l*`` baselines (requires the upstream ictp package)."""

from __future__ import annotations

import pytest

pytest.importorskip("ictp")

from baselines import AVAILABLE_MODELS, CONFIGS_DIR  # noqa: E402
from baselines.ictp import (  # noqa: E402
    ICTPDataWrapper,
    build_ictp_l1,
    build_ictp_l2,
    build_ictp_l3,
)
from cliffordstf.io.config import load_config  # noqa: E402


def test_ictp_variants_are_registered_in_catalog() -> None:
    for name in ("ictp_l1", "ictp_l2", "ictp_l3"):
        assert name in AVAILABLE_MODELS


def test_ictp_l1_yaml_resolves() -> None:
    cfg = load_config(argv=["model.name=ictp_l1"], extra_search_paths=[CONFIGS_DIR])
    assert cfg.model.l_max_hidden_feats == 1
    assert cfg.model.n_hidden_feats == 10
    assert cfg.model.l_max_edge_attrs == 3


def test_ictp_l2_yaml_resolves() -> None:
    cfg = load_config(argv=["model.name=ictp_l2"], extra_search_paths=[CONFIGS_DIR])
    assert cfg.model.l_max_hidden_feats == 2
    assert cfg.model.n_hidden_feats == 8


def test_ictp_l3_yaml_resolves() -> None:
    cfg = load_config(argv=["model.name=ictp_l3"], extra_search_paths=[CONFIGS_DIR])
    assert cfg.model.l_max_hidden_feats == 3
    assert cfg.model.n_hidden_feats == 7


def test_build_each_ictp_variant_returns_wrapper() -> None:
    base = [
        "model.n_hidden_feats=4",
        "model.n_product_feats=4",
        "model.n_basis=4",
        "model.cutoff=4.0",
        "model.n_species=10",
        "model.max_neighbors=10",
        "dataset.task_type=energy_forces",
    ]
    for name, builder in (
        ("ictp_l1", build_ictp_l1),
        ("ictp_l2", build_ictp_l2),
        ("ictp_l3", build_ictp_l3),
    ):
        cfg = load_config(
            argv=[f"model.name={name}", "model.l_max_hidden_feats=1", *base],
            extra_search_paths=[CONFIGS_DIR],
        )
        assert isinstance(builder(cfg), ICTPDataWrapper)
