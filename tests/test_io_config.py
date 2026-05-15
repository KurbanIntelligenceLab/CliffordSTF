"""Tests for ``cliffordstf.io.config.load_config``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from omegaconf import OmegaConf

from cliffordstf.io.config import load_config

if TYPE_CHECKING:
    pass


@pytest.fixture
def configs_root(tmp_path: Path) -> Path:
    """Build a self-contained config tree for testing."""
    root = tmp_path / "configs"
    (root / "dataset").mkdir(parents=True)
    (root / "model").mkdir(parents=True)
    (root / "base.yaml").write_text(
        "training:\n  lr: 1e-3\n  batch_size: 32\ndataset: {}\nmodel: {}\n"
    )
    (root / "dataset" / "tiny.yaml").write_text("dataset:\n  name: tiny\n  num_samples: 100\n")
    (root / "model" / "linear.yaml").write_text("model:\n  name: linear\n  hidden: 64\n")
    return root


def test_load_config_merges_base_dataset_model(configs_root: Path) -> None:
    cfg = load_config(
        argv=["dataset.name=tiny", "model.name=linear"],
        extra_search_paths=[configs_root],
    )
    assert cfg.training.lr == 1e-3
    assert cfg.training.batch_size == 32
    assert cfg.dataset.name == "tiny"
    assert cfg.dataset.num_samples == 100
    assert cfg.model.name == "linear"
    assert cfg.model.hidden == 64


def test_cli_overrides_have_highest_precedence(configs_root: Path) -> None:
    cfg = load_config(
        argv=["dataset.name=tiny", "model.name=linear", "training.lr=5e-4"],
        extra_search_paths=[configs_root],
    )
    assert cfg.training.lr == 5e-4
    assert cfg.dataset.num_samples == 100


def test_missing_dataset_yaml_is_silently_ignored(configs_root: Path) -> None:
    cfg = load_config(
        argv=["dataset.name=does_not_exist", "model.name=linear"],
        extra_search_paths=[configs_root],
    )
    assert cfg.dataset.name == "does_not_exist"
    assert cfg.model.name == "linear"


def test_override_yaml_layered_between_dataset_and_cli(
    configs_root: Path,
    tmp_path: Path,
) -> None:
    override = tmp_path / "user_override.yaml"
    override.write_text("training:\n  lr: 7e-4\nmodel:\n  hidden: 128\n")
    cfg = load_config(
        argv=[
            "dataset.name=tiny",
            "model.name=linear",
            "--config",
            str(override),
            "training.lr=9e-9",
        ],
        extra_search_paths=[configs_root],
    )
    assert cfg.training.lr == 9e-9
    assert cfg.model.hidden == 128


def test_load_config_uses_defaults_dict(configs_root: Path) -> None:
    cfg = load_config(
        argv=["model.name=linear"],
        defaults={"dataset": {"name": "tiny"}},
        extra_search_paths=[configs_root],
    )
    assert cfg.dataset.name == "tiny"
    assert cfg.dataset.num_samples == 100


def test_extra_search_paths_layered_after_default(configs_root: Path, tmp_path: Path) -> None:
    extras = tmp_path / "baselines_configs"
    (extras / "model").mkdir(parents=True)
    (extras / "model" / "plugged_in.yaml").write_text("model:\n  name: plugged_in\n  hidden: 256\n")
    cfg = load_config(
        argv=["model.name=plugged_in"],
        extra_search_paths=[configs_root, extras],
    )
    assert cfg.model.name == "plugged_in"
    assert cfg.model.hidden == 256


def test_load_config_returns_dictconfig(configs_root: Path) -> None:
    cfg = load_config(argv=["model.name=linear"], extra_search_paths=[configs_root])
    assert OmegaConf.is_dict(cfg)
