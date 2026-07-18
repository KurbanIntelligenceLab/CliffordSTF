"""Tests for ``cliffordstf.data.md17.build_md17_loaders``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from cliffordstf.data import LoaderSet
from cliffordstf.data.md17 import build_md17_loaders


def _write_npz(data_root: Path, molecule: str, *, n_structures: int = 20, natoms: int = 5) -> None:
    """Write a synthetic ``rmd17_{molecule}.npz`` under ``data_root/{molecule}``."""
    mol_dir = data_root / molecule
    mol_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    np.savez(
        mol_dir / f"rmd17_{molecule}.npz",
        nuclear_charges=rng.integers(1, 30, size=(natoms,), dtype=np.int64),
        coords=rng.standard_normal(size=(n_structures, natoms, 3)).astype(np.float32),
        energies=rng.standard_normal(size=(n_structures,)).astype(np.float32),
        forces=rng.standard_normal(size=(n_structures, natoms, 3)).astype(np.float32),
    )


def _cfg(data_root: Path, **overrides: object) -> OmegaConf:
    base = {
        "seed": 0,
        "dataset": {
            "name": "md17",
            "data_root": str(data_root),
            "molecule": "aspirin",
            "k_folds": 1,
            "val_frac": 0.2,
            "max_train_samples": 1000,
            "test_samples": 4,
        },
        "training": {"batch_size": 2, "num_workers": 0},
    }
    return OmegaConf.create({**base, **overrides})


def test_build_md17_loaders_k_folds_1_returns_single_set(tmp_path: Path) -> None:
    _write_npz(tmp_path, "aspirin", n_structures=20, natoms=5)
    loader_sets = build_md17_loaders(_cfg(tmp_path))
    assert len(loader_sets) == 1
    ls = loader_sets[0]
    assert isinstance(ls, LoaderSet)
    assert ls.fold == 1
    assert ls.k_folds == 1
    assert ls.extra_parts == ("aspirin", "fold1")
    assert ls.test is not None


def test_build_md17_loaders_k_folds_3_returns_three_sets(tmp_path: Path) -> None:
    _write_npz(tmp_path, "aspirin", n_structures=30, natoms=5)
    loader_sets = build_md17_loaders(
        _cfg(
            tmp_path,
            **{
                "dataset": {
                    "name": "md17",
                    "data_root": str(tmp_path),
                    "molecule": "aspirin",
                    "k_folds": 3,
                    "val_frac": 0.2,
                }
            },
        )
    )
    assert len(loader_sets) == 3
    folds = [ls.fold for ls in loader_sets]
    assert folds == [1, 2, 3]
    assert all(ls.k_folds == 3 for ls in loader_sets)


def test_build_md17_loaders_batch_has_pyg_attrs(tmp_path: Path) -> None:
    _write_npz(tmp_path, "aspirin", n_structures=20, natoms=5)
    loader_sets = build_md17_loaders(_cfg(tmp_path))
    batch = next(iter(loader_sets[0].train))
    assert hasattr(batch, "z")
    assert hasattr(batch, "pos")
    assert hasattr(batch, "energy")
    assert hasattr(batch, "force")
    assert batch.pos.dtype == torch.float32
    assert batch.force.shape[-1] == 3
    assert batch.force.dtype == torch.float32


def test_build_md17_loaders_caps_max_train_samples(tmp_path: Path) -> None:
    _write_npz(tmp_path, "aspirin", n_structures=40, natoms=5)
    cfg = _cfg(
        tmp_path,
        dataset={
            "name": "md17",
            "data_root": str(tmp_path),
            "molecule": "aspirin",
            "k_folds": 1,
            "val_frac": 0.0,
            "max_train_samples": 5,
            "test_samples": 4,
        },
    )
    loader_sets = build_md17_loaders(cfg)
    train_loader = loader_sets[0].train
    n_train_samples = sum(b.num_graphs for b in train_loader)
    assert n_train_samples == 5


def test_build_md17_loaders_unknown_molecule_raises(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        dataset={
            "name": "md17",
            "data_root": str(tmp_path),
            "molecule": "unobtanium",
        },
    )
    with pytest.raises(ValueError, match="Unknown MD17 molecule"):
        build_md17_loaders(cfg)


def test_build_md17_loaders_all_picks_only_present_molecules(tmp_path: Path) -> None:
    _write_npz(tmp_path, "aspirin", n_structures=20, natoms=5)
    _write_npz(tmp_path, "benzene", n_structures=20, natoms=4)
    cfg = _cfg(
        tmp_path,
        dataset={
            "name": "md17",
            "data_root": str(tmp_path),
            "molecule": "all",
            "k_folds": 1,
            "val_frac": 0.2,
            "test_samples": 4,
        },
    )
    loader_sets = build_md17_loaders(cfg)
    molecules = sorted({ls.extra_parts[0] for ls in loader_sets})
    assert molecules == ["aspirin", "benzene"]


def test_build_md17_loaders_no_data_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No MD17 molecule subdirectories"):
        build_md17_loaders(
            _cfg(tmp_path, dataset={"name": "md17", "data_root": str(tmp_path), "molecule": "all"})
        )


def test_build_md17_loaders_runtime_stats_is_empty_in_step_6a(tmp_path: Path) -> None:
    _write_npz(tmp_path, "aspirin", n_structures=20, natoms=5)
    loader_sets = build_md17_loaders(_cfg(tmp_path))
    assert dict(loader_sets[0].runtime_stats) == {}
