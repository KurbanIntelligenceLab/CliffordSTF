"""Tests for ``cliffordstf.data.molecule3d``."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data, InMemoryDataset

from cliffordstf.data import LoaderSet
from cliffordstf.data.molecule3d import build_molecule3d_loaders


def _write_split(processed_dir: Path, name: str, *, n: int, natoms: int) -> None:
    """Write a collated ``{name}.pt`` to ``processed_dir``."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator().manual_seed(hash(name) & 0xFFFF)
    data_list: list[Data] = []
    for _ in range(n):
        props = torch.randn(8, generator=rng)
        data_list.append(
            Data(
                z=torch.randint(1, 10, (natoms,), generator=rng, dtype=torch.long),
                pos=torch.randn(natoms, 3, generator=rng) * 0.5,
                props=props,
                y=props[0].view(1),
            )
        )
    collated = InMemoryDataset.collate(data_list)
    torch.save(collated, processed_dir / f"{name}.pt")


def _build_fixture(root: Path, *, split_mode: str = "random") -> None:
    processed = root / f"processed_downstream_{split_mode}"
    _write_split(processed, "train", n=24, natoms=5)
    _write_split(processed, "val", n=6, natoms=5)
    _write_split(processed, "test", n=6, natoms=5)


def _cfg(root: Path, **dataset_overrides: object) -> OmegaConf:
    base = {
        "seed": 0,
        "dataset": {
            "name": "molecule3d",
            "data_root": str(root),
            "task_type": "scalar",
            "target_name": "homo_lumo_gap",
            "target": 0,
            "split_mode": "random",
            **dataset_overrides,
        },
        "training": {"batch_size": 4, "num_workers": 0},
    }
    return OmegaConf.create(base)


def test_build_molecule3d_loaders_returns_single_loader_set(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    loader_sets = build_molecule3d_loaders(_cfg(tmp_path))
    assert len(loader_sets) == 1
    ls = loader_sets[0]
    assert isinstance(ls, LoaderSet)
    assert ls.fold == 1
    assert ls.k_folds == 1
    assert ls.test is not None


def test_build_molecule3d_loaders_runtime_stats_has_target_name(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    ls = build_molecule3d_loaders(_cfg(tmp_path))[0]
    stats = dict(ls.runtime_stats)
    assert stats["target_name"] == "homo_lumo_gap"
    assert isinstance(stats["mean"], torch.Tensor)
    assert isinstance(stats["std"], torch.Tensor)
    assert stats["std"].item() > 0.0


def test_build_molecule3d_loaders_split_counts_match_fixture(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    ls = build_molecule3d_loaders(_cfg(tmp_path))[0]
    n_train = sum(b.num_graphs for b in ls.train)
    n_val = sum(b.num_graphs for b in ls.val)
    n_test = sum(b.num_graphs for b in ls.test)
    assert n_train == 24
    assert n_val == 6
    assert n_test == 6


def test_build_molecule3d_loaders_respects_max_train_samples(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    cfg = _cfg(tmp_path, max_train_samples=10)
    ls = build_molecule3d_loaders(cfg)[0]
    n_train = sum(b.num_graphs for b in ls.train)
    assert n_train == 10


def test_build_molecule3d_loaders_scaffold_split_dir(tmp_path: Path) -> None:
    _build_fixture(tmp_path, split_mode="scaffold")
    ls = build_molecule3d_loaders(_cfg(tmp_path, split_mode="scaffold"))[0]
    assert sum(b.num_graphs for b in ls.train) == 24


def test_build_molecule3d_loaders_missing_processed_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(Exception):  # noqa: B017 - PyG raises various IOErrors
        build_molecule3d_loaders(_cfg(tmp_path))
