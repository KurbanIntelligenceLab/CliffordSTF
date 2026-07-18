"""Tests for ``cliffordstf.data.qm9.build_qm9_loaders``."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data, InMemoryDataset

from cliffordstf.data import LoaderSet
from cliffordstf.data.qm9 import build_qm9_loaders


def _write_qm9_pt(root: Path, *, n_samples: int = 60, n_atoms: int = 4) -> None:
    """Write a synthetic ``qm9.pt`` mirroring PyG's QM9 collation."""
    root.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator().manual_seed(0)
    data_list = []
    for _ in range(n_samples):
        data_list.append(
            Data(
                z=torch.randint(1, 10, (n_atoms,), generator=rng, dtype=torch.long),
                pos=torch.randn(n_atoms, 3, generator=rng) * 0.5,
                y=torch.randn(1, 12, generator=rng) * 2.0,
            )
        )
    collated = InMemoryDataset.collate(data_list)
    torch.save(collated, root / "qm9.pt")


def _cfg(data_root: Path, **dataset_overrides: object) -> OmegaConf:
    base = {
        "seed": 0,
        "dataset": {
            "name": "qm9",
            "data_root": str(data_root),
            "target_name": "U0",
            "task_type": "scalar",
            **dataset_overrides,
        },
        "training": {"batch_size": 4, "num_workers": 0},
    }
    return OmegaConf.create(base)


def test_build_qm9_loaders_returns_single_loader_set(tmp_path: Path) -> None:
    _write_qm9_pt(tmp_path, n_samples=60, n_atoms=4)
    loader_sets = build_qm9_loaders(_cfg(tmp_path))
    assert len(loader_sets) == 1
    ls = loader_sets[0]
    assert isinstance(ls, LoaderSet)
    assert ls.fold == 1
    assert ls.k_folds == 1
    assert ls.test is not None


def test_build_qm9_loaders_runtime_stats_carries_target(tmp_path: Path) -> None:
    _write_qm9_pt(tmp_path, n_samples=60, n_atoms=4)
    ls = build_qm9_loaders(_cfg(tmp_path))[0]
    stats = dict(ls.runtime_stats)
    assert stats["target_name"] == "U0"
    assert stats["target_idx"] == 7
    assert isinstance(stats["mean"], torch.Tensor)
    assert isinstance(stats["std"], torch.Tensor)
    assert stats["std"].item() > 0.0


def test_build_qm9_loaders_batch_has_normalized_energy(tmp_path: Path) -> None:
    _write_qm9_pt(tmp_path, n_samples=60, n_atoms=4)
    ls = build_qm9_loaders(_cfg(tmp_path))[0]
    batch = next(iter(ls.train))
    assert hasattr(batch, "energy")
    assert batch.energy.dim() == 1
    assert torch.isfinite(batch.energy).all()


def test_build_qm9_loaders_zscore_train_mean_is_zero(tmp_path: Path) -> None:
    _write_qm9_pt(tmp_path, n_samples=80, n_atoms=4)
    ls = build_qm9_loaders(_cfg(tmp_path))[0]
    train_energy = torch.cat([b.energy for b in ls.train])
    assert abs(train_energy.mean().item()) < 1e-5
    assert abs(train_energy.std().item() - 1.0) < 1e-3


def test_build_qm9_loaders_split_sizes_are_80_10_10(tmp_path: Path) -> None:
    _write_qm9_pt(tmp_path, n_samples=100, n_atoms=4)
    ls = build_qm9_loaders(_cfg(tmp_path))[0]
    n_train = sum(b.num_graphs for b in ls.train)
    n_val = sum(b.num_graphs for b in ls.val)
    n_test = sum(b.num_graphs for b in ls.test)
    assert n_train == 80
    assert n_val == 10
    assert n_test == 10


def test_build_qm9_loaders_respects_debug_subset(tmp_path: Path) -> None:
    _write_qm9_pt(tmp_path, n_samples=60, n_atoms=4)
    cfg = _cfg(tmp_path, debug_subset=20)
    ls = build_qm9_loaders(cfg)[0]
    total = sum(b.num_graphs for b in ls.train) + sum(b.num_graphs for b in ls.val)
    total += sum(b.num_graphs for b in ls.test)
    assert total == 20


def test_build_qm9_loaders_alternative_target(tmp_path: Path) -> None:
    _write_qm9_pt(tmp_path, n_samples=60, n_atoms=4)
    cfg = _cfg(tmp_path, target_name="gap")
    ls = build_qm9_loaders(cfg)[0]
    stats = dict(ls.runtime_stats)
    assert stats["target_idx"] == 4
    assert stats["target_name"] == "gap"


def test_build_qm9_loaders_missing_pt_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_qm9_loaders(_cfg(tmp_path))
