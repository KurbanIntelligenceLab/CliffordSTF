"""Tests for ``cliffordstf.data.oc20`` and the LMDB dataset class."""

from __future__ import annotations

import pickle
from pathlib import Path

import lmdb
import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from cliffordstf.data import LoaderSet
from cliffordstf.data._oc20_io import OC20LMDBDataset
from cliffordstf.data.oc20 import (
    build_oc20_is2re_loaders,
    build_oc20_s2ef_loaders,
)


def _write_lmdb_shard(
    path: Path,
    *,
    n_entries: int,
    natoms: int,
    has_force: bool,
) -> None:
    """Write a synthetic OC20-style LMDB shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), subdir=False, map_size=1 << 26, writemap=False)
    rng = torch.Generator().manual_seed(0)
    with env.begin(write=True) as txn:
        for i in range(n_entries):
            data = Data(
                z=torch.randint(1, 30, (natoms,), generator=rng, dtype=torch.long),
                pos=torch.randn(natoms, 3, generator=rng) * 0.5,
                y=torch.randn(1, generator=rng),
                cell=torch.eye(3) * 10.0,
                tags=torch.randint(0, 3, (natoms,), generator=rng, dtype=torch.long),
                fixed=torch.zeros(natoms, dtype=torch.bool),
                sid=i,
                fid=0,
            )
            if has_force:
                data.force = torch.randn(natoms, 3, generator=rng)
            txn.put(str(i).encode("ascii"), pickle.dumps(data))
        txn.put(b"length", pickle.dumps(n_entries))
    env.close()


def _make_oc20_layout(
    root: Path, task: str, split: str, *, n_entries: int = 8, natoms: int = 4
) -> None:
    """Lay out an OC20-style LMDB directory: {root}/{task}/{split}/data.lmdb."""
    shard_path = root / task / split / "data.lmdb"
    _write_lmdb_shard(shard_path, n_entries=n_entries, natoms=natoms, has_force=(task == "s2ef"))


def test_oc20_lmdb_dataset_reads_synthetic_shard(tmp_path: Path) -> None:
    _make_oc20_layout(tmp_path, "s2ef", "val_id", n_entries=8, natoms=4)
    ds = OC20LMDBDataset(root=tmp_path, task="s2ef", split="val_id")
    assert len(ds) == 8
    item = ds[0]
    assert isinstance(item, Data)
    assert item.z.shape == (4,)
    assert item.pos.shape == (4, 3)
    assert item.y.shape == (1,)
    assert item.force.shape == (4, 3)
    assert item.cell.shape == (3, 3)


def test_oc20_lmdb_dataset_respects_max_samples(tmp_path: Path) -> None:
    _make_oc20_layout(tmp_path, "s2ef", "val_id", n_entries=8, natoms=4)
    ds = OC20LMDBDataset(root=tmp_path, task="s2ef", split="val_id", max_samples=3)
    assert len(ds) == 3


def test_oc20_lmdb_dataset_close_envs_resets_handles(tmp_path: Path) -> None:
    _make_oc20_layout(tmp_path, "s2ef", "val_id", n_entries=4, natoms=4)
    ds = OC20LMDBDataset(root=tmp_path, task="s2ef", split="val_id")
    _ = ds[0]
    assert ds._envs[0] is not None
    ds._close_envs()
    assert ds._envs == [None]
    _ = ds[0]  # reopens lazily
    assert ds._envs[0] is not None


def test_oc20_lmdb_dataset_pickle_round_trip_closes_envs(tmp_path: Path) -> None:
    _make_oc20_layout(tmp_path, "s2ef", "val_id", n_entries=4, natoms=4)
    ds = OC20LMDBDataset(root=tmp_path, task="s2ef", split="val_id")
    _ = ds[0]
    blob = pickle.dumps(ds)
    assert ds._envs[0] is None
    restored: OC20LMDBDataset = pickle.loads(blob)
    assert len(restored) == 4


def test_oc20_lmdb_dataset_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="LMDB directory not found"):
        OC20LMDBDataset(root=tmp_path, task="s2ef", split="val_id")


def test_oc20_is2re_layout_resolves_under_all_subdir(tmp_path: Path) -> None:
    # OC20 IS2RE tarball lays out as is2re/all/{split}/
    shard_path = tmp_path / "is2re" / "all" / "val_id" / "data.lmdb"
    _write_lmdb_shard(shard_path, n_entries=5, natoms=3, has_force=False)
    ds = OC20LMDBDataset(root=tmp_path, task="is2re", split="val_id")
    assert len(ds) == 5


def _cfg(tmp_path: Path, name: str, task_type: str, *, split_test: str | None = None) -> OmegaConf:
    base = {
        "seed": 0,
        "dataset": {
            "name": name,
            "data_root": str(tmp_path),
            "task_type": task_type,
            "task": task_type,
            "split_train": "train",
            "split_val": "val_id",
            "split_test": split_test,
        },
        "training": {"batch_size": 2, "num_workers": 0},
    }
    return OmegaConf.create(base)


def test_build_oc20_s2ef_returns_single_loader_set(tmp_path: Path) -> None:
    _make_oc20_layout(tmp_path, "s2ef", "train", n_entries=8, natoms=4)
    _make_oc20_layout(tmp_path, "s2ef", "val_id", n_entries=4, natoms=4)
    loader_sets = build_oc20_s2ef_loaders(_cfg(tmp_path, "oc20_s2ef", "s2ef"))
    assert len(loader_sets) == 1
    ls = loader_sets[0]
    assert isinstance(ls, LoaderSet)
    assert ls.test is None
    assert sum(b.num_graphs for b in ls.train) == 8
    assert sum(b.num_graphs for b in ls.val) == 4


def test_build_oc20_s2ef_with_test_split(tmp_path: Path) -> None:
    _make_oc20_layout(tmp_path, "s2ef", "train", n_entries=4, natoms=4)
    _make_oc20_layout(tmp_path, "s2ef", "val_id", n_entries=4, natoms=4)
    _make_oc20_layout(tmp_path, "s2ef", "val_ood_ads", n_entries=2, natoms=4)
    cfg = _cfg(tmp_path, "oc20_s2ef", "s2ef", split_test="val_ood_ads")
    ls = build_oc20_s2ef_loaders(cfg)[0]
    assert ls.test is not None
    assert sum(b.num_graphs for b in ls.test) == 2


def test_build_oc20_is2re_uses_all_subdir(tmp_path: Path) -> None:
    _write_lmdb_shard(
        tmp_path / "is2re" / "all" / "train" / "data.lmdb",
        n_entries=6,
        natoms=4,
        has_force=False,
    )
    _write_lmdb_shard(
        tmp_path / "is2re" / "all" / "val_id" / "data.lmdb",
        n_entries=3,
        natoms=4,
        has_force=False,
    )
    ls = build_oc20_is2re_loaders(_cfg(tmp_path, "oc20_is2re", "is2re"))[0]
    assert sum(b.num_graphs for b in ls.train) == 6
    assert sum(b.num_graphs for b in ls.val) == 3
    batch = next(iter(ls.val))
    assert not hasattr(batch, "force") or batch.force is None
