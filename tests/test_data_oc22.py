"""Tests for ``cliffordstf.data.oc22``."""

from __future__ import annotations

import pickle
from pathlib import Path

import lmdb
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from cliffordstf.data import LoaderSet
from cliffordstf.data.oc22 import build_oc22_is2re_loaders, build_oc22_s2ef_loaders


def _write_lmdb_shard(path: Path, *, n_entries: int, natoms: int, has_force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), subdir=False, map_size=1 << 26, writemap=False)
    rng = torch.Generator().manual_seed(7)
    with env.begin(write=True) as txn:
        for i in range(n_entries):
            data = Data(
                z=torch.randint(1, 30, (natoms,), generator=rng, dtype=torch.long),
                pos=torch.randn(natoms, 3, generator=rng) * 0.5,
                y=torch.randn(1, generator=rng),
                cell=torch.eye(3) * 10.0,
            )
            if has_force:
                data.force = torch.randn(natoms, 3, generator=rng)
            txn.put(str(i).encode("ascii"), pickle.dumps(data))
        txn.put(b"length", pickle.dumps(n_entries))
    env.close()


def _cfg(tmp_path: Path, name: str, task: str) -> OmegaConf:
    return OmegaConf.create(
        {
            "seed": 0,
            "dataset": {
                "name": name,
                "data_root": str(tmp_path),
                "task_type": task,
                "task": task,
                "split_train": "train",
                "split_val": "val_id",
                "split_test": None,
            },
            "training": {"batch_size": 2, "num_workers": 0},
        }
    )


def test_build_oc22_s2ef_uses_total_subdir(tmp_path: Path) -> None:
    # OC22 lays out as {root}/s2ef-total/{split}/
    _write_lmdb_shard(
        tmp_path / "s2ef-total" / "train" / "data.lmdb",
        n_entries=6,
        natoms=4,
        has_force=True,
    )
    _write_lmdb_shard(
        tmp_path / "s2ef-total" / "val_id" / "data.lmdb",
        n_entries=3,
        natoms=4,
        has_force=True,
    )
    loader_sets = build_oc22_s2ef_loaders(_cfg(tmp_path, "oc22_s2ef", "s2ef"))
    assert len(loader_sets) == 1
    ls = loader_sets[0]
    assert isinstance(ls, LoaderSet)
    assert sum(b.num_graphs for b in ls.train) == 6
    assert sum(b.num_graphs for b in ls.val) == 3
    batch = next(iter(ls.train))
    assert hasattr(batch, "force")
    assert batch.force.shape[-1] == 3


def test_build_oc22_is2re_uses_total_subdir_without_force(tmp_path: Path) -> None:
    _write_lmdb_shard(
        tmp_path / "is2re-total" / "train" / "data.lmdb",
        n_entries=4,
        natoms=3,
        has_force=False,
    )
    _write_lmdb_shard(
        tmp_path / "is2re-total" / "val_id" / "data.lmdb",
        n_entries=2,
        natoms=3,
        has_force=False,
    )
    ls = build_oc22_is2re_loaders(_cfg(tmp_path, "oc22_is2re", "is2re"))[0]
    assert sum(b.num_graphs for b in ls.train) == 4
    assert sum(b.num_graphs for b in ls.val) == 2
