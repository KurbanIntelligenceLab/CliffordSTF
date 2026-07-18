"""Tests for ``cliffordstf.data.loader.build_loader``."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from cliffordstf.data.loader import build_loader


class _TinyDataset(Dataset[torch.Tensor]):
    def __init__(self, n: int = 8) -> None:
        self._data = [torch.tensor([i], dtype=torch.float32) for i in range(n)]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._data[idx]


class _LmdbAwareDataset(_TinyDataset):
    closed: bool = False

    def _close_envs(self) -> None:
        type(self).closed = True


def test_build_loader_respects_batch_size_and_shuffle() -> None:
    cfg = OmegaConf.create({"training": {"batch_size": 4, "num_workers": 0}})
    ds = _TinyDataset(n=8)
    loader = build_loader(ds, cfg, shuffle=False)
    batches = list(loader)
    assert len(batches) == 2
    assert batches[0].shape == (4, 1)


def test_build_loader_uses_lmdb_worker_init_for_lmdb_datasets() -> None:
    cfg = OmegaConf.create({"training": {"batch_size": 2, "num_workers": 0}})
    ds: Any = _LmdbAwareDataset(n=4)
    loader = build_loader(ds, cfg)
    assert loader.worker_init_fn is not None


def test_build_loader_omits_worker_init_for_plain_datasets() -> None:
    cfg = OmegaConf.create({"training": {"batch_size": 2, "num_workers": 0}})
    ds = _TinyDataset(n=4)
    loader = build_loader(ds, cfg)
    assert loader.worker_init_fn is None


def test_build_loader_omits_prefetch_factor_when_num_workers_zero() -> None:
    cfg = OmegaConf.create({"training": {"batch_size": 2, "num_workers": 0, "prefetch_factor": 8}})
    ds = _TinyDataset(n=4)
    loader = build_loader(ds, cfg)
    assert loader.prefetch_factor is None
