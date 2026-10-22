"""Shared ``DataLoader`` construction helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch.utils.data import Dataset
    from torch_geometric.loader import DataLoader as PyGDataLoader


def _lmdb_worker_init_fn(_worker_id: int) -> None:
    """Reopen LMDB environments inside DataLoader workers."""
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    dataset = info.dataset
    close = getattr(dataset, "_close_envs", None)
    if callable(close):
        close()


def build_loader(
    dataset: Dataset[Any],
    cfg: DictConfig,
    *,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
) -> PyGDataLoader:
    """Construct a ``torch_geometric.loader.DataLoader`` using shared config."""
    from torch_geometric.loader import DataLoader

    batch_size = int(cfg.training.batch_size)
    num_workers = int(cfg.training.get("num_workers", 0))
    pin_memory = bool(cfg.training.get("pin_memory", False))
    persistent = bool(cfg.training.get("persistent_workers", False)) and num_workers > 0
    prefetch_factor = (
        int(cfg.training.get("prefetch_factor", 2)) if num_workers > 0 else None
    )

    worker_init = _lmdb_worker_init_fn if hasattr(dataset, "_close_envs") else None

    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent,
        "generator": generator,
    }
    if prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    if worker_init is not None:
        kwargs["worker_init_fn"] = worker_init

    return DataLoader(dataset, **kwargs)


__all__ = ["build_loader"]
