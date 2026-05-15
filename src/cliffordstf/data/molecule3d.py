"""Molecule3D dataset factory."""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import torch

from cliffordstf.data._molecule3d_io import Molecule3DProps
from cliffordstf.data.loader import build_loader

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch.utils.data import Dataset
    from torch_geometric.data import Data

    from cliffordstf.data import LoaderSet


_logger = logging.getLogger(__name__)


def _maybe_truncate(dataset: object, limit: int | None) -> object:
    """Return ``dataset[:limit]`` when ``limit`` is a positive int; else return ``dataset``."""
    if not isinstance(limit, int) or limit <= 0:
        return dataset
    if len(dataset) > limit:  # type: ignore[arg-type]
        return dataset[:limit]  # type: ignore[index]
    return dataset


def build_molecule3d_loaders(cfg: DictConfig) -> list[LoaderSet]:
    """Build Molecule3D train/val/test LoaderSet (single fold).

    Reads from ``cfg.dataset``:
        ``data_root``: directory containing the Molecule3D raw + processed
            files.
        ``split_mode``: ``"random"`` (default) or ``"scaffold"``.
        ``target``: property index passed to ``Molecule3DProps``
            (default ``0``).
        ``target_name``: human-readable name carried in
            ``runtime_stats`` (default ``"homo_lumo_gap"``).
        ``max_base_mols``: optional cap on the base-test preprocessing
            output.
        ``debug_subset``: optional ``int`` truncating each split to the
            first ``N`` samples.
        ``max_train_samples`` / ``max_val_samples``: optional caps on
            the produced loaders.

    Returns a single-element list with ``runtime_stats =
    {"mean", "std", "target_name"}`` computed from the train split.
    """
    from cliffordstf.data import LoaderSet  # local import: break circular dep

    ds_cfg = cfg.dataset
    split_mode = ds_cfg.get("split_mode", "random")
    target_id = int(ds_cfg.get("target", 0))
    target_name = str(ds_cfg.get("target_name", "homo_lumo_gap"))
    max_base_mols = ds_cfg.get("max_base_mols", None)
    debug_subset = ds_cfg.get("debug_subset", None)
    max_train = ds_cfg.get("max_train_samples", None)
    max_val = ds_cfg.get("max_val_samples", None)

    train_dataset: object = Molecule3DProps(
        root=ds_cfg.data_root,
        split="train",
        split_mode=split_mode,
        target_id=target_id,
        max_base_mols=max_base_mols,
    )
    val_dataset: object = Molecule3DProps(
        root=ds_cfg.data_root,
        split="val",
        split_mode=split_mode,
        target_id=target_id,
        max_base_mols=max_base_mols,
    )
    test_dataset: object = Molecule3DProps(
        root=ds_cfg.data_root,
        split="test",
        split_mode=split_mode,
        target_id=target_id,
        max_base_mols=max_base_mols,
    )

    if isinstance(debug_subset, int) and debug_subset > 0:
        train_dataset = _maybe_truncate(train_dataset, debug_subset)
        val_dataset = _maybe_truncate(val_dataset, min(debug_subset, len(val_dataset)))  # type: ignore[arg-type]
        test_dataset = _maybe_truncate(test_dataset, min(debug_subset, len(test_dataset)))  # type: ignore[arg-type]

    train_dataset = _maybe_truncate(train_dataset, max_train)
    val_dataset = _maybe_truncate(val_dataset, max_val)

    train_loader = build_loader(cast("Dataset[Data]", train_dataset), cfg, shuffle=True)
    val_loader = build_loader(cast("Dataset[Data]", val_dataset), cfg, shuffle=False)
    test_loader = build_loader(cast("Dataset[Data]", test_dataset), cfg, shuffle=False)

    y_train_chunks: list[torch.Tensor] = []
    for batch in train_loader:
        y_train_chunks.append(batch.y.view(-1))
    y_train = torch.cat(y_train_chunks, dim=0).float()
    mean = y_train.mean()
    std = y_train.std().clamp_min(1e-12)

    _logger.info(
        "Molecule3D target=%s mean=%.6f std=%.6f (n_train=%d)",
        target_name,
        float(mean.item()),
        float(std.item()),
        len(y_train),
    )

    return [
        LoaderSet(
            train=train_loader,
            val=val_loader,
            test=test_loader,
            extra_parts=(),
            runtime_stats=MappingProxyType(
                {
                    "mean": mean.detach().clone(),
                    "std": std.detach().clone(),
                    "target_name": target_name,
                }
            ),
            fold=1,
            k_folds=1,
        )
    ]


__all__ = ["build_molecule3d_loaders"]
