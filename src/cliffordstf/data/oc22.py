"""OC22 dataset factories: oxide electrocatalysis (S2EF-Total + IS2RE-Total)."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from cliffordstf.data._oc20_io import OC20LMDBDataset
from cliffordstf.data.loader import build_loader

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from cliffordstf.data import LoaderSet


def _build_oc22(cfg: DictConfig, *, task: str) -> list[LoaderSet]:
    from cliffordstf.data import LoaderSet  # local import: break circular dep

    ds_cfg = cfg.dataset
    max_train = ds_cfg.get("max_train_samples", None)
    max_val = ds_cfg.get("max_val_samples", None)
    val_frac = ds_cfg.get("val_subsample_frac", None)
    split_test = ds_cfg.get("split_test", None)

    train_dataset = OC20LMDBDataset(
        root=ds_cfg.data_root,
        task=task,
        split=ds_cfg.get("split_train", "train"),
        max_samples=max_train,
        oc22=True,
    )
    val_dataset = OC20LMDBDataset(
        root=ds_cfg.data_root,
        task=task,
        split=ds_cfg.get("split_val", "val_id"),
        max_samples=max_val,
        oc22=True,
    )
    if val_frac is not None and max_val is None:
        val_dataset._total_len = max(1, int(val_frac * len(val_dataset)))

    train_loader = build_loader(train_dataset, cfg, shuffle=True)
    val_loader = build_loader(val_dataset, cfg, shuffle=False)
    test_loader = None
    if split_test:
        test_dataset = OC20LMDBDataset(
            root=ds_cfg.data_root,
            task=task,
            split=split_test,
            max_samples=ds_cfg.get("max_test_samples", None),
            oc22=True,
        )
        test_loader = build_loader(test_dataset, cfg, shuffle=False)

    return [
        LoaderSet(
            train=train_loader,
            val=val_loader,
            test=test_loader,
            extra_parts=(),
            runtime_stats=MappingProxyType({}),
            fold=1,
            k_folds=1,
        )
    ]


def build_oc22_s2ef_loaders(cfg: DictConfig) -> list[LoaderSet]:
    """OC22 S2EF-Total (energy + forces, oxide electrocatalysis)."""
    return _build_oc22(cfg, task="s2ef")


def build_oc22_is2re_loaders(cfg: DictConfig) -> list[LoaderSet]:
    """OC22 IS2RE-Total (scalar relaxed-energy regression)."""
    return _build_oc22(cfg, task="is2re")


__all__ = ["build_oc22_is2re_loaders", "build_oc22_s2ef_loaders"]
