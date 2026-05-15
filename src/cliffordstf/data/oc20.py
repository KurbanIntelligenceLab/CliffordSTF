"""OC20 dataset factories: S2EF, IS2RE, and three Tier-3 adsorbate subsets."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from cliffordstf.data._oc20_io import OC20LMDBDataset
from cliffordstf.data._oc20_subsets import (
    C2_ADSORBATES,
    CO2RR_ADSORBATES,
    NRR_ADSORBATES,
    build_oc20_adsorbate_subset,
)
from cliffordstf.data.loader import build_loader

if TYPE_CHECKING:
    from collections.abc import Callable

    from omegaconf import DictConfig
    from torch.utils.data import Dataset
    from torch_geometric.data import Data

    from cliffordstf.data import LoaderSet


def _val_with_subsample(
    dataset: object,
    val_frac: float | None,
    max_val: int | None,
) -> None:
    """Apply ``val_subsample_frac`` to either an LMDB dataset or a Subset."""
    if val_frac is None or max_val is not None:
        return
    if isinstance(dataset, OC20LMDBDataset):
        dataset._total_len = max(1, int(val_frac * len(dataset)))
        return
    existing = getattr(dataset, "indices", None)
    if existing is not None:
        new_len = max(1, int(val_frac * len(existing)))
        dataset.indices = existing[:new_len]  # type: ignore[attr-defined]


def _build_with_dataset_factory(
    cfg: DictConfig,
    make_dataset: Callable[[str, int | None], Dataset[Data]],
) -> list[LoaderSet]:
    """Common scaffolding: build train/val/(test) loaders from a dataset factory."""
    from cliffordstf.data import LoaderSet  # local import: break circular dep

    ds_cfg = cfg.dataset
    max_train = ds_cfg.get("max_train_samples", None)
    max_val = ds_cfg.get("max_val_samples", None)
    split_test = ds_cfg.get("split_test", None)
    val_frac = ds_cfg.get("val_subsample_frac", None)

    train_dataset = make_dataset(ds_cfg.get("split_train", "train"), max_train)
    val_dataset = make_dataset(ds_cfg.get("split_val", "val_id"), max_val)
    _val_with_subsample(val_dataset, val_frac, max_val)

    train_loader = build_loader(train_dataset, cfg, shuffle=True)
    val_loader = build_loader(val_dataset, cfg, shuffle=False)

    test_loader = None
    if split_test:
        test_dataset = make_dataset(split_test, ds_cfg.get("max_test_samples", None))
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


def _oc20_lmdb_factory(cfg: DictConfig, *, task: str) -> list[LoaderSet]:
    ds_cfg = cfg.dataset

    def make_dataset(split: str, max_samples: int | None) -> OC20LMDBDataset:
        return OC20LMDBDataset(
            root=ds_cfg.data_root, task=task, split=split, max_samples=max_samples
        )

    return _build_with_dataset_factory(cfg, make_dataset)


def _oc20_subset_factory(
    cfg: DictConfig,
    *,
    adsorbates: frozenset[str],
    cache_subdir: str,
) -> list[LoaderSet]:
    ds_cfg = cfg.dataset

    def make_dataset(split: str, max_samples: int | None) -> Dataset[Data]:
        return build_oc20_adsorbate_subset(
            root=ds_cfg.data_root,
            split=split,
            adsorbates=adsorbates,
            cache_subdir=cache_subdir,
            max_samples=max_samples,
        )

    return _build_with_dataset_factory(cfg, make_dataset)


def build_oc20_s2ef_loaders(cfg: DictConfig) -> list[LoaderSet]:
    """OC20 S2EF (energy + forces) LoaderSet."""
    task = cfg.dataset.get("task", cfg.dataset.get("task_type", "s2ef"))
    return _oc20_lmdb_factory(cfg, task=task if task in ("s2ef", "is2re") else "s2ef")


def build_oc20_is2re_loaders(cfg: DictConfig) -> list[LoaderSet]:
    """OC20 IS2RE (scalar relaxed-energy regression) LoaderSet."""
    return _oc20_lmdb_factory(cfg, task="is2re")


def build_oc20_s2ef_co2rr_loaders(cfg: DictConfig) -> list[LoaderSet]:
    """OC20 S2EF filtered to CO2 reduction adsorbates."""
    return _oc20_subset_factory(cfg, adsorbates=CO2RR_ADSORBATES, cache_subdir="co2rr")


def build_oc20_s2ef_nrr_loaders(cfg: DictConfig) -> list[LoaderSet]:
    """OC20 S2EF filtered to nitrogen reduction adsorbates."""
    return _oc20_subset_factory(cfg, adsorbates=NRR_ADSORBATES, cache_subdir="nrr")


def build_oc20_s2ef_c2_loaders(cfg: DictConfig) -> list[LoaderSet]:
    """OC20 S2EF filtered to C-C coupling adsorbates."""
    return _oc20_subset_factory(cfg, adsorbates=C2_ADSORBATES, cache_subdir="c2")


__all__ = [
    "build_oc20_is2re_loaders",
    "build_oc20_s2ef_c2_loaders",
    "build_oc20_s2ef_co2rr_loaders",
    "build_oc20_s2ef_loaders",
    "build_oc20_s2ef_nrr_loaders",
]
