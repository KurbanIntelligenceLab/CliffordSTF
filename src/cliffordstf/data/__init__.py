"""Built-in dataset factories and the ``build_dataloaders`` resolver."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from cliffordstf.data.md17 import build_md17_loaders
from cliffordstf.data.oc20 import (
    build_oc20_is2re_loaders,
    build_oc20_s2ef_loaders,
)
from cliffordstf.data.oc22 import build_oc22_is2re_loaders, build_oc22_s2ef_loaders
from cliffordstf.domain import DatasetFactory

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch_geometric.loader import DataLoader as PyGDataLoader


@dataclass(frozen=True, slots=True)
class LoaderSet:
    """One training fold of built loaders plus optional dataset metadata."""

    train: PyGDataLoader
    val: PyGDataLoader
    test: PyGDataLoader | None = None
    extra_parts: tuple[str, ...] = ()
    runtime_stats: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    fold: int = 0
    k_folds: int = 1


AVAILABLE_DATASETS: Mapping[str, DatasetFactory] = MappingProxyType(
    {
        "md17": build_md17_loaders,
        "oc20_s2ef": build_oc20_s2ef_loaders,
        "oc20_is2re": build_oc20_is2re_loaders,
        "oc22_s2ef": build_oc22_s2ef_loaders,
        "oc22_is2re": build_oc22_is2re_loaders,
    }
)
"""Dataset tasks used in the manuscript: rMD17 and OC20/OC22 S2EF and IS2RE."""


def build_dataloaders(
    cfg: DictConfig,
    *,
    extras: Mapping[str, DatasetFactory] | None = None,
) -> list[LoaderSet]:
    """Build LoaderSets for ``cfg.dataset.name``; mirrors ``build_model``."""
    name = cfg.dataset.name
    catalog: dict[str, DatasetFactory] = {**AVAILABLE_DATASETS, **(extras or {})}
    if name not in catalog:
        raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(catalog)}")
    return catalog[name](cfg)


__all__ = ["AVAILABLE_DATASETS", "LoaderSet", "build_dataloaders"]
