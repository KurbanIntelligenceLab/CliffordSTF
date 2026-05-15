"""Built-in dataset factories and the ``build_dataloaders`` resolver.

Mirrors ``cliffordstf.models``: no global registry, no decorators.
Each factory is a plain function imported here and registered in
``AVAILABLE_DATASETS`` via a frozen ``MappingProxyType``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from cliffordstf.data.md17 import build_md17_loaders
from cliffordstf.data.molecule3d import build_molecule3d_loaders
from cliffordstf.data.oc20 import (
    build_oc20_is2re_loaders,
    build_oc20_s2ef_c2_loaders,
    build_oc20_s2ef_co2rr_loaders,
    build_oc20_s2ef_loaders,
    build_oc20_s2ef_nrr_loaders,
)
from cliffordstf.data.oc22 import build_oc22_is2re_loaders, build_oc22_s2ef_loaders
from cliffordstf.data.qm9 import build_qm9_loaders
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
    runtime_stats: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    fold: int = 0
    k_folds: int = 1


AVAILABLE_DATASETS: Mapping[str, DatasetFactory] = MappingProxyType(
    {
        "md17": build_md17_loaders,
        "qm9": build_qm9_loaders,
        "oc20_s2ef": build_oc20_s2ef_loaders,
        "oc20_is2re": build_oc20_is2re_loaders,
        "oc20_s2ef_co2rr": build_oc20_s2ef_co2rr_loaders,
        "oc20_s2ef_nrr": build_oc20_s2ef_nrr_loaders,
        "oc20_s2ef_c2": build_oc20_s2ef_c2_loaders,
        "oc22_s2ef": build_oc22_s2ef_loaders,
        "oc22_is2re": build_oc22_is2re_loaders,
        "molecule3d": build_molecule3d_loaders,
    }
)
"""Built-in cliffordstf datasets (10 entries; full coverage of source registry)."""


def build_dataloaders(
    cfg: DictConfig,
    *,
    extras: Mapping[str, DatasetFactory] | None = None,
) -> list[LoaderSet]:
    """Build LoaderSets for ``cfg.dataset.name``; mirrors ``build_model``.

    Args:
        cfg: Resolved config. ``cfg.dataset.name`` selects the factory.
        extras: Optional extra factories that override / extend
            ``AVAILABLE_DATASETS`` for one call (used by the
            ``baselines`` package to plug in its own datasets).

    Returns:
        One ``LoaderSet`` per training fold (single-element list for
        non-cross-validated datasets).
    """
    name = cfg.dataset.name
    catalog: dict[str, DatasetFactory] = {**AVAILABLE_DATASETS, **(extras or {})}
    if name not in catalog:
        raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(catalog)}")
    return catalog[name](cfg)


__all__ = ["AVAILABLE_DATASETS", "LoaderSet", "build_dataloaders"]
