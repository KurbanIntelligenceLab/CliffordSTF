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


AVAILABLE_DATASETS: Mapping[str, DatasetFactory] = MappingProxyType({"md17": build_md17_loaders})
"""Built-in cliffordstf datasets. Extended in Step 6b with QM9, OC20, OC22, Molecule3D."""


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
