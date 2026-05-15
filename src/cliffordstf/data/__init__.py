"""Built-in dataset factories and a pure ``build_dataset`` resolver.

Mirrors ``cliffordstf.models``: no global registry, no decorators.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from cliffordstf.domain import DatasetFactory

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch.utils.data import Dataset


AVAILABLE_DATASETS: Mapping[str, DatasetFactory] = MappingProxyType({})
"""Built-in cliffordstf datasets. Populated in Step 6."""


def build_dataset(
    name: str,
    cfg: DictConfig,
    *,
    extras: Mapping[str, DatasetFactory] | None = None,
) -> Dataset[Any]:
    """Build a dataset by name; mirrors ``build_model``."""
    catalog: dict[str, DatasetFactory] = {**AVAILABLE_DATASETS, **(extras or {})}
    if name not in catalog:
        raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(catalog)}")
    return catalog[name](cfg)


__all__ = ["AVAILABLE_DATASETS", "build_dataset"]
