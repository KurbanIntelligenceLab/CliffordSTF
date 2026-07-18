"""Shared domain types used by ``cliffordstf`` modules and external plug-ins.

Anything imported by two or more modules lives here per
``CODING_RULES.md`` §C.4.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch import nn

    from cliffordstf.data import LoaderSet


ModelFactory = Callable[["DictConfig"], "nn.Module"]
"""A callable that turns a resolved Hydra config into an ``nn.Module``."""

DatasetFactory = Callable[["DictConfig"], "list[LoaderSet]"]
"""A callable that turns a resolved config into a list of ``LoaderSet``.

A factory returns one ``LoaderSet`` per training fold (single-element
list for non-cross-validated datasets, multi-element for k-fold).
"""


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Filesystem locations for a single training run."""

    output_dir: Path
    checkpoint_dir: Path
    logs_dir: Path


class ProvenanceSink(Protocol):
    """Anything that accepts a ``meta.json`` payload for an artifact."""

    def write(self, artifact: Path, payload: dict[str, object]) -> None: ...
