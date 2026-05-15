"""Shared domain types used by ``cliffordstf`` modules and external plug-ins.

Anything imported by two or more modules lives here per
``CODING_RULES.md`` §C.4.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch import nn
    from torch.utils.data import Dataset


ModelFactory = Callable[["DictConfig"], "nn.Module"]
"""A callable that turns a resolved Hydra config into an ``nn.Module``."""

DatasetFactory = Callable[["DictConfig"], "Dataset[Any]"]
"""A callable that turns a resolved Hydra config into a torch ``Dataset``.

The element type is left as ``Any`` because data items are heterogeneous
across MLIP datasets (``torch_geometric.data.Data``, dicts, custom Batch
objects) and validation happens inside the model wrapper, not at this
boundary (``CODING_RULES.md`` §C.2 boundary exception).
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
