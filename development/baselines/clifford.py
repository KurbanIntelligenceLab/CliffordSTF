"""Base Clifford baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cliffordstf.models import build_clifford_stf

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from cliffordstf.models.wrapper import CliffordSTFWrapper


def build_clifford(cfg: DictConfig) -> CliffordSTFWrapper:
    """Build the Clifford baseline from a resolved config."""
    return build_clifford_stf(cfg)
