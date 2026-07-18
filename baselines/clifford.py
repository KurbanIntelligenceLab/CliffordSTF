"""Base Clifford baseline.

The model is the in-tree :class:`~cliffordstf.models.wrapper.CliffordSTFWrapper`
with all STF features deactivated via
``baselines/configs/model/clifford.yaml`` (matches
``ABLATION_CONFIGS["baseline"]`` in the wrapper). A standalone factory
is kept here only so the plug-in seam from
``cliffordstf.models.build_model(extras=...)`` resolves ``clifford`` to
this baseline rather than to a cliffordstf variant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cliffordstf.models import build_clifford_stf

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from cliffordstf.models.wrapper import CliffordSTFWrapper


def build_clifford(cfg: DictConfig) -> CliffordSTFWrapper:
    """Build the Clifford baseline from a resolved config."""
    return build_clifford_stf(cfg)
