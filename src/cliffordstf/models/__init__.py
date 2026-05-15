"""Built-in model factories and a pure ``build_model`` resolver.

No global registry, no decorators, no import-time side effects beyond the
explicit factory imports below (``CODING_RULES.md`` §C.4 / §L).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from cliffordstf.domain import ModelFactory

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch import nn


AVAILABLE_MODELS: Mapping[str, ModelFactory] = MappingProxyType({})
"""Built-in cliffordstf models. Populated as variants are ported (Step 4)."""


def build_model(
    name: str,
    cfg: DictConfig,
    *,
    extras: Mapping[str, ModelFactory] | None = None,
) -> nn.Module:
    """Build a model by name, optionally extending with plug-in factories.

    Args:
        name: Name of the model variant (e.g. ``"clifford_stf_full"``).
        cfg: Resolved Hydra config; passed straight through to the factory.
        extras: Optional supplementary catalog merged on top of
            ``AVAILABLE_MODELS``. The CLI uses this to plug in
            ``baselines.AVAILABLE_MODELS`` when the optional baselines
            package is installed.

    Returns:
        The constructed ``nn.Module``.

    Raises:
        KeyError: If ``name`` is in neither catalog.
    """
    catalog: dict[str, ModelFactory] = {**AVAILABLE_MODELS, **(extras or {})}
    if name not in catalog:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(catalog)}")
    return catalog[name](cfg)


__all__ = ["AVAILABLE_MODELS", "build_model"]
