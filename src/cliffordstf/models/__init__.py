"""Built-in model factories and a pure ``build_model`` resolver.

No global registry, no decorators, no import-time side effects beyond the
explicit factory imports below (``CODING_RULES.md`` §C.4 / §L).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from cliffordstf.domain import ModelFactory
from cliffordstf.models.wrapper import CliffordSTFWrapper

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch import nn


def _build(cfg: DictConfig) -> CliffordSTFWrapper:
    """Construct a ``CliffordSTFWrapper`` from a resolved config.

    Reads ``cfg.model.*`` for the three named variants ``clifford_stf`` /
    ``clifford_stf_full`` / ``clifford_stf_full_10m``. The variants differ only
    in YAML hyperparameters; the factory body is shared.
    """
    m = cfg.model
    return CliffordSTFWrapper(
        n_atom_types=m.get("n_atom_types", 100),
        n_channels=m.n_channels,
        n_interactions=m.n_interactions,
        n_rbf=m.get("n_rbf", 20),
        cutoff=m.cutoff,
        n_hidden_output=m.get("n_hidden_output", 64),
        max_neighbors=m.get("max_neighbors", 50),
        direct_forces=m.get("direct_forces", True),
        use_attention=m.get("use_attention", True),
        use_self_interaction=m.get("use_self_interaction", False),
        max_body_order=m.get("max_body_order", 3),
        use_l2=m.get("use_l2", True),
        use_multiscale=m.get("use_multiscale", True),
        use_gp_readout=m.get("use_gp_readout", False),
        n_heads=m.get("n_heads", 4),
        stf_mode=m.get("stf_mode", "stf2"),
        use_hodge_forces=m.get("use_hodge_forces", True),
        use_adaptive_routing=m.get("use_adaptive_routing", False),
        routing_mode=m.get("routing_mode", "none"),
        use_cross_track=m.get("use_cross_track", True),
        skip_clifford_output=m.get("skip_clifford_output", False),
        use_compile=m.get("use_compile", False),
        compile_mode=m.get("compile_mode", "reduce-overhead"),
    )


def build_clifford_stf(cfg: DictConfig) -> CliffordSTFWrapper:
    return _build(cfg)


def build_clifford_stf_full(cfg: DictConfig) -> CliffordSTFWrapper:
    return _build(cfg)


def build_clifford_stf_full_10m(cfg: DictConfig) -> CliffordSTFWrapper:
    return _build(cfg)


AVAILABLE_MODELS: Mapping[str, ModelFactory] = MappingProxyType(
    {
        "clifford_stf": build_clifford_stf,
        "clifford_stf_full": build_clifford_stf_full,
        "clifford_stf_full_10m": build_clifford_stf_full_10m,
    }
)
"""Built-in cliffordstf models. Variants differ only by config YAML."""


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


__all__ = [
    "AVAILABLE_MODELS",
    "build_clifford_stf",
    "build_clifford_stf_full",
    "build_clifford_stf_full_10m",
    "build_model",
]
