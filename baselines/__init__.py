"""Optional plug-in baselines for cliffordstf.

Exposes :data:`AVAILABLE_MODELS` and :data:`CONFIGS_DIR` so
``cliffordstf.cli`` can merge third-party model factories into
:func:`cliffordstf.models.build_model` via its ``extras`` argument and
extend :func:`cliffordstf.io.config.load_config`'s
``extra_search_paths`` with the baselines' packaged YAMLs.

Heavy dependencies are gated behind ``pip install cliffordstf[baselines]``
and populated incrementally (one baseline per commit).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from baselines.clifford import build_clifford
from baselines.dimenetpp import build_dimenetpp
from baselines.painn import build_painn
from baselines.schnet import build_schnet
from baselines.torchmdnet import build_torchmdnet
from baselines.visnet import build_visnet
from cliffordstf.domain import ModelFactory

CONFIGS_DIR: Path = Path(__file__).resolve().parent / "configs"
"""Packaged ``baselines/configs/`` directory."""

AVAILABLE_MODELS: Mapping[str, ModelFactory] = MappingProxyType(
    {
        "clifford": build_clifford,
        "dimenetpp": build_dimenetpp,
        "painn": build_painn,
        "schnet": build_schnet,
        "torchmdnet": build_torchmdnet,
        "visnet": build_visnet,
    }
)

__all__ = ["AVAILABLE_MODELS", "CONFIGS_DIR"]
