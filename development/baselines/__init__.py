"""Optional plug-in baselines for cliffordstf."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from baselines.clifford import build_clifford
from baselines.dimenetpp import build_dimenetpp
from baselines.faenet import build_faenet
from baselines.gotennet import build_gotennet
from baselines.ictp import build_ictp_l1, build_ictp_l2, build_ictp_l3
from baselines.mace import build_mace_l1, build_mace_l2, build_mace_l3
from baselines.nequip import build_nequip
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
        "faenet": build_faenet,
        "gotennet": build_gotennet,
        "ictp_l1": build_ictp_l1,
        "ictp_l2": build_ictp_l2,
        "ictp_l3": build_ictp_l3,
        "mace_l1": build_mace_l1,
        "mace_l2": build_mace_l2,
        "mace_l3": build_mace_l3,
        "nequip": build_nequip,
        "painn": build_painn,
        "schnet": build_schnet,
        "torchmdnet": build_torchmdnet,
        "visnet": build_visnet,
    }
)

__all__ = ["AVAILABLE_MODELS", "CONFIGS_DIR"]
