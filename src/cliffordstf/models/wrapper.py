"""CliffordSTF Clifford wrapper - ablation-ready ``nn.Module``.

Backward-compatible: ``ABLATION_CONFIGS["baseline"]`` reproduces the original
``CliffordWrapper`` behaviour. All twelve named ablations from the paper are
preserved as :data:`ABLATION_CONFIGS` for documentation and reproduction.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph

from cliffordstf.models.interaction import CliffordSTF
from cliffordstf.training.ema import ExponentialMovingAverage

_logger = logging.getLogger(__name__)


ABLATION_CONFIGS: dict[str, dict[str, Any]] = {
    "baseline": {
        "stf_mode": "none",
        "use_hodge_forces": False,
        "use_adaptive_routing": False,
        "routing_mode": "none",
        "use_cross_track": False,
    },
    "hodge_only": {
        "stf_mode": "none",
        "use_hodge_forces": True,
        "use_adaptive_routing": False,
        "routing_mode": "none",
        "use_cross_track": False,
    },
    "stf2_no_hodge": {
        "stf_mode": "stf2",
        "use_hodge_forces": False,
        "use_adaptive_routing": False,
        "routing_mode": "none",
        "use_cross_track": True,
    },
    "stf2": {
        "stf_mode": "stf2",
        "use_hodge_forces": True,
        "use_adaptive_routing": False,
        "routing_mode": "none",
        "use_cross_track": True,
    },
    "stf2_no_cross": {
        "stf_mode": "stf2",
        "use_hodge_forces": True,
        "use_adaptive_routing": False,
        "routing_mode": "none",
        "use_cross_track": False,
    },
    "stf2_stf3": {
        "stf_mode": "stf2+stf3",
        "use_hodge_forces": True,
        "use_adaptive_routing": False,
        "routing_mode": "none",
        "use_cross_track": True,
    },
    "stf2_static_routing": {
        "stf_mode": "stf2",
        "use_hodge_forces": True,
        "use_adaptive_routing": True,
        "routing_mode": "static",
        "use_cross_track": True,
    },
    "stf2_learned_routing": {
        "stf_mode": "stf2",
        "use_hodge_forces": True,
        "use_adaptive_routing": True,
        "routing_mode": "learned",
        "use_cross_track": True,
    },
    "full": {
        "stf_mode": "stf2+stf3",
        "use_hodge_forces": True,
        "use_adaptive_routing": True,
        "routing_mode": "learned",
        "use_cross_track": True,
    },
    "full_no_routing": {
        "stf_mode": "stf2+stf3",
        "use_hodge_forces": True,
        "use_adaptive_routing": False,
        "routing_mode": "none",
        "use_cross_track": True,
    },
    "full_no_cross": {
        "stf_mode": "stf2+stf3",
        "use_hodge_forces": True,
        "use_adaptive_routing": False,
        "routing_mode": "none",
        "use_cross_track": False,
    },
    "stf_only_output": {
        "stf_mode": "stf2+stf3",
        "use_hodge_forces": False,
        "use_adaptive_routing": False,
        "routing_mode": "none",
        "use_cross_track": True,
        "skip_clifford_output": True,
    },
}


class CliffordSTFWrapper(nn.Module):
    """CliffordSTF Clifford encoder with the ``data_wrapper`` interface.

    Drop-in replacement for the original ``CliffordWrapper`` with augmented
    features (STF tracks, Hodge forces, adaptive routing).

    Example:
        model = CliffordSTFWrapper(
            n_channels=128,
            n_interactions=5,
            stf_mode="stf2",
            use_hodge_forces=True,
        )
        energy, forces = model(data)
    """

    def __init__(
        self,
        n_atom_types: int = 100,
        n_channels: int = 128,
        n_interactions: int = 5,
        n_rbf: int = 20,
        cutoff: float = 5.0,
        n_hidden_output: int = 64,
        max_neighbors: int = 50,
        direct_forces: bool = True,
        use_attention: bool = True,
        use_self_interaction: bool = True,
        max_body_order: int = 3,
        use_l2: bool = True,
        use_multiscale: bool = True,
        use_gp_readout: bool = True,
        n_heads: int = 4,
        use_dens: bool = False,
        dens_noise_std: float = 0.01,
        stf_mode: str = "stf2",
        use_hodge_forces: bool = True,
        use_adaptive_routing: bool = False,
        routing_mode: str = "none",
        use_cross_track: bool = True,
        skip_clifford_output: bool = False,
        use_compile: bool = False,
        compile_mode: str = "reduce-overhead",
        use_ema: bool = True,
        ema_decay: float = 0.999,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.use_ema = use_ema
        self.use_dens = use_dens

        self._model: nn.Module
        self._model = CliffordSTF(
            n_atom_types=n_atom_types,
            n_channels=n_channels,
            n_interactions=n_interactions,
            n_rbf=n_rbf,
            cutoff=cutoff,
            n_hidden_output=n_hidden_output,
            max_neighbors=max_neighbors,
            direct_forces=direct_forces,
            use_attention=use_attention,
            use_self_interaction=use_self_interaction,
            max_body_order=max_body_order,
            use_l2=use_l2,
            use_multiscale=use_multiscale,
            use_gp_readout=use_gp_readout,
            n_heads=n_heads,
            use_dens=use_dens,
            dens_noise_std=dens_noise_std,
            stf_mode=stf_mode,
            use_hodge_forces=use_hodge_forces,
            use_adaptive_routing=use_adaptive_routing,
            routing_mode=routing_mode,
            use_cross_track=use_cross_track,
            skip_clifford_output=skip_clifford_output,
        )

        if use_compile:
            try:
                self._model = cast(
                    nn.Module,
                    torch.compile(self._model, mode=compile_mode, dynamic=True),
                )
                _logger.info("torch.compile enabled (mode=%s)", compile_mode)
            except Exception as exc:
                _logger.warning("torch.compile failed: %s", exc)

        self._ema: ExponentialMovingAverage | None = None
        self._ema_decay = ema_decay

    def init_ema(self) -> None:
        if self.use_ema:
            model = self._get_raw_model()
            self._ema = ExponentialMovingAverage(model, decay=self._ema_decay)

    def _get_raw_model(self) -> CliffordSTF:
        m = self._model
        if hasattr(m, "_orig_mod"):
            return cast(CliffordSTF, m._orig_mod)
        return cast(CliffordSTF, m)

    def update_ema(self) -> None:
        if self._ema is not None:
            self._ema.update()

    def apply_ema(self) -> None:
        if self._ema is not None:
            self._ema.apply_shadow()

    def restore_from_ema(self) -> None:
        if self._ema is not None:
            self._ema.restore()

    def _build_edges(self, data: Data) -> torch.Tensor:
        return cast(
            torch.Tensor,
            radius_graph(
                data.pos,
                r=self.cutoff,
                batch=data.batch,
                max_num_neighbors=self.max_neighbors,
            ),
        )

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        edge_index = self._build_edges(data)
        energy, forces = self._model(data.z, data.pos, edge_index, data.batch)
        if self._get_raw_model().direct_forces:
            return energy.view(-1), forces
        return cast(torch.Tensor, energy.view(-1))

    def forward_with_dens(self, data: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_index = self._build_edges(data)
        energy, forces, dens_loss = self._get_raw_model().forward_with_dens(
            data.z, data.pos, edge_index, data.batch
        )
        return energy.view(-1), forces, dens_loss


def build_from_ablation(
    config_name: str,
    n_channels: int = 128,
    n_interactions: int = 5,
    **overrides: object,
) -> CliffordSTFWrapper:
    """Build a wrapper from a named ablation config.

    Args:
        config_name: Key in :data:`ABLATION_CONFIGS`.
        n_channels: Channel width override (always set).
        n_interactions: Number of interaction layers (always set).
        **overrides: Any other ``CliffordSTFWrapper`` argument.

    Example:
        model = build_from_ablation("stf2", n_channels=64, cutoff=6.0)
    """
    if config_name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(ABLATION_CONFIGS)}")

    cfg = {**ABLATION_CONFIGS[config_name]}
    cfg.update(overrides)
    cfg["n_channels"] = n_channels
    cfg["n_interactions"] = n_interactions

    return CliffordSTFWrapper(**cfg)
