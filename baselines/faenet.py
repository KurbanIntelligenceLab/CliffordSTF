"""FAENet baseline with stochastic 3D frame averaging.

Wraps the upstream ``faenet`` package's ``FAENet`` so the cliffordstf
trainer can drive it via a PyG ``Data`` object. ``faenet`` is imported
lazily inside the constructor; importing this module without the
``cliffordstf[baselines]`` extra installed is therefore safe.

Frame averaging is implemented through the upstream
``faenet.frame_averaging.frame_averaging_3D`` helper. The upstream
``compute_frames`` builds its ``plus_minus`` tensors on CPU even when
the eigenvector matrix is on CUDA, which raises a device-mismatch
error on GPU. We patch ``faenet.frame_averaging.compute_frames``
at module import time with a device-aware re-implementation. The
patch is a no-op on CPU.

The wrapper returns bare per-graph energy; the trainer derives forces
via autograd from ``-dE/dR``.
"""

from __future__ import annotations

import itertools
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch_geometric.data import Data


def _patched_compute_frames(
    eigenvec: torch.Tensor,
    pos: torch.Tensor,
    cell: torch.Tensor | None,
    fa_method: str = "stochastic",
    pos_3d: torch.Tensor | None = None,
    det_index: int = 0,
) -> tuple[list[torch.Tensor], list[torch.Tensor | None], list[torch.Tensor]]:
    """Device-aware re-implementation of ``faenet.frame_averaging.compute_frames``.

    The upstream version constructs ``plus_minus`` permutations on CPU,
    which collides with CUDA ``eigenvec``. We allocate every working
    tensor on ``pos.device`` instead.
    """
    dim = pos.shape[1]
    plus_minus_list: list[torch.Tensor] = [
        torch.tensor(x, device=pos.device, dtype=pos.dtype)
        for x in itertools.product([1, -1], repeat=dim)
    ]
    all_fa_pos: list[torch.Tensor] = []
    all_cell: list[torch.Tensor | None] = []
    all_rots: list[torch.Tensor] = []
    se3 = fa_method in {"se3-all", "se3-stochastic", "se3-det"}
    fa_cell = deepcopy(cell)

    if fa_method in ("det", "se3-det"):
        sum_eigenvec = torch.sum(eigenvec, axis=0)
        plus_minus_list = [torch.where(sum_eigenvec >= 0, 1.0, -1.0)]

    for pm in plus_minus_list:
        new_eigenvec = pm * eigenvec
        fa_pos = pos @ new_eigenvec
        if pos_3d is not None:
            full_eigenvec = torch.eye(3, device=pos.device, dtype=pos.dtype)
            fa_pos = torch.cat((fa_pos, pos_3d.unsqueeze(1)), dim=1)
            full_eigenvec[:2, :2] = new_eigenvec
            new_eigenvec = full_eigenvec
        if cell is not None:
            fa_cell = cell @ new_eigenvec
        if se3 and torch.det(new_eigenvec) < 0:
            continue
        all_fa_pos.append(fa_pos)
        all_cell.append(fa_cell)
        all_rots.append(new_eigenvec)

    if fa_method in ("stochastic", "se3-stochastic"):
        idx = torch.randint(len(all_fa_pos), (1,)).item()
        return [all_fa_pos[idx]], [all_cell[idx]], [all_rots[idx]]
    if fa_method in ("det", "se3-det"):
        i = det_index % len(all_fa_pos)
        return [all_fa_pos[i]], [all_cell[i]], [all_rots[i]]
    return all_fa_pos, all_cell, all_rots


def _install_faenet_device_patch() -> None:
    """Apply the device-aware ``compute_frames`` patch (idempotent)."""
    import faenet.frame_averaging as fa_mod

    if getattr(fa_mod, "_cliffordstf_patched", False):
        return
    fa_mod.compute_frames = _patched_compute_frames  # type: ignore[assignment]
    fa_mod._cliffordstf_patched = True  # type: ignore[attr-defined]


class FAENetDataWrapper(nn.Module):
    """Adapt ``faenet.FAENet`` to ``data_wrapper`` with stochastic 3D FA."""

    def __init__(
        self,
        *,
        cutoff: float,
        max_num_neighbors: int,
        hidden_channels: int,
        num_interactions: int,
        num_gaussians: int,
        num_filters: int,
        act: str,
        skip_co: str,
        mp_type: str,
        second_layer_mlp: bool,
        graph_norm: bool,
        complex_mp: bool,
        phys_embeds: bool,
        phys_hidden_channels: int,
        tag_hidden_channels: int,
        pg_hidden_channels: int,
        frame_averaging: str,
        fa_method: str,
    ) -> None:
        super().__init__()
        from faenet import FAENet
        from faenet.utils import base_preprocess

        _install_faenet_device_patch()

        self.frame_averaging = frame_averaging
        self.fa_method = fa_method
        self._backbone = FAENet(
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            hidden_channels=hidden_channels,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            num_filters=num_filters,
            act=act,
            skip_co=skip_co,
            mp_type=mp_type,
            second_layer_MLP=second_layer_mlp,
            graph_norm=graph_norm,
            complex_mp=complex_mp,
            phys_embeds=phys_embeds,
            phys_hidden_channels=phys_hidden_channels,
            tag_hidden_channels=tag_hidden_channels,
            pg_hidden_channels=pg_hidden_channels,
            preprocess=base_preprocess,
            regress_forces=None,
        )

    def forward(self, data: Data) -> torch.Tensor:
        if not hasattr(data, "atomic_numbers") or data.atomic_numbers is None:
            data.atomic_numbers = data.z
        if not hasattr(data, "natoms") or data.natoms is None:
            data.natoms = torch.unique(data.batch, return_counts=True)[1]

        if self.frame_averaging:
            from faenet.frame_averaging import frame_averaging_3D

            fa_pos_list, _, _ = frame_averaging_3D(data.pos, cell=None, fa_method=self.fa_method)
            original_pos = data.pos
            energies: list[torch.Tensor] = []
            for fa_pos in fa_pos_list:
                data.pos = fa_pos
                preds: dict[str, Any] = self._backbone.energy_forward(data)
                energies.append(cast("torch.Tensor", preds["energy"]))
            data.pos = original_pos
            energy = sum(energies) / len(energies)
        else:
            preds = self._backbone.energy_forward(data)
            energy = cast("torch.Tensor", preds["energy"])
        return energy.view(-1)


def build_faenet(cfg: DictConfig) -> FAENetDataWrapper:
    """Build the FAENet baseline from a resolved config."""
    m = cfg.model
    return FAENetDataWrapper(
        cutoff=m.get("cutoff", 6.0),
        max_num_neighbors=m.get("max_num_neighbors", 40),
        hidden_channels=m.get("hidden_channels", 320),
        num_interactions=m.get("num_interactions", 4),
        num_gaussians=m.get("num_gaussians", 50),
        num_filters=m.get("num_filters", 128),
        act=m.get("act", "swish"),
        skip_co=m.get("skip_co", "concat"),
        mp_type=m.get("mp_type", "updownscale_base"),
        second_layer_mlp=m.get("second_layer_MLP", True),
        graph_norm=m.get("graph_norm", True),
        complex_mp=m.get("complex_mp", False),
        phys_embeds=m.get("phys_embeds", True),
        phys_hidden_channels=m.get("phys_hidden_channels", 0),
        tag_hidden_channels=m.get("tag_hidden_channels", 0),
        pg_hidden_channels=m.get("pg_hidden_channels", 0),
        frame_averaging=m.get("frame_averaging", "3D"),
        fa_method=m.get("fa_method", "stochastic"),
    )
