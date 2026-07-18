"""Clifford message-passing building blocks consumed by clifford_stf.

Aggressively trimmed: only the radial basis + cutoff, the equivariant
dot-product attention, the multi-scale per-layer energy readout, and the
``scatter_softmax`` eager helper survive. All other classes from the original
``models/clifford/interaction.py`` (full message-passing layers, the public
``CliffordNet`` base, and the module's smoke test) live in the baseline
package, not in this private subtree.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn
from torch_scatter import scatter, scatter_softmax

from cliffordstf.models._algebra.geometric_algebra import GeometricAlgebra


# scatter_softmax is a torch_scatter C++ custom op that Dynamo cannot trace
# through (fails at fake-tensor propagation). Wrap it in an eager helper so
# torch.compile graph-breaks here and keeps going for the rest of the model.
def _scatter_softmax_eager_raw(logits: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, scatter_softmax(logits, dst, dim=0))


_scatter_softmax_eager: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cast(
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    torch.compiler.disable(_scatter_softmax_eager_raw),  # type: ignore[no-untyped-call]
)


# ============================================================
# Radial basis + cutoff
# ============================================================


class RadialBasisFunctions(nn.Module):
    """Gaussian RBF for distance encoding."""

    offsets: torch.Tensor
    widths: torch.Tensor

    def __init__(self, n_rbf: int = 20, cutoff: float = 5.0, trainable: bool = False) -> None:
        super().__init__()
        self.n_rbf = n_rbf
        offsets = torch.linspace(0.0, cutoff, n_rbf)
        widths = torch.full((n_rbf,), (offsets[1] - offsets[0]).item())
        if trainable:
            self.offsets = nn.Parameter(offsets)
            self.widths = nn.Parameter(widths)
        else:
            self.register_buffer("offsets", offsets)
            self.register_buffer("widths", widths)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        if dist.dim() == 1:
            dist = dist.unsqueeze(-1)
        return torch.exp(-0.5 * ((dist - self.offsets) / self.widths) ** 2)


class CosineCutoff(nn.Module):
    """Smooth cosine cutoff envelope."""

    def __init__(self, cutoff: float = 5.0) -> None:
        super().__init__()
        self.cutoff = cutoff

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.cos(dist * torch.pi / self.cutoff) + 1.0) * (dist < self.cutoff).float()


# ============================================================
# Equivariant dot-product attention
# ============================================================


class CliffordAttention(nn.Module):
    """Equivariant dot-product attention over neighbors.

    Query/Key from grade-0 (invariant) for attention weights.
    Value is the full multivector message.
    Compatible with ``scatter_softmax`` for variable neighbor counts.
    """

    def __init__(self, n_channels: int, n_heads: int = 4, n_rbf: int = 20) -> None:
        super().__init__()
        if n_channels % n_heads != 0:
            raise ValueError(f"n_channels={n_channels} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = n_channels // n_heads

        self.q_proj = nn.Linear(n_channels, n_channels, bias=False)
        self.k_proj = nn.Linear(n_channels, n_channels, bias=False)
        self.rbf_proj = nn.Linear(n_rbf, n_channels, bias=False)
        self.rbf = RadialBasisFunctions(n_rbf)

        self.scale = self.head_dim**-0.5

    def forward(
        self,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        dist: torch.Tensor,
        dst: torch.Tensor,
        n_nodes: int,
        rbf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns ``(E, 1, 1)`` attention weights for each edge.

        Args:
            h_i: ``(E, C)`` receiver grade-0 features.
            h_j: ``(E, C)`` sender grade-0 features.
            dist: ``(E,)`` edge distances.
            dst: ``(E,)`` destination indices.
            n_nodes: Number of nodes (unused; preserved for call-site stability).
            rbf: ``(E, n_rbf)`` pre-computed RBF; recomputed from ``dist`` if None.
        """
        del n_nodes  # unused; kept in signature for call-site stability
        q = self.q_proj(h_i)
        k = self.k_proj(h_j)
        rbf_w = self.rbf_proj(rbf if rbf is not None else self.rbf(dist))

        n_edges, _ = q.shape
        q = q.view(n_edges, self.n_heads, self.head_dim)
        k = k.view(n_edges, self.n_heads, self.head_dim)
        rbf_w = rbf_w.view(n_edges, self.n_heads, self.head_dim)

        attn_logits = (q * k * rbf_w).sum(-1) * self.scale

        attn = _scatter_softmax_eager(attn_logits, dst)
        attn = attn.mean(dim=-1)

        return attn.unsqueeze(-1).unsqueeze(-1)


# ============================================================
# Multi-scale per-layer energy readout
# ============================================================


class PerLayerEnergyReadout(nn.Module):
    """Lightweight per-layer energy head for multi-scale readout.

    Reads the grade-0 (scalar) slice of the multivector and projects to a
    per-atom scalar, optionally aggregated across a batch.
    """

    def __init__(
        self,
        n_channels: int,
        n_hidden: int = 64,
        algebra: GeometricAlgebra | None = None,
    ) -> None:
        super().__init__()
        if algebra is None:
            from cliffordstf.models._algebra.clifford import _get_default_algebra

            algebra = _get_default_algebra()
        self.algebra = algebra
        self.head = nn.Sequential(
            nn.Linear(n_channels, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        batch: torch.Tensor | None = None,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        s0, e0 = self.algebra.grade_ranges[0]
        atom_e = self.head(h[..., s0:e0].squeeze(-1)).squeeze(-1)
        if batch is not None:
            # Callers in the hot path should pass num_graphs to avoid the
            # GPU->CPU sync from batch.max().item().
            if num_graphs is None:
                num_graphs = int(batch.max().item()) + 1
            return cast(
                torch.Tensor,
                scatter(atom_e, batch, dim=0, dim_size=num_graphs, reduce="sum"),
            )
        return cast(torch.Tensor, atom_e.sum(dim=0, keepdim=True))
