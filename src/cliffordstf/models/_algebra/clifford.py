"""Clifford algebra primitives for equivariant neural networks.

Signature-parameterized via `GeometricAlgebra`.

Behaviour:
  - `CliffordAlgebra(GeometricAlgebra((3,0)))` keeps the Cl(3,0) geometric
    product as a single einsum kernel for bit-exact behaviour.
  - Other signatures dispatch to einsum on `algebra.cayley`.
  - `CliffordLinear`, `CliffordNorm`, `CliffordGateActivation` read grade
    ranges, max_grade, and grade_dims from the algebra rather than relying on
    module-level Cl(3,0) constants.

Memory layout (Cl(3,0) default): [s, e1, e2, e3, e12, e13, e23, e123]
        index:                     0   1   2   3    4    5    6     7
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from cliffordstf.models._algebra.geometric_algebra import GeometricAlgebra

# ============================================================
# Backward-compat module-level constants for Cl(3,0) callers
# (clifford_stf and the equivariance test helper below).
# New code should derive these from a GeometricAlgebra instance instead.
# ============================================================

DIM = 8
N_GRADES = 4
GRADE_RANGES: dict[int, tuple[int, int]] = {0: (0, 1), 1: (1, 4), 2: (4, 7), 3: (7, 8)}
GRADE_DIMS: dict[int, int] = {0: 1, 1: 3, 2: 3, 3: 1}
ALL_GRADES: tuple[int, ...] = (0, 1, 2, 3)

# Basis-blade indices for Cl(3,0): [s, e1, e2, e3, e12, e13, e23, e123]
S, E1, E2, E3, E12, E13, E23, E123 = range(8)


# Cached default algebra for backward-compat callers that don't pass one.
# Lazily constructed to avoid any import-time ordering issues.
_DEFAULT_ALGEBRA: GeometricAlgebra | None = None


def _get_default_algebra() -> GeometricAlgebra:
    global _DEFAULT_ALGEBRA
    if _DEFAULT_ALGEBRA is None:
        _DEFAULT_ALGEBRA = GeometricAlgebra((3, 0))
    return _DEFAULT_ALGEBRA


# ============================================================
# Grade bookkeeping helpers (signature-independent)
# ============================================================


# Lookup: GP(grade_a, grade_b) -> set of output grades (Cl(3,0) conventions;
# matches general (p,q) signatures because |a-b| <= out <= a+b with step 2,
# capped at p+q).
GP_GRADE_TABLE: dict[tuple[int, int], set[int]] = {
    (0, 0): {0},
    (0, 1): {1},
    (0, 2): {2},
    (0, 3): {3},
    (1, 0): {1},
    (1, 1): {0, 2},
    (1, 2): {1, 3},
    (1, 3): {2},
    (2, 0): {2},
    (2, 1): {1, 3},
    (2, 2): {0, 2},
    (2, 3): {1},
    (3, 0): {3},
    (3, 1): {2},
    (3, 2): {1},
    (3, 3): {0},
}


def compute_gp_output_grades(
    grades_a: tuple[int, ...], grades_b: tuple[int, ...]
) -> tuple[int, ...]:
    """Compute which grades are produced by GP(a, b) given input grades."""
    result: set[int] = set()
    for ga in grades_a:
        for gb in grades_b:
            result |= GP_GRADE_TABLE.get((ga, gb), set())
    return tuple(sorted(result))


def compute_layer_grades(
    n_layers: int,
    edge_grades: tuple[int, ...] = (0, 1),
    max_grade: int = 3,
) -> list[tuple[int, ...]]:
    """Progressive grade activation schedule.

    Args:
        n_layers: Number of interaction layers to schedule.
        edge_grades: Grades carried on edges (defaults to scalar + vector).
        max_grade: Cap on the highest grade to activate. ``1`` = scalar+vector
            only (L=1), ``3`` = full Cl(3,0) (default).

    Returns:
        List of ``n_layers`` grade tuples, one per layer.
    """
    node_grades: tuple[int, ...] = (0,)
    layer_grades: list[tuple[int, ...]] = []
    for _ in range(n_layers):
        gp_grades = compute_gp_output_grades(node_grades, edge_grades)
        node_grades = tuple(sorted(set(node_grades) | set(gp_grades)))
        node_grades = tuple(g for g in node_grades if g <= max_grade)
        layer_grades.append(node_grades)
    return layer_grades


# ============================================================
# Core algebra - wraps GeometricAlgebra, Cl(3,0) fast-path
# ============================================================


class CliffordAlgebra(nn.Module):
    """Clifford algebra wrapper holding a GeometricAlgebra instance.

    For ``signature=(3,0)`` the geometric product is a single einsum kernel.
    All other signatures dispatch to the same einsum path on
    ``algebra.cayley``.
    """

    def __init__(self, algebra: GeometricAlgebra | None = None) -> None:
        super().__init__()
        if algebra is None:
            algebra = _get_default_algebra()
        self.algebra = algebra
        self.dim = algebra.dim
        self._is_cl30 = algebra.signature == (3, 0)

    def geometric_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Full GP: a * b. ``(..., dim) x (..., dim) -> (..., dim)``.

        Single einsum with the algebra's precomputed Cayley tensor.
        """
        return torch.einsum("...i,...j,ijk->...k", a, b, self.algebra.cayley)

    def gp_scalar_times_mv(self, scalar_mv: torch.Tensor, mv: torch.Tensor) -> torch.Tensor:
        s0, e0 = self.algebra.grade_ranges[0]
        return scalar_mv[..., s0:e0] * mv

    def dispatch_gp(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        grades_a: tuple[int, ...],
        grades_b: tuple[int, ...],
    ) -> torch.Tensor:
        if grades_a == (0,):
            return self.gp_scalar_times_mv(a, b)
        if grades_b == (0,):
            return self.gp_scalar_times_mv(b, a)
        return self.geometric_product(a, b)

    def geometric_product_reference(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...i,...j,ijk->...k", a, b, self.algebra.cayley)

    def sandwich_product(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return self.geometric_product(self.geometric_product(r, x), self.reverse(r))

    def reverse(self, mv: torch.Tensor) -> torch.Tensor:
        return mv * self.algebra.reverse_signs

    def grade_select(self, mv: torch.Tensor, grade: int) -> torch.Tensor:
        return self.algebra.slice_grade(mv, grade)

    def grade_decompose(self, mv: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self.algebra.grade_decompose(mv))

    def norm_squared(self, mv: torch.Tensor) -> torch.Tensor:
        prod = self.geometric_product(mv, self.reverse(mv))
        s0, e0 = self.algebra.grade_ranges[0]
        return prod[..., s0:e0]

    def grade_norms(self, mv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        parts = self.algebra.grade_decompose(mv)
        out: list[torch.Tensor] = []
        for k, part in enumerate(parts):
            if self.algebra.grade_dims[k] == 1:
                out.append(torch.abs(part))
            else:
                out.append(torch.sqrt(torch.sum(part**2, dim=-1, keepdim=True) + eps))
        return torch.cat(out, dim=-1)

    def rotor_from_bivector(self, bv: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        return self.algebra.rotor_from_bivector(bv, angle)


# ============================================================
# L=2 feature augmentation (Cl(3,0) shim)
# ============================================================


def compute_l2_features(direction: torch.Tensor) -> torch.Tensor:
    """Cl(3,0) backward-compat shim.

    Returns the 5 L=2 symmetric-traceless components of ``d (outer) d`` for a
    unit direction in 3D. Prefer ``GeometricAlgebra.compute_l2_features`` in
    new code.
    """
    dx, dy, dz = direction.unbind(-1)
    third = 1.0 / 3.0
    return torch.stack(
        [dx * dx - third, dx * dy, dx * dz, dy * dy - third, dy * dz],
        dim=-1,
    )


# ============================================================
# Multivector construction (signature-aware)
# ============================================================


def make_scalar_mv(s: torch.Tensor, algebra: GeometricAlgebra | None = None) -> torch.Tensor:
    """``(*, C)`` scalar -> ``(*, C, algebra.dim)`` with only grade-0 populated."""
    alg = algebra if algebra is not None else _get_default_algebra()
    out = s.new_zeros(*s.shape, alg.dim)
    s0, e0 = alg.grade_ranges[0]
    out[..., s0:e0] = s.unsqueeze(-1)
    return out


def make_vec_mv(v: torch.Tensor, algebra: GeometricAlgebra | None = None) -> torch.Tensor:
    """``(*, C, grade_dims[1])`` vector -> ``(*, C, algebra.dim)`` grade-1 only."""
    alg = algebra if algebra is not None else _get_default_algebra()
    out = v.new_zeros(*v.shape[:-1], alg.dim)
    s1, e1 = alg.grade_ranges[1]
    out[..., s1:e1] = v
    return out


def make_grades01_mv(
    g0: torch.Tensor,
    g1: torch.Tensor,
    algebra: GeometricAlgebra | None = None,
) -> torch.Tensor:
    """``(*, C, 1)`` scalar + ``(*, C, grade_dims[1])`` vector -> ``(*, C, algebra.dim)``."""
    alg = algebra if algebra is not None else _get_default_algebra()
    out = g0.new_zeros(*g0.shape[:-1], alg.dim)
    s0, e0 = alg.grade_ranges[0]
    s1, e1 = alg.grade_ranges[1]
    out[..., s0:e0] = g0
    out[..., s1:e1] = g1
    return out


# ============================================================
# Grade-aware neural network layers (signature-aware)
# ============================================================


class CliffordLinear(nn.Module):
    """Grade-preserving linear map. One weight matrix per active grade."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        bias: bool = True,
        active_grades: tuple[int, ...] | None = None,
        algebra: GeometricAlgebra | None = None,
    ) -> None:
        super().__init__()
        if algebra is None:
            algebra = _get_default_algebra()
        self.c_in = c_in
        self.c_out = c_out
        self.algebra = algebra
        if active_grades is None:
            active_grades = tuple(range(algebra.max_grade + 1))
        self.active = tuple(sorted(active_grades))

        for g in self.active:
            w = nn.Parameter(torch.empty(c_out, c_in))
            nn.init.normal_(w, std=c_in**-0.5)
            setattr(self, f"w{g}", w)

        top_grade = algebra.max_grade
        self.has_b0 = bias and 0 in self.active
        self.has_b_top = bias and top_grade in self.active and top_grade != 0
        if self.has_b0:
            self.b0 = nn.Parameter(torch.zeros(c_out, 1))
        if self.has_b_top:
            self.b_top = nn.Parameter(torch.zeros(c_out, 1))
        self._top_grade = top_grade

    def forward(self, mv: torch.Tensor) -> torch.Tensor:
        out = mv.new_zeros(*mv.shape[:-2], self.c_out, self.algebra.dim)
        for g in self.active:
            s, e = self.algebra.grade_ranges[g]
            w = getattr(self, f"w{g}")
            out[..., s:e] = torch.einsum("oc,...cd->...od", w, mv[..., s:e])
        if self.has_b0:
            s0, e0 = self.algebra.grade_ranges[0]
            out[..., s0:e0] = out[..., s0:e0] + self.b0
        if self.has_b_top:
            st, et = self.algebra.grade_ranges[self._top_grade]
            out[..., st:et] = out[..., st:et] + self.b_top
        return out


class CliffordNorm(nn.Module):
    """Grade-wise normalization."""

    def __init__(
        self,
        n_channels: int,
        active_grades: tuple[int, ...] | None = None,
        eps: float = 1e-8,
        algebra: GeometricAlgebra | None = None,
    ) -> None:
        super().__init__()
        if algebra is None:
            algebra = _get_default_algebra()
        self.eps = eps
        self.algebra = algebra
        if active_grades is None:
            active_grades = tuple(range(algebra.max_grade + 1))
        self.active = set(active_grades)
        self.n_channels = n_channels
        self._top_grade = algebra.max_grade

        for g in range(algebra.max_grade + 1):
            if g in self.active:
                setattr(self, f"s{g}", nn.Parameter(torch.ones(n_channels, 1)))

    def forward(self, mv: torch.Tensor) -> torch.Tensor:
        slices: list[torch.Tensor] = []
        for g in range(self.algebra.max_grade + 1):
            s, e = self.algebra.grade_ranges[g]
            x = mv[..., s:e]
            if g not in self.active:
                slices.append(x)
                continue
            scale = getattr(self, f"s{g}")
            if self.algebra.grade_dims[g] == 1:
                mean = x.mean(dim=-2, keepdim=True)
                std = x.std(dim=-2, keepdim=True) + self.eps
                slices.append(scale * (x - mean) / std)
            else:
                ch_norm_sq = torch.sum(x**2, dim=-1, keepdim=True)
                rms = torch.sqrt(torch.mean(ch_norm_sq, dim=-2, keepdim=True) + self.eps)
                slices.append(scale * x / rms)
        return torch.cat(slices, dim=-1)


class CliffordGateActivation(nn.Module):
    """Equivariant norm-gated activation."""

    def __init__(
        self,
        n_channels: int,
        active_grades: tuple[int, ...] | None = None,
        algebra: GeometricAlgebra | None = None,
    ) -> None:
        super().__init__()
        if algebra is None:
            algebra = _get_default_algebra()
        self.algebra = algebra
        if active_grades is None:
            active_grades = tuple(range(algebra.max_grade + 1))
        self.active = set(active_grades)
        self.scalar_act = nn.SiLU()

        for g in range(1, algebra.max_grade):
            if g in self.active and algebra.grade_dims[g] > 1:
                gate = nn.Sequential(
                    nn.Linear(n_channels, n_channels),
                    nn.SiLU(),
                    nn.Linear(n_channels, n_channels),
                    nn.Sigmoid(),
                )
                setattr(self, f"gate_g{g}", gate)

    def forward(self, mv: torch.Tensor) -> torch.Tensor:
        slices: list[torch.Tensor] = []
        for g in range(self.algebra.max_grade + 1):
            s, e = self.algebra.grade_ranges[g]
            x = mv[..., s:e]
            if g not in self.active:
                slices.append(x)
                continue
            if self.algebra.grade_dims[g] == 1:
                slices.append(self.scalar_act(x))
            else:
                gate_fn = getattr(self, f"gate_g{g}")
                x_norm = torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True) + 1e-8)
                gate = gate_fn(x_norm.squeeze(-1)).unsqueeze(-1)
                slices.append(x * gate)
        return torch.cat(slices, dim=-1)


# ============================================================
# Backward-compat: Cl(3,0) equivariance smoke check
# ============================================================


def test_equivariance(
    model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    n_atoms: int = 5,
    n_channels: int = 16,
    atol: float = 1e-5,
    seed: int = 42,
) -> dict[str, float | bool]:
    """Numerically verify O(3) equivariance (Cl(3,0) backward-compat shim).

    Not a pytest test - this is a hand-rolled equivariance checker that
    ``clifford_stf`` re-exports as ``test_clifford_equivariance``.
    """
    torch.manual_seed(seed)
    alg = CliffordAlgebra()

    a = torch.randn(n_atoms, n_channels, 8)
    b = torch.randn(n_atoms, n_channels, 8)

    bv = torch.randn(3)
    bv = bv / (bv.norm() + 1e-8)
    angle = torch.tensor([torch.pi * torch.rand(1).item()])
    rotor = alg.rotor_from_bivector(bv.unsqueeze(0), angle.unsqueeze(0)).squeeze(0)

    def rotate(mv: torch.Tensor) -> torch.Tensor:
        return alg.sandwich_product(mv, rotor.expand_as(mv))

    results: dict[str, float | bool] = {}

    gp_then_rot = rotate(alg.geometric_product(a, b))
    rot_then_gp = alg.geometric_product(rotate(a), rotate(b))
    gp_err = (gp_then_rot - rot_then_gp).abs().max().item()
    results["gp_equivariance_error"] = gp_err
    results["gp_equivariant"] = gp_err < atol

    norms_orig = alg.grade_norms(a)
    norms_rot = alg.grade_norms(rotate(a))
    norm_err = (norms_orig - norms_rot).abs().max().item()
    results["grade_norm_invariance_error"] = norm_err
    results["grade_norms_invariant"] = norm_err < atol

    if model_fn is not None:
        with torch.no_grad():
            out_then_rot = rotate(model_fn(a, b))
            rot_then_out = model_fn(rotate(a), rotate(b))
        model_err = (out_then_rot - rot_then_out).abs().max().item()
        results["model_equivariance_error"] = model_err
        results["model_equivariant"] = model_err < atol

    return results


# Tell pytest not to auto-collect this as a test: the `test_` prefix is
# preserved for the historical re-export as `test_clifford_equivariance`.
test_equivariance.__test__ = False  # type: ignore[attr-defined]
