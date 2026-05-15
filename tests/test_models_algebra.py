"""Tests for ``cliffordstf.models._algebra`` Clifford primitives."""

from __future__ import annotations

from itertools import pairwise

import pytest
import torch

from cliffordstf.models._algebra.clifford import (
    ALL_GRADES,
    DIM,
    GRADE_DIMS,
    GRADE_RANGES,
    N_GRADES,
    CliffordAlgebra,
    CliffordGateActivation,
    CliffordLinear,
    CliffordNorm,
    compute_gp_output_grades,
    compute_l2_features,
    compute_layer_grades,
    make_grades01_mv,
    make_scalar_mv,
    make_vec_mv,
    test_equivariance,
)
from cliffordstf.models._algebra.geometric_algebra import GeometricAlgebra
from cliffordstf.models._algebra.interaction import (
    CliffordAttention,
    CosineCutoff,
    PerLayerEnergyReadout,
    RadialBasisFunctions,
)

# ============================================================
# GeometricAlgebra
# ============================================================


def test_geometric_algebra_cl_3_0_signature() -> None:
    alg = GeometricAlgebra((3, 0))
    assert alg.signature == (3, 0)
    assert alg.dim == 8
    assert alg.max_grade == 3
    # Grade dims: [scalar, vector, bivector, trivector] = [1, 3, 3, 1]
    assert alg.grade_dims == {0: 1, 1: 3, 2: 3, 3: 1}
    # Grade ranges must form a contiguous partition of [0, dim)
    expected_ranges = {0: (0, 1), 1: (1, 4), 2: (4, 7), 3: (7, 8)}
    assert alg.grade_ranges == expected_ranges
    # Cayley table shape
    assert alg.cayley.shape == (8, 8, 8)
    # Reverse signs: grade 0,1 -> +1; grade 2,3 -> -1 (Cl(3,0))
    assert torch.allclose(alg.reverse_signs[0:4], torch.tensor([1.0, 1.0, 1.0, 1.0]))
    assert torch.allclose(alg.reverse_signs[4:8], torch.tensor([-1.0, -1.0, -1.0, -1.0]))


def test_geometric_algebra_alternate_signature() -> None:
    # Cl(2, 1): p=2, q=1, n_basis=3, dim=8
    alg = GeometricAlgebra((2, 1))
    assert alg.dim == 8
    assert alg.n_basis == 3
    assert alg.grade_dims == {0: 1, 1: 3, 2: 3, 3: 1}
    # Cl(2, 0): p=2, q=0, n_basis=2, dim=4
    alg2 = GeometricAlgebra((2, 0))
    assert alg2.dim == 4
    assert alg2.grade_dims == {0: 1, 1: 2, 2: 1}


def test_geometric_algebra_rejects_zero_signature() -> None:
    with pytest.raises(ValueError, match="invalid signature"):
        GeometricAlgebra((0, 0))


# ============================================================
# CliffordAlgebra
# ============================================================


def test_clifford_algebra_geometric_product_shape() -> None:
    torch.manual_seed(0)
    alg = CliffordAlgebra()
    a = torch.randn(2, 4, 8)
    b = torch.randn(2, 4, 8)
    out = alg.geometric_product(a, b)
    assert out.shape == (2, 4, 8)
    assert torch.isfinite(out).all()


def test_clifford_algebra_reverse_involution() -> None:
    torch.manual_seed(1)
    alg = CliffordAlgebra()
    mv = torch.randn(3, 5, 8)
    assert torch.allclose(alg.reverse(alg.reverse(mv)), mv, atol=1e-6)


def test_clifford_algebra_grade_decompose_concat_roundtrip() -> None:
    torch.manual_seed(2)
    alg = CliffordAlgebra()
    mv = torch.randn(2, 3, 8)
    parts = alg.grade_decompose(mv)
    assert len(parts) == 4
    reconstructed = torch.cat(list(parts), dim=-1)
    assert torch.allclose(reconstructed, mv)


def test_clifford_algebra_rotor_vector_equivariance() -> None:
    """GP commutes with rotor sandwich: rot(GP(a, b)) = GP(rot(a), rot(b))."""
    torch.manual_seed(3)
    alg = CliffordAlgebra()
    a = torch.randn(2, 4, 8)
    b = torch.randn(2, 4, 8)

    bv = torch.randn(3)
    bv = bv / (bv.norm() + 1e-8)
    bv_mv = bv.unsqueeze(0)
    angle = torch.tensor([[0.7]])
    rotor = alg.rotor_from_bivector(bv_mv, angle).squeeze(0)

    rot_ab = alg.sandwich_product(alg.geometric_product(a, b), rotor.expand_as(a))
    ab_rot = alg.geometric_product(
        alg.sandwich_product(a, rotor.expand_as(a)),
        alg.sandwich_product(b, rotor.expand_as(b)),
    )
    assert torch.allclose(rot_ab, ab_rot, atol=1e-4)


def test_clifford_algebra_grade_norms_invariant_under_rotation() -> None:
    torch.manual_seed(4)
    alg = CliffordAlgebra()
    mv = torch.randn(2, 3, 8)
    bv = torch.randn(3)
    bv = bv / (bv.norm() + 1e-8)
    rotor = alg.rotor_from_bivector(bv.unsqueeze(0), torch.tensor([[0.5]])).squeeze(0)

    norms_orig = alg.grade_norms(mv)
    norms_rot = alg.grade_norms(alg.sandwich_product(mv, rotor.expand_as(mv)))
    assert torch.allclose(norms_orig, norms_rot, atol=1e-4)


# ============================================================
# Grade bookkeeping
# ============================================================


def test_compute_gp_output_grades_vector_vector() -> None:
    # GP(L=1, L=1) -> {scalar, bivector}
    assert compute_gp_output_grades((1,), (1,)) == (0, 2)


def test_compute_gp_output_grades_scalar_anything() -> None:
    # Scalar times anything preserves the grade
    assert compute_gp_output_grades((0,), (0, 1, 2, 3)) == (0, 1, 2, 3)


def test_compute_layer_grades_progresses_monotonically() -> None:
    grades = compute_layer_grades(n_layers=3, edge_grades=(0, 1), max_grade=3)
    assert len(grades) == 3
    # Each layer's grade set should be a superset of the previous
    for prev, curr in pairwise(grades):
        assert set(prev).issubset(set(curr))
    # Last layer cannot exceed max_grade
    assert all(g <= 3 for g in grades[-1])


def test_compute_layer_grades_respects_max_grade_cap() -> None:
    grades = compute_layer_grades(n_layers=4, edge_grades=(0, 1), max_grade=1)
    assert all(set(g) <= {0, 1} for g in grades)


def test_module_constants_consistent_with_cl_3_0() -> None:
    assert DIM == 8
    assert N_GRADES == 4
    assert GRADE_RANGES == {0: (0, 1), 1: (1, 4), 2: (4, 7), 3: (7, 8)}
    assert GRADE_DIMS == {0: 1, 1: 3, 2: 3, 3: 1}
    assert ALL_GRADES == (0, 1, 2, 3)


# ============================================================
# L=2 features & multivector constructors
# ============================================================


def test_compute_l2_features_shape_and_traceless() -> None:
    torch.manual_seed(5)
    direction = torch.randn(7, 3)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    l2 = compute_l2_features(direction)
    assert l2.shape == (7, 5)
    assert torch.isfinite(l2).all()
    # Symmetric-traceless reconstruction: S_zz = -S_xx - S_yy, trace(S) = 0
    s_xx, _s_xy, _s_xz, s_yy, _s_yz = l2.unbind(-1)
    s_zz = -s_xx - s_yy
    trace = s_xx + s_yy + s_zz
    assert torch.allclose(trace, torch.zeros_like(trace), atol=1e-6)


def test_make_scalar_mv_populates_grade_zero_only() -> None:
    s = torch.randn(2, 4)
    mv = make_scalar_mv(s)
    assert mv.shape == (2, 4, 8)
    assert torch.allclose(mv[..., 0], s)
    assert torch.allclose(mv[..., 1:], torch.zeros_like(mv[..., 1:]))


def test_make_vec_mv_populates_grade_one_only() -> None:
    v = torch.randn(3, 4, 3)
    mv = make_vec_mv(v)
    assert mv.shape == (3, 4, 8)
    assert torch.allclose(mv[..., 1:4], v)
    assert torch.allclose(mv[..., 0:1], torch.zeros_like(mv[..., 0:1]))
    assert torch.allclose(mv[..., 4:], torch.zeros_like(mv[..., 4:]))


def test_make_grades01_mv_populates_grade_zero_and_one() -> None:
    g0 = torch.randn(2, 3, 1)
    g1 = torch.randn(2, 3, 3)
    mv = make_grades01_mv(g0, g1)
    assert mv.shape == (2, 3, 8)
    assert torch.allclose(mv[..., 0:1], g0)
    assert torch.allclose(mv[..., 1:4], g1)
    assert torch.allclose(mv[..., 4:], torch.zeros_like(mv[..., 4:]))


# ============================================================
# Neural building blocks
# ============================================================


def test_clifford_linear_forward_shape_and_finite() -> None:
    torch.manual_seed(6)
    layer = CliffordLinear(c_in=4, c_out=8)
    mv = torch.randn(2, 4, 8)
    out = layer(mv)
    assert out.shape == (2, 8, 8)
    assert torch.isfinite(out).all()


def test_clifford_norm_forward_shape_and_finite() -> None:
    torch.manual_seed(7)
    norm = CliffordNorm(n_channels=4)
    mv = torch.randn(3, 4, 8)
    out = norm(mv)
    assert out.shape == (3, 4, 8)
    assert torch.isfinite(out).all()


def test_clifford_gate_activation_forward_shape_and_finite() -> None:
    torch.manual_seed(8)
    act = CliffordGateActivation(n_channels=4)
    mv = torch.randn(3, 4, 8)
    out = act(mv)
    assert out.shape == (3, 4, 8)
    assert torch.isfinite(out).all()


# ============================================================
# Radial basis + cosine cutoff
# ============================================================


def test_radial_basis_functions_output_shape() -> None:
    rbf = RadialBasisFunctions(n_rbf=12, cutoff=5.0)
    dist = torch.linspace(0.1, 4.9, 7)
    out = rbf(dist)
    assert out.shape == (7, 12)
    assert torch.isfinite(out).all()
    assert (out >= 0).all()


def test_cosine_cutoff_boundaries() -> None:
    cutoff = CosineCutoff(cutoff=5.0)
    # At r=0, cos(0)=1 -> value 1.0
    assert torch.allclose(cutoff(torch.tensor([0.0])), torch.tensor([1.0]))
    # At r=cutoff, cos(pi)=-1 -> value 0.0
    assert torch.allclose(cutoff(torch.tensor([5.0])), torch.tensor([0.0]))
    # Beyond cutoff, masked to 0
    assert torch.allclose(cutoff(torch.tensor([6.0])), torch.tensor([0.0]))


# ============================================================
# CliffordAttention (exercises _scatter_softmax_eager)
# ============================================================


def test_clifford_attention_forward_tiny_graph() -> None:
    torch.manual_seed(9)
    attn = CliffordAttention(n_channels=8, n_heads=2, n_rbf=4)
    # Tiny graph: 3 nodes, 2 edges (0->1, 1->2)
    n_edges = 2
    n_nodes = 3
    h_i = torch.randn(n_edges, 8)
    h_j = torch.randn(n_edges, 8)
    dist = torch.tensor([1.0, 1.5])
    dst = torch.tensor([1, 2])
    out = attn(h_i, h_j, dist, dst, n_nodes)
    assert out.shape == (n_edges, 1, 1)
    assert torch.isfinite(out).all()


def test_clifford_attention_rejects_non_divisible_channels() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        CliffordAttention(n_channels=7, n_heads=2)


# ============================================================
# PerLayerEnergyReadout
# ============================================================


def test_per_layer_energy_readout_no_batch() -> None:
    torch.manual_seed(10)
    readout = PerLayerEnergyReadout(n_channels=4, n_hidden=8)
    h = torch.randn(5, 4, 8)
    out = readout(h)
    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_per_layer_energy_readout_with_batch() -> None:
    torch.manual_seed(11)
    readout = PerLayerEnergyReadout(n_channels=4, n_hidden=8)
    h = torch.randn(6, 4, 8)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    out = readout(h, batch=batch, num_graphs=2)
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


# ============================================================
# test_equivariance helper
# ============================================================


def test_test_equivariance_returns_expected_keys() -> None:
    results = test_equivariance(n_atoms=3, n_channels=4)
    assert "gp_equivariance_error" in results
    assert "gp_equivariant" in results
    assert "grade_norm_invariance_error" in results
    assert "grade_norms_invariant" in results
    # The Cl(3,0) algebra is exactly equivariant -> both flags must be True
    assert results["gp_equivariant"] is True
    assert results["grade_norms_invariant"] is True
