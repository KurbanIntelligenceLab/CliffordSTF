"""Tests for ``cliffordstf.models._algebra`` Clifford primitives."""

from __future__ import annotations

import torch

from cliffordstf.models._algebra.clifford import (
    CliffordAlgebra,
    compute_gp_output_grades,
)
from cliffordstf.models._algebra.geometric_algebra import GeometricAlgebra
from cliffordstf.models._algebra.interaction import CosineCutoff


def test_geometric_algebra_cl_3_0_signature() -> None:
    alg = GeometricAlgebra((3, 0))
    assert alg.signature == (3, 0)
    assert alg.dim == 8
    assert alg.max_grade == 3

    assert alg.grade_dims == {0: 1, 1: 3, 2: 3, 3: 1}

    expected_ranges = {0: (0, 1), 1: (1, 4), 2: (4, 7), 3: (7, 8)}
    assert alg.grade_ranges == expected_ranges

    assert alg.cayley.shape == (8, 8, 8)

    assert torch.allclose(alg.reverse_signs[0:4], torch.tensor([1.0, 1.0, 1.0, 1.0]))
    assert torch.allclose(
        alg.reverse_signs[4:8], torch.tensor([-1.0, -1.0, -1.0, -1.0])
    )


def test_clifford_algebra_reverse_involution() -> None:
    torch.manual_seed(1)
    alg = CliffordAlgebra()
    mv = torch.randn(3, 5, 8)
    assert torch.allclose(alg.reverse(alg.reverse(mv)), mv, atol=1e-6)


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


def test_compute_gp_output_grades_vector_vector() -> None:

    assert compute_gp_output_grades((1,), (1,)) == (0, 2)


def test_compute_gp_output_grades_scalar_anything() -> None:

    assert compute_gp_output_grades((0,), (0, 1, 2, 3)) == (0, 1, 2, 3)


def test_cosine_cutoff_boundaries() -> None:
    cutoff = CosineCutoff(cutoff=5.0)

    assert torch.allclose(cutoff(torch.tensor([0.0])), torch.tensor([1.0]))

    assert torch.allclose(cutoff(torch.tensor([5.0])), torch.tensor([0.0]))

    assert torch.allclose(cutoff(torch.tensor([6.0])), torch.tensor([0.0]))
