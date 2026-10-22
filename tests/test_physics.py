"""Check tensor identities and molecular symmetries."""

import itertools

import pytest
import torch
from torch_geometric.data import Data

from cliffordstf.models.stf import compute_stf2_product, compute_stf3_product
from cliffordstf.models.wrapper import build_from_ablation


def matrix(s):
    """Expand five STF components into a symmetric matrix."""
    xx, xy, xz, yy, yz = s.unbind(-1)
    return torch.stack([xx, xy, xz, xy, yy, yz, xz, yz, -xx - yy], -1).reshape(
        *s.shape[:-1], 3, 3
    )


def rank3(t):
    """Expand seven STF components into a symmetric tensor."""
    xxx, xxy, xxz, xyy, xyz, yyy, yyz = t.unbind(-1)
    values = [xxx, xxy, xxz, xyy, xyz, -xxx - xyy, yyy, yyz, -xxy - yyy, -xxz - yyz]
    out = t.new_zeros(*t.shape[:-1], 3, 3, 3)
    for indices, value in zip(
        itertools.combinations_with_replacement(range(3), 3), values
    ):
        for idx in set(itertools.permutations(indices)):
            out[..., idx[0], idx[1], idx[2]] = value
    return out


def test_stf_projectors_and_rotations():
    """Compare STF products with Cartesian projectors in double precision."""
    torch.manual_seed(42)
    u, v = torch.randn(2, 1000, 3, dtype=torch.float64)
    eye = torch.eye(3, dtype=u.dtype)
    outer = u[..., :, None] * v[..., None, :]
    expected = (outer + outer.transpose(-1, -2)) / 2 - (u * v).sum(-1)[
        ..., None, None
    ] * eye / 3
    s = matrix(compute_stf2_product(u, v))
    torch.testing.assert_close(s, expected, atol=1e-11, rtol=0)
    sv = torch.einsum("nij,nj->ni", s, v)
    sym = (
        torch.einsum("nij,nk->nijk", s, v)
        + torch.einsum("nik,nj->nijk", s, v)
        + torch.einsum("njk,ni->nijk", s, v)
    ) / 3
    correction = (
        (
            torch.einsum("ij,nk->nijk", eye, sv)
            + torch.einsum("ik,nj->nijk", eye, sv)
            + torch.einsum("jk,ni->nijk", eye, sv)
        )
        * 2
        / 15
    )
    t = rank3(compute_stf3_product(compute_stf2_product(u, v), v))
    torch.testing.assert_close(t, sym - correction, atol=1e-11, rtol=0)
    torch.testing.assert_close(
        t.diagonal(dim1=-3, dim2=-2).sum(-1), torch.zeros_like(v), atol=1e-11, rtol=0
    )
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=u.dtype))
    q[:, 0] *= torch.det(q)
    rotated_s = compute_stf2_product(u @ q.T, v @ q.T)
    torch.testing.assert_close(matrix(rotated_s), q @ s @ q.T, atol=1e-11, rtol=0)
    rotated_t = rank3(compute_stf3_product(rotated_s, v @ q.T))
    torch.testing.assert_close(
        rotated_t, torch.einsum("ia,jb,kc,nabc->nijk", q, q, q, t), atol=1e-11, rtol=0
    )


VARIANTS = [
    ("baseline", {"use_l2": False}),
    ("baseline", {"use_l2": True}),
    ("hodge_only", {}),
    ("stf2_no_hodge", {}),
    ("stf2", {}),
    ("stf2_no_cross", {}),
    ("stf2_stf3", {}),
    ("stf2_static_routing", {}),
    ("stf2_learned_routing", {}),
    ("full", {}),
    ("full_no_cross", {}),
    ("stf_only_output", {}),
]


def molecule(pos=None):
    """Return a non-collinear four-atom geometry."""
    if pos is None:
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [1.1, 0.2, 0.1], [0.2, 1.2, -0.2], [-0.3, 0.4, 1.3]]
        )
    return Data(
        z=torch.tensor([6, 1, 8, 1]), pos=pos, batch=torch.zeros(4, dtype=torch.long)
    )


def model(variant="full", **overrides):
    """Build a small instance of a manuscript architecture."""
    return build_from_ablation(
        variant,
        n_channels=8,
        n_interactions=2,
        n_rbf=8,
        n_hidden_output=8,
        n_heads=2,
        cutoff=4.0,
        max_neighbors=20,
        use_self_interaction=False,
        use_gp_readout=False,
        use_ema=False,
        **overrides,
    ).eval()


@pytest.mark.parametrize("variant,overrides", VARIANTS)
def test_molecular_rotation_translation_permutation(variant, overrides):
    """Check energy invariance and force covariance for each ablation."""
    torch.manual_seed(7)
    net = model(variant, **overrides)
    data = molecule()
    energy, forces = net(data)
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    q[:, 0] *= torch.det(q)
    transformed = molecule(data.pos @ q.T + torch.tensor([2.0, -1.0, 0.5]))
    e, f = net(transformed)
    torch.testing.assert_close(e, energy, atol=1e-3, rtol=0)
    torch.testing.assert_close(f, forces @ q.T, atol=1e-3, rtol=0)
    order = torch.tensor([2, 0, 3, 1])
    permuted = Data(z=data.z[order], pos=data.pos[order], batch=data.batch)
    e, f = net(permuted)
    torch.testing.assert_close(e, energy, atol=1e-3, rtol=0)
    torch.testing.assert_close(f, forces[order], atol=1e-3, rtol=0)


def test_energy_gradient_forces_and_conservation():
    """Check conservative forces against finite differences and zero net force and torque."""
    torch.manual_seed(17)
    net = model(direct_forces=False).double()
    pos = molecule().pos.double()
    _, forces = net(molecule(pos.clone()))
    step = 1e-5
    numerical = torch.zeros_like(pos)
    for atom in range(4):
        for axis in range(3):
            plus, minus = pos.clone(), pos.clone()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            ep, _ = net(molecule(plus))
            em, _ = net(molecule(minus))
            numerical[atom, axis] = -(ep - em).item() / (2 * step)
    torch.testing.assert_close(forces, numerical, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(
        forces.sum(0), torch.zeros(3, dtype=pos.dtype), atol=1e-9, rtol=0
    )
    torch.testing.assert_close(
        torch.linalg.cross(pos, forces).sum(0),
        torch.zeros(3, dtype=pos.dtype),
        atol=1e-9,
        rtol=0,
    )
