"""CliffordSTF Algebra: Cl(3,0) + STF_2 + STF_3.

Core theorem exploited here:
    For L=1 vectors u, v, the full CG product 1 (x) 1 -> 0 + 1 + 2 decomposes as
        GP(u, v) = u . v + u ^ v        -> L=0 (1D) + L=1 (3D)
        STF(u, v) = sym traceless(u (x) v) -> L=2 (5D)
    Together GP + STF = full CG, computed without CG coefficients.

Layout of CliffordSTF multivector (per channel):
    [0:8]   Cl(3,0) multivector  - grades 0,1,2,3 = 2 . L=0 + 2 . L=1
    [8:13]  STF_2                - symmetric traceless rank-2 = L=2 (5D)
    [13:20] STF_3                - symmetric traceless rank-3 = L=3 (7D)

    8D  mode: stf_mode="none"    (original Clifford)
    13D mode: stf_mode="stf2"    (+ L=2)
    20D mode: stf_mode="stf2+stf3" (+ L=2 + L=3)

STF_2 storage: [S_xx, S_xy, S_xz, S_yy, S_yz]  (S_zz = -S_xx - S_yy)
STF_3 storage: [T_xxx, T_xxy, T_xxz, T_xyy, T_xyz, T_yyy, T_yyz]
              (T_xzz = -T_xxx - T_xyy, T_yzz = -T_xxy - T_yyy,
               T_zzz = -T_xxz - T_yyz)
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from cliffordstf.models._algebra.clifford import (
    ALL_GRADES,
    DIM,
    CliffordAlgebra,
    CliffordGateActivation,
    CliffordLinear,
    CliffordNorm,
    make_vec_mv,
)

CL_DIM: int = DIM

STF2_DIM = 5
STF3_DIM = 7
AUG13_DIM = CL_DIM + STF2_DIM
AUG20_DIM = CL_DIM + STF2_DIM + STF3_DIM

AUG_CL_RANGE = (0, CL_DIM)
AUG_STF2_RANGE = (CL_DIM, AUG13_DIM)
AUG_STF3_RANGE = (AUG13_DIM, AUG20_DIM)

STF_MODES: dict[str, int] = {"none": CL_DIM, "stf2": AUG13_DIM, "stf2+stf3": AUG20_DIM}

TRACK_CL = "cl"
TRACK_STF2 = "stf2"
TRACK_STF3 = "stf3"


def clifford_stf_dim(stf_mode: str) -> int:
    return STF_MODES[stf_mode]


def compute_stf2_product(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Symmetric traceless product of two L=1 vectors.

    This is the L=2 channel of CG(1 (x) 1) that the geometric product misses.

    Args:
        u: ``(..., 3)`` vector.
        v: ``(..., 3)`` vector.

    Returns:
        ``(..., 5)`` STF_2 components ``[S_xx, S_xy, S_xz, S_yy, S_yz]``.
    """
    ux, uy, uz = u.unbind(-1)
    vx, vy, vz = v.unbind(-1)

    dot = ux * vx + uy * vy + uz * vz
    third = dot / 3.0

    s_xx = ux * vx - third
    s_xy = 0.5 * (ux * vy + uy * vx)
    s_xz = 0.5 * (ux * vz + uz * vx)
    s_yy = uy * vy - third
    s_yz = 0.5 * (uy * vz + uz * vy)

    return torch.stack([s_xx, s_xy, s_xz, s_yy, s_yz], dim=-1)


def stf2_norm_sq(s: torch.Tensor) -> torch.Tensor:
    """Frobenius norm^2 of STF_2 tensor -> L=0 invariant.

    ||S||^2 = S_xx^2 + S_yy^2 + S_zz^2 + 2 (S_xy^2 + S_xz^2 + S_yz^2),
    where ``S_zz = -S_xx - S_yy``.
    """
    s_xx, s_xy, s_xz, s_yy, s_yz = s.unbind(-1)
    s_zz = -s_xx - s_yy
    return (s_xx**2 + s_yy**2 + s_zz**2 + 2.0 * (s_xy**2 + s_xz**2 + s_yz**2)).unsqueeze(-1)


def stf2_inner(s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    """Inner product of two STF_2 tensors -> L=0 invariant."""
    a_xx, a_xy, a_xz, a_yy, a_yz = s1.unbind(-1)
    b_xx, b_xy, b_xz, b_yy, b_yz = s2.unbind(-1)
    a_zz = -a_xx - a_yy
    b_zz = -b_xx - b_yy
    return (
        a_xx * b_xx + a_yy * b_yy + a_zz * b_zz + 2.0 * (a_xy * b_xy + a_xz * b_xz + a_yz * b_yz)
    ).unsqueeze(-1)


def reconstruct_stf2_matrix(s: torch.Tensor) -> torch.Tensor:
    """Reconstruct full 3x3 symmetric traceless matrix from 5 components."""
    s_xx, s_xy, s_xz, s_yy, s_yz = s.unbind(-1)
    s_zz = -s_xx - s_yy
    row0 = torch.stack([s_xx, s_xy, s_xz], dim=-1)
    row1 = torch.stack([s_xy, s_yy, s_yz], dim=-1)
    row2 = torch.stack([s_xz, s_yz, s_zz], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def contract_stf2_vec(s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Contract STF_2 with vector: S . v -> L=1 vector.

    L=2 (x) L=1 -> L=1 channel (CG contraction). Implemented via the reconstructed
    3x3 matrix and a single einsum.
    """
    s_mat = reconstruct_stf2_matrix(s)
    return torch.einsum("...ij,...j->...i", s_mat, v)


def compute_stf3_product(
    stf2: torch.Tensor,
    v: torch.Tensor,
    precomputed_sv: torch.Tensor | None = None,
) -> torch.Tensor:
    """STF_3 from STF_2 (x) L=1 -> L=3 channel.

    Computes the symmetric traceless part of ``S_ij * v_k``, the L=3 channel of
    CG(2 (x) 1 -> 1 + 2 + 3).
    """
    s_xx, s_xy, s_xz, s_yy, s_yz = stf2.unbind(-1)
    vx, vy, vz = v.unbind(-1)

    t_xxx = s_xx * vx
    t_xxy = (s_xx * vy + 2.0 * s_xy * vx) / 3.0
    t_xxz = (s_xx * vz + 2.0 * s_xz * vx) / 3.0
    t_xyy = (2.0 * s_xy * vy + s_yy * vx) / 3.0
    t_xyz = (s_xy * vz + s_xz * vy + s_yz * vx) / 3.0
    t_yyy = s_yy * vy
    t_yyz = (s_yy * vz + 2.0 * s_yz * vy) / 3.0

    sv = contract_stf2_vec(stf2, v) if precomputed_sv is None else precomputed_sv
    ax, ay, az = (sv * (2.0 / 3.0)).unbind(-1)

    inv5 = 1.0 / 5.0
    o_xxx = t_xxx - 3.0 * inv5 * ax
    o_xxy = t_xxy - inv5 * ay
    o_xxz = t_xxz - inv5 * az
    o_xyy = t_xyy - inv5 * ax
    o_xyz = t_xyz
    o_yyy = t_yyy - 3.0 * inv5 * ay
    o_yyz = t_yyz - inv5 * az

    return torch.stack([o_xxx, o_xxy, o_xxz, o_xyy, o_xyz, o_yyy, o_yyz], dim=-1)


def stf3_norm_sq(t: torch.Tensor) -> torch.Tensor:
    """Norm^2 of STF_3 tensor -> L=0 invariant."""
    t_xxx, t_xxy, t_xxz, t_xyy, t_xyz, t_yyy, t_yyz = t.unbind(-1)
    t_xzz = -t_xxx - t_xyy
    t_yzz = -t_xxy - t_yyy
    t_zzz = -t_xxz - t_yyz

    return (
        t_xxx**2
        + t_yyy**2
        + t_zzz**2
        + 3.0 * (t_xxy**2 + t_xxz**2 + t_xyy**2 + t_xzz**2 + t_yyz**2 + t_yzz**2)
        + 6.0 * t_xyz**2
    ).unsqueeze(-1)


def contract_stf3_vec_to_stf2(t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Contract STF_3 with vector -> STF_2 (L=3 (x) L=1 -> L=2 channel).

    ``R_ij = sum_k T_ijk * v_k`` (symmetric, projected to traceless).
    """
    t_xxx, t_xxy, t_xxz, t_xyy, t_xyz, t_yyy, t_yyz = t.unbind(-1)
    vx, vy, vz = v.unbind(-1)
    t_xzz = -t_xxx - t_xyy
    t_yzz = -t_xxy - t_yyy
    t_zzz = -t_xxz - t_yyz

    r_xx = t_xxx * vx + t_xxy * vy + t_xxz * vz
    r_xy = t_xxy * vx + t_xyy * vy + t_xyz * vz
    r_xz = t_xxz * vx + t_xyz * vy + t_xzz * vz
    r_yy = t_xyy * vx + t_yyy * vy + t_yyz * vz
    r_yz = t_xyz * vx + t_yyz * vy + t_yzz * vz
    r_zz = t_xzz * vx + t_yzz * vy + t_zzz * vz

    tr = r_xx + r_yy + r_zz
    third_tr = tr / 3.0

    return torch.stack(
        [r_xx - third_tr, r_xy, r_xz, r_yy - third_tr, r_yz],
        dim=-1,
    )


def contract_stf2_vec_to_stf2(s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """L=2 (x) L=1 -> L=2 channel (antisymmetric-like contraction).

    Middle CG channel of ``2 (x) 1 -> 1 + 2 + 3``. Computed as the
    cross-product-like operation between rows of ``S`` and ``v``, then
    symmetrized and projected to traceless.
    """
    s_xx, s_xy, s_xz, s_yy, s_yz = s.unbind(-1)
    vx, vy, vz = v.unbind(-1)
    s_zz = -s_xx - s_yy

    cx_x = s_xy * vz - s_xz * vy
    cx_y = s_xz * vx - s_xx * vz
    cx_z = s_xx * vy - s_xy * vx

    cy_x = s_yy * vz - s_yz * vy
    cy_y = s_yz * vx - s_xy * vz
    cy_z = s_xy * vy - s_yy * vx

    r_xx = cx_x
    r_xy = 0.5 * (cx_y + cy_x)
    r_yy = cy_y

    cz_x = s_yz * vz - s_zz * vy
    cz_y = s_zz * vx - s_xz * vz

    r_xz = 0.5 * (cx_z + cz_x)
    r_yz = 0.5 * (cy_z + cz_y)

    cz_z = s_xz * vy - s_yz * vx
    raw_trace = cx_x + cy_y + cz_z
    third_tr = raw_trace / 3.0

    return torch.stack(
        [r_xx - third_tr, r_xy, r_xz, r_yy - third_tr, r_yz],
        dim=-1,
    )


def hodge_star_g2_to_vec(mv: torch.Tensor) -> torch.Tensor:
    """Extract L=1 pseudovector from grade-2 bivectors via Hodge dual.

    In Cl(3,0): ``*e12 = e3``, ``*e13 = -e2``, ``*e23 = e1``. Grade-2 layout is
    ``[e12, e13, e23]`` at indices ``[4, 5, 6]``.
    """
    return torch.stack([mv[..., 6], -mv[..., 5], mv[..., 4]], dim=-1)


def make_aug_mv(
    cl: torch.Tensor,
    stf2: torch.Tensor | None = None,
    stf3: torch.Tensor | None = None,
) -> torch.Tensor:
    """Construct augmented multivector from components.

    Returns ``(..., C, D)`` where ``D in {8, 13, 20}``.
    """
    parts = [cl]
    if stf2 is not None:
        parts.append(stf2)
    if stf3 is not None:
        assert stf2 is not None, "Cannot have STF3 without STF2"
        parts.append(stf3)
    return torch.cat(parts, dim=-1)


def split_aug_mv(
    aug: torch.Tensor, stf_mode: str = "stf2+stf3"
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Split augmented multivector into Clifford / STF_2 / STF_3 components."""
    cl = aug[..., :CL_DIM]
    stf2 = aug[..., CL_DIM:AUG13_DIM] if stf_mode != "none" else None
    stf3 = aug[..., AUG13_DIM:AUG20_DIM] if stf_mode == "stf2+stf3" else None
    return cl, stf2, stf3


def pad_cl_to_aug(cl: torch.Tensor, stf_mode: str) -> torch.Tensor:
    """Pad a Clifford multivector to augmented size with zeros."""
    if stf_mode == "none":
        return cl
    d = clifford_stf_dim(stf_mode)
    pad_size = d - CL_DIM
    padding = cl.new_zeros(*cl.shape[:-1], pad_size)
    return torch.cat([cl, padding], dim=-1)


class CliffordSTFAlgebra(CliffordAlgebra):
    """Cl(3,0) + STF_2 + STF_3 with cross-track products.

    Inherits all GP variants from :class:`CliffordAlgebra`. Adds an augmented
    product that couples the Clifford and STF tracks.
    """

    def __init__(self) -> None:
        super().__init__()

    def augmented_product(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        stf_mode: str = "stf2",
        grades_a: tuple[int, ...] = ALL_GRADES,
        grades_b: tuple[int, ...] = ALL_GRADES,
    ) -> torch.Tensor:
        """Full augmented product: GP on Clifford + STF generation/coupling."""
        d = clifford_stf_dim(stf_mode)
        a_cl, a_s2, a_s3 = split_aug_mv(a, stf_mode)
        b_cl, b_s2, b_s3 = split_aug_mv(b, stf_mode)

        out_cl = self.dispatch_gp(a_cl, b_cl, grades_a, grades_b)

        if stf_mode == "none":
            return out_cl

        a_vec = a_cl[..., 1:4]
        b_vec = b_cl[..., 1:4]

        out_s2 = compute_stf2_product(a_vec, b_vec)

        if a_s2 is not None:
            out_s2 = out_s2 + contract_stf2_vec_to_stf2(a_s2, b_vec)
        if b_s2 is not None:
            out_s2 = out_s2 + contract_stf2_vec_to_stf2(b_s2, a_vec)

        if a_s2 is not None and b_s2 is not None:
            s2_scalar = stf2_inner(a_s2, b_s2)
            out_cl[..., 0:1] = out_cl[..., 0:1] + s2_scalar

        out_s3: torch.Tensor | None = None
        if stf_mode == "stf2+stf3":
            out_s3 = a_cl.new_zeros(*a_vec.shape[:-1], STF3_DIM)
            if a_s2 is not None:
                out_s3 = out_s3 + compute_stf3_product(a_s2, b_vec)
            if b_s2 is not None:
                out_s3 = out_s3 + compute_stf3_product(b_s2, a_vec)

            if a_s3 is not None:
                out_s2 = out_s2 + contract_stf3_vec_to_stf2(a_s3, b_vec)
            if b_s3 is not None:
                out_s2 = out_s2 + contract_stf3_vec_to_stf2(b_s3, a_vec)

        out = torch.empty(*a.shape[:-1], d, device=a.device, dtype=a.dtype)
        out[..., :CL_DIM] = out_cl
        out[..., CL_DIM:AUG13_DIM] = out_s2
        if stf_mode == "stf2+stf3":
            assert out_s3 is not None
            out[..., AUG13_DIM:AUG20_DIM] = out_s3

        return out


class CliffordSTFLinear(nn.Module):
    """Block-diagonal linear map for augmented multivectors.

    Separate weight matrices for the Clifford track (8D, grade-preserving), the
    STF_2 track (5D), and the STF_3 track (7D). No cross-track linear mixing -
    cross-track coupling happens only through products.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        bias: bool = True,
        active_grades: tuple[int, ...] = ALL_GRADES,
        stf_mode: str = "stf2",
    ) -> None:
        super().__init__()
        self.stf_mode = stf_mode

        self.cl_linear = CliffordLinear(c_in, c_out, bias=bias, active_grades=active_grades)

        if stf_mode != "none":
            self.stf2_linear = nn.Linear(c_in, c_out, bias=False)
            nn.init.normal_(self.stf2_linear.weight, std=c_in**-0.5)

        if stf_mode == "stf2+stf3":
            self.stf3_linear = nn.Linear(c_in, c_out, bias=False)
            nn.init.normal_(self.stf3_linear.weight, std=c_in**-0.5)

    def forward(self, aug: torch.Tensor) -> torch.Tensor:
        cl, s2, s3 = split_aug_mv(aug, self.stf_mode)

        out_cl = cast(torch.Tensor, self.cl_linear(cl))

        if self.stf_mode == "none":
            return out_cl

        assert s2 is not None
        out_s2 = torch.einsum("oc,...ci->...oi", self.stf2_linear.weight, s2)

        parts = [out_cl, out_s2]

        if self.stf_mode == "stf2+stf3":
            assert s3 is not None
            out_s3 = torch.einsum("oc,...ci->...oi", self.stf3_linear.weight, s3)
            parts.append(out_s3)

        return torch.cat(parts, dim=-1)


class CliffordSTFNorm(nn.Module):
    """Grade-wise + track-wise normalization.

    Clifford uses :class:`CliffordNorm` (per-grade norm). STF_2 and STF_3 each
    use a Frobenius-norm RMS normalization over channels.
    """

    def __init__(
        self,
        n_channels: int,
        active_grades: tuple[int, ...] = ALL_GRADES,
        stf_mode: str = "stf2",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.stf_mode = stf_mode
        self.eps = eps

        self.cl_norm = CliffordNorm(n_channels, active_grades, eps)

        if stf_mode != "none":
            self.stf2_scale = nn.Parameter(torch.ones(n_channels, 1))

        if stf_mode == "stf2+stf3":
            self.stf3_scale = nn.Parameter(torch.ones(n_channels, 1))

    def forward(self, aug: torch.Tensor) -> torch.Tensor:
        cl, s2, s3 = split_aug_mv(aug, self.stf_mode)

        out_cl = cast(torch.Tensor, self.cl_norm(cl))

        if self.stf_mode == "none":
            return out_cl

        assert s2 is not None
        s2_frob = stf2_norm_sq(s2)
        s2_rms = torch.sqrt(torch.mean(s2_frob, dim=-2, keepdim=True) + self.eps)
        out_s2 = self.stf2_scale * s2 / s2_rms

        parts = [out_cl, out_s2]

        if self.stf_mode == "stf2+stf3":
            assert s3 is not None
            s3_frob = stf3_norm_sq(s3)
            s3_rms = torch.sqrt(torch.mean(s3_frob, dim=-2, keepdim=True) + self.eps)
            out_s3 = self.stf3_scale * s3 / s3_rms
            parts.append(out_s3)

        return torch.cat(parts, dim=-1)


class CliffordSTFGateActivation(nn.Module):
    """Equivariant nonlinearity for augmented multivectors.

    Clifford uses :class:`CliffordGateActivation` (norm-gating per grade). STF_2
    and STF_3 each use a scalar gate driven by their Frobenius norm.
    """

    def __init__(
        self,
        n_channels: int,
        active_grades: tuple[int, ...] = ALL_GRADES,
        stf_mode: str = "stf2",
    ) -> None:
        super().__init__()
        self.stf_mode = stf_mode

        self.cl_act = CliffordGateActivation(n_channels, active_grades)

        if stf_mode != "none":
            self.stf2_gate = nn.Sequential(
                nn.Linear(n_channels, n_channels),
                nn.SiLU(),
                nn.Linear(n_channels, n_channels),
                nn.Sigmoid(),
            )

        if stf_mode == "stf2+stf3":
            self.stf3_gate = nn.Sequential(
                nn.Linear(n_channels, n_channels),
                nn.SiLU(),
                nn.Linear(n_channels, n_channels),
                nn.Sigmoid(),
            )

    def forward(self, aug: torch.Tensor) -> torch.Tensor:
        cl, s2, s3 = split_aug_mv(aug, self.stf_mode)

        out_cl = cast(torch.Tensor, self.cl_act(cl))

        if self.stf_mode == "none":
            return out_cl

        assert s2 is not None
        s2_norm = torch.sqrt(stf2_norm_sq(s2) + 1e-8)
        gate2 = self.stf2_gate(s2_norm.squeeze(-1)).unsqueeze(-1)
        out_s2 = s2 * gate2

        parts = [out_cl, out_s2]

        if self.stf_mode == "stf2+stf3":
            assert s3 is not None
            s3_norm = torch.sqrt(stf3_norm_sq(s3) + 1e-8)
            gate3 = self.stf3_gate(s3_norm.squeeze(-1)).unsqueeze(-1)
            out_s3 = s3 * gate3
            parts.append(out_s3)

        return torch.cat(parts, dim=-1)


def test_stf2_equivariance(
    n: int = 32, c: int = 8, atol: float = 1e-5, seed: int = 42
) -> dict[str, float | bool]:
    """Verify STF_2 transforms correctly under SO(3) rotation."""
    torch.manual_seed(seed)

    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, 0] *= -1

    u = torch.randn(n, c, 3)
    v = torch.randn(n, c, 3)

    stf = compute_stf2_product(u, v)
    s_xx, s_xy, s_xz, s_yy, s_yz = stf.unbind(-1)
    s_zz = -s_xx - s_yy
    s_mat = torch.stack(
        [
            torch.stack([s_xx, s_xy, s_xz], -1),
            torch.stack([s_xy, s_yy, s_yz], -1),
            torch.stack([s_xz, s_yz, s_zz], -1),
        ],
        -2,
    )
    s_rot = q @ s_mat @ q.T

    u_rot = torch.einsum("ij,...j->...i", q, u)
    v_rot = torch.einsum("ij,...j->...i", q, v)
    stf_rot = compute_stf2_product(u_rot, v_rot)
    s2_xx, s2_xy, s2_xz, s2_yy, s2_yz = stf_rot.unbind(-1)

    err = max(
        (s_rot[..., 0, 0] - s2_xx).abs().max().item(),
        (s_rot[..., 0, 1] - s2_xy).abs().max().item(),
        (s_rot[..., 0, 2] - s2_xz).abs().max().item(),
        (s_rot[..., 1, 1] - s2_yy).abs().max().item(),
        (s_rot[..., 1, 2] - s2_yz).abs().max().item(),
    )
    return {"stf2_equivariance_error": err, "stf2_equivariant": err < atol}


test_stf2_equivariance.__test__ = False  # type: ignore[attr-defined]


def test_stf3_equivariance(
    n: int = 16, c: int = 4, atol: float = 1e-4, seed: int = 42
) -> dict[str, float | bool]:
    """Verify STF_3 transforms correctly under SO(3)."""
    torch.manual_seed(seed)

    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, 0] *= -1

    stf2 = torch.randn(n, c, 5)
    v = torch.randn(n, c, 3)

    t = compute_stf3_product(stf2, v)

    s_xx, s_xy, s_xz, s_yy, s_yz = stf2.unbind(-1)
    s_zz = -s_xx - s_yy
    s_mat = torch.stack(
        [
            torch.stack([s_xx, s_xy, s_xz], -1),
            torch.stack([s_xy, s_yy, s_yz], -1),
            torch.stack([s_xz, s_yz, s_zz], -1),
        ],
        -2,
    )
    s_rot = q @ s_mat @ q.T
    stf2_rot = torch.stack(
        [
            s_rot[..., 0, 0],
            s_rot[..., 0, 1],
            s_rot[..., 0, 2],
            s_rot[..., 1, 1],
            s_rot[..., 1, 2],
        ],
        dim=-1,
    )

    v_rot = torch.einsum("ij,...j->...i", q, v)
    t_rot_from_rot_inputs = compute_stf3_product(stf2_rot, v_rot)

    t_xxx, t_xxy, t_xxz, t_xyy, t_xyz, t_yyy, t_yyz = t.unbind(-1)
    t_xzz = -t_xxx - t_xyy
    t_yzz = -t_xxy - t_yyy
    t_zzz = -t_xxz - t_yyz

    t_full = torch.zeros(*t.shape[:-1], 3, 3, 3, device=t.device)
    idx_map = {
        (0, 0, 0): t_xxx,
        (0, 0, 1): t_xxy,
        (0, 0, 2): t_xxz,
        (0, 1, 1): t_xyy,
        (0, 1, 2): t_xyz,
        (0, 2, 2): t_xzz,
        (1, 1, 1): t_yyy,
        (1, 1, 2): t_yyz,
        (1, 2, 2): t_yzz,
        (2, 2, 2): t_zzz,
    }
    for (i, j, k), val in idx_map.items():
        for pi, pj, pk in {
            (i, j, k),
            (i, k, j),
            (j, i, k),
            (j, k, i),
            (k, i, j),
            (k, j, i),
        }:
            t_full[..., pi, pj, pk] = val

    t_rot = torch.einsum("ia,jb,kc,...abc->...ijk", q, q, q, t_full)

    t_rot_direct = torch.stack(
        [
            t_rot[..., 0, 0, 0],
            t_rot[..., 0, 0, 1],
            t_rot[..., 0, 0, 2],
            t_rot[..., 0, 1, 1],
            t_rot[..., 0, 1, 2],
            t_rot[..., 1, 1, 1],
            t_rot[..., 1, 1, 2],
        ],
        dim=-1,
    )

    err = (t_rot_direct - t_rot_from_rot_inputs).abs().max().item()
    return {"stf3_equivariance_error": err, "stf3_equivariant": err < atol}


test_stf3_equivariance.__test__ = False  # type: ignore[attr-defined]


def test_gp_plus_stf_equals_cg(
    n: int = 32, c: int = 8, atol: float = 1e-5, seed: int = 42
) -> dict[str, float | bool]:
    """Verify GP + STF = full CG(1 (x) 1) numerically.

    For two vectors ``u``, ``v``:
      - CG(1 (x) 1) produces 9 components: L=0(1) + L=1(3) + L=2(5).
      - GP(u, v) produces L=0(1) + L=1(3) (in indices ``0``, ``4``, ``5``, ``6``).
      - STF(u, v) produces L=2(5).
    """
    torch.manual_seed(seed)
    alg = CliffordAlgebra()

    u = torch.randn(n, c, 3)
    v = torch.randn(n, c, 3)

    outer = torch.einsum("...i,...j->...ij", u, v)
    dot = torch.einsum("...ii->...", outer)

    antisym = 0.5 * (outer - outer.transpose(-1, -2))

    sym = 0.5 * (outer + outer.transpose(-1, -2))
    stl = sym - (dot / 3.0).unsqueeze(-1).unsqueeze(-1) * torch.eye(3)

    u_mv = make_vec_mv(u)
    v_mv = make_vec_mv(v)
    gp = alg.geometric_product(u_mv, v_mv)

    gp_scalar = gp[..., 0]
    err_l0 = (gp_scalar - dot).abs().max().item()

    gp_e12 = gp[..., 4]
    gp_e13 = gp[..., 5]
    gp_e23 = gp[..., 6]
    err_l1 = max(
        (gp_e12 - 2.0 * antisym[..., 0, 1]).abs().max().item(),
        (gp_e13 - 2.0 * antisym[..., 0, 2]).abs().max().item(),
        (gp_e23 - 2.0 * antisym[..., 1, 2]).abs().max().item(),
    )

    stf = compute_stf2_product(u, v)
    s_xx, s_xy, s_xz, s_yy, s_yz = stf.unbind(-1)
    err_l2 = max(
        (s_xx - stl[..., 0, 0]).abs().max().item(),
        (s_xy - stl[..., 0, 1]).abs().max().item(),
        (s_xz - stl[..., 0, 2]).abs().max().item(),
        (s_yy - stl[..., 1, 1]).abs().max().item(),
        (s_yz - stl[..., 1, 2]).abs().max().item(),
    )

    return {
        "l0_error": err_l0,
        "l1_error": err_l1,
        "l2_error": err_l2,
        "gp_plus_stf_equals_cg": max(err_l0, err_l1, err_l2) < atol,
    }


test_gp_plus_stf_equals_cg.__test__ = False  # type: ignore[attr-defined]
