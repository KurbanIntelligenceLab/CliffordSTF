"""CliffordSTF Message Passing - Dual-Track Architecture.

Extends ``_algebra/interaction.py`` with:
  - STF_2 / STF_3 tracks propagating as equivariant features through MP.
  - Hodge dual force readout (grade-1 + *grade-2).
  - Per-edge STF_2 force decomposition (L=2 (x) L=1 -> L=1).
  - Environment-adaptive angular momentum routing.
  - Cross-track coupling via equivariant bilinear products only.

Backward compatible: ``stf_mode="none"`` + ``use_hodge_forces=False`` recovers
the original Clifford model.
"""

from __future__ import annotations

import logging
from typing import cast

import torch
from torch import nn
from torch.nn.functional import mse_loss
from torch_scatter import scatter

from cliffordstf.models._algebra.clifford import (
    ALL_GRADES,
    CliffordAlgebra,
    CliffordLinear,
    compute_gp_output_grades,
    compute_layer_grades,
    make_grades01_mv,
    make_scalar_mv,
)
from cliffordstf.models._algebra.interaction import (
    CliffordAttention,
    CosineCutoff,
    PerLayerEnergyReadout,
    RadialBasisFunctions,
)
from cliffordstf.models.stf import (
    CL_DIM,
    STF2_DIM,
    STF3_DIM,
    CliffordSTFAlgebra,
    CliffordSTFGateActivation,
    CliffordSTFLinear,
    CliffordSTFNorm,
    clifford_stf_dim,
    compute_stf2_product,
    compute_stf3_product,
    contract_stf2_vec,
    hodge_star_g2_to_vec,
    split_aug_mv,
    stf2_norm_sq,
    stf3_norm_sq,
)

_logger = logging.getLogger(__name__)


class CliffordSTFEdgeEmbedding(nn.Module):
    """Edge features with equivariant L=2 STF_2 track.

    Clifford track holds the original grade-(0, 1) multivectors. The STF_2 track
    holds ``direction (x) direction`` as a proper L=2 equivariant feature.
    """

    def __init__(
        self,
        n_rbf: int = 20,
        n_channels: int = 128,
        cutoff: float = 5.0,
        use_l2: bool = True,
        stf_mode: str = "stf2",
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.use_l2 = use_l2
        self.stf_mode = stf_mode
        self.rbf = RadialBasisFunctions(n_rbf, cutoff)
        self.cutoff_fn = CosineCutoff(cutoff)

        in_dim = n_rbf + (1 if use_l2 else 0)

        self.scalar_net = nn.Sequential(
            nn.Linear(in_dim, n_channels),
            nn.SiLU(),
            nn.Linear(n_channels, n_channels),
        )
        self.vector_net = nn.Sequential(
            nn.Linear(in_dim, n_channels),
            nn.SiLU(),
            nn.Linear(n_channels, n_channels),
        )

        if stf_mode != "none":
            self.stf2_net = nn.Sequential(
                nn.Linear(in_dim, n_channels),
                nn.SiLU(),
                nn.Linear(n_channels, n_channels),
            )

    def forward(self, dist: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        rbf = self.rbf(dist)
        env = self.cutoff_fn(dist)

        if self.use_l2:
            from cliffordstf.models._algebra.clifford import compute_l2_features

            l2 = compute_l2_features(direction)
            s_xx, s_xy, s_xz, s_yy, s_yz = l2.unbind(-1)
            s_zz = -s_xx - s_yy
            l2_norm_sq = s_xx**2 + s_yy**2 + s_zz**2 + 2.0 * (s_xy**2 + s_xz**2 + s_yz**2)
            l2_norm = torch.sqrt(l2_norm_sq.unsqueeze(-1) + 1e-8)
            feat = torch.cat([rbf, l2_norm], dim=-1)
        else:
            feat = rbf

        g0 = (self.scalar_net(feat) * env.unsqueeze(-1)).unsqueeze(-1)
        v_w = self.vector_net(feat) * env.unsqueeze(-1)
        g1 = v_w.unsqueeze(-1) * direction.unsqueeze(-2)

        cl_mv = make_grades01_mv(g0, g1)

        if self.stf_mode == "none":
            return cl_mv

        dir_stf2 = compute_stf2_product(direction, direction)
        stf2_w = self.stf2_net(feat) * env.unsqueeze(-1)
        stf2_feat = stf2_w.unsqueeze(-1) * dir_stf2.unsqueeze(-2)

        parts = [cl_mv, stf2_feat]

        if self.stf_mode == "stf2+stf3":
            stf3_feat = cl_mv.new_zeros(*cl_mv.shape[:-1], STF3_DIM)
            parts.append(stf3_feat)

        return torch.cat(parts, dim=-1)


class CliffordSTFAtomEmbedding(nn.Module):
    """Atomic numbers -> scalar-only augmented multivectors.

    STF tracks initialised to small noise (not zeros) for gradient flow.
    """

    def __init__(
        self,
        n_atom_types: int = 100,
        n_channels: int = 128,
        stf_mode: str = "stf2",
        init_noise: float = 1e-3,
    ) -> None:
        super().__init__()
        self.stf_mode = stf_mode
        self.init_noise = init_noise
        self.embed = nn.Embedding(n_atom_types, n_channels)

    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        s = self.embed(atomic_numbers)
        cl = make_scalar_mv(s)

        if self.stf_mode == "none":
            return cl

        d = clifford_stf_dim(self.stf_mode)
        pad_size = d - CL_DIM
        if self.training:
            padding = (
                torch.randn(*cl.shape[:-1], pad_size, device=cl.device, dtype=cl.dtype)
                * self.init_noise
            )
        else:
            padding = cl.new_zeros(*cl.shape[:-1], pad_size)

        return torch.cat([cl, padding], dim=-1)


class AdaptiveRouting(nn.Module):
    """Per-atom angular momentum routing.

    Modes:
        ``"none"``: all atoms use full augmented features.
        ``"static"``: atom-type lookup (H/C/N/O -> low, metals -> high).
        ``"learned"``: MLP from invariant local descriptors.

    Output: per-atom soft gate in ``[0, 1]`` for each STF track. Applied as a
    multiplicative mask (``torch.compile``-friendly).
    """

    def __init__(
        self,
        n_channels: int,
        mode: str = "none",
        stf_mode: str = "stf2",
        n_atom_types: int = 100,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.stf_mode = stf_mode
        self.n_tracks = 1 + (1 if stf_mode != "none" else 0) + (1 if stf_mode == "stf2+stf3" else 0)

        if mode == "static":
            n_gates = self.n_tracks - 1
            self.type_gates = nn.Embedding(n_atom_types, n_gates)
            nn.init.ones_(self.type_gates.weight)

        elif mode == "learned":
            n_gates = self.n_tracks - 1
            self.gate_net = nn.Sequential(
                nn.Linear(4, n_channels // 4),
                nn.SiLU(),
                nn.Linear(n_channels // 4, n_gates),
                nn.Sigmoid(),
            )

    def forward(
        self,
        aug: torch.Tensor,
        atomic_numbers: torch.Tensor | None = None,
        invariant_desc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mode == "none":
            return aug

        cl, s2, s3 = split_aug_mv(aug, self.stf_mode)

        if self.mode == "static":
            assert atomic_numbers is not None
            gates = torch.sigmoid(self.type_gates(atomic_numbers))
        elif self.mode == "learned":
            assert invariant_desc is not None
            gates = self.gate_net(invariant_desc)
        else:
            return aug

        parts = [cl]
        if s2 is not None:
            g2 = gates[:, 0:1].unsqueeze(-1)
            parts.append(s2 * g2)
        if s3 is not None:
            g3 = gates[:, 1:2].unsqueeze(-1)
            parts.append(s3 * g3)

        return torch.cat(parts, dim=-1)

    def precompute_geometric_invariants(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        dist: torch.Tensor,
    ) -> torch.Tensor:
        """Compute layer-independent portion of invariant descriptors.

        These terms depend only on positions / edges / distances, not on the
        per-layer features ``h``. Caching them and calling
        :meth:`combine_with_g0` inside the interaction loop avoids recomputing
        three scatter ops per layer.
        """
        src, dst = edge_index
        n = pos.shape[0]

        coord_num = cast(
            torch.Tensor,
            scatter(
                torch.ones_like(dst, dtype=torch.float),
                dst,
                dim=0,
                dim_size=n,
                reduce="sum",
            ),
        ).unsqueeze(-1)

        mean_dist = cast(
            torch.Tensor,
            scatter(dist, dst, dim=0, dim_size=n, reduce="mean"),
        ).unsqueeze(-1)

        direction = (pos[src] - pos[dst]) / (dist.unsqueeze(-1) + 1e-8)
        mean_dir = cast(
            torch.Tensor,
            scatter(direction, dst, dim=0, dim_size=n, reduce="mean"),
        )
        mean_dir_norm = torch.sqrt(torch.sum(mean_dir**2, dim=-1, keepdim=True) + 1e-8)
        angular_var = 1.0 - mean_dir_norm

        return torch.cat([coord_num, angular_var, mean_dist], dim=-1)

    def combine_with_g0(
        self,
        geometric_invariants: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Append per-layer ``g0_norm`` to cached geometric invariants."""
        g0_norm = torch.sqrt(torch.sum(h[..., 0] ** 2, dim=-1, keepdim=True) + 1e-8).mean(
            dim=-1, keepdim=True
        )
        return torch.cat([geometric_invariants, g0_norm], dim=-1)

    def compute_invariant_descriptors(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        dist: torch.Tensor,
        h: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Compute per-atom invariant descriptors for learned routing.

        Kept for backward compatibility. Prefer
        :meth:`precompute_geometric_invariants` + :meth:`combine_with_g0` in
        hot paths.
        """
        geom = self.precompute_geometric_invariants(pos, edge_index, dist)
        return self.combine_with_g0(geom, h)


class CliffordSTFMessageFunction(nn.Module):
    """Dual-track message with GP on Clifford + bilinear on STF tracks.

    Clifford track is the original (GP dispatch + attention/gating). STF_2 track
    is generated from grade-1 x edge direction (L=1 (x) L=1 -> L=2) and
    propagated from sender STF_2 via scalar modulation. Cross-track contraction
    STF_2 (x) grade-1 -> grade-1 feeds back into the Clifford message.
    """

    def __init__(
        self,
        n_channels: int,
        n_rbf: int,
        edge_grades: tuple[int, ...],
        node_input_grades: tuple[int, ...],
        gp_output_grades: tuple[int, ...],
        use_attention: bool = True,
        n_heads: int = 4,
        stf_mode: str = "stf2",
        use_cross_track: bool = True,
    ) -> None:
        super().__init__()
        self.alg = CliffordSTFAlgebra()
        self.n_channels = n_channels
        self.node_input_grades = node_input_grades
        self.edge_grades = edge_grades
        self.use_attention = use_attention
        self.stf_mode = stf_mode
        self.use_cross_track = use_cross_track

        self.pre_edge = CliffordLinear(
            n_channels, n_channels, bias=False, active_grades=edge_grades
        )
        self.proj_he = CliffordLinear(
            n_channels, n_channels, bias=False, active_grades=gp_output_grades
        )
        self.proj_eh = CliffordLinear(
            n_channels, n_channels, bias=False, active_grades=gp_output_grades
        )
        self.skip = nn.Linear(n_channels, n_channels, bias=False)

        if use_attention:
            self.attention = CliffordAttention(n_channels, n_heads, n_rbf)
        else:
            self.rbf = RadialBasisFunctions(n_rbf)
            self.radial_gate = nn.Sequential(
                nn.Linear(n_rbf, n_channels),
                nn.SiLU(),
                nn.Linear(n_channels, n_channels),
                nn.Sigmoid(),
            )

        if stf_mode != "none":
            self.stf2_proj_sender = nn.Linear(n_channels, n_channels, bias=False)
            self.stf2_proj_gen = nn.Linear(n_channels, n_channels, bias=False)
            self.stf2_radial = nn.Sequential(
                nn.Linear(n_rbf, n_channels),
                nn.SiLU(),
                nn.Linear(n_channels, n_channels),
            )

            if use_cross_track:
                self.cross_proj = nn.Linear(n_channels, n_channels, bias=False)

        if stf_mode == "stf2+stf3":
            self.stf3_proj_sender = nn.Linear(n_channels, n_channels, bias=False)
            self.stf3_proj_gen = nn.Linear(n_channels, n_channels, bias=False)

    def forward(
        self,
        h_j: torch.Tensor,
        h_i: torch.Tensor,
        edge_mv: torch.Tensor,
        dist: torch.Tensor,
        direction: torch.Tensor,
        dst: torch.Tensor,
        n_nodes: int,
        rbf_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_j_cl = h_j[..., :CL_DIM]
        h_i_cl = h_i[..., :CL_DIM]
        edge_cl = edge_mv[..., :CL_DIM]

        e = self.pre_edge(edge_cl)
        gp_he = self.alg.dispatch_gp(h_j_cl, e, self.node_input_grades, self.edge_grades)
        gp_eh = self.alg.dispatch_gp(e, h_j_cl, self.edge_grades, self.node_input_grades)

        skip_s = self.skip(h_j_cl[..., 0])
        skip_mv = make_scalar_mv(skip_s)

        msg_cl = (
            cast(torch.Tensor, self.proj_he(gp_he))
            + cast(torch.Tensor, self.proj_eh(gp_eh))
            + skip_mv
        )

        if self.stf_mode != "none" and self.use_cross_track:
            h_j_s2 = h_j[..., CL_DIM : CL_DIM + STF2_DIM]
            h_j_vec = h_j_cl[..., 1:4]

            cross_vec = contract_stf2_vec(h_j_s2, h_j_vec)
            cross_vec = self.cross_proj(cross_vec.transpose(-1, -2)).transpose(-1, -2)

            msg_cl[..., 1:4] = msg_cl[..., 1:4] + cross_vec

        if self.use_attention:
            attn = cast(
                torch.Tensor,
                self.attention(h_i_cl[..., 0], h_j_cl[..., 0], dist, dst, n_nodes),
            )
            msg_cl = msg_cl * attn
        else:
            gate = cast(torch.Tensor, self.radial_gate(self.rbf(dist))).unsqueeze(-1)
            msg_cl = msg_cl * gate

        if self.stf_mode == "none":
            return msg_cl

        assert rbf_feats is not None, (
            "rbf_feats required when stf_mode != 'none'. "
            "Pass pre-computed RBF features from the interaction block."
        )
        h_j_s2 = h_j[..., CL_DIM : CL_DIM + STF2_DIM]

        h_j_vec = h_j_cl[..., 1:4]
        dir_exp = direction.unsqueeze(-2).expand_as(h_j_vec)
        s2_gen = compute_stf2_product(h_j_vec, dir_exp)

        rbf_w = self.stf2_radial(rbf_feats)

        s2_gen_proj = torch.einsum("oc,...ci->...oi", self.stf2_proj_gen.weight, s2_gen)
        s2_send_proj = torch.einsum("oc,...ci->...oi", self.stf2_proj_sender.weight, h_j_s2)
        msg_s2 = (s2_gen_proj + s2_send_proj) * rbf_w.unsqueeze(-1)

        parts = [msg_cl, msg_s2]

        if self.stf_mode == "stf2+stf3":
            h_j_s3 = h_j[..., CL_DIM + STF2_DIM :]

            s3_gen = compute_stf3_product(h_j_s2, dir_exp)
            s3_gen_proj = torch.einsum("oc,...ci->...oi", self.stf3_proj_gen.weight, s3_gen)
            s3_send_proj = torch.einsum("oc,...ci->...oi", self.stf3_proj_sender.weight, h_j_s3)
            msg_s3 = (s3_gen_proj + s3_send_proj) * rbf_w.unsqueeze(-1)
            parts.append(msg_s3)

        return torch.cat(parts, dim=-1)


class CliffordSTFMultiBodyInteraction(nn.Module):
    """Multi-body with STF generation from iterated products.

    2-body: ``W2 . agg``.
    3-body: ``GP(agg, agg)`` on Clifford + STF_2 from grade-1 x grade-1.
    4-body: ``GP(3body, agg)`` + STF_3 from STF_2 x grade-1.
    """

    def __init__(
        self,
        n_channels: int,
        active_grades: tuple[int, ...] = ALL_GRADES,
        max_body_order: int = 3,
        stf_mode: str = "stf2",
    ) -> None:
        super().__init__()
        self.alg = CliffordSTFAlgebra()
        self.max_body_order = max_body_order
        self.stf_mode = stf_mode

        self.w2 = CliffordSTFLinear(
            n_channels,
            n_channels,
            bias=False,
            active_grades=active_grades,
            stf_mode=stf_mode,
        )
        self.w3 = CliffordSTFLinear(
            n_channels,
            n_channels,
            bias=False,
            active_grades=active_grades,
            stf_mode=stf_mode,
        )
        if max_body_order >= 4:
            self.w4 = CliffordSTFLinear(
                n_channels,
                n_channels,
                bias=False,
                active_grades=active_grades,
                stf_mode=stf_mode,
            )

    def forward(self, agg: torch.Tensor) -> torch.Tensor:
        out = cast(torch.Tensor, self.w2(agg))

        if self.max_body_order >= 3:
            if self.stf_mode == "none":
                agg_cl = agg
                three_body_cl = self.alg.geometric_product(agg_cl, agg_cl)
                three_body = three_body_cl
            else:
                agg_cl = agg
                three_body = self.alg.augmented_product(agg, agg, stf_mode=self.stf_mode)
            out = out + self.w3(three_body)

            if self.max_body_order >= 4:
                if self.stf_mode == "none":
                    four_body = self.alg.geometric_product(three_body, agg_cl)
                else:
                    four_body = self.alg.augmented_product(three_body, agg, stf_mode=self.stf_mode)
                out = out + self.w4(four_body)

        return out


class CliffordSTFSelfInteraction(nn.Module):
    """Per-node self-interaction via augmented product."""

    def __init__(
        self,
        n_channels: int,
        active_grades: tuple[int, ...] = ALL_GRADES,
        stf_mode: str = "stf2",
    ) -> None:
        super().__init__()
        self.alg = CliffordSTFAlgebra()
        self.stf_mode = stf_mode
        self.proj = CliffordSTFLinear(
            n_channels,
            n_channels,
            bias=False,
            active_grades=active_grades,
            stf_mode=stf_mode,
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_proj = self.proj(h)
        if self.stf_mode == "none":
            return self.alg.geometric_product(h_proj, h)
        return self.alg.augmented_product(h_proj, h, stf_mode=self.stf_mode)


class CliffordSTFUpdateFunction(nn.Module):
    """Node update with augmented features + residual."""

    def __init__(
        self,
        n_channels: int,
        input_grades: tuple[int, ...],
        output_grades: tuple[int, ...],
        use_self_interaction: bool = True,
        stf_mode: str = "stf2",
    ) -> None:
        super().__init__()
        self.use_self_interaction = use_self_interaction
        self.stf_mode = stf_mode

        cat_mult = 3 if use_self_interaction else 2
        self.linear_in = CliffordSTFLinear(
            cat_mult * n_channels,
            n_channels,
            bias=True,
            active_grades=output_grades,
            stf_mode=stf_mode,
        )
        self.activation = CliffordSTFGateActivation(
            n_channels,
            active_grades=output_grades,
            stf_mode=stf_mode,
        )
        self.linear_out = CliffordSTFLinear(
            n_channels,
            n_channels,
            bias=True,
            active_grades=output_grades,
            stf_mode=stf_mode,
        )
        self.norm = CliffordSTFNorm(
            n_channels,
            active_grades=output_grades,
            stf_mode=stf_mode,
        )

        if use_self_interaction:
            self.self_int = CliffordSTFSelfInteraction(
                n_channels,
                active_grades=output_grades,
                stf_mode=stf_mode,
            )

    def forward(self, h: torch.Tensor, agg: torch.Tensor) -> torch.Tensor:
        if self.use_self_interaction:
            si = self.self_int(h)
            combined = torch.cat([h, agg, si], dim=-2)
        else:
            combined = torch.cat([h, agg], dim=-2)

        out = self.linear_in(combined)
        out = self.activation(out)
        out = self.linear_out(out)

        if h.shape[-2] == out.shape[-2]:
            return cast(torch.Tensor, self.norm(out + h))
        return cast(torch.Tensor, self.norm(out))


class CliffordSTFInteractionBlock(nn.Module):
    """Single augmented Clifford MP layer."""

    def __init__(
        self,
        n_channels: int,
        n_rbf: int,
        edge_grades: tuple[int, ...],
        node_input_grades: tuple[int, ...],
        node_output_grades: tuple[int, ...],
        use_attention: bool = True,
        use_self_interaction: bool = True,
        max_body_order: int = 3,
        n_heads: int = 4,
        stf_mode: str = "stf2",
        use_cross_track: bool = True,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.stf_mode = stf_mode

        if stf_mode != "none":
            self.shared_rbf = RadialBasisFunctions(n_rbf)

        gp_out = compute_gp_output_grades(node_input_grades, edge_grades)

        self.message_fn = CliffordSTFMessageFunction(
            n_channels,
            n_rbf,
            edge_grades=edge_grades,
            node_input_grades=node_input_grades,
            gp_output_grades=gp_out,
            use_attention=use_attention,
            n_heads=n_heads,
            stf_mode=stf_mode,
            use_cross_track=use_cross_track,
        )

        union_grades = tuple(sorted(set(node_input_grades) | set(gp_out)))

        self.multi_body = CliffordSTFMultiBodyInteraction(
            n_channels,
            active_grades=union_grades,
            max_body_order=max_body_order,
            stf_mode=stf_mode,
        )

        self.update_fn = CliffordSTFUpdateFunction(
            n_channels,
            input_grades=union_grades,
            output_grades=node_output_grades,
            use_self_interaction=use_self_interaction,
            stf_mode=stf_mode,
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mv: torch.Tensor,
        dist: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        h_j = h[src]
        h_i = h[dst]
        n = h.shape[0]

        rbf_feats = self.shared_rbf(dist) if self.stf_mode != "none" else None

        messages = self.message_fn(
            h_j,
            h_i,
            edge_mv,
            dist,
            direction,
            dst,
            n,
            rbf_feats=rbf_feats,
        )

        agg = cast(
            torch.Tensor,
            scatter(messages, dst, dim=0, dim_size=n, reduce="sum"),
        )

        agg = self.multi_body(agg)

        return cast(torch.Tensor, self.update_fn(h, agg))


class CliffordSTFOutputBlock(nn.Module):
    """Energy/force output with Hodge dual and STF_2 force decomposition.

    Energy: grade-0 scalars + GP readout + STF_2 norm invariants + STF_3 norm
    invariants.

    Forces: grade-1 vectors + *grade-2 (Hodge dual) + per-edge STF_2 . r-hat
    contraction.

    If ``skip_clifford_output=True`` the Clifford-track contribution is removed
    from both energy (no grade-0 scalars, no GP readout) and forces (no grade-1
    vectors, no Hodge dual). Only STF invariants and the per-edge STF_2 force
    decomposition drive the output. Requires ``stf_mode != 'none'`` because
    otherwise the output head would have no inputs.
    """

    def __init__(
        self,
        n_channels: int = 128,
        n_hidden: int = 64,
        use_gp_readout: bool = True,
        use_hodge_forces: bool = True,
        stf_mode: str = "stf2",
        skip_clifford_output: bool = False,
    ) -> None:
        super().__init__()
        if skip_clifford_output and stf_mode == "none":
            raise ValueError(
                "skip_clifford_output=True requires stf_mode != 'none': "
                "with no Clifford track AND no STF track, there is nothing to "
                "read out."
            )
        self.use_gp_readout = use_gp_readout
        self.use_hodge_forces = use_hodge_forces
        self.stf_mode = stf_mode
        self.skip_clifford_output = skip_clifford_output
        self.alg = CliffordAlgebra()

        if not skip_clifford_output:
            self.pre_mix = CliffordLinear(
                n_channels, n_channels, bias=False, active_grades=ALL_GRADES
            )

        energy_in = (
            0 if skip_clifford_output else (2 * n_channels if use_gp_readout else n_channels)
        )
        if stf_mode != "none":
            energy_in += n_channels
        if stf_mode == "stf2+stf3":
            energy_in += n_channels

        self.energy_head = nn.Sequential(
            nn.Linear(energy_in, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 1),
        )

        if not skip_clifford_output:
            force_vec_channels = 2 * n_channels if use_hodge_forces else n_channels
            self.force_head = nn.Linear(force_vec_channels, 1, bias=False)

        if stf_mode != "none":
            self.stf2_force_head = nn.Sequential(
                nn.Linear(n_channels, n_hidden),
                nn.SiLU(),
                nn.Linear(n_hidden, 1),
            )

    def forward(
        self,
        h: torch.Tensor,
        batch: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        dist: torch.Tensor | None = None,
        direction: torch.Tensor | None = None,
        num_graphs: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_cl = h[..., :CL_DIM]

        if self.skip_clifford_output:
            scalars = h_cl.new_empty(*h_cl.shape[:-2], 0)
        elif self.use_gp_readout:
            h_pre = self.pre_mix(h_cl)
            h_mixed = self.alg.geometric_product(h_pre, h_cl)
            scalars = torch.cat([h_cl[..., 0], h_mixed[..., 0]], dim=-1)
        else:
            scalars = h_cl[..., 0]

        s2_norm_sq_cached: torch.Tensor | None = None
        if self.stf_mode != "none":
            h_s2 = h[..., CL_DIM : CL_DIM + STF2_DIM]
            s2_norm_sq_cached = stf2_norm_sq(h_s2)
            s2_inv = torch.sqrt(s2_norm_sq_cached + 1e-8).squeeze(-1)
            scalars = torch.cat([scalars, s2_inv], dim=-1)

        if self.stf_mode == "stf2+stf3":
            h_s3 = h[..., CL_DIM + STF2_DIM :]
            s3_inv = torch.sqrt(stf3_norm_sq(h_s3) + 1e-8).squeeze(-1)
            scalars = torch.cat([scalars, s3_inv], dim=-1)

        atom_energy = self.energy_head(scalars).squeeze(-1)

        if batch is not None:
            if num_graphs is None:
                num_graphs = int(batch.max().item()) + 1
            energy = cast(
                torch.Tensor,
                scatter(atom_energy, batch, dim=0, dim_size=num_graphs, reduce="sum"),
            )
        else:
            energy = atom_energy.sum(dim=0, keepdim=True)

        if self.skip_clifford_output:
            forces = h_cl.new_zeros(h_cl.shape[0], 3)
        else:
            v_force = h_cl[..., 1:4]

            if self.use_hodge_forces:
                pv_force = hodge_star_g2_to_vec(h_cl)
                combined_v = torch.cat([v_force, pv_force], dim=-2)
            else:
                combined_v = v_force

            forces = self.force_head(combined_v.transpose(-1, -2)).squeeze(-1)

        if self.stf_mode != "none" and edge_index is not None and direction is not None:
            assert s2_norm_sq_cached is not None
            src, dst = edge_index
            h_s2_src = h[src, :, CL_DIM : CL_DIM + STF2_DIM]
            h_s2_dst = h[dst, :, CL_DIM : CL_DIM + STF2_DIM]

            dir_exp = direction.unsqueeze(-2).expand_as(h_s2_src[..., :3])
            f_s2_src = contract_stf2_vec(h_s2_src, dir_exp)
            f_s2_dst = contract_stf2_vec(h_s2_dst, dir_exp)

            s2_inv_edge = torch.sqrt(s2_norm_sq_cached[src].squeeze(-1) + 1e-8)
            s2_weight = self.stf2_force_head(s2_inv_edge).unsqueeze(-1)

            edge_force = s2_weight * (f_s2_src + f_s2_dst)
            edge_force = edge_force.mean(dim=-2)
            stf2_forces = cast(
                torch.Tensor,
                scatter(edge_force, dst, dim=0, dim_size=h.shape[0], reduce="sum"),
            )
            forces = forces + stf2_forces

        return energy, forces


class CliffordSTF(nn.Module):
    """CliffordSTF GNN - Cl(3,0) + STF_2 + STF_3.

    Ablation flags:
        ``stf_mode``: ``"none"`` / ``"stf2"`` / ``"stf2+stf3"``.
        ``use_hodge_forces``: grade-1 + *grade-2 force readout.
        ``use_adaptive_routing``: per-atom L_max selection.
        ``routing_mode``: ``"none"`` / ``"static"`` / ``"learned"``.
        ``use_cross_track``: Clifford <-> STF bilinear coupling.
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
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.direct_forces = direct_forces
        self.n_channels = n_channels
        self.use_multiscale = use_multiscale
        self.use_dens = use_dens
        self.dens_noise_std = dens_noise_std
        self.stf_mode = stf_mode
        self.use_adaptive_routing = use_adaptive_routing
        self.routing_mode = routing_mode
        self.skip_clifford_output = skip_clifford_output

        self.atom_embed = CliffordSTFAtomEmbedding(n_atom_types, n_channels, stf_mode=stf_mode)
        self.edge_embed = CliffordSTFEdgeEmbedding(
            n_rbf, n_channels, cutoff, use_l2=use_l2, stf_mode=stf_mode
        )

        edge_grades = (0, 1)
        layer_output_grades = compute_layer_grades(n_interactions, edge_grades)

        self.interactions = nn.ModuleList()
        node_grades: tuple[int, ...] = (0,)
        for i in range(n_interactions):
            out_grades = layer_output_grades[i]
            self.interactions.append(
                CliffordSTFInteractionBlock(
                    n_channels,
                    n_rbf,
                    edge_grades=edge_grades,
                    node_input_grades=node_grades,
                    node_output_grades=out_grades,
                    use_attention=use_attention,
                    use_self_interaction=use_self_interaction,
                    max_body_order=max_body_order,
                    n_heads=n_heads,
                    stf_mode=stf_mode,
                    use_cross_track=use_cross_track,
                )
            )
            node_grades = out_grades

        if use_multiscale:
            self.layer_readouts = nn.ModuleList(
                [PerLayerEnergyReadout(n_channels, n_hidden_output) for _ in range(n_interactions)]
            )

        self.output = CliffordSTFOutputBlock(
            n_channels,
            n_hidden_output,
            use_gp_readout=use_gp_readout,
            use_hodge_forces=use_hodge_forces,
            stf_mode=stf_mode,
            skip_clifford_output=skip_clifford_output,
        )

        if use_adaptive_routing:
            self.router = AdaptiveRouting(
                n_channels,
                mode=routing_mode,
                stf_mode=stf_mode,
                n_atom_types=n_atom_types,
            )

        self._grade_schedule = layer_output_grades

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.direct_forces:
            pos = pos.clone().requires_grad_(True)

        src, dst = edge_index
        rel_pos = pos[dst] - pos[src]
        dist = torch.sqrt(torch.sum(rel_pos**2, dim=-1) + 1e-8)
        direction = rel_pos / (dist.unsqueeze(-1) + 1e-8)

        h = self.atom_embed(atomic_numbers)
        edge_mv = self.edge_embed(dist, direction)

        num_graphs: int | None = None
        if batch is not None:
            num_graphs = int(batch.max().item()) + 1

        geometric_invariants_cache: torch.Tensor | None = None
        if self.use_adaptive_routing and self.routing_mode == "learned":
            geometric_invariants_cache = self.router.precompute_geometric_invariants(
                pos, edge_index, dist
            )

        layer_energies: list[torch.Tensor] = []
        for i, interaction in enumerate(self.interactions):
            h = interaction(h, edge_index, edge_mv, dist, direction)

            if self.use_adaptive_routing:
                if self.routing_mode == "static":
                    h = self.router(h, atomic_numbers=atomic_numbers)
                elif self.routing_mode == "learned":
                    assert geometric_invariants_cache is not None
                    inv_desc = self.router.combine_with_g0(
                        geometric_invariants_cache, h[..., :CL_DIM]
                    )
                    h = self.router(h, invariant_desc=inv_desc)

            if self.use_multiscale:
                layer_energies.append(
                    self.layer_readouts[i](h[..., :CL_DIM], batch, num_graphs=num_graphs)
                )

        energy, forces = self.output(h, batch, edge_index, dist, direction, num_graphs=num_graphs)

        if self.use_multiscale and layer_energies:
            for le in layer_energies:
                energy = energy + le

        if not self.direct_forces:
            forces = -torch.autograd.grad(
                energy.sum(),
                pos,
                create_graph=self.training,
                retain_graph=self.training,
            )[0]

        return energy, forces

    def forward_with_dens(
        self,
        atomic_numbers: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward with DeNS denoising auxiliary loss."""
        energy, forces = self.forward(atomic_numbers, pos, edge_index, batch)

        dens_loss = torch.tensor(0.0, device=pos.device)
        if self.training and self.use_dens:
            noise = torch.randn_like(pos) * self.dens_noise_std
            pos_noisy = pos + noise

            rel_pos_n = pos_noisy[edge_index[1]] - pos_noisy[edge_index[0]]
            dist_n = torch.sqrt(torch.sum(rel_pos_n**2, dim=-1) + 1e-8)
            direction_n = rel_pos_n / (dist_n.unsqueeze(-1) + 1e-8)

            h_n = self.atom_embed(atomic_numbers)
            edge_mv_n = self.edge_embed(dist_n, direction_n)

            for interaction in self.interactions:
                h_n = interaction(h_n, edge_index, edge_mv_n, dist_n, direction_n)

            h_cl_n = h_n[..., :CL_DIM]
            noise_pred = h_cl_n[..., 1:4].mean(dim=-2)

            if self.use_hodge_forces:
                hodge_pred = hodge_star_g2_to_vec(h_cl_n).mean(dim=-2)
                noise_pred = noise_pred + hodge_pred

            dens_loss = mse_loss(noise_pred, -noise)

        return energy, forces, dens_loss

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def use_hodge_forces(self) -> bool:
        return self.output.use_hodge_forces

    def print_config(self) -> None:
        _logger.info("CliffordSTF GNN:")
        _logger.info("  stf_mode:       %s", self.stf_mode)
        _logger.info("  hodge_forces:   %s", self.use_hodge_forces)
        _logger.info(
            "  adaptive_route: %s (%s)",
            self.use_adaptive_routing,
            self.routing_mode,
        )
        _logger.info("  clifford_stf_dim: %d", clifford_stf_dim(self.stf_mode))
        _logger.info("  parameters:     %d", self.num_params)
        _logger.info("  Grade schedule:")
        _logger.info("    Edge:  (0, 1)")
        _logger.info("    Atoms: (0,)")
        for i, g in enumerate(self._grade_schedule):
            _logger.info("    Layer %d: %s", i, g)
