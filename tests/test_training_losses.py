"""Tests for ``cliffordstf.training.losses``."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from cliffordstf.training.losses import (
    compute_forces,
    compute_loss,
    forward_model,
    get_free_atom_mask,
    l2norm_loss,
    per_atom_mae_loss,
)


def _make_data(n_atoms: int = 4) -> Data:
    torch.manual_seed(0)
    return Data(
        z=torch.randint(1, 30, (n_atoms,)),
        pos=torch.randn(n_atoms, 3, requires_grad=True) * 2.0,
        batch=torch.zeros(n_atoms, dtype=torch.long),
        energy=torch.randn(1),
        force=torch.randn(n_atoms, 3),
    )


def test_per_atom_mae_loss_matches_manual():
    pred = torch.tensor([2.0, 6.0])
    target = torch.tensor([1.0, 4.0])
    natoms = torch.tensor([2.0, 4.0])
    expected = ((pred / natoms - target / natoms).abs()).mean()
    assert torch.allclose(per_atom_mae_loss(pred, target, natoms), expected)


def test_l2norm_loss_matches_manual():
    pred = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    target = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    expected = torch.tensor((1.0 + 2.0) / 2.0)
    assert torch.allclose(l2norm_loss(pred, target), expected)


def test_get_free_atom_mask_uses_fixed_attribute():
    data = _make_data()
    data.fixed = torch.tensor([True, False, True, False])
    mask = get_free_atom_mask(data)
    assert mask is not None
    assert torch.equal(mask, torch.tensor([False, True, False, True]))


def test_get_free_atom_mask_falls_back_to_tags():
    data = _make_data()
    data.tags = torch.tensor([0, 1, 2, 0])
    mask = get_free_atom_mask(data)
    assert mask is not None
    assert torch.equal(mask, torch.tensor([False, True, True, False]))


def test_get_free_atom_mask_returns_none_when_unavailable():
    data = _make_data()
    assert get_free_atom_mask(data) is None


def test_compute_forces_via_autograd_recovers_negative_gradient():
    pos = torch.randn(3, 3, requires_grad=True)
    energy = (pos**2).sum().unsqueeze(0)
    forces = compute_forces(energy, pos, create_graph=False)
    expected = -2.0 * pos
    assert torch.allclose(forces, expected, atol=1e-5)


class _DirectForcesModel(torch.nn.Module):
    """Minimal ``data_wrapper``-style model that emits direct forces."""

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        energy = data.pos.sum().unsqueeze(0)
        forces = torch.ones_like(data.pos)
        return energy, forces


def test_forward_model_returns_tuple_for_direct_forces():
    model = _DirectForcesModel()
    data = _make_data()
    out = forward_model(model, data)
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0].shape == (1,)
    assert out[1].shape == data.pos.shape


def test_compute_loss_combines_energy_and_force_components():
    model = _DirectForcesModel()
    data = _make_data()
    cfg = OmegaConf.create({"training": {"loss": "mse"}, "dataset": {}})
    loss = compute_loss(model, data, cfg, training=False)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_compute_loss_respects_weights():
    model = _DirectForcesModel()
    data = _make_data()
    cfg_high_e = OmegaConf.create(
        {
            "training": {"loss": "mse"},
            "dataset": {"energy_weight": 100.0, "force_weight": 0.0},
        }
    )
    cfg_high_f = OmegaConf.create(
        {
            "training": {"loss": "mse"},
            "dataset": {"energy_weight": 0.0, "force_weight": 100.0},
        }
    )
    loss_e = compute_loss(model, data, cfg_high_e, training=False)
    loss_f = compute_loss(model, _make_data(), cfg_high_f, training=False)
    assert not torch.allclose(loss_e, loss_f)
