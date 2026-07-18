"""Tests for ``cliffordstf.training.evaluate``."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from cliffordstf.training.evaluate import evaluate_epoch


class _ConstModel(nn.Module):
    """``data_wrapper``-style model returning fixed energy/forces.

    Used to verify the metric arithmetic without exercising a full model.
    """

    def __init__(self, energy_value: float, force_value: float) -> None:
        super().__init__()
        self.energy_value = energy_value
        self.force_value = force_value
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        n_graphs = int(data.batch.max().item()) + 1
        energy = torch.full((n_graphs,), self.energy_value, device=data.pos.device)
        forces = torch.full_like(data.pos, self.force_value)
        return energy + 0.0 * self._dummy.sum(), forces


def _make_loader(n_atoms: int = 4) -> DataLoader:
    torch.manual_seed(0)
    data = Data(
        z=torch.randint(1, 10, (n_atoms,)),
        pos=torch.randn(n_atoms, 3),
        batch=torch.zeros(n_atoms, dtype=torch.long),
        energy=torch.tensor([1.0]),
        force=torch.zeros(n_atoms, 3),
    )
    return DataLoader([data], batch_size=1, shuffle=False)


def test_evaluate_epoch_returns_expected_metric_keys():
    model = _ConstModel(energy_value=1.0, force_value=0.0)
    loader = _make_loader()
    cfg = OmegaConf.create({"dataset": {}})
    metrics = evaluate_epoch(model, loader, cfg, torch.device("cpu"))
    assert set(metrics.keys()) == {"energy_mae", "force_mae", "force_cos", "efwt"}


def test_evaluate_epoch_zero_error_perfect_prediction():
    model = _ConstModel(energy_value=1.0, force_value=0.0)
    loader = _make_loader()
    cfg = OmegaConf.create({"dataset": {}})
    metrics = evaluate_epoch(model, loader, cfg, torch.device("cpu"))
    assert metrics["energy_mae"] == 0.0
    assert metrics["force_mae"] == 0.0
    assert metrics["efwt"] == 100.0


def test_evaluate_epoch_force_mae_with_known_offset():
    model = _ConstModel(energy_value=1.0, force_value=0.5)
    loader = _make_loader()
    cfg = OmegaConf.create({"dataset": {}})
    metrics = evaluate_epoch(model, loader, cfg, torch.device("cpu"))
    assert metrics["force_mae"] == 0.5


def test_evaluate_epoch_rescales_energy_mae_by_runtime_std():
    model = _ConstModel(energy_value=3.0, force_value=0.0)
    loader = _make_loader()
    cfg = OmegaConf.create({"dataset": {}})
    metrics = evaluate_epoch(
        model,
        loader,
        cfg,
        torch.device("cpu"),
        runtime_stats={"std": 4.0, "mean": 0.0},
    )
    assert metrics["energy_mae"] == 2.0 * 4.0


class _ScalarModel(nn.Module):
    """Energy-only model for scalar-task evaluation."""

    def __init__(self, energy_value: float) -> None:
        super().__init__()
        self.energy_value = energy_value

    def forward(self, data: Data) -> torch.Tensor:
        n_graphs = int(data.batch.max().item()) + 1
        return torch.full((n_graphs,), self.energy_value, device=data.pos.device)


def _make_scalar_loader() -> DataLoader:
    data = Data(
        z=torch.tensor([1, 6]),
        pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        batch=torch.zeros(2, dtype=torch.long),
        energy=torch.tensor([1.0]),
    )
    return DataLoader([data], batch_size=1, shuffle=False)


def test_evaluate_epoch_scalar_returns_only_energy_mae():
    model = _ScalarModel(energy_value=1.5)
    loader = _make_scalar_loader()
    cfg = OmegaConf.create({"dataset": {"task_type": "scalar"}})
    metrics = evaluate_epoch(model, loader, cfg, torch.device("cpu"))
    assert set(metrics.keys()) == {"energy_mae"}
    assert abs(metrics["energy_mae"] - 0.5) < 1e-7


def test_evaluate_epoch_scalar_rescales_via_runtime_stats():
    model = _ScalarModel(energy_value=1.0)
    loader = _make_scalar_loader()
    cfg = OmegaConf.create({"dataset": {"task_type": "scalar"}})
    raw = evaluate_epoch(model, loader, cfg, torch.device("cpu"))
    rescaled = evaluate_epoch(
        model,
        loader,
        cfg,
        torch.device("cpu"),
        runtime_stats={"std": torch.tensor(3.0)},
    )
    assert abs(rescaled["energy_mae"] - raw["energy_mae"] * 3.0) < 1e-7


def test_evaluate_epoch_is2re_alias_triggers_scalar_branch():
    """is2re task_type should route through the energy-only branch."""
    model = _ScalarModel(energy_value=1.5)
    loader = _make_scalar_loader()
    cfg = OmegaConf.create({"dataset": {"task_type": "is2re"}})
    metrics = evaluate_epoch(model, loader, cfg, torch.device("cpu"))
    assert set(metrics.keys()) == {"energy_mae"}
