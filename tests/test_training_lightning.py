"""Tests for ``CliffordSTFLightningModule`` (Phase-2 Step 18)."""

from __future__ import annotations

from typing import cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch_geometric.data import Batch, Data

from cliffordstf.training.lightning import CliffordSTFLightningModule


class _TinyEnergyForcesModel(nn.Module):
    """Minimal data_wrapper-style model: per-graph energy + autograd forces."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1, bias=False)

    def forward(self, data: Data) -> torch.Tensor:
        per_atom = self.linear(data.pos).squeeze(-1)
        return torch.zeros(
            int(data.batch.max().item()) + 1, device=data.pos.device, dtype=data.pos.dtype
        ).index_add_(0, data.batch, per_atom)


def _tiny_cfg(loss: str = "mse", lr_scheduler: dict[str, object] | None = None) -> DictConfig:
    base: dict[str, object] = {
        "training": {
            "lr": 1e-3,
            "optimizer": "adam",
            "weight_decay": 0.0,
            "loss": loss,
            "lr_scheduler": lr_scheduler,
        },
        "dataset": {
            "task_type": "energy_forces",
            "energy_weight": 1.0,
            "force_weight": 1.0,
        },
    }
    return cast(DictConfig, OmegaConf.create(base))


def _tiny_batch() -> Batch:
    g1 = Data(
        z=torch.tensor([1, 6], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32),
        energy=torch.tensor([0.5]),
        force=torch.tensor([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=torch.float32),
    )
    g2 = Data(
        z=torch.tensor([1, 8, 1], dtype=torch.long),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ),
        energy=torch.tensor([-0.3]),
        force=torch.tensor(
            [[0.0, 0.1, 0.0], [0.0, -0.05, 0.0], [0.0, -0.05, 0.0]], dtype=torch.float32
        ),
    )
    return Batch.from_data_list([g1, g2])


def test_training_step_returns_finite_loss() -> None:
    module = CliffordSTFLightningModule(_TinyEnergyForcesModel(), _tiny_cfg())
    module.train()
    loss = module.training_step(_tiny_batch(), batch_idx=0)
    assert torch.isfinite(loss).item()


def test_validation_step_returns_finite_loss() -> None:
    module = CliffordSTFLightningModule(_TinyEnergyForcesModel(), _tiny_cfg())
    module.eval()
    loss = module.validation_step(_tiny_batch(), batch_idx=0)
    assert torch.isfinite(loss).item()


def test_configure_optimizers_without_scheduler() -> None:
    module = CliffordSTFLightningModule(_TinyEnergyForcesModel(), _tiny_cfg())
    config = module.configure_optimizers()
    assert "optimizer" in config
    assert "lr_scheduler" not in config


def test_configure_optimizers_with_scheduler() -> None:
    cfg = _tiny_cfg(lr_scheduler={"type": "cosine", "T_max": 10})
    # ``build_scheduler``'s eager default evaluation reads cfg.training.epochs
    # even when ``T_max`` is explicit; pin it so the test stays self-contained.
    cfg.training.epochs = 10
    module = CliffordSTFLightningModule(_TinyEnergyForcesModel(), cfg)
    config = module.configure_optimizers()
    assert "optimizer" in config
    assert "lr_scheduler" in config


def test_forward_delegates_to_wrapped_model() -> None:
    model = _TinyEnergyForcesModel()
    module = CliffordSTFLightningModule(model, _tiny_cfg())
    batch = _tiny_batch()
    out = module.forward(batch)
    expected = model(batch)
    assert torch.allclose(out, expected)
