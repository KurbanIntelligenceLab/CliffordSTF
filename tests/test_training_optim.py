"""Tests for ``cliffordstf.training.optim``."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
)

from cliffordstf.training.optim import build_optimizer, build_scheduler


def _model() -> nn.Module:
    return nn.Linear(4, 3)


def test_build_optimizer_defaults_to_adam():
    cfg = OmegaConf.create({"training": {"lr": 1e-3}})
    opt = build_optimizer(cfg, _model())
    assert isinstance(opt, torch.optim.Adam)
    assert opt.param_groups[0]["lr"] == 1e-3
    assert opt.param_groups[0]["weight_decay"] == 0.0


def test_build_optimizer_adamw_with_weight_decay():
    cfg = OmegaConf.create({"training": {"lr": 5e-4, "weight_decay": 1e-2, "optimizer": "adamw"}})
    opt = build_optimizer(cfg, _model())
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == 5e-4
    assert opt.param_groups[0]["weight_decay"] == 1e-2


def test_build_optimizer_unknown_raises():
    cfg = OmegaConf.create({"training": {"lr": 1e-3, "optimizer": "sgd"}})
    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_optimizer(cfg, _model())


def test_build_scheduler_returns_none_when_unset():
    cfg = OmegaConf.create({"training": {"lr": 1e-3, "epochs": 10}})
    opt = build_optimizer(cfg, _model())
    assert build_scheduler(cfg, opt) is None


def test_build_scheduler_step():
    cfg = OmegaConf.create(
        {
            "training": {
                "lr": 1e-3,
                "epochs": 10,
                "lr_scheduler": {"type": "step", "step_size": 3, "gamma": 0.5},
            }
        }
    )
    opt = build_optimizer(cfg, _model())
    sched = build_scheduler(cfg, opt)
    assert isinstance(sched, StepLR)
    assert sched.step_size == 3
    assert sched.gamma == 0.5


def test_build_scheduler_cosine_uses_epochs_as_default_t_max():
    cfg = OmegaConf.create(
        {"training": {"lr": 1e-3, "epochs": 25, "lr_scheduler": {"type": "cosine"}}}
    )
    opt = build_optimizer(cfg, _model())
    sched = build_scheduler(cfg, opt)
    assert isinstance(sched, CosineAnnealingLR)
    assert sched.T_max == 25


def test_build_scheduler_plateau():
    cfg = OmegaConf.create(
        {
            "training": {
                "lr": 1e-3,
                "epochs": 10,
                "lr_scheduler": {"type": "plateau", "patience": 4, "factor": 0.25},
            }
        }
    )
    opt = build_optimizer(cfg, _model())
    sched = build_scheduler(cfg, opt)
    assert isinstance(sched, ReduceLROnPlateau)
    assert sched.patience == 4
    assert sched.factor == 0.25


def test_build_scheduler_unknown_raises():
    cfg = OmegaConf.create(
        {
            "training": {
                "lr": 1e-3,
                "epochs": 10,
                "lr_scheduler": {"type": "weird"},
            }
        }
    )
    opt = build_optimizer(cfg, _model())
    with pytest.raises(ValueError, match="Unknown scheduler type"):
        build_scheduler(cfg, opt)
