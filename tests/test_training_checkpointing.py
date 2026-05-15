"""Tests for ``cliffordstf.training.checkpointing``."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from cliffordstf.training.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    save_logs,
    wait_for_checkpoint,
)


def test_save_and_load_checkpoint_round_trips_model_state(tmp_path: Path):
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    cfg = OmegaConf.create({"foo": "bar"})

    path = tmp_path / "ckpt.pth"
    save_checkpoint(
        path,
        model,
        optimizer=optimizer,
        epoch=7,
        cfg=cfg,
        training_state={"best_val": 0.5},
    )
    wait_for_checkpoint()
    assert path.exists()

    new_model = nn.Linear(4, 3)
    new_opt = torch.optim.Adam(new_model.parameters(), lr=1e-3)
    epoch, training_state = load_checkpoint(path, new_model, new_opt)
    assert epoch == 7
    assert training_state == {"best_val": 0.5}
    for p_old, p_new in zip(model.parameters(), new_model.parameters(), strict=True):
        assert torch.equal(p_old.data, p_new.data)


def test_load_checkpoint_returns_zero_when_epoch_missing(tmp_path: Path):
    model = nn.Linear(4, 3)
    path = tmp_path / "minimal.pth"
    torch.save({"model_state_dict": model.state_dict()}, path)
    epoch, training_state = load_checkpoint(path, nn.Linear(4, 3))
    assert epoch == 0
    assert training_state is None


def test_save_logs_writes_json_with_timestamp(tmp_path: Path):
    cfg = OmegaConf.create({"seed": 42})
    path = tmp_path / "logs.json"
    save_logs(path, {"train_loss": [0.1, 0.05]}, cfg=cfg)

    payload = json.loads(path.read_text())
    assert payload["train_loss"] == [0.1, 0.05]
    assert payload["config"] == {"seed": 42}
    assert "timestamp" in payload


def test_wait_for_checkpoint_is_idempotent_when_idle():
    wait_for_checkpoint()
    wait_for_checkpoint()
