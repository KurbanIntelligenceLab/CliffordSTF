"""End-to-end smoke test for ``cliffordstf.training.trainer``."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from cliffordstf.training.trainer import (
    build_output_dirs,
    resolve_device,
    train,
    train_one_run,
)


def _tiny_data(seed: int) -> Data:
    g = torch.Generator().manual_seed(seed)
    n = 4
    return Data(
        z=torch.randint(1, 30, (n,), generator=g),
        pos=torch.randn(n, 3, generator=g) * 2.0,
        batch=torch.zeros(n, dtype=torch.long),
        energy=torch.randn(1, generator=g),
        force=torch.randn(n, 3, generator=g),
    )


def _make_loaders() -> dict[str, DataLoader]:
    train_loader = DataLoader([_tiny_data(i) for i in range(4)], batch_size=2)
    val_loader = DataLoader([_tiny_data(100 + i) for i in range(2)], batch_size=2)
    return {"train": train_loader, "val": val_loader}


def _tiny_cfg(tmp_output: Path) -> OmegaConf:
    return OmegaConf.create(
        {
            "seed": 0,
            "device": "cpu",
            "output_root": str(tmp_output),
            "model": {
                "name": "clifford_stf",
                "n_channels": 8,
                "n_interactions": 1,
                "cutoff": 4.0,
                "n_rbf": 8,
                "n_hidden_output": 16,
                "max_neighbors": 10,
                "n_heads": 2,
                "use_self_interaction": False,
                "use_gp_readout": False,
                "use_multiscale": False,
            },
            "dataset": {
                "name": "synthetic",
                "task_type": "energy_forces",
                "energy_weight": 1.0,
                "force_weight": 1.0,
            },
            "training": {
                "epochs": 1,
                "lr": 1e-3,
                "weight_decay": 0.0,
                "loss": "mse",
                "amp": False,
                "grad_accum_steps": 1,
                "val_every": 1,
            },
        }
    )


def test_resolve_device_returns_cpu_when_explicit():
    cfg = OmegaConf.create({"device": "cpu"})
    assert resolve_device(cfg) == torch.device("cpu")


def test_build_output_dirs_creates_layout(tmp_path: Path):
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "output_root": str(tmp_path),
            "model": {"name": "m"},
            "dataset": {"name": "d", "task_type": "energy_forces"},
        }
    )
    base, model_dir, logs_dir = build_output_dirs(cfg)
    assert base == tmp_path / "m" / "d" / "energy_forces" / "0"
    assert model_dir.exists()
    assert logs_dir.exists()


def test_train_one_run_produces_finite_loss_and_writes_artifacts(tmp_path: Path):
    cfg = _tiny_cfg(tmp_path)
    loaders = _make_loaders()
    logs = train_one_run(cfg, loaders, torch.device("cpu"))

    assert len(logs["train_loss"]) == 1
    assert torch.isfinite(torch.tensor(logs["train_loss"][0]))
    assert len(logs["val_metrics"]) == 1
    assert "energy_mae" in logs["val_metrics"][0]
    assert logs["best_epoch"] == 1

    base = tmp_path / cfg.model.name / cfg.dataset.name / cfg.dataset.task_type / str(cfg.seed)
    assert (base / "models" / "ckpt_last.pth").exists()
    assert (base / "models" / "ckpt_best_val.pth").exists()

    written_logs = json.loads((base / "logs" / "logs.json").read_text())
    assert written_logs["dataset"] == "synthetic"
    assert written_logs["model"] == "clifford_stf"


def test_train_top_level_entrypoint_runs(tmp_path: Path):
    cfg = _tiny_cfg(tmp_path)
    loaders = _make_loaders()
    logs = train(cfg, loaders)
    assert logs["best_epoch"] == 1


def test_build_output_dirs_appends_extra_parts(tmp_path: Path):
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "output_root": str(tmp_path),
            "model": {"name": "m"},
            "dataset": {"name": "d", "task_type": "energy_forces"},
        }
    )
    base, _, _ = build_output_dirs(cfg, ("aspirin", "fold1"))
    assert base == tmp_path / "m" / "d" / "energy_forces" / "0" / "aspirin" / "fold1"
    assert base.exists()


def test_train_with_extra_parts_writes_under_nested_dir(tmp_path: Path):
    cfg = _tiny_cfg(tmp_path)
    loaders = _make_loaders()
    logs = train(cfg, loaders, extra_parts=("aspirin", "fold1"))
    assert logs["best_epoch"] == 1
    nested = (
        tmp_path
        / cfg.model.name
        / cfg.dataset.name
        / cfg.dataset.task_type
        / str(cfg.seed)
        / "aspirin"
        / "fold1"
    )
    assert (nested / "models" / "ckpt_last.pth").exists()


def test_train_forwards_runtime_stats_to_evaluate(tmp_path: Path):
    cfg = _tiny_cfg(tmp_path)
    loaders = _make_loaders()
    logs_raw = train(cfg, loaders, extra_parts=("raw",))
    logs_scaled = train(
        cfg, loaders, runtime_stats={"std": 4.0, "mean": 0.0}, extra_parts=("scaled",)
    )
    raw_mae = logs_raw["val_metrics"][-1]["energy_mae"]
    scaled_mae = logs_scaled["val_metrics"][-1]["energy_mae"]
    assert abs(scaled_mae - raw_mae * 4.0) < 1e-5
