"""End-to-end tests for :class:`LegacyCheckpointCallback` (Phase 2 Step 21)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from cliffordstf.training.lightning_trainer import train_lightning


def _md17_like(n: int = 8, natoms: int = 3) -> list[Data]:
    rng = np.random.default_rng(0)
    return [
        Data(
            z=torch.from_numpy(rng.integers(1, 30, size=(natoms,), dtype=np.int64)),
            pos=torch.tensor(rng.standard_normal((natoms, 3)) * 0.5, dtype=torch.float32),
            energy=torch.tensor([float(rng.standard_normal())], dtype=torch.float32),
            force=torch.tensor(rng.standard_normal((natoms, 3)), dtype=torch.float32),
        )
        for _ in range(n)
    ]


def _tiny_cfg(tmp_path: Path, epochs: int = 1) -> OmegaConf:
    return OmegaConf.create(
        {
            "seed": 0,
            "device": "cpu",
            "output_root": str(tmp_path / "outputs"),
            "training": {
                "lr": 1e-3,
                "optimizer": "adam",
                "weight_decay": 0.0,
                "loss": "mse",
                "epochs": epochs,
                "batch_size": 2,
                "val_every": 1,
                "grad_clip": 0.0,
                "grad_accum_steps": 1,
                "amp": False,
                "amp_dtype": "bfloat16",
                "early_stopping": {"patience": 0},
            },
            "dataset": {
                "name": "md17",
                "task_type": "energy_forces",
                "energy_weight": 1.0,
                "force_weight": 1.0,
            },
            "model": {
                "name": "clifford_stf",
                "interface": "data_wrapper",
                "n_atom_types": 100,
                "n_channels": 8,
                "n_interactions": 1,
                "n_rbf": 8,
                "cutoff": 4.0,
                "n_hidden_output": 8,
                "max_neighbors": 10,
                "stf_mode": "stf2+stf3",
                "use_hodge_forces": True,
                "use_cross_track": True,
                "use_self_interaction": False,
                "use_gp_readout": False,
                "use_compile": False,
            },
        }
    )


def test_lightning_trainer_writes_ckpt_last(tmp_path: Path) -> None:
    cfg = _tiny_cfg(tmp_path)
    items = _md17_like()
    loaders = {
        "train": DataLoader(items[:4], batch_size=2, shuffle=False),
        "val": DataLoader(items[4:], batch_size=2, shuffle=False),
    }
    result = train_lightning(cfg, loaders, extra_parts=("aspirin", "fold1"))
    model_dir = Path(result["output_dir"]) / "models"
    assert (model_dir / "ckpt_last.pth").is_file()


def test_lightning_trainer_writes_ckpt_best_val(tmp_path: Path) -> None:
    cfg = _tiny_cfg(tmp_path, epochs=2)
    items = _md17_like()
    loaders = {
        "train": DataLoader(items[:4], batch_size=2, shuffle=False),
        "val": DataLoader(items[4:], batch_size=2, shuffle=False),
    }
    result = train_lightning(cfg, loaders, extra_parts=("aspirin", "fold1"))
    model_dir = Path(result["output_dir"]) / "models"
    assert (model_dir / "ckpt_best_val.pth").is_file()


def test_ckpt_last_schema_matches_legacy(tmp_path: Path) -> None:
    cfg = _tiny_cfg(tmp_path)
    items = _md17_like()
    loaders = {
        "train": DataLoader(items[:4], batch_size=2, shuffle=False),
        "val": DataLoader(items[4:], batch_size=2, shuffle=False),
    }
    result = train_lightning(cfg, loaders, extra_parts=("aspirin", "fold1"))
    ckpt_path = Path(result["output_dir"]) / "models" / "ckpt_last.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt
    assert "epoch" in ckpt
    assert "config" in ckpt
    assert "training_state" in ckpt
    assert "best_val" in ckpt["training_state"]
