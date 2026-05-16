"""End-to-end tests for ``train_lightning`` (Phase 2 Step 19)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from cliffordstf.training.lightning_trainer import train_lightning


def _md17_like_batch(n_structures: int = 6, n_atoms: int = 3) -> list[Data]:
    rng = np.random.default_rng(0)
    items: list[Data] = []
    for _ in range(n_structures):
        z = torch.from_numpy(rng.integers(1, 30, size=(n_atoms,), dtype=np.int64))
        pos = torch.tensor(rng.standard_normal((n_atoms, 3)) * 0.5, dtype=torch.float32)
        energy = torch.tensor([float(rng.standard_normal())], dtype=torch.float32)
        force = torch.tensor(rng.standard_normal((n_atoms, 3)), dtype=torch.float32)
        items.append(Data(z=z, pos=pos, energy=energy, force=force))
    return items


def _tiny_cliffordstf_cfg(tmp_path: Path) -> OmegaConf:
    cfg = {
        "seed": 0,
        "device": "cpu",
        "output_root": str(tmp_path / "outputs"),
        "training": {
            "lr": 1e-3,
            "optimizer": "adam",
            "weight_decay": 0.0,
            "loss": "mse",
            "epochs": 1,
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
    return OmegaConf.create(cfg)


def test_train_lightning_runs_one_epoch(tmp_path: Path) -> None:
    cfg = _tiny_cliffordstf_cfg(tmp_path)
    items = _md17_like_batch()
    loaders = {
        "train": DataLoader(items[:4], batch_size=2, shuffle=False),
        "val": DataLoader(items[4:], batch_size=2, shuffle=False),
    }
    result = train_lightning(cfg, loaders, extra_parts=("aspirin", "fold1"))
    assert "output_dir" in result
    assert Path(result["output_dir"]).is_dir()


def test_train_lightning_with_test_loader(tmp_path: Path) -> None:
    cfg = _tiny_cliffordstf_cfg(tmp_path)
    items = _md17_like_batch(n_structures=8)
    loaders = {
        "train": DataLoader(items[:4], batch_size=2, shuffle=False),
        "val": DataLoader(items[4:6], batch_size=2, shuffle=False),
        "test": DataLoader(items[6:], batch_size=2, shuffle=False),
    }
    result = train_lightning(cfg, loaders, extra_parts=("aspirin", "fold1"))
    assert isinstance(result["test_metrics"], list)


def test_train_lightning_logs_energy_forces_metrics(tmp_path: Path) -> None:
    """val_energy_mae / val_force_mae / val_force_cos / val_efwt are populated."""
    import pytorch_lightning as pl

    from cliffordstf.models import build_model
    from cliffordstf.training.lightning import CliffordSTFLightningModule

    cfg = _tiny_cliffordstf_cfg(tmp_path)
    items = _md17_like_batch()
    model = build_model(cfg.model.name, cfg)
    module = CliffordSTFLightningModule(model, cfg)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        inference_mode=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )
    trainer.fit(
        module,
        train_dataloaders=DataLoader(items[:4], batch_size=2, shuffle=False),
        val_dataloaders=DataLoader(items[4:], batch_size=2, shuffle=False),
    )
    metrics = trainer.callback_metrics
    for key in ("val_energy_mae", "val_force_mae", "val_force_cos", "val_efwt"):
        assert key in metrics, f"missing {key} in {sorted(metrics)}"
        assert torch.isfinite(metrics[key]).item()
