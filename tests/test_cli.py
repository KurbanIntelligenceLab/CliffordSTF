"""End-to-end CLI smoke tests for ``cliffordstf.cli``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cliffordstf import cli


def _write_md17_npz(
    data_root: Path, molecule: str, *, n_structures: int = 24, natoms: int = 5
) -> None:
    mol_dir = data_root / molecule
    mol_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    np.savez(
        mol_dir / f"rmd17_{molecule}.npz",
        nuclear_charges=rng.integers(1, 30, size=(natoms,), dtype=np.int64),
        coords=(rng.standard_normal((n_structures, natoms, 3)) * 0.5).astype(np.float32),
        energies=rng.standard_normal((n_structures,)).astype(np.float32),
        forces=rng.standard_normal((n_structures, natoms, 3)).astype(np.float32),
    )


def _write_override_yaml(path: Path, *, data_root: Path, output_root: Path) -> Path:
    path.write_text(
        f"""
seed: 0
device: cpu
output_root: {output_root}

dataset:
  name: md17
  data_root: {data_root}
  task_type: energy_forces
  molecule: aspirin
  k_folds: 1
  val_frac: 0.25
  max_train_samples: 12
  test_samples: 4
  energy_weight: 1.0
  force_weight: 1.0

model:
  name: clifford_stf
  n_channels: 8
  n_interactions: 1
  cutoff: 4.0
  n_rbf: 8
  n_hidden_output: 16
  max_neighbors: 10
  n_heads: 2
  use_self_interaction: false
  use_gp_readout: false
  use_multiscale: false

training:
  batch_size: 4
  num_workers: 0
  epochs: 1
  lr: 1.0e-3
  loss: mse
  val_every: 1
""".strip()
    )
    return path


def test_cli_main_runs_md17_end_to_end(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "outputs"
    yaml_path = _write_override_yaml(
        tmp_path / "cfg.yaml", data_root=data_root, output_root=output_root
    )
    _write_md17_npz(data_root, "aspirin", n_structures=24, natoms=5)

    rc = cli.main(["--config", str(yaml_path)])
    assert rc == 0

    fold_dir = output_root / "clifford_stf" / "md17" / "energy_forces" / "0" / "aspirin" / "fold1"
    assert (fold_dir / "models" / "ckpt_last.pth").exists()
    assert (fold_dir / "models" / "ckpt_best_val.pth").exists()
    assert (fold_dir / "logs" / "logs.json").exists()


def test_cli_main_unknown_dataset_raises(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="Unknown dataset 'no_such_dataset'"):
        cli.main(["dataset.name=no_such_dataset", "model.name=clifford_stf"])
