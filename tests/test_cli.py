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


def _write_dataset_training_override_yaml(
    path: Path, *, data_root: Path, output_root: Path
) -> Path:
    """Override YAML that pins dataset + training shrink knobs only.

    Model fields are left to the packaged ``configs/model/clifford_stf.yaml``
    plus a handful of CLI dot-overrides on heavy numeric knobs (see
    ``test_cli_main_uses_packaged_clifford_stf_yaml``).
    """
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
  max_train_samples: 4
  test_samples: 2
  energy_weight: 1.0
  force_weight: 1.0

training:
  batch_size: 2
  num_workers: 0
  epochs: 1
  lr: 1.0e-3
  loss: mse
  val_every: 1
""".strip()
    )
    return path


def test_cli_main_uses_packaged_clifford_stf_yaml(tmp_path: Path) -> None:
    """Drive ``cli.main`` with the packaged model YAML (no inline model fields).

    Heavy numeric knobs (``n_channels``, ``n_interactions``, ``cutoff``,
    ``n_rbf``, ``n_hidden_output``, ``max_neighbors``) are shrunk via CLI
    dot-overrides for CPU speed. The semantic fields (``stf_mode``,
    ``use_hodge_forces``, ``use_cross_track``, ``use_self_interaction``,
    ``use_gp_readout``, ``use_compile``) are left to come from
    ``configs/model/clifford_stf.yaml``. The packaged YAML being loaded
    is verified separately by re-resolving the same argv via
    ``load_config`` and asserting those semantic fields.
    """
    from cliffordstf.io.config import load_config

    data_root = tmp_path / "data"
    output_root = tmp_path / "outputs"
    yaml_path = _write_dataset_training_override_yaml(
        tmp_path / "cfg.yaml", data_root=data_root, output_root=output_root
    )
    _write_md17_npz(data_root, "aspirin", n_structures=8, natoms=3)

    argv = [
        "--config",
        str(yaml_path),
        "model.name=clifford_stf",
        "model.n_channels=8",
        "model.n_interactions=1",
        "model.n_rbf=8",
        "model.cutoff=4.0",
        "model.n_hidden_output=16",
        "model.max_neighbors=10",
    ]

    # Confirm the packaged YAML is actually on the search path and supplies
    # the semantic fields we did not override.
    cfg = load_config(argv)
    assert cfg.model.name == "clifford_stf"
    assert cfg.model.stf_mode == "stf2+stf3"
    assert cfg.model.use_hodge_forces is True
    assert cfg.model.use_cross_track is True
    assert cfg.model.use_self_interaction is False
    assert cfg.model.use_gp_readout is False
    assert cfg.model.use_compile is False
    # Shrink overrides applied with highest precedence.
    assert cfg.model.n_channels == 8
    assert cfg.model.n_interactions == 1

    rc = cli.main(argv)
    assert rc == 0

    fold_dir = output_root / "clifford_stf" / "md17" / "energy_forces" / "0" / "aspirin" / "fold1"
    assert (fold_dir / "models" / "ckpt_last.pth").exists()
    assert (fold_dir / "logs" / "logs.json").exists()


def test_cli_main_uses_packaged_clifford_baseline(tmp_path: Path) -> None:
    """Drive ``cli.main`` with the baselines-plug-in ``clifford`` variant.

    Exercises the silent try-import seam: cli.main picks up
    ``baselines.AVAILABLE_MODELS`` (which contains ``"clifford"``) and
    ``baselines.CONFIGS_DIR`` (which contains the packaged YAML), so
    ``model.name=clifford`` resolves end-to-end without inline model
    fields. Heavy numeric knobs are shrunk via CLI dot-overrides for
    CPU speed; the deactivated-STF fields are left to come from the
    packaged ``baselines/configs/model/clifford.yaml``.
    """
    from cliffordstf.io.config import load_config

    data_root = tmp_path / "data"
    output_root = tmp_path / "outputs"
    yaml_path = _write_dataset_training_override_yaml(
        tmp_path / "cfg.yaml", data_root=data_root, output_root=output_root
    )
    _write_md17_npz(data_root, "aspirin", n_structures=8, natoms=3)

    from baselines import CONFIGS_DIR

    argv = [
        "--config",
        str(yaml_path),
        "model.name=clifford",
        "model.n_channels=8",
        "model.n_interactions=1",
        "model.n_rbf=8",
        "model.cutoff=4.0",
        "model.n_hidden_output=16",
        "model.max_neighbors=10",
    ]

    cfg = load_config(argv, extra_search_paths=[CONFIGS_DIR])
    assert cfg.model.name == "clifford"
    assert cfg.model.stf_mode == "none"
    assert cfg.model.use_hodge_forces is False
    assert cfg.model.use_cross_track is False
    assert cfg.model.use_l2 is False
    assert cfg.model.n_channels == 8

    rc = cli.main(argv)
    assert rc == 0

    fold_dir = output_root / "clifford" / "md17" / "energy_forces" / "0" / "aspirin" / "fold1"
    assert (fold_dir / "models" / "ckpt_last.pth").exists()
    assert (fold_dir / "logs" / "logs.json").exists()


def test_cli_main_uses_packaged_schnet_baseline(tmp_path: Path) -> None:
    """Drive ``cli.main`` with the ``schnet`` baseline.

    Confirms the plug-in seam works for a PyG-native baseline (SchNet
    returns bare energy; the trainer computes forces via autograd from
    ``-dE/dR``). Heavy SchNet knobs are shrunk via CLI dot-overrides
    for CPU speed.
    """
    data_root = tmp_path / "data"
    output_root = tmp_path / "outputs"
    yaml_path = _write_dataset_training_override_yaml(
        tmp_path / "cfg.yaml", data_root=data_root, output_root=output_root
    )
    _write_md17_npz(data_root, "aspirin", n_structures=8, natoms=3)

    argv = [
        "--config",
        str(yaml_path),
        "model.name=schnet",
        "model.hidden_channels=16",
        "model.num_filters=16",
        "model.num_interactions=1",
        "model.num_gaussians=8",
        "model.cutoff=4.0",
    ]

    rc = cli.main(argv)
    assert rc == 0

    fold_dir = output_root / "schnet" / "md17" / "energy_forces" / "0" / "aspirin" / "fold1"
    assert (fold_dir / "models" / "ckpt_last.pth").exists()
    assert (fold_dir / "logs" / "logs.json").exists()
