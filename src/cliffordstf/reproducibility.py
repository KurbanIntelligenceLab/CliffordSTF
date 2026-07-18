"""Seed management and provenance writing.

Per ``CODING_RULES.md`` §E:

- ``SeedManager.set_global_seed`` is called exactly once at every entrypoint
  before any random / numpy / torch / CUDA call.
- ``ProvenanceWriter`` writes a sibling ``<artifact>.meta.json`` for every
  artifact under ``outputs/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig

import numpy as np


@dataclass(frozen=True, slots=True)
class SeedManager:
    """Coordinated seeding for ``random``, ``numpy``, PyTorch, and CUDA."""

    seed: int

    @classmethod
    def set_global_seed(cls, seed: int) -> SeedManager:
        """Seed every supported RNG and return a record of what was set.

        Call this exactly once per entrypoint, before any random sampling,
        dataset loading, or model construction. Exact bit-for-bit
        reproducibility is not guaranteed across PyTorch / CUDA / cuDNN
        releases (``CODING_RULES.md`` §E.1).
        """
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

        try:
            import torch
        except ImportError:
            pass
        else:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        return cls(seed=seed)


class ProvenanceWriter:
    """Writes ``<artifact>.meta.json`` sidecars with required §E.2 fields."""

    def __init__(
        self,
        *,
        config: DictConfig,
        config_path: Path,
        seed: int,
        package_version: str,
        dataset_manifest: Path | None = None,
        split_manifest: Path | None = None,
    ) -> None:
        self._config = config
        self._config_path = config_path
        self._seed = seed
        self._package_version = package_version
        self._dataset_manifest = dataset_manifest
        self._split_manifest = split_manifest

    def write(self, artifact: Path, payload: dict[str, object] | None = None) -> Path:
        """Write ``<artifact>.meta.json`` next to ``artifact`` and return its path."""
        from omegaconf import OmegaConf

        config_yaml = OmegaConf.to_yaml(self._config, resolve=True)
        meta: dict[str, object] = {
            "git_sha": _git_sha(),
            "git_dirty": _git_dirty(),
            "config_hash": "sha256:" + hashlib.sha256(config_yaml.encode()).hexdigest(),
            "config_path": str(self._config_path),
            "dataset_manifest": str(self._dataset_manifest) if self._dataset_manifest else None,
            "split_manifest": str(self._split_manifest) if self._split_manifest else None,
            "seed": self._seed,
            "python_version": platform.python_version(),
            "package_version": self._package_version,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "hostname": socket.gethostname(),
            "gpu_model": _gpu_model(),
            "cuda_version": _cuda_version(),
            "driver_version": _driver_version(),
            "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
            "python_executable": sys.executable,
        }
        if payload:
            meta.update(payload)

        meta_path = artifact.with_suffix(artifact.suffix + ".meta.json")
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
        return meta_path


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip()


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return bool(out.stdout.strip())


def _gpu_model() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


def _cuda_version() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    return torch.version.cuda


def _driver_version() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip().splitlines()[0] if out.stdout.strip() else None
