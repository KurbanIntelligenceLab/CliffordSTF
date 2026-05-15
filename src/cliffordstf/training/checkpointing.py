"""Asynchronous checkpoint I/O and JSON log writing.

Checkpoint writes are dispatched to a single background-thread executor so the
training loop never blocks on disk I/O. Always call :func:`wait_for_checkpoint`
before exit so the last write completes.
"""

from __future__ import annotations

import copy
import json
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from omegaconf import OmegaConf
from torch import nn

if TYPE_CHECKING:
    from typing import Protocol

    from omegaconf import DictConfig
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

    class _StateDictProtocol(Protocol):
        def state_dict(self) -> dict[str, Any]: ...

        def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...


_CKPT_EXECUTOR = ThreadPoolExecutor(max_workers=1)
_CKPT_FUTURE: Future[None] | None = None


def _save_ckpt_to_disk(ckpt: dict[str, Any], path: Path) -> None:
    torch.save(ckpt, path)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    epoch: int = 0,
    metrics: dict[str, Any] | None = None,
    cfg: DictConfig | None = None,
    scheduler: LRScheduler | ReduceLROnPlateau | None = None,
    scaler: _StateDictProtocol | None = None,
    training_state: dict[str, Any] | None = None,
) -> None:
    """Snapshot training state and dispatch the disk write to a background thread.

    Tensors are cloned to CPU before submission so the write thread does not
    pin GPU memory. The previous in-flight write is waited on before a new one
    is queued, so at most one save is ever pending.
    """
    global _CKPT_FUTURE
    if _CKPT_FUTURE is not None:
        _CKPT_FUTURE.result()

    state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    ckpt: dict[str, Any] = {
        "model_state_dict": state_dict_cpu,
        "epoch": epoch,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = copy.deepcopy(optimizer.state_dict())
    if metrics is not None:
        ckpt["metrics"] = metrics
    if cfg is not None:
        ckpt["config"] = OmegaConf.to_container(cfg, resolve=True)
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = copy.deepcopy(scheduler.state_dict())
    if scaler is not None:
        ckpt["scaler_state_dict"] = copy.deepcopy(scaler.state_dict())
    if training_state is not None:
        ckpt["training_state"] = copy.deepcopy(training_state)

    _CKPT_FUTURE = _CKPT_EXECUTOR.submit(_save_ckpt_to_disk, ckpt, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | ReduceLROnPlateau | None = None,
    scaler: _StateDictProtocol | None = None,
    device: torch.device | str | None = None,
) -> tuple[int, dict[str, Any] | None]:
    """Load a checkpoint and restore optimizer / scheduler / scaler state.

    Returns ``(epoch, training_state)``. ``training_state`` carries fields like
    ``best_val`` and ``best_epoch`` and is ``None`` if absent.
    """
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    epoch: int = ckpt.get("epoch", 0)
    training_state: dict[str, Any] | None = ckpt.get("training_state", None)
    return epoch, training_state


def wait_for_checkpoint() -> None:
    """Block until the last async checkpoint save completes."""
    global _CKPT_FUTURE
    if _CKPT_FUTURE is not None:
        _CKPT_FUTURE.result()
        _CKPT_FUTURE = None


def save_logs(
    path: Path,
    logs: dict[str, Any],
    cfg: DictConfig | None = None,
) -> None:
    """Write ``logs`` to a JSON file with the resolved config embedded."""
    output: dict[str, Any] = dict(logs)
    if cfg is not None:
        output["config"] = OmegaConf.to_container(cfg, resolve=True)
    output["timestamp"] = datetime.now(UTC).isoformat()
    path.write_text(json.dumps(output, indent=4))
