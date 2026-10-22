"""Training entrypoint and main loop for energy + forces models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from tqdm import tqdm

from cliffordstf.models import build_model
from cliffordstf.reproducibility import SeedManager
from cliffordstf.training.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    save_logs,
    wait_for_checkpoint,
)
from cliffordstf.training.evaluate import evaluate_epoch
from cliffordstf.training.losses import compute_loss
from cliffordstf.training.optim import build_optimizer, build_scheduler

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sized
    from typing import Protocol

    from omegaconf import DictConfig

    from cliffordstf.domain import ModelFactory

    class _Loader(Iterable[Any], Sized, Protocol):
        """Data loader protocol."""

    class _EmaModule(Protocol):
        def init_ema(self) -> None: ...

        def update_ema(self) -> None: ...

        def apply_ema(self) -> None: ...

        def restore_from_ema(self) -> None: ...


PRIMARY_METRIC = "energy_mae"


def _monitor_value(val_metrics: dict, cfg: object) -> float:
    """Value used for early-stopping, best-checkpoint, and plateau LR."""
    try:
        mode = cfg.training.get("monitor_metric", "energy_mae")  # type: ignore[attr-defined]
    except Exception:
        mode = "energy_mae"
    if mode == "force_mae":
        return float(val_metrics.get("force_mae", float("inf")))
    if mode == "combined":
        try:
            ew = float(cfg.dataset.get("energy_weight", 1.0))  # type: ignore[attr-defined]
            fw = float(cfg.dataset.get("force_weight", 1.0))  # type: ignore[attr-defined]
        except Exception:
            ew, fw = 1.0, 1.0
        return ew * float(val_metrics.get("energy_mae", float("inf"))) + fw * float(
            val_metrics.get("force_mae", float("inf"))
        )
    return float(val_metrics.get("energy_mae", float("inf")))


def resolve_device(cfg: DictConfig) -> torch.device:
    """Resolve a torch device from ``cfg.device``. ``"auto"`` picks CUDA when available."""
    d = cfg.get("device", "auto")
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def build_output_dirs(
    cfg: DictConfig,
    extra_parts: tuple[str, ...] = (),
) -> tuple[Path, Path, Path]:
    """Create and return ``(base_dir, model_dir, logs_dir)``."""
    target = cfg.dataset.get("target_name", cfg.dataset.get("task_type", "default"))
    base_dir = Path(
        cfg.get("output_root", "outputs"),
        cfg.model.name,
        cfg.dataset.name,
        str(target),
        str(cfg.seed),
        *extra_parts,
    )
    model_dir = base_dir / "models"
    logs_dir = base_dir / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, model_dir, logs_dir


def _is_plateau(scheduler: LRScheduler | ReduceLROnPlateau | None) -> bool:
    return isinstance(scheduler, ReduceLROnPlateau)


def _format_metrics(metrics: dict[str, float]) -> str:
    parts = [f"{k}: {v:.6f}" for k, v in metrics.items()]
    return " | ".join(parts)


def train_one_run(
    cfg: DictConfig,
    loaders: dict[str, object],
    device: torch.device,
    *,
    runtime_stats: Mapping[str, object] | None = None,
    extra_parts: tuple[str, ...] = (),
    model_extras: Mapping[str, ModelFactory] | None = None,
) -> dict[str, Any]:
    """Train ``cfg.model`` for ``cfg.training.epochs`` epochs and return logs."""
    _, model_dir, logs_dir = build_output_dirs(cfg, extra_parts)

    model = build_model(cfg.model.name, cfg, extras=model_extras).to(device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    if hasattr(model, "init_ema"):
        cast("_EmaModule", model).init_ema()

    epochs = cfg.training.epochs
    print_every = cfg.logging.get("print_every", 1) if "logging" in cfg else 1
    grad_clip = cfg.training.get("grad_clip", None)
    grad_accum_steps = cfg.training.get("grad_accum_steps", 1)
    val_every = cfg.training.get("val_every", 1)

    use_amp = cfg.training.get("amp", False) and device.type == "cuda"
    amp_dtype_str = cfg.training.get("amp_dtype", "bfloat16")
    amp_dtype = torch.bfloat16 if amp_dtype_str == "bfloat16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    es_cfg = cfg.training.get("early_stopping", None)
    es_patience = es_cfg.get("patience", 0) if es_cfg else 0
    es_counter = 0

    logs: dict[str, Any] = {
        "dataset": cfg.dataset.name,
        "model": cfg.model.name,
        "train_loss": [],
        "val_metrics": [],
        "duration": [],
        "best_val": None,
        "best_epoch": None,
    }

    best_val = float("inf")
    best_epoch = 0
    start_epoch = 1
    last_val_metrics: dict[str, float] | None = None

    ckpt_last_path = model_dir / "ckpt_last.pth"
    if ckpt_last_path.exists():
        resumed_epoch, training_state = load_checkpoint(
            ckpt_last_path, model, optimizer, scheduler, scaler, device
        )
        start_epoch = resumed_epoch + 1
        if training_state:
            best_val = training_state.get("best_val", float("inf"))
            best_epoch = training_state.get("best_epoch", 0)
            es_counter = training_state.get("es_counter", 0)
            logs = training_state.get("logs", logs)

    epoch_bar = tqdm(range(start_epoch, epochs + 1), desc=cfg.model.name, unit="ep")
    train_start = time.time()

    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0
        n_batches = 0
        start = time.time()
        optimizer.zero_grad()

        train_loader = cast("_Loader", loaders["train"])
        batch_iter = tqdm(
            enumerate(train_loader, 1),
            total=len(train_loader),
            desc=f"  Epoch {epoch:03d}",
            unit="batch",
            leave=False,
        )
        for batch_idx, data in batch_iter:
            data = data.to(device)

            with torch.autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                loss = compute_loss(model, data, cfg, training=True)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            n_train = len(train_loader)
            if batch_idx % grad_accum_steps == 0 or batch_idx == n_train:
                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if hasattr(model, "update_ema"):
                    cast("_EmaModule", model).update_ema()

            total_loss += loss.item() * grad_accum_steps
            n_batches += 1

        if scheduler is not None and not _is_plateau(scheduler):
            scheduler.step()

        duration = time.time() - start
        train_loss = total_loss / max(1, n_batches)

        run_val = (epoch % val_every == 0) or (epoch == epochs)

        if run_val:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if hasattr(model, "apply_ema"):
                cast("_EmaModule", model).apply_ema()
            try:
                val_metrics = evaluate_epoch(
                    model,
                    loaders["val"],
                    cfg,
                    device,
                    amp_dtype=amp_dtype if use_amp else None,
                    runtime_stats=runtime_stats,
                )
            finally:
                if hasattr(model, "restore_from_ema"):
                    cast("_EmaModule", model).restore_from_ema()
            last_val_metrics = val_metrics

            if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(_monitor_value(val_metrics, cfg))

            logs["val_metrics"].append(val_metrics)

            val_primary = _monitor_value(val_metrics, cfg)
            if val_primary < best_val:
                best_val = val_primary
                best_epoch = epoch
                es_counter = 0

                if hasattr(model, "apply_ema"):
                    cast("_EmaModule", model).apply_ema()
                try:
                    save_checkpoint(
                        model_dir / "ckpt_best_val.pth",
                        model,
                        optimizer,
                        epoch,
                        metrics={"best_val_monitor": best_val, **val_metrics},
                        cfg=cfg,
                    )
                finally:
                    if hasattr(model, "restore_from_ema"):
                        cast("_EmaModule", model).restore_from_ema()
            else:
                es_counter += 1
        else:
            val_metrics = last_val_metrics if last_val_metrics is not None else {}
            val_primary = _monitor_value(val_metrics, cfg)

        logs["train_loss"].append(train_loss)
        logs["duration"].append(duration)
        logs["best_val"] = best_val
        logs["best_epoch"] = best_epoch
        logs["current_epoch"] = epoch

        if epoch % print_every == 0:
            if val_metrics:
                tqdm.write(
                    f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | "
                    f"Val {_format_metrics(val_metrics)} | "
                    f"Best {PRIMARY_METRIC}: {best_val:.6f} (ep {best_epoch}) | "
                    f"{duration:.1f}s"
                )
            else:
                tqdm.write(
                    f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | "
                    f"Best {PRIMARY_METRIC}: {best_val:.6f} (ep {best_epoch}) | "
                    f"{duration:.1f}s"
                )

        save_checkpoint(
            model_dir / "ckpt_last.pth",
            model,
            optimizer,
            epoch,
            metrics={"last_train_loss": train_loss},
            cfg=cfg,
            scheduler=scheduler,
            scaler=scaler,
            training_state={
                "best_val": best_val,
                "best_epoch": best_epoch,
                "es_counter": es_counter,
                "logs": logs,
            },
        )
        save_logs(logs_dir / "logs.json", logs, cfg=cfg)

        if run_val and es_patience > 0 and es_counter >= es_patience:
            tqdm.write(
                f"Early stopping at epoch {epoch} (no improvement for {es_patience} epochs)"
            )
            break

    wait_for_checkpoint()

    total_duration = time.time() - train_start
    logs["best_val"] = best_val
    logs["best_epoch"] = best_epoch
    logs["total_duration_sec"] = total_duration

    if "test" in loaders:
        if device.type == "cuda":
            import gc as _gc

            _gc.collect()
            torch.cuda.empty_cache()
        if hasattr(model, "apply_ema"):
            cast("_EmaModule", model).apply_ema()
        try:
            test_metrics = evaluate_epoch(
                model,
                loaders["test"],
                cfg,
                device,
                amp_dtype=amp_dtype if use_amp else None,
                runtime_stats=runtime_stats,
            )
        finally:
            if hasattr(model, "restore_from_ema"):
                cast("_EmaModule", model).restore_from_ema()
        logs["test_metrics"] = test_metrics

    save_logs(logs_dir / "logs.json", logs, cfg=cfg)
    return logs


def train(
    cfg: DictConfig,
    loaders: dict[str, object],
    *,
    runtime_stats: Mapping[str, object] | None = None,
    extra_parts: tuple[str, ...] = (),
    model_extras: Mapping[str, ModelFactory] | None = None,
) -> dict[str, Any]:
    """Top-level training entrypoint: seed, resolve device, run the loop."""
    SeedManager.set_global_seed(cfg.seed)
    device = resolve_device(cfg)
    return train_one_run(
        cfg,
        loaders,
        device,
        runtime_stats=runtime_stats,
        extra_parts=extra_parts,
        model_extras=model_extras,
    )
