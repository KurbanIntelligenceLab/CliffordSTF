"""``cliffordstf-train`` command-line entrypoint."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cliffordstf.data import build_dataloaders
from cliffordstf.io.config import load_config
from cliffordstf.training.lightning_trainer import train_lightning
from cliffordstf.training.trainer import train

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omegaconf import DictConfig


SUPPORTED_ENGINES: frozenset[str] = frozenset({"legacy", "lightning"})


def _resolve_engine(cfg: DictConfig) -> str:
    """Read and validate ``cfg.training.engine`` (defaults to ``"legacy"``)."""
    engine = cfg.training.get("engine", "legacy") if "training" in cfg else "legacy"
    if engine not in SUPPORTED_ENGINES:
        raise ValueError(
            f"Unsupported cfg.training.engine={engine!r}. "
            f"Expected one of {sorted(SUPPORTED_ENGINES)}."
        )
    return str(engine)


def main(argv: Sequence[str] | None = None) -> int:
    """Build loaders, run the trainer, and return ``0`` on success."""
    cfg = load_config(argv)
    engine = _resolve_engine(cfg)
    runner = train_lightning if engine == "lightning" else train

    loader_sets = build_dataloaders(cfg)
    results: list[dict[str, Any]] = []
    for loader_set in loader_sets:
        loaders: dict[str, object] = {
            "train": loader_set.train,
            "val": loader_set.val,
        }
        if loader_set.test is not None:
            loaders["test"] = loader_set.test
        results.append(
            runner(
                cfg,
                loaders,
                runtime_stats=loader_set.runtime_stats,
                extra_parts=loader_set.extra_parts,
            )
        )
    return 0


__all__ = ["SUPPORTED_ENGINES", "main"]
