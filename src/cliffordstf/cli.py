"""``cliffordstf-train`` command-line entrypoint.

Wires :func:`cliffordstf.io.config.load_config` ->
:func:`cliffordstf.data.build_dataloaders` -> training. The training
backend is selected by ``cfg.training.engine``:

* ``"legacy"`` (default) - :func:`cliffordstf.training.trainer.train`.
* ``"lightning"`` - :func:`cliffordstf.training.lightning_trainer.train_lightning`.

One training run is launched per ``LoaderSet`` returned by the dataset
factory (multi-fold datasets such as MD17 trigger one run per fold).

The optional ``baselines`` package, if installed, contributes both
extra model factories (via ``baselines.AVAILABLE_MODELS``) and an
extra config search path (``baselines.CONFIGS_DIR``). Both are picked
up via a silent try-import so cliffordstf works either way.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from cliffordstf.data import build_dataloaders
from cliffordstf.io.config import load_config
from cliffordstf.training.lightning_trainer import train_lightning
from cliffordstf.training.trainer import train

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from omegaconf import DictConfig

    from cliffordstf.domain import ModelFactory

SUPPORTED_ENGINES: frozenset[str] = frozenset({"legacy", "lightning"})


def _load_baseline_plugins() -> tuple[Mapping[str, ModelFactory] | None, list[Path] | None]:
    """Return (extra model factories, extra config search paths) or (None, None)."""
    try:
        from baselines import AVAILABLE_MODELS, CONFIGS_DIR
    except ImportError:
        return None, None
    return AVAILABLE_MODELS, [CONFIGS_DIR]


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
    model_extras, extra_search_paths = _load_baseline_plugins()
    cfg = load_config(argv, extra_search_paths=extra_search_paths)
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
                model_extras=model_extras,
            )
        )
    return 0


__all__ = ["SUPPORTED_ENGINES", "main"]
