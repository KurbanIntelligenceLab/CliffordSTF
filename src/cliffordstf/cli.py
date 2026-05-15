"""``cliffordstf-train`` command-line entrypoint.

Wires :func:`cliffordstf.io.config.load_config` ->
:func:`cliffordstf.data.build_dataloaders` -> :func:`cliffordstf.training.trainer.train`.
One training run is launched per ``LoaderSet`` returned by the dataset
factory (multi-fold datasets such as MD17 trigger one run per fold).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cliffordstf.data import build_dataloaders
from cliffordstf.io.config import load_config
from cliffordstf.training.trainer import train

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Build loaders, run the trainer, and return ``0`` on success."""
    cfg = load_config(argv)
    loader_sets = build_dataloaders(cfg)
    results: list[dict[str, Any]] = []
    for loader_set in loader_sets:
        loaders: dict[str, object] = {
            "train": loader_set.train,
            "val": loader_set.val,
        }
        if loader_set.test is not None:
            loaders["test"] = loader_set.test
        results.append(train(cfg, loaders))
    return 0


__all__ = ["main"]
