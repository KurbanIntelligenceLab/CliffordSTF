"""MD17 dataset factory."""

from __future__ import annotations

import logging
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import torch
from sklearn.model_selection import KFold
from torch.utils.data import random_split

from cliffordstf.data._md17_io import _MD17_MOLECULES, MD17Dataset
from cliffordstf.data.loader import build_loader

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch.utils.data import Dataset

    from cliffordstf.data import LoaderSet


_logger = logging.getLogger(__name__)


def _resolve_molecules(ds_cfg: DictConfig) -> list[str]:
    """Resolve the dataset config's ``molecule`` selector into concrete names.

    Accepts a single name, a list of names, or ``"all"``. Strips the
    optional ``rmd17_`` prefix, validates against the known molecule
    table, and keeps only molecules whose subdirectory exists under
    ``ds_cfg.data_root``.
    """
    data_root = Path(ds_cfg.data_root)
    mol_cfg = ds_cfg.get("molecule", "benzene")

    if mol_cfg == "all":
        candidates: list[str] = list(_MD17_MOLECULES)
    elif isinstance(mol_cfg, str):
        candidates = [mol_cfg]
    else:
        candidates = [str(m) for m in mol_cfg]

    resolved: list[str] = []
    for raw in candidates:
        name = raw[6:] if raw.startswith("rmd17_") else raw
        if name not in _MD17_MOLECULES:
            raise ValueError(f"Unknown MD17 molecule {raw!r}. Available: {sorted(_MD17_MOLECULES)}")
        if (data_root / name).is_dir():
            resolved.append(name)
        else:
            _logger.warning(
                "MD17 molecule %s: directory %s not found, skipping.",
                name,
                data_root / name,
            )

    if not resolved:
        raise FileNotFoundError(
            f"No MD17 molecule subdirectories found under {data_root}. Requested: {candidates}."
        )
    return resolved


def build_md17_loaders(cfg: DictConfig) -> list[LoaderSet]:
    """Build MD17 LoaderSets for the configured molecules and folds.

    Reads from ``cfg.dataset``:
        ``data_root``: root directory containing per-molecule subdirs.
        ``molecule``: a name, a list of names, or ``"all"``.
        ``k_folds``: number of CV folds (default ``3``). ``1`` selects the
            random train/test split.
        ``val_frac``: fraction of the training pool used for validation
            (default ``0.1``).
        ``max_train_samples``: cap on training-set size to avoid
            time-series correlation across consecutive frames (default
            ``1000``).
        ``test_samples``: size of the held-out test set when
            ``k_folds == 1`` (default ``1000``).
        ``debug_subset``: optional ``int`` that truncates train/test to
            the first ``N`` samples.

    Also reads ``cfg.seed`` for fold splitting (combined with the fold
    index for the train/val random split) and ``cfg.training.*`` keys
    for DataLoader options (via :func:`build_loader`).
    """
    from cliffordstf.data import LoaderSet  # local import: break circular dep

    ds_cfg = cfg.dataset
    seed = int(cfg.seed)

    molecules = _resolve_molecules(ds_cfg)
    k_folds = int(ds_cfg.get("k_folds", 3))
    val_frac = float(ds_cfg.get("val_frac", 0.1))
    debug_subset = ds_cfg.get("debug_subset", None)
    max_train_samples = int(ds_cfg.get("max_train_samples", 1000))

    fold_sets: list[LoaderSet] = []

    for molecule in molecules:
        full_dataset = MD17Dataset(root=ds_cfg.data_root, molecule=molecule)

        if k_folds == 1:
            test_samples = int(ds_cfg.get("test_samples", 1000))
            n = len(full_dataset)
            indices = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
            test_idx = indices[-test_samples:]
            train_idx = indices[:-test_samples]
            splits: list[tuple[int, list[int], list[int]]] = [(1, train_idx, test_idx)]
        else:
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
            splits = [
                (fold, list(tr), list(te))
                for fold, (tr, te) in enumerate(kf.split(range(len(full_dataset))), start=1)
            ]

        for fold, train_idx_, test_idx_ in splits:
            ds_train = cast("Dataset[object]", full_dataset[train_idx_])
            ds_test = cast("Dataset[object]", full_dataset[test_idx_])

            if len(ds_train) > max_train_samples:  # type: ignore[arg-type]
                _logger.info(
                    "MD17 %s: limiting training to %d samples (time-series correlation).",
                    molecule,
                    max_train_samples,
                )
                ds_train = cast("Dataset[object]", ds_train[:max_train_samples])

            if isinstance(debug_subset, int) and debug_subset > 0:
                ds_train = cast("Dataset[object]", ds_train[:debug_subset])
                ds_test = cast("Dataset[object]", ds_test[:debug_subset])

            n_train_pool = len(ds_train)  # type: ignore[arg-type]
            val_size = int(val_frac * n_train_pool)
            train_size = n_train_pool - val_size
            train_dataset, val_dataset = random_split(
                ds_train,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(seed + fold),
            )

            generator = torch.Generator().manual_seed(seed + fold)
            train_loader = build_loader(
                cast("Dataset[object]", train_dataset), cfg, shuffle=True, generator=generator
            )
            val_loader = build_loader(cast("Dataset[object]", val_dataset), cfg, shuffle=False)
            test_loader = build_loader(ds_test, cfg, shuffle=False)

            fold_sets.append(
                LoaderSet(
                    train=train_loader,
                    val=val_loader,
                    test=test_loader,
                    extra_parts=(molecule, f"fold{fold}"),
                    runtime_stats=MappingProxyType({}),
                    fold=fold,
                    k_folds=k_folds,
                )
            )

    return fold_sets


__all__ = ["build_md17_loaders"]
