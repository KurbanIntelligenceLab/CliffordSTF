"""QM9 dataset factory."""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import torch
from torch.utils.data import Dataset

from cliffordstf.data._qm9_io import _QM9Flat
from cliffordstf.data.loader import build_loader

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch_geometric.data import Data

    from cliffordstf.data import LoaderSet


_logger = logging.getLogger(__name__)

_DEFAULT_QM9_TARGETS: dict[str, int] = {
    "mu": 0,
    "alpha": 1,
    "homo": 2,
    "lumo": 3,
    "gap": 4,
    "r2": 5,
    "zpve": 6,
    "U0": 7,
    "U": 8,
    "H": 9,
    "G": 10,
    "Cv": 11,
}


class _QM9ScalarSubset(Dataset["Data"]):
    """Index a ``_QM9Flat`` and inject a z-score-normalized ``data.energy``.

    PyG's ``InMemoryDataset`` returns each sample with ``data.y`` of shape
    ``[1, 12]``. The trainer expects a scalar energy via ``data.energy``
    (or ``data.y.view(-1)`` as a fallback). We compute
    ``data.energy = (data.y[0, target_idx] - mean) / std`` per sample so
    the loss runs in normalized space; ``runtime_stats`` carries the
    inverse-transform parameters for metric reporting.
    """

    def __init__(
        self,
        base: _QM9Flat,
        indices: list[int],
        target_idx: int,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        self._base = base
        self._indices = list(indices)
        self._target_idx = target_idx
        self._mean = mean
        self._std = std

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Data:
        data = cast("Data", self._base[self._indices[idx]])
        target = data.y.view(-1)[self._target_idx]
        data.energy = ((target - self._mean) / self._std).view(1)
        return data


def _resolve_target(ds_cfg: DictConfig) -> tuple[str, int]:
    """Resolve ``(target_name, target_idx)`` from config + defaults."""
    target_name = str(ds_cfg.get("target_name", "U0"))
    target_map_raw = ds_cfg.get("qm9_targets", None)
    target_map: dict[str, int] = (
        {str(k): int(v) for k, v in dict(target_map_raw).items()}
        if target_map_raw
        else dict(_DEFAULT_QM9_TARGETS)
    )
    if target_name in target_map:
        target_idx = target_map[target_name]
    else:
        target_idx = int(ds_cfg.get("target_idx", 7))
    return target_name, target_idx


def build_qm9_loaders(cfg: DictConfig) -> list[LoaderSet]:
    """Build the QM9 train/val/test LoaderSet (single fold).

    Reads from ``cfg.dataset``:
        ``data_root``: directory containing ``qm9.pt``.
        ``target_name``: target name; defaults to ``"U0"``.
        ``qm9_targets``: optional ``{name: idx}`` map; defaults to the
            standard PyG 12-target ordering.
        ``target_idx``: fallback index when ``target_name`` is not in the
            ``qm9_targets`` map (default ``7``).
        ``debug_subset``: optional ``int`` truncating the dataset to the
            first ``N`` samples.

    Also reads ``cfg.seed`` for the 80/10/10 random split.

    Returns a single-element list with the LoaderSet carrying
    ``runtime_stats = {"mean", "std", "target_idx", "target_name"}``.
    """
    from cliffordstf.data import LoaderSet  # local import: break circular dep

    ds_cfg = cfg.dataset
    seed = int(cfg.seed)

    target_name, target_idx = _resolve_target(ds_cfg)
    debug_subset = ds_cfg.get("debug_subset", None)

    full_dataset = _QM9Flat(ds_cfg.data_root)
    n_total = len(full_dataset)
    if isinstance(debug_subset, int) and debug_subset > 0:
        n_total = min(n_total, debug_subset)

    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    perm = torch.randperm(len(full_dataset), generator=torch.Generator().manual_seed(seed)).tolist()
    perm = perm[:n_total]
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val : n_train + n_val + n_test]

    all_y = cast(torch.Tensor, full_dataset.data.y)
    y_train_raw = all_y[train_idx, target_idx].to(torch.float32)
    mean = y_train_raw.mean()
    std = y_train_raw.std().clamp_min(1e-12)

    train_ds = _QM9ScalarSubset(full_dataset, train_idx, target_idx, mean, std)
    val_ds = _QM9ScalarSubset(full_dataset, val_idx, target_idx, mean, std)
    test_ds = _QM9ScalarSubset(full_dataset, test_idx, target_idx, mean, std)

    generator = torch.Generator().manual_seed(seed)
    train_loader = build_loader(train_ds, cfg, shuffle=True, generator=generator)
    val_loader = build_loader(val_ds, cfg, shuffle=False)
    test_loader = build_loader(test_ds, cfg, shuffle=False)

    _logger.info(
        "QM9 target=%s idx=%d mean=%.6f std=%.6f (n_train=%d n_val=%d n_test=%d)",
        target_name,
        target_idx,
        float(mean.item()),
        float(std.item()),
        n_train,
        n_val,
        n_test,
    )

    return [
        LoaderSet(
            train=train_loader,
            val=val_loader,
            test=test_loader,
            extra_parts=(),
            runtime_stats=MappingProxyType(
                {
                    "mean": mean.detach().clone(),
                    "std": std.detach().clone(),
                    "target_idx": target_idx,
                    "target_name": target_name,
                }
            ),
            fold=1,
            k_folds=1,
        )
    ]


__all__ = ["build_qm9_loaders"]
