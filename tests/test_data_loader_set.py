"""Tests for the ``LoaderSet`` frozen dataclass."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields, is_dataclass
from types import MappingProxyType
from typing import cast

import pytest
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader

from cliffordstf.data import LoaderSet


class _NumberDataset(Dataset[torch.Tensor]):
    def __init__(self, n: int = 4) -> None:
        self._items = [torch.tensor([i], dtype=torch.float32) for i in range(n)]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._items[idx]


def _make_loader() -> PyGDataLoader:
    return PyGDataLoader(_NumberDataset(2), batch_size=1)


def test_loader_set_is_frozen_dataclass_with_slots() -> None:
    assert is_dataclass(LoaderSet)
    assert "__slots__" in LoaderSet.__dict__


def test_loader_set_has_expected_fields() -> None:
    names = {f.name for f in fields(LoaderSet)}
    assert names == {
        "train",
        "val",
        "test",
        "extra_parts",
        "runtime_stats",
        "fold",
        "k_folds",
    }


def test_loader_set_defaults_are_safe() -> None:
    ls = LoaderSet(train=_make_loader(), val=_make_loader())
    assert ls.test is None
    assert ls.extra_parts == ()
    assert ls.fold == 0
    assert ls.k_folds == 1
    assert isinstance(ls.runtime_stats, Mapping)
    assert dict(ls.runtime_stats) == {}


def test_loader_set_runtime_stats_default_is_read_only() -> None:
    ls = LoaderSet(train=_make_loader(), val=_make_loader())
    with pytest.raises(TypeError):
        cast("dict[str, object]", ls.runtime_stats)["sneak_in"] = 1.0


def test_loader_set_attributes_cannot_be_reassigned() -> None:
    ls = LoaderSet(
        train=_make_loader(),
        val=_make_loader(),
        runtime_stats=MappingProxyType({"mean": 0.5}),
    )
    with pytest.raises(FrozenInstanceError):
        ls.fold = 7  # type: ignore[misc]
