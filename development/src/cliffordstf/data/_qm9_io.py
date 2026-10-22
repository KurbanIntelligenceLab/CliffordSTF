"""QM9 raw-file loader."""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset


class _QM9Flat(InMemoryDataset):  # type: ignore[misc]
    """Flat QM9 reader: loads ``qm9.pt`` without download/process."""

    def __init__(self, root: str | Path) -> None:
        super(InMemoryDataset, self).__init__()
        self.transform = None
        self.pre_transform = None
        self.pre_filter = None
        pt_path = Path(root) / "qm9.pt"
        loaded = torch.load(pt_path, map_location="cpu", weights_only=False)
        self.data, self.slices = loaded


__all__ = ["_QM9Flat"]
