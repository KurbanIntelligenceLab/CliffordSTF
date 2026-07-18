"""QM9 raw-file loader.

Loads pre-processed QM9 data from a flat ``qm9.pt`` archive. The full
XYZ-tar parsing pipeline is not ported because every consumer of this
loader uses the cached flat file. See the source dataset documentation
on figshare (``https://doi.org/10.6084/m9.figshare.978904``) for the
upstream layout.

Expected on-disk layout::

    {root} / qm9.pt  # collated (data, slices) tuple from PyG InMemoryDataset
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset


class _QM9Flat(InMemoryDataset):  # type: ignore[misc]
    """Flat QM9 reader: loads ``qm9.pt`` without download/process.

    The standard PyG ``InMemoryDataset.__init__`` walks ``raw_dir`` /
    ``processed_dir`` and triggers ``download`` and ``process`` when the
    cache is missing. This subclass bypasses that machinery entirely by
    calling ``torch.utils.data.Dataset.__init__`` directly and then
    populating ``self.data`` / ``self.slices`` from the cache file.
    """

    def __init__(self, root: str | Path) -> None:
        super(InMemoryDataset, self).__init__()
        self.transform = None
        self.pre_transform = None
        self.pre_filter = None
        pt_path = Path(root) / "qm9.pt"
        loaded = torch.load(pt_path, map_location="cpu", weights_only=False)
        self.data, self.slices = loaded


__all__ = ["_QM9Flat"]
