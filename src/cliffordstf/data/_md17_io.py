"""MD17 raw-file loader.

Loads revised MD17 (rMD17) ``.npz`` files and emits PyG ``Data`` objects
with ``z``, ``pos``, ``energy``, and ``force`` attributes.

Dataset reference:
    Christensen & von Lilienfeld, "On the role of gradients for machine
    learning of molecular energies and forces", Machine Learning: Science
    and Technology, 2020.

Source: https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038

Expected on-disk layout (flat per molecule)::

    {root}/{molecule}/rmd17_{molecule}.npz   # raw numpy archive
    {root}/{molecule}/{molecule}.pt          # processed PyG cache (auto-created)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

if TYPE_CHECKING:
    from typing import Any

_logger = logging.getLogger(__name__)

_MD17_MOLECULES: Mapping[str, str] = MappingProxyType(
    {
        "benzene": "rmd17_benzene",
        "uracil": "rmd17_uracil",
        "naphthalene": "rmd17_naphthalene",
        "aspirin": "rmd17_aspirin",
        "salicylic": "rmd17_salicylic",
        "malonaldehyde": "rmd17_malonaldehyde",
        "ethanol": "rmd17_ethanol",
        "toluene": "rmd17_toluene",
        "paracetamol": "rmd17_paracetamol",
        "azobenzene": "rmd17_azobenzene",
    }
)
"""rMD17 molecule name -> raw ``.npz`` basename (without extension)."""


class MD17Dataset(InMemoryDataset):  # type: ignore[misc]
    """Revised MD17 dataset as a PyG ``InMemoryDataset``.

    Each sample is one geometry with a scalar energy and per-atom forces.
    The full rMD17 release contains 100k structures per molecule at the
    PBE/def2-SVP level. Use at most 1000 training samples to avoid time-
    series correlation between consecutive trajectory frames.
    """

    def __init__(
        self,
        root: str | Path,
        molecule: str = "benzene",
        transform: Callable[[Data], Data] | None = None,
        pre_transform: Callable[[Data], Data] | None = None,
        pre_filter: Callable[[Data], bool] | None = None,
    ) -> None:
        if molecule not in _MD17_MOLECULES:
            raise ValueError(f"Unknown molecule {molecule!r}. Available: {sorted(_MD17_MOLECULES)}")
        self.molecule = molecule
        self.source_name = _MD17_MOLECULES[molecule]
        self._mol_root = Path(root) / molecule

        super().__init__(str(self._mol_root), transform, pre_transform, pre_filter)
        loaded = torch.load(self.processed_paths[0], map_location="cpu", weights_only=False)
        self.data, self.slices = loaded

    @property
    def raw_dir(self) -> str:
        return str(self._mol_root)

    @property
    def processed_dir(self) -> str:
        return str(self._mol_root)

    @property
    def raw_file_names(self) -> list[str]:
        return [f"{self.source_name}.npz"]

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{self.molecule}.pt"]

    def process(self) -> None:
        npz_path = Path(self.raw_dir) / f"{self.source_name}.npz"
        raw: Any = np.load(npz_path)

        z = torch.tensor(raw["nuclear_charges"], dtype=torch.long)
        coords = raw["coords"]
        energies = raw["energies"]
        forces = raw["forces"]

        data_list: list[Data] = []
        for i in range(len(energies)):
            data = Data(
                z=z.clone(),
                pos=torch.tensor(coords[i], dtype=torch.float32),
                energy=torch.tensor([energies[i]], dtype=torch.float32),
                force=torch.tensor(forces[i], dtype=torch.float32),
            )
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
        _logger.info("MD17 %s: %d structures saved.", self.molecule, len(data_list))


__all__ = ["MD17Dataset"]
