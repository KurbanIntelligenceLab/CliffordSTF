"""Molecule3D raw loader and per-split InMemoryDataset.

Molecule3D ships ~3.9M molecules as four SDF files plus a property CSV.
The base-test subset (~780k molecules) is preprocessed once into three
PyG cache files (``train.pt`` / ``val.pt`` / ``test.pt``) under
``{root}/{process_dir_base}_{split_mode}/``. Only the preprocessing step
needs RDKit + pandas; routine training reads the cached files.

Reference:
    Xu et al., "Molecule3D: A Benchmark for Predicting 3D Geometries
    from Molecular Graphs", arXiv:2110.01717.

Expected on-disk layout::

    {root}/raw/
        combined_mols_0_to_1000000.sdf
        combined_mols_1000000_to_2000000.sdf
        combined_mols_2000000_to_3000000.sdf
        combined_mols_3000000_to_3899647.sdf
        properties.csv
        random_split_inds.json     # or scaffold_split_inds.json
        random_test_split_inds.json
    {root}/processed_downstream_{split_mode}/
        train.pt
        val.pt
        test.pt
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal

import torch
from torch_geometric.data import Data, InMemoryDataset

_logger = logging.getLogger(__name__)

Split = Literal["train", "val", "test"]
SplitMode = Literal["random", "scaffold"]

_SDF_FILES: tuple[str, ...] = (
    "combined_mols_0_to_1000000.sdf",
    "combined_mols_1000000_to_2000000.sdf",
    "combined_mols_2000000_to_3000000.sdf",
    "combined_mols_3000000_to_3899647.sdf",
)


class Molecule3DProps(InMemoryDataset):  # type: ignore[misc]
    """Molecule3D property prediction dataset.

    Preprocesses only the ~780k base-test subset (per
    ``{split_mode}_split_inds.json``) and partitions it into
    train/val/test using ``{split_mode}_test_split_inds.json``. The
    ``target_id`` parameter selects one property column from
    ``properties.csv``.
    """

    def __init__(
        self,
        root: str | Path,
        split: Split = "train",
        split_mode: SplitMode = "random",
        target_id: int = 0,
        transform: Callable[[Data], Data] | None = None,
        pre_transform: Callable[[Data], Data] | None = None,
        pre_filter: Callable[[Data], bool] | None = None,
        process_dir_base: str = "processed_downstream",
        max_base_mols: int | None = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split!r}.")
        if split_mode not in ("random", "scaffold"):
            raise ValueError(f"Unknown split_mode {split_mode!r}.")
        self.split_mode = split_mode
        self.split = split
        self.target_id = target_id
        self.process_dir_base = process_dir_base
        self.max_base_mols = max_base_mols
        self._root = Path(root)

        super().__init__(str(self._root), transform, pre_transform, pre_filter)

        split_idx = {"train": 0, "val": 1, "test": 2}[split]
        loaded = torch.load(self.processed_paths[split_idx], map_location="cpu", weights_only=False)
        self.data, self.slices = loaded

    @property
    def raw_dir(self) -> str:
        return str(self._root / "raw")

    @property
    def processed_dir(self) -> str:
        return str(self._root / f"{self.process_dir_base}_{self.split_mode}")

    @property
    def raw_file_names(self) -> list[str]:
        return [
            *_SDF_FILES,
            "properties.csv",
            f"{self.split_mode}_split_inds.json",
            f"{self.split_mode}_test_split_inds.json",
        ]

    @property
    def processed_file_names(self) -> list[str]:
        return ["train.pt", "val.pt", "test.pt"]

    def _load_base_test_abs_indices(self) -> set[int]:
        path = Path(self.raw_dir) / f"{self.split_mode}_split_inds.json"
        with path.open() as handle:
            inds = json.load(handle)
        return set(inds["test"])

    def _load_downstream_split_indices(self) -> dict[str, list[int]]:
        path = Path(self.raw_dir) / f"{self.split_mode}_test_split_inds.json"
        with path.open() as handle:
            return json.load(handle)  # type: ignore[no-any-return]

    def _sdf_paths(self) -> list[Path]:
        return [Path(self.raw_dir) / name for name in _SDF_FILES]

    def _pre_process_base_test_subset(self) -> tuple[list[Data], dict[int, int]]:
        try:
            import pandas as pd  # type: ignore[import-untyped]
            from rdkit import Chem  # type: ignore[import-not-found]
            from tqdm import tqdm
        except ImportError as exc:
            raise RuntimeError(
                "Molecule3D preprocessing requires `rdkit`, `pandas`, and `tqdm`. "
                "Install RDKit via conda-forge (or a compatible wheel for your "
                "NumPy version) and rerun. Cached .pt files can be reused across "
                "environments without these deps."
            ) from exc

        target_df = pd.read_csv(Path(self.raw_dir) / "properties.csv")
        base_test_abs = self._load_base_test_abs_indices()

        data_list: list[Data] = []
        index_map: dict[int, int] = {}
        abs_idx = -1
        base_test_pos = -1
        seen = 0

        for i, sdf_path in enumerate(self._sdf_paths()):
            suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
            for j in tqdm(range(len(suppl)), desc=f"sdf {i + 1}/4", ncols=80):
                abs_idx += 1
                if abs_idx not in base_test_abs:
                    continue
                base_test_pos += 1
                mol = suppl[j]
                if mol is None:
                    continue

                coords = mol.GetConformer().GetPositions()
                atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
                props = torch.tensor(target_df.iloc[abs_idx, 1:].values, dtype=torch.float32)

                data = Data()
                data.z = torch.tensor(atomic_numbers, dtype=torch.long)
                data.pos = torch.tensor(coords, dtype=torch.float32)
                data.props = props
                data.y = props[self.target_id].view(1)

                index_map[base_test_pos] = len(data_list)
                data_list.append(data)
                seen += 1
                if (
                    isinstance(self.max_base_mols, int)
                    and self.max_base_mols > 0
                    and seen >= self.max_base_mols
                ):
                    return data_list, index_map

        skipped = (base_test_pos + 1) - len(data_list)
        if skipped > 0:
            _logger.info("Skipped %d molecules with RDKit parse failures.", skipped)
        return data_list, index_map

    def _subset(
        self, indices: Iterable[int], index_map: dict[int, int], base_list: list[Data]
    ) -> list[Data]:
        out: list[Data] = []
        skipped = 0
        for k in indices:
            mapped = index_map.get(k)
            if mapped is None:
                skipped += 1
                continue
            out.append(base_list[mapped])
        if skipped > 0:
            _logger.info("Dropped %d split indices (RDKit parse failures).", skipped)
        return out

    def process(self) -> None:
        downstream = self._load_downstream_split_indices()
        base_list, index_map = self._pre_process_base_test_subset()

        train_list = self._subset(downstream["train"], index_map, base_list)
        val_list = self._subset(downstream["valid"], index_map, base_list)
        test_list = self._subset(downstream["test"], index_map, base_list)

        if self.pre_filter is not None:
            train_list = [d for d in train_list if self.pre_filter(d)]
            val_list = [d for d in val_list if self.pre_filter(d)]
            test_list = [d for d in test_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            train_list = [self.pre_transform(d) for d in train_list]
            val_list = [self.pre_transform(d) for d in val_list]
            test_list = [self.pre_transform(d) for d in test_list]

        torch.save(self.collate(train_list), self.processed_paths[0])
        torch.save(self.collate(val_list), self.processed_paths[1])
        torch.save(self.collate(test_list), self.processed_paths[2])


__all__ = ["Molecule3DProps"]
