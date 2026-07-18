"""OC20 adsorbate-filtered subsets (CO2RR, NRR, C2 coupling).

The three reaction-pathway subsets in the source were three near-identical
files differing only by the adsorbate set and cache subdirectory name.
This module consolidates them into one parameterised builder plus three
frozen adsorbate sets.

Each builder filters OC20 S2EF samples by the system-id (``sid``) of
catalyst/adsorbate pairs listed in ``oc20_data_mapping.pkl``. The
caller is responsible for downloading that mapping from the OC20 docs:
https://fair-chem.github.io/catalysts/datasets/oc20.html
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

from torch.utils.data import Subset

from cliffordstf.data._oc20_io import OC20LMDBDataset

if TYPE_CHECKING:
    from torch_geometric.data import Data

_logger = logging.getLogger(__name__)

CO2RR_ADSORBATES: frozenset[str] = frozenset(
    {
        "*CO",
        "*COOH",
        "*CHO",
        "*CO2",
        "*CO₂",
        "*HCOO",
        "*COH",
        "*CH2O",
        "*H",
        "*OH",
        "*O",
        "*CH",
        "*CH2",
        "*C",
    }
)
"""CO2 reduction / C1 pathway intermediates."""

NRR_ADSORBATES: frozenset[str] = frozenset(
    {
        "*N",
        "*N2",
        "*NH",
        "*NH3",
        "*NHNH",
        "*N*NH",
        "*N*NO",
        "*NO",
        "*NO2",
        "*NO3",
        "*ONH",
        "*NONH",
        "*ONNH2",
    }
)
"""Nitrogen reduction / NOx intermediates."""

C2_ADSORBATES: frozenset[str] = frozenset(
    {
        "*C*C",
        "*CCH",
        "*CCH2",
        "*CCH3",
        "*CH*CH",
        "*CHCH2",
        "*CH2CH3",
        "*CCO",
        "*CHCO",
        "CH2*CO",
    }
)
"""C-C coupling pathway intermediates."""

_MAPPING_FILENAME = "oc20_data_mapping.pkl"
_MAPPING_DOC = "https://fair-chem.github.io/catalysts/datasets/oc20.html"


def _load_mapping(root: Path) -> dict[str, dict[str, object]]:
    mapping_path = root / _MAPPING_FILENAME
    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"OC20 mapping not found at {mapping_path}. "
            f"Download {_MAPPING_FILENAME} per {_MAPPING_DOC}."
        )
    with mapping_path.open("rb") as handle:
        return pickle.load(handle)  # type: ignore[no-any-return]


def _sids_for_adsorbates(
    mapping: dict[str, dict[str, object]], adsorbates: frozenset[str]
) -> set[str]:
    """Return LMDB ``sid`` strings whose ``ads_symbols`` are in ``adsorbates``."""
    matched: set[str] = set()
    for key, meta in mapping.items():
        if not isinstance(meta, dict) or meta.get("ads_symbols") not in adsorbates:
            continue
        key_str = str(key)
        if key_str.startswith("random"):
            matched.add(key_str[len("random") :])
        else:
            matched.add(key_str)
    return matched


def _build_indices(
    base: OC20LMDBDataset,
    valid_sids: set[str],
    cache_path: Path,
    *,
    max_collect: int | None,
    max_scan: int | None,
) -> list[int]:
    if cache_path.is_file():
        cached = [int(x) for x in cache_path.read_text().split()]
        if max_collect is not None and max_collect > 0:
            return cached[:max_collect]
        return cached

    indices: list[int] = []
    total = len(base)
    scan_limit = total if max_scan is None else min(total, max_scan)
    for i in range(scan_limit):
        if max_collect is not None and len(indices) >= max_collect:
            break
        if (i + 1) % 50000 == 0:
            _logger.info(
                "OC20 subset scan: %d / %d (%d matched)",
                i + 1,
                scan_limit,
                len(indices),
            )
        try:
            data: Data = base[i]
        except Exception:
            continue
        sid = getattr(data, "sid", None)
        if sid is None:
            continue
        sid_str = str(sid)
        if sid_str in valid_sids or (
            sid_str.startswith("random") and sid_str[len("random") :] in valid_sids
        ):
            indices.append(i)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n".join(str(i) for i in indices))
    if max_collect is not None and max_collect > 0:
        return indices[:max_collect]
    return indices


def build_oc20_adsorbate_subset(
    root: str | Path,
    split: str,
    adsorbates: frozenset[str],
    cache_subdir: str,
    max_samples: int | None = None,
) -> Subset[Data]:
    """Build an OC20 S2EF subset filtered to ``adsorbates``.

    Args:
        root: Directory containing the OC20 LMDB shards plus
            ``oc20_data_mapping.pkl``.
        split: OC20 split name (e.g. ``"train"``, ``"val_id"``).
        adsorbates: Frozen set of ``ads_symbols`` strings to keep.
        cache_subdir: Subdirectory under ``root`` for the indices cache
            (e.g. ``"co2rr"``, ``"nrr"``, ``"c2"``).
        max_samples: Optional cap on the returned subset. Small caps
            (< 10000) also bound the LMDB scan limit so smoke tests
            finish quickly.
    """
    root_path = Path(root)
    base = OC20LMDBDataset(root=root_path, task="s2ef", split=split, max_samples=None)
    mapping = _load_mapping(root_path)
    valid_sids = _sids_for_adsorbates(mapping, adsorbates)
    cache_path = root_path / cache_subdir / f"{split}_indices.txt"

    max_collect: int | None = (
        max_samples if (max_samples is not None and max_samples < 10000) else None
    )
    max_scan: int | None = 50000 if (max_collect is not None and max_collect < 1000) else None
    indices = _build_indices(
        base, valid_sids, cache_path, max_collect=max_collect, max_scan=max_scan
    )
    if max_samples is not None and max_samples > 0 and max_collect is None:
        indices = indices[:max_samples]
    if not indices:
        if max_samples is not None and max_samples < 100:
            indices = list(range(min(max_samples, len(base))))
        else:
            scan_str = str(max_scan) if max_scan is not None else str(len(base))
            raise ValueError(
                f"No samples in OC20 {split} matched adsorbates {sorted(adsorbates)} "
                f"(scanned up to {scan_str}). Try a larger split or raise max_scan."
            )
    return Subset(base, indices)


__all__ = [
    "C2_ADSORBATES",
    "CO2RR_ADSORBATES",
    "NRR_ADSORBATES",
    "build_oc20_adsorbate_subset",
]
