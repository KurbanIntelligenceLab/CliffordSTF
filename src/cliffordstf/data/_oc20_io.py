"""OC20/OC22 LMDB dataset loader.

Reads LMDB shards from the Open Catalyst datasets and normalises the
entries to a PyG ``Data`` schema with ``z``, ``pos``, ``y``, ``force``,
``cell``, ``tags``, ``fixed``, ``sid``, ``fid``. Used by both S2EF and
IS2RE tasks.

References:
    OC20: https://fair-chem.github.io/catalysts/datasets/oc20.html
    OC22: https://fair-chem.github.io/catalysts/datasets/oc22.html
"""

from __future__ import annotations

import bisect
import contextlib
import pickle
from pathlib import Path
from typing import Any

import lmdb
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

# Documented but not Literal-enforced: cfg-driven splits flow through as plain
# strings (config files reference values like "val_ood_ads" that callers may
# extend without breaking the type hint).
Task = str
Split = str

_S2EF_TRAIN_SIZES: tuple[str, ...] = ("200k", "2M", "20M", "all")
_OC22_TASK_DIRS: dict[str, str] = {"s2ef": "s2ef-total", "is2re": "is2re-total"}


def _resolve_lmdb_dir(root: str | Path, task: str, split: str, oc22: bool = False) -> Path:
    """Resolve the LMDB directory across the three known layouts."""
    root_path = Path(root)
    candidates: list[Path] = [root_path / task / split]

    if task == "is2re" and not oc22:
        candidates.insert(0, root_path / task / "all" / split)
    if oc22 and task in _OC22_TASK_DIRS:
        candidates.insert(0, root_path / _OC22_TASK_DIRS[task] / split)
    if task == "s2ef":
        if split == "train":
            candidates.extend(root_path / task / size / split for size in _S2EF_TRAIN_SIZES)
        else:
            candidates.append(root_path / task / "all" / split)

    for cand in candidates:
        if cand.is_dir() and any(p.suffix == ".lmdb" for p in cand.iterdir()):
            return cand
    docs = (
        "OC22: https://fair-chem.github.io/catalysts/datasets/oc22.html"
        if oc22
        else "OC20: https://fair-chem.github.io/catalysts/datasets/oc20.html"
    )
    raise FileNotFoundError(
        f"LMDB directory not found for {task}/{split}. Tried: {candidates}\n"
        f"Download instructions: {docs}"
    )


def _as_tensor(value: object) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else torch.tensor(value)


def _is_atomic_data_like(obj: object) -> bool:
    """Duck-type FairChem/OCP ``AtomicData`` (non-PyG, non-dict)."""
    return all(hasattr(obj, attr) for attr in ("atomic_numbers", "pos", "energy", "forces"))


class OC20LMDBDataset(Dataset[Data]):
    """Map-style dataset over OC20/OC22 LMDB shards.

    Each LMDB shard is opened lazily on the first ``__getitem__`` so
    DataLoader workers can fork safely. ``_close_envs`` resets the
    handles before the parent forks; ``__getstate__`` closes envs
    automatically when the dataset is pickled into a worker.
    """

    def __init__(
        self,
        root: str | Path,
        task: Task = "s2ef",
        split: Split = "train",
        max_samples: int | None = None,
        oc22: bool = False,
    ) -> None:
        super().__init__()
        self.root = str(root)
        self.task = task
        self.split = split
        self.max_samples = max_samples

        self.lmdb_dir = _resolve_lmdb_dir(root, task, split, oc22=oc22)
        self.lmdb_paths: list[str] = sorted(
            str(p) for p in self.lmdb_dir.iterdir() if p.suffix == ".lmdb"
        )
        if not self.lmdb_paths:
            raise FileNotFoundError(
                f"No .lmdb files found in {self.lmdb_dir}. Extract the dataset first."
            )

        self._shard_lengths: list[int] = []
        self._cumulative_lengths: list[int] = []
        cumulative = 0
        for path in self.lmdb_paths:
            env = lmdb.open(
                path,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1,
            )
            with env.begin() as txn:
                raw_len = txn.get(b"length")
                n = pickle.loads(raw_len) if raw_len is not None else txn.stat()["entries"]
            env.close()
            self._shard_lengths.append(n)
            cumulative += n
            self._cumulative_lengths.append(cumulative)

        self._total_len = cumulative
        if max_samples is not None and max_samples > 0:
            self._total_len = min(self._total_len, max_samples)

        self._envs: list[lmdb.Environment | None] = [None] * len(self.lmdb_paths)

    def _get_env(self, shard_idx: int) -> lmdb.Environment:
        env = self._envs[shard_idx]
        if env is None:
            env = lmdb.open(
                self.lmdb_paths[shard_idx],
                subdir=False,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=256,
            )
            self._envs[shard_idx] = env
        return env

    def _global_to_shard(self, global_idx: int) -> tuple[int, int]:
        shard_idx = bisect.bisect_right(self._cumulative_lengths, global_idx)
        if shard_idx >= len(self._cumulative_lengths):
            raise IndexError(f"Global index {global_idx} out of range.")
        local_idx = global_idx - (self._cumulative_lengths[shard_idx - 1] if shard_idx > 0 else 0)
        return shard_idx, local_idx

    def __len__(self) -> int:
        return self._total_len

    def __getitem__(self, idx: int) -> Data:
        if idx < 0 or idx >= self._total_len:
            raise IndexError(f"Index {idx} out of range for dataset of size {self._total_len}.")
        shard_idx, local_idx = self._global_to_shard(idx)
        env = self._get_env(shard_idx)
        with env.begin() as txn:
            raw = txn.get(str(local_idx).encode("ascii"))
            if raw is None:
                raise KeyError(f"Key {local_idx} not found in shard {self.lmdb_paths[shard_idx]}.")
        return self._to_pyg_data(pickle.loads(raw))

    def _to_pyg_data(self, obj: object) -> Data:
        """Normalise an LMDB entry to the trainer's expected schema."""
        if isinstance(obj, Data):
            return _from_pyg_data(obj)
        if _is_atomic_data_like(obj):
            return _from_atomic_data(obj)
        if isinstance(obj, dict):
            return _from_dict(obj)
        raise TypeError(f"Unexpected LMDB entry type: {type(obj).__name__}.")

    def _close_envs(self) -> None:
        """Close every open LMDB environment so they reopen lazily.

        Called by ``build_loader`` via ``_lmdb_worker_init_fn`` so each
        DataLoader worker gets its own file descriptors after fork.
        """
        if not hasattr(self, "_envs"):
            return
        for i, env in enumerate(self._envs):
            if env is not None:
                with contextlib.suppress(Exception):
                    env.close()
                self._envs[i] = None

    def __getstate__(self) -> dict[str, Any]:
        self._close_envs()
        state = self.__dict__.copy()
        state["_envs"] = [None] * len(self.lmdb_paths)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def close(self) -> None:
        self._close_envs()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


def _from_pyg_data(obj: Data) -> Data:
    """Re-emit a stored PyG Data with the trainer's expected attribute set."""
    raw = dict(getattr(obj, "__dict__", {}))
    store = raw.get("_store")
    if store is not None and hasattr(store, "keys"):
        for key in store:
            if key not in raw:
                raw[key] = store[key]
    data = Data()
    if "atomic_numbers" in raw:
        data.z = _as_tensor(raw["atomic_numbers"]).long()
    elif "z" in raw:
        data.z = _as_tensor(raw["z"]).long()
    if "pos" in raw:
        data.pos = _as_tensor(raw["pos"]).float()
    if raw.get("y_relaxed") is not None:
        data.y = _as_tensor(raw["y_relaxed"]).float().view(-1)
    elif raw.get("y") is not None:
        data.y = _as_tensor(raw["y"]).float().view(-1)
    elif raw.get("energy") is not None:
        data.y = _as_tensor(raw["energy"]).float().view(-1)
    if raw.get("force") is not None:
        data.force = _as_tensor(raw["force"]).float()
    elif raw.get("forces") is not None:
        data.force = _as_tensor(raw["forces"]).float()
    if "cell" in raw:
        cell = _as_tensor(raw["cell"])
        data.cell = cell.squeeze(0).float() if cell.dim() == 3 else cell.float()
    if "tags" in raw:
        data.tags = _as_tensor(raw["tags"]).long()
    if "fixed" in raw:
        data.fixed = _as_tensor(raw["fixed"]).bool()
    if data.pos is None:
        raise ValueError("LMDB entry missing `pos`.")
    data.natoms = data.pos.size(0)
    data.sid = raw.get("sid", -1)
    data.fid = raw.get("fid", -1)
    if not hasattr(data, "z") or data.z is None:
        data.z = data.pos.new_zeros(data.natoms, dtype=torch.long)
    if not hasattr(data, "y") or data.y is None:
        data.y = data.pos.new_zeros(1)
    return data


def _from_atomic_data(obj: Any) -> Data:  # noqa: ANN401 - duck-typed FairChem AtomicData
    """Convert a FairChem/OCP-style AtomicData to PyG ``Data``."""
    data = Data()
    data.z = _as_tensor(obj.atomic_numbers).long()
    data.pos = _as_tensor(obj.pos).float()
    if getattr(obj, "energy", None) is not None:
        data.y = _as_tensor(obj.energy).float().view(-1)
    else:
        data.y = data.pos.new_zeros(1)
    if getattr(obj, "forces", None) is not None:
        data.force = _as_tensor(obj.forces).float()
    else:
        data.force = data.pos.new_zeros(data.pos.shape[0], 3)
    cell = _as_tensor(obj.cell)
    data.cell = cell.squeeze(0).float() if cell.dim() == 3 else cell.float()
    data.tags = _as_tensor(obj.tags).long()
    data.fixed = _as_tensor(obj.fixed).bool()
    data.natoms = data.pos.size(0)
    sid = getattr(obj, "sid", None)
    if isinstance(sid, int | str):
        data.sid = sid
    elif isinstance(sid, list | tuple) and sid:
        data.sid = sid[0]
    else:
        data.sid = -1
    return data


def _from_dict(raw: dict[str, Any]) -> Data:
    """Convert a dict-encoded entry to PyG ``Data``."""
    data = Data()
    data.z = torch.tensor(raw.get("atomic_numbers", raw.get("z")), dtype=torch.long)
    data.pos = torch.tensor(raw["pos"], dtype=torch.float32)
    if "y" in raw:
        data.y = torch.tensor([raw["y"]], dtype=torch.float32)
    elif "y_relaxed" in raw:
        data.y = torch.tensor([raw["y_relaxed"]], dtype=torch.float32)
    if "force" in raw:
        data.force = torch.tensor(raw["force"], dtype=torch.float32)
    elif "forces" in raw:
        data.force = torch.tensor(raw["forces"], dtype=torch.float32)
    if "cell" in raw:
        data.cell = torch.tensor(raw["cell"], dtype=torch.float32).view(3, 3)
    if "tags" in raw:
        data.tags = torch.tensor(raw["tags"], dtype=torch.long)
    if "fixed" in raw:
        data.fixed = torch.tensor(raw["fixed"], dtype=torch.bool)
    data.natoms = data.pos.size(0)
    data.sid = raw.get("sid", -1)
    data.fid = raw.get("fid", -1)
    return data


__all__ = ["OC20LMDBDataset"]
