"""Peak-memory benchmark: base Clifford vs CliffordSTF at matched ~1M-param budget."""

from __future__ import annotations

import datetime as _dt


if not hasattr(_dt, "UTC"):
    _dt.UTC = _dt.timezone.utc

import json
import sys

import torch


try:
    import torch_geometric.nn.pool as _pyg_pool
    from torch_cluster import radius_graph as _tc_radius_graph

    _pyg_pool.radius_graph = _tc_radius_graph
except Exception as _e:
    print("radius_graph patch skipped:", _e)
from torch_geometric.data import Batch, Data


import cliffordstf.training.lightning_trainer  # noqa: F401
from cliffordstf.models import wrapper as _wrapper_mod
from cliffordstf.models.wrapper import CliffordSTFWrapper


try:
    from torch_cluster import radius_graph as _tc_rg2

    _wrapper_mod.radius_graph = _tc_rg2
except Exception:
    pass


ASPIRIN_Z = [8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1]
ASPIRIN_POS = [
    [1.7317, 0.2611, -0.0331],
    [3.0512, -1.5361, 0.5163],
    [-1.1610, 1.5518, 0.0844],
    [-2.4529, -0.2657, -0.4327],
    [0.5145, -0.4108, -0.0201],
    [0.4249, -1.8103, 0.0299],
    [-0.7947, -2.4741, 0.0459],
    [-1.9755, -1.7377, 0.0100],
    [-1.8917, -0.3446, -0.0430],
    [-0.6660, 0.3252, -0.0610],
    [2.8642, -0.3596, 0.1439],
    [4.0879, 0.4680, -0.0176],
    [-1.3722, 2.0, 0.9],
    [1.3376, -2.3854, 0.0555],
    [-0.8574, -3.5566, 0.0862],
    [-2.9243, -2.2559, 0.0233],
    [4.9, 0.0, 0.6],
    [4.3, 0.5, -1.08],
    [3.9, 1.45, 0.32],
    [-1.9, 1.6, -0.5],
    [0.9, 2.0, 0.5],
]


def make_aspirin_batch(batch_size: int, device: torch.device) -> Batch:
    z = torch.tensor(ASPIRIN_Z, dtype=torch.long)
    pos = torch.tensor(ASPIRIN_POS, dtype=torch.float)
    data_list = [Data(z=z.clone(), pos=pos.clone()) for _ in range(batch_size)]
    return Batch.from_data_list(data_list).to(device)


def build_models() -> dict[str, CliffordSTFWrapper]:

    base = CliffordSTFWrapper(
        n_atom_types=100,
        n_channels=80,
        n_interactions=5,
        n_rbf=50,
        cutoff=6.0,
        n_hidden_output=80,
        max_neighbors=50,
        use_l2=False,
        stf_mode="none",
        use_hodge_forces=False,
        use_adaptive_routing=False,
        routing_mode="none",
        use_cross_track=False,
        use_ema=False,
    )

    full = CliffordSTFWrapper(
        n_atom_types=100,
        n_channels=60,
        n_interactions=5,
        n_rbf=50,
        cutoff=6.0,
        n_hidden_output=60,
        max_neighbors=50,
        stf_mode="stf2+stf3",
        use_hodge_forces=True,
        use_adaptive_routing=True,
        routing_mode="learned",
        use_cross_track=True,
        use_self_interaction=False,
        use_gp_readout=False,
        use_ema=False,
    )
    return {"Clifford (base)": base, "CliffordSTF (full)": full}


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def measure(
    model: CliffordSTFWrapper,
    batch: Batch,
    device: torch.device,
    *,
    train: bool,
    warmup: int = 10,
    iters: int = 20,
) -> dict:
    is_cuda = device.type == "cuda"
    model = model.to(device)
    if train:
        model.train()
    else:
        model.eval()

    def one_step():
        if train:
            model.zero_grad(set_to_none=True)
            batch.pos.requires_grad_(False)
            energy, forces = model(batch)
            loss = energy.pow(2).mean() + forces.pow(2).mean()
            loss.backward()
        else:
            with torch.no_grad():
                model(batch)

    for _ in range(warmup):
        one_step()
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(iters):
        one_step()
    if is_cuda:
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        peak = float("nan")
    return {"peak_mem_MiB": peak}


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    torch.manual_seed(0)
    batch = make_aspirin_batch(batch_size, device)
    n_atoms = batch.z.numel()

    try:
        from torch_cluster import radius_graph as _rg

        ei = _rg(batch.pos, r=6.0, batch=batch.batch, max_num_neighbors=50)
        n_edges = int(ei.shape[1])
    except Exception:
        n_edges = -1
    results = {
        "device": device.type,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "batch_size": batch_size,
        "atoms_per_batch": int(n_atoms),
        "edges_per_batch": n_edges,
        "models": {},
    }
    for name, model in build_models().items():
        entry = {"n_params": n_params(model)}
        if device.type == "cuda":
            torch.cuda.empty_cache()
        entry["inference"] = measure(model, batch, device, train=False)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        entry["training"] = measure(model, batch, device, train=True)
        results["models"][name] = entry
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps(results, indent=2))
    with open("mem_bench_result.json", "w") as f:
        json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
