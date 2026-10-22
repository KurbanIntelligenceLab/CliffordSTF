"""Run train."""

import sys


try:
    from torch_cluster import radius_graph as _tc_rg
    import torch_geometric.nn.pool as _pool

    _pool.radius_graph = _tc_rg
    import torch_geometric.nn as _nn

    _nn.radius_graph = _tc_rg
except Exception as e:
    print("radius_graph patch skipped:", e)


import cliffordstf.training.lightning_trainer  # noqa: F401
from cliffordstf.cli import main

if __name__ == "__main__":
    rc = main(sys.argv[1:])
    print("TRAIN_RC", rc)
    sys.exit(rc or 0)
