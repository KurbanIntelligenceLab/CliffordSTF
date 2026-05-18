# CliffordSTF

**Geometric Algebra Meets Cartesian Tensors: Higher-Order Equivariance for Interatomic Potentials**

This repository accompanies an anonymous submission. It contains the
`cliffordstf` package (model, training loop, dataset loaders, CLI),
a `baselines/` plug-in package that registers the comparison
baselines used in the paper, and the tooling needed to reproduce the
experiments on supported datasets.

## Purpose

`Cl(3,0)` interatomic potentials predict force *magnitudes*
accurately but force *directions* poorly: across ten rMD17 molecules,
every `L ≤ 1` baseline in our twelve-model study attains aggregate
force-cosine similarity below `0.25`. The cause is structural — the
geometric product of two vectors in `ℝ³` realises only the `L=0` and
`L=1` pieces of its irrep content, leaving the symmetric-traceless
rank-2 component absent from the per-edge bilinear that drives every
message-passing layer.

`CliffordSTF` closes the gap by coupling the Clifford multivector to
closed-form symmetric-traceless tensor tracks at ranks two and three
through bilinear cross-track contractions. A single learned bilinear
does the fusion; the implementation uses **no Clebsch–Gordan tables,
no Wigner-D matrices, and no `e3nn` calls**.

Headline results (from the accompanying paper):

- **rMD17.** Aggregate force-cosine similarity rises from `0.055`
  (base Clifford) to `0.551` — an order-of-magnitude relative
  directional gain at improved magnitude accuracy (force MAE
  `15.8%` lower, energy MAE `10.9%` lower). CliffordSTF outperforms
  every CG-free or body-ordered baseline evaluated (all `≤ 0.17`).
- **Catalysis.** Best out-of-distribution S2EF energy MAE on OC22 in
  our study, and best in-distribution energy MAE among `L ≥ 2`
  methods on OC22 IS2RE.
- **Ablation.** An eleven-variant ablation confirms the two tracks
  are empirically complementary: neither alone approaches the
  hybrid.

Three packaged variants ship under `model.name`:

- `clifford_stf` — base variant.
- `clifford_stf_full` — ~1.06M parameters; adaptive routing + Hodge
  forces; primary rMD17 configuration.
- `clifford_stf_full_10m` — ~10.17M parameters; scaled-up counterpart
  used on OC20 / OC22.

## Installation

The project is managed with [uv](https://github.com/astral-sh/uv) and
pinned to Python 3.12. From the repository root:

```bash
uv sync                    # creates .venv/ and resolves uv.lock
uv run cliffordstf-train --help
```

PyTorch is pulled in via the lock file. `torch_scatter`, `torch_sparse`,
and `torch_cluster` are built against the resolved `torch` version
(see `[tool.uv.extra-build-dependencies]` in `pyproject.toml`).

The optional `baselines` extra pulls in the conflict-free third-party
MLIP baselines:

```bash
uv sync --extra baselines
```

Two baseline families need a dedicated environment because they pin
incompatible `e3nn` ranges:

```bash
uv pip install mace-torch       # mace_l1 / mace_l2 / mace_l3 / mace_l2_10m
uv pip install nequip           # nequip
uv pip install ictp             # ictp_l1 / ictp_l2 / ictp_l3
```

Each baseline test file is guarded by `pytest.importorskip(...)`, so
the suite stays green on any subset of installed extras.

## Data access

Raw datasets are external and **not committed** to this repository.
The 10 built-in dataset keys exposed by
`cliffordstf.data.AVAILABLE_DATASETS` are:

| Key | Task | Source |
| --- | --- | --- |
| `md17` | energy + forces | revised MD17 (rMD17) NPZ files |
| `qm9` | scalar properties | QM9 |
| `oc20_s2ef`, `oc20_is2re` | structure-to-energy/forces, IS-to-RE | Open Catalyst 2020 |
| `oc20_s2ef_co2rr`, `_nrr`, `_c2` | OC20 Tier-3 subsets | Open Catalyst 2020 |
| `oc22_s2ef`, `oc22_is2re` | s2ef, is2re | Open Catalyst 2022 |
| `molecule3d` | scalar properties | Molecule3D |

Each dataset loader resolves `cfg.dataset.data_root` against the
location of the raw archives. See the corresponding factory in
`src/cliffordstf/data/<key>.py` for the expected on-disk layout.

## Reproduce experiments

A training run is configured by composing packaged YAML defaults with
optional CLI dot-overrides:

```bash
uv run cliffordstf-train \
  dataset.name=md17 \
  model.name=clifford_stf_full \
  dataset.data_root=/path/to/rmd17 \
  dataset.molecule=aspirin \
  training.epochs=200 \
  training.batch_size=8
```

`load_config` merges, in order: `src/cliffordstf/configs/base.yaml`,
optional `defaults` injected by the trainer, the matching
`configs/dataset/<name>.yaml`, the matching `configs/model/<name>.yaml`
(plus `baselines/configs/model/<name>.yaml` when a baseline is
installed), an optional `--config <path>.yaml` overlay, and finally
any CLI dot-overrides. Each fold writes a config snapshot and a
`manifest.json` capturing git SHA, config hash, seed, dataset/split
manifests, Python version, and hardware context under
`<output_root>/<model>/<dataset>/<task>/<seed>/<molecule>/foldN/`.

Two training engines are wired behind `cfg.training.engine`:

- `legacy` (default) — `cliffordstf.training.trainer.train`, the
  hand-rolled loop with explicit EMA / AMP / checkpoint handling.
- `lightning` — `cliffordstf.training.lightning_trainer.train_lightning`,
  the same cfg surface driven through a `pl.Trainer` plus matching
  EMA / checkpoint / logs callbacks. Both engines write checkpoints
  in the same schema, so a fold trained under one engine can be
  resumed under the other.

GPU is required to reach paper-grade results in a reasonable wall
clock. The full test suite runs on CPU.

## Repository layout

```
src/cliffordstf/      package code (CODING_RULES.md §A)
  configs/            packaged base + dataset + model YAMLs
  data/               dataset factories (10 entries)
  io/config.py        OmegaConf-based config loader
  models/             CliffordSTF wrapper + algebra internals
  training/           legacy trainer, Lightning trainer + callbacks,
                      optim/scheduler/losses/checkpointing, EMA
  cli.py              `cliffordstf-train` entrypoint
tests/                pytest suite (mirrors src/ layout)
baselines/            plug-in package for third-party baselines
  clifford.py, schnet.py, dimenetpp.py, visnet.py, painn.py,
  torchmdnet.py, gotennet.py, faenet.py, nequip.py, mace.py, ictp.py
tools/                check_anonymity.py and other repo utilities
analysis/             paper-side artefacts (not part of the package)
```

## Citation

This repository accompanies an anonymous submission. A citation block
will be added on de-anonymisation.

## License

[MIT](LICENSE).

## Known limitations

- **Anonymous submission.** Author metadata, affiliations, and external
  links are omitted by design. The `LICENSE` copyright holder is
  intentionally redacted until de-anonymisation.
- **CPU smoke only.** The default test suite shrinks model and data
  scales aggressively so the suite finishes in seconds on CPU; the
  packaged YAMLs are sized for GPU training and are not exercised at
  full scale in CI.
- **`torch.compile` disabled.** All packaged model YAMLs ship with
  `use_compile: false`. `torch_scatter`'s FX-trace incompatibility and
  the dynamic-shape access pattern of the interaction layer block a
  clean compile path today. Revisit when `torch_scatter` exposes
  opaque custom-op registration.
- **Lightning resume restores model weights only.** The
  `cfg.training.engine=lightning` path auto-resumes from a legacy
  `ckpt_last.pth` for model state continuity but does not restore
  optimizer / scheduler state; use `cfg.training.engine=legacy` for
  bit-exact resume.
- **Baseline e3nn conflict.** `mace-torch` (pins `e3nn==0.4.4`) and
  `nequip` (needs `e3nn>=0.5.6`) cannot coexist; install the family
  you need in its own virtual environment.
- **Pre-commit anonymity hook.** Runs `tools/check_anonymity.py`,
  which scans every git-tracked text file for a fixed list of source
  identifiers. If a commit is rejected by this hook, the offending
  lines are written to stderr.
