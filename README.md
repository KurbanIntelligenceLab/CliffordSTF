# CliffordSTF

A steerable tensor-field model on Clifford algebras for machine-learning
interatomic potentials (MLIPs).

This repository accompanies an anonymous submission. It contains the
`cliffordstf` package (model, training loop, dataset loaders, CLI), an
empty `baselines/` plug-in package for third-party baselines, and the
tooling needed to reproduce the experiments on supported datasets.

## Purpose

The package implements a two-track architecture that combines a
Clifford-algebra geometric track (`Cl(3,0)`) with a symmetric
tensor-field track (`STF₂`, `STF₃`), optionally fused via Hodge dual
forces, learned adaptive routing, and cross-track interactions. Three
packaged variants ship under `model.name`:

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

PyTorch is pulled in via the lock file. `torch_scatter` and
`torch_cluster` are built against the resolved `torch` version (see
`[tool.uv.extra-build-dependencies]` in `pyproject.toml`).

The optional `baselines` extra is reserved for third-party MLIP
baselines:

```bash
uv sync --extra baselines
```

It is currently empty; baselines land one at a time as the project
progresses.

## Data access

Raw datasets are external and **not committed** to this repository.
The 10 built-in dataset keys exposed by `cliffordstf.data.AVAILABLE_DATASETS`
are:

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
`src/cliffordstf/data/<key>.py` for the expected on-disk layout
(typically the upstream release format).

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
`configs/dataset/<name>.yaml`, the matching `configs/model/<name>.yaml`,
an optional `--config <path>.yaml` overlay, and finally any CLI
dot-overrides. Each fold/run writes a complete config snapshot and a
`manifest.json` capturing git SHA, config hash, seed, dataset/split
manifests, Python version, and hardware context under
`<output_root>/<model>/<dataset>/<task>/<seed>/<molecule>/foldN/`.

GPU is required to reach paper-grade results in a reasonable wall
clock. The full smoke test suite runs on CPU.

## Repository layout

```
src/cliffordstf/      package code (CODING_RULES.md §A)
  configs/            packaged base + dataset + model YAMLs
  data/               dataset factories (10 entries)
  io/config.py        OmegaConf-based config loader
  models/             CliffordSTF wrapper + algebra internals
  training/           trainer, optimizer/scheduler, losses, checkpointing
  cli.py              `cliffordstf-train` entrypoint
tests/                pytest suite (mirrors src/ layout)
baselines/            plug-in package for third-party baselines (skeleton)
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
- **Baselines skeleton.** `baselines/` is empty; third-party baseline
  ports land incrementally and are gated behind the `baselines` extra.
- **Pre-commit anonymity hook.** Runs `tools/check_anonymity.py`,
  which scans every git-tracked text file for a fixed list of source
  identifiers. If a commit is rejected by this hook, the offending
  lines are written to stderr.
