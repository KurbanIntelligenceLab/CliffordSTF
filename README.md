# CliffordSTF

Geometric algebra meets Cartesian tensors: higher-order equivariance for interatomic potentials

## Abstract

Clifford-algebra interatomic potentials use three-dimensional multivectors with scalar and vector angular channels (L = 0, 1). CliffordSTF adds symmetric traceless tensors of ranks two and three (L = 2, 3) through closed-form generation and contraction operations while retaining geometric-product message passing. On revised MD17, aggregate force cosine similarity increases from 0.051 to 0.507, a factor of 10.0, and force MAE decreases by 13.9% relative to CliffordIP. Force error is lower on all ten molecules; aggregate energy error is unchanged at the reported precision. Eleven variants plus the baseline show that the rank-2 edge descriptor supplies most of the directional improvement. OC20 results improve relaxed-energy and force prediction over CliffordIP, while OC22 results are mixed.

The primary experiments use direct force predictions. The separate 1.9M- and 10M-parameter rMD17 experiments use energy-gradient forces and different training recipes. Their results should be read as a joint change in capacity and training settings.

## Manuscript data

Baseline accuracy values come from the [published CliffordIP study](https://doi.org/10.1038/s41524-026-02259-8). The larger-model comparison uses the published reference values identified in ESI Table S7. Published values and our measurements are separate fields in the files below.

| Manuscript result | Data | Records and units |
| --- | --- | --- |
| Table 1: rMD17 energy and force MAE | [published_comparison.json](manuscript_work/data/published_comparison.json) | `published_rmd17` and `cliffordstf_rmd17`; energy in eV, force in eV/Å; means over four seeds and aggregates over ten molecules |
| Abstract and rMD17 force cosine | [cliffordip_published_and_allseed.json](manuscript_work/data/cliffordip_published_and_allseed.json) | `rmd17_all_seed_aggregates_meV`: `C` is cosine, `E` is meV, `F` is meV/Å; relative improvements use unrounded errors |
| Table 2 and ESI Table S2: ablations | [ablation_rmd17.json](manuscript_work/data/ablation_rmd17.json) | Twelve records with `name`, `flags`, and `[mean, standard deviation]` pairs: `f` in meV/Å, `e` in meV, `c` dimensionless; four molecules averaged within each of six seeds |
| Table 3 and ESI Tables S5 and S6: OC20 and OC22 | [published_comparison.json](manuscript_work/data/published_comparison.json) | `published_oc`, `cliffordstf_oc`, and `columns`; energy in eV, force in eV/Å; all three OC20 seeds and all four OC22 seeds included |
| ESI Tables S3 and S4: training and architecture settings | [source_results.json](manuscript_work/data/source_results.json) | `sheets` benchmark rows contain `model_config` and `training_config`; larger-model settings are specified in ESI Section S15 |
| ESI Tables S7 and S8 and Figure S1: larger rMD17 models and seed spread | [rmd17_multiseed.json](manuscript_work/data/rmd17_multiseed.json) | `configurations.cliffordstf_1p9M`, `configurations.cliffordstf_10M`, `seeds`, and `published_table4_force_mae`; 80 evaluations, force MAE in kcal/mol/Å, force cosine and stopping epochs |
| ESI Table S9: computational timing | [wallclock.csv](manuscript_work/data/wallclock.csv) | Per-model inference and training step means and standard deviations in seconds, batch size, parameter counts, and measurement settings |
| Computational-cost memory comparison | [mem_bench_result.json](manuscript_work/data/mem_bench_result.json) | `models` contains peak inference and training memory in MiB; hardware and batch settings are included |
| ESI Tables S10 and S11: cross-track stability | [cross_track_stability.json](manuscript_work/data/cross_track_stability.json) | `variants.stf_full`, `variants.stf_full_no_cross`, and `normalization_medians`; per-run energy MAE in meV, force cosine, loss ratios, and seed-mean energy errors |

[source_results.json](manuscript_work/data/source_results.json) also contains the per-seed primary benchmark and ablation records. Each sheet begins with its column names. The STF-only readout runs are in `sheets.NPJ_Ablation`. Model selection retains all seeds and completion statuses, including the early-diverging OC22 IS2RE run. [rmd17_summary.json](manuscript_work/data/rmd17_summary.json) contains the larger-model aggregate statistics.

Figure 1 describes the architecture, and ESI Table S1 gives analytical parity labels; neither represents measured data.

## Installation

Use Python 3.12:

```bash
uv sync --frozen
```

PyTorch Geometric extensions must match the installed PyTorch and accelerator versions. Baseline adapters are retired; published comparison results do not require baseline packages.

## Training and evaluation

Download the datasets separately and set `dataset.data_root`. Supported tasks are `md17`, `oc20_s2ef`, `oc20_is2re`, `oc22_s2ef`, and `oc22_is2re`. rMD17 expects `<data_root>/<molecule>/rmd17_<molecule>.npz`. OC20 and OC22 use LMDB splits under the data root.

For a primary-protocol rMD17 run:

```bash
uv run cliffordstf-train \
  dataset.name=md17 dataset.molecule=aspirin \
  dataset.data_root=/path/to/rmd17 \
  model.name=clifford_stf_full model.direct_forces=true \
  model.use_ema=false training.epochs=250 \
  training.batch_size=32 training.lr=1e-4 \
  training.optimizer=adam training.weight_decay=0 \
  training.grad_clip=10 training.early_stopping.patience=30 \
  seed=1 output_root=outputs
```

The primary rMD17 protocol uses a 900/100/1,000 train/validation/test split and seeds 1, 2, 3, and 4. OC20 uses seeds 1, 2, and 3; OC22 uses seeds 1, 2, 3, and 4. ESI Section S13 specifies the benchmark settings. The legacy trainer evaluates the best checkpoint on the test split after training and saves configuration snapshots, checkpoints, and metrics under `output_root`.

The larger rMD17 experiments use split 01, 950/50/1,000 configurations, seeds 42, 43, 44, and 45, and energy-gradient forces. Their model and optimizer settings are in ESI Section S15. The original launchers in `reproduction/` require the staged data bundle and CUDA environment used for those runs.

## Scientific checks

```bash
uv run pytest
```

The tests check Clifford algebra identities, STF projectors, rotational equivariance, energy invariance, permutation symmetry, and conservative-force consistency. Conservation checks apply to the energy-gradient model. Tests run manually; the repository has no CI or type-checking gate.

## Citation

```bibtex
@article{polat2026geometric,
  title={Geometric Algebra Meets Cartesian Tensors: Higher-Order Equivariance for Interatomic Potentials},
  author={Polat, Can and Serpedin, Erchin and Kurban, Mustafa and Kurban, Hasan},
  journal={arXiv preprint arXiv:2606.29584},
  year={2026}
}
```

## License

[MIT](LICENSE).
