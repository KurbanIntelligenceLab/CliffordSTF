# MD17 Per-Molecule Comparison Plot — Design

## Purpose

Paper figure (main body) summarizing MD17 results across 3 metrics (test force MAE,
test energy MAE, test force cosine similarity) for the top 6 baselines plus the two
Clifford variants. Complements the full 17-model LaTeX table, which moves to the
supplementary information.

## Data

- **Source:** `analysis/clifford_results_tables_v4.xlsx`, sheet `MD17`.
- **Materials:** all 10 MD17 molecules (`aspirin`, `azobenzene`, `benzene`, `ethanol`,
  `malonaldehyde`, `naphthalene`, `paracetamol`, `salicylic`, `toluene`, `uracil`).
- **Metrics:**
  - `test_force_mae_mean` — lower is better
  - `test_energy_mae_mean` — lower is better
  - `test_force_cos_mean` — higher is better (bounded ≈ [-1, 1])
- **Models included (8 total):**
  - **Baselines (top 6 by average force MAE across molecules):** EquiformerV2, FAENet,
    TorchMD-Net, SchNet, DimeNet++, PaiNN.
  - **Ours:** Clifford, CliffordSTF (= `clifford_stf_full` in the data).
- **Excluded:** `clifford_stf` (smaller variant), and the remaining 9 baselines
  (they belong to the SI table).

## Layout

2×5 grid of subplots, one per MD17 molecule (main content = 10 panels).

Each panel:

- X axis: 8 models, grouped. Order within a panel:
  `[EquiformerV2, FAENet, TorchMD-Net, SchNet, DimeNet++, PaiNN] | [Clifford, CliffordSTF]`
  with a thin dashed vertical divider separating the two groups.
- Per model: **3 adjacent bars** — force MAE, energy MAE, 1 − force cos.
- Y axis: normalized value in [0, 1], lower = better.
- Panel title: molecule name.

Shared elements:

- Single figure-level legend at the top: `force MAE`, `energy MAE`, `1 − force cos`
  (metric colors — e.g. blue / purple / pink), plus markers indicating the two
  Clifford models use a colored label strip under the bar group.
- Y-axis label on leftmost panels only (`normalized error`).
- All bars use the same 3 metric colors (one color per metric, consistent across
  models and panels). The Clifford/CliffordSTF group gets a light shaded background
  strip plus the dashed divider on its left, signaling "ours" without introducing
  a separate color scheme.

## Normalization

Per-molecule, per-metric normalization so that each panel's three metrics share
a [0, 1] axis. Performed **before** plotting.

For each `(molecule, metric)` pair, over the 8 models shown:

- **Force MAE, energy MAE** (log-min-max):
  `v' = (log(v) - log(min_v)) / (log(max_v) - log(min_v))`
- **Force cosine similarity**: converted to `1 - cos` and min-max normalized linearly:
  `v' = (1 - cos - min) / (max - min)`
  (cos is already bounded, and log is not meaningful for ≤ 0 values.)

Normalization range is computed over the **8 displayed models only**, not the full 17.
This keeps the figure self-contained: the highest bar in each panel/metric marks the
worst of the displayed models for that molecule, the lowest (zero) marks the best.

A footnote in the caption will state that raw values appear in the SI table.

## Visual design

- Paper style: clean, publication-ready. Matplotlib with `seaborn-v0_8-whitegrid`
  or equivalent; sans-serif font (default or Helvetica).
- Color palette: colorblind-safe categorical (e.g. matplotlib `tab10` subset or
  `colorbrewer`). Three metric colors consistent across all panels.
- Figure size: ~7.0 in × 3.2 in for a two-column layout (tunable). The implementation
  exposes `figsize` as a CLI arg.
- Output formats: PDF (primary, for LaTeX) and PNG (for quick viewing).

## Deliverables

1. `analysis/scripts/md17_per_molecule_plot.py` — the plotting script.
   - Reads the xlsx.
   - Filters and renames models per the rules above.
   - Applies the per-panel normalization.
   - Emits `analysis/figures/md17_per_molecule.pdf` and `.png`.
   - Accepts `--figsize`, `--output-dir`, and `--dpi` flags with sensible defaults.
2. (No changes to the existing `md17_latex_table.py`; that table stays as-is but is
   re-targeted for the SI — that's a caption concern, not a code concern.)

## Non-goals

- No error bars (seed-level std dev is in the xlsx but would clutter the bars;
  can be added in a revision if the paper asks for it).
- No per-molecule ranking / annotations.
- No interactive version.
- No change to which molecules appear — all 10.

## Open questions / revision hooks

- If reviewers want error bars, the script can add them from
  `test_*_mae_std` and `test_force_cos_std` columns (propagate through log
  transform for MAE metrics).
- If the 2×5 layout is too wide for a single-column figure, the same script can
  produce a 5×2 (tall) variant with a `--orientation` flag. Not implemented in
  v1 unless requested.
