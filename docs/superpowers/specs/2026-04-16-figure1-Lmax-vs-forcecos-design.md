# Figure 1 — $L_{\max}$ vs Force Cosine Similarity (MD17)

## Thesis

One figure, one sentence: **directional accuracy is governed by the maximum
angular-momentum order $L_{\max}$ of a model's features, not by other
architectural choices.** $L \leq 1$ models cluster near zero on force cosine
similarity; $L \geq 2$ models climb substantially higher. CliffordSTF adds STF
channels of rank $L \geq 2$ to a $\mathrm{Cl}(3,0)$ backbone and crosses the gap.

## Plot

- **X-axis:** integer $L_{\max} \in \{0, 1, 2, 3\}$. Small horizontal jitter so
  points at the same $L$ don't overlap. X-tick labels: `L=0`, `L=1`, `L=2`, `L=3`.
- **Y-axis:** mean test force cosine similarity across the 10 MD17 molecules.
  Range: roughly $[-0.05, 1.0]$. Linear scale.
- **One point per model.**
  - **Color** encodes the $L \leq 1$ vs $L \geq 2$ category (two colors from
    the Okabe–Ito palette). This reinforces the bifurcation both spatially
    (x-axis) and chromatically.
  - **Marker shape** encodes convergence: filled circle for converged, open
    circle for non-converged (MACE/ICTP on MD17, per README caveat).
  - **Clifford** and **CliffordSTF** use a distinct highlighted marker
    (filled star) and bold text labels.
  - Individual model names are printed next to their markers (no legend
    entry per model — the x position + label is enough).
- **Error bars:** $\pm 1$ std across the 10 MD17 molecules (molecule-level
  variance, not seed variance). Thin vertical lines only.
- **Annotation:** dashed curved arrow from Clifford ($L=1$) to CliffordSTF
  ($L=2$), labelled `+ STF (L $\geq$ 2)`. Makes the paper's contribution a
  visual moment rather than prose.
- **Group cue:** a faint horizontal line at each L column equal to the mean cos
  over converged models at that L. Shows the trend without adding clutter.
- **Shaded region:** light vertical band spanning $L \in [-0.5, 1.5]$ in a
  cool tint and $L \in [1.5, 3.5]$ in a warm tint. Keeps the bifurcation
  readable even in grayscale print.

## Model-to-$L_{\max}$ mapping (authoritative for this figure)

| Canonical key       | Paper label        | $L_{\max}$ | Converged on MD17? |
|---------------------|--------------------|------------|--------------------|
| `schnet`            | SchNet             | 0          | yes                |
| `dimenetpp`         | DimeNet++          | 0          | yes                |
| `faenet`            | FAENet             | 0          | yes                |
| `painn`             | PaiNN              | 1          | yes                |
| `torchmdnet`        | TorchMD-Net        | 1          | yes                |
| `visnet`            | ViSNet             | 1          | yes                |
| `nequip`            | NequIP             | 1          | yes                |
| `mace_l1`           | MACE ($L=1$)       | 1          | no (†)             |
| `ictp_l1`           | ICTP ($L=1$)       | 1          | no (†)             |
| `clifford`          | Clifford           | 1          | yes                |
| `mace_l2`           | MACE ($L=2$)       | 2          | no (†)             |
| `ictp_l2`           | ICTP ($L=2$)       | 2          | no (†)             |
| `gotennet`          | GotenNet           | 2          | yes                |
| `equiformer_v2`     | EquiformerV2       | 2          | yes                |
| `clifford_stf_full` | CliffordSTF        | 2          | yes                |
| `mace_l3`           | MACE ($L=3$)       | 3          | no (†)             |
| `ictp_l3`           | ICTP ($L=3$)       | 3          | no (†)             |

`clifford_stf` (the smaller ablation variant) is excluded.

(†) "Non-converged" = failed to train under the shared hyperparameter protocol
per README caveat #1. Shown as hollow markers; footnote in caption explains.

This table lives in `analysis/scripts/style.py` as `MODEL_L` and
`MODEL_CONVERGED_MD17`, so it's a single source of truth reusable by later
plots.

## Rendering details

- Uses `style.py` for colors, fonts, rcParams, and display names.
- Figure size default: `(4.0, 3.2)` inches (single-column figure).
- Output: PDF (primary) + PNG (preview) to `analysis/figures/`.
- Legend: one compact legend at the bottom or right — "family", "converged /
  non-converged", plus Clifford/CliffordSTF highlight markers. Family legend
  can be trimmed to the few families actually present; no need to label every
  individual model in the legend (labels go next to markers).
- Caption cue (informational; caption lives in LaTeX, not here):
  - Thesis statement (one line).
  - (†) note for hollow markers.
  - Reference to SI table for raw values.

## Deliverables

1. Extend `analysis/scripts/style.py` with:
   - `MODEL_L`: dict mapping canonical key → int.
   - `MODEL_CONVERGED_MD17`: dict mapping canonical key → bool.
   - `L_CATEGORY_COLORS`: `{'low': ..., 'high': ...}` — color per L category
     ($\leq 1$ and $\geq 2$).
2. New script `analysis/scripts/figure1_lmax_vs_cos.py`:
   - Reads MD17 sheet.
   - Filters out `clifford_stf`.
   - Computes per-model: mean & std of `test_force_cos_mean` across the 10
     molecules.
   - Plots the scatter per the spec above.
   - Emits `analysis/figures/figure1_lmax_vs_cos.pdf` + `.png`.
   - CLI args: `--figsize`, `--dpi`, `--output-dir`.
3. Keep existing scripts intact (`md17_latex_table.py`,
   `md17_per_molecule_plot.py` — the latter's per-molecule view still serves
   the SI even if it's no longer the main figure).

## Non-goals

- No MAE metrics in Figure 1. The paper's claim "comparable MAE while improving
  cos" is best shown by numbers in a table, not by overlaying MAE bars on the
  thesis figure. An MAE comparison figure, if needed, is a separate spec.
- No cross-benchmark aggregation. MD17-only keeps the figure honest to the
  abstract's 0.05 → 0.46 headline.
- No seed-level error bars. The 10-molecule variance is the more informative
  quantity here; seed std is in the xlsx if ever needed.

## Revision hooks

- If reviewers ask "does the L trend hold across benchmarks?", a v2 of this
  figure can aggregate cos across MD17 + OC20-S2EF + OC22-S2EF; the data is
  in sibling sheets.
- If EquiformerV2 was run at a different $L_{\max}$ (e.g. $L=6$), the mapping
  updates in one place (`style.py`) and the figure re-emits.
