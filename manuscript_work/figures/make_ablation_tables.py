from inline_tables import update_tables

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ROWS = json.load(open(os.path.join(ROOT, "data", "ablation_rmd17.json")))
MAIN = [
    "CliffordIP ($L{\\leq}1$)",
    "$L{=}2$ edge descriptor only",
    "Hodge only",
    "STF2+STF3",
    "Full no cross",
    "Full",
    "STF-only readout",
]
HEAD = [
    "& \\multicolumn{7}{c}{\\textbf{Configuration}} & \\multicolumn{3}{c}{\\textbf{Metrics}} \\\\",
    "\\cmidrule(lr){2-8}\\cmidrule(lr){9-11}",
    "Variant & $L{=}2$ edge & STF tracks & Hodge & Cross & Routing & Mode & Clifford readout & $f_\\mathrm{MAE}$ (meV\\,\\AA$^{-1}$) & $e_\\mathrm{MAE}$ (meV) & $f_\\mathrm{cos}\\uparrow$ \\\\",
]


def marks(rows):
    out = {}
    for key, nd, hi in [("f", 3, False), ("e", 3, False), ("c", 3, True)]:
        u = sorted(set(round(r[key][0], nd) for r in rows), reverse=hi)
        out[key] = (u[0], u[1] if len(u) > 1 else None, nd)
    return out


def cell(r, key, m):
    b, s, nd = m[key]
    v, sd = round(r[key][0], nd), r[key][1]
    a = f"{v:.{nd}f}"
    if v == b:
        a = f"\\textbf{{{a}}}"
    elif v == s:
        a = f"\\underline{{{a}}}"
    return f"{a} $\\pm$ {sd:.{nd}f}"


def table(rows, env, label, caption, split):
    m = marks(rows)
    L = (
        [
            f"\\begin{{{env}}}[{'t' if env == 'table*' else '!htbp'}]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{l cccccc c ccc}",
            "\\toprule",
        ]
        + HEAD
        + ["\\midrule"]
    )
    for k, r in enumerate(rows):
        if split and k == split:
            L.append("\\midrule")
        L.append(
            " & ".join(
                [r["name"]]
                + r["flags"]
                + [cell(r, "f", m), cell(r, "e", m), cell(r, "c", m)]
            )
            + " \\\\"
        )
    L += ["\\bottomrule", "\\end{tabular}}", f"\\end{{{env}}}"]
    return "\n".join(L) + "\n"


def main_table(rows):
    labels = {
        "CliffordIP ($L{\\leq}1$)": "CliffordIP baseline",
        "$L{=}2$ edge descriptor only": "Rank-2 edge only",
        "Hodge only": "Rank-2 edge + Hodge readout",
        "STF2+STF3": "STF tracks, no routing",
        "Full no cross": "Full, no cross or routing",
        "Full": "Full CliffordSTF",
        "STF-only readout": "STF-only readout",
    }
    best = {
        k: (max if k == "c" else min)(r[k][0] for r in ROWS) for k in ["c", "f", "e"]
    }

    def value(row, key):
        mean, sd = row[key]
        text = f"{mean:.3f}"
        if mean == best[key]:
            text = r"\textbf{" + text + "}"
        return text + r" $\pm$ " + f"{sd:.3f}"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{Key rMD17 ablations.} Mean $\pm$ standard deviation over six seeds, averaging aspirin, benzene, ethanol and salicylic acid within each seed; approximately $10^6$ parameters per variant. All variants except the baseline use the rank-2 edge descriptor. Bold marks the best mean in the full ablation; all eleven variants plus the baseline and their flags are in ESI Table~\ref{E-tab:ablation_full}.}",
        r"\label{tab:ablation}",
        r"\fontsize{8.5}{10}\selectfont",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrr@{}}",
        r"\toprule",
        r"Variant & STF node ranks & Force cosine $\uparrow$ & Force MAE (meV\,\AA$^{-1}$) & Energy MAE (meV) \\",
        r"\midrule",
    ]
    for row in rows:
        nodes = "2, 3" if row["flags"][1] == "stf2+stf3" else "None"
        lines.append(
            " & ".join(
                [labels[row["name"]], nodes]
                + [value(row, key) for key in ["c", "f", "e"]]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    by = {r["name"]: r for r in ROWS}
    main_rows = [by[n] for n in MAIN]
    cap_full = r"\textbf{Full rMD17 ablation} (eleven variants plus CliffordIP; mean $\pm$ standard deviation over six seeds on aspirin, benzene, ethanol and salicylic acid; best value per column in bold, second best underlined). Edge denotes the rank-2 edge descriptor; tracks denote STF node features; Hodge denotes the axial-vector readout; Cross denotes cross-track coupling. Routing and Mode specify the invariant gates; Clifford readout indicates whether Clifford features enter the force head."
    update_tables("experiments", {"tab_ablation": main_table(main_rows)})
    update_tables(
        "supplementary",
        {"tab_ablation_full": table(ROWS, "table", "tab:ablation_full", cap_full, 10)},
    )
    print("ok")
