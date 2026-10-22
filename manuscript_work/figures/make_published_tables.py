from inline_tables import update_tables

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = json.load(open(os.path.join(ROOT, "data", "published_comparison.json")))
PUB = D["published_oc"]
R17 = D["published_rmd17"]
STF = D["cliffordstf_oc"]
S17 = D["cliffordstf_rmd17"]
OCMODS = [
    "SchNet",
    "DimeNet++",
    "PaiNN",
    "TorchMD-Net",
    "NequIP",
    "GotenNet",
    "EquiformerV2 (L=2)",
    "EquiformerV2 (L=4)",
    "MACE (L=1)",
    "MACE (L=2)",
    "ICTP (L=1)",
    "ICTP (L=2)",
    "CliffordIP",
]
MOLS = [
    "aspirin",
    "azobenzene",
    "benzene",
    "ethanol",
    "malonaldehyde",
    "naphthalene",
    "paracetamol",
    "salicylic",
    "toluene",
    "uracil",
]
STFNAME = "\\textbf{CliffordSTF} (this work)"


def tex_name(n):
    return re.sub(r"\(L=(\d)\)", r"$L{=}\1$", n)


def ranked(vals, nd, higher):
    r = [round(v, nd) for v in vals]
    u = sorted(set(r), reverse=higher)
    return r, u[0], (u[1] if len(u) > 1 else None)


def fmt(v, sd, nd, best, second):
    a = f"{v:.{nd}f}"
    if v == best:
        a = f"\\textbf{{{a}}}"
    elif v == second:
        a = f"\\underline{{{a}}}"
    return f"{a} $\\pm$ {sd:.{nd}f}"


def column(task, i, nd, higher, names):
    vals = [
        STF[task]["values"][i][0] if n == "CliffordSTF" else PUB[task][n][i][0]
        for n in names
    ]
    sds = [
        STF[task]["values"][i][1] if n == "CliffordSTF" else PUB[task][n][i][1]
        for n in names
    ]
    r, b, sc = ranked(vals, nd, higher)
    return [fmt(r[k], sds[k], nd, b, sc) for k in range(len(names))]


def build_oc_main():
    spec = [
        ("OC20_S2EF", 0, 2, False),
        ("OC20_S2EF", 1, 2, False),
        ("OC20_S2EF", 2, 2, False),
        ("OC20_S2EF", 3, 2, False),
        ("OC20_S2EF", 4, 3, False),
        ("OC20_IS2RE", 0, 2, False),
        ("OC20_IS2RE", 3, 2, False),
        ("OC22_S2EF", 0, 2, False),
        ("OC22_S2EF", 2, 3, False),
    ]
    names = ["CliffordSTF"] + OCMODS
    cols = [column(t, i, nd, hi, names) for t, i, nd, hi in spec]
    L = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{\\textbf{OC20 and OC22 validation results} (mean $\\pm$ standard deviation over seeds; best value per column in bold, second best underlined). Baselines from ref.~\\citenum{polat2026cliffordip}; protocol in Section~\\ref{sec:benchmarks}. CliffordSTF averages all seeds (3 on OC20, 4 on OC22). Energy MAE in eV, force MAE in eV\\,\\AA$^{-1}$. Force cosine similarity, the remaining IS2RE splits, OC22 out-of-distribution energy and OC22 IS2RE are in ESI Tables~\\ref{E-tab:oc20_full}--\\ref{E-tab:oc22_full}.}",
        "\\label{tab:oc_main}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccccccccc}",
        "\\toprule",
        "& \\multicolumn{5}{c}{OC20 S2EF} & \\multicolumn{2}{c}{OC20 IS2RE} & \\multicolumn{2}{c}{OC22 S2EF} \\\\",
        "\\cmidrule(lr){2-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}",
        "Model & $e$ ID & $e$ OOD-Ads. & $e$ OOD-Cat. & $e$ OOD-Both & $f$ ID & ID & OOD-Both & $e$ ID & $f$ ID \\\\",
        "\\midrule",
    ]
    for r, n in enumerate(names):
        lab = STFNAME if n == "CliffordSTF" else tex_name(n)
        L.append(lab + " & " + " & ".join(c[r] for c in cols) + " \\\\")
        if n == "CliffordSTF":
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}"]
    return "\n".join(L) + "\n"


def oc_panel(task, title, heads, nds, his):
    names = ["CliffordSTF"] + OCMODS
    cols = [column(task, i, nds[i], his[i], names) for i in range(len(heads))]
    width = len(heads) + 1
    lines = [
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l"
        + "c" * len(heads)
        + "@{}}",
        r"\toprule",
        f"\\multicolumn{{{width}}}{{@{{}}l}}{{\\textbf{{{title}}}}}" + r" \\",
        r"\addlinespace[2pt]",
        "Model & " + " & ".join(heads) + r" \\",
        r"\midrule",
    ]
    for row, name in enumerate(names):
        label = STFNAME if name == "CliffordSTF" else tex_name(name)
        lines.append(label + " & " + " & ".join(col[row] for col in cols) + r" \\")
        if name == "CliffordSTF":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular*}"]
    return "\n".join(lines)


def combined_oc_table(dataset, seeds, panels, note=""):
    caption = (
        rf"\textbf{{{dataset} validation results.}} (a) S2EF energy MAE, force MAE and force cosine similarity; (b) IS2RE relaxed-energy MAE. Mean $\pm$ standard deviation; best value per column in bold, second best underlined. Baselines from ref.~\citenum{{polat2026cliffordip}}; CliffordSTF averages all {seeds} seeds. Energy MAE in eV and force MAE in eV\,\AA$^{{-1}}$."
        + note
    )
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{" + caption + "}",
        rf"\label{{tab:{dataset.lower()}_full}}",
        r"\fontsize{8.5}{10.5}\selectfont",
        r"\setlength{\tabcolsep}{3pt}",
        *(
            panel
            for i, content in enumerate(panels)
            for panel in ([r"\par\vspace{8pt}", content] if i else [content])
        ),
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def build_oc20_full():
    a = oc_panel(
        "OC20_S2EF",
        "(a) S2EF",
        [
            "$e_\\mathrm{MAE}$ ID",
            "$e_\\mathrm{MAE}$ OOD-Ads.",
            "$e_\\mathrm{MAE}$ OOD-Cat.",
            "$e_\\mathrm{MAE}$ OOD-Both",
            "$f_\\mathrm{MAE}$ ID",
            "$f_\\mathrm{cos}$ ID$\\uparrow$",
        ],
        [2, 2, 2, 2, 3, 2],
        [False] * 5 + [True],
    )
    b = oc_panel(
        "OC20_IS2RE",
        "(b) IS2RE",
        ["ID", "OOD-Ads.", "OOD-Cat.", "OOD-Both"],
        [2, 2, 2, 2],
        [False] * 4,
    )
    return combined_oc_table("OC20", 3, [a, b])


def build_oc22_full():
    names = ["CliffordSTF"] + OCMODS
    spec = [
        ("OC22_S2EF", 0, 2, False),
        ("OC22_S2EF", 1, 2, False),
        ("OC22_S2EF", 2, 3, False),
        ("OC22_S2EF", 3, 2, True),
        ("OC22_IS2RE", 0, 2, False),
        ("OC22_IS2RE", 1, 2, False),
    ]
    cols = [column(task, i, nd, hi, names) for task, i, nd, hi in spec]
    lines = [
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}",
        r"\toprule",
        r"& \multicolumn{4}{c}{\textbf{(a) S2EF}} & \multicolumn{2}{c}{\textbf{(b) IS2RE}} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
        r"Model & $e_\mathrm{MAE}$ ID & $e_\mathrm{MAE}$ OOD & $f_\mathrm{MAE}$ ID & $f_\mathrm{cos}$ ID$\uparrow$ & ID & OOD \\",
        r"\midrule",
    ]
    for row, name in enumerate(names):
        label = STFNAME if name == "CliffordSTF" else tex_name(name)
        lines.append(label + " & " + " & ".join(col[row] for col in cols) + r" \\")
        if name == "CliffordSTF":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular*}"]
    return combined_oc_table(
        "OC22",
        4,
        ["\n".join(lines)],
        r" One IS2RE seed diverged early (13.28\,eV ID); the other three give 4.09\,eV ID.",
    )


def mark(vals, nd):
    r = [round(v, nd) for v in vals]
    u = sorted(set(r))
    best = u[0]
    second = u[1] if len(u) > 1 else None
    out = []
    for v in r:
        t = f"{v:.{nd}f}"
        if v == best:
            t = f"\\textbf{{{t}}}"
        out.append(t)
    return out


def build_rmd17_permol():
    abbr = ["Asp", "Azo", "Bnz", "Eth", "Mal", "Nap", "Par", "Sal", "Tol", "Ura"]
    L = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{rMD17 test MAE per molecule.} Mean over four seeds; $f$: force MAE (eV\,\AA$^{-1}$), $e$: energy MAE (eV), Agg.: mean over ten molecules. Best value among the models shown per metric and column in bold. Baselines from ref.~\citenum{polat2026cliffordip}; protocol details are given in Section~\ref{sec:cliffordip} and ESI Section~\ref{E-supp:protocol}.}",
        r"\label{tab:rmd17_cip}",
        r"\fontsize{8.5}{9.6}\selectfont",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\renewcommand{\arraystretch}{1.02}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll" + "r" * 11 + r"@{}}",
        r"\toprule",
        "Model & & " + " & ".join(abbr) + r" & Agg. \\",
        r"\midrule",
    ]
    names = [n for n in R17 if not n.startswith("EquiformerV2")] + ["CliffordSTF"]

    def get(n, k):
        if n == "CliffordSTF":
            return [S17[f"{k}_mol"][m] for m in MOLS] + [S17[f"{k}_agg"]]
        return R17[n][k]

    names = sorted(
        names, key=lambda n: (round(get(n, "F")[-1], 3), round(get(n, "E")[-1], 3))
    )
    cells = {}
    for k in ["F", "E"]:
        cols = [mark([get(n, k)[c] for n in names], 3) for c in range(11)]
        for r, n in enumerate(names):
            cells[(n, k)] = [cols[c][r] for c in range(11)]
    for r, n in enumerate(names):
        for k, lab in [("F", "$f$"), ("E", "$e$")]:
            first = (STFNAME if n == "CliffordSTF" else tex_name(n)) if k == "F" else ""
            L.append(f"{first} & {lab} & " + " & ".join(cells[(n, k)]) + " \\\\")
        if r < len(names) - 1:
            L.append("\\addlinespace[1pt]")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    update_tables(
        "experiments",
        {"tab_rmd17_permol": build_rmd17_permol(), "tab_oc_main": build_oc_main()},
    )
    update_tables(
        "supplementary",
        {"tab_oc20_full": build_oc20_full(), "tab_oc22_full": build_oc22_full()},
    )
    print("ok")
