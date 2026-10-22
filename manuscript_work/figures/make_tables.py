from inline_tables import update_tables

import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = json.load(open(os.path.join(ROOT, "data", "rmd17_multiseed.json")))
PUB = D["published_table4_force_mae"]
BIG = D["configurations"]["cliffordstf_10M"]
SMALL = D["configurations"]["cliffordstf_1p9M"]
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
LABEL = {"salicylic": "Salicylic acid"}


def name(m):
    return LABEL.get(m, m.capitalize())


def ms(v):
    return f"{st.mean(v):.3f} $\\pm$ {st.stdev(v):.3f}"


def main_table():
    lines = []
    lines.append("\\begin{table*}[!htbp]")
    lines.append("\\centering")
    lines.append(
        "\\caption{\\textbf{rMD17 force MAE under a published training recipe} (kcal\\,mol$^{-1}$\\,\\AA$^{-1}$, test set, mean $\\pm$ standard deviation over four seeds). CliffordSTF-1.9M uses the released GotenNet rMD17 configuration; CliffordSTF-10M uses its own schedule (Section~\\ref{supp:published}). Both use split 01 and a force-weighted loss, whereas the published entries average five splits. Published values are taken from Table~4 of ref.~\\citenum{aykent2025gotennet}; ``best'' and ``widest'' are the lowest and highest of the eight published entries for each molecule. Force cosine similarity lies between 0.9954 and 1.0000 for all 80 CliffordSTF evaluations.}"
    )
    lines.append("\\label{tab:published}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccccccc}")
    lines.append("\\toprule")
    lines.append(
        "Molecule & CliffordSTF-10M & CliffordSTF-1.9M & Best published & Widest published & GotenNet & 10M/widest & 10M/GotenNet \\\\"
    )
    lines.append("\\midrule")
    xw, xg, mb, msm = [], [], [], []
    for m in MOLS:
        p = PUB[m]
        best_k = min(p, key=p.get)
        wide_k = max(p, key=p.get)
        a = st.mean(BIG[m]["force_mae"])
        w = a / p[wide_k]
        g = a / p["GotenNet"]
        xw.append(w)
        xg.append(g)
        mb.append(a)
        msm.append(st.mean(SMALL[m]["force_mae"]))
        wcell = f"\\textbf{{{w:.2f}}}" if w < 1 else f"{w:.2f}"
        lines.append(
            f"{name(m)} & {ms(BIG[m]['force_mae'])} & {ms(SMALL[m]['force_mae'])} & {p[best_k]:.4f} ({best_k}) & {p[wide_k]:.4f} ({wide_k}) & {p['GotenNet']:.4f} & {wcell} & {g:.2f} \\\\"
        )
    lines.append("\\midrule")
    lines.append(
        f"Median & {st.median(mb):.3f} & {st.median(msm):.3f} & & & & {st.median(xw):.2f} & {st.median(xg):.2f} \\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def seed_table():
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{\\textbf{Per-seed rMD17 test force MAE} (kcal\\,mol$^{-1}$\\,\\AA$^{-1}$) for both CliffordSTF configurations under the published-recipe protocol, with the stopping epoch of each run and the minimum force cosine similarity across the four seeds. ``Ratio'' is the longest divided by the shortest stopping epoch; ``rel.\\ std'' is the standard deviation divided by the mean.}"
    )
    lines.append("\\label{tab:seeds}")
    lines.append("\\footnotesize")
    lines.append("\\begin{tabular}{llccccccc}")
    lines.append("\\toprule")
    lines.append(
        "Config. & Molecule & Seed 42 & Seed 43 & Seed 44 & Seed 45 & Stopping epochs & Ratio & Rel.\\ std / min cos \\\\"
    )
    lines.append("\\midrule")
    for tag, C in [("10M", BIG), ("1.9M", SMALL)]:
        for i, m in enumerate(MOLS):
            v = C[m]["force_mae"]
            e = C[m]["stopping_epoch"]
            rel = 100 * st.stdev(v) / st.mean(v)
            lead = tag if i == 0 else ""
            cells = " & ".join(f"{x:.4f}" for x in v)
            lines.append(
                f"{lead} & {name(m)} & {cells} & {', '.join(str(x) for x in e)} & {max(e) / min(e):.2f} & {rel:.1f}\\% / {min(C[m]['force_cos']):.4f} \\\\"
            )
        if tag == "10M":
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def pearson(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den


def summary():
    out = {}
    for tag, C in [("10M", BIG), ("1.9M", SMALL)]:
        er = [max(C[m]["stopping_epoch"]) / min(C[m]["stopping_epoch"]) for m in MOLS]
        rs = [
            100 * st.stdev(C[m]["force_mae"]) / st.mean(C[m]["force_mae"]) for m in MOLS
        ]
        means = [st.mean(C[m]["force_mae"]) for m in MOLS]
        cos = [c for m in MOLS for c in C[m]["force_cos"]]
        out[tag] = {
            "median_force_mae": round(st.median(means), 4),
            "median_x_widest": round(
                st.median(
                    [st.mean(C[m]["force_mae"]) / max(PUB[m].values()) for m in MOLS]
                ),
                2,
            ),
            "median_x_gotennet": round(
                st.median(
                    [st.mean(C[m]["force_mae"]) / PUB[m]["GotenNet"] for m in MOLS]
                ),
                2,
            ),
            "mean_rel_std_pct": round(st.mean(rs), 1),
            "pearson_epoch_ratio_vs_rel_std": round(pearson(er, rs), 2),
            "force_cos_min": round(min(cos), 4),
            "force_cos_max": round(max(cos), 4),
        }
    return out


if __name__ == "__main__":
    update_tables(
        "supplementary", {"tab_published": main_table(), "tab_seeds": seed_table()}
    )
    json.dump(
        summary(), open(os.path.join(ROOT, "data", "rmd17_summary.json"), "w"), indent=1
    )
    print(json.dumps(summary(), indent=1))
