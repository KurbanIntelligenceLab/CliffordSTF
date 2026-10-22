import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = json.load(open(os.path.join(ROOT, "data", "rmd17_multiseed.json")))
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
CFG = [
    ("cliffordstf_10M", "CliffordSTF-10M", "#1f4e9c", "o"),
    ("cliffordstf_1p9M", "CliffordSTF-1.9M", "#c47a1a", "s"),
]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
    }
)


def pearson(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den


fig, (a, b) = plt.subplots(
    1, 2, figsize=(6.9, 2.55), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.32}
)

for key, lab, col, mk in CFG:
    C = D["configurations"][key]
    er = [max(C[m]["stopping_epoch"]) / min(C[m]["stopping_epoch"]) for m in MOLS]
    rs = [100 * st.stdev(C[m]["force_mae"]) / st.mean(C[m]["force_mae"]) for m in MOLS]
    r = pearson(er, rs)
    a.scatter(
        er,
        rs,
        s=22,
        color=col,
        marker=mk,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
        label=f"{lab} (r = " + f"{r:.2f}".replace("-", "\u2212") + ")",
    )
    i = MOLS.index("benzene")
    ha = "left" if er[i] < 3 else "right"
    dx = 6 if er[i] < 3 else -6
    a.annotate(
        "benzene",
        (er[i], rs[i]),
        xytext=(dx, 0),
        textcoords="offset points",
        ha=ha,
        va="center",
        fontsize=6.5,
        color=col,
    )

a.set_xscale("log")
a.set_xticks([1, 2, 5, 10])
a.set_xticklabels(["1", "2", "5", "10"])
a.set_xlabel("Longest / shortest stopping epoch across seeds")
a.set_ylabel("Seed spread of force MAE (%)")
a.set_title("Seed spread and stopping-epoch variation", loc="left")
a.legend(frameon=False, loc="upper left", handletextpad=0.3)
a.margins(0.06)

for key, lab, col, mk in CFG:
    C = D["configurations"][key]["benzene"]
    pts = sorted(zip(C["stopping_epoch"], C["force_mae"]))
    if key == "cliffordstf_10M":
        b.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            color=col,
            lw=0.9,
            alpha=0.6,
            zorder=2,
        )
    b.scatter(
        [p[0] for p in pts],
        [p[1] for p in pts],
        s=24,
        color=col,
        marker=mk,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
        label=lab,
    )

b.set_xlabel("Stopping epoch")
b.set_ylabel("Benzene force MAE (kcal mol$^{-1}$ Å$^{-1}$)")
b.set_title("Benzene: error and stopping epoch", loc="left")
b.legend(frameon=False, loc="upper right", handletextpad=0.3)
b.margins(0.08)

for ax, letter in [(a, "a"), (b, "b")]:
    ax.text(
        -0.16,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
    )

fig.savefig(
    os.path.join(os.path.dirname(ROOT), "dd_template", "figures", "fig_seed.pdf"),
    bbox_inches="tight",
)
