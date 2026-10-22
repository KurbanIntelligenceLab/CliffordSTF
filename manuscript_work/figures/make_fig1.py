"""Build the manuscript Figure 1 from editable equations and vector geometry."""

from __future__ import annotations
from pathlib import Path
from html import escape
import hashlib, re, shutil, subprocess, tempfile
import xml.etree.ElementTree as ET

OUTPUT = Path(__file__).resolve().parents[2] / "dd_template" / "figures" / "fig1.pdf"
CACHE = Path(tempfile.gettempdir()) / "cliffordstf-equation-cache"
CACHE.mkdir(exist_ok=True)
BLACK = "#202020"
GRAY = "#555555"
RULE = "#B8B8B8"
BLUE = "#0072B2"
ORANGE = "#B65F00"
NS = "http://www.w3.org/2000/svg"
EQ = {
    "decomp": r"u\otimes v=\underbrace{\tfrac13(u\cdot v)I}_{L=0}+\underbrace{\tfrac12(uv^{\mathsf T}-vu^{\mathsf T})}_{L=1}+\underbrace{\mathrm{STF}_2(u,v)}_{L=2}",
    "stf": r"\mathrm{STF}_2(u,v)=\tfrac12(uv^{\mathsf T}+vu^{\mathsf T})-\tfrac13(u\cdot v)I",
    "input": r"\{Z_i,\mathbf r_i\}_{i=1}^{N}",
    "edge2": r"Q_{ij}=\mathrm{STF}_2(\hat r_{ij},\hat r_{ij})",
    "state": r"h_i^{\mathrm{Cl}}\in\mathrm{Cl}(3,0)\otimes\mathbb R^C",
    "stfstate": r"S_i\in\mathrm{STF}_2\otimes\mathbb R^C",
    "generate": r"\mathrm{STF}_2(v_j,\hat r_{ij})",
    "contract": r"S_j\cdot v_j",
    "agg": r"M_i=\sum_{j\in\mathcal N(i)}m_{j\to i}",
    "energy": r"E=\sum_{t,i}\varepsilon_i^{(t)}",
    "inv": r"\varepsilon_i^{(t)}=\rho_t\!\left(\mathrm{Inv}(h_i^{(t)})\right)",
    "force": r"\langle h\rangle_1,\quad\star\langle h\rangle_2,\quad S\hat r",
    "forceout": r"\mathbf F_i",
}


def equation(tex):
    key = hashlib.sha256(tex.encode()).hexdigest()[:16]
    f = CACHE / (key + ".svg")
    if not f.exists():
        src = (
            r"\documentclass[border=0pt]{standalone}"
            + "\n"
            + r"\usepackage{amsmath,amssymb,newtxtext,newtxmath}"
            + "\n"
            + r"\begin{document}$\displaystyle "
            + tex
            + r"$\end{document}"
            + "\n"
        )
        (CACHE / (key + ".tex")).write_text(src)
        for cmd in [
            [
                shutil.which("latex") or "/Library/TeX/texbin/latex",
                "-interaction=batchmode",
                "-halt-on-error",
                key + ".tex",
            ],
            [
                shutil.which("dvisvgm") or "/Library/TeX/texbin/dvisvgm",
                "--no-fonts",
                "--exact-bbox",
                "-o",
                key + ".svg",
                key + ".dvi",
            ],
        ]:
            r = subprocess.run(cmd, cwd=CACHE, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(r.stdout + r.stderr)
    root = ET.fromstring(f.read_text())
    vb = [float(v) for v in root.attrib["viewBox"].split()]
    raw = "".join(ET.tostring(child, encoding="unicode") for child in root)

    raw = re.sub(r'(?<=id=")([^\"]+)', lambda m: key + "-" + m.group(1), raw)
    raw = re.sub(r'(?<=href=")#([^\"]+)', lambda m: "#" + key + "-" + m.group(1), raw)
    return raw, vb


class Figure:
    def __init__(self, name, h):
        self.name = name
        self.h = h
        self.count = 0
        self.p = [
            f'<svg xmlns="{NS}" xmlns:xlink="http://www.w3.org/1999/xlink" width="1400" height="{h}" viewBox="0 0 1400 {h}" role="img" aria-label="{name}"><rect width="1400" height="{h}" fill="white"/><defs>'
        ]
        for c in [BLACK, GRAY, BLUE, ORANGE]:
            self.p.append(
                f'<marker id="arrow{c[1:]}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L8 4 L0 8" fill="{c}"/></marker>'
            )
        self.p.append("</defs>")

    def text(self, x, y, s, size=23, weight=400, anchor="start", color=BLACK):
        self.p.append(
            f'<text x="{x}" y="{y}" font-family="Arial,Helvetica,sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">{escape(s)}</text>'
        )

    def panel(self, x, y, l, title):
        self.text(x, y, l, 27, 700)
        self.text(x + 35, y, title, 24)

    def path(self, d, color=BLACK, arrow=False, width=1.6, dash=""):
        self.p.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="square"'
            + (f' marker-end="url(#arrow{color[1:]})"' if arrow else "")
            + (f' stroke-dasharray="{dash}"' if dash else "")
            + "/>"
        )

    def rule(self, x, y, x2, y2, color=RULE, width=1):
        self.path(f"M{x} {y} L{x2} {y2}", color, False, width)

    def arrow(self, x, y, x2, y2, color=BLACK):
        self.path(f"M{x} {y} L{x2} {y2}", color, True)

    def rect(self, x, y, w, h, color=RULE):
        self.p.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="white" stroke="{color}" stroke-width="1.5"/>'
        )

    def math(self, x, y, key, size=28, maxw=None, anchor="middle"):
        tex = EQ.get(key, key)
        raw, vb = equation(tex)
        scale = size / 10
        if maxw:
            scale = min(scale, maxw / vb[2])
        w, h = vb[2] * scale, vb[3] * scale
        left = x - w / 2 if anchor == "middle" else x
        self.count += 1
        prefix = f"e{self.count}-"
        raw = re.sub(r'(?<=id=")([^\"]+)', lambda m: prefix + m.group(1), raw)
        raw = re.sub(r'(?<=href=")#([^\"]+)', lambda m: "#" + prefix + m.group(1), raw)
        self.p.append(
            f'<g class="equation" data-latex="{escape(tex, quote=True)}" data-size="{size}" transform="translate({left:.3f} {y - h / 2:.3f}) scale({scale:.5f}) translate({-vb[0]} {-vb[1]})" fill="{BLACK}">{raw}</g>'
        )

    def molecule(self, x, y):
        pts = [(-44, -22), (0, -47), (46, -18), (45, 35), (0, 60), (-46, 32)]
        for i, (a, b) in enumerate(pts):
            c, d = pts[(i + 1) % len(pts)]
            self.rule(x + a, y + b, x + c, y + d, GRAY, 1.8)
        for i, (a, b) in enumerate(pts):
            self.p.append(
                f'<circle cx="{x + a}" cy="{y + b}" r="{6 if i else 8}" fill="{BLACK if i else BLUE}"/>'
            )
        self.math(x + 65, y + 60, r"\mathbf r_i", 23)

    def save(self):
        import cairosvg

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        cairosvg.svg2pdf(
            bytestring=("".join(self.p) + "</svg>").encode(), write_to=str(OUTPUT)
        )


f = Figure("fig1", 780)
f.panel(30, 38, "a", "Angular decomposition")
f.math(720, 120, "decomp", 34, 1210)
f.rule(32, 210, 1368, 210)
f.panel(30, 254, "b", "Coupled features and node update")
f.panel(1007, 254, "c", "Readout")
f.rule(977, 237, 977, 650)
f.molecule(138, 356)
f.math(139, 453, "input", 28, 230)
f.math(139, 507, "edge2", 27, 244)
f.path("M260 411 H291 V345 H346", BLACK, True)
f.path("M291 411 V538 H346", BLACK, True)
f.text(359, 304, "Clifford", 23)
f.rule(359, 316, 715, 316, BLUE, 3)
f.rect(347, 330, 381, 75)
f.math(537, 367, "state", 30, 352)
f.text(359, 610, "STF rank 2; rank 3 optional", 21)
f.rule(359, 579, 715, 579, ORANGE, 3)
f.rect(347, 501, 381, 65)
f.math(537, 533, "stfstate", 30, 352)
f.arrow(380, 411, 380, 494, BLUE)
f.math(467, 438, "generate", 26, 195)
f.arrow(697, 495, 697, 411, ORANGE)
f.math(637, 477, "contract", 28, 112)
f.path("M728 367 H748 V445 H783", BLACK, True)
f.path("M728 533 H748 V445")
f.math(871, 399, "agg", 29, 184)
f.text(871, 465, "Augmented products", 21, anchor="middle")
f.text(871, 503, "Gate · residual", 21, anchor="middle")
f.text(871, 537, "Normalise", 21, anchor="middle")
f.text(920, 657, "Repeat across layers", 20, anchor="end")
f.text(1023, 312, "Invariant scalar head", 23)
f.math(1190, 369, "energy", 33, 319)
f.math(1190, 432, "inv", 29, 330)
f.text(1023, 495, "Direct force sources", 23)
f.math(1190, 548, "force", 32, 322)
f.arrow(1190, 578, 1190, 609)
f.math(1190, 638, "forceout", 34)
f.rule(32, 692, 1368, 692)
f.math(700, 738, "stf", 29, 1200)
f.save()
