"""Figure 3N: correct choice rate in trials 1-400 and 401-1000 (left) and trials to the 70% criterion (right).

Bars, mean +/- SEM over mice; circles, individual mice. Exact two-sided permutation tests on per-mouse values
(Holm-adjusted over the two trial windows). Drawn in the style of the other Prism panels of Figure 3 (Arial 10 pt).
Inputs: mouse_level_values.csv and stats_primary_exact.csv (02), trials_to_criterion.csv and criterion_test.csv (03).
Outputs: Fig3N.pdf/.svg/.png and the per-mouse values in Prism table layout (Fig3N_data_for_Prism.xlsx).
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = ["WT", "Rescued"]
FILL = {"WT": "#d4d4d4", "Rescued": "#8babd3"}
TEXT = {"WT": "#000000", "Rescued": "#2f5597"}
LABEL = {"WT": "WT", "Rescued": "GtdKO+AAV-Gαt2"}
BAR_W, BAR_LW = 14.57, 1.456                 # pt, as in panel H
DOT, DOT_LW = 5.83, 0.728
DMIN = DOT + DOT_LW                          # centre distance at which two circles touch
ERR_LW, CAP = 0.728, 3.52                    # capsize = half the 7.04-pt cap
AX_LW, TICK = 1.456, 2.94
FS = 10
PAD = 8.0                                    # axis to outermost bar edge
GROUP_PITCH = 55.0
W, H = 262.0, 190.0                          # page (pt)
Y0, AH = 32.5, 118.0                         # baseline above the bottom edge; axis height

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": FS,
                     "pdf.fonttype": 42, "svg.fonttype": "none",
                     "axes.linewidth": AX_LW, "xtick.major.width": AX_LW, "ytick.major.width": AX_LW,
                     "xtick.major.size": TICK, "ytick.major.size": TICK, "xtick.major.pad": 2.5,
                     "ytick.major.pad": 3.0, "xtick.labelsize": FS, "ytick.labelsize": FS, "axes.labelsize": FS,
                     "axes.spines.top": False, "axes.spines.right": False})

# ------------------------------------------------------------------ data
mv = pd.read_csv(os.path.join(HERE, "mouse_level_values.csv"))
crit = pd.read_csv(os.path.join(HERE, "trials_to_criterion.csv"))
crit = crit[crit.group.isin(GROUPS)]
prim = pd.read_csv(os.path.join(HERE, "stats_primary_exact.csv"))
p_early = prim.loc[prim.test.str.contains("1-400"), "p_holm"].item()
p_late = prim.loc[prim.test.str.contains("401-1000"), "p_holm"].item()
p_crit = pd.read_csv(os.path.join(HERE, "criterion_test.csv"))["p_perm_two_sided"].item()


def stars(p):
    for thr, s in [(1e-4, "****"), (1e-3, "***"), (1e-2, "**"), (0.05, "*")]:
        if p < thr:
            return s
    return "n.s."


# ------------------------------------------------------------------ drawing helpers
fig = plt.figure(figsize=(W / 72, H / 72))


def add_axes(x0, width, ymax):
    """Axes whose box is the plotting area; x in pt from the y-axis, y in data units."""
    ax = fig.add_axes([x0 / W, Y0 / H, width / W, AH / H])
    ax.set_xlim(0, width)
    ax.set_ylim(0, ymax)
    for side in ("left", "bottom"):
        ax.spines[side].set_capstyle("projecting")
    return ax


def swarm(values, pt_per_unit):
    """Sideways offsets (pt) so circles do not overlap, symmetric about the bar centre (Prism column scatter)."""
    y = np.asarray(values, float) * pt_per_unit
    steps = [0.0] + [s * k * DMIN / 4 for k in range(1, 17) for s in (1, -1)]
    x = np.zeros(len(y))
    done = []
    for i in np.argsort(y):
        for c in steps:
            if all(np.hypot(c - x[j], y[i] - y[j]) >= DMIN * 0.999 for j in done):
                x[i] = c
                break
        done.append(i)
    # centre each cluster of touching circles on the bar, unless that would create an overlap
    n = len(y)
    adj = [[j for j in range(n) if j != i and np.hypot(x[i] - x[j], y[i] - y[j]) < DMIN * 1.05] for i in range(n)]
    seen = set()
    for i in range(n):
        if i in seen:
            continue
        comp, stack = [], [i]
        while stack:
            k = stack.pop()
            if k in seen:
                continue
            seen.add(k)
            comp.append(k)
            stack += adj[k]
        shifted = x.copy()
        shifted[comp] -= x[comp].mean()
        if all(np.hypot(shifted[a] - shifted[b], y[a] - y[b]) >= DMIN * 0.999
               for a in range(n) for b in range(a + 1, n)):
            x = shifted
    return x


def bar(ax, xc, vals, group, pt_per_unit):
    vals = np.asarray(vals, float)
    m, s = vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))
    ax.bar(xc, m, width=BAR_W, color=FILL[group], edgecolor="black", linewidth=BAR_LW, zorder=2)
    ax.errorbar(xc, m, yerr=s, fmt="none", ecolor="black", elinewidth=ERR_LW, capsize=CAP, capthick=ERR_LW,
                zorder=3)
    ax.plot(xc + swarm(vals, pt_per_unit), vals, "o", ms=DOT, mfc="none", mec="black", mew=DOT_LW, zorder=4,
            clip_on=False)
    return (vals.max() * pt_per_unit + DMIN / 2) / pt_per_unit           # top of the highest circle


def significance(ax, x1, x2, y, p):
    ax.plot([x1, x2], [y, y], color="black", lw=1.0, solid_capstyle="butt", clip_on=False, zorder=5)
    ax.annotate(stars(p), xy=((x1 + x2) / 2, y), xytext=(0, 0.6), textcoords="offset points", ha="center",
                va="bottom", fontsize=FS, annotation_clip=False)


# ------------------------------------------------------------------ left: correct choice rate by window
XL = 40.0
centres = [PAD + BAR_W, PAD + BAR_W + GROUP_PITCH]
wl = centres[-1] + BAR_W + PAD
ax = add_axes(XL, wl, 1.0)
ppu = AH / 1.0
ax.plot([0, wl], [0.7, 0.7], color="#ff0000", lw=1.0, dashes=(4, 3), zorder=2.5)
for c, col, p in zip(centres, ["early", "late"], [p_early, p_late]):
    tops = [bar(ax, c + (k - 0.5) * BAR_W, mv.loc[mv.group == g, col], g, ppu) for k, g in enumerate(GROUPS)]
    significance(ax, c - BAR_W / 2, c + BAR_W / 2, max(tops) + 4 / ppu, p)
ax.set_xticks(centres)
ax.set_xticklabels(["Trials\n1–400", "Trials\n401–1000"], linespacing=1.0)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_yticklabels([f"{v:.1f}" for v in np.arange(0, 1.01, 0.2)])
ax.set_ylabel("Correct choice rate", labelpad=4)

# ------------------------------------------------------------------ right: trials to criterion
XR = XL + wl + 64.0
wr = PAD + 2 * BAR_W + PAD
ax = add_axes(XR, wr, 500)
ppu = AH / 500
tops = [bar(ax, PAD + (k + 0.5) * BAR_W, crit.loc[crit.group == g, "trials_to_criterion"], g, ppu)
        for k, g in enumerate(GROUPS)]
significance(ax, PAD + 0.5 * BAR_W, PAD + 1.5 * BAR_W, max(tops) + 4 / ppu, p_crit)
ax.set_xticks([])
ax.set_yticks(np.arange(0, 501, 100))
ax.set_ylabel("Trials to 70% criterion", labelpad=4)

# ------------------------------------------------------------------ legend (outlined swatches, coloured text)
for k, g in enumerate(GROUPS):
    yc = H - 10 - 12 * k
    fig.add_artist(Rectangle(((XL - 28) / W, (yc - 3.97) / H), 15.69 / W, 7.94 / H, transform=fig.transFigure,
                             facecolor=FILL[g], edgecolor="black", lw=1.0))
    fig.text((XL - 28 + 15.69 + 4) / W, yc / H, LABEL[g], color=TEXT[g], ha="left", va="center", fontsize=FS)

for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
    fig.savefig(os.path.join(HERE, f"Fig3N.{ext}"), **kw)
print(f"saved {W:.0f} x {H:.0f} pt; p = {p_early:.4f}, {p_late:.3f}, {p_crit:.4f}")

# ------------------------------------------------------------------ per-mouse values in Prism table layout
with pd.ExcelWriter(os.path.join(HERE, "Fig3N_data_for_Prism.xlsx")) as xw:
    mice = [mv.loc[mv.group == g, "mouse"].tolist() for g in GROUPS]
    rows = [[""] + [LABEL[g] if i == 0 else "" for g, ms in zip(GROUPS, mice) for i in range(len(ms))],
            [""] + [m for ms in mice for m in ms]]
    for col, name in (("early", "Trials 1-400"), ("late", "Trials 401-1000")):
        rows.append([name] + [v for g in GROUPS for v in mv.loc[mv.group == g, col].tolist()])
    pd.DataFrame(rows).to_excel(xw, sheet_name="Correct choice rate (grouped)", header=False, index=False)
    c = {LABEL[g]: crit.loc[crit.group == g, "trials_to_criterion"].tolist() for g in GROUPS}
    m = {f"{LABEL[g]} mouse": crit.loc[crit.group == g, "mouse"].tolist() for g in GROUPS}
    pd.DataFrame({**c, **m}).to_excel(xw, sheet_name="Trials to criterion (column)", index=False)
    pd.DataFrame({"comparison": ["WT vs GtdKO+AAV-Gat2, trials 1-400", "WT vs GtdKO+AAV-Gat2, trials 401-1000",
                                 "WT vs GtdKO+AAV-Gat2, trials to 70% criterion"],
                  "test": ["exact two-sided permutation test, Holm-adjusted over the two windows"] * 2
                  + ["exact two-sided permutation test"],
                  "p": [p_early, p_late, p_crit], "label": [stars(p_early), stars(p_late), stars(p_crit)]}
                 ).to_excel(xw, sheet_name="Statistics (use these)", index=False)
