"""Supplementary Figure 4: the Figure 6D result with three dsPCA-independent measures.

One row per method: a schematic of the method (A, C, E) and the result (B, D, F), in the format of Figure 6D.
  A-B  condition-averaged PCA (top 3 PCs of the 36 condition means)
  C-D  cross-validated Mahalanobis distance (crossnobis)
  E-F  cross-validated linear discriminability (shrinkage LDA, balanced accuracy)
Same neurons, trials, groups and statistics (ART mixed model, mouse as a random intercept) as Figure 6D.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
C_WT, C_PRE, C_POST = "#bebebe", "#1f77b4", "#800080"          # Fig. 6D colors
SF = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64]

# Fig. 6A schematic palette
C_ACT = "#f1dbc0"        # "original activity" box
C_SPACE = "#dfe8f2"      # reduced space (new; pastel like the Fig. 6A subspace boxes)
C_ARROW = "#4d4d4d"
C_ACCENT = "#c34b0a"     # Fig. 6A "Orthogonal"
C_TRAIN, C_TEST = "#bdbdbd", "white"
ORI_COL = plt.cm.turbo(np.linspace(0, 1, 7)[:-1])                 # Fig. 6 orientation colors
C_A, C_B = ORI_COL[3], ORI_COL[4]                                 # 90 and 120 deg

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                     "mathtext.bf": "Arial:bold"})

d = pd.read_csv(os.path.join(HERE, "geometry_metrics_long.csv"))
st = pd.read_csv(os.path.join(HERE, "stats_geometry_alternatives.csv")).set_index("metric")

# metric, y label, y range, y ticks, tick format, reference line
PANELS = [
    ("pca3", "Mean distance between\nadjacent orientations", (0.2, 0.9), np.arange(0.2, 0.91, 0.1), "{:.1f}", None),
    ("crossnobis", "Squared Mahalanobis\ndistance per neuron", (-0.01, 0.05), np.arange(-0.01, 0.051, 0.01),
     "{:.2f}", 0.0),
    ("lda_acc", "Decoding accuracy\n(adjacent orientations)", (0.45, 0.70), np.arange(0.45, 0.701, 0.05),
     "{:.2f}", 0.5),
]


# ------------------------------------------------------------------ data panels
def stars(p):
    for thr, s in [(1e-4, "****"), (1e-3, "***"), (1e-2, "**"), (0.05, "*")]:
        if p < thr:
            return s
    return "n.s."


def group_curve(metric, group, training):
    x = d[(d.metric == metric) & (d.group == group) & (d.training == training)]
    m = x.pivot_table(index="mouse", columns="sf", values="value").reindex(columns=SF)
    return m.mean(0).to_numpy(), (m.std(0, ddof=1) / np.sqrt(m.notna().sum(0))).to_numpy()


def comparison_block(ax, metric):
    """Fig. 6D-style comparison lines above the plot, group names in group colors."""
    p = st.loc[metric]
    lines = [[("WT/", "black"), ("AAV-Gαt2(pre)", C_PRE),
              (f": {stars(p.WT_vs_rescued_pre_p)}", "black")],
             [("WT/", "black"), ("AAV-Gαt2(post)", C_POST),
              (f": {stars(p.WT_vs_rescued_post_p)}", "black")],
             [("AAV-Gαt2(pre)", C_PRE), ("/", "black"), ("(post)", C_POST),
              (f": {stars(p.rescued_pre_vs_post_p)}", "black")]]
    rows = [HPacker(children=[TextArea(t, textprops=dict(color=c, fontsize=6.5)) for t, c in line],
                    align="baseline", pad=0, sep=0) for line in lines]
    box = AnchoredOffsetbox(loc="lower right", child=VPacker(children=rows, align="right", pad=0, sep=1.5),
                            bbox_to_anchor=(1.0, 1.03), bbox_transform=ax.transAxes,
                            frameon=False, pad=0, borderpad=0)
    ax.add_artist(box)


def draw_data(ax, spec):
    metric, ylabel, ylim, yticks, fmt, ref = spec
    if ref is not None:
        ax.axhline(ref, color="#808080", lw=0.6, ls="--", zorder=1)
    for group, training, col in (("WT", "pre", C_WT), ("Rescued", "post", C_POST), ("Rescued", "pre", C_PRE)):
        mean, sem = group_curve(metric, group, training)
        ax.errorbar(SF, mean, yerr=sem, color=col, mfc=col, ecolor="black", marker="o", ms=4.2, mec="black",
                    mew=0.5, lw=1.0, capsize=1.8, elinewidth=0.6, capthick=0.6, zorder=3)
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xlim(0.0168, 0.76)
    ax.set_xticks(SF)
    ax.set_xticklabels([f"{s:g}" for s in SF])
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels([fmt.format(t + 0.0).replace("-", "−") for t in yticks])
    ax.set_xlabel("Spatial frequency (cpd)")
    ax.set_ylabel(ylabel)
    comparison_block(ax, metric)


# ------------------------------------------------------------------ schematic helpers (units: inches)
SW, SH = 4.10, 1.95                                    # schematic panel size
E1, E2, E3 = (0.00, 1.22), (1.66, 2.84), (3.12, 4.00)  # x-extent of the three elements
BOX_Y = (0.10, 1.70)                                   # y-extent of the element boxes


def box(ax, xr, color, label=None):
    ax.add_patch(Rectangle((xr[0], BOX_Y[0]), xr[1] - xr[0], BOX_Y[1] - BOX_Y[0], facecolor=color,
                           edgecolor="none", zorder=0))
    if label:
        ax.text(np.mean(xr), BOX_Y[1] - 0.06, label, ha="center", va="top", fontsize=6.5)


def arrow(ax, x0, x1, y=0.90, above=None, below=None):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=7, color=C_ARROW, lw=1.0,
                                 shrinkA=0, shrinkB=0))
    xm = (x0 + x1) / 2
    if above:
        ax.text(xm, y + 0.07, above, ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    if below:
        ax.text(xm, y - 0.07, below, ha="center", va="top", fontsize=6)


def thin_arrow(ax, p0, p1, color=C_ARROW, lw=0.8, style="-|>", ms=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=4))


def setup(ax, title):
    ax.set_xlim(0, SW)
    ax.set_ylim(0, SH)
    ax.axis("off")
    ax.text(0, SH, title, ha="left", va="top", fontsize=7.5, fontweight="bold")


def neural_axes(ax, origin, length):
    ox, oy = origin
    thin_arrow(ax, (ox, oy), (ox + length, oy))
    thin_arrow(ax, (ox, oy), (ox, oy + length))
    ax.text(ox + length, oy - 0.04, "Neuron 1", ha="right", va="top", fontsize=5.5)
    ax.text(ox - 0.04, oy + length, "Neuron 2", ha="right", va="top", fontsize=5.5, rotation=90)


def clouds(ax, rng, centers, n=22, sd=(0.15, 0.045), angle=40):
    """Single-trial clouds with correlated noise (2-SD ellipses); returns the covariance."""
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    cov = R @ np.diag(np.square(sd)) @ R.T
    for c, col in zip(centers, (C_A, C_B)):
        pts = rng.multivariate_normal(c, cov, n)
        ax.scatter(pts[:, 0], pts[:, 1], s=3.5, color=col, edgecolor="#555555", linewidth=0.2, zorder=2)
        ax.add_patch(Ellipse(c, 4 * sd[0], 4 * sd[1], angle=angle, facecolor="none", edgecolor=col, lw=0.8,
                             zorder=3))
        ax.scatter(*c, s=16, color=col, edgecolor="black", linewidth=0.6, zorder=5)
    return cov


CLOUD_CENTERS = [(0.52, 0.86), (0.80, 0.62)]          # theta, theta + 30 deg (difference across the noise)


def cloud_labels(ax):
    ax.text(0.30, 1.08, "θ", color="black", fontsize=6.5, ha="center")
    ax.text(1.06, 0.46, "θ+30°", color="black", fontsize=6.5, ha="center")


# ------------------------------------------------------------------ schematics
def schematic_pca(ax):
    setup(ax, "Condition-averaged PCA")
    rng = np.random.default_rng(3)
    # E1: condition-averaged responses, 36 conditions x neurons
    box(ax, E1, C_ACT, "Condition-averaged\nresponses")
    n_neur = 14
    ori = np.repeat(np.arange(6), 6)
    sfi = np.tile(np.arange(6), 6)
    pref_o, pref_s = rng.integers(0, 6, n_neur), rng.uniform(0, 5, n_neur)
    dori = np.minimum(np.abs(ori[:, None] - pref_o[None]), 6 - np.abs(ori[:, None] - pref_o[None]))
    resp = np.exp(-dori ** 2 / 1.5) * np.exp(-(sfi[:, None] - pref_s[None]) ** 2 / 4) + rng.normal(0, 0.08, (36, n_neur))
    x0, x1, y0, y1 = 0.33, 1.13, 0.22, 1.28
    ax.imshow(resp, extent=(x0, x1, y0, y1), cmap="Greys", vmin=-0.1, vmax=1.4, aspect="auto",
              interpolation="nearest", zorder=1)
    h = (y1 - y0) / 36
    for k in range(36):
        ax.add_patch(Rectangle((x0 - 0.07, y1 - (k + 1) * h), 0.05, h, facecolor=ORI_COL[ori[k]], edgecolor="none",
                               zorder=1))
    ax.text((x0 + x1) / 2, y1 + 0.03, "Neurons", ha="center", va="bottom", fontsize=5.5)
    ax.text(x0 - 0.11, (y0 + y1) / 2, "36 conditions\n(6 orientations × 6 SFs)", ha="right", va="center",
            fontsize=5.5, rotation=90, multialignment="center")
    # arrow: PCA
    arrow(ax, E1[1] + 0.06, E2[0] - 0.06, above="PCA", below="no stimulus\nlabels")
    # E2: top 3 PCs with one SF's orientation ring
    box(ax, E2, C_SPACE, "Top 3 PCs")
    o = np.array([E2[0] + 0.22, 0.28])
    thin_arrow(ax, o, o + (0.88, 0))
    thin_arrow(ax, o, o + (0, 1.05))
    thin_arrow(ax, o, o + (0.40, 0.27))
    ax.text(*(o + (0.88, -0.04)), "PC1", ha="right", va="top", fontsize=5.5)
    ax.text(*(o + (-0.04, 1.05)), "PC2", ha="right", va="top", fontsize=5.5)
    ax.text(*(o + (0.43, 0.25)), "PC3", ha="left", va="center", fontsize=5.5)
    c, rx, ry, tilt = o + (0.56, 0.78), 0.30, 0.15, np.deg2rad(12)
    ang = np.deg2rad(np.arange(6) * 60 + 90)
    pts = np.c_[rx * np.cos(ang), ry * np.sin(ang)] @ np.array([[np.cos(tilt), np.sin(tilt)],
                                                                 [-np.sin(tilt), np.cos(tilt)]]) + c
    loop = np.vstack([pts, pts[:1]])
    ax.plot(loop[:, 0], loop[:, 1], color="#8c8c8c", lw=0.7, zorder=2)
    for k in range(6):
        ax.scatter(*pts[k], s=16, color=ORI_COL[k], edgecolor="black", linewidth=0.5, zorder=3)
    thin_arrow(ax, pts[3], pts[4], color=C_ACCENT, lw=0.9, style="<|-|>", ms=4)
    mid = (pts[3] + pts[4]) / 2
    out = (mid - c) / np.linalg.norm(mid - c)
    ax.text(*(mid + 0.08 * out), "d", color=C_ACCENT, fontsize=6.5, fontstyle="italic", ha="center",
            va="center")
    ax.text(c[0], c[1] + ry + 0.07, "one SF", ha="center", va="bottom", fontsize=5.5, color="#555555")
    # E3: measure
    arrow(ax, E2[1] + 0.06, E3[0] - 0.06)
    ax.text(np.mean(E3), 1.19, "Mean distance between\nadjacent orientations\n(30° apart), per SF",
            ha="center", va="top", fontsize=6.5, multialignment="center")
    ax.text(np.mean(E3), 0.68, r"$\bar{d}=\langle\Vert\mathbf{x}_{\theta}-\mathbf{x}_{\theta+30^\circ}\Vert\rangle_{\theta}$",
            ha="center", va="center", fontsize=7)


def schematic_crossnobis(ax):
    setup(ax, "Cross-validated Mahalanobis distance (crossnobis)")
    rng = np.random.default_rng(7)
    # E1: single trials of two adjacent orientations, correlated noise
    box(ax, E1, C_ACT, "Single-trial responses")
    neural_axes(ax, (0.16, 0.24), 1.0)
    clouds(ax, rng, CLOUD_CENTERS)
    cloud_labels(ax)
    ax.text(E1[1] - 0.04, 1.36, "Σ: trial-to-trial\nnoise covariance", ha="right", va="center", fontsize=5.5,
            multialignment="center")
    ax.plot([0.93, 0.76], [1.26, 1.07], color="#555555", lw=0.5)
    # arrow: split trials
    arrow(ax, E1[1] + 0.06, E2[0] - 0.06, above="Split", below="trials into\n5 folds")
    # E2: folds and fold-wise differences
    box(ax, E2, C_SPACE, "Fold-wise differences,\nsplit repeated 5 times")
    w, gap = 0.13, 0.035
    x_start = np.mean(E2) - (5 * w + 4 * gap) / 2 + 0.10
    for k in range(5):
        xk = x_start + k * (w + gap)
        ax.add_patch(Rectangle((xk, 1.08), w, 0.20, facecolor=C_A, edgecolor="#555555", lw=0.4))
        ax.add_patch(Rectangle((xk, 0.81), w, 0.20, facecolor=C_B, edgecolor="#555555", lw=0.4))
        ax.text(xk + w / 2, 1.18, str(k + 1), ha="center", va="center", fontsize=5)
        ax.text(xk + w / 2, 0.91, str(k + 1), ha="center", va="center", fontsize=5)
        thin_arrow(ax, (xk + w / 2, 0.77), (xk + w / 2, 0.64), lw=0.6, ms=3.5)
        ax.text(xk + w / 2, 0.60, rf"$\Delta_{k + 1}$", ha="center", va="top", fontsize=6.5)
    ax.text(x_start - 0.05, 1.18, "θ", ha="right", va="center", fontsize=6.5)
    ax.text(x_start - 0.05, 0.91, "θ+30°", ha="right", va="center", fontsize=5.5)
    ax.text(np.mean(E2), 0.24, r"$\Delta_k$" + " = difference between the\ntwo orientations’ means in fold "
            + r"$k$", ha="center", va="center", fontsize=5.5, multialignment="center", color="#555555")
    # E3: estimator
    arrow(ax, E2[1] + 0.06, E3[0] - 0.06)
    ax.text(np.mean(E3), 1.14, "Noise-whitened\nsquared distance", ha="center", va="top", fontsize=6.5,
            multialignment="center")
    ax.text(np.mean(E3), 0.73, r"$d^{2}=\langle\Delta_{k}^{\top}\Sigma^{-1}\Delta_{l}\rangle_{k\neq l}\,/\,N$",
            ha="center", va="center", fontsize=7)


def schematic_lda(ax):
    setup(ax, "Cross-validated linear decoding (LDA)")
    rng = np.random.default_rng(7)
    # E1: single trials with the LDA boundary
    box(ax, E1, C_ACT, "Single-trial responses")
    neural_axes(ax, (0.16, 0.24), 1.0)
    cov = clouds(ax, rng, CLOUD_CENTERS)
    cloud_labels(ax)
    m0, m1 = np.array(CLOUD_CENTERS[0]), np.array(CLOUD_CENTERS[1])
    wv = np.linalg.solve(cov, m1 - m0)
    wv /= np.linalg.norm(wv)
    mid = (m0 + m1) / 2
    tang = np.array([-wv[1], wv[0]])
    b0, b1 = mid - 0.42 * tang, mid + 0.42 * tang
    ax.plot([b0[0], b1[0]], [b0[1], b1[1]], color="black", lw=0.9, ls=(0, (3, 2)), zorder=4)
    ax.text(E1[1] - 0.04, b1[1] + 0.05, "Decision\nboundary", ha="right", va="bottom", fontsize=5.5)
    # arrow: train / test
    arrow(ax, E1[1] + 0.06, E2[0] - 0.06, above="Train", below="on 4/5,\ntest on 1/5")
    # E2: 5-fold cross-validation scheme
    box(ax, E2, C_SPACE, "5-fold cross-validation,\nrepeated 3 times")
    w, h, gx, gy, y_top = 0.15, 0.13, 0.03, 0.05, 1.34
    x_start = np.mean(E2) - (5 * w + 4 * gx) / 2 + 0.04
    for r in range(5):
        y = y_top - r * (h + gy) - h
        for k in range(5):
            ax.add_patch(Rectangle((x_start + k * (w + gx), y), w, h, facecolor=C_TEST if k == r else C_TRAIN,
                                   edgecolor="#555555", lw=0.4))
        ax.text(x_start - 0.05, y + h / 2, f"{r + 1}", ha="right", va="center", fontsize=5)
    ax.add_patch(Rectangle((x_start + 0.10, 0.22), 0.10, 0.08, facecolor=C_TRAIN, edgecolor="#555555", lw=0.4))
    ax.text(x_start + 0.23, 0.26, "train", ha="left", va="center", fontsize=5.5)
    ax.add_patch(Rectangle((x_start + 0.52, 0.22), 0.10, 0.08, facecolor=C_TEST, edgecolor="#555555", lw=0.4))
    ax.text(x_start + 0.65, 0.26, "test", ha="left", va="center", fontsize=5.5)
    # E3: measure
    arrow(ax, E2[1] + 0.06, E3[0] - 0.06)
    ax.text(np.mean(E3), 1.18, "Accuracy on\nheld-out trials", ha="center", va="top", fontsize=6.5,
            multialignment="center")
    ax.text(np.mean(E3), 0.84, "Balanced accuracy,\nchance = 0.5", ha="center", va="top", fontsize=6.5,
            multialignment="center")


# ------------------------------------------------------------------ layout (inches)
W, H = 1.67, 1.17                                   # data plot area, as in Fig. 6D
FIG_W = 6.90
DATA_LEFT = FIG_W - 0.25 - W
SCH_LEFT = 0.22
BOTTOM, PITCH, TOP = 0.45, 2.34, 0.40
fig_h = BOTTOM + 2 * PITCH + H + 0.50 + TOP
fig = plt.figure(figsize=(FIG_W, fig_h), dpi=300)
schematics = [schematic_pca, schematic_crossnobis, schematic_lda]
letters = iter("ABCDEF")
for k, (spec, schem) in enumerate(zip(PANELS, schematics)):
    y_data = BOTTOM + (2 - k) * PITCH
    row_top = y_data + H + 0.50
    sax = fig.add_axes([SCH_LEFT / FIG_W, (row_top - SH) / fig_h, SW / FIG_W, SH / fig_h])
    schem(sax)
    dax = fig.add_axes([DATA_LEFT / FIG_W, y_data / fig_h, W / FIG_W, H / fig_h])
    draw_data(dax, spec)
    for x_letter in (SCH_LEFT - 0.20, DATA_LEFT - 0.64):
        fig.text(x_letter / FIG_W, row_top / fig_h, next(letters), ha="left", va="top", fontweight="bold",
                 fontsize=9)

handles = [Line2D([], [], color=c, marker="o", mfc=c, mec="black", mew=0.5, ms=4.2, lw=1.0, label=l)
           for c, l in ((C_WT, "WT"), (C_PRE, "GtdKO+AAV-Gαt2 pre-training"),
                        (C_POST, "GtdKO+AAV-Gαt2 post-training"))]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(1 - 0.12 / FIG_W, 1 - 0.04 / fig_h), frameon=False,
           fontsize=6.5, ncol=3, handlelength=1.6, columnspacing=1.2)

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"SuppFig4.{ext}"))
print(f"saved ({FIG_W:.2f} x {fig_h:.2f} in)")
