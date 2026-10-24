"""Figures 4 and 5, drawn from the data.

Figure 4: WT and GtdKO+AAV-Gat2 mice before VR training (GtdKO mice in 4C). Figure 5: GtdKO+AAV-Gat2 mice before
and after VR training, with WT mice before training. Responsive fractions, single-neuron tuning curves aligned to
the cross-validated preferred orientation (absolute and normalized to 0 deg; descriptive), cross-validated OSI and
population decoding with the shuffled-label null (95% CI). Comparison statistics are written inside the plots.
Illustrations (Fig. 4A, B and the decoder schematic) are read from assets/.
Inputs: fig45_session_metrics.csv, fig45_cells.csv (01), fig45_stats.csv (02).
"""
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "pipeline")]
import config  # noqa: E402
ASSETS = os.path.join(HERE, "assets")
NULL = os.path.join(config.RESULTS_ROOT, "rescued_pre_shuffled_continuous_null_iter100_v3.npz")
SCALE_BAR_FRACTION = 30 / 251          # 100 um = 30 of the 251 px width of the example image

COL = {"WT": "#bebebe", "Pre": "#1f77b4", "Post": "#800080", "GtdKO": "#eb4f52"}
TEXT_COL = {"WT": "#8c8c8c", "Pre": "#1f77b4", "Post": "#800080", "GtdKO": "#eb4f52", "band": "#a6a6a6"}
NAME = {"WT": "WT", "Pre": "AAV-G\u03b1t2(pre)", "Post": "AAV-G\u03b1t2(post)"}
LEGEND_NAME = {"WT": "WT", "Pre": "GtdKO+AAV-G\u03b1t2 (pre-training)", "Post": "GtdKO+AAV-G\u03b1t2 (post-training)",
               "GtdKO": "GtdKO"}
BAND = "#dcdcdc"
SF6 = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
SF5 = SF6[:5]
ORI = [0, 30, 60, 90, 120, 150]
NN = list(range(10, 81, 10))
OFFSETS = [-60, -30, 0, 30, 60, 90]
PW, PH = 0.92, 0.92                    # plot area (inches)
EB = dict(marker="o", ms=3.6, mec="black", mew=0.5, lw=1.0, ecolor="black", elinewidth=0.6, capsize=1.6,
          capthick=0.6)

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6.5,
                     "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "legend.fontsize": 6.3,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic"})

d = pd.read_csv(os.path.join(HERE, "fig45_session_metrics.csv"))
cells = pd.read_csv(os.path.join(HERE, "fig45_cells.csv"))
st = pd.read_csv(os.path.join(HERE, "fig45_stats.csv"))


# ------------------------------------------------------------------ data helpers
def curve(metric, group, levels):
    x = d[(d.metric == metric) & (d.group == group)]
    w = x.pivot_table(index="mouse", columns="level", values="value")
    w = w[[c for lv in levels for c in w.columns if np.isclose(c, lv)]]
    return w.mean(0).to_numpy(), (w.std(0, ddof=1) / np.sqrt(w.notna().sum(0))).to_numpy()


def cell_curve(group, col):
    w = cells[cells.group == group].pivot(index="cell", columns="offset", values=col)[OFFSETS]
    return w.mean(0).to_numpy(), (w.std(0, ddof=1) / np.sqrt(len(w))).to_numpy()


def stars(metric, comparison, level=None):
    x = st[(st.metric == metric) & (st.comparison == comparison)]
    x = x[x.level.isna()] if level is None else x[np.isclose(pd.to_numeric(x.level, errors="coerce"), level)]
    return x.stars.item()


def null_band(metric):
    z = np.load(NULL, allow_pickle=True)
    sf_null = np.asarray(z["sf_by_orientation"], float)
    ori_null = np.asarray(z["orientation_by_sf"], float)
    s = {"sf_dec_vs_n": np.nanmean(sf_null, axis=2), "ori_dec_vs_n": np.nanmean(ori_null[:, :, 1:6], axis=2),
         "sf_dec_by_ori": sf_null[:, -1, :], "ori_dec_by_sf": ori_null[:, -1, 1:6]}[metric]
    mean = np.nanmean(s, 0)
    half = 1.96 * np.nanstd(s, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(s), 0))
    return mean - half, mean + half


def pptx_image(name):
    """Illustrations (Fig. 4A, B and the decoder schematic) from assets/."""
    return Image.open(os.path.join(ASSETS, name)).convert("RGBA")


# ------------------------------------------------------------------ drawing helpers
def series(ax, x, group, mean, sem, zorder=3, **overrides):
    # post-training error bars in purple
    style = dict(EB, ecolor=COL[group]) if group == "Post" else dict(EB)
    style.update(overrides)
    ax.errorbar(x, mean, yerr=sem, color=COL[group], mfc=COL[group], zorder=zorder, **style)


def stat_block(ax, lines, fontsize=6):
    """Comparison lines inside the plot, top right, group names in their colours ."""
    rows = [HPacker(children=[TextArea(t, textprops=dict(color=c, fontsize=fontsize)) for t, c in line],
                    align="baseline", pad=0, sep=0) for line in lines]
    ax.add_artist(AnchoredOffsetbox(loc="upper right", child=VPacker(children=rows, align="right", pad=0, sep=1.0),
                                    bbox_to_anchor=(1.0, 1.0), bbox_transform=ax.transAxes, frameon=False,
                                    pad=0, borderpad=0.15, zorder=5))


def vs(a, b, s):
    """'WT/AAV-Gat2(pre): *' style line."""
    return [(f"{NAME[a]}/", "black" if a == "WT" else COL[a]), (NAME[b], COL[b]), (f": {s}", "black")]


def pre_post(s):
    return [(NAME["Pre"], COL["Pre"]), ("/", "black"), ("(post)", COL["Post"]), (f": {s}", "black")]


def group_legend(ax, groups, band_label, **kw):
    """Line legend with group-coloured text, ."""
    handles = [Line2D([], [], color=COL[g], lw=1.2) for g in groups] + [Patch(color=BAND)]
    labels = [LEGEND_NAME[g] for g in groups] + [band_label]
    colors = [TEXT_COL[g] for g in groups] + [TEXT_COL["band"]]
    leg = ax.legend(handles, labels, frameon=False, handlelength=1.4, handletextpad=0.5, labelspacing=0.15,
                    borderaxespad=0, **kw)
    for t, c in zip(leg.get_texts(), colors):
        t.set_color(c)
    leg.set_clip_on(False)
    return leg


def yaxis(ax, lo, hi, step, fmt="{:.1f}", ticks=None):
    ax.set_ylim(lo, hi)
    ticks = np.round(np.arange(lo, hi + step / 2, step), 3) if ticks is None else np.asarray(ticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels([fmt.format(t).replace("-", "\u2212") for t in ticks])


def sf_axis(ax, levels):
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xlim(levels[0] / 1.12, levels[-1] * 1.12)
    ax.set_xticks(levels)
    ax.set_xticklabels([f"{v:g}" for v in levels], fontsize=5.5 if len(levels) > 5 else 6)
    ax.set_xlabel("Spatial frequency (cpd)")


def ori_axis(ax):
    ax.set_xlim(-12, 162)
    ax.set_xticks(ORI)
    ax.set_xlabel("Orientation (\u00b0)")


def n_axis(ax):
    ax.set_xlim(4, 86)
    ax.set_xticks(NN)
    ax.set_xlabel("Number of neurons used\nfor decoding")


def letter(fig, fig_h, x_in, y_top_in, s):
    fig.text(x_in / fig.get_figwidth(), y_top_in / fig_h, s, fontweight="bold", fontsize=9, ha="left", va="top")


def add_axes(fig, fig_h, x, y, w=PW, h=PH):
    return fig.add_axes([x / fig.get_figwidth(), y / fig_h, w / fig.get_figwidth(), h / fig_h])


# ------------------------------------------------------------------ panel functions
def panel_resp_sf(ax, groups, marks):
    """Responsive fraction by SF. marks: (comparison, colour, per-SF placement), where each placement is
    (anchor group, x factor, 'mean' or 'top' of the anchor, dy, ha, va)."""
    ax.axhspan(0, 0.05, color=BAND, lw=0, zorder=0)
    if "GtdKO" in groups:
        g = d[(d.metric == "resp_by_sf") & (d.group == "GtdKO")]
        for _, m in g.groupby("mouse"):
            m = m.sort_values("level")
            ax.plot(m.level, m.value, color=COL["GtdKO"], lw=0.8, alpha=0.85, zorder=1)
    means = {}
    for gname in [g for g in groups if g != "GtdKO"]:
        means[gname] = curve("resp_by_sf", gname, SF6)
        series(ax, SF6, gname, *means[gname])
    for comparison, color, placement in marks:
        for k, (sf, (anchor, fx, base, dy, ha, va)) in enumerate(zip(SF6, placement)):
            s = stars("resp_by_sf", comparison, sf)
            mean, sem = means[anchor]
            y = (mean[k] + sem[k] if base == "top" else mean[k]) + dy
            ax.text(sf * fx, y, s, color=color, ha=ha, va=va, fontsize=5.5 if s == "n.s." else 7, clip_on=False,
                    zorder=6)
    sf_axis(ax, SF6)
    yaxis(ax, 0, 1.0, 0.2)
    ax.set_ylabel("Responsive fraction")


def panel_session(ax, metric, groups, levels, axis, ylim, lines, band=False, fontsize=6):
    if band:
        lo, hi = null_band(metric)
        ax.fill_between(levels, lo, hi, color=BAND, lw=0, zorder=0)
    for gname in groups:
        mean, sem = curve(metric, gname, levels)
        series(ax, levels, gname, mean, sem)
    axis(ax)
    yaxis(ax, *ylim)
    stat_block(ax, lines, fontsize)


def panel_tuning(fig, fig_h, x, y, groups, w=PW):
    """Single-cell tuning curves, one panel: absolute (top) and normalized to 0 deg (bottom), shared x-axis.
    Descriptive (no statistics)."""
    gap = 0.16
    h = (PH - gap) / 2
    top = add_axes(fig, fig_h, x, y + h + gap, w=w, h=h)
    bottom = add_axes(fig, fig_h, x, y, w=w, h=h)
    for ax, col in ((top, "value"), (bottom, "norm_0deg")):
        for gname in groups:
            mean, sem = cell_curve(gname, col)
            series(ax, OFFSETS, gname, mean, sem, marker="none", lw=0.6)     # thin line, standard error bars
            # small markers drawn separately: in errorbar() the marker outline width also sets the cap thickness
            ax.plot(OFFSETS, mean, "o", ms=2.0, mfc=COL[gname], mec="black", mew=0.3, zorder=4)
        ax.set_xlim(-72, 102)
        ax.set_xticks(OFFSETS)
    top.set_xticklabels([])
    bottom.set_xticklabels([f"{o:d}".replace("-", "\u2212") for o in OFFSETS])
    yaxis(top, 0.08, 0.21, None, "{:.2f}", ticks=[0.1, 0.2])
    yaxis(bottom, 0.5, 1.08, None, "{:.1f}", ticks=[0.6, 0.8, 1.0])
    # two complete labels, so they cannot be read as one ("Normalized response")
    top.set_ylabel("Response\n(z-scored)", fontsize=6.5)
    bottom.set_ylabel("Normalized\nto 0\u00b0", fontsize=6.5)
    bottom.set_xlabel("Orientation relative\nto preferred (\u00b0)")
    for ax in (top, bottom):
        ax.yaxis.set_label_coords(-0.42, 0.5)
    return top, bottom


def draw_image(fig, fig_h, img, x, y, w, h=None, scale_bar=False):
    h = h if h is not None else w * img.height / img.width
    ax = fig.add_axes([x / fig.get_figwidth(), y / fig_h, w / fig.get_figwidth(), h / fig_h])
    ax.imshow(img, interpolation="none")
    ax.axis("off")
    if scale_bar:
        W, H = img.width, img.height
        L = SCALE_BAR_FRACTION * W
        x1, yb = W * 0.94, H * 0.93
        ax.plot([x1 - L, x1], [yb, yb], color="#ffd400", lw=2.2, solid_capstyle="butt")
        ax.text(x1 - L / 2, yb - H * 0.03, "100 \u00b5m", color="#ffd400", ha="center", va="bottom", fontsize=6)
    return ax


def decoding_schematic(fig, fig_h, x, y, w, h):
    """Linear decoder: traces of neurons 1..N weighted and summed ."""
    ax = fig.add_axes([x / fig.get_figwidth(), y / fig_h, w / fig.get_figwidth(), h / fig_h])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    traces = np.asarray(pptx_image("image2.png"))
    strips = [traces[25:170], traces[225:370], traces[425:570]]       # neurons 1, 2 and N
    rows = [h - 0.15, h - 0.46, 0.38]
    tx0, tw = 0.15, 0.30 * w                                         # horizontal layout scales with the width
    for strip, yc in zip(strips, rows):
        th = tw * strip.shape[0] / strip.shape[1]
        ax.imshow(strip, extent=(tx0, tx0 + tw, yc - th / 2, yc + th / 2), zorder=2, interpolation="none")
    y_dots = np.linspace(rows[1] - 0.15, rows[2] + 0.15, 3)
    cx, r = 0.48 * w, 0.085
    for xd in (tx0 + tw / 2, cx):
        ax.scatter([xd] * 3, y_dots, s=1.2, color="black", zorder=3)
    ax.text(0.04, (rows[0] + rows[-1]) / 2, "Neuron index", rotation=90, ha="center", va="center", fontsize=6)
    ax.text(tx0 + tw / 2, 0.04, "Deconvolved\nfluorescence", ha="center", va="bottom", fontsize=5.3,
            multialignment="center")
    sx, sy, sr = 0.70 * w, (rows[0] + rows[-1]) / 2, 0.095
    for label, yc, wl in zip(["1", "2", "N"], rows, ["$w_1$", "$w_2$", "$w_N$"]):
        ax.add_patch(Circle((cx, yc), r, facecolor="white", edgecolor="black", lw=0.7, zorder=3))
        ax.text(cx, yc, label, ha="center", va="center", fontsize=6, zorder=4)
        start = np.array([cx + r, yc])
        v = np.array([sx, sy]) - start
        end = np.array([sx, sy]) - v / np.linalg.norm(v) * sr
        ax.add_patch(FancyArrowPatch(tuple(start), tuple(end), arrowstyle="-|>", mutation_scale=4.5, lw=0.6,
                                     color="black", shrinkA=0, shrinkB=0))
        nrm = np.array([-v[1], v[0]]) / np.linalg.norm(v)             # unit normal, upper side of the arrow
        lab = start + 0.42 * (end - start) + 0.06 * nrm
        ax.text(lab[0], lab[1], wl, fontsize=5.5, ha="center", va="center")
    ax.add_patch(Circle((sx, sy), sr, facecolor="white", edgecolor="black", lw=0.7))
    ax.text(sx, sy, "\u03a3", ha="center", va="center", fontsize=6.5)
    bx = 0.775 * w
    bw = w - bx - 0.01
    ax.add_patch(FancyArrowPatch((sx + sr, sy), (bx, sy), arrowstyle="-|>", mutation_scale=4.5, lw=0.6,
                                 color="black", shrinkA=0, shrinkB=0))
    ax.add_patch(FancyBboxPatch((bx, sy - 0.2), bw, 0.4, boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor="white", edgecolor="black", lw=0.7))
    ax.text(bx + bw / 2, sy, "Spatial\nfrequency /\norientation", ha="center", va="center",
            fontsize=5.5 if bw > 0.5 else 5, multialignment="center")


DEC_SPECS = [("sf_dec_vs_n", NN, n_axis, (-0.1, 0.5, 0.1), "SF decoding\n(Pearson r)"),
             ("ori_dec_vs_n", NN, n_axis, (-0.1, 0.4, 0.1), "Orientation decoding\n(circular r)"),
             ("sf_dec_by_ori", ORI, ori_axis, (-0.1, 0.6, 0.1), "SF decoding\n(Pearson r)"),
             ("ori_dec_by_sf", SF5, lambda a: sf_axis(a, SF5), (-0.1, 0.4, 0.1), "Orientation decoding\n(circular r)")]


def decoding_panels(fig, fig_h, xs, y, groups, lines_for, letters, top_y, legend_band, fontsize=6):
    for k, (metric, levels, axis, ylim, ylabel) in enumerate(DEC_SPECS):
        ax = add_axes(fig, fig_h, xs[k], y)
        panel_session(ax, metric, groups, levels, axis, ylim, lines_for(metric), band=True, fontsize=fontsize)
        ax.set_ylabel(ylabel)
        if k == 0:                                   # legend above the first decoding panel
            group_legend(ax, groups, legend_band, loc="lower left", bbox_to_anchor=(-0.05, 1.04))
        if k >= 2:
            ax.text(0.5, 1.04, "N = 80 cells", transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5)
        letter(fig, fig_h, xs[k] - 0.58, top_y, letters[k])


# ------------------------------------------------------------------ per-SF significance placements
# (anchor, x factor, base, dy, ha, va) for 0.02 ... 0.64 cpd
R, U = ("left", "center"), ("left", "bottom")
F4C_MARKS = [
    ("WT vs Pre", "black", [("WT", 0.99, "top", 0.02, "left", "bottom"), ("WT", 1.07, "top", 0.004, *U),
                            ("WT", 1.12, "mean", 0.012, *R), ("WT", 1.12, "mean", 0.012, *R),
                            ("WT", 1.12, "mean", 0.012, *R), ("WT", 1.03, "top", 0.07, "center", "bottom")]),
    ("Pre vs chance", COL["Pre"], [("Pre", 1.12, "mean", 0.012, *R), ("Pre", 1.12, "mean", 0.012, *R),
                                   ("Pre", 1.07, "top", 0.004, *U), ("Pre", 1.00, "top", 0.012, "center", "bottom"),
                                   ("Pre", 1.00, "top", 0.012, "center", "bottom"),
                                   ("Pre", 1.12, "mean", 0.02, *R)]),
]
F5A_MARKS = [
    ("WT vs Post", "#8c8c8c", [("Post", 1.03, "top", 0.015, "left", "bottom"), ("WT", 1.07, "top", 0.004, *U),
                               ("WT", 1.07, "top", 0.004, *U), ("WT", 1.07, "top", 0.004, *U),
                               ("WT", 1.07, "top", 0.004, *U), ("WT", 1.03, "top", 0.07, "center", "bottom")]),
    ("Pre vs Post", COL["Pre"], [("Pre", 1.12, "mean", 0.03, *R), ("Pre", 1.16, "mean", 0.005, *R),
                                 ("Pre", 1.07, "top", 0.004, *U), ("Pre", 1.07, "top", 0.004, *U),
                                 ("Pre", 0.94, "top", 0.008, "right", "bottom"), ("Pre", 1.12, "mean", 0.02, *R)]),
]


# ------------------------------------------------------------------ Figure 4
def figure4():
    FIG_W = 7.0
    cols = [0.60, 2.24, 3.88, 5.52]                                     # rows 2-3, aligned columns
    y3 = 0.52                                                           # plot bottoms
    y2 = y3 + PH + 0.42 + 0.55                                          # legend above H; x-labels of row 2
    y1 = y2 + PH + 0.30 + 0.52
    fig_h = y1 + PH + 0.52
    fig = plt.figure(figsize=(FIG_W, fig_h), dpi=300)
    groups = ["WT", "Pre"]
    one = lambda metric: [vs("WT", "Pre", stars(metric, "WT vs Pre"))]
    top1, top2, top3 = y1 + PH + 0.46, y2 + PH + 0.26, y3 + PH + 0.40

    # row 1: illustrations, responsiveness
    img_a = pptx_image("image3.png")
    draw_image(fig, fig_h, img_a, 0.08, y1 - 0.30, 1.54 * img_a.width / img_a.height, 1.54)
    letter(fig, fig_h, 0.02, top1, "A")
    draw_image(fig, fig_h, pptx_image("image4.png"), 1.30, y1 - 0.24, 1.46, scale_bar=True)
    letter(fig, fig_h, 1.22, top1, "B")
    xc = 3.40
    ax = add_axes(fig, fig_h, xc, y1, w=1.48)
    panel_resp_sf(ax, ["WT", "Pre", "GtdKO"], F4C_MARKS)
    group_legend(ax, ["WT", "Pre", "GtdKO"], "Chance fraction", loc="lower left", bbox_to_anchor=(0.12, 0.80))
    letter(fig, fig_h, xc - 0.58, top1, "C")
    ax = add_axes(fig, fig_h, cols[3], y1)
    panel_session(ax, "resp_by_ori", groups, ORI, ori_axis, (0, 1.0, 0.2), one("resp_by_ori"))
    ax.set_ylabel("Responsive fraction")
    letter(fig, fig_h, cols[3] - 0.58, top1, "D")

    # row 2: tuning curves, cross-validated OSI, decoding schematic
    panel_tuning(fig, fig_h, cols[0], y2, groups)
    letter(fig, fig_h, cols[0] - 0.58, top2, "E")
    ax = add_axes(fig, fig_h, cols[1], y2)
    panel_session(ax, "osi_cv", groups, SF5, lambda a: sf_axis(a, SF5), (0, 0.5, 0.1), one("osi_cv"))
    ax.set_ylabel("Cross-validated OSI")
    letter(fig, fig_h, cols[1] - 0.58, top2, "F")
    x_sch = cols[2] - 0.30                                              # spans the last two columns
    decoding_schematic(fig, fig_h, x_sch, y2 - 0.16, cols[3] + PW - x_sch, 1.25)
    letter(fig, fig_h, cols[2] - 0.58, top2, "G")

    # row 3: decoding
    decoding_panels(fig, fig_h, cols, y3, groups, one, "HIJK", top3, "95% CI shuffled")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"Fig4.{ext}"))
    print(f"Fig 4 saved ({FIG_W:.2f} x {fig_h:.2f} in)")


# ------------------------------------------------------------------ Figure 5
def figure5():
    FIG_W = 7.0
    cols = [0.60, 2.66, 4.22, 5.78]                                     # B, C, D above F, G, H
    y2 = 0.52
    y1 = y2 + PH + 0.50 + 0.55                                          # legend above E; x-labels of row 1
    fig_h = y1 + PH + 0.60
    fig = plt.figure(figsize=(FIG_W, fig_h), dpi=300)
    groups = ["WT", "Pre", "Post"]
    top1, top2 = y1 + PH + 0.54, y2 + PH + 0.48

    def two(metric):
        return [vs("WT", "Post", stars(metric, "WT vs Post")), pre_post(stars(metric, "Pre vs Post"))]

    # row 1: responsiveness (A wide), tuning curves, cross-validated OSI
    ax = add_axes(fig, fig_h, cols[0], y1, w=1.46)
    panel_resp_sf(ax, groups, F5A_MARKS)
    group_legend(ax, groups, "Chance fraction", loc="lower left", bbox_to_anchor=(0.20, 0.84))
    letter(fig, fig_h, cols[0] - 0.58, top1, "A")
    ax = add_axes(fig, fig_h, cols[1], y1)
    panel_session(ax, "resp_by_ori", groups, ORI, ori_axis, (0, 1.0, 0.2), two("resp_by_ori"), fontsize=5.5)
    ax.set_ylabel("Responsive fraction")
    letter(fig, fig_h, cols[1] - 0.58, top1, "B")
    panel_tuning(fig, fig_h, cols[2], y1, groups)
    letter(fig, fig_h, cols[2] - 0.58, top1, "C")
    ax = add_axes(fig, fig_h, cols[3], y1)
    panel_session(ax, "osi_cv", groups, SF5, lambda a: sf_axis(a, SF5), (0, 0.5, 0.1), two("osi_cv"), fontsize=5.5)
    ax.set_ylabel("Cross-validated OSI")
    letter(fig, fig_h, cols[3] - 0.58, top1, "D")

    # row 2: decoding
    decoding_panels(fig, fig_h, cols, y2, groups, two, "EFGH", top2, "95% CI shuffled", fontsize=5.5)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"Fig5.{ext}"))
    print(f"Fig 5 saved ({FIG_W:.2f} x {fig_h:.2f} in)")


if __name__ == "__main__":
    figure4()
    figure5()
