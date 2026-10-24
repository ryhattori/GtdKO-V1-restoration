"""Supplementary Figure 3: training-associated changes in V1 of WT mice.

A-I: WT mice before vs after VR training: responsive fraction by SF (A) and orientation (B), single-neuron tuning
  curves (C; absolute, top; normalized to 0 deg, bottom), cross-validated OSI (D), population decoding (E-H) and
  adjacent-orientation distance in the dsPCA orientation plane (I).
J-R: change after training (post - pre), WT vs GtdKO+AAV-Gat2 (mean +/- SEM over mice), in the same order.
Statistics: ART mixed models, mouse as a random intercept (STATS_MODEL): A-I training effect (per-SF ART-C contrasts
  with BH in A), J-R group x training interaction. Tuning curves (C, L) are descriptive.
Shuffled-label null (E-H): the GtdKO+AAV-Gat2 pre-training null used for the decoding panels of Figures 4-5.
Inputs: session_metrics_long.csv (01), stats_training_effect.csv and stats_resp_by_sf_per_sf.csv (02),
osi_cross_validated.csv (03), stats_osi_cross_validated.csv (04), tuning_curves_cells.csv (05).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "pipeline")]
import config  # noqa: E402
NULL_CACHE = os.path.join(config.RESULTS_ROOT, "rescued_pre_shuffled_continuous_null_iter100_v3.npz")
C_PRE, C_POST = "#b0b0b0", "#303030"          # WT pre / post
C_WT, C_RESC = "#8c8c8c", "#8b1a89"           # change: WT / GtdKO+AAV-Gat2
BAND = "#d9d9d9"

# "(1|mouse)": model used in the figures; "(1+training|mouse)": random-slope sensitivity analysis
# (the cross-validated OSI was tested with the "(1|mouse)" model only)
STATS_MODEL = "(1|mouse)"
P_TRAIN = "WT_training_p_(1|mouse)" if STATS_MODEL == "(1|mouse)" else "WT_training_p"
P_INTER = "group_x_training_p_(1|mouse)" if STATS_MODEL == "(1|mouse)" else "group_x_training_p"

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6.5,
                     "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})

# ------------------------------------------------------------------ data
d = pd.read_csv(os.path.join(HERE, "session_metrics_long.csv"))
d = d[d.group.isin(["WT", "Rescued"])]
osi = pd.read_csv(os.path.join(HERE, "osi_cross_validated.csv"))
osi = osi[osi.selection == "responsive"]
d = pd.concat([d, pd.DataFrame(dict(group=osi.group, mouse=osi.mouse, training=osi.training, metric="osi_cv",
                                    level=osi.sf, value=osi.osi_cv))], ignore_index=True)
cells = pd.read_csv(os.path.join(HERE, "tuning_curves_cells.csv"))

st = pd.read_csv(os.path.join(HERE, "stats_training_effect.csv"))
st = st[st.analysis == "primary"].set_index("metric")
st_osi = pd.read_csv(os.path.join(HERE, "stats_osi_cross_validated.csv"))
st_osi = st_osi[(st_osi.selection == "responsive") & (st_osi.measure == "osi_cv")].iloc[0]
per_sf = pd.read_csv(os.path.join(HERE, "stats_resp_by_sf_per_sf.csv"))
per_sf = per_sf[(per_sf.group == "WT") & (per_sf.model == STATS_MODEL)]

SF6 = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
SF5 = SF6[:5]
ORI = [0, 30, 60, 90, 120, 150]
NN = list(range(10, 81, 10))
OFFSETS = [-60, -30, 0, 30, 60, 90]
EB = dict(marker="o", ms=3.2, mec="black", mew=0.4, lw=1.0, capsize=1.6, elinewidth=0.6)


def p_train(metric):
    return st_osi.WT_training_p if metric == "osi_cv" else st.loc[metric, P_TRAIN]


def p_inter(metric):
    return st_osi.group_x_training_p if metric == "osi_cv" else st.loc[metric, P_INTER]


def stars(p):
    for thr, s in [(1e-4, "****"), (1e-3, "***"), (1e-2, "**"), (0.05, "*")]:
        if p < thr:
            return s
    return "n.s."


def mouse_matrix(metric, group, training, levels):
    x = d[(d.metric == metric) & (d.group == group) & (d.training == training)]
    x = x[x.level.apply(lambda v: any(abs(v - l) < 1e-9 for l in levels))]
    return x.pivot_table(index="mouse", columns="level", values="value").reindex(columns=levels)


def mean_sem(m):
    return m.mean(0).to_numpy(), (m.std(0, ddof=1) / np.sqrt(m.notna().sum(0))).to_numpy()


def null_band(metric):
    """95% CI of the shuffled-label null (mean +/- 1.96 SEM over iterations), as in the decoding panels."""
    z = np.load(NULL_CACHE, allow_pickle=True)
    sf_null = np.asarray(z["sf_by_orientation"], float)       # (iterations, N, orientation)
    ori_null = np.asarray(z["orientation_by_sf"], float)      # (iterations, N, SF incl. 0)
    samples = {"sf_dec_vs_n": np.nanmean(sf_null, axis=2),
               "ori_dec_vs_n": np.nanmean(ori_null[:, :, 1:6], axis=2),
               "sf_dec_by_ori": sf_null[:, -1, :],
               "ori_dec_by_sf": ori_null[:, -1, 1:6]}[metric]
    mean = np.nanmean(samples, 0)
    half = 1.96 * np.nanstd(samples, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(samples), 0))
    return mean - half, mean + half


# ------------------------------------------------------------------ panel specs
# metric, levels, x-axis kind, x label, WT y label, WT y range/ticks, change y label, change y range/ticks
LOGX = {6: (0.02 / 1.12, 0.64 * 1.12), 5: (0.02 / 1.12, 0.32 * 1.12)}
SPECS = {
    "resp_by_sf": (SF6, "sf", "Spatial frequency (cpd)", "Responsive fraction", (0, 1.0), np.arange(0, 1.01, 0.2),
                   "\u0394 Responsive fraction\n(post \u2212 pre)", (-0.2, 0.5), [-0.2, 0, 0.2, 0.4]),
    "resp_by_ori": (ORI, "ori", "Orientation (\u00b0)", "Responsive fraction", (0, 1.0), np.arange(0, 1.01, 0.2),
                    "\u0394 Responsive fraction\n(post \u2212 pre)", (-0.2, 0.5), [-0.2, 0, 0.2, 0.4]),
    "osi_cv": (SF5, "sf", "Spatial frequency (cpd)", "Cross-validated OSI", (0, 0.5), np.arange(0, 0.51, 0.1),
               "\u0394 Cross-validated OSI\n(post \u2212 pre)", (-0.1, 0.3), [-0.1, 0, 0.1, 0.2, 0.3]),
    "sf_dec_vs_n": (NN, "n", "Number of neurons used\nfor decoding", "SF decoding\n(Pearson r)", (-0.1, 0.5),
                    np.arange(-0.1, 0.51, 0.1), "\u0394 SF decoding\n(post \u2212 pre)", (-0.3, 0.45),
                    [-0.2, 0, 0.2, 0.4]),
    "ori_dec_vs_n": (NN, "n", "Number of neurons used\nfor decoding", "Orientation decoding\n(circular r)",
                     (-0.1, 0.4), np.arange(-0.1, 0.41, 0.1), "\u0394 Orientation decoding\n(post \u2212 pre)",
                     (-0.3, 0.45), [-0.2, 0, 0.2, 0.4]),
    "sf_dec_by_ori": (ORI, "ori", "Orientation (\u00b0)", "SF decoding\n(Pearson r)", (-0.1, 0.6),
                      np.arange(-0.1, 0.61, 0.1), "\u0394 SF decoding\n(post \u2212 pre)", (-0.3, 0.45),
                      [-0.2, 0, 0.2, 0.4]),
    "ori_dec_by_sf": (SF5, "sf", "Spatial frequency (cpd)", "Orientation decoding\n(circular r)", (-0.1, 0.5),
                      np.arange(-0.1, 0.51, 0.1), "\u0394 Orientation decoding\n(post \u2212 pre)", (-0.3, 0.45),
                      [-0.2, 0, 0.2, 0.4]),
    "adj_dist_by_sf": (SF6, "sf", "Spatial frequency (cpd)", "Mean distance between\nadjacent orientations",
                       (0.1, 0.7), np.arange(0.1, 0.71, 0.1), "\u0394 Adjacent-orientation\ndistance (post \u2212 pre)",
                       (-0.2, 0.65), [-0.2, 0, 0.2, 0.4, 0.6]),
}


def style_x(ax, levels, kind, xlabel):
    if kind == "sf":
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xlim(*LOGX[len(levels)])
        ax.set_xticks(levels)
        ax.set_xticklabels([f"{l:g}" for l in levels], fontsize=5.5 if len(levels) > 5 else 6)
    elif kind == "ori":
        ax.set_xlim(-12, 162)
        ax.set_xticks(levels)
    else:
        ax.set_xlim(4, 86)
        ax.set_xticks(levels)
    ax.set_xlabel(xlabel)


def style_y(ax, ylim, yticks, fmt="{:.1f}"):
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels([fmt.format(t + 0.0).replace("-", "\u2212") for t in yticks])


def label_above(ax, text):
    ax.text(1.0, 1.04, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=6)


# ------------------------------------------------------------------ panels
def draw_wt(ax, metric):
    levels, kind, xl, yl, ylim, yticks, *_ = SPECS[metric]
    if metric == "resp_by_sf":
        ax.axhspan(0, 0.05, color=BAND, lw=0, zorder=0)
    if metric in ("sf_dec_vs_n", "ori_dec_vs_n", "sf_dec_by_ori", "ori_dec_by_sf"):
        lo, hi = null_band(metric)
        ax.fill_between(levels, lo, hi, color=BAND, lw=0, zorder=0)
    for training, col in (("pre", C_PRE), ("post", C_POST)):
        mean, sem = mean_sem(mouse_matrix(metric, "WT", training, levels))
        ax.errorbar(levels, mean, yerr=sem, color=col, mfc=col, **EB, zorder=3)
    if metric == "resp_by_sf":          # per-SF pre vs post (ART-C, BH), above the post-training points
        mean, sem = mean_sem(mouse_matrix(metric, "WT", "post", levels))
        for k, lvl in enumerate(levels):
            lab = stars(per_sf.loc[per_sf.contrast.str.endswith(f",{lvl:g}"), "q_BH"].item())
            ax.text(lvl, mean[k] + sem[k] + 0.03, lab, ha="center", va="bottom",
                    fontsize=5 if lab == "n.s." else 7)
    label_above(ax, f"WT pre/post: {stars(p_train(metric))}")
    style_x(ax, levels, kind, xl)
    style_y(ax, ylim, yticks, "{:.1f}")
    ax.set_ylabel(yl)
    if metric in ("sf_dec_by_ori", "ori_dec_by_sf"):
        ax.text(0.97, 0.97, "N = 80 cells", transform=ax.transAxes, ha="right", va="top", fontsize=6)


def draw_change(ax, metric):
    levels, kind, xl, _, _, _, dyl, dylim, dyticks = SPECS[metric]
    ax.axhline(0, color="black", lw=0.6, ls="--", zorder=1)
    for grp, col in (("WT", C_WT), ("Rescued", C_RESC)):
        mean, sem = mean_sem(mouse_matrix(metric, grp, "post", levels) - mouse_matrix(metric, grp, "pre", levels))
        ax.errorbar(levels, mean, yerr=sem, color=col, mfc=col, **EB, zorder=3)
    label_above(ax, f"Group \u00d7 training: {stars(p_inter(metric))}")
    style_x(ax, levels, kind, xl)
    style_y(ax, dylim, dyticks)
    ax.set_ylabel(dyl)


def tuning_axes(fig, fig_h, x, y, w, h):
    gap, h = 0.14, h + TUNE_EXTRA
    hh = (h - gap) / 2
    return add_axes(fig, fig_h, x, y + hh + gap, w, hh), add_axes(fig, fig_h, x, y, w, hh)


def tuning_series(ax, mean, sem, col):
    ax.errorbar(OFFSETS, mean, yerr=sem, color=col, marker="none", lw=0.6, capsize=1.6, elinewidth=0.6,
                mew=0.4, zorder=3)
    ax.plot(OFFSETS, mean, "o", ms=2.0, mfc=col, mec="black", mew=0.3, zorder=4)


def tuning_x(ax, bottom):
    ax.set_xlim(-72, 102)
    ax.set_xticks(OFFSETS)
    if bottom:
        ax.set_xticklabels([f"{o:d}".replace("-", "\u2212") for o in OFFSETS], fontsize=5.5)
        ax.set_xlabel("Orientation relative\nto preferred (\u00b0)")
    else:
        ax.set_xticklabels([])


def draw_tuning_wt(fig, fig_h, x, y, w, h):
    """WT single-cell tuning curves before vs after training (cells as n), as in Fig. 5C."""
    top, bottom = tuning_axes(fig, fig_h, x, y, w, h)
    for ax, col in ((top, "value"), (bottom, "norm_0deg")):
        for training, color in (("pre", C_PRE), ("post", C_POST)):
            wcur = cells[(cells.group == "WT") & (cells.training == training)].pivot(
                index="cell", columns="offset", values=col)[OFFSETS]
            mean, sem = wcur.mean(0).to_numpy(), (wcur.std(0, ddof=1) / np.sqrt(len(wcur))).to_numpy()
            tuning_series(ax, mean, sem, color)
        tuning_x(ax, ax is bottom)
    style_y(top, (0.08, 0.21), [0.1, 0.2], "{:.2f}")
    style_y(bottom, (0.5, 1.08), [0.6, 0.8, 1.0])
    top.set_ylabel("Response\n(z-scored)", fontsize=6.5)
    bottom.set_ylabel("Normalized\nto 0\u00b0", fontsize=6.5)
    for ax in (top, bottom):
        ax.yaxis.set_label_coords(-0.45, 0.5)
    return top


def draw_tuning_change(fig, fig_h, x, y, w, h):
    """Change of the session-average tuning curve after training (post - pre per mouse), WT vs rescued."""
    top, bottom = tuning_axes(fig, fig_h, x, y, w, h)
    sess = cells.groupby(["group", "mouse", "training", "offset"])[["value", "norm_0deg"]].mean().reset_index()
    for ax, col in ((top, "value"), (bottom, "norm_0deg")):
        ax.axhline(0, color="black", lw=0.6, ls="--", zorder=1)
        for grp, color in (("WT", C_WT), ("Rescued", C_RESC)):
            s = sess[sess.group == grp].pivot_table(index="mouse", columns=["training", "offset"], values=col)
            delta = s["post"][OFFSETS] - s["pre"][OFFSETS]
            tuning_series(ax, *mean_sem(delta), color)
        tuning_x(ax, ax is bottom)
    style_y(top, (-0.03, 0.10), [0, 0.05, 0.1], "{:.2f}")
    style_y(bottom, (-0.4, 0.1), [-0.4, -0.2, 0])
    top.set_ylabel("\u0394 Response\n(z-scored)", fontsize=6.5)
    bottom.set_ylabel("\u0394 Normalized\nto 0\u00b0", fontsize=6.5)
    for ax in (top, bottom):
        ax.yaxis.set_label_coords(-0.45, 0.5)
    return top


# ------------------------------------------------------------------ layout (inches)
PH = 0.85                                      # plot height (all panels)
TUNE_EXTRA = 0.12                              # tuning panels also use the height of the statistics label
GAP = 0.60                                     # space for the next panel's y-axis label
ROW_A = [("resp_by_sf", 1.30), ("resp_by_ori", 0.85), ("tuning", 0.85), ("osi_cv", 0.85)]
ROW_B = [("sf_dec_vs_n", 0.85), ("ori_dec_vs_n", 0.85), ("sf_dec_by_ori", 0.85), ("ori_dec_by_sf", 0.85),
         ("adj_dist_by_sf", round(0.85 * 1.43, 2))]              # Fig. 6D aspect
LEFT, BOTTOM = 0.62, 0.52
ROW_PITCH, HALF_GAP = 1.78, 2.05                                # plot bottom to plot bottom
fig_w = LEFT + sum(w for _, w in ROW_B) + GAP * (len(ROW_B) - 1) + 0.1
rows_y = [0.0] * 4
rows_y[3] = BOTTOM
rows_y[2] = rows_y[3] + ROW_PITCH
rows_y[1] = rows_y[2] + HALF_GAP
rows_y[0] = rows_y[1] + ROW_PITCH
fig_h = rows_y[0] + PH + 0.40
fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)


def add_axes(fig, fig_h, x, y, w, h):
    return fig.add_axes([x / fig.get_figwidth(), y / fig_h, w / fig.get_figwidth(), h / fig_h])


letters = iter("ABCDEFGHIJKLMNOPQR")
for part, (row_panels, r_y) in enumerate([(ROW_A, rows_y[0]), (ROW_B, rows_y[1]), (ROW_A, rows_y[2]),
                                          (ROW_B, rows_y[3])]):
    change = part >= 2
    x = LEFT
    for metric, w in row_panels:
        if metric == "tuning":
            ax = (draw_tuning_change if change else draw_tuning_wt)(fig, fig_h, x, r_y, w, PH)
        else:
            ax = add_axes(fig, fig_h, x, r_y, w, PH)
            (draw_change if change else draw_wt)(ax, metric)
        fig.text((x - 0.56) / fig_w, (r_y + PH + 0.36) / fig_h, next(letters), fontweight="bold", fontsize=9,
                 ha="left", va="top")
        x += w + GAP

# legends in the free space at the right of rows 1 and 3
x_leg = (LEFT + 4 * (ROW_B[0][1] + GAP) - 0.12) / fig_w                  # above/below panel I/R
fig.legend(handles=[Line2D([], [], color=C_PRE, marker="o", mec="black", mew=0.4, ms=3.2, lw=1, label="WT pre-training"),
                    Line2D([], [], color=C_POST, marker="o", mec="black", mew=0.4, ms=3.2, lw=1, label="WT post-training"),
                    Patch(color=BAND, label="Chance fraction (A)"),
                    Patch(color=BAND, label="95% CI shuffled (E\u2013H)")],
           loc="center left", bbox_to_anchor=(x_leg, (rows_y[0] + PH / 2) / fig_h), frameon=False, fontsize=6.5,
           handlelength=1.6)
fig.legend(handles=[Line2D([], [], color=C_WT, marker="o", mec="black", mew=0.4, ms=3.2, lw=1, label="WT"),
                    Line2D([], [], color=C_RESC, marker="o", mec="black", mew=0.4, ms=3.2, lw=1,
                           label="GtdKO+AAV-G\u03b1t2")],
           loc="center left", bbox_to_anchor=(x_leg, (rows_y[2] + PH / 2) / fig_h), frameon=False, fontsize=6.5,
           handlelength=1.6)

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"SuppFig3.{ext}"))
print(f"saved ({fig_w:.2f} x {fig_h:.2f} in)")
