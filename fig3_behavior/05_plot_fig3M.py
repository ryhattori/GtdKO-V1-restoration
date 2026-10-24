"""Figure 3M: learning curves of the VR visual discrimination task.

Correct choice (1/0) of the kept trials of each mouse (trials_clean.csv, 01), 30-trial moving average, x = last trial
of the window; mean +/- SEM over mice up to trial 1000. Trials 1-400 shaded; 70% criterion as a red dashed line.
Output: Fig3M.pdf/.svg/.png.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MA_WINDOW, X_MAX = 30, 1000
COLOR = {"WT": "#000000", "Rescued": "#2f5597"}
SHADE = {"WT": "#7f7f7f", "Rescued": "#8babd3"}
LABEL = {"WT": "WT", "Rescued": "GtdKO+AAV-Gαt2"}
AX_LW, TICK, FS = 1.456, 2.94, 10
W, H = 262.0, 190.0                           # page (pt), as Fig3M_prism_style (now panel N)
X0, Y0, AW, AH = 42.0, 32.5, 205.0, 118.0     # axis box (pt); same baseline and height as panel N

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": FS,
                     "pdf.fonttype": 42, "svg.fonttype": "none",
                     "axes.linewidth": AX_LW, "xtick.major.width": AX_LW, "ytick.major.width": AX_LW,
                     "xtick.major.size": TICK, "ytick.major.size": TICK, "xtick.major.pad": 2.5,
                     "ytick.major.pad": 3.0, "xtick.labelsize": FS, "ytick.labelsize": FS, "axes.labelsize": FS,
                     "axes.spines.top": False, "axes.spines.right": False})

t = pd.read_csv(os.path.join(HERE, "trials_clean.csv"), low_memory=False)
t = t[t["keep"] & t["group"].isin(["WT", "Rescued"])].sort_values(["mouse", "clean_idx"])

fig = plt.figure(figsize=(W / 72, H / 72))
ax = fig.add_axes([X0 / W, Y0 / H, AW / W, AH / H])
ax.axvspan(0, 400, color="#ebebeb", lw=0, zorder=0)
ns = {}
for grp in ["WT", "Rescued"]:
    curves = []
    for _, d in t[t.group == grp].groupby("mouse"):
        c = d["Correctness"].to_numpy(float)
        curves.append(np.convolve(c, np.ones(MA_WINDOW) / MA_WINDOW, mode="valid")[:X_MAX])
    mat = np.full((len(curves), max(map(len, curves))), np.nan)
    for i, cv in enumerate(curves):
        mat[i, :len(cv)] = cv
    n = np.sum(~np.isnan(mat), 0)
    mean = np.nanmean(mat, 0)
    sem = np.where(n > 1, np.nanstd(mat, 0, ddof=1) / np.sqrt(np.maximum(n, 1)), np.nan)
    x = np.arange(1, mat.shape[1] + 1) + MA_WINDOW - 1
    keep = x <= X_MAX
    ax.fill_between(x[keep], (mean - sem)[keep], (mean + sem)[keep], color=SHADE[grp], alpha=0.35, lw=0, zorder=1)
    ax.plot(x[keep], mean[keep], color=COLOR[grp], lw=1.0, zorder=3)
    ns[grp] = len(curves)
ax.plot([0, X_MAX], [0.7, 0.7], color="#ff0000", lw=1.0, dashes=(4, 3), zorder=2)
ax.set_xlim(0, X_MAX); ax.set_ylim(0, 1.0)
ax.set_xticks(np.arange(0, 1001, 200)); ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_yticklabels([f"{v:.1f}" for v in np.arange(0, 1.01, 0.2)])
for side in ("left", "bottom"):
    ax.spines[side].set_capstyle("projecting")
ax.set_xlabel("Trial number", labelpad=2)
ax.set_ylabel("Correct choice rate", labelpad=4)
for xc, s in ((200, "Trials 1–400"), (700, "Trials 401–1000")):
    ax.text(xc, 1.03, s, ha="center", va="bottom", fontsize=FS, clip_on=False)
for k, grp in enumerate(["WT", "Rescued"]):
    yc = 0.17 - 0.1 * k
    ax.plot([420, 480], [yc, yc], color=COLOR[grp], lw=1.0, solid_capstyle="butt")
    ax.text(495, yc, f"{LABEL[grp]} (n = {ns[grp]})", color=COLOR[grp], ha="left", va="center", fontsize=FS)

for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
    fig.savefig(os.path.join(HERE, f"Fig3M.{ext}"), **kw)
print("saved", ns)
