"""Source data workbooks (one per figure, one sheet per panel) for Figs 3M-N, 4, 5, 6 and Supplementary Figs 3-4.

Each panel sheet has the per-mouse (or per-neuron) values that were plotted, followed by the plotted summary
(mean, SEM, n). Reference bands (chance level, shuffled-label null) are given where the panel shows them.
Every number is read from the analysis outputs used to draw the figures:
  Fig 3M-N     fig3_behavior/            trials_clean.csv, mouse_level_values.csv, trials_to_criterion.csv, stats
  Figs 4-5     fig4_fig5/          fig45_session_metrics.csv, fig45_cells.csv, fig45_stats.csv
  Fig 6B-C     source_data/             fig6B_pseudopop_projections.csv, fig6C_orientation_plane_points.csv
                                        (01_fig6_projections.py)
  Fig 6D, S3   session_metrics/ session_metrics_long.csv, osi_cross_validated.csv, tuning_curves_cells.csv
  S4           suppfig4_dspca_alternatives/   geometry_metrics_long.csv
  statistics   supp_table1/stats_table.csv and the per-panel stats files above
Shuffled-label null for the decoding panels: rescued_pre_shuffled_continuous_null_iter100_v3.npz.
"""
import os
import sys

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import config  # noqa: E402
OUT = os.path.join(HERE, "output")
NULL = os.path.join(config.RESULTS_ROOT, "rescued_pre_shuffled_continuous_null_iter100_v3.npz")
os.makedirs(OUT, exist_ok=True)

RESC = "GtdKO+AAV-Gαt2"
SF6 = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
SF5 = SF6[:5]
ORI = [0, 30, 60, 90, 120, 150]
NN = list(range(10, 81, 10))
OFFSETS = [-60, -30, 0, 30, 60, 90]

# measure: (levels, x-axis label, value description)
SPEC = {
    "resp_by_sf": (SF6, "Spatial frequency (cpd)", "Fraction of visually responsive neurons"),
    "resp_by_ori": (ORI, "Orientation (°)", "Fraction of visually responsive neurons"),
    "osi_cv": (SF5, "Spatial frequency (cpd)", "Cross-validated OSI"),
    "sf_dec_vs_n": (NN, "Number of neurons", "SF decoding accuracy (cross-validated Pearson correlation)"),
    "ori_dec_vs_n": (NN, "Number of neurons", "Orientation decoding accuracy (cross-validated circular correlation)"),
    "sf_dec_by_ori": (ORI, "Orientation (°)", "SF decoding accuracy, 80 neurons (cross-validated Pearson correlation)"),
    "ori_dec_by_sf": (SF5, "Spatial frequency (cpd)",
                      "Orientation decoding accuracy, 80 neurons (cross-validated circular correlation)"),
    "adj_dist_by_sf": (SF6, "Spatial frequency (cpd)",
                       "Mean distance between adjacent orientations (30° apart) in the dsPCA orientation plane"),
    "pca3": (SF6, "Spatial frequency (cpd)", "Mean distance between adjacent orientations, top 3 PCs"),
    "crossnobis": (SF6, "Spatial frequency (cpd)", "Cross-validated Mahalanobis distance (per neuron)"),
    "lda_acc": (SF6, "Spatial frequency (cpd)", "Cross-validated linear decoding accuracy (balanced accuracy)"),
}

# ------------------------------------------------------------------ inputs
f45 = pd.read_csv(os.path.join(ROOT, "fig4_fig5", "fig45_session_metrics.csv"))
cells45 = pd.read_csv(os.path.join(ROOT, "fig4_fig5", "fig45_cells.csv"))
st45 = pd.read_csv(os.path.join(ROOT, "fig4_fig5", "fig45_stats.csv"))
R16 = os.path.join(ROOT, "session_metrics")
r16 = pd.read_csv(os.path.join(R16, "session_metrics_long.csv"))
osi = pd.read_csv(os.path.join(R16, "osi_cross_validated.csv"))
osi = osi[osi.selection == "responsive"]
r16 = pd.concat([r16, pd.DataFrame(dict(group=osi.group, mouse=osi.mouse, training=osi.training, metric="osi_cv",
                                        level=osi.sf, value=osi.osi_cv))], ignore_index=True)
cells16 = pd.read_csv(os.path.join(R16, "tuning_curves_cells.csv"))
geo = pd.read_csv(os.path.join(ROOT, "suppfig4_dspca_alternatives", "geometry_metrics_long.csv"))
geo = geo.rename(columns={"sf": "level"})
stab = pd.read_csv(os.path.join(ROOT, "supp_table1", "stats_table.csv"))
F3 = os.path.join(ROOT, "fig3_behavior")


def null_band(metric):
    """95% CI of the shuffled-label null (mean +/- 1.96 SEM over 100 iterations), as drawn in the decoding panels."""
    z = np.load(NULL, allow_pickle=True)
    sf_null = np.asarray(z["sf_by_orientation"], float)
    ori_null = np.asarray(z["orientation_by_sf"], float)
    s = {"sf_dec_vs_n": np.nanmean(sf_null, axis=2), "ori_dec_vs_n": np.nanmean(ori_null[:, :, 1:6], axis=2),
         "sf_dec_by_ori": sf_null[:, -1, :], "ori_dec_by_sf": ori_null[:, -1, 1:6]}[metric]
    mean = np.nanmean(s, 0)
    half = 1.96 * np.nanstd(s, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(s), 0))
    return mean - half, mean + half


def wide(x, levels):
    """Mouse x level table from long data (columns matched to levels with a float tolerance)."""
    w = x.pivot_table(index="mouse", columns="level", values="value")
    cols = {c: lv for c in w.columns for lv in levels if np.isclose(c, lv)}
    return w.rename(columns=cols).reindex(columns=levels)


class Book:
    """Collects sheets as lists of blocks; a block is (title_lines, DataFrame)."""

    def __init__(self, name, readme):
        self.name, self.sheets = name, []

    def add(self, sheet, blocks):
        """Keep only the plotted data: the first block of each panel, with a one-line title."""
        if sheet in ("Statistics", "Fig6C_plotted"):
            return
        lines, df = blocks[0]
        self.sheets.append((sheet, [(lines[:1], df)]))

    def save(self):
        path = os.path.join(OUT, self.name)
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            layout = {}
            for sheet, blocks in self.sheets:
                row, marks = 0, []
                for lines, df in blocks:
                    pd.DataFrame({"": lines}).to_excel(xw, sheet_name=sheet, startrow=row, index=False, header=False)
                    marks.append(("title", row, len(lines)))
                    row += len(lines)
                    if df is not None:
                        df.to_excel(xw, sheet_name=sheet, startrow=row, index=False)
                        marks.append(("header", row, len(df.columns)))
                        row += len(df) + 1
                    row += 1
                layout[sheet] = marks
        wb = load_workbook(path)
        for sheet, marks in layout.items():
            ws = wb[sheet]
            for kind, r, n in marks:
                if kind == "title":
                    ws.cell(r + 1, 1).font = Font(bold=True)
                    for k in range(1, n):
                        ws.cell(r + 1 + k, 1).font = Font(italic=True)
                else:
                    for c in range(1, n + 1):
                        cell = ws.cell(r + 1, c)
                        cell.font = Font(bold=True)
                        cell.fill = PatternFill("solid", fgColor="E7E6E6")
                        cell.alignment = Alignment(wrap_text=True, vertical="top")
            ws.column_dimensions["A"].width = 34 if sheet != "README" else 140
            for col in "BCDEFGHIJKLMNOP":
                ws.column_dimensions[col].width = 14
            if sheet == "README":
                for row in ws.iter_rows():
                    for cell in row:
                        cell.alignment = Alignment(wrap_text=True, vertical="top")
        wb.save(path)
        print("saved", path)


def per_mouse_blocks(title, metric, groups, band=None, note=None):
    """groups: list of (label, long DataFrame for that group). Per-mouse table, then mean/SEM/n."""
    levels, xlab, ylab = SPEC[metric]
    lv_names = [f"{v:g}" for v in levels]
    rows, summ = [], []
    for label, x in groups:
        w = wide(x, levels)
        for mouse, r in w.iterrows():
            rows.append([label, mouse] + r.tolist())
        n = w.notna().sum(0)
        summ.append([label, "mean"] + w.mean(0).tolist())
        summ.append([label, "SEM"] + (w.std(0, ddof=1) / np.sqrt(n)).tolist())
        summ.append([label, "n (mice)"] + n.astype(int).tolist())
    cols = ["Group", "Mouse"] + lv_names
    head = [title, f"Values: {ylab}. Columns: {xlab}."] + ([note] if note else [])
    blocks = [(head, pd.DataFrame(rows, columns=cols)),
              (["Plotted summary (mean ± SEM over mice)"], pd.DataFrame(summ, columns=["Group", "Statistic"] + lv_names))]
    if band is not None:
        blocks.append(band(lv_names))
    return blocks


def chance_band(lv_names):
    return (["Gray band: chance fraction of visually responsive neurons (5%)"],
            pd.DataFrame([["chance level"] + [0.05] * len(lv_names)], columns=["Reference"] + lv_names))


def shuffle_band(metric):
    def f(lv_names):
        lo, hi = null_band(metric)
        return (["Gray band: 95% CI of the shuffled-label null (100 shuffles of GtdKO+AAV-Gαt2 pre-training sessions)"],
                pd.DataFrame([["lower"] + list(lo), ["upper"] + list(hi)], columns=["Reference"] + lv_names))
    return f


def tuning_blocks(title, groups, n_note):
    """groups: list of (label, cells DataFrame). Per-neuron aligned tuning curves (raw and normalized), then summary."""
    rows, summ = [], []
    for label, c in groups:
        raw = c.pivot(index="cell", columns="offset", values="value")[OFFSETS]
        nrm = c.pivot(index="cell", columns="offset", values="norm_0deg")[OFFSETS]
        mouse = c.drop_duplicates("cell").set_index("cell")["mouse"]
        for cell in raw.index:
            rows.append([label, mouse[cell], cell] + raw.loc[cell].tolist() + nrm.loc[cell].tolist())
        for name, w in (("response", raw), ("normalized", nrm)):
            summ.append([label, name, "mean"] + w.mean(0).tolist())
            summ.append([label, name, "SEM"] + (w.std(0, ddof=1) / np.sqrt(len(w))).tolist())
            summ.append([label, name, "n (neurons)"] + [len(w)] * len(OFFSETS))
    off = [f"{o:+d}°" if o else "0°" for o in OFFSETS]
    cols = ["Group", "Mouse", "Neuron"] + [f"Response {o}" for o in off] + [f"Normalized {o}" for o in off]
    head = [title, "Values: z-scored response, by orientation relative to the preferred orientation (cross-validated); "
                   "Normalized: divided by the mean response at 0° of all neurons in the same session.", n_note]
    return [(head, pd.DataFrame(rows, columns=cols)),
            (["Plotted summary (mean ± SEM over neurons)"],
             pd.DataFrame(summ, columns=["Group", "Curve", "Statistic"] + off))]


def stats_block(figs, extra=None):
    x = stab[stab.figure.isin(figs)].copy()
    x["F (df1, df2)"] = [f"{f:.2f} ({a:.0f}, {b:.0f})" for f, a, b in zip(x.F, x.df1, x.df2)]
    x = x[["figure", "panel", "measure", "comparison", "term", "n_mice", "neurons", "mean_sem_1", "mean_sem_2",
           "F (df1, df2)", "p", "eta2p"]]
    x.columns = ["Figure", "Panel", "Measure", "Comparison", "Model term", "n mice (1 / 2)", "Neurons (1 / 2)",
                 "Group 1 mean ± SEM", "Group 2 mean ± SEM", "F (df1, df2)", "p", "Partial η²"]
    blocks = [(["Statistics: ART mixed models with mouse as a random intercept (as in Supplementary Table 1)",
                "Group 1 / Group 2 are the two groups named in the comparison, in that order."], x)]
    return blocks + (extra or [])


# ================================================================== Figure 3M-N
t = pd.read_csv(os.path.join(F3, "trials_clean.csv"), low_memory=False)
t = t[t["keep"] & t["group"].isin(["WT", "Rescued"])].sort_values(["mouse", "clean_idx"])
curves = {}
for (grp, mouse), x in t.groupby(["group", "mouse"]):
    c = np.convolve(x["Correctness"].to_numpy(float), np.ones(30) / 30, mode="valid")[:1000]
    trial = np.arange(1, len(c) + 1) + 29
    curves[(grp, mouse)] = pd.Series(c[trial <= 1000], index=trial[trial <= 1000])
lab3 = {"WT": "WT", "Rescued": RESC}
cm = pd.DataFrame({f"{lab3[g]} {m}": s for (g, m), s in sorted(curves.items(), key=lambda k: k[0][0] != "WT")})
cm.insert(0, "Trial number", cm.index)
mv = pd.read_csv(os.path.join(F3, "mouse_level_values.csv"))
crit = pd.read_csv(os.path.join(F3, "trials_to_criterion.csv"))
crit = crit[crit.group.isin(["WT", "Rescued"])]
prim = pd.read_csv(os.path.join(F3, "stats_primary_exact.csv"))
ctest = pd.read_csv(os.path.join(F3, "criterion_test.csv"))

b3 = Book("Fig3MN_source_data.xlsx",
          "Source data for Figure 3M and 3N (VR-based visual discrimination task).\n"
          "WT, 5 mice; GtdKO+AAV-Gαt2, 5 mice. Trials from all behavioral sessions of each mouse were concatenated in "
          "order after removing invalid trials (see Methods).\n"
          "3M: correct choice (1/0) averaged over a moving window of 30 trials; the trial number is the last trial of "
          "the window (first value at trial 30), up to trial 1000. Mean ± SEM over mice.\n"
          "3N: correct choice rate in trials 1–400 and 401–1000, and number of trials needed to reach ≥70% correct over "
          "100 consecutive trials. Exact two-sided permutation tests on per-mouse values (252 permutations); trial "
          "windows Holm-adjusted.\nPanels A–L of Figure 3 are not included here.")
b3.add("Fig3M", [(["Figure 3M. Learning curves: correct choice rate (30-trial moving average)",
                   "One column per mouse; mean, SEM and n over mice at the end."], cm)])
n_rows = []
for g in ("WT", "Rescued"):
    for _, r in mv[mv.group == g].iterrows():
        c = crit[(crit.group == g) & (crit.mouse == r.mouse)]
        n_rows.append([lab3[g], r.mouse, r.early, r.late, int(c.trials_to_criterion.item())])
nd = pd.DataFrame(n_rows, columns=["Group", "Mouse", "Correct choice rate, trials 1–400",
                                   "Correct choice rate, trials 401–1000", "Trials to 70% criterion"])
ns = []
for g in ("WT", RESC):
    x = nd[nd.Group == g].iloc[:, 2:]
    ns += [[g, "mean"] + x.mean().tolist(), [g, "SEM"] + (x.std(ddof=1) / np.sqrt(len(x))).tolist(),
           [g, "n (mice)"] + [len(x)] * 3]
st3 = pd.DataFrame({
    "Comparison": ["WT vs GtdKO+AAV-Gαt2, trials 1–400", "WT vs GtdKO+AAV-Gαt2, trials 401–1000",
                   "Group × window (improvement, GtdKO+AAV-Gαt2 vs WT)", "WT vs GtdKO+AAV-Gαt2, trials to criterion"],
    "Test": ["exact two-sided permutation test, Holm-adjusted over the two windows"] * 2
            + ["exact two-sided permutation test on the per-mouse improvement"] * 1 + ["exact two-sided permutation test"],
    "Difference (GtdKO+AAV-Gαt2 − WT)": [f"{prim.diff_pp[0]:.1f} percentage points",
                                         f"{prim.diff_pp[1]:.1f} percentage points",
                                         f"{prim.diff_pp[2]:.1f} percentage points", f"{ctest['diff'][0]:.1f} trials"],
    "p": [prim.p_holm[0], prim.p_holm[1], prim.exact_p[2], ctest.p_perm_two_sided[0]],
    "Label in figure": ["*", "n.s.", "(legend)", "**"]})
b3.add("Fig3N", [(["Figure 3N. Correct choice rate in trials 1–400 and 401–1000 (left) and trials to criterion (right)"], nd),
                 (["Plotted summary (mean ± SEM over mice)"], pd.DataFrame(ns, columns=["Group", "Statistic"] + list(nd.columns[2:]))),
                 (["Statistics"], st3)])
b3.save()

# ================================================================== Figures 4 and 5
def g45(metric, group):
    return f45[(f45.metric == metric) & (f45.group == group)]


L45 = {"WT": "WT", "Pre": f"{RESC} pre-training", "Post": f"{RESC} post-training", "GtdKO": "GtdKO"}
N45 = ("Mice: WT VD017–VD020, VD022; GtdKO+AAV-Gαt2 VD024, VD031–VD034 (same mice before and after training); "
       "GtdKO VD025–VD027. One imaging session per mouse and time point.")
lvl_stats = st45[st45.level.notna()][["figure", "metric", "comparison", "level", "test", "p", "stars"]]
lvl_stats.columns = ["Figure", "Measure", "Comparison", "Level (SF, cpd)", "Test", "p (BH-corrected)", "Label"]

b4 = Book("Fig4_source_data.xlsx",
          "Source data for Figure 4 (V1 two-photon imaging before VR training).\n"
          "WT, 5 mice, 741 neurons; GtdKO+AAV-Gαt2 before training, 5 mice, 513 neurons; GtdKO, 3 mice, 279 neurons "
          "(Fig. 4C only). One imaging session per mouse.\n"
          "C, D, F, H–K: one value per mouse and x-axis point; the plotted mean ± SEM is over mice.\n"
          "E: per-neuron tuning curves of visually responsive neurons (621 WT, 320 GtdKO+AAV-Gαt2 neurons); the plotted "
          "mean ± SEM is over neurons.\nA, B, G are schematics or an example image and have no source data.\n" + N45)
b4.add("Fig4C", per_mouse_blocks("Figure 4C. Fraction of visually responsive neurons by spatial frequency", "resp_by_sf",
                                 [(L45[g], g45("resp_by_sf", g)) for g in ("WT", "Pre", "GtdKO")], chance_band,
                                 "GtdKO mice are shown as individual lines in the figure."))
b4.add("Fig4D", per_mouse_blocks("Figure 4D. Fraction of visually responsive neurons by orientation", "resp_by_ori",
                                 [(L45[g], g45("resp_by_ori", g)) for g in ("WT", "Pre")]))
b4.add("Fig4E", tuning_blocks("Figure 4E. Orientation tuning curves of visually responsive neurons",
                              [(L45[g], cells45[cells45.group == g]) for g in ("WT", "Pre")],
                              "n = 621 WT and 320 GtdKO+AAV-Gαt2 neurons from 5 mice each."))
b4.add("Fig4F", per_mouse_blocks("Figure 4F. Cross-validated OSI", "osi_cv",
                                 [(L45[g], g45("osi_cv", g)) for g in ("WT", "Pre")]))
for p, m in (("H", "sf_dec_vs_n"), ("I", "ori_dec_vs_n"), ("J", "sf_dec_by_ori"), ("K", "ori_dec_by_sf")):
    b4.add(f"Fig4{p}", per_mouse_blocks(f"Figure 4{p}. {SPEC[m][2]}", m,
                                        [(L45[g], g45(m, g)) for g in ("WT", "Pre")], shuffle_band(m)))
b4.add("Statistics", stats_block(["Fig. 4"], [(["Per-SF tests in Figure 4C (asterisks above the points)"],
                                               lvl_stats[lvl_stats.Figure.astype(str) == "4"])]))
b4.save()

b5 = Book("Fig5_source_data.xlsx",
          "Source data for Figure 5 (V1 two-photon imaging after VR training).\n"
          "WT, 5 mice imaged before VR training (the sessions in Figure 4), 741 neurons; GtdKO+AAV-Gαt2 before and after "
          "training, 5 mice, 513 and 455 neurons. One imaging session per mouse and time point.\n"
          "A, B, D–H: one value per mouse and x-axis point; the plotted mean ± SEM is over mice.\n"
          "C: per-neuron tuning curves (621 WT, 320 pre- and 356 post-training GtdKO+AAV-Gαt2 neurons); mean ± SEM over "
          "neurons.\n" + N45)
G5 = ("WT", "Pre", "Post")
b5.add("Fig5A", per_mouse_blocks("Figure 5A. Fraction of visually responsive neurons by spatial frequency", "resp_by_sf",
                                 [(L45[g], g45("resp_by_sf", g)) for g in G5], chance_band))
b5.add("Fig5B", per_mouse_blocks("Figure 5B. Fraction of visually responsive neurons by orientation", "resp_by_ori",
                                 [(L45[g], g45("resp_by_ori", g)) for g in G5]))
b5.add("Fig5C", tuning_blocks("Figure 5C. Orientation tuning curves of visually responsive neurons",
                              [(L45[g], cells45[cells45.group == g]) for g in G5],
                              "n = 621 WT, 320 pre- and 356 post-training GtdKO+AAV-Gαt2 neurons (5 mice each)."))
b5.add("Fig5D", per_mouse_blocks("Figure 5D. Cross-validated OSI", "osi_cv", [(L45[g], g45("osi_cv", g)) for g in G5]))
for p, m in (("E", "sf_dec_vs_n"), ("F", "ori_dec_vs_n"), ("G", "sf_dec_by_ori"), ("H", "ori_dec_by_sf")):
    b5.add(f"Fig5{p}", per_mouse_blocks(f"Figure 5{p}. {SPEC[m][2]}", m, [(L45[g], g45(m, g)) for g in G5],
                                        shuffle_band(m)))
b5.add("Statistics", stats_block(["Fig. 5"], [(["Per-SF tests in Figure 5A (asterisks above the points)"],
                                               lvl_stats[lvl_stats.Figure.astype(str) == "5"])]))
b5.save()


# ================================================================== Figure 6
def g16(metric, group, training, data=r16):
    return data[(data.metric == metric) & (data.group == group) & (data.training == training)]


L16 = {("WT", "pre"): "WT", ("Rescued", "pre"): f"{RESC} pre-training", ("Rescued", "post"): f"{RESC} post-training"}
b6b = pd.read_csv(os.path.join(HERE, "fig6B_pseudopop_projections.csv"))
b6b = b6b[b6b.sf_cpd > 0]                    # the blank-level conditions were fitted but are not plotted
b6b = b6b.sort_values(["group", "sf_cpd", "orientation_deg"], key=lambda s: s.map(
    {"WT": 0, f"{RESC} pre-training": 1, f"{RESC} post-training": 2}) if s.name == "group" else s)
b6b.columns = ["Group", "Neurons in pseudopopulation", "Orientation (°)", "SF (cpd)",
               "Stimulus-coding: log2(SF) axis", "Stimulus-coding: sin(2θ) axis", "Stimulus-coding: cos(2θ) axis",
               "Stimulus-null: PC1", "Stimulus-null: PC2", "Stimulus-null: PC3",
               "PC1 variance (%)", "PC2 variance (%)", "PC3 variance (%)"]
c6 = pd.read_csv(os.path.join(HERE, "fig6C_orientation_plane_points.csv"))
c6rows = []
for (grp, sf, o), x in c6.groupby(["group", "sf_cpd", "orientation_deg"], sort=False):
    n = x.sin2theta.notna().sum()
    se = x[["sin2theta", "cos2theta"]].std(ddof=1) / np.sqrt(n)
    c6rows.append([grp, sf, o, x.sin2theta.mean(), x.cos2theta.mean(), se.sin2theta, se.cos2theta,
                   float(np.sqrt((se ** 2).sum())), n])
c6s = pd.DataFrame(c6rows, columns=["Group", "SF (cpd)", "Orientation (°)", "sin(2θ) mean", "cos(2θ) mean",
                                    "sin(2θ) SEM", "cos(2θ) SEM", "Radius of SEM circle", "n (mice)"])
c6 = c6.drop(columns="session")
c6.columns = ["Group", "Mouse", "SF (cpd)", "Orientation (°)", "sin(2θ) axis", "cos(2θ) axis"]

b6 = Book("Fig6_source_data.xlsx",
          "Source data for Figure 6 (dsPCA of V1 population activity).\n"
          "WT, 5 mice imaged before VR training; GtdKO+AAV-Gαt2 before and after training, 5 mice. For dsPCA, 80 neurons "
          "were randomly subsampled from each session (all neurons if fewer; fixed seed per session).\n"
          "B: condition-averaged responses (6 orientations × 6 spatial frequencies) of a pseudopopulation "
          "combining the subsampled neurons of all mice in each group, projected onto the three dsPCA stimulus-coding "
          "axes and onto the first three principal components of the stimulus-null subspace (variance explained as "
          "printed on the axes).\n"
          "C: each mouse projected separately onto its own orientation-coding plane; at each SF the six orientation "
          "points of each mouse are centred on their mean. The figure shows the mean over mice; the SEM circle radius is "
          "sqrt(SEM_sin² + SEM_cos²).\n"
          "D: mean distance between adjacent orientations (30° apart) in the orientation-coding plane of each mouse, "
          "per SF (the six adjacent pairs averaged); mean ± SEM over mice.\nA is a schematic.")
b6b = b6b.drop(columns=["Neurons in pseudopopulation", "PC1 variance (%)", "PC2 variance (%)", "PC3 variance (%)"])
b6.add("Fig6B", [(["Figure 6B. Pseudopopulation activity in the stimulus-coding and stimulus-null subspaces",
                   "36 plotted conditions (6 orientations × 6 SFs). dsPCA was fitted on 42 conditions, including "
                   "6 blank-level (SF = 0) conditions that are not plotted. Values are projection scores, as on the "
                   "figure axes."], b6b)])
b6.add("Fig6C", [(["Figure 6C. Orientation-plane coordinates of each mouse (centred per SF)"], c6)])
b6.add("Fig6C_plotted", [(["Figure 6C. Plotted mean over mice and SEM circle radius"], c6s)])
b6.add("Fig6D", per_mouse_blocks("Figure 6D. Mean distance between adjacent orientations", "adj_dist_by_sf",
                                 [(L16[k], g16("adj_dist_by_sf", *k)) for k in L16]))
b6.add("Statistics", stats_block(["Fig. 6"]))
b6.save()

# ================================================================== Supplementary Figure 3
LS3 = {("WT", "pre"): "WT pre-training", ("WT", "post"): "WT post-training"}
s3 = Book("SuppFig3_source_data.xlsx",
          "Source data for Supplementary Figure 3 (training-associated changes in V1 of WT mice).\n"
          "WT, 5 mice imaged before and after VR training (741 and 630 neurons); GtdKO+AAV-Gαt2, 5 mice (513 and 455 "
          "neurons). One imaging session per mouse and time point.\n"
          "A–I: WT before vs after training; C is per neuron (621 pre- and 517 post-training neurons), all other "
          "panels one value per mouse.\n"
          "J–R: change after training (post − pre) of each mouse, at each x-axis point of A–I; in L, the change of the "
          "mean tuning curve of each mouse.\nWT mice: VD017–VD020, VD022. GtdKO+AAV-Gαt2 mice: VD024, VD031–VD034.")
panels = (("A", "resp_by_sf", chance_band), ("B", "resp_by_ori", None), ("D", "osi_cv", None),
          ("E", "sf_dec_vs_n", shuffle_band("sf_dec_vs_n")), ("F", "ori_dec_vs_n", shuffle_band("ori_dec_vs_n")),
          ("G", "sf_dec_by_ori", shuffle_band("sf_dec_by_ori")), ("H", "ori_dec_by_sf", shuffle_band("ori_dec_by_sf")),
          ("I", "adj_dist_by_sf", None))
for p, m, band in panels:
    if p == "D":
        s3.add("SFig3C", tuning_blocks("Supplementary Figure 3C. Orientation tuning curves of visually responsive "
                                       "neurons, WT before and after training",
                                       [(LS3[k], cells16[(cells16.group == "WT") & (cells16.training == k[1])])
                                        for k in LS3], "n = 621 pre- and 517 post-training WT neurons from 5 mice."))
    s3.add(f"SFig3{p}", per_mouse_blocks(f"Supplementary Figure 3{p}. {SPEC[m][2]}", m,
                                         [(LS3[k], g16(m, *k)) for k in LS3], band))
change = dict(zip("ABDEFGHI", "JKMNOPQR"))
for p, m, _ in panels:
    q = change[p]
    if q == "M":
        sess = cells16.groupby(["group", "mouse", "training", "offset"])[["value", "norm_0deg"]].mean().reset_index()
        rows, summ = [], []
        for grp, lab in (("WT", "WT"), ("Rescued", RESC)):
            for col, name in (("value", "Δ response"), ("norm_0deg", "Δ normalized")):
                w = sess[sess.group == grp].pivot_table(index="mouse", columns=["training", "offset"], values=col)
                dlt = w["post"][OFFSETS] - w["pre"][OFFSETS]
                for mouse, r in dlt.iterrows():
                    rows.append([lab, name, mouse] + r.tolist())
                summ += [[lab, name, "mean"] + dlt.mean(0).tolist(),
                         [lab, name, "SEM"] + (dlt.std(0, ddof=1) / np.sqrt(len(dlt))).tolist(),
                         [lab, name, "n (mice)"] + [len(dlt)] * len(OFFSETS)]
        off = [f"{o:+d}°" if o else "0°" for o in OFFSETS]
        s3.add("SFig3L", [(["Supplementary Figure 3L. Change after training of the mean tuning curve of each mouse "
                            "(post − pre)", "The normalized curves equal 1 at 0° by construction, so their change "
                                            "there is 0."], pd.DataFrame(rows, columns=["Group", "Curve", "Mouse"] + off)),
                          (["Plotted summary (mean ± SEM over mice)"],
                           pd.DataFrame(summ, columns=["Group", "Curve", "Statistic"] + off))])
    levels = SPEC[m][0]
    deltas = []
    for grp, lab in (("WT", "WT"), ("Rescued", RESC)):
        dw = wide(g16(m, grp, "post"), levels) - wide(g16(m, grp, "pre"), levels)
        deltas.append((lab, dw.stack().rename("value").reset_index().rename(columns={"level": "level"})))
    s3.add(f"SFig3{q}", per_mouse_blocks(f"Supplementary Figure 3{q}. Change after training (post − pre): "
                                         f"{SPEC[m][2]}", m, deltas))
st16 = pd.read_csv(os.path.join(R16, "stats_resp_by_sf_per_sf.csv"))
st16 = st16[(st16.group == "WT") & (st16.model == "(1|mouse)")][["contrast", "t", "p", "q_BH"]]
st16.columns = ["ART-C contrast (WT)", "t", "p", "p (BH-corrected)"]
s3.add("Statistics", stats_block(["Suppl. Fig. 3"], [(["Per-SF tests in Supplementary Figure 3A"], st16)]))
s3.sheets = sorted(s3.sheets, key=lambda s: s[0])
s3.save()

# ================================================================== Supplementary Figure 4
s4 = Book("SuppFig4_source_data.xlsx",
          "Source data for Supplementary Figure 4 (dsPCA-independent measures of adjacent-orientation separation).\n"
          "Same neurons and trials as Figure 6D. WT, 5 mice imaged before VR training; GtdKO+AAV-Gαt2 before and after "
          "training, 5 mice. One value per mouse and spatial frequency (mean of the six adjacent-orientation pairs); "
          "mean ± SEM over mice.\nA, C, E are schematics.")
for p, m in (("B", "pca3"), ("D", "crossnobis"), ("F", "lda_acc")):
    s4.add(f"SFig4{p}", per_mouse_blocks(f"Supplementary Figure 4{p}. {SPEC[m][2]}", m,
                                         [(L16[k], g16(m, *k, data=geo)) for k in L16]))
sg = pd.read_csv(os.path.join(ROOT, "suppfig4_dspca_alternatives", "stats_geometry_alternatives.csv"))
sg = sg[sg.metric.isin(["pca2", "pca3", "pca5", "pca10"])][["metric", "WT_vs_rescued_pre_p"]]
sg.columns = ["Number of PCs", "p, WT vs GtdKO+AAV-Gαt2 pre-training (group effect)"]
sg["Number of PCs"] = sg["Number of PCs"].str.replace("pca", "")
s4.add("Statistics", stats_block(["Suppl. Fig. 4"],
                                 [(["Supplementary Figure 4B legend: pre-training difference with 2, 3, 5 and 10 PCs"], sg)]))
s4.save()
