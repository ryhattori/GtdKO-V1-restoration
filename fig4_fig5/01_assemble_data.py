"""Figures 4 and 5: per-mouse metrics and per-neuron tuning curves in two tidy tables.

fig45_session_metrics.csv  one value per mouse (session) and x-axis level
    group: WT (WT mice before VR training), Pre / Post (GtdKO+AAV-Gat2 before / after VR training),
           GtdKO (Fig. 4C only)
    metric: resp_by_sf, resp_by_ori, osi_cv (cross-validated OSI), sf_dec_vs_n, ori_dec_vs_n,
            sf_dec_by_ori, ori_dec_by_sf
fig45_cells.csv            per-neuron tuning curves aligned to the cross-validated preferred orientation
Inputs: session_metrics/session_metrics_long.csv, osi_cross_validated.csv and tuning_curves_cells.csv, and
responsive_fraction_by_sf.npy of the GtdKO sessions (config.RESULTS_ROOT).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R16 = os.path.join(os.path.dirname(HERE), "session_metrics")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "pipeline")]
import config  # noqa: E402
ROOT = config.RESULTS_ROOT
GTDKO = {"VD026": r"250425\VD026\Left\170um", "VD027": r"250515\VD027\Left\150um", "VD025": r"250715\VD025\Right"}
SF6 = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
METRICS = ["resp_by_sf", "resp_by_ori", "sf_dec_vs_n", "ori_dec_vs_n", "sf_dec_by_ori", "ori_dec_by_sf"]


def group_of(group, training):
    if group == "WT":
        return "WT" if training == "pre" else None
    if group == "Rescued":
        return {"pre": "Pre", "post": "Post"}[training]
    return None


def main():
    s = pd.read_csv(os.path.join(R16, "session_metrics_long.csv"))
    s["g"] = [group_of(g, t) for g, t in zip(s.group, s.training)]
    s = s[s.g.notna() & s.metric.isin(METRICS)]
    parts = [s[["g", "mouse", "metric", "level", "value"]]]

    o = pd.read_csv(os.path.join(R16, "osi_cross_validated.csv"))
    o = o[o.selection == "responsive"]
    o["g"] = [group_of(g, t) for g, t in zip(o.group, o.training)]
    o = o[o.g.notna()]
    parts.append(pd.DataFrame(dict(g=o.g, mouse=o.mouse, metric="osi_cv", level=o.sf, value=o.osi_cv)))

    rows = []
    for mouse, session in GTDKO.items():
        frac = np.load(os.path.join(ROOT, session, "responsive_fraction_by_sf.npy"))[1:7]    # index 0 = blank
        rows += [dict(g="GtdKO", mouse=mouse, metric="resp_by_sf", level=sf, value=v) for sf, v in zip(SF6, frac)]
    parts.append(pd.DataFrame(rows))

    d = pd.concat(parts, ignore_index=True).rename(columns={"g": "group"})
    d.to_csv(os.path.join(HERE, "fig45_session_metrics.csv"), index=False)
    print(d.groupby(["metric", "group"]).mouse.nunique().unstack("group").to_string())

    c = pd.read_csv(os.path.join(R16, "tuning_curves_cells.csv"))
    c["g"] = [group_of(g, t) for g, t in zip(c.group, c.training)]
    c = c[c.g.notna()].drop(columns=["group", "training"]).rename(columns={"g": "group"})
    c.to_csv(os.path.join(HERE, "fig45_cells.csv"), index=False)
    print(c[c.offset == 0].groupby("group").cell.nunique().to_string())


if __name__ == "__main__":
    main()
