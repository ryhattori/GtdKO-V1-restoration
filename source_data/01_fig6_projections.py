"""Fig. 6B and 6C plotted coordinates, regenerated with the analysis code that made the figure
(pipeline/group_analysis.py), read-only: bytecode writing and all file output of the
module are disabled. Writes fig6B_pseudopop_projections.csv and fig6C_orientation_plane_points.csv here.
  6B: run_pseudopopulation_dspca(subsample_neurons=True) -- pseudopopulation of 80 neurons per session (seeded per
      session), standard dsPCA; matches the PC variances printed in the figure.
  6C: _session_points_by_sf_ori(cv50=False) -- per-session dspca_results.npz (80 neurons), each SF polygon centred;
      group mean and SEM radius = sqrt(SEM_x^2 + SEM_y^2), as in _plot_sessionavg_dspca_orient_plane_by_sf_overlay.
"""
import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline", "group_analysis.py")
HERE = os.path.dirname(os.path.abspath(__file__))

def _blocked(*a, **k):
    raise RuntimeError("file output blocked")

spec = importlib.util.spec_from_file_location("summary_plots_ro", SRC)
sp = importlib.util.module_from_spec(spec)
plt.savefig = lambda *a, **k: None
matplotlib.figure.Figure.savefig = lambda *a, **k: None
spec.loader.exec_module(sp)
for name in ("_pseudopopulation_dspca_make_plots", "_pseudopopulation_dspca_make_detailed_plots",
             "_pseudopopulation_dspca_make_superimposed_and_distance_plots"):
    setattr(sp, name, lambda *a, **k: None)
sp.np = type("NP", (), {**{k: getattr(np, k) for k in dir(np) if not k.startswith("__")},
                        "save": _blocked, "savez": _blocked, "savez_compressed": _blocked})
sp.pd.DataFrame.to_csv = _blocked

GROUPS6 = {"WT_BEFORE": "WT", "RESCUED_BEFORE": "GtdKO+AAV-Gαt2 pre-training",
           "RESCUED_AFTER": "GtdKO+AAV-Gαt2 post-training"}

# ---------------------------------------------------------------- 6B
res = sp.run_pseudopopulation_dspca(subsample_neurons=True, out_tag="")
rows = []
for g, label in GROUPS6.items():
    r = res[g]
    pct = 100 * np.asarray(r["targetfree_var"])[:3] / float(r["total_var"])
    print(g, "n =", r["n_neurons"], "PC% =", np.round(pct, 1))
    t = np.asarray(r["targets"]); pt = np.asarray(r["projection_target"]); pf = np.asarray(r["projection_targetfree"])
    ori = (np.degrees(np.arctan2(t[:, 1], t[:, 2])) / 2) % 180
    for i in range(len(t)):
        rows.append({"group": label, "n_neurons": int(r["n_neurons"]), "orientation_deg": round(float(ori[i])),
                     "sf_cpd": float(r["mapped_sf_targets"][i]),
                     "log2SF_axis": pt[i, 0], "sin2theta_axis": pt[i, 1], "cos2theta_axis": pt[i, 2],
                     "null_PC1": pf[i, 0], "null_PC2": pf[i, 1], "null_PC3": pf[i, 2],
                     "null_PC1_var_pct": pct[0], "null_PC2_var_pct": pct[1], "null_PC3_var_pct": pct[2]})
pd.DataFrame(rows).to_csv(os.path.join(HERE, "fig6B_pseudopop_projections.csv"), index=False,
                          ) if False else None
out6b = pd.DataFrame(rows)

# ---------------------------------------------------------------- 6C
mapped = sorted({float(sp.convert_sf_value(float(s))) for s in sp.TARGET_SF_VALUES if float(s) > 0})
oris = np.arange(0, 180, 30, dtype=float)
rows = []
for g, label in GROUPS6.items():
    for sess in sp.GROUPS[g]:
        pts = sp._session_points_by_sf_ori(sess, cv50=False, mapped_levels=mapped, ori_levels=oris)
        mouse = [p for p in sess.replace("/", "\\").split("\\") if p.startswith("VD")][0]
        for sf in mapped:
            for o in oris:
                xy = (pts or {}).get(sf, {}).get(float(o))
                rows.append({"group": label, "mouse": mouse, "session": sess, "sf_cpd": sf, "orientation_deg": int(o),
                             "sin2theta": np.nan if xy is None else xy[0], "cos2theta": np.nan if xy is None else xy[1]})
out6c = pd.DataFrame(rows)
# restore pandas writer for our own outputs
del sp.pd.DataFrame.to_csv
import importlib; importlib.reload(pd)
from pandas.core.generic import NDFrame
pd.DataFrame.to_csv = NDFrame.to_csv
out6b.to_csv(os.path.join(HERE, "fig6B_pseudopop_projections.csv"), index=False)
out6c.to_csv(os.path.join(HERE, "fig6C_orientation_plane_points.csv"), index=False)
print(out6c.groupby("group").mouse.nunique())
