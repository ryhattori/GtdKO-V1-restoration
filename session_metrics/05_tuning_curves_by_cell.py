"""Tuning curves with cells as the unit (companion to 07, which averages per mouse).

Cells: neurons responsive (the Fig. 5C selection) at >= 1 SF in 0.02-0.32 cpd. At each SF where a cell is
responsive, its tuning curve is aligned to the cross-validated preferred orientation exactly as in 07 (same
random splits); the cell's curve is the mean over those SFs.
Normalized to 0 deg (norm_0deg): each cell's curve divided by the mean response at the preferred orientation
(0 deg) of all cells in its session, so the pooled curve is exactly 1 at 0 deg and session-level differences
in response gain are removed. Dividing each cell by its own 0-deg response is not used: with
cross-validated alignment that single value is near zero for some weakly tuned cells, and the ratios
explode (e.g. rescued pre-training, 90 deg: mean 3.1, median 0.8, maximum 483).
Statistics (13_tuning_cells_stats.R): ART mixed models on cell-level data with mouse, session,
session x orientation and cell as random intercepts, so cells are not treated as independent replicates.
"""
import hashlib
import importlib.util
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("tuning07", os.path.join(HERE, "tuning_curve_helpers.py"))
t07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t07)
load, align, SF, ORIS, OFFSETS, PAIRS = t07.load, t07.align, t07.SF, t07.ORIS, t07.OFFSETS, t07.PAIRS


def cell_curves(session, rng):
    """Per-cell aligned curves, averaged over the SFs where the cell is responsive (same splits as 07)."""
    delta, ori, sf, mask = load(session)
    total = np.zeros((delta.shape[0], 6))
    count = np.zeros(delta.shape[0])
    for k, s in enumerate(SF, start=1):
        idx = np.flatnonzero(mask[k])
        if idx.size == 0:
            continue
        x = delta[idx]
        at = np.isclose(sf, s)
        trials = [np.flatnonzero((ori == o) & at) for o in ORIS]
        halves = [np.array_split(rng.permutation(t), 2) for t in trials]
        RA = np.stack([np.nanmean(x[:, h[0]], 1) for h in halves], 1)
        RB = np.stack([np.nanmean(x[:, h[1]], 1) for h in halves], 1)
        total[idx] += (align(RB, RA.argmax(1)) + align(RA, RB.argmax(1))) / 2
        count[idx] += 1
    keep = np.flatnonzero(count > 0)
    return keep, total[keep] / count[keep, None], count[keep]


def main():
    rows = []
    for group, pairs in PAIRS.items():
        for mouse, pre, post in pairs:
            for training, session in (("pre", pre), ("post", post)):
                rng = np.random.default_rng(int(hashlib.md5(session.encode()).hexdigest()[:8], 16))
                cells, curves, n_sf = cell_curves(session, rng)
                session_peak = curves[:, list(OFFSETS).index(0)].mean()     # session mean response at 0 deg
                for c, curve, n in zip(cells, curves, n_sf):
                    for j, off in enumerate(OFFSETS):
                        rows.append(dict(group=group, mouse=mouse, training=training,
                                         session=f"{mouse}_{training}", cell=f"{mouse}_{training}_{c}", n_sf=int(n),
                                         offset=int(off), value=curve[j], norm_0deg=curve[j] / session_peak))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(HERE, "tuning_curves_cells.csv"), index=False)
    n = d[d.offset == 0].groupby(["group", "training", "mouse"]).size().unstack("training")[["pre", "post"]]
    print("cells per session:\n", n.to_string())
    s = d[d.offset == 0].groupby(["group", "training"]).size()
    print("\ncells per group:\n", s.to_string())
    q = d.groupby(["group", "training", "offset"]).norm_0deg.mean().unstack("offset")[[-60, -30, 0, 30, 60, 90]]
    print("\nmean normalized to 0 deg:\n", q.round(3).to_string())


if __name__ == "__main__":
    main()
