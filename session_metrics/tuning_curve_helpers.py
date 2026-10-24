"""Orientation tuning curves before vs after training (context for the Fig. 5C OSI change).

Neurons: at each SF shown in Fig. 5C (0.02-0.32 cpd), the neurons classified as visually responsive at
that SF (the Fig. 5C selection, responsive_neurons_by_sf.npy).
Response: mean response per orientation at that SF (per-neuron z-scored deconvolved activity, minus the
mean on blank trials; delta_per_trial of the 260319 pipeline).
Alignment: each neuron's curve is aligned to its preferred orientation. The preference is cross-validated:
the trials of each orientation are split into two random halves, the preference is taken from one half
and the responses from the other, and the two directions are averaged. This keeps trial noise from
inflating the aligned peak (which would otherwise inflate it more in noisier, weaker neurons). The
conventional version (preference and responses from all trials) is saved too.
Averaging: neurons -> SF -> mouse (SFs weighted equally); mean +/- SEM over mice (n = 5 per group).
Statistics: ART mixed models with mouse as a random intercept, as in Figs 4-6:
value ~ training * orientation + (1 | mouse), per group (08_tuning_curve_stats.R).
Figure: 11_plot_osi_tuning_figure.py.
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "pipeline")]
import config  # noqa: E402
import trial_cache  # noqa: E402
ROOT = config.RESULTS_ROOT
HERE = os.path.dirname(os.path.abspath(__file__))
ORIS = np.array([0, 30, 60, 90, 120, 150.])
SF = [0.02, 0.04, 0.08, 0.16, 0.32]                       # SFs shown in Fig. 5C
OFFSETS = np.array([0, 30, 60, 90, -60, -30])             # offset of column j after alignment

PAIRS = {
    "WT": [("VD017", r"250423\VD017\Left\170um", r"250716\VD017\Left"),
           ("VD018", r"250424\VD018\Left\190um", r"250717\VD018\Right"),
           ("VD019", r"250424\VD019\Left\120um", r"250717\VD019\Left"),
           ("VD020", r"250625\VD020\Left\170 um", r"250707\VD020\Left"),
           ("VD022", r"250724\VD022\Left", r"250912\VD022\Left")],
    "Rescued": [("VD024", r"250424\VD024\Right\150um", r"250716\VD024\Right"),
                ("VD031", r"250701\VD031\Left\170", r"250812\VD031\Left"),
                ("VD032", r"250715\VD032\Left", r"251115\VD032\Left2"),
                ("VD033", r"250814\VD033\Left", r"251115\VD033\Left"),
                ("VD034", r"260223\VD034\Left", r"260320\VD034\Left")],
}


def load(session):
    z = trial_cache.load_trials(session)
    mask = np.load(os.path.join(ROOT, session, "responsive_neurons_by_sf.npy")).astype(bool)   # (7 SF incl. 0, N)
    return z["delta"].astype(float), z["ori"], z["sf"], mask


def align(R, pref):
    """Rotate each row so that its preferred orientation is column 0 (columns = OFFSETS)."""
    cols = (pref[:, None] + np.arange(6)[None]) % 6
    return np.take_along_axis(R, cols, axis=1)


def session_curves(session, rng):
    delta, ori, sf, mask = load(session)
    out = []
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
        R = np.stack([np.nanmean(x[:, t], 1) for t in trials], 1)
        cv = (align(RB, RA.argmax(1)) + align(RA, RB.argmax(1))) / 2
        conv = align(R, R.argmax(1))
        out.append(dict(sf=s, n=idx.size, cv=cv.mean(0), conventional=conv.mean(0)))
    return out


def main():
    rows = []
    for group, pairs in PAIRS.items():
        for mouse, pre, post in pairs:
            for training, session in (("pre", pre), ("post", post)):
                rng = np.random.default_rng(int(hashlib.md5(session.encode()).hexdigest()[:8], 16))
                for c in session_curves(session, rng):
                    for version in ("cv", "conventional"):
                        for j, off in enumerate(OFFSETS):
                            rows.append(dict(group=group, mouse=mouse, training=training, sf=c["sf"],
                                             n_neurons=c["n"], version=version, offset=int(off),
                                             value=c[version][j]))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(HERE, "tuning_curves_by_sf.csv"), index=False)
    # mouse-level curves: mean over SFs; normalized version divides by the response at the preferred orientation
    m = d.groupby(["version", "group", "mouse", "training", "offset"], as_index=False).value.mean()
    pref = m[m.offset == 0].set_index(["version", "group", "mouse", "training"]).value
    m["normalized"] = m.value / pref.loc[list(zip(m.version, m.group, m.mouse, m.training))].to_numpy()
    m.to_csv(os.path.join(HERE, "tuning_curves_mouse.csv"), index=False)


if __name__ == "__main__":
    main()
