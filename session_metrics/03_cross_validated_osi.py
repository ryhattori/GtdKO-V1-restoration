"""Orientation selectivity corrected for trial-noise bias.

The Fig. 5C OSI (vector-sum OSI of each neuron's rectified mean responses, averaged over neurons) is
biased upward by trial-to-trial noise: noise adds a random vector to every neuron's vector sum, so even
an untuned neuron gets OSI > 0, and the bias is larger when responses are weaker relative to the
noise. A change in response strength therefore changes the Fig. 5C OSI even if tuning is unchanged
(simulation in _cvosi_simulation(): with the true OSI fixed at 0, 0.1 or 0.3, the Fig. 5C OSI falls as
the response amplitude rises).

Cross-validated (population) OSI, per session and SF:
    q_n  = Re( V_n(A) * conj(V_n(B)) ),  V = sum_theta R(theta) exp(2i theta), from two random halves
           A and B of each orientation's trials (averaged over 10 random splits); E[q_n] = |V_n|^2
    OSI_cv = sqrt( max(mean_n q_n, 0) ) / mean_n S_n,   S_n = sum_theta R_n(theta) (all trials)
i.e. the root-mean-square tuned amplitude over the mean response of the population. The simulation
recovers the true OSI at every response amplitude (per-neuron ratios are unstable at low SNR, so the
ratio is taken after averaging over neurons).

Neuron sets per SF (0.02-0.32 cpd): responsive (the Fig. 5C selection) and matched (top-k by response
strength, k = the mouse's pre-training responsive fraction; see 05_osi_selection_check.py).
Also saved: the Fig. 5C OSI and its shuffle null (orientation labels shuffled within SF).
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "pipeline")]
import config  # noqa: E402
import trial_cache  # noqa: E402
ROOT = config.RESULTS_ROOT
HERE = os.path.dirname(os.path.abspath(__file__))
ORIS = np.array([0, 30, 60, 90, 120, 150.])
E = np.exp(2j * np.deg2rad(ORIS))
SF = [0.02, 0.04, 0.08, 0.16, 0.32]
N_SPLITS, N_SHUFFLES = 10, 10

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


def fig5c_osi(R):
    Rr = R.clip(min=0)
    den = Rr.sum(1)
    return np.where(den > 0, np.abs(Rr @ E) / np.where(den > 0, den, 1), np.nan)


def cv_osi(x, trials, rng):
    """Population cross-validated OSI for neurons x (rows) given each orientation's trial indices."""
    q = np.zeros(len(x))
    for _ in range(N_SPLITS):
        halves = [np.array_split(rng.permutation(t), 2) for t in trials]
        RA = np.stack([np.nanmean(x[:, h[0]], 1) for h in halves], 1)
        RB = np.stack([np.nanmean(x[:, h[1]], 1) for h in halves], 1)
        q += np.real((RA @ E) * np.conj(RB @ E)) / N_SPLITS
    S = np.stack([np.nanmean(x[:, t], 1) for t in trials], 1).sum(1)
    return np.sqrt(max(q.mean(), 0.0)) / S.mean()


def session_rows(delta, ori, sf, mask, f_pre, rng):
    rows = []
    for k, s in enumerate(SF, start=1):
        at = np.isclose(sf, s)
        trials = [np.flatnonzero((ori == o) & at) for o in ORIS]
        t = stats.ttest_ind(delta[:, at], delta[:, sf == 0], axis=1, equal_var=False).statistic
        n_top = max(1, int(round(f_pre[k] * len(t))))
        sets = {"responsive": np.flatnonzero(mask[k]), "matched": np.argsort(-t)[:n_top]}
        for name, idx in sets.items():
            if idx.size == 0:
                continue
            x = delta[idx]
            R = np.stack([np.nanmean(x[:, tt], 1) for tt in trials], 1)
            allt = np.concatenate(trials)
            lab = np.concatenate([[j] * len(tt) for j, tt in enumerate(trials)])
            null_std, null_cv = [], []
            for _ in range(N_SHUFFLES):
                perm = rng.permutation(lab)
                shuf = [allt[perm == j] for j in range(6)]
                null_std.append(np.nanmean(fig5c_osi(np.stack([np.nanmean(x[:, tt], 1) for tt in shuf], 1))))
                null_cv.append(cv_osi(x, shuf, rng))
            rows.append(dict(sf=s, selection=name, n=idx.size, mean_response=R.mean(),
                             osi_fig5c=np.nanmean(fig5c_osi(R)), osi_fig5c_null=np.mean(null_std),
                             osi_cv=cv_osi(x, trials, rng), osi_cv_null=np.mean(null_cv)))
    return rows


def _cvosi_simulation(n_neur=4000, n_trials=60, noise=0.6, seed=0):
    """Fig. 5C OSI vs cross-validated OSI for simulated populations with known OSI."""
    rng = np.random.default_rng(seed)
    out = []
    for amp in (0.05, 0.1, 0.2):
        for true in (0.0, 0.1, 0.3):
            pref = rng.uniform(0, 180, n_neur)
            mu = amp * (1 + 2 * true * np.cos(2 * np.deg2rad(ORIS[None] - pref[:, None])))
            X = mu[:, :, None] + rng.normal(0, noise, (n_neur, 6, n_trials))
            flat = X.reshape(n_neur, -1)
            trials = [np.arange(j * n_trials, (j + 1) * n_trials) for j in range(6)]
            out.append(dict(amplitude=amp, true_osi=true, osi_fig5c=np.nanmean(fig5c_osi(X.mean(2))),
                            osi_cv=cv_osi(flat, trials, rng)))
    return pd.DataFrame(out)


def main():
    rows = []
    for group, pairs in PAIRS.items():
        for mouse, pre, post in pairs:
            data = {tr: load(s) for tr, s in (("pre", pre), ("post", post))}
            f_pre = data["pre"][3].mean(1)                       # pre-training responsive fraction per SF
            for tr, s in (("pre", pre), ("post", post)):
                rng = np.random.default_rng(int(hashlib.md5(s.encode()).hexdigest()[:8], 16))
                for r in session_rows(*data[tr], f_pre, rng):
                    rows.append(dict(group=group, mouse=mouse, training=tr, **r))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(HERE, "osi_cross_validated.csv"), index=False)
    sim = _cvosi_simulation()
    sim.to_csv(os.path.join(HERE, "osi_cv_simulation.csv"), index=False)
    print(sim.round(3).to_string(index=False))
    g = (d.groupby(["selection", "group", "mouse", "training"])[["mean_response", "osi_fig5c", "osi_fig5c_null",
                                                                  "osi_cv", "osi_cv_null"]].mean()
         .groupby(level=[0, 1, 3]).mean())
    print(g.round(3))


if __name__ == "__main__":
    main()
