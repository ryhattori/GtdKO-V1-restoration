"""Supplementary Figure 4: adjacent-orientation separation measured without dsPCA.

Figure 6D measures, per session and SF, the mean distance between population states for adjacent orientations
(30 deg apart, six pairs around the 180-deg loop) in the dsPCA orientation plane. The same quantity is computed here
without dsPCA, on the same neurons (the 80-neuron dsPCA subset of each session, dspca_results.npz:
neuron_subset_idx) and the same trials:

  dspca       Fig. 6D metric, recomputed from the saved dsPCA projections (reference)
  pca3        condition-averaged PCA: PCA of the 36 condition means (centered), distance in the top 3 PCs
  crossnobis  cross-validated Mahalanobis distance between the trial means of the two orientations (5 folds x 5
              repeats; Ledoit-Wolf noise covariance from the residuals of all stimulus trials; squared distance per
              neuron); 0 in expectation if the two orientations evoke the same mean response
  lda_acc     cross-validated linear discriminability: held-out balanced accuracy of a shrinkage LDA classifier
              separating the two orientations (standardized features, equal class priors, stratified 5-fold x 3
              repeats); balanced accuracy with equal priors keeps chance at 0.5 with unequal trial counts

Additional measures (not in the figure): pca2, pca5, pca10 (condition-averaged PCA with 2, 5 or 10 PCs), euclid
(Euclidean distance between condition means in the full neural space) and xeuclid (cross-validated Euclidean
distance), and the variance composition of the top 3 PCs and of the dsPCA orientation plane
(variance_composition.csv).

Trial data: baseline-subtracted single-trial responses of all neurons (pipeline/trial_cache.py).
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "pipeline")]
import config  # noqa: E402
import trial_cache  # noqa: E402
ROOT = config.RESULTS_ROOT
HERE = os.path.dirname(os.path.abspath(__file__))
SF = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
ORIS = [0, 30, 60, 90, 120, 150]
PAIRS_ADJ = [(ORIS[i], ORIS[(i + 1) % 6]) for i in range(6)]          # includes 150 -> 0
N_FOLDS, N_REPEATS_XNOBIS, N_REPEATS_LDA = 5, 5, 3

SESSIONS = {  # group, mouse, training, session (pipeline/group_analysis.py GROUPS)
    ("WT", "VD017"): (r"250423\VD017\Left\170um", r"250716\VD017\Left"),
    ("WT", "VD018"): (r"250424\VD018\Left\190um", r"250717\VD018\Right"),
    ("WT", "VD019"): (r"250424\VD019\Left\120um", r"250717\VD019\Left"),
    ("WT", "VD020"): (r"250625\VD020\Left\170 um", r"250707\VD020\Left"),
    ("WT", "VD022"): (r"250724\VD022\Left", r"250912\VD022\Left"),
    ("Rescued", "VD024"): (r"250424\VD024\Right\150um", r"250716\VD024\Right"),
    ("Rescued", "VD031"): (r"250701\VD031\Left\170", r"250812\VD031\Left"),
    ("Rescued", "VD032"): (r"250715\VD032\Left", r"251115\VD032\Left2"),
    ("Rescued", "VD033"): (r"250814\VD033\Left", r"251115\VD033\Left"),
    ("Rescued", "VD034"): (r"260223\VD034\Left", r"260320\VD034\Left"),
}


def adjacent_mean(points_by_ori):
    """Mean Euclidean distance between adjacent orientations around the loop."""
    return float(np.mean([np.linalg.norm(points_by_ori[a] - points_by_ori[b]) for a, b in PAIRS_ADJ]))


def cv_distances(xa, xb, precision, rng):
    """Cross-validated squared Mahalanobis (crossnobis) and Euclidean distances per neuron
    between two trial sets, from the same folds."""
    n = xa.shape[1]
    maha, eucl = [], []
    for _ in range(N_REPEATS_XNOBIS):
        fa = np.array_split(rng.permutation(len(xa)), N_FOLDS)
        fb = np.array_split(rng.permutation(len(xb)), N_FOLDS)
        D = np.stack([xa[ia].mean(0) - xb[ib].mean(0) for ia, ib in zip(fa, fb)])     # (folds, n)
        for G, out in ((D @ precision @ D.T, maha), (D @ D.T, eucl)):
            out.append((G.sum() - np.trace(G)) / (N_FOLDS * (N_FOLDS - 1)))
    return float(np.mean(maha)) / n, float(np.mean(eucl)) / n


def lda_accuracy(xa, xb, seed):
    X = np.vstack([xa, xb]); y = np.r_[np.zeros(len(xa)), np.ones(len(xb))]
    clf = make_pipeline(StandardScaler(),
                        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]))
    accs = [cross_val_score(clf, X, y, scoring="balanced_accuracy",
                            cv=StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed + r)).mean()
            for r in range(N_REPEATS_LDA)]
    return float(np.mean(accs))


def variance_composition(trials, bases):
    """Share of condition-mean variance in each subspace due to SF, orientation, their
    interaction, and (expected) trial noise."""
    X = np.stack([[trials[(o, s)].mean(0) for s in SF] for o in ORIS])       # (ori, sf, n)
    Xc = X - X.mean((0, 1))
    ori_eff = np.broadcast_to(Xc.mean(1, keepdims=True), Xc.shape)
    sf_eff = np.broadcast_to(Xc.mean(0, keepdims=True), Xc.shape)
    parts = {"sf": sf_eff, "ori": ori_eff, "sf_x_ori": Xc - ori_eff - sf_eff}
    resid = np.vstack([t - t.mean(0) for t in trials.values()])
    noise_cov = resid.T @ resid / (len(resid) - len(trials))                  # pooled, single trial
    k = len(trials)
    inv_n = sum(1 / len(t) for t in trials.values())
    out = {}
    for name, B in bases.items():
        Q, _ = np.linalg.qr(B)                                                 # orthonormal basis
        total = ((Xc.reshape(36, -1) @ Q) ** 2).sum()
        for part, A in parts.items():
            out[f"{part}_{name}"] = float(((A.reshape(36, -1) @ Q) ** 2).sum() / total)
        out[f"noise_{name}"] = float((k - 1) / k * inv_n * np.trace(Q.T @ noise_cov @ Q) / total)
    return out


def session_metrics(session, seed):
    z = trial_cache.load_trials(session)
    ds = np.load(os.path.join(ROOT, session, "dspca_results.npz"), allow_pickle=True)
    idx = np.asarray(ds["neuron_subset_idx"], int)
    delta = z["delta"].astype(float)[idx]                                  # (n, trials)
    ori, sf = z["ori"], z["sf"]
    trials = {(o, s): delta[:, (ori == o) & np.isclose(sf, s)].T for o in ORIS for s in SF}   # (60, n)
    X = np.stack([trials[(o, s)].mean(0) for o in ORIS for s in SF])                          # (36, n)
    keys = [(o, s) for o in ORIS for s in SF]

    # dsPCA reference: saved projections onto the sin/cos(2 theta) axes (as in Fig. 6D)
    co, cs = np.asarray(ds["condition_orientations"], float), np.asarray(ds["condition_spatialfreqs"], float)
    proj = np.asarray(ds["projection_target"], float)[:, 1:3]
    dspca_pts = {(o, s): proj[(np.isclose(co, o)) & np.isclose(cs, s)][0] for o in ORIS for s in SF}

    # condition-averaged PCA (unsupervised): top k PCs (k = 3 in the figure)
    Xc = X - X.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pca_pts = {k: dict(zip(keys, Xc @ Vt[:k].T)) for k in (2, 3, 5, 10)}
    pca_pts["all"] = dict(zip(keys, X))
    var3 = float((S[:3] ** 2).sum() / (S ** 2).sum())
    composition = variance_composition(trials, {"pc3": Vt[:3].T,
                                                "dspca_ori": np.asarray(ds["ax_targets"], float)[:, 1:3]})

    # noise covariance from all stimulus trials' residuals
    resid = np.vstack([t - t.mean(0) for t in trials.values()])
    precision = np.linalg.inv(LedoitWolf().fit(resid).covariance_)

    rng = np.random.default_rng(seed)
    rows = []
    for s in SF:
        rows.append(dict(sf=s, metric="dspca", value=adjacent_mean({o: dspca_pts[(o, s)] for o in ORIS})))
        for k, name in ((3, "pca3"), (2, "pca2"), (5, "pca5"), (10, "pca10"), ("all", "euclid")):
            rows.append(dict(sf=s, metric=name, value=adjacent_mean({o: pca_pts[k][(o, s)] for o in ORIS})))
        cv = np.array([cv_distances(trials[(a, s)], trials[(b, s)], precision, rng) for a, b in PAIRS_ADJ])
        rows.append(dict(sf=s, metric="crossnobis", value=float(cv[:, 0].mean())))
        rows.append(dict(sf=s, metric="xeuclid", value=float(cv[:, 1].mean())))
        rows.append(dict(sf=s, metric="lda_acc",
                         value=float(np.mean([lda_accuracy(trials[(a, s)], trials[(b, s)], seed)
                                              for a, b in PAIRS_ADJ]))))
    return rows, len(idx), var3, composition


def main():
    out, info = [], []
    for k, ((group, mouse), (pre, post)) in enumerate(SESSIONS.items()):
        for training, session in (("pre", pre), ("post", post)):
            seed = int(hashlib.md5(session.encode()).hexdigest()[:8], 16)      # reproducible per session
            rows, n_used, var3, composition = session_metrics(session, seed)
            for r in rows:
                r.update(group=group, mouse=mouse, training=training)
            out += rows
            info.append(dict(group=group, mouse=mouse, training=training, session=session,
                             n_neurons_used=n_used, pca_top3_variance=var3, **composition))
            print(f"done {group:7s} {mouse} {training:4s} ({n_used} neurons; top-3 PCs explain {100 * var3:.0f}%)")
    pd.DataFrame(out).to_csv(os.path.join(HERE, "geometry_metrics_long.csv"), index=False)
    pd.DataFrame(info).to_csv(os.path.join(HERE, "variance_composition.csv"), index=False)
    if os.path.exists(os.path.join(HERE, "sessions.csv")):
        os.remove(os.path.join(HERE, "sessions.csv"))                     # superseded by variance_composition.csv


if __name__ == "__main__":
    main()
