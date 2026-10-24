"""Per-session V1 metrics of WT and GtdKO+AAV-Gat2 mice before and after VR training.

Reads the per-session results (config.RESULTS_ROOT; pipeline/session_analysis.py) and applies the same derivations
as pipeline/group_analysis.py (prepare_session_summaries and the decoding-vs-N functions):

  resp_by_sf     responsive fraction by spatial frequency (non-zero SFs)
  resp_by_ori    responsive fraction by orientation, 0.64 cpd excluded
  osi_by_sf      mean conventional OSI of neurons responsive at that SF (not used in the figures)
  sf_dec_vs_n    ridge SF decoding (Pearson r), mean over orientations, N = 10..80
  ori_dec_vs_n   ridge orientation decoding (circular r), mean over 0.02-0.32 cpd, N = 10..80
  sf_dec_by_ori  ridge SF decoding by orientation, N = 80
  ori_dec_by_sf  ridge orientation decoding by SF (0.02-0.32 cpd), N = 80
  adj_dist_by_sf mean distance between adjacent orientations (30 deg) in the dsPCA orientation plane, per SF
                 (dspca_results.npz, 80 neurons)                                                   Fig. 6D

Outputs: session_metrics_long.csv (one row per mouse x session x metric x level) and sessions.csv (inventory).
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "pipeline")]
import config  # noqa: E402
RESULTS_ROOT = config.RESULTS_ROOT
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# (mouse, pre, post) as in MOUSE_PAIRS of pipeline/group_analysis.py.
PAIRS = {
    "WT": [
        ("VD017", r"250423\VD017\Left\170um", r"250716\VD017\Left"),
        ("VD018", r"250424\VD018\Left\190um", r"250717\VD018\Right"),
        ("VD019", r"250424\VD019\Left\120um", r"250717\VD019\Left"),
        ("VD020", r"250625\VD020\Left\170 um", r"250707\VD020\Left"),
        ("VD022", r"250724\VD022\Left", r"250912\VD022\Left"),
    ],
    "Rescued": [
        ("VD024", r"250424\VD024\Right\150um", r"250716\VD024\Right"),
        ("VD031", r"250701\VD031\Left\170", r"250812\VD031\Left"),
        ("VD032", r"250715\VD032\Left", r"251115\VD032\Left2"),
        ("VD033", r"250814\VD033\Left", r"251115\VD033\Left"),
        ("VD034", r"260223\VD034\Left", r"260320\VD034\Left"),
    ],
    # Removed from the manuscript for training failure; collected only for a
    # sensitivity analysis.
    "Rescued_VD029_excluded": [
        ("VD029", r"250701\VD029\Left", r"250812\VD029\Left"),
    ],
}

SF_LABELS = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
ORI_DEC_SF = [0.02, 0.04, 0.08, 0.16, 0.32]          # 0.64 cpd excluded, as in Fig. 5E/G
CELL_COUNTS = [10, 20, 30, 40, 50, 60, 70, 80]


def load(session, name):
    path = os.path.join(RESULTS_ROOT, session, name)
    return np.load(path, allow_pickle=True) if os.path.exists(path) else None


def session_metrics(session):
    meta = load(session, "orientation_decoding_metadata.npz")
    if meta is None:
        return None
    sfs = np.asarray(meta["unique_spatialfreqs"], float)
    oris = np.asarray(meta["unique_orientations"], float)
    nz = sfs > 0
    if nz.sum() != len(SF_LABELS):
        raise ValueError(f"{session}: unexpected SF set {sfs}")
    ori_sf_idx = np.flatnonzero(nz)[:len(ORI_DEC_SF)]
    out = {"n_neurons": int(np.asarray(load(session, "n_neurons.npy")))}

    out["resp_by_sf"] = dict(zip(SF_LABELS, np.asarray(load(session, "responsive_fraction_by_sf.npy"), float)[nz]))

    rbo = load(session, "responsive_fraction_by_orientation_excluding_064.npz")
    if rbo is not None:
        out["resp_by_ori"] = dict(zip(np.asarray(rbo["orientations"], float),
                                      np.asarray(rbo["responsive_fraction_by_orientation"], float)))

    resp = np.asarray(load(session, "responsive_neurons_by_sf.npy")).astype(bool)   # (SF, neurons)
    osi = np.asarray(load(session, "osi_spf.npy"), float)                          # (neurons, SF)
    osi_mean = [np.nanmean(osi[resp[k], k]) if resp[k].any() else np.nan for k in range(resp.shape[0])]
    out["osi_by_sf"] = dict(zip(SF_LABELS, np.asarray(osi_mean)[nz]))

    dec = load(session, "decoding_cellcount_results.npz")
    cc = list(np.asarray(dec["cell_counts"], int))
    ori_by_sf = np.asarray(dec["orientation_continuous_circular_corr_by_sf"], float)   # (N, SF incl. 0)
    sf_by_ori = np.asarray(dec["sf_continuous_pearson_by_orientation"], float)          # (N, orientation)
    out["sf_dec_vs_n"] = {n: np.nanmean(sf_by_ori[cc.index(n)]) for n in CELL_COUNTS}
    out["ori_dec_vs_n"] = {n: np.nanmean(ori_by_sf[cc.index(n)][ori_sf_idx]) for n in CELL_COUNTS}
    i80 = cc.index(80)
    out["sf_dec_by_ori"] = dict(zip(oris, sf_by_ori[i80]))
    out["ori_dec_by_sf"] = dict(zip(ORI_DEC_SF, ori_by_sf[i80][ori_sf_idx]))

    # Fig. 6D: six adjacent edge lengths (wrapped) in the sin/cos(2 theta) plane, averaged per SF
    ds = load(session, "dspca_results.npz")
    proj = np.asarray(ds["projection_target"], float)
    c_ori = np.mod(np.round(np.asarray(ds["condition_orientations"], float) / 30.0) * 30.0, 180.0)
    c_sf = np.asarray(ds["condition_spatialfreqs"], float)
    adj = {}
    for sf in SF_LABELS:
        pts = np.vstack([np.nanmean(proj[np.isclose(c_sf, sf) & np.isclose(c_ori, o), 1:3], axis=0)
                         for o in range(0, 180, 30)])
        loop = np.vstack([pts, pts[:1]])
        adj[sf] = float(np.nanmean(np.sqrt((np.diff(loop, axis=0) ** 2).sum(1))))
    out["adj_dist_by_sf"] = adj
    return out


def main():
    rows, inv = [], []
    for group, pairs in PAIRS.items():
        for mouse, pre, post in pairs:
            for training, session in (("pre", pre), ("post", post)):
                m = session_metrics(session)
                inv.append({"group": group, "mouse": mouse, "training": training, "session": session,
                            "available": m is not None, "n_neurons": m["n_neurons"] if m else np.nan,
                            "has_resp_by_ori": bool(m and "resp_by_ori" in m)})
                if m is None:
                    print(f"[missing] {group} {mouse} {training}: {session}")
                    continue
                for metric, vals in m.items():
                    if metric == "n_neurons":
                        continue
                    for level, v in vals.items():
                        rows.append({"group": group, "mouse": mouse, "training": training,
                                     "metric": metric, "level": level, "value": float(v)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "session_metrics_long.csv"), index=False)
    inv = pd.DataFrame(inv)
    inv.to_csv(os.path.join(OUT_DIR, "sessions.csv"), index=False)
    print(inv.to_string(index=False))


if __name__ == "__main__":
    main()
