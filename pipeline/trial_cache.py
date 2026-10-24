"""Single-trial responses of each session, cached as one small npz file per session.

load_trials(session) returns delta (neurons x trials, baseline-subtracted response), ori (deg; NaN for blank
trials) and sf (cpd; NaN for blank trials), taken from the continuous-decoder inputs of group_analysis.py.
Used by the cross-validated OSI, the tuning curves and the dsPCA-independent measures.
"""
import hashlib
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

_module = None


def _group_analysis():
    global _module
    if _module is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "group_analysis.py")
        spec = importlib.util.spec_from_file_location("group_analysis", path)
        _module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_module)
    return _module


def load_trials(session):
    path = os.path.join(config.TRIAL_CACHE, f"trials_{hashlib.md5(session.encode()).hexdigest()[:10]}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    ga = _group_analysis()
    d = ga._load_session_continuous_decoder_inputs(session)
    delta = np.asarray(d["delta_per_trial"], float)
    ori = np.full(delta.shape[1], np.nan)
    sf = np.full(delta.shape[1], np.nan)
    for o, value in enumerate(d["unique_orientations"]):
        ori[d["trial_inds_valid_by_ori"][o]] = value
    for s, value in enumerate(d["unique_spatialfreqs"]):
        sf[d["trial_inds_by_sf"][s]] = ga.convert_sf_value(value)
    out = dict(delta=delta.astype(np.float32), ori=ori.astype(np.float32), sf=sf.astype(np.float32))
    os.makedirs(config.TRIAL_CACHE, exist_ok=True)
    np.savez_compressed(path, **out)
    return out
