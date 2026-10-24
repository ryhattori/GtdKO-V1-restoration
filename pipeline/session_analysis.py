import os
import glob
import zlib
from typing import List
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
import seaborn as sns
import json
from scipy.optimize import curve_fit
from scipy.stats import ttest_rel, wilcoxon
import warnings


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA

from dspca.dspca import dsPCA
from dspca.utils import adjust_lightness

# ─────────────────────────────────────────────
# ROOT PATHS AND SESSION LIST
# ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
IMAGE_ROOT       = config.RAW_IMAGING_ROOT
PROC_IMAGE_ROOT  = config.SUITE2P_ROOT
RESULTS_ROOT     = config.RESULTS_ROOT

SESSION_LIST = [
    r"250423\VD017\Left\170um",
    r"250424\VD018\Left\190um",
    r"250424\VD019\Left\120um",
    r"250625\VD020\Left\170 um",
    r"250724\VD022\Left",
    r"250716\VD017\Left",
    r"250717\VD018\Right",
    r"250717\VD019\Left",
    r"250707\VD020\Left",
    r"250912\VD022\Left",
    r"250424\VD024\Right\150um",
    r"250701\VD029\Left",
    r"250701\VD031\Left\170",
    r"250715\VD032\Left",
    r"250814\VD033\Left",
    r"250716\VD024\Right",
    r"250812\VD029\Left",
    r"250812\VD031\Left",
    r"251115\VD032\Left2",
    r"251115\VD033\Left",
    r"250425\VD026\Left\170um",
    r"250515\VD027\Left\150um",
    r"250715\VD025\Right",
    r"260223\VD034\Left",
    r"260320\VD034\Left",
]


def _safe_nanmean(x: np.ndarray) -> float:
    """np.nanmean that returns NaN instead of emitting warnings on empty/all-NaN."""
    x = np.asarray(x, dtype=float)
    if x.size == 0 or not np.any(np.isfinite(x)):
        return float('nan')
    return float(np.nanmean(x))


def _safe_sem(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute SEM with NaN for entries with <2 finite samples (avoids ddof warnings)."""
    x = np.asarray(x, dtype=float)
    n = np.sum(np.isfinite(x), axis=axis)
    sem = np.full(x.shape[1] if (x.ndim == 2 and axis == 0) else x.shape[0], np.nan, dtype=float)
    if x.ndim == 2 and axis == 0:
        for j in range(x.shape[1]):
            nj = int(n[j])
            if nj >= 2:
                sem[j] = float(np.nanstd(x[:, j], ddof=1) / np.sqrt(nj))
        return sem
    with np.errstate(invalid='ignore', divide='ignore'):
        sem = np.nanstd(x, axis=axis, ddof=1) / np.sqrt(np.maximum(n, 1))
    sem = np.where(n >= 2, sem, np.nan)
    return sem


def _safe_nanmean_axis0(mat: np.ndarray) -> np.ndarray:
    """Column-wise nanmean that returns NaN for columns with no finite values."""
    mat = np.asarray(mat, dtype=float)
    if mat.ndim != 2:
        return np.asarray(np.nanmean(mat), dtype=float)
    out = np.full(mat.shape[1], np.nan, dtype=float)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        if np.any(np.isfinite(col)):
            out[j] = float(np.nanmean(col))
    return out


# For fair Before vs After comparisons, subsample the same number of neurons
# within each session for PCA/dsPCA.
SUBSAMPLE_N_NEURONS_FOR_PCA_DSPCA = 80
SUBSAMPLE_NEURON_SEED = 12345

# Target spatial frequency values as used in the single-session script
# Map original SF values to desired values prior to log-scaling
# [7.0184, 14.7623, 29.8831, 59.9448, 119.9789, 240.0023] -> [0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
ORIG_SF_MAP = {
    7.0184: 0.02,
    14.7623: 0.04,
    29.8831: 0.08,
    59.9448: 0.16,
    119.9789: 0.32,
    240.0023: 0.64,
}


def convert_sf_value(sf: float) -> float:
    """Convert original SF to desired value using nearest match with tolerance."""
    if sf <= 0:
        return 0.0
    for k, v in ORIG_SF_MAP.items():
        if abs(sf - k) < 1e-3:
            return v
    closest_k = min(ORIG_SF_MAP.keys(), key=lambda x: abs(sf - x))
    return ORIG_SF_MAP[closest_k]


def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r with NaN returned for degenerate inputs."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return float('nan')
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def _safe_circular_mean(x_rad: np.ndarray) -> float:
    """Circular mean in radians, returning NaN if input is empty."""
    x_rad = np.asarray(x_rad, dtype=float)
    x_rad = x_rad[np.isfinite(x_rad)]
    if x_rad.size == 0:
        return float('nan')
    return float(np.angle(np.mean(np.exp(1j * x_rad))))


def _safe_circular_corr(alpha_rad: np.ndarray, beta_rad: np.ndarray) -> float:
    """Circular-circular correlation (Jammalamadaka-Sarma style)."""
    alpha_rad = np.asarray(alpha_rad, dtype=float).ravel()
    beta_rad = np.asarray(beta_rad, dtype=float).ravel()
    mask = np.isfinite(alpha_rad) & np.isfinite(beta_rad)
    if np.sum(mask) < 2:
        return float('nan')

    alpha_rad = alpha_rad[mask]
    beta_rad = beta_rad[mask]
    alpha_mean = _safe_circular_mean(alpha_rad)
    beta_mean = _safe_circular_mean(beta_rad)
    if not np.isfinite(alpha_mean) or not np.isfinite(beta_mean):
        return float('nan')

    alpha_centered = np.sin(alpha_rad - alpha_mean)
    beta_centered = np.sin(beta_rad - beta_mean)
    denom = np.sqrt(np.sum(alpha_centered ** 2) * np.sum(beta_centered ** 2))
    if denom < 1e-12:
        return float('nan')
    return float(np.sum(alpha_centered * beta_centered) / denom)


def _orientation_targets_from_deg(angles_deg: np.ndarray) -> np.ndarray:
    """Encode orientation as sin/cos of 2theta to respect 180-degree periodicity."""
    theta_rad = np.deg2rad(np.asarray(angles_deg, dtype=float))
    return np.column_stack((np.sin(2 * theta_rad), np.cos(2 * theta_rad)))


def _orientation_phase_from_targets(targets: np.ndarray) -> np.ndarray:
    """Recover doubled-angle phase from [sin(2theta), cos(2theta)] targets."""
    targets = np.asarray(targets, dtype=float)
    if targets.ndim == 1:
        targets = targets[None, :]
    return np.mod(np.arctan2(targets[:, 0], targets[:, 1]), 2 * np.pi)


def _decode_orientation_from_targets(targets: np.ndarray) -> np.ndarray:
    """Recover orientation in degrees from [sin(2theta), cos(2theta)] targets."""
    phase = _orientation_phase_from_targets(targets)
    return np.mod(np.rad2deg(0.5 * phase), 180.0)


def _fit_ridge_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float) -> np.ndarray:
    """Fit a standardized ridge decoder and predict held-out targets."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = Ridge(alpha=float(alpha))
    model.fit(X_train_scaled, y_train)
    return model.predict(X_test_scaled)


def _run_nested_linear_decoder(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha_grid: List[float],
    outer_cv,
    inner_cv,
    scorer,
) -> tuple[float, float, List[np.ndarray], List[np.ndarray]]:
    """Nested-CV ridge decoder returning pooled out-of-fold score, best alpha, preds, truths."""
    if X.shape[0] < 4:
        return float('nan'), float('nan'), [], []

    best_alpha = float(alpha_grid[0])
    best_score = -np.inf
    for alpha in alpha_grid:
        inner_scores = []
        for train_idx, test_idx in inner_cv.split(X):
            try:
                y_pred = _fit_ridge_predict(X[train_idx], y[train_idx], X[test_idx], alpha)
                score = scorer(y[test_idx], y_pred)
            except Exception:
                score = np.nan
            if np.isfinite(score):
                inner_scores.append(float(score))
        if inner_scores:
            mean_score = float(np.nanmean(inner_scores))
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = float(alpha)

    y_all_pred: List[np.ndarray] = []
    y_all_true: List[np.ndarray] = []
    for train_idx, test_idx in outer_cv.split(X):
        try:
            y_pred = _fit_ridge_predict(X[train_idx], y[train_idx], X[test_idx], best_alpha)
        except Exception:
            y_pred = None
        if y_pred is not None:
            y_all_pred.append(np.asarray(y_pred, dtype=float))
            y_all_true.append(np.asarray(y[test_idx], dtype=float))

    pooled_outer = float('nan')
    if y_all_pred and y_all_true:
        try:
            pooled_outer = float(scorer(np.concatenate(y_all_true, axis=0), np.concatenate(y_all_pred, axis=0)))
        except Exception:
            pooled_outer = float('nan')
    return pooled_outer, best_alpha, y_all_pred, y_all_true


def _sf_regression_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation score for continuous log-SF decoding."""
    return _safe_pearsonr(y_true, y_pred)


def _orientation_regression_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Circular correlation score for continuous orientation decoding."""
    true_phase = _orientation_phase_from_targets(y_true)
    pred_phase = _orientation_phase_from_targets(y_pred)
    return _safe_circular_corr(true_phase, pred_phase)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grating task per-session analysis (supports parallel launcher).",
        add_help=True,
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Launch one subprocess per session and run sessions concurrently.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max concurrent sessions when using --parallel (default: CPU count - 1).",
    )
    parser.add_argument(
        "--session",
        action="append",
        default=None,
        help="Run only this session (relative path). Can be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which sessions/commands would run and exit.",
    )
    # Notebook kernels (ipykernel) inject their own argv entries like:
    #   --f=<path-to-connection-file>
    # Use parse_known_args so running this script in Jupyter/VS Code notebooks
    # doesn't raise "unrecognized arguments".
    args, _unknown = parser.parse_known_args()
    return args


_ARGS = _parse_cli_args()

# If user specifies one or more sessions, restrict SESSION_LIST.
if _ARGS.session:
    SESSION_LIST = list(_ARGS.session)


def _launch_parallel_sessions(sessions: List[str], *, max_workers: int, dry_run: bool) -> None:
    script_path = os.path.abspath(__file__)

    def _run_one(session_rel: str) -> tuple[str, int, str]:
        save_dir = os.path.join(RESULTS_ROOT, session_rel)
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, "parallel_run.log")
        cmd = [sys.executable, script_path, "--session", session_rel]
        if dry_run:
            return (session_rel, 0, log_path)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("COMMAND: " + " ".join(cmd) + "\n\n")
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        return (session_rel, int(p.returncode), log_path)

    if dry_run:
        print("[dry-run] Parallel sessions would run:")
        for s in sessions:
            print(f"  - {s}")
        return

    print("\n" + "=" * 70)
    print(f"PARALLEL SESSION PROCESSING | workers={max_workers} | n_sessions={len(sessions)}")
    print("=" * 70)

    failures: List[tuple[str, int, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_one, s): s for s in sessions}
        for fut in as_completed(futures):
            session_rel, code, log_path = fut.result()
            if code == 0:
                print(f"[ok]   {session_rel}")
            else:
                print(f"[FAIL] {session_rel} (exit={code}) | log: {log_path}")
                failures.append((session_rel, code, log_path))

    if failures:
        print("\n[summary] Failures:")
        for session_rel, code, log_path in failures:
            print(f"  - {session_rel} (exit={code}) | log: {log_path}")
        raise SystemExit(1)


# If --parallel is requested (and we are not already in a per-session run), launch subprocesses and exit.
if _ARGS.parallel and not _ARGS.session:
    cpu = os.cpu_count() or 2
    default_workers = max(1, cpu - 1)
    max_workers = int(_ARGS.max_workers) if (_ARGS.max_workers is not None) else default_workers
    if max_workers < 1:
        max_workers = 1
    _launch_parallel_sessions(SESSION_LIST, max_workers=max_workers, dry_run=bool(_ARGS.dry_run))
    raise SystemExit(0)

# Target SFs and column names for Excel sheet 1
TARGET_SF_VALUES = [0.0000, 7.0184, 14.7623, 29.8831, 59.9448, 119.9789, 240.0023]
TARGET_SF_COLS   = [
    "SF_0.0000",
    "SF_7.0184",
    "SF_14.7623",
    "SF_29.8831",
    "SF_59.9448",
    "SF_119.9789",
    "SF_240.0023",
]

# Collectors for Excel summaries
responsive_summary_rows = []   # for "Responsive_frac_by_SF"
tuning_summary_rows     = []   # for "Tuning_curves"

# ─────────────────────────────────────────────
# GAUSSIAN FUNCTION
# ─────────────────────────────────────────────
def single_gaussian(x, A, mu, sigma, offset):
    """Single Gaussian function to model one peak."""
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + offset


CV_N_FOLDS = 10


def _split_trials_into_cv_folds(
    trial_indices: np.ndarray,
    *,
    n_folds: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Split one condition's trial indices into reproducible held-out folds."""
    trial_indices = np.asarray(trial_indices, dtype=int)
    if trial_indices.size == 0:
        return [np.array([], dtype=int) for _ in range(int(n_folds))]
    perm = rng.permutation(trial_indices)
    return [np.sort(np.asarray(chunk, dtype=int)) for chunk in np.array_split(perm, int(n_folds))]


def _weighted_orientation_mean_deg(angles_deg: np.ndarray, weights: np.ndarray) -> float:
    """Weighted circular mean for orientations in degrees (180-degree periodic)."""
    angles_deg = np.asarray(angles_deg, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(angles_deg) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return float('nan')
    theta = np.deg2rad(2.0 * angles_deg[mask])
    z = np.sum(weights[mask] * np.exp(1j * theta))
    if np.abs(z) < 1e-12:
        return float('nan')
    return float(np.mod(np.rad2deg(0.5 * np.angle(z)), 180.0))


def _safe_sign_from_dot(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Return +1/-1 based on the sign of the dot product."""
    dot = float(np.dot(np.ravel(np.asarray(vec_a, dtype=float)), np.ravel(np.asarray(vec_b, dtype=float))))
    return -1.0 if dot < 0 else 1.0


def _orthogonal_rotation_to_reference(source_axes: np.ndarray, reference_axes: np.ndarray) -> np.ndarray:
    """Best-fit orthogonal rotation that maps source axes onto reference axes."""
    source_axes = np.asarray(source_axes, dtype=float)
    reference_axes = np.asarray(reference_axes, dtype=float)
    if source_axes.ndim != 2 or reference_axes.ndim != 2 or source_axes.shape != reference_axes.shape:
        raise ValueError(f"Axis mismatch for alignment: {source_axes.shape} vs {reference_axes.shape}")
    if source_axes.shape[1] == 0:
        return np.eye(0, dtype=float)
    m = source_axes.T @ reference_axes
    u, _s, vt = np.linalg.svd(m, full_matrices=False)
    rotation = u @ vt
    if rotation.shape[0] == rotation.shape[1] and np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def _align_component_projection_signs(
    projection: np.ndarray,
    fold_axes: np.ndarray,
    reference_axes: np.ndarray,
) -> np.ndarray:
    """Align component signs so fold projections can be combined across CV splits."""
    projection = np.asarray(projection, dtype=float)
    fold_axes = np.asarray(fold_axes, dtype=float)
    reference_axes = np.asarray(reference_axes, dtype=float)
    aligned = np.array(projection, copy=True)
    n_comp = min(aligned.shape[1], fold_axes.shape[1], reference_axes.shape[1])
    for comp_idx in range(n_comp):
        aligned[:, comp_idx] *= _safe_sign_from_dot(fold_axes[:, comp_idx], reference_axes[:, comp_idx])
    return aligned


def _align_target_projection_to_reference(
    projection_target_fold: np.ndarray,
    fold_axes: np.ndarray,
    reference_axes: np.ndarray,
) -> np.ndarray:
    """Align dsPCA target projections to a reference basis across CV folds."""
    projection_target_fold = np.asarray(projection_target_fold, dtype=float)
    fold_axes = np.asarray(fold_axes, dtype=float)
    reference_axes = np.asarray(reference_axes, dtype=float)
    aligned = np.array(projection_target_fold, copy=True)
    n_comp = min(aligned.shape[1], fold_axes.shape[1], reference_axes.shape[1])
    if n_comp >= 1:
        aligned[:, 0] *= _safe_sign_from_dot(fold_axes[:, 0], reference_axes[:, 0])
    if n_comp >= 3:
        rotation = _orthogonal_rotation_to_reference(fold_axes[:, 1:3], reference_axes[:, 1:3])
        aligned[:, 1:3] = aligned[:, 1:3] @ rotation
        for comp_idx in range(3, n_comp):
            aligned[:, comp_idx] *= _safe_sign_from_dot(fold_axes[:, comp_idx], reference_axes[:, comp_idx])
    else:
        for comp_idx in range(1, n_comp):
            aligned[:, comp_idx] *= _safe_sign_from_dot(fold_axes[:, comp_idx], reference_axes[:, comp_idx])
    return aligned


# ─────────────────────────────────────────────
# MAIN LOOP OVER SESSIONS
# ─────────────────────────────────────────────
for date_name_side_depth in SESSION_LIST:
    print(f"\nProcessing session: {date_name_side_depth}")

    # Paths for this session
    data_path    = os.path.join(IMAGE_ROOT, date_name_side_depth)
    suite2p_path = os.path.join(PROC_IMAGE_ROOT, date_name_side_depth, 'suite2p', 'plane0')
    save_path    = os.path.join(RESULTS_ROOT, date_name_side_depth)

    # Create the save directory if it does not exist
    os.makedirs(save_path, exist_ok=True)

    # Read unity behavior data in data_path
    json_files = glob.glob(os.path.join(data_path, '*.json'))
    if len(json_files) == 0:
        print(f"  No JSON file found for {date_name_side_depth}, skipping.")
        continue
    json_file = json_files[0]
    with open(json_file, 'r') as f:
        json_data = [json.loads(line) for line in f][-1]

    # Paths to Bonsai timestamp csv files
    try:
        scanimage_frame_path = glob.glob(os.path.join(data_path, '*Scanimage_Arm1_Frame*.csv'))[0]
        stimulus_time_path   = glob.glob(os.path.join(data_path, '*DataUnity_TrialStart*.csv'))[0]
        unity_frame_path     = glob.glob(os.path.join(data_path, '*DataUnity_Frame*.csv'))[0]
    except IndexError:
        print(f"  Missing one or more Bonsai CSV files for {date_name_side_depth}, skipping.")
        continue

    # Extract True onset timestamps for scanimage_frame
    scanimage_frame_time_all   = np.genfromtxt(scanimage_frame_path, delimiter=',', comments='//', usecols=0)
    scanimage_frame_boolean    = np.genfromtxt(scanimage_frame_path, delimiter=',', comments='//', usecols=1, dtype=str)
    scanimage_frame_onset_bool = np.zeros_like(scanimage_frame_boolean, dtype=int)
    scanimage_frame_onset_bool[1:] = ((scanimage_frame_boolean[1:] == 'True') &
                                      (scanimage_frame_boolean[:-1] == 'False')).astype(int)
    scanimage_frame_onset = scanimage_frame_time_all[scanimage_frame_onset_bool == 1]

    # Extract True onset timestamps for stimulus_time
    stimulus_time_all   = np.genfromtxt(stimulus_time_path, delimiter=',', comments='//', usecols=0)
    stimulus_time_bool  = np.genfromtxt(stimulus_time_path, delimiter=',', comments='//', usecols=1, dtype=str)
    stimulus_onset_bool = np.zeros_like(stimulus_time_bool, dtype=int)
    stimulus_onset_bool[1:] = ((stimulus_time_bool[1:] == 'True') &
                               (stimulus_time_bool[:-1] == 'False')).astype(int)
    stimulus_onset = stimulus_time_all[stimulus_onset_bool == 1]

    # Extract True onset timestamps for unity_frame
    unity_frame_time_all   = np.genfromtxt(unity_frame_path, delimiter=',', comments='//', usecols=0)
    unity_frame_boolean    = np.genfromtxt(unity_frame_path, delimiter=',', comments='//', usecols=1, dtype=str)
    unity_frame_onset_bool = np.zeros_like(unity_frame_boolean, dtype=int)
    unity_frame_onset_bool[1:] = ((unity_frame_boolean[1:] == 'True') &
                                  (unity_frame_boolean[:-1] == 'False')).astype(int)
    unity_frame_onset = unity_frame_time_all[unity_frame_onset_bool == 1]

    # If stimulus_onset is empty, fall back to unity_frame_onset
    if ('stimulus_onset' in locals() or 'stimulus_onset' in globals()):
        if getattr(stimulus_onset, "size", len(stimulus_onset)) == 0 and unity_frame_onset.size > 0:
            print("  stimulus_onset empty — copying unity_frame_onset into stimulus_onset")
            stimulus_onset = unity_frame_onset.copy()
            # remove trailing onsets if the gap before them is > 500 ms
            if stimulus_onset.size >= 2:
                removed = 0
                while stimulus_onset.size >= 2 and (np.diff(stimulus_onset)[-1] > 500):
                    stimulus_onset = stimulus_onset[:-1]
                    removed += 1
                if removed > 0:
                    print(f"  Removed {removed} trailing unity_frame_onset(s) with preceding gap > 500 ms")

    # # plot the distribution of np.diff(unity_frame_onset)
    # diffs = np.diff(unity_frame_onset)
    # plt.figure(figsize=(8, 6))
    # plt.hist(diffs, bins=50, color='blue', alpha=0.7)
    # plt.xlabel('Time Difference (ms)')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of np.diff(unity_frame_onset)')
    # plt.savefig(os.path.join(save_path, 'unity_frame_onset_diff_distribution.svg'))
    # plt.close()

    # Compute differences between consecutive timestamps for scanimage_frame_raw
    diffs = np.diff(scanimage_frame_onset)
    plt.figure(figsize=(8, 6))
    plt.hist(diffs, bins=50, color='blue', alpha=0.7)
    plt.xlabel('Time Difference (ms)')
    plt.ylabel('Frequency')
    plt.title('Distribution of np.diff(scanimage_frame_onset)')
    plt.savefig(os.path.join(save_path, 'scanimage_frame_onset_diff_distribution.svg'))
    plt.close()

    # Load suite2p data
    try:
        iscell = np.load(os.path.join(suite2p_path, 'iscell.npy'))
        spks   = np.load(os.path.join(suite2p_path, 'spks.npy'))
        F      = np.load(os.path.join(suite2p_path, 'F.npy'))
    except FileNotFoundError:
        print(f"  Suite2p outputs not found for {date_name_side_depth}, skipping.")
        continue

    spks_cell          = spks[iscell[:, 0] == 1]
    spks_cell_zscored  = zscore(spks_cell, axis=1)
    F_cell             = F[iscell[:, 0] == 1]
    F_cell_zscored     = zscore(F_cell, axis=1)

    # ===== Filter out neurons with NaN values or zero variance =====
    # Check for NaN in zscored data
    has_nan = np.isnan(spks_cell_zscored).any(axis=1)
    
    # Check for zero variance (or very small variance) in original spike data
    neuron_variance = np.var(spks_cell, axis=1)
    has_zero_var = neuron_variance < 1e-10
    
    # Combine filters: keep only neurons without NaN and with non-zero variance
    valid_neurons = ~(has_nan | has_zero_var)
    n_removed = np.sum(~valid_neurons)
    
    if n_removed > 0:
        print(f"  Removing {n_removed} neurons (NaN: {np.sum(has_nan)}, Zero var: {np.sum(has_zero_var)})")
        spks_cell = spks_cell[valid_neurons]
        spks_cell_zscored = spks_cell_zscored[valid_neurons]
        F_cell = F_cell[valid_neurons]
        F_cell_zscored = F_cell_zscored[valid_neurons]

    # Create a neural activity tensor for the grating stimulus
    mean_dt = np.mean(np.diff(scanimage_frame_onset))  # ms, assuming timestamps in ms

    negative_window = int(round(50 / mean_dt))    # ~ -50 ms
    positive_window = int(round(250 / mean_dt))   # ~ +250 ms
    window_length   = negative_window + positive_window + 1

    n_trials_raw = len(stimulus_onset)
    n_neurons    = spks_cell_zscored.shape[0]
    n_frames     = spks_cell_zscored.shape[1]

    # Build trial slices and remember which trials are actually kept
    trial_slices       = []
    kept_trial_indices = []  # indices into original trial list

    for trial_i, onset in enumerate(stimulus_onset):
        trial_idx = np.argmin(np.abs(scanimage_frame_onset - onset))
        start_idx = trial_idx - negative_window
        end_idx   = trial_idx + positive_window + 1

        if start_idx < 0 or end_idx > n_frames:
            continue

        trial_slice = spks_cell_zscored[:, start_idx:end_idx]
        trial_slices.append(trial_slice)
        kept_trial_indices.append(trial_i)

    if len(trial_slices) == 0:
        print(f"  No valid trial windows for {date_name_side_depth}, skipping.")
        continue

    kept_trial_indices = np.array(kept_trial_indices, dtype=int)
    spks_tensor = np.stack(trial_slices, axis=-1)   # shape: (neurons, time, n_kept_trials)
    n_trials = spks_tensor.shape[2]

    # Save basic counts using kept trials
    np.save(os.path.join(save_path, 'n_trials.npy'),  n_trials)
    np.save(os.path.join(save_path, 'n_neurons.npy'), n_neurons)
    np.save(os.path.join(save_path, 'n_frames.npy'),  n_frames)

    print("  spks_tensor shape:", spks_tensor.shape)

    # Plot the heatmap and mean activity of the neural activity tensor
    time_axis = (np.arange(window_length) - negative_window) * (mean_dt)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    im = axes[0].imshow(np.mean(spks_tensor, axis=2), aspect='auto', cmap='viridis',
                        origin='lower', extent=[time_axis[0], time_axis[-1], 0, n_neurons])
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Neuron Index')
    axes[0].set_title('Heatmap of Averaged Neural Activity')
    axes[0].axvline(x=0,   color='red',   linestyle='--', linewidth=2, label='Stimulus Onset')
    axes[0].axvline(x=250, color='green', linestyle='--', linewidth=2, label='Stimulus Offset')
    axes[0].legend(bbox_to_anchor=(0.5, 1.25), loc='upper center', ncol=3)
    fig.colorbar(im, ax=axes[0], label='Average Activity')

    mean_activity_trace = np.mean(spks_tensor, axis=(0, 2))
    axes[1].plot(time_axis, mean_activity_trace, color='blue', linewidth=2, label='Mean Activity')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Mean Activity')
    axes[1].set_title('Mean Neural Activity (Averaged Over Neurons)')
    axes[1].axvline(x=0,   color='red',   linestyle='--', linewidth=2, label='Stimulus Onset')
    axes[1].axvline(x=250, color='green', linestyle='--', linewidth=2, label='Stimulus Offset')
    axes[1].legend(bbox_to_anchor=(0.5, 1.25), loc='upper center', ncol=3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'neural_activity_heatmap_and_mean_activity.svg'), format='svg')
    plt.close()

    # Identify visually responsive neurons for each spatial frequency.
    orientation_history = np.array(json_data['orientation_history'])
    spatialfreq_history = np.array(json_data['spatialfreq_history'])

    unique_orientations = np.unique(orientation_history)
    unique_spatialfreqs = np.unique(spatialfreq_history)

    n_sf = len(unique_spatialfreqs)
    responsive_neurons_by_sf   = np.zeros((n_sf, n_neurons), dtype=bool)
    responsive_fraction_by_sf  = np.zeros(n_sf)
    response_idx               = np.where((time_axis >= 0) & (time_axis <= 250))[0]

    # Map original trial indices → kept-trial indices for spks_tensor
    index_map = {orig_idx: kept_idx for kept_idx, orig_idx in enumerate(kept_trial_indices)}

    def map_to_kept(idxs):
        """Map array of original trial indices to kept-trial indices (drop those that were discarded)."""
        return np.array([index_map[i] for i in idxs if i in index_map], dtype=int)

    # Baseline trials (original indices -> kept indices)
    baseline_trial_inds_all = np.where(spatialfreq_history == unique_spatialfreqs[0])[0]
    baseline_trial_inds     = map_to_kept(baseline_trial_inds_all)

    if baseline_trial_inds.size == 0:
        print(f"  No valid baseline trials (after slicing) for {date_name_side_depth}, skipping session.")
        continue

    for sf_idx, sf in enumerate(unique_spatialfreqs[1:], start=1):
        trial_inds_all = np.where(spatialfreq_history == sf)[0]
        trial_inds     = map_to_kept(trial_inds_all)

        if trial_inds.size == 0:
            print(f"  No valid trials for spatial frequency {sf} (after slicing) in {date_name_side_depth}")
            continue

        baseline_activity_sf = np.mean(
            spks_tensor[:, response_idx, :][:, :, baseline_trial_inds],
            axis=1
        )
        response_activity_sf = np.mean(
            spks_tensor[:, response_idx, :][:, :, trial_inds],
            axis=1
        )

        baseline_mean_sf = np.nanmean(baseline_activity_sf, axis=1)
        response_mean_sf = np.nanmean(response_activity_sf, axis=1)

        # Robust paired t-test: handle unequal trial counts + NaNs
        p_values_sf = np.ones(n_neurons)
        for i in range(n_neurons):
            base_vals = baseline_activity_sf[i, :]
            resp_vals = response_activity_sf[i, :]

            # drop NaNs
            base_vals = base_vals[~np.isnan(base_vals)]
            resp_vals = resp_vals[~np.isnan(resp_vals)]

            # make them the same length (paired)
            n_pair = min(base_vals.size, resp_vals.size)
            if n_pair < 2:
                p_values_sf[i] = 1.0  # not enough data, mark non-significant
                continue

            stat, p = ttest_rel(resp_vals[:n_pair], base_vals[:n_pair])
            p_values_sf[i] = p

        responsive_neurons = np.where((p_values_sf < 0.05) & (response_mean_sf > baseline_mean_sf))[0]
        responsive_neurons_by_sf[sf_idx, responsive_neurons] = True
        responsive_fraction_by_sf[sf_idx] = len(responsive_neurons) / n_neurons
        print(f"  SF {sf} | Fraction responsive: {responsive_fraction_by_sf[sf_idx]}")

    np.save(os.path.join(save_path, 'responsive_neurons_by_sf.npy'),  responsive_neurons_by_sf)
    np.save(os.path.join(save_path, 'responsive_fraction_by_sf.npy'), responsive_fraction_by_sf)

    # Plot fraction responsive vs SF
    plt.figure(figsize=(6, 4))
    plt.plot(unique_spatialfreqs[1:], responsive_fraction_by_sf[1:], marker='o',
             color='blue', linestyle='-', alpha=0.7)
    plt.xscale('log')
    plt.xlabel('Spatial Frequency (log scale)')
    plt.ylabel('Fraction of Responsive Neurons')
    plt.title('Fraction of Visually Responsive Neurons for Each Spatial Frequency')
    plt.xticks(unique_spatialfreqs[1:], labels=[0.02, 0.04, 0.08, 0.16, 0.32, 0.64], rotation=45)
    plt.axhline(y=0.025, color='red', linestyle='--', label='Chance Level')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'fraction_responsive_neurons_by_sf_logscale.svg'), format='svg')
    plt.close()

    # Save row for Excel Sheet 1 (Responsive_frac_by_SF)
    resp_row = {"Session": date_name_side_depth}
    for col_name, sf_target in zip(TARGET_SF_COLS, TARGET_SF_VALUES):
        idx_match = np.where(np.isclose(unique_spatialfreqs, sf_target, rtol=1e-4, atol=1e-4))[0]
        if idx_match.size > 0:
            resp_row[col_name] = float(responsive_fraction_by_sf[idx_match[0]])
        else:
            resp_row[col_name] = np.nan
    responsive_summary_rows.append(resp_row)

    # ------------------------------------------------------------------
    # Downstream tuning/OSI analyses per spatial frequency
    # (each SF uses its own responsive neuron mask)
    # ------------------------------------------------------------------

    # Per-trial response (mean over response window) for all neurons
    response_per_trial = np.nanmean(spks_tensor[:, response_idx, :], axis=1)  # (n_neurons, n_trials)
    baseline_mean = np.nanmean(response_per_trial[:, baseline_trial_inds], axis=1, keepdims=True)  # (n_neurons, 1)
    delta_per_trial = response_per_trial - baseline_mean

    n_ori = len(unique_orientations)
    orientation_tuning = np.full((n_neurons, n_ori, n_sf), np.nan, dtype=float)
    spatialfreq_tuning = np.full((n_neurons, n_sf, n_ori), np.nan, dtype=float)

    # Compute tuning curves, but only populate neurons responsive at each SF
    for k, sf in enumerate(unique_spatialfreqs):
        responsive_mask_k = responsive_neurons_by_sf[k]
        if not np.any(responsive_mask_k):
            continue

        for j, ori in enumerate(unique_orientations):
            trial_idx_all = np.where((orientation_history == ori) & (spatialfreq_history == sf))[0]
            trial_idx = map_to_kept(trial_idx_all)
            if trial_idx.size == 0:
                continue

            vals = delta_per_trial[responsive_mask_k][:, trial_idx]
            orientation_tuning[responsive_mask_k, j, k] = np.nanmean(vals, axis=1)

    for j, sf in enumerate(unique_spatialfreqs):
        responsive_mask_j = responsive_neurons_by_sf[j]
        if not np.any(responsive_mask_j):
            continue

        for k, ori in enumerate(unique_orientations):
            trial_idx_all = np.where((spatialfreq_history == sf) & (orientation_history == ori))[0]
            trial_idx = map_to_kept(trial_idx_all)
            if trial_idx.size == 0:
                continue

            vals = delta_per_trial[responsive_mask_j][:, trial_idx]
            spatialfreq_tuning[responsive_mask_j, j, k] = np.nanmean(vals, axis=1)

    np.save(os.path.join(save_path, 'orientation_tuning.npy'), orientation_tuning)
    np.save(os.path.join(save_path, 'spatialfreq_tuning.npy'), spatialfreq_tuning)

    # Visualization for orientation tuning curves for each spatial frequency (responsive neurons for that SF)
    fig, axes = plt.subplots(nrows=1, ncols=n_sf, figsize=(4 * n_sf, 5))
    for k in range(n_sf):
        ax = axes[k] if n_sf > 1 else axes
        mask_k = responsive_neurons_by_sf[k]
        tuning_img = orientation_tuning[mask_k, :, k]
        if tuning_img.shape[0] == 0:
            tuning_img = np.full((1, n_ori), np.nan)
        im = ax.imshow(tuning_img, aspect='auto', cmap='viridis', origin='lower',
                       extent=[unique_orientations[0], unique_orientations[-1], 0, tuning_img.shape[0]])
        ax.set_xlabel('Orientation (degrees)')
        ax.set_title(f'SF: {unique_spatialfreqs[k]}')
        fig.colorbar(im, ax=ax, label='ΔActivity (Response - Baseline)')
    plt.suptitle('Orientation Tuning Curves (ΔActivity) for each Spatial Frequency')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_path, 'orientation_tuning_curves.svg'), format='svg')
    plt.close()

    # Visualization for spatial frequency tuning curves for each orientation
    # (show neurons responsive to any non-baseline SF; non-responsive entries are NaN)
    responsive_any = responsive_neurons_by_sf[1:].any(axis=0) if n_sf > 1 else responsive_neurons_by_sf.any(axis=0)
    n_resp_any = int(np.sum(responsive_any))
    fig, axes = plt.subplots(nrows=1, ncols=n_ori, figsize=(4 * n_ori, 5))
    for k in range(n_ori):
        ax = axes[k] if n_ori > 1 else axes
        img = spatialfreq_tuning[responsive_any, :, k]
        if img.shape[0] == 0:
            img = np.full((1, n_sf), np.nan)
        im = ax.imshow(img, aspect='auto', cmap='viridis', origin='lower',
                       extent=[unique_spatialfreqs[0], unique_spatialfreqs[-1], 0, img.shape[0]])
        ax.set_xlabel('Spatial Frequency')
        ax.set_title(f'Orientation: {unique_orientations[k]}°')
        fig.colorbar(im, ax=ax, label='ΔActivity (Response - Baseline)')
    plt.suptitle('Spatial Frequency Tuning Curves (ΔActivity) for each Orientation')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_path, 'spatialfreq_tuning_curves.svg'), format='svg')
    plt.close()

    # Add rows for Excel Sheet 2 (Tuning_curves)
    for j, ori in enumerate(unique_orientations):
        for k, sf in enumerate(unique_spatialfreqs):
            mean_act = _safe_nanmean(orientation_tuning[:, j, k])
            idx_match = np.where(np.isclose(unique_spatialfreqs, sf, rtol=1e-4, atol=1e-4))[0]
            if idx_match.size > 0:
                frac_resp = float(responsive_fraction_by_sf[idx_match[0]])
            else:
                frac_resp = np.nan
            tuning_summary_rows.append({
                "Session":      date_name_side_depth,
                "Orientation":  float(ori),
                "SpatialFreq":  float(sf),
                "MeanActivity": mean_act,
                "Fractionresponsive": frac_resp
            })

    # OSI via vector sum method (compute only for neurons responsive at each SF)
    osi_spf = np.full((n_neurons, n_sf), np.nan, dtype=float)
    theta_rad = np.deg2rad(unique_orientations)
    for k in range(n_sf):
        mask_k = responsive_neurons_by_sf[k]
        if not np.any(mask_k):
            continue
        responses = np.maximum(orientation_tuning[mask_k, :, k], 0)
        denom = np.nansum(responses, axis=1)
        valid = denom > 0
        vector_sum = np.nansum(responses * np.exp(1j * 2 * theta_rad), axis=1)
        out = np.full(responses.shape[0], np.nan, dtype=float)
        out[valid] = np.abs(vector_sum[valid]) / denom[valid]
        osi_spf[mask_k, k] = out
    np.save(os.path.join(save_path, 'osi_spf.npy'), osi_spf)

    # Histogram of OSI vector sum (per SF responsive set)
    fig, axes = plt.subplots(1, n_sf, figsize=(4 * n_sf, 5))
    for k in range(n_sf):
        ax = axes[k] if n_sf > 1 else axes
        mask_k = responsive_neurons_by_sf[k]
        valid_osi = osi_spf[mask_k, k]
        valid_osi = valid_osi[~np.isnan(valid_osi)]
        ax.hist(valid_osi, bins=20, color='blue', alpha=0.7)
        ax.set_xlabel('OSI')
        ax.set_title(f'SF: {unique_spatialfreqs[k]}')
        mean_val = _safe_nanmean(osi_spf[mask_k, k])
        ax.text(0.05, 0.95, f'Mean OSI: {mean_val:.2f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top', color='black')
    fig.suptitle('Distribution of OSI Values for each Spatial Frequency')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_path, 'osi_vector_sum_distributions.svg'), format='svg')
    plt.close()

    # OSI via difference method (discrete; per SF responsive set)
    osi_diff = np.full((n_neurons, n_sf), np.nan, dtype=float)
    for k in range(n_sf):
        mask_k = responsive_neurons_by_sf[k]
        if not np.any(mask_k):
            continue

        tuning_k = orientation_tuning[mask_k, :, k]
        safe = np.where(np.isnan(tuning_k), -np.inf, tuning_k)
        all_nan = np.all(np.isnan(tuning_k), axis=1)
        pref_idx = np.argmax(safe, axis=1)

        preferred_ori = unique_orientations[pref_idx]
        orth_ori = (preferred_ori + 90) % 180
        R_pref = tuning_k[np.arange(tuning_k.shape[0]), pref_idx]
        orth_idx = np.array([np.argmin(np.abs(unique_orientations - o)) for o in orth_ori])
        R_orth = tuning_k[np.arange(tuning_k.shape[0]), orth_idx]

        out = np.full(tuning_k.shape[0], np.nan, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = (R_pref - R_orth) / (R_pref + R_orth)
        out[all_nan] = np.nan
        osi_diff[mask_k, k] = out
    np.save(os.path.join(save_path, 'osi_diff.npy'), osi_diff)

    fig, axes = plt.subplots(1, n_sf, figsize=(4 * n_sf, 5))
    for k in range(n_sf):
        ax = axes[k] if n_sf > 1 else axes
        mask_k = responsive_neurons_by_sf[k]
        valid_osi = osi_diff[mask_k, k]
        valid_osi = valid_osi[~np.isnan(valid_osi)]
        ax.hist(valid_osi, bins=20, color='blue', alpha=0.7)
        ax.set_xlabel('OSI')
        ax.set_title(f'SF: {unique_spatialfreqs[k]}')
        mean_val = _safe_nanmean(osi_diff[mask_k, k])
        ax.text(0.05, 0.95, f'Mean OSI: {mean_val:.2f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top', color='black')
    fig.suptitle('Distribution of OSI (Difference Method) for each Spatial Frequency')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_path, 'osi_difference_distributions.svg'), format='svg')
    plt.close()

    # OSI via difference method with Gaussian fit (fit only responsive neurons per SF)
    fit_tuning_curves = np.full_like(orientation_tuning, np.nan, dtype=float)
    preferred_orientations = np.full((n_neurons, n_sf), np.nan, dtype=float)
    osi_diff_gauss = np.full((n_neurons, n_sf), np.nan, dtype=float)

    x = unique_orientations.astype(float)
    for k in range(n_sf):
        mask_k = responsive_neurons_by_sf[k]
        if not np.any(mask_k):
            continue
        neuron_indices = np.where(mask_k)[0]
        for i in neuron_indices:
            y = orientation_tuning[i, :, k]
            if np.all(np.isnan(y)):
                continue

            init_A      = np.nanmax(y) - np.nanmin(y)
            init_mu     = x[np.nanargmax(y)]
            init_sigma  = 20.0
            init_offset = np.nanmin(y)
            p0 = [init_A, init_mu, init_sigma, init_offset]

            popt = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    popt, _ = curve_fit(single_gaussian, x, y, p0=p0)
                mu_fit = popt[1] % 180
                fit_tuning_curves[i, :, k] = single_gaussian(x, *popt)
            except Exception:
                pref_idx = np.nanargmax(y)
                mu_fit   = x[pref_idx]
                fit_tuning_curves[i, :, k] = y

            preferred_orientations[i, k] = mu_fit
            orth_ori = (mu_fit + 90) % 180

            if popt is not None:
                R_pref = single_gaussian(mu_fit, *popt)
                orth_idx = np.argmin(np.abs(x - orth_ori))
                R_orth = single_gaussian(x[orth_idx], *popt)
            else:
                pref_idx = np.nanargmax(y)
                R_pref   = y[pref_idx]
                orth_idx = np.argmin(np.abs(x - orth_ori))
                R_orth   = y[orth_idx]

            if (R_pref + R_orth) > 0:
                osi_diff_gauss[i, k] = (R_pref - R_orth) / (R_pref + R_orth)

    np.save(os.path.join(save_path, 'fit_tuning_curves.npy'),      fit_tuning_curves)
    np.save(os.path.join(save_path, 'preferred_orientations.npy'), preferred_orientations)

    fig, axes = plt.subplots(1, n_sf, figsize=(4 * n_sf, 5))
    for k in range(n_sf):
        ax = axes[k] if n_sf > 1 else axes
        mask_k = responsive_neurons_by_sf[k]
        valid_osi = osi_diff_gauss[mask_k, k]
        valid_osi = valid_osi[~np.isnan(valid_osi)]
        ax.hist(valid_osi, bins=20, color='blue', alpha=0.7)
        ax.set_xlabel('OSI')
        ax.set_title(f'SF: {unique_spatialfreqs[k]}')
        mean_val = _safe_nanmean(osi_diff_gauss[mask_k, k])
        ax.text(0.05, 0.95, f'Mean OSI: {mean_val:.2f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top', color='black')
    fig.suptitle('Distribution of OSI (Difference Method with Single Gaussian Fit)\nfor each Spatial Frequency')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_path, 'osi_gaussian_fit_distributions.svg'), format='svg')
    plt.close()

    # Mean aligned orientation tuning curves ± SEM using Gaussian fits
    center_idx = 2
    aligned_tuning = np.full_like(fit_tuning_curves, np.nan, dtype=float)
    for k in range(n_sf):
        mask_k = responsive_neurons_by_sf[k]
        if not np.any(mask_k):
            continue
        neuron_indices = np.where(mask_k)[0]
        for i in neuron_indices:
            curve = fit_tuning_curves[i, :, k]
            if np.all(np.isnan(curve)):
                continue
            pref_ori = preferred_orientations[i, k]
            if np.isnan(pref_ori):
                continue
            pref_idx = np.argmin(np.abs(unique_orientations - pref_ori))
            shift    = center_idx - pref_idx
            aligned_tuning[i, :, k] = np.roll(curve, shift)

    xticks      = [-60, -30, 0, 30, 60, 90]
    xticklabels = ['-60', '-30', 'Pref', '+30', '+60', '+90']

    fig, axes = plt.subplots(1, n_sf, figsize=(4 * n_sf, 5), sharey=True)
    for k in range(n_sf):
        ax = axes[k] if n_sf > 1 else axes
        tuning_k = aligned_tuning[:, :, k]
        mean_curve = _safe_nanmean_axis0(tuning_k)
        n_valid    = np.sum(~np.isnan(tuning_k), axis=0)
        sem_curve = np.full_like(mean_curve, np.nan)
        valid_cols = n_valid > 0
        if np.any(valid_cols):
            std_vals = np.nanstd(tuning_k[:, valid_cols], axis=0)
            sem_curve[valid_cols] = std_vals / np.sqrt(n_valid[valid_cols])

        ax.errorbar(xticks, mean_curve, yerr=sem_curve, fmt='-o', capsize=5,
                    color='blue', ecolor='gray')
        ax.set_xlabel('Relative Orientation (deg)')
        ax.set_title(f'SF: {unique_spatialfreqs[k]}')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

    axes[0].set_ylabel('ΔActivity (Response - Baseline)')
    plt.suptitle('Mean Aligned Orientation Tuning Curves ± SEM\n(Aligned to peak = "pref")')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_path, 'aligned_tuning_gaussian_fit.svg'), format='svg')
    plt.close()

    # Mean aligned orientation tuning curves ± SEM using original data
    aligned_tuning_original = np.full_like(orientation_tuning, np.nan, dtype=float)
    for k in range(n_sf):
        mask_k = responsive_neurons_by_sf[k]
        if not np.any(mask_k):
            continue
        neuron_indices = np.where(mask_k)[0]
        for i in neuron_indices:
            curve = orientation_tuning[i, :, k]
            if np.all(np.isnan(curve)):
                continue
            pref_ori = preferred_orientations[i, k]
            if np.isnan(pref_ori):
                continue
            pref_idx = np.argmin(np.abs(unique_orientations - pref_ori))
            shift    = center_idx - pref_idx
            aligned_tuning_original[i, :, k] = np.roll(curve, shift)

    fig, axes = plt.subplots(1, n_sf, figsize=(4 * n_sf, 5), sharey=True)
    for k in range(n_sf):
        ax = axes[k] if n_sf > 1 else axes
        tuning_k = aligned_tuning_original[:, :, k]
        mean_curve = _safe_nanmean_axis0(tuning_k)
        n_valid    = np.sum(~np.isnan(tuning_k), axis=0)
        sem_curve = np.full_like(mean_curve, np.nan)
        valid_cols = n_valid > 0
        if np.any(valid_cols):
            std_vals = np.nanstd(tuning_k[:, valid_cols], axis=0)
            sem_curve[valid_cols] = std_vals / np.sqrt(n_valid[valid_cols])

        ax.errorbar(xticks, mean_curve, yerr=sem_curve, fmt='-o', capsize=5,
                    color='blue', ecolor='gray')
        ax.set_xlabel('Relative Orientation (deg)')
        ax.set_title(f'SF: {unique_spatialfreqs[k]}')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

    axes[0].set_ylabel('ΔActivity (Response - Baseline)')
    plt.suptitle('Mean Aligned Orientation Tuning Curves ± SEM\n(Aligned to peak = "pref", Original Data)')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_path, 'aligned_tuning_original_data.svg'), format='svg')
    plt.close()

    # ─────────────────────────────────────────────
    # 10-FOLD CROSS-VALIDATION ALIGNED TUNING CURVES
    # - Split trials independently within each (orientation, SF) condition
    # - Each trial appears in the held-out set exactly once
    # - Aggregate held-out responses across folds for downstream plotting
    # ─────────────────────────────────────────────
    rng_cv_tuning = np.random.default_rng(int(42) ^ int(zlib.crc32(str(save_path).encode('utf-8'))))
    min_trials_tuning = np.inf
    tuning_trial_indices = {}
    for k, sf in enumerate(unique_spatialfreqs):
        for j, ori in enumerate(unique_orientations):
            trial_idx_all = np.where((orientation_history == ori) & (spatialfreq_history == sf))[0]
            trial_idx = map_to_kept(trial_idx_all)
            tuning_trial_indices[(j, k)] = trial_idx
            if trial_idx.size > 0:
                min_trials_tuning = min(min_trials_tuning, trial_idx.size)

    if (not np.isfinite(min_trials_tuning)) or (int(min_trials_tuning) < CV_N_FOLDS):
        found_trials = int(min_trials_tuning) if np.isfinite(min_trials_tuning) else 0
        print(f"  [CV10] Skipped aligned tuning CV: need >= {CV_N_FOLDS} trials per (ori, SF), found {found_trials}")
    else:
        cv_train_means = np.full((CV_N_FOLDS, n_neurons, n_ori, n_sf), np.nan, dtype=float)
        cv_test_means = np.full((CV_N_FOLDS, n_neurons, n_ori, n_sf), np.nan, dtype=float)
        cv_train_counts = np.zeros((CV_N_FOLDS, n_ori, n_sf), dtype=int)
        cv_test_counts = np.zeros((CV_N_FOLDS, n_ori, n_sf), dtype=int)

        for k, sf in enumerate(unique_spatialfreqs):
            mask_k = responsive_neurons_by_sf[k]
            if not np.any(mask_k):
                continue
            for j, ori in enumerate(unique_orientations):
                trial_idx = tuning_trial_indices[(j, k)]
                if trial_idx.size < CV_N_FOLDS:
                    continue
                fold_splits = _split_trials_into_cv_folds(trial_idx, n_folds=CV_N_FOLDS, rng=rng_cv_tuning)
                for fold_idx, test_idx in enumerate(fold_splits):
                    if test_idx.size == 0:
                        continue
                    train_idx = np.setdiff1d(trial_idx, test_idx, assume_unique=False)
                    if train_idx.size == 0:
                        continue
                    cv_train_counts[fold_idx, j, k] = int(train_idx.size)
                    cv_test_counts[fold_idx, j, k] = int(test_idx.size)
                    cv_train_means[fold_idx, mask_k, j, k] = np.nanmean(delta_per_trial[mask_k][:, train_idx], axis=1)
                    cv_test_means[fold_idx, mask_k, j, k] = np.nanmean(delta_per_trial[mask_k][:, test_idx], axis=1)

        train_weighted_sum = np.nansum(cv_train_means * cv_train_counts[:, None, :, :], axis=0)
        test_weighted_sum = np.nansum(cv_test_means * cv_test_counts[:, None, :, :], axis=0)
        train_weight_total = np.sum(cv_train_counts, axis=0)
        test_weight_total = np.sum(cv_test_counts, axis=0)
        orientation_tuning_cv_fold1 = np.divide(
            train_weighted_sum,
            train_weight_total[None, :, :],
            out=np.full((n_neurons, n_ori, n_sf), np.nan, dtype=float),
            where=train_weight_total[None, :, :] > 0,
        )
        orientation_tuning_cv_fold2 = np.divide(
            test_weighted_sum,
            test_weight_total[None, :, :],
            out=np.full((n_neurons, n_ori, n_sf), np.nan, dtype=float),
            where=test_weight_total[None, :, :] > 0,
        )

        np.save(os.path.join(save_path, 'orientation_tuning_cv_fold1.npy'), orientation_tuning_cv_fold1)
        np.save(os.path.join(save_path, 'orientation_tuning_cv_fold2.npy'), orientation_tuning_cv_fold2)
        np.savez(
            os.path.join(save_path, 'orientation_tuning_cv10.npz'),
            train_means_by_fold=cv_train_means,
            test_means_by_fold=cv_test_means,
            train_counts_by_fold=cv_train_counts,
            test_counts_by_fold=cv_test_counts,
            n_folds=int(CV_N_FOLDS),
            cv_note='10-fold CV: train on 9 folds, aggregate held-out condition means across all folds',
        )

        fit_tuning_curves_cv_folds = np.full_like(cv_train_means, np.nan, dtype=float)
        preferred_orientations_cv_folds = np.full((CV_N_FOLDS, n_neurons, n_sf), np.nan, dtype=float)
        aligned_fit_sum = np.zeros((n_neurons, n_ori, n_sf), dtype=float)
        aligned_fit_weight = np.zeros((n_neurons, n_ori, n_sf), dtype=float)
        aligned_test_sum = np.zeros((n_neurons, n_ori, n_sf), dtype=float)
        aligned_test_weight = np.zeros((n_neurons, n_ori, n_sf), dtype=float)

        for fold_idx in range(CV_N_FOLDS):
            for k in range(n_sf):
                mask_k = responsive_neurons_by_sf[k]
                if not np.any(mask_k):
                    continue
                neuron_indices = np.where(mask_k)[0]
                for i in neuron_indices:
                    y = cv_train_means[fold_idx, i, :, k]
                    if np.all(np.isnan(y)):
                        continue
                    init_A = np.nanmax(y) - np.nanmin(y)
                    init_mu = x[np.nanargmax(y)]
                    init_sigma = 20.0
                    init_offset = np.nanmin(y)
                    p0 = [init_A, init_mu, init_sigma, init_offset]
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            popt, _ = curve_fit(single_gaussian, x, y, p0=p0)
                        mu_fit = popt[1] % 180
                        fit_curve = single_gaussian(x, *popt)
                    except Exception:
                        pref_idx = np.nanargmax(y)
                        mu_fit = x[pref_idx]
                        fit_curve = y
                    fit_tuning_curves_cv_folds[fold_idx, i, :, k] = fit_curve
                    preferred_orientations_cv_folds[fold_idx, i, k] = mu_fit

            for k in range(n_sf):
                mask_k = responsive_neurons_by_sf[k]
                if not np.any(mask_k):
                    continue
                neuron_indices = np.where(mask_k)[0]
                for i in neuron_indices:
                    pref_ori = preferred_orientations_cv_folds[fold_idx, i, k]
                    if np.isnan(pref_ori):
                        continue
                    pref_idx = np.argmin(np.abs(unique_orientations - pref_ori))
                    shift = center_idx - pref_idx

                    fit_curve = fit_tuning_curves_cv_folds[fold_idx, i, :, k]
                    test_curve = cv_test_means[fold_idx, i, :, k]
                    train_counts_fold = np.roll(cv_train_counts[fold_idx, :, k], shift)
                    test_counts_fold = np.roll(cv_test_counts[fold_idx, :, k], shift)

                    rolled_fit = np.roll(fit_curve, shift)
                    valid_fit = np.isfinite(rolled_fit) & (train_counts_fold > 0)
                    aligned_fit_sum[i, valid_fit, k] += rolled_fit[valid_fit] * train_counts_fold[valid_fit]
                    aligned_fit_weight[i, valid_fit, k] += train_counts_fold[valid_fit]

                    rolled_test = np.roll(test_curve, shift)
                    valid_test = np.isfinite(rolled_test) & (test_counts_fold > 0)
                    aligned_test_sum[i, valid_test, k] += rolled_test[valid_test] * test_counts_fold[valid_test]
                    aligned_test_weight[i, valid_test, k] += test_counts_fold[valid_test]

        aligned_tuning_cv_fold1 = np.divide(
            aligned_fit_sum,
            aligned_fit_weight,
            out=np.full_like(aligned_fit_sum, np.nan, dtype=float),
            where=aligned_fit_weight > 0,
        )
        aligned_tuning_cv_fold2 = np.divide(
            aligned_test_sum,
            aligned_test_weight,
            out=np.full_like(aligned_test_sum, np.nan, dtype=float),
            where=aligned_test_weight > 0,
        )

        preferred_orientations_cv = np.full((n_neurons, n_sf), np.nan, dtype=float)
        fold_sf_weights = np.sum(cv_train_counts, axis=1)
        for k in range(n_sf):
            mask_k = responsive_neurons_by_sf[k]
            if not np.any(mask_k):
                continue
            for i in np.where(mask_k)[0]:
                preferred_orientations_cv[i, k] = _weighted_orientation_mean_deg(
                    preferred_orientations_cv_folds[:, i, k],
                    fold_sf_weights[:, k],
                )

        np.save(os.path.join(save_path, 'preferred_orientations_cv.npy'), preferred_orientations_cv)

        fig, axes = plt.subplots(1, n_sf, figsize=(4 * n_sf, 5), sharey=True)
        for k in range(n_sf):
            ax = axes[k] if n_sf > 1 else axes
            tuning_k = aligned_tuning_cv_fold1[:, :, k]
            mean_curve = _safe_nanmean_axis0(tuning_k)
            sem_curve = _safe_sem(tuning_k, axis=0)
            ax.errorbar(xticks, mean_curve, yerr=sem_curve, fmt='-o', capsize=5,
                        color='blue', ecolor='gray')
            ax.set_xlabel('Relative Orientation (deg)')
            ax.set_title(f'SF: {unique_spatialfreqs[k]}')
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)

        axes[0].set_ylabel('ΔActivity (Response - Baseline)')
        plt.suptitle('Mean Aligned Orientation Tuning Curves ± SEM\n(10-fold CV: Gaussian Fit on Training Folds)')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(save_path, 'aligned_tuning_cv_gaussian_fit.svg'), format='svg')
        plt.close()

        fig, axes = plt.subplots(1, n_sf, figsize=(4 * n_sf, 5), sharey=True)
        for k in range(n_sf):
            ax = axes[k] if n_sf > 1 else axes
            tuning_k = aligned_tuning_cv_fold2[:, :, k]
            mean_curve = _safe_nanmean_axis0(tuning_k)
            sem_curve = _safe_sem(tuning_k, axis=0)
            ax.errorbar(xticks, mean_curve, yerr=sem_curve, fmt='-o', capsize=5,
                        color='orange', ecolor='gray')
            ax.set_xlabel('Relative Orientation (deg)')
            ax.set_title(f'SF: {unique_spatialfreqs[k]}')
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)

        axes[0].set_ylabel('ΔActivity (Response - Baseline)')
        plt.suptitle('Mean Aligned Orientation Tuning Curves ± SEM\n(10-fold CV: Concatenated Held-out Data)')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(save_path, 'aligned_tuning_cv_heldout_data.svg'), format='svg')
        plt.close()

    # ------------------------------------------------------------------
    # Decoding analyses across neuron counts (baseline-subtracted 0-250ms mean)
    # - Orientation decoding per SF and SF decoding per orientation
    # - Evaluate for multiple neuron counts: 10..80
    # - For each neuron count: build subsets until all neurons are used at least once
    # - Cross-validated, regularized multinomial logistic regression
    # ------------------------------------------------------------------

    DECODING_CELL_COUNTS = [10, 20, 30, 40, 50, 60, 70, 80]
    DECODING_DEFAULT_N = 80
    DECODING_BASE_SEED = 42

    # Hyperparameter grid for regularization strength
    C_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    RIDGE_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    # Relative bins: [-60, -30, Correct(0), +30, +60, +90]
    rel_bins = np.array([-60, -30, 0, 30, 60, 90], dtype=int)
    theta_rad = np.deg2rad(rel_bins.astype(float))

    # Continuous decoding targets.
    # SF is decoded as log2(mapped_sf) over non-baseline trials.
    # Orientation is decoded as [sin(2theta), cos(2theta)] to preserve circular structure.
    mapped_sf_kept = np.array([convert_sf_value(sf) for sf in spatialfreq_history[kept_trial_indices]], dtype=float)
    log_sf_kept = np.full(mapped_sf_kept.shape, np.nan, dtype=float)
    nonzero_mapped_sf = mapped_sf_kept > 0
    log_sf_kept[nonzero_mapped_sf] = np.log2(mapped_sf_kept[nonzero_mapped_sf])

    # Precompute kept-trial indices by SF and by orientation (used for all N)
    trial_inds_by_sf: List[np.ndarray] = []
    for sf in unique_spatialfreqs:
        trial_idx_all = np.where(spatialfreq_history == sf)[0]
        trial_idx = map_to_kept(trial_idx_all)
        trial_inds_by_sf.append(trial_idx)

    trial_inds_by_ori: List[np.ndarray] = []
    for ori in unique_orientations:
        trial_idx_all = np.where(orientation_history == ori)[0]
        trial_idx = map_to_kept(trial_idx_all)
        trial_inds_by_ori.append(trial_idx)

    # Track chance level for SF decoding plot (1 / # of non-gray SF classes present)
    nonzero_sfs_global = np.unique(spatialfreq_history[spatialfreq_history != 0])
    sf_chance_level = 1.0 / nonzero_sfs_global.size if nonzero_sfs_global.size > 0 else np.nan

    def _make_neuron_subsets(n_total: int, n_sub: int, *, seed: int) -> List[np.ndarray]:
        """Create subsets of size n_sub so every neuron is used at least once.

        Rule:
        - Start from a random permutation (no replacement) and chunk by n_sub.
        - If the last chunk is smaller than n_sub, pad it with additional neurons
          sampled WITHOUT replacement from neurons not already in that chunk.
        This ensures every neuron appears at least once across subsets.
        """
        if n_total <= 0:
            return [np.array([], dtype=int)]

        if n_total <= n_sub:
            return [np.arange(n_total, dtype=int)]

        rng_local = np.random.default_rng(seed)
        perm = rng_local.permutation(n_total)
        subsets_local = [perm[i:i + n_sub] for i in range(0, n_total, n_sub)]
        if subsets_local and subsets_local[-1].size < n_sub:
            last = subsets_local[-1]
            need = int(n_sub - last.size)
            remaining = np.setdiff1d(np.arange(n_total, dtype=int), last, assume_unique=True)
            pad = rng_local.choice(remaining, size=need, replace=False)
            subsets_local[-1] = np.concatenate([last, pad])
        return subsets_local

    # Pre-build subsets for each N (deterministic per session)
    subsets_by_count: List[List[np.ndarray]] = []
    n_repeats_by_count = np.zeros(len(DECODING_CELL_COUNTS), dtype=int)
    for i, n_sub in enumerate(DECODING_CELL_COUNTS):
        seed_i = int(DECODING_BASE_SEED) ^ int(n_sub * 1009) ^ int(zlib.crc32(str(save_path).encode('utf-8')))
        subs = _make_neuron_subsets(int(n_neurons), int(n_sub), seed=seed_i)
        subsets_by_count.append(subs)
        n_repeats_by_count[i] = int(len(subs))

    max_repeats = int(np.max(n_repeats_by_count)) if n_repeats_by_count.size > 0 else 0

    # Allocate outputs (padded with NaNs along repeats axis)
    decoding_acc_by_sf_all = np.full((len(DECODING_CELL_COUNTS), n_sf), np.nan, dtype=float)
    decoding_acc_repeats_all = np.full((len(DECODING_CELL_COUNTS), max_repeats, n_sf), np.nan, dtype=float)
    best_C_values_all = np.full((len(DECODING_CELL_COUNTS), max_repeats, n_sf), np.nan, dtype=float)
    dec_ori_acc_all = np.full((len(DECODING_CELL_COUNTS), n_sf, n_ori), np.nan, dtype=float)
    frac_rel_all = np.full((len(DECODING_CELL_COUNTS), n_sf, rel_bins.size), np.nan, dtype=float)
    osi_decoding_all = np.full((len(DECODING_CELL_COUNTS), n_sf), np.nan, dtype=float)

    sf_decoding_acc_by_ori_all = np.full((len(DECODING_CELL_COUNTS), n_ori), np.nan, dtype=float)
    sf_decoding_acc_repeats_all = np.full((len(DECODING_CELL_COUNTS), max_repeats, n_ori), np.nan, dtype=float)
    sf_best_C_values_all = np.full((len(DECODING_CELL_COUNTS), max_repeats, n_ori), np.nan, dtype=float)

    orientation_continuous_corr_by_sf_all = np.full((len(DECODING_CELL_COUNTS), n_sf), np.nan, dtype=float)
    orientation_continuous_corr_repeats_all = np.full((len(DECODING_CELL_COUNTS), max_repeats, n_sf), np.nan, dtype=float)
    orientation_continuous_best_alpha_by_sf_all = np.full((len(DECODING_CELL_COUNTS), max_repeats, n_sf), np.nan, dtype=float)

    sf_continuous_corr_by_ori_all = np.full((len(DECODING_CELL_COUNTS), n_ori), np.nan, dtype=float)
    sf_continuous_corr_repeats_all = np.full((len(DECODING_CELL_COUNTS), max_repeats, n_ori), np.nan, dtype=float)
    sf_continuous_best_alpha_by_ori_all = np.full((len(DECODING_CELL_COUNTS), max_repeats, n_ori), np.nan, dtype=float)

    for i_count, subsets in enumerate(subsets_by_count):
        # ---------- Orientation decoding per SF ----------
        decoding_acc_repeats = np.full((len(subsets), n_sf), np.nan, dtype=float)
        best_C_values = np.full((len(subsets), n_sf), np.nan, dtype=float)

        preds_by_sf = [list() for _ in range(n_sf)]
        truths_by_sf = [list() for _ in range(n_sf)]

        for r, neuron_subset in enumerate(subsets):
            for k, trial_idx in enumerate(trial_inds_by_sf):
                # Ignore baseline/gray trials (SF == 0) for decoding analyses.
                # Keep the slot in output arrays (will remain NaN), but do not fit a classifier.
                try:
                    sf_val = float(unique_spatialfreqs[k])
                except Exception:
                    sf_val = None
                if sf_val is not None and np.isfinite(sf_val) and abs(sf_val) < 1e-12:
                    continue
                if trial_idx.size == 0:
                    continue

                y = orientation_history[kept_trial_indices[trial_idx]]
                classes, counts = np.unique(y, return_counts=True)
                if classes.size < 2:
                    continue

                min_count = int(counts.min())
                if min_count < 2:
                    continue

                n_splits = min(5, min_count)
                if n_splits < 2:
                    continue

                X = delta_per_trial[neuron_subset][:, trial_idx].T  # (n_trials_sf, n_sub)

                cv_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_inner = StratifiedKFold(n_splits=min(3, min_count), shuffle=True, random_state=43)

                best_C = 1.0
                best_score = -np.inf
                for C_val in C_grid:
                    clf_inner = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            penalty='l2',
                            C=C_val,
                            solver='lbfgs',
                            max_iter=2000,
                        ),
                    )
                    try:
                        scores_inner = cross_val_score(clf_inner, X, y, cv=cv_inner, scoring='accuracy')
                        mean_score = float(np.nanmean(scores_inner))
                        if mean_score > best_score:
                            best_score = mean_score
                            best_C = C_val
                    except Exception:
                        continue

                y_all_pred: List[np.ndarray] = []
                y_all_true: List[np.ndarray] = []
                for train_idx, test_idx in cv_outer.split(X, y):
                    clf = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            penalty='l2',
                            C=best_C,
                            solver='lbfgs',
                            max_iter=20000,
                        ),
                    )
                    clf.fit(X[train_idx], y[train_idx])
                    y_pred = clf.predict(X[test_idx])
                    y_all_pred.append(y_pred)
                    y_all_true.append(y[test_idx])

                pooled_acc = float('nan')
                if y_all_pred and y_all_true:
                    y_pred_concat = np.concatenate(y_all_pred, axis=0)
                    y_true_concat = np.concatenate(y_all_true, axis=0)
                    pooled_acc = float(np.mean(y_pred_concat == y_true_concat))
                decoding_acc_repeats[r, k] = pooled_acc
                best_C_values[r, k] = best_C
                if len(y_all_pred) > 0:
                    preds_by_sf[k].append(np.concatenate(y_all_pred))
                    truths_by_sf[k].append(np.concatenate(y_all_true))

        decoding_acc_by_sf = np.nanmean(decoding_acc_repeats, axis=0)

        # Concatenate predictions and truths per SF
        preds_concat: List[np.ndarray] = []
        truths_concat: List[np.ndarray] = []
        for k in range(n_sf):
            if len(preds_by_sf[k]) > 0:
                preds_concat.append(np.concatenate(preds_by_sf[k]))
                truths_concat.append(np.concatenate(truths_by_sf[k]))
            else:
                preds_concat.append(np.array([], dtype=float))
                truths_concat.append(np.array([], dtype=float))

        # Compute orientation-specific accuracies per SF
        orient_acc_by_sf = np.full((n_sf, n_ori), np.nan, dtype=float)
        for k in range(n_sf):
            y_true_k = truths_concat[k]
            y_pred_k = preds_concat[k]
            if y_true_k.size == 0:
                continue
            for j, ori in enumerate(unique_orientations):
                mask = (y_true_k == ori)
                if np.any(mask):
                    orient_acc_by_sf[k, j] = float(np.mean(y_pred_k[mask] == y_true_k[mask]))

        # Compute fractions over relative decoded orientations per SF
        frac_rel_by_sf = np.full((n_sf, rel_bins.size), np.nan, dtype=float)
        for k in range(n_sf):
            y_true_k = truths_concat[k]
            y_pred_k = preds_concat[k]
            if y_true_k.size == 0:
                continue
            diffs = (y_pred_k - y_true_k) % 180
            diffs = np.where(diffs > 90, diffs - 180, diffs)  # now in [-90,90]
            snapped = np.array([rel_bins[int(np.argmin(np.abs(rel_bins - d)))] for d in diffs], dtype=int)
            total = int(snapped.size)
            if total > 0:
                for b_idx, b in enumerate(rel_bins):
                    frac_rel_by_sf[k, b_idx] = float(np.sum(snapped == b) / total)

        # Decoding-accuracy-based vector sum OSI as a function of SF
        osi_decoding_by_sf = np.full(n_sf, np.nan, dtype=float)
        for k in range(n_sf):
            frac_k = frac_rel_by_sf[k]
            if np.any(~np.isnan(frac_k)):
                valid = ~np.isnan(frac_k)
                f = frac_k[valid]
                t = theta_rad[valid]
                denom = float(np.sum(f))
                if denom > 0:
                    vec = np.sum(f * np.exp(1j * 2 * t))
                    osi_decoding_by_sf[k] = float(np.abs(vec) / denom)

        # ---------- SF decoding per orientation ----------
        sf_decoding_acc_repeats = np.full((len(subsets), n_ori), np.nan, dtype=float)
        sf_best_C_values = np.full((len(subsets), n_ori), np.nan, dtype=float)

        # ---------- Continuous decoding: orientation and SF ----------
        orientation_continuous_corr_repeats = np.full((len(subsets), n_sf), np.nan, dtype=float)
        orientation_continuous_best_alpha = np.full((len(subsets), n_sf), np.nan, dtype=float)
        sf_continuous_corr_repeats = np.full((len(subsets), n_ori), np.nan, dtype=float)
        sf_continuous_best_alpha = np.full((len(subsets), n_ori), np.nan, dtype=float)

        for r, neuron_subset in enumerate(subsets):
            for j, trial_idx in enumerate(trial_inds_by_ori):
                if trial_idx.size == 0:
                    continue

                y_raw = spatialfreq_history[kept_trial_indices[trial_idx]]
                nonzero_mask = y_raw != 0
                if not np.any(nonzero_mask):
                    continue

                trial_idx_nz = trial_idx[nonzero_mask]
                y_raw_nz = y_raw[nonzero_mask]

                y_rounded = np.round(y_raw_nz, decimals=6)
                _, y = np.unique(y_rounded, return_inverse=True)
                classes, counts = np.unique(y, return_counts=True)
                if classes.size < 2:
                    continue

                min_count = int(counts.min())
                if min_count < 2:
                    continue

                n_splits = min(5, min_count)
                if n_splits < 2:
                    continue

                X = delta_per_trial[neuron_subset][:, trial_idx_nz].T  # (n_trials_ori_nonzero_sf, n_sub)

                cv_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=52)
                cv_inner = StratifiedKFold(n_splits=min(3, min_count), shuffle=True, random_state=53)

                best_C = 1.0
                best_score = -np.inf
                for C_val in C_grid:
                    clf_inner = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            penalty='l2',
                            C=C_val,
                            solver='lbfgs',
                            max_iter=20000,
                        ),
                    )
                    try:
                        scores_inner = cross_val_score(clf_inner, X, y, cv=cv_inner, scoring='accuracy')
                        mean_score = float(np.nanmean(scores_inner))
                        if mean_score > best_score:
                            best_score = mean_score
                            best_C = C_val
                    except Exception:
                        continue

                y_all_pred: List[np.ndarray] = []
                y_all_true: List[np.ndarray] = []
                for train_idx, test_idx in cv_outer.split(X, y):
                    clf = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            penalty='l2',
                            C=best_C,
                            solver='lbfgs',
                            max_iter=20000,
                        ),
                    )
                    clf.fit(X[train_idx], y[train_idx])
                    y_pred = clf.predict(X[test_idx])
                    y_all_pred.append(y_pred)
                    y_all_true.append(y[test_idx])

                pooled_acc = float('nan')
                if y_all_pred and y_all_true:
                    y_pred_concat = np.concatenate(y_all_pred, axis=0)
                    y_true_concat = np.concatenate(y_all_true, axis=0)
                    pooled_acc = float(np.mean(y_pred_concat == y_true_concat))
                sf_decoding_acc_repeats[r, j] = pooled_acc
                sf_best_C_values[r, j] = best_C

            for k, trial_idx in enumerate(trial_inds_by_sf):
                try:
                    sf_val = float(unique_spatialfreqs[k])
                except Exception:
                    sf_val = None
                if sf_val is not None and np.isfinite(sf_val) and abs(sf_val) < 1e-12:
                    continue
                if trial_idx.size < 4:
                    continue

                y_ori_deg = np.asarray(orientation_history[kept_trial_indices[trial_idx]], dtype=float)
                if np.unique(y_ori_deg).size < 2:
                    continue

                X = delta_per_trial[neuron_subset][:, trial_idx].T
                y = _orientation_targets_from_deg(y_ori_deg)
                outer_cv = KFold(n_splits=min(5, X.shape[0]), shuffle=True, random_state=62)
                inner_cv = KFold(n_splits=min(3, X.shape[0]), shuffle=True, random_state=63)

                score, best_alpha, _y_pred, _y_true = _run_nested_linear_decoder(
                    X,
                    y,
                    alpha_grid=RIDGE_ALPHA_GRID,
                    outer_cv=outer_cv,
                    inner_cv=inner_cv,
                    scorer=_orientation_regression_score,
                )
                orientation_continuous_corr_repeats[r, k] = score
                orientation_continuous_best_alpha[r, k] = best_alpha

            for j, trial_idx in enumerate(trial_inds_by_ori):
                if trial_idx.size < 4:
                    continue

                y_log_sf = log_sf_kept[trial_idx]
                valid_mask = np.isfinite(y_log_sf)
                if np.sum(valid_mask) < 4:
                    continue

                trial_idx_valid = trial_idx[valid_mask]
                y_log_sf = y_log_sf[valid_mask]
                if np.unique(y_log_sf).size < 2:
                    continue

                X = delta_per_trial[neuron_subset][:, trial_idx_valid].T
                y = y_log_sf[:, None]
                outer_cv = KFold(n_splits=min(5, X.shape[0]), shuffle=True, random_state=72)
                inner_cv = KFold(n_splits=min(3, X.shape[0]), shuffle=True, random_state=73)

                score, best_alpha, _y_pred, _y_true = _run_nested_linear_decoder(
                    X,
                    y,
                    alpha_grid=RIDGE_ALPHA_GRID,
                    outer_cv=outer_cv,
                    inner_cv=inner_cv,
                    scorer=_sf_regression_score,
                )
                sf_continuous_corr_repeats[r, j] = score
                sf_continuous_best_alpha[r, j] = best_alpha

        sf_decoding_acc_by_ori = np.nanmean(sf_decoding_acc_repeats, axis=0)
        orientation_continuous_corr_by_sf = np.nanmean(orientation_continuous_corr_repeats, axis=0)
        sf_continuous_corr_by_ori = np.nanmean(sf_continuous_corr_repeats, axis=0)

        # ---------- Store into global arrays (pad repeats) ----------
        decoding_acc_by_sf_all[i_count, :] = decoding_acc_by_sf
        dec_ori_acc_all[i_count, :, :] = orient_acc_by_sf
        frac_rel_all[i_count, :, :] = frac_rel_by_sf
        osi_decoding_all[i_count, :] = osi_decoding_by_sf
        sf_decoding_acc_by_ori_all[i_count, :] = sf_decoding_acc_by_ori
        orientation_continuous_corr_by_sf_all[i_count, :] = orientation_continuous_corr_by_sf
        sf_continuous_corr_by_ori_all[i_count, :] = sf_continuous_corr_by_ori

        decoding_acc_repeats_all[i_count, : decoding_acc_repeats.shape[0], :] = decoding_acc_repeats
        best_C_values_all[i_count, : best_C_values.shape[0], :] = best_C_values
        sf_decoding_acc_repeats_all[i_count, : sf_decoding_acc_repeats.shape[0], :] = sf_decoding_acc_repeats
        sf_best_C_values_all[i_count, : sf_best_C_values.shape[0], :] = sf_best_C_values
        orientation_continuous_corr_repeats_all[i_count, : orientation_continuous_corr_repeats.shape[0], :] = orientation_continuous_corr_repeats
        orientation_continuous_best_alpha_by_sf_all[i_count, : orientation_continuous_best_alpha.shape[0], :] = orientation_continuous_best_alpha
        sf_continuous_corr_repeats_all[i_count, : sf_continuous_corr_repeats.shape[0], :] = sf_continuous_corr_repeats
        sf_continuous_best_alpha_by_ori_all[i_count, : sf_continuous_best_alpha.shape[0], :] = sf_continuous_best_alpha

    # Save all cell-count decoding results for grouped analyses
    np.savez(
        os.path.join(save_path, 'decoding_cellcount_results.npz'),
        cell_counts=np.array(DECODING_CELL_COUNTS, dtype=int),
        unique_spatialfreqs=unique_spatialfreqs,
        unique_orientations=unique_orientations,
        n_neurons=n_neurons,
        n_repeats_by_count=n_repeats_by_count,
        rel_bins=rel_bins,
        C_grid=np.array(C_grid, dtype=float),
        orientation_decoding_accuracy_by_sf=decoding_acc_by_sf_all,
        orientation_decoding_accuracy_repeats_by_sf=decoding_acc_repeats_all,
        orientation_decoding_best_C_values=best_C_values_all,
        decoder_orientation_accuracy_by_sf=dec_ori_acc_all,
        decoder_fraction_relative_orientation_by_sf=frac_rel_all,
        decoding_based_osi_by_sf=osi_decoding_all,
        sf_decoding_accuracy_by_orientation=sf_decoding_acc_by_ori_all,
        sf_decoding_accuracy_repeats_by_orientation=sf_decoding_acc_repeats_all,
        sf_decoding_best_C_values_by_orientation=sf_best_C_values_all,
        orientation_continuous_circular_corr_by_sf=orientation_continuous_corr_by_sf_all,
        orientation_continuous_circular_corr_repeats_by_sf=orientation_continuous_corr_repeats_all,
        orientation_continuous_best_alpha_by_sf=orientation_continuous_best_alpha_by_sf_all,
        sf_continuous_pearson_by_orientation=sf_continuous_corr_by_ori_all,
        sf_continuous_pearson_repeats_by_orientation=sf_continuous_corr_repeats_all,
        sf_continuous_best_alpha_by_orientation=sf_continuous_best_alpha_by_ori_all,
        ridge_alpha_grid=np.array(RIDGE_ALPHA_GRID, dtype=float),
    )

    # Keep legacy single-N outputs/plots (default N=80) for backward compatibility.
    try:
        default_idx = int(np.where(np.array(DECODING_CELL_COUNTS, dtype=int) == int(DECODING_DEFAULT_N))[0][0])
    except Exception:
        default_idx = int(len(DECODING_CELL_COUNTS) - 1)

    decoding_acc_by_sf = decoding_acc_by_sf_all[default_idx]
    decoding_acc_repeats = decoding_acc_repeats_all[default_idx, : n_repeats_by_count[default_idx], :]
    best_C_values = best_C_values_all[default_idx, : n_repeats_by_count[default_idx], :]
    orient_acc_by_sf = dec_ori_acc_all[default_idx]
    frac_rel_by_sf = frac_rel_all[default_idx]
    osi_decoding_by_sf = osi_decoding_all[default_idx]
    sf_decoding_acc_by_ori = sf_decoding_acc_by_ori_all[default_idx]
    sf_decoding_acc_repeats = sf_decoding_acc_repeats_all[default_idx, : n_repeats_by_count[default_idx], :]
    sf_best_C_values = sf_best_C_values_all[default_idx, : n_repeats_by_count[default_idx], :]
    orientation_continuous_corr_by_sf = orientation_continuous_corr_by_sf_all[default_idx]
    orientation_continuous_corr_repeats = orientation_continuous_corr_repeats_all[default_idx, : n_repeats_by_count[default_idx], :]
    orientation_continuous_best_alpha = orientation_continuous_best_alpha_by_sf_all[default_idx, : n_repeats_by_count[default_idx], :]
    sf_continuous_corr_by_ori = sf_continuous_corr_by_ori_all[default_idx]
    sf_continuous_corr_repeats = sf_continuous_corr_repeats_all[default_idx, : n_repeats_by_count[default_idx], :]
    sf_continuous_best_alpha = sf_continuous_best_alpha_by_ori_all[default_idx, : n_repeats_by_count[default_idx], :]

    # Plot: orientation accuracies per SF (line plot per SF)
    fig, axes = plt.subplots(1, n_sf, figsize=(4 * n_sf, 4), sharey=True)
    for k in range(n_sf):
        ax = axes[k] if n_sf > 1 else axes
        yvals = orient_acc_by_sf[k]
        ax.plot(unique_orientations, yvals, marker='o')
        ax.set_xlabel('Orientation (deg)')
        ax.set_title(f'SF: {unique_spatialfreqs[k]}')
        ax.set_ylim(0, 1)
    axes[0].set_ylabel('Decoding accuracy')
    plt.suptitle(f'Decoder Accuracy by Orientation (per SF) | N={DECODING_DEFAULT_N}')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_path, 'decoder_accuracy_by_orientation_per_sf.svg'), format='svg')
    plt.close()

    # Plot: fractions over relative decoded orientations per SF (line plot per SF)
    fig, axes = plt.subplots(1, n_sf, figsize=(4 * n_sf, 4), sharey=True)
    for k in range(n_sf):
        ax = axes[k] if n_sf > 1 else axes
        yvals = frac_rel_by_sf[k]
        ax.plot(rel_bins, yvals, marker='o')
        ax.set_xticks(rel_bins)
        ax.set_xticklabels(['-60', '-30', 'Correct', '+30', '+60', '+90'])
        ax.set_xlabel('Decoded relative orientation')
        ax.set_title(f'SF: {unique_spatialfreqs[k]}')
        ax.set_ylim(0, 1)
    axes[0].set_ylabel('Fraction of trials')
    plt.suptitle(f'Decoder Output Fractions by Relative Orientation (per SF) | N={DECODING_DEFAULT_N}')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_path, 'decoder_output_fraction_relative_orientation_per_sf.svg'), format='svg')
    plt.close()

    # Plot OSI (decoding-based) vs SF
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(unique_spatialfreqs, osi_decoding_by_sf, marker='o')
    ax.set_xscale('log')
    ax.set_xlabel('Spatial Frequency (log scale)')
    ax.set_ylabel('Decoding-based OSI (vector sum)')
    ax.set_ylim(0, 1)
    ax.set_title(f'Decoding-based OSI vs Spatial Frequency | N={DECODING_DEFAULT_N}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'decoding_based_osi_vs_sf.svg'), format='svg')
    plt.close()

    # Plot decoding accuracy by spatial frequency (default N)
    n_reps_per_sf = np.sum(~np.isnan(decoding_acc_repeats), axis=0)
    sem = np.full_like(decoding_acc_by_sf, np.nan, dtype=float)
    valid = n_reps_per_sf > 0
    if np.any(valid):
        sem_vals = np.nanstd(decoding_acc_repeats[:, valid], axis=0)
        sem[valid] = sem_vals / np.sqrt(n_reps_per_sf[valid])

    x = np.arange(len(unique_spatialfreqs))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(x, decoding_acc_by_sf, yerr=sem, fmt='-o', capsize=5, color='C0', ecolor='gray')

    # Analytic chance per SF (1 / n_classes present for that SF)
    chance_by_sf = np.full(len(unique_spatialfreqs), np.nan, dtype=float)
    for k, trial_idx in enumerate(trial_inds_by_sf):
        if trial_idx.size == 0:
            continue
        labels = orientation_history[kept_trial_indices[trial_idx]]
        classes = np.unique(labels)
        if classes.size > 0:
            chance_by_sf[k] = 1.0 / float(classes.size)
    valid_chance = ~np.isnan(chance_by_sf)
    if np.any(valid_chance):
        ax.plot(
            x[valid_chance],
            chance_by_sf[valid_chance],
            linestyle='--',
            color='red',
            marker='o',
            label='Analytic chance (1 / n_classes)',
        )
        ax.legend(loc='upper right')

    ax.set_xticks(x)
    ax.set_xticklabels([f"{sf:.4g}" for sf in unique_spatialfreqs], rotation=45)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Decoding accuracy')
    ax.set_title(f'Orientation decoding accuracy by Spatial Frequency | N={DECODING_DEFAULT_N}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'orientation_decoding_accuracy_by_sf.svg'), format='svg')
    plt.close()

    # Plot SF decoding accuracy by orientation (default N)
    n_reps_per_ori = np.sum(~np.isnan(sf_decoding_acc_repeats), axis=0)
    sf_sem = np.full_like(sf_decoding_acc_by_ori, np.nan, dtype=float)
    valid_sf = n_reps_per_ori > 0
    if np.any(valid_sf):
        sf_sem_vals = np.nanstd(sf_decoding_acc_repeats[:, valid_sf], axis=0)
        sf_sem[valid_sf] = sf_sem_vals / np.sqrt(n_reps_per_ori[valid_sf])

    x_ori = np.arange(len(unique_orientations))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(x_ori, sf_decoding_acc_by_ori, yerr=sf_sem, fmt='-o', capsize=5, color='C1', ecolor='gray')
    if not np.isnan(sf_chance_level):
        ax.axhline(y=sf_chance_level, linestyle='--', color='red', label='Chance (1 / n_SF classes)')
        ax.legend(loc='upper right')
    ax.set_xticks(x_ori)
    ax.set_xticklabels([f"{ori:.0f}" for ori in unique_orientations], rotation=45)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Orientation (deg)')
    ax.set_ylabel('SF decoding accuracy')
    ax.set_title(f'Spatial Frequency decoding accuracy by Orientation | N={DECODING_DEFAULT_N}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'sf_decoding_accuracy_by_orientation.svg'), format='svg')
    plt.close()

    # Plot continuous orientation decoding by SF (default N)
    n_reps_per_sf_cont = np.sum(~np.isnan(orientation_continuous_corr_repeats), axis=0)
    ori_cont_sem = np.full_like(orientation_continuous_corr_by_sf, np.nan, dtype=float)
    valid_ori_cont = n_reps_per_sf_cont > 0
    if np.any(valid_ori_cont):
        ori_cont_sem_vals = np.nanstd(orientation_continuous_corr_repeats[:, valid_ori_cont], axis=0)
        ori_cont_sem[valid_ori_cont] = ori_cont_sem_vals / np.sqrt(n_reps_per_sf_cont[valid_ori_cont])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(x, orientation_continuous_corr_by_sf, yerr=ori_cont_sem, fmt='-o', capsize=5, color='C2', ecolor='gray')
    ax.axhline(y=0.0, linestyle='--', color='red', linewidth=1.0, label='Null correlation')
    ax.legend(loc='lower right')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{sf:.4g}" for sf in unique_spatialfreqs], rotation=45)
    ax.set_ylim(-1, 1)
    ax.set_ylabel('Circular correlation')
    ax.set_title(f'Continuous orientation decoding by Spatial Frequency | N={DECODING_DEFAULT_N}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'orientation_continuous_circular_corr_by_sf.svg'), format='svg')
    plt.close()

    # Plot continuous SF decoding by orientation (default N)
    n_reps_per_ori_cont = np.sum(~np.isnan(sf_continuous_corr_repeats), axis=0)
    sf_cont_sem = np.full_like(sf_continuous_corr_by_ori, np.nan, dtype=float)
    valid_sf_cont = n_reps_per_ori_cont > 0
    if np.any(valid_sf_cont):
        sf_cont_sem_vals = np.nanstd(sf_continuous_corr_repeats[:, valid_sf_cont], axis=0)
        sf_cont_sem[valid_sf_cont] = sf_cont_sem_vals / np.sqrt(n_reps_per_ori_cont[valid_sf_cont])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(x_ori, sf_continuous_corr_by_ori, yerr=sf_cont_sem, fmt='-o', capsize=5, color='C3', ecolor='gray')
    ax.axhline(y=0.0, linestyle='--', color='red', linewidth=1.0, label='Null correlation')
    ax.legend(loc='lower right')
    ax.set_xticks(x_ori)
    ax.set_xticklabels([f"{ori:.0f}" for ori in unique_orientations], rotation=45)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Orientation (deg)')
    ax.set_ylabel('Pearson r')
    ax.set_title(f'Continuous SF decoding by Orientation | N={DECODING_DEFAULT_N}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'sf_continuous_pearson_by_orientation.svg'), format='svg')
    plt.close()

    # Legacy saves consumed by grouped pipeline (default N)
    np.save(os.path.join(save_path, 'decoding_based_osi_by_sf.npy'), osi_decoding_by_sf)
    np.save(os.path.join(save_path, 'orientation_decoding_accuracy_repeats_by_sf.npy'), decoding_acc_repeats)
    np.save(os.path.join(save_path, 'orientation_decoding_best_C_values.npy'), best_C_values)
    np.save(os.path.join(save_path, 'decoder_orientation_accuracy_by_sf.npy'), orient_acc_by_sf)
    np.save(os.path.join(save_path, 'decoder_fraction_relative_orientation_by_sf.npy'), frac_rel_by_sf)
    np.save(os.path.join(save_path, 'sf_decoding_accuracy_repeats_by_orientation.npy'), sf_decoding_acc_repeats)
    np.save(os.path.join(save_path, 'sf_decoding_best_C_values_by_orientation.npy'), sf_best_C_values)
    np.save(os.path.join(save_path, 'sf_decoding_accuracy_by_orientation.npy'), sf_decoding_acc_by_ori)
    np.save(os.path.join(save_path, 'orientation_continuous_circular_corr_by_sf.npy'), orientation_continuous_corr_by_sf)
    np.save(os.path.join(save_path, 'orientation_continuous_circular_corr_repeats_by_sf.npy'), orientation_continuous_corr_repeats)
    np.save(os.path.join(save_path, 'orientation_continuous_best_alpha_by_sf.npy'), orientation_continuous_best_alpha)
    np.save(os.path.join(save_path, 'sf_continuous_pearson_by_orientation.npy'), sf_continuous_corr_by_ori)
    np.save(os.path.join(save_path, 'sf_continuous_pearson_repeats_by_orientation.npy'), sf_continuous_corr_repeats)
    np.save(os.path.join(save_path, 'sf_continuous_best_alpha_by_orientation.npy'), sf_continuous_best_alpha)
    np.savez(
        os.path.join(save_path, 'orientation_decoding_metadata.npz'),
        unique_spatialfreqs=unique_spatialfreqs,
        unique_orientations=unique_orientations,
        n_neurons=n_neurons,
        n_subsample=int(DECODING_DEFAULT_N),
        n_repeats=int(decoding_acc_repeats.shape[0]),
        best_C_values=best_C_values,
        C_grid=np.array(C_grid, dtype=float),
        sf_decoding_accuracy_by_orientation=sf_decoding_acc_by_ori,
        sf_decoding_accuracy_repeats_by_orientation=sf_decoding_acc_repeats,
        sf_decoding_best_C_values_by_orientation=sf_best_C_values,
        orientation_continuous_circular_corr_by_sf=orientation_continuous_corr_by_sf,
        orientation_continuous_circular_corr_repeats_by_sf=orientation_continuous_corr_repeats,
        orientation_continuous_best_alpha_by_sf=orientation_continuous_best_alpha,
        sf_continuous_pearson_by_orientation=sf_continuous_corr_by_ori,
        sf_continuous_pearson_repeats_by_orientation=sf_continuous_corr_repeats,
        sf_continuous_best_alpha_by_orientation=sf_continuous_best_alpha,
        ridge_alpha_grid=np.array(RIDGE_ALPHA_GRID, dtype=float),
        decoding_cellcount_results_file='decoding_cellcount_results.npz',
    )


    # ------------------------------------------------------------------
    # Signal vs noise correlations and their SF/orientation dependence
    # - Signal corr: pairwise correlation of condition-mean responses
    # - Noise corr: pairwise correlation of residuals (trial - condition mean)
    # ------------------------------------------------------------------

    # Build condition means per (orientation, SF) excluding baseline (sf == 0)
    nonbaseline_sfs = [sf for sf in unique_spatialfreqs if sf != 0]
    condition_keys = []
    condition_means = []  # shape will be (n_neurons, n_conditions)
    condition_trial_indices = []
    for ori in unique_orientations:
        for sf in nonbaseline_sfs:
            cond_trials_orig = np.where((orientation_history == ori) & (spatialfreq_history == sf))[0]
            cond_trials_kept = map_to_kept(cond_trials_orig)
            if cond_trials_kept.size == 0:
                continue
            condition_keys.append((ori, sf))
            condition_trial_indices.append(cond_trials_kept)
            cond_mean = np.nanmean(delta_per_trial[:, cond_trials_kept], axis=1)  # (n_neurons,)
            condition_means.append(cond_mean)

    if condition_means:
        condition_means = np.stack(condition_means, axis=1)  # (n_neurons, n_conditions)
    else:
        condition_means = np.empty((n_neurons, 0))

    def corrcoef_safe(mat):
        """Compute correlation matrix dropping columns with NaNs or zero variance."""
        if mat.size == 0 or mat.shape[1] < 2:
            return None
        valid = ~np.any(np.isnan(mat), axis=0)
        if not np.any(valid):
            return None
        mat_valid = mat[:, valid]
        # Drop zero-variance columns to avoid NaNs
        var_nonzero = np.nanstd(mat_valid, axis=0) > 0
        if not np.any(var_nonzero):
            return None
        mat_use = mat_valid[:, var_nonzero]
        if mat_use.shape[1] < 2:
            return None
        return np.corrcoef(mat_use)

    def upper_tri_mean(mat):
        if mat is None:
            return np.nan
        idx = np.triu_indices_from(mat, k=1)
        if idx[0].size == 0:
            return np.nan
        return float(np.nanmean(mat[idx]))

    # Signal correlation overall
    signal_corr_overall = corrcoef_safe(condition_means)

    # Noise residuals per trial: subtract condition mean
    residuals = np.full_like(delta_per_trial, np.nan)
    trial_condition_mean = np.full_like(delta_per_trial, np.nan)
    for cond_mean, trial_idx in zip(condition_means.T, condition_trial_indices):
        # broadcast condition mean to each trial in this condition
        trial_condition_mean[:, trial_idx] = cond_mean[:, None]
    residuals = delta_per_trial - trial_condition_mean

    noise_corr_overall = corrcoef_safe(residuals)

    # Mean correlations binned by SF (condition-based)
    signal_corr_by_sf = np.full(n_sf, np.nan, dtype=float)
    noise_corr_by_sf = np.full(n_sf, np.nan, dtype=float)
    for sf_idx, sf in enumerate(unique_spatialfreqs):
        if sf == 0:
            continue
        # Signal: use condition means for this SF across orientations
        cols = [i for i, (ori_k, sf_k) in enumerate(condition_keys) if sf_k == sf]
        if len(cols) >= 2:
            mat_sf = condition_means[:, cols]
            signal_corr_by_sf[sf_idx] = upper_tri_mean(corrcoef_safe(mat_sf))

        # Noise: use residuals from trials with this SF (all orientations)
        trials_sf_orig = np.where(spatialfreq_history == sf)[0]
        trials_sf = map_to_kept(trials_sf_orig)
        if trials_sf.size >= 2:
            noise_corr_by_sf[sf_idx] = upper_tri_mean(corrcoef_safe(residuals[:, trials_sf]))

    # Mean correlations binned by orientation (condition-based)
    signal_corr_by_ori = np.full(n_ori, np.nan, dtype=float)
    noise_corr_by_ori = np.full(n_ori, np.nan, dtype=float)
    for ori_idx, ori in enumerate(unique_orientations):
        cols = [i for i, (ori_k, sf_k) in enumerate(condition_keys) if ori_k == ori]
        if len(cols) >= 2:
            mat_ori = condition_means[:, cols]
            signal_corr_by_ori[ori_idx] = upper_tri_mean(corrcoef_safe(mat_ori))

        trials_ori_orig = np.where((orientation_history == ori) & (spatialfreq_history != 0))[0]
        trials_ori = map_to_kept(trials_ori_orig)
        if trials_ori.size >= 2:
            noise_corr_by_ori[ori_idx] = upper_tri_mean(corrcoef_safe(residuals[:, trials_ori]))

    # Signal - noise deltas by SF/orientation (mean upper-tri values)
    signal_minus_noise_by_sf = signal_corr_by_sf - noise_corr_by_sf
    signal_minus_noise_by_ori = signal_corr_by_ori - noise_corr_by_ori

    # Save correlation outputs
    if signal_corr_overall is not None:
        np.save(os.path.join(save_path, 'signal_corr_overall.npy'), signal_corr_overall)
    if noise_corr_overall is not None:
        np.save(os.path.join(save_path, 'noise_corr_overall.npy'), noise_corr_overall)
    np.save(os.path.join(save_path, 'signal_corr_mean_by_sf.npy'), signal_corr_by_sf)
    np.save(os.path.join(save_path, 'noise_corr_mean_by_sf.npy'), noise_corr_by_sf)
    np.save(os.path.join(save_path, 'signal_minus_noise_corr_by_sf.npy'), signal_minus_noise_by_sf)
    np.save(os.path.join(save_path, 'signal_corr_mean_by_orientation.npy'), signal_corr_by_ori)
    np.save(os.path.join(save_path, 'noise_corr_mean_by_orientation.npy'), noise_corr_by_ori)
    np.save(os.path.join(save_path, 'signal_minus_noise_corr_by_orientation.npy'), signal_minus_noise_by_ori)

    # Plots: overall histograms
    if signal_corr_overall is not None:
        triu_sig = signal_corr_overall[np.triu_indices_from(signal_corr_overall, k=1)]
        plt.figure(figsize=(5, 4))
        plt.hist(triu_sig, bins=40, color='blue', alpha=0.7)
        plt.xlabel('Signal correlation')
        plt.ylabel('Count')
        plt.title('Distribution of signal correlations (all neuron pairs)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'signal_correlation_hist.svg'), format='svg')
        plt.close()

    if noise_corr_overall is not None:
        triu_noise = noise_corr_overall[np.triu_indices_from(noise_corr_overall, k=1)]
        plt.figure(figsize=(5, 4))
        plt.hist(triu_noise, bins=40, color='orange', alpha=0.7)
        plt.xlabel('Noise correlation')
        plt.ylabel('Count')
        plt.title('Distribution of noise correlations (all neuron pairs)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'noise_correlation_hist.svg'), format='svg')
        plt.close()

    # Overlay noise vs signal (paired upper-tri values) — swapped x and y
    if (signal_corr_overall is not None) and (noise_corr_overall is not None):
        triu_sig = signal_corr_overall[np.triu_indices_from(signal_corr_overall, k=1)]
        triu_noise = noise_corr_overall[np.triu_indices_from(noise_corr_overall, k=1)]
        finite_mask = np.isfinite(triu_sig) & np.isfinite(triu_noise)
        if np.any(finite_mask):
            plt.figure(figsize=(5, 5))
            # swap x and y: plot noise on x-axis, signal on y-axis
            plt.scatter(triu_noise[finite_mask], triu_sig[finite_mask], s=8, alpha=0.4, color='purple')
            lims = [min(triu_noise[finite_mask].min(), triu_sig[finite_mask].min()),
                    max(triu_noise[finite_mask].max(), triu_sig[finite_mask].max())]
            plt.plot(lims, lims, 'k--', linewidth=1, label='Identity')
            plt.xlabel('Noise correlation (pairwise)')
            plt.ylabel('Signal correlation (pairwise)')
            plt.title('Noise vs Signal correlations (paired neuron pairs)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'noise_vs_signal_correlation_scatter.svg'), format='svg')
            plt.close()

    # Plots: mean correlation vs SF and orientation
    x_sf = np.arange(len(unique_spatialfreqs))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_sf, signal_corr_by_sf, '-o', label='Signal corr', color='blue')
    ax.plot(x_sf, noise_corr_by_sf, '-o', label='Noise corr', color='orange')
    ax.set_xticks(x_sf)
    ax.set_xticklabels([f"{sf:.4g}" for sf in unique_spatialfreqs], rotation=45)
    ax.set_ylabel('Mean pairwise correlation (upper-tri mean)')
    ax.set_title('Signal/Noise correlations vs Spatial Frequency')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'signal_noise_correlations_by_sf.svg'), format='svg')
    plt.close()

    x_o = np.arange(len(unique_orientations))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_o, signal_corr_by_ori, '-o', label='Signal corr', color='blue')
    ax.plot(x_o, noise_corr_by_ori, '-o', label='Noise corr', color='orange')
    ax.set_xticks(x_o)
    ax.set_xticklabels([f"{ori:.0f}" for ori in unique_orientations], rotation=45)
    ax.set_ylabel('Mean pairwise correlation (upper-tri mean)')
    ax.set_title('Signal/Noise correlations vs Orientation')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'signal_noise_correlations_by_orientation.svg'), format='svg')
    plt.close()


    # ------------------------------------------------------------------
    # Dimensionality & geometry of population codes
    # - PCA on baseline-subtracted responses (non-gray trials)
    # - Participation ratio as an effective dimensionality metric
    # - Variance spectrum and power-law fit
    # - Train/test split comparison of dimensionality and variance allocation
    # ------------------------------------------------------------------

    def participation_ratio(eigvals):
        eigvals = np.asarray(eigvals, dtype=float)
        eigvals = eigvals[eigvals > 0]
        if eigvals.size == 0:
            return np.nan
        return (np.sum(eigvals) ** 2) / np.sum(eigvals ** 2)

    # Subsample neurons for PCA/dsPCA within this session (reproducible).
    # Use the same subset for PCA, manifold visualization, and dsPCA.
    neuron_subset_idx = None
    n_neurons_used = int(n_neurons)
    delta_per_trial_sub = delta_per_trial
    condition_means_sub = condition_means
    try:
        # Prefer neurons that have any finite condition-mean value.
        valid = np.isfinite(condition_means).any(axis=1) if (condition_means is not None and condition_means.ndim == 2) else None
        candidates = np.where(valid)[0] if (valid is not None and np.any(valid)) else np.arange(n_neurons)
        n_select = int(min(SUBSAMPLE_N_NEURONS_FOR_PCA_DSPCA, candidates.size))
        if n_select >= 2:
            seed_session = int(SUBSAMPLE_NEURON_SEED) ^ int(zlib.crc32(str(save_path).encode('utf-8')))
            rng_cells = np.random.default_rng(seed_session)
            if candidates.size > n_select:
                neuron_subset_idx = np.sort(rng_cells.choice(candidates, size=n_select, replace=False))
            else:
                neuron_subset_idx = np.sort(candidates)
            delta_per_trial_sub = delta_per_trial[neuron_subset_idx, :]
            condition_means_sub = condition_means[neuron_subset_idx, :]
            n_neurons_used = int(delta_per_trial_sub.shape[0])
    except Exception:
        # Fall back to using all neurons.
        neuron_subset_idx = None
        n_neurons_used = int(n_neurons)
        delta_per_trial_sub = delta_per_trial
        condition_means_sub = condition_means

    print(f"[PCA/dsPCA] Using {n_neurons_used} randomly selected neurons (of {n_neurons} total)")

    trial_sf_kept = spatialfreq_history[kept_trial_indices]
    nonbaseline_mask = trial_sf_kept != 0

    pca_eigvals = np.array([])
    pca_explained_ratio = np.array([])
    pca_cum_ratio = np.array([])
    powerlaw_alpha = np.nan
    powerlaw_intercept = np.nan
    pr_all = np.nan

    pr_train = np.nan
    pr_test = np.nan
    evr_train = np.array([])
    evr_test = np.array([])

    pr_by_sf = np.full(n_sf, np.nan, dtype=float)
    knee_pc = np.nan  # PC index at which cumulative variance reaches 80%

    if np.sum(nonbaseline_mask) >= 2 and n_neurons_used >= 2:
        X_all = delta_per_trial_sub[:, nonbaseline_mask].T  # samples x neurons
        pca = PCA(svd_solver='full')
        pca.fit(X_all)
        pca_eigvals = pca.explained_variance_
        pca_explained_ratio = pca.explained_variance_ratio_
        pca_cum_ratio = np.cumsum(pca_explained_ratio)
        pr_all = participation_ratio(pca_eigvals)
        
        # Find knee: first PC where cumulative variance >= 80%
        knee_idx = np.where(pca_cum_ratio >= 0.8)[0]
        if knee_idx.size > 0:
            knee_pc = int(knee_idx[0] + 1)  # +1 for 1-based indexing

        # Power-law fit of eigenvalue spectrum (log-log)
        eig_pos = pca_eigvals[pca_eigvals > 0]
        if eig_pos.size >= 2:
            ranks = np.arange(1, eig_pos.size + 1)
            slope, intercept = np.polyfit(np.log(ranks), np.log(eig_pos), 1)
            powerlaw_alpha = -slope  # eig ~ r^{-alpha}
            powerlaw_intercept = intercept

        # Train/test split to see if dimensionality shifts
        rng = np.random.default_rng(123)
        trial_indices_nb = np.where(nonbaseline_mask)[0]
        perm = rng.permutation(trial_indices_nb)
        split = len(perm) // 2
        train_idx = perm[:split]
        test_idx = perm[split:]
        if split >= 2 and (len(perm) - split) >= 2:
            X_train = delta_per_trial_sub[:, train_idx].T
            X_test = delta_per_trial_sub[:, test_idx].T

            pca_train = PCA(svd_solver='full')
            pca_train.fit(X_train)
            evr_train = pca_train.explained_variance_ratio_
            pr_train = participation_ratio(pca_train.explained_variance_)

            pca_test = PCA(svd_solver='full')
            pca_test.fit(X_test)
            evr_test = pca_test.explained_variance_ratio_
            pr_test = participation_ratio(pca_test.explained_variance_)

        # Participation ratio per SF (non-gray only)
        for sf_idx, sf in enumerate(unique_spatialfreqs):
            if sf == 0:
                continue
            trials_sf_orig = np.where(spatialfreq_history == sf)[0]
            trials_sf = map_to_kept(trials_sf_orig)
            if trials_sf.size < 2:
                continue
            X_sf = delta_per_trial_sub[:, trials_sf].T
            pca_sf = PCA(svd_solver='full')
            pca_sf.fit(X_sf)
            pr_by_sf[sf_idx] = participation_ratio(pca_sf.explained_variance_)

    # Compute normalized metrics for cross-dataset comparison
    # Normalized PR: PR / N_neurons (ranges 0-1, where 1 = maximally high-D)
    pr_all_normalized = pr_all / n_neurons_used if (np.isfinite(pr_all) and n_neurons_used > 0) else np.nan
    pr_train_normalized = pr_train / n_neurons_used if (np.isfinite(pr_train) and n_neurons_used > 0) else np.nan
    pr_test_normalized = pr_test / n_neurons_used if (np.isfinite(pr_test) and n_neurons_used > 0) else np.nan
    pr_by_sf_normalized = pr_by_sf / n_neurons_used if n_neurons_used > 0 else np.full_like(pr_by_sf, np.nan)
    
    # Dimensionality ratio: knee_pc / N_neurons (fraction of neurons needed for 80% variance)
    knee_ratio = knee_pc / n_neurons_used if (np.isfinite(knee_pc) and n_neurons_used > 0) else np.nan
    
    # Effective dimensionality percentage: (PR / N) * 100
    effective_dim_pct = pr_all_normalized * 100 if np.isfinite(pr_all_normalized) else np.nan

    # Save dimensionality outputs with normalized metrics
    np.savez(
        os.path.join(save_path, 'pca_dimensionality_summary.npz'),
        eigvals=pca_eigvals,
        explained_variance_ratio=pca_explained_ratio,
        cumulative_explained_variance_ratio=pca_cum_ratio,
        participation_ratio_all=pr_all,
        powerlaw_alpha=powerlaw_alpha,
        powerlaw_intercept=powerlaw_intercept,
        participation_ratio_train=pr_train,
        participation_ratio_test=pr_test,
        explained_variance_ratio_train=evr_train,
        explained_variance_ratio_test=evr_test,
        participation_ratio_by_sf=pr_by_sf,
        unique_spatialfreqs=unique_spatialfreqs,
        knee_pc_80pct=knee_pc,
        n_neurons_total=n_neurons,
        n_neurons_used=n_neurons_used,
        neuron_subset_idx=neuron_subset_idx if neuron_subset_idx is not None else np.array([], dtype=int),
        n_trials_nonbaseline=int(np.sum(nonbaseline_mask)),
        # Normalized metrics for cross-dataset comparison
        participation_ratio_all_normalized=pr_all_normalized,
        participation_ratio_train_normalized=pr_train_normalized,
        participation_ratio_test_normalized=pr_test_normalized,
        participation_ratio_by_sf_normalized=pr_by_sf_normalized,
        knee_ratio=knee_ratio,
        effective_dimensionality_pct=effective_dim_pct,
    )
    
    # Print normalized metrics for easy comparison
    print(f"\n--- Normalized Dimensionality Metrics (for cross-dataset comparison) ---")
    print(f"  Participation Ratio (raw): {pr_all:.2f} / {n_neurons_used} neurons (subset of {n_neurons})")
    print(f"  Participation Ratio (normalized): {pr_all_normalized:.4f} (PR/N)")
    print(f"  Effective Dimensionality: {effective_dim_pct:.2f}% of neurons")
    print(f"  Knee Ratio: {knee_ratio:.4f} (Knee PC / N)")
    print(f"  Power-law α: {powerlaw_alpha:.4f} (neuron-count independent)")
    print(f"------------------------------------------------------------------------")

    # Plots: scree (EV ratio) and cumulative for all non-gray trials
    if pca_explained_ratio.size > 0:
        fig, ax1 = plt.subplots(figsize=(6, 4))
        comps = np.arange(1, pca_explained_ratio.size + 1)
        ax1.plot(comps, pca_explained_ratio, '-o', color='C0', label='Explained variance ratio')
        ax1.set_xlabel('PC index')
        ax1.set_ylabel('Explained variance ratio')
        ax1.tick_params(axis='y', labelcolor='C0')

        ax2 = ax1.twinx()
        ax2.plot(comps, pca_cum_ratio, '-o', color='C1', label='Cumulative')
        ax2.set_ylabel('Cumulative explained variance', color='C1')
        ax2.tick_params(axis='y', labelcolor='C1')
        
        # Add knee marker and annotation
        if np.isfinite(knee_pc):
            ax2.axvline(knee_pc, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
            ax2.text(0.98, 0.02, f'Knee: PC{knee_pc} ({knee_ratio:.2%} of N)\nPR: {pr_all:.1f} (norm={pr_all_normalized:.3f})\nEff.Dim: {effective_dim_pct:.1f}%',
                    transform=ax2.transAxes, fontsize=9, verticalalignment='bottom',
                    horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax1.set_title('PCA variance spectrum (non-gray trials)')
        fig.tight_layout()
        plt.savefig(os.path.join(save_path, 'pca_variance_spectrum_scree.svg'), format='svg')
        plt.close()

    # Plot: eigenvalue spectrum with power-law fit
    eig_pos = pca_eigvals[pca_eigvals > 0]
    if eig_pos.size >= 2:
        ranks = np.arange(1, eig_pos.size + 1)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ranks, eig_pos, 'o', color='C0', label='Eigenvalues')
        ax.set_xscale('log')
        ax.set_yscale('log')
        if np.isfinite(powerlaw_alpha):
            fit_vals = np.exp(powerlaw_intercept) * (ranks ** (-powerlaw_alpha))
            ax.plot(ranks, fit_vals, '-', color='C2', label=f'Power-law fit (alpha={powerlaw_alpha:.2f})')
            # Add text box with alpha and normalized PR
            ax.text(0.02, 0.02, f'α = {powerlaw_alpha:.3f} (N-independent)\nPR = {pr_all:.1f} (norm={pr_all_normalized:.3f})\nEff.Dim = {effective_dim_pct:.1f}%',
                   transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        ax.set_xlabel('Rank')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('Eigenvalue spectrum (log-log)')
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'pca_eigenvalue_powerlaw.svg'), format='svg')
        plt.close()

    # Plot: participation ratio split (all vs train/test)
    bars = []
    labels = []
    colors = []
    if np.isfinite(pr_all):
        bars.append(pr_all)
        labels.append('All')
        colors.append('C0')
    if np.isfinite(pr_train):
        bars.append(pr_train)
        labels.append('Train')
        colors.append('C1')
    if np.isfinite(pr_test):
        bars.append(pr_test)
        labels.append('Test')
        colors.append('C2')
    if bars:
        fig, ax = plt.subplots(figsize=(5, 4))
        bar_plot = ax.bar(np.arange(len(bars)), bars, color=colors)
        ax.set_xticks(np.arange(len(bars)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Participation ratio')
        ax.set_title(f'Dimensionality (PR, N_used={n_neurons_used}, N_total={n_neurons})')
        # Add value labels on bars with normalized values
        norm_vals = [v/n_neurons for v in bars]
        for i, (bar, val, norm_val) in enumerate(zip(bar_plot, bars, norm_vals)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}\n({norm_val:.3f})', ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'pca_participation_ratio_split.svg'), format='svg')
        plt.close()

    # Plot: variance allocation comparison all vs train/test
    if pca_explained_ratio.size > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        comps = np.arange(1, pca_explained_ratio.size + 1)
        ax.plot(comps, pca_explained_ratio, '-o', color='C0', label='All')
        if evr_train.size > 0:
            comps_t = np.arange(1, evr_train.size + 1)
            ax.plot(comps_t, evr_train, '-o', color='C1', label='Train')
        if evr_test.size > 0:
            comps_te = np.arange(1, evr_test.size + 1)
            ax.plot(comps_te, evr_test, '-o', color='C2', label='Test')
        ax.set_xlabel('PC index')
        ax.set_ylabel('Explained variance ratio')
        ax.set_title('Variance allocation (all vs train/test)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'pca_variance_spectrum_train_test.svg'), format='svg')
        plt.close()

    # Plot: participation ratio by spatial frequency (non-gray)
    if np.any(~np.isnan(pr_by_sf)):
        x_sf = np.arange(len(unique_spatialfreqs))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x_sf, pr_by_sf, '-o', color='C3')
        ax.set_xticks(x_sf)
        ax.set_xticklabels([f"{sf:.4g}" for sf in unique_spatialfreqs], rotation=45)
        ax.set_ylabel('Participation ratio')
        ax.set_title('Dimensionality by Spatial Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'pca_participation_ratio_by_sf.svg'), format='svg')
        plt.close()

    # ------------------------------------------------------------------
    # 3D manifold visualization: population activity in PC space
    # - Project condition means onto top 3 PCs
    # - Visualize orientation and SF manifolds
    # ------------------------------------------------------------------

    if pca_eigvals.size >= 3 and condition_means_sub.shape[1] > 0:
        from mpl_toolkits.mplot3d import Axes3D
        
        # Project condition means onto top 3 PCs
        pca_transform = PCA(n_components=3, svd_solver='full')
        condition_means_centered = condition_means_sub - np.mean(condition_means_sub, axis=1, keepdims=True)
        pc_projections = pca_transform.fit_transform(condition_means_centered.T)  # (n_conditions, 3)
        
        # Extract orientation and SF labels for each condition
        condition_oris = np.array([ori for ori, sf in condition_keys])
        condition_sfs = np.array([sf for ori, sf in condition_keys])
        
        # 3D plot colored by orientation
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color map for orientations
        unique_oris = np.unique(condition_oris)
        ori_colors = plt.cm.hsv(np.linspace(0, 1, len(unique_oris) + 1)[:-1])
        
        for ori_idx, ori in enumerate(unique_oris):
            mask = condition_oris == ori
            ax.plot(pc_projections[mask, 0], pc_projections[mask, 1], pc_projections[mask, 2],
                   'o-', color=ori_colors[ori_idx], label=f'{ori:.0f}°', markersize=6, alpha=0.7)
        
        ax.set_xlabel(f'PC1 ({pca_transform.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca_transform.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_zlabel(f'PC3 ({pca_transform.explained_variance_ratio_[2]*100:.1f}%)')
        ax.set_title('Population manifold: Orientation coding\n(trajectories across SF)')
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1, fontsize=8, title='Orientation')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'pca_manifold_3d_orientation.svg'), format='svg', bbox_inches='tight')
        plt.close()
        
        # 3D plot colored by spatial frequency
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color map for SFs (log scale coloring)
        unique_sfs_sorted = np.sort(np.unique(condition_sfs))
        sf_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_sfs_sorted)))
        sf_color_map = {sf: sf_colors[i] for i, sf in enumerate(unique_sfs_sorted)}
        
        for sf_idx, sf in enumerate(unique_sfs_sorted):
            mask = condition_sfs == sf
            ax.plot(pc_projections[mask, 0], pc_projections[mask, 1], pc_projections[mask, 2],
                   'o-', color=sf_color_map[sf], label=f'SF {sf:.2f}', markersize=6, alpha=0.7)
        
        ax.set_xlabel(f'PC1 ({pca_transform.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca_transform.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_zlabel(f'PC3 ({pca_transform.explained_variance_ratio_[2]*100:.1f}%)')
        ax.set_title('Population manifold: Spatial Frequency coding\n(trajectories across orientation)')
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1, fontsize=8, title='SF')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'pca_manifold_3d_sf.svg'), format='svg', bbox_inches='tight')
        plt.close()
        
        # Combined view: orientation as color, SF as marker size
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Normalize SF for marker sizes (larger SF = larger marker)
        sf_norm = (condition_sfs - condition_sfs.min()) / (condition_sfs.max() - condition_sfs.min() + 1e-10)
        marker_sizes = 20 + 80 * sf_norm
        
        for ori_idx, ori in enumerate(unique_oris):
            mask = condition_oris == ori
            ax.scatter(pc_projections[mask, 0], pc_projections[mask, 1], pc_projections[mask, 2],
                      c=[ori_colors[ori_idx]], s=marker_sizes[mask], alpha=0.7, 
                      label=f'{ori:.0f}°', edgecolors='black', linewidths=0.5)
        
        ax.set_xlabel(f'PC1 ({pca_transform.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca_transform.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_zlabel(f'PC3 ({pca_transform.explained_variance_ratio_[2]*100:.1f}%)')
        ax.set_title('Population manifold: Combined view\n(color=orientation, size=SF)')
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1, fontsize=8, title='Orientation')
        
        # Add text annotation for marker size meaning
        ax.text2D(0.02, 0.98, 'Marker size ∝ Spatial Frequency', transform=ax.transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'pca_manifold_3d_combined.svg'), format='svg', bbox_inches='tight')
        plt.close()
        
        # Save PC projections for further analysis
        np.savez(
            os.path.join(save_path, 'pca_manifold_projections.npz'),
            pc_projections=pc_projections,
            condition_orientations=condition_oris,
            condition_spatialfreqs=condition_sfs,
            explained_variance_ratio_top3=pca_transform.explained_variance_ratio_,
        )

    # ------------------------------------------------------------------
    # dsPCA: Demixed Subspace Principal Component Analysis
    # - Separate orientation-specific vs SF-specific variance components
    # - Identify target axes for orientation and SF
    # - Find target-free subspace orthogonal to task parameters
    # ------------------------------------------------------------------

    print("\n" + "="*60)
    print("Running dsPCA (Demixed PCA) Analysis")
    print("="*60)
    
    # Prepare data: condition_means.T gives (n_conditions × n_neurons)
    dspca_data = condition_means_sub.T  # Shape: (n_conditions, n_neurons_used)

    # Prepare targets (RAW, for labeling/masks): [SF_code, sin(2θ), cos(2θ)] for each condition
    # - SF_code is log2(mapped_sf) z-scored across nonzero SF levels; baseline (SF<=0) kept at 0.
    # - Orientation uses a circular basis (sin/cos of 2θ) for 180° periodicity.
    mapped_sfs_by_level = [float(convert_sf_value(sf)) for sf in unique_spatialfreqs]
    raw_log_sf_by_level = [float(np.log2(msf)) if msf > 0 else 0.0 for msf in mapped_sfs_by_level]
    nonzero_raw = np.array([v for v, msf in zip(raw_log_sf_by_level, mapped_sfs_by_level) if msf > 0], dtype=float)
    if nonzero_raw.size:
        sf_mu = float(np.mean(nonzero_raw))
        sf_sigma = float(np.std(nonzero_raw, ddof=0))
    else:
        sf_mu = 0.0
        sf_sigma = 1.0
    if not np.isfinite(sf_sigma) or sf_sigma <= 0:
        sf_sigma = 1.0
    sf_code_by_level = []
    for msf, raw_log in zip(mapped_sfs_by_level, raw_log_sf_by_level):
        if msf > 0:
            sf_code_by_level.append(float((raw_log - sf_mu) / sf_sigma))
        else:
            sf_code_by_level.append(0.0)

    # Build raw targets in the same ordering as condition_keys
    # Also track mapped SF per condition (linear) for downstream metrics.
    targets_dspca_raw = np.zeros((len(condition_keys), 3), dtype=float)
    mapped_sf_targets = np.zeros(len(condition_keys), dtype=float)
    for i, (ori, sf) in enumerate(condition_keys):
        mapped_sf = float(convert_sf_value(sf))
        mapped_sf_targets[i] = mapped_sf
        # map sf to its level index (nearest match)
        if mapped_sf > 0:
            sf_idx = int(np.argmin([abs(mapped_sf - v) for v in mapped_sfs_by_level]))
            targets_dspca_raw[i, 0] = float(sf_code_by_level[sf_idx])
        else:
            targets_dspca_raw[i, 0] = 0.0
        theta = np.deg2rad(float(ori))
        targets_dspca_raw[i, 1] = float(np.sin(2 * theta))
        targets_dspca_raw[i, 2] = float(np.cos(2 * theta))

    # Z-score ALL regressors for dsPCA fitting (keeps dsPCA weighting balanced across regressors)
    mu = np.nanmean(targets_dspca_raw, axis=0)
    sigma = np.nanstd(targets_dspca_raw, axis=0, ddof=0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)
    targets_dspca_fit = (targets_dspca_raw - mu) / sigma
    
    # Run dsPCA
    print(f"Data shape: {dspca_data.shape} (conditions × neurons_used)")
    print(f"Targets shape: {targets_dspca_fit.shape} (conditions × [SF_code_z, sin_z, cos_z])")
    
    projection_target, projection_targetfree, ax_targets, ax_targetfree, \
    target_signal, targetfree_signal, target_var, targetfree_var, total_var, dot_target_ax = \
        dsPCA(data=dspca_data, targets=targets_dspca_fit)

    # ------------------------------------------------------------------
    # dsPCA CV: fit 10 train/test splits, align held-out projections, and
    # aggregate all held-out conditions so each trial contributes once.
    # ------------------------------------------------------------------
    projection_target_test = None
    projection_targetfree_test = None
    projection_target_train = None
    projection_targetfree_train = None
    train_center = None
    try:
        min_trials_dspca = np.inf
        for trials_kept in condition_trial_indices:
            trials_kept = np.asarray(trials_kept, dtype=int)
            trials_kept = trials_kept[np.isfinite(trials_kept)]
            if trials_kept.size > 0:
                min_trials_dspca = min(min_trials_dspca, trials_kept.size)

        if (not np.isfinite(min_trials_dspca)) or (int(min_trials_dspca) < CV_N_FOLDS):
            found_trials = int(min_trials_dspca) if np.isfinite(min_trials_dspca) else 0
            print(f"  [CV10] Skipped dsPCA CV: need >= {CV_N_FOLDS} trials per non-baseline condition, found {found_trials}")
        else:
            rng_cv = np.random.default_rng(int(4242) ^ int(zlib.crc32(str(save_path).encode('utf-8'))))
            condition_fold_splits = [
                _split_trials_into_cv_folds(np.asarray(trials_kept, dtype=int), n_folds=CV_N_FOLDS, rng=rng_cv)
                for trials_kept in condition_trial_indices
            ]

            projection_target_train_folds = []
            projection_targetfree_train_folds = []
            projection_target_test_folds = []
            projection_targetfree_test_folds = []
            fold_train_counts = []
            fold_test_counts = []
            target_var_folds = []
            targetfree_var_folds = []
            total_var_folds = []
            train_center_folds = []
            ref_ax_targets = np.asarray(ax_targets, dtype=float)
            ref_ax_targetfree = np.asarray(ax_targetfree, dtype=float)

            for fold_idx in range(CV_N_FOLDS):
                fold_train_means = []
                fold_test_means = []
                fold_train_count = []
                fold_test_count = []
                cv_ok = True

                for cond_idx, trials_kept in enumerate(condition_trial_indices):
                    trials_kept = np.asarray(trials_kept, dtype=int)
                    test_idx = np.asarray(condition_fold_splits[cond_idx][fold_idx], dtype=int)
                    train_idx = np.setdiff1d(trials_kept, test_idx, assume_unique=False)
                    if train_idx.size == 0 or test_idx.size == 0:
                        cv_ok = False
                        break
                    fold_train_means.append(np.nanmean(delta_per_trial_sub[:, train_idx], axis=1))
                    fold_test_means.append(np.nanmean(delta_per_trial_sub[:, test_idx], axis=1))
                    fold_train_count.append(int(train_idx.size))
                    fold_test_count.append(int(test_idx.size))

                if not cv_ok:
                    continue

                dspca_data_train = np.stack(fold_train_means, axis=1).T
                dspca_data_test = np.stack(fold_test_means, axis=1).T
                pt_tr, ptf_tr, ax_tr, ax_tf, _ts_tr, _tfs_tr, _tv_tr, _tfv_tr, _tot_tr, _dot_tr = dsPCA(
                    data=dspca_data_train,
                    targets=targets_dspca_fit,
                )
                fold_center = np.nanmean(dspca_data_train, axis=0, keepdims=True)
                pt_te = (dspca_data_test - fold_center) @ ax_tr
                ptf_te = (dspca_data_test - fold_center) @ ax_tf

                projection_target_train_folds.append(
                    _align_target_projection_to_reference(pt_tr, ax_tr, ref_ax_targets)
                )
                projection_target_test_folds.append(
                    _align_target_projection_to_reference(pt_te, ax_tr, ref_ax_targets)
                )
                projection_targetfree_train_folds.append(
                    _align_component_projection_signs(ptf_tr, ax_tf, ref_ax_targetfree)
                )
                projection_targetfree_test_folds.append(
                    _align_component_projection_signs(ptf_te, ax_tf, ref_ax_targetfree)
                )
                fold_train_counts.append(np.asarray(fold_train_count, dtype=float))
                fold_test_counts.append(np.asarray(fold_test_count, dtype=float))
                train_center_folds.append(fold_center)

            if projection_target_test_folds:
                projection_target_train_folds_arr = np.stack(projection_target_train_folds, axis=0)
                projection_targetfree_train_folds_arr = np.stack(projection_targetfree_train_folds, axis=0)
                projection_target_test_folds_arr = np.stack(projection_target_test_folds, axis=0)
                projection_targetfree_test_folds_arr = np.stack(projection_targetfree_test_folds, axis=0)
                fold_train_counts_arr = np.stack(fold_train_counts, axis=0)
                fold_test_counts_arr = np.stack(fold_test_counts, axis=0)
                train_center = np.nanmean(np.concatenate(train_center_folds, axis=0), axis=0, keepdims=True)

                projection_target_train = np.divide(
                    np.nansum(projection_target_train_folds_arr * fold_train_counts_arr[:, :, None], axis=0),
                    np.sum(fold_train_counts_arr, axis=0)[:, None],
                    out=np.full_like(projection_target_train_folds_arr[0], np.nan, dtype=float),
                    where=np.sum(fold_train_counts_arr, axis=0)[:, None] > 0,
                )
                projection_targetfree_train = np.divide(
                    np.nansum(projection_targetfree_train_folds_arr * fold_train_counts_arr[:, :, None], axis=0),
                    np.sum(fold_train_counts_arr, axis=0)[:, None],
                    out=np.full_like(projection_targetfree_train_folds_arr[0], np.nan, dtype=float),
                    where=np.sum(fold_train_counts_arr, axis=0)[:, None] > 0,
                )
                projection_target_test = np.divide(
                    np.nansum(projection_target_test_folds_arr * fold_test_counts_arr[:, :, None], axis=0),
                    np.sum(fold_test_counts_arr, axis=0)[:, None],
                    out=np.full_like(projection_target_test_folds_arr[0], np.nan, dtype=float),
                    where=np.sum(fold_test_counts_arr, axis=0)[:, None] > 0,
                )
                projection_targetfree_test = np.divide(
                    np.nansum(projection_targetfree_test_folds_arr * fold_test_counts_arr[:, :, None], axis=0),
                    np.sum(fold_test_counts_arr, axis=0)[:, None],
                    out=np.full_like(projection_targetfree_test_folds_arr[0], np.nan, dtype=float),
                    where=np.sum(fold_test_counts_arr, axis=0)[:, None] > 0,
                )
                target_var_cv = np.var(projection_target_test, axis=0, ddof=0)
                targetfree_var_cv = np.var(projection_targetfree_test, axis=0, ddof=0)
                total_var_cv = float(np.sum(np.var(dspca_data, axis=0, ddof=0)))

                np.savez(
                    os.path.join(save_path, 'dspca_results_cv50.npz'),
                    projection_target_train=projection_target_train,
                    projection_targetfree_train=projection_targetfree_train,
                    projection_target_test=projection_target_test,
                    projection_targetfree_test=projection_targetfree_test,
                    projection_target_train_folds=projection_target_train_folds_arr,
                    projection_targetfree_train_folds=projection_targetfree_train_folds_arr,
                    projection_target_test_folds=projection_target_test_folds_arr,
                    projection_targetfree_test_folds=projection_targetfree_test_folds_arr,
                    fold_train_counts=fold_train_counts_arr,
                    fold_test_counts=fold_test_counts_arr,
                    n_folds=int(CV_N_FOLDS),
                    ax_targets=ref_ax_targets,
                    ax_targetfree=ref_ax_targetfree,
                    target_var=target_var_cv,
                    targetfree_var=targetfree_var_cv,
                    total_var=total_var_cv,
                    targets=targets_dspca_raw,
                    targets_fit=targets_dspca_fit,
                    targets_fit_stats=np.stack([mu, sigma], axis=0),
                    train_center=train_center,
                    condition_orientations=np.array([ori for ori, sf in condition_keys]),
                    condition_spatialfreqs=mapped_sf_targets,
                    n_neurons_total=n_neurons,
                    n_neurons_used=n_neurons_used,
                    neuron_subset_idx=neuron_subset_idx if neuron_subset_idx is not None else np.array([], dtype=int),
                    cv_note='10-fold CV: fit each 9/10 split, align held-out projections to the full-data reference basis, aggregate all held-out folds',
                )
                print("  [CV10] Saved dspca_results_cv50.npz")
            else:
                print("  [CV10] Skipped dsPCA CV: no valid folds were produced")
    except Exception as e:
        print(f"  [CV10] Failed: {type(e).__name__}: {e}")
    
    # Print summary statistics
    print(f"\nDemixed Variance Explained:")
    sf_pct = 100 * target_var[0] / total_var
    ori_pct = 100 * (target_var[1] + target_var[2]) / total_var
    print(f"  SF axis: {sf_pct:.2f}%")
    print(f"  Orientation (sin+cos): {ori_pct:.2f}%")
    print(f"  Total target variance: {100*np.sum(target_var)/total_var:.2f}%")
    
    n_targetfree_show = min(10, len(targetfree_var))
    print(f"  Top {n_targetfree_show} target-free dims: {100*targetfree_var[:n_targetfree_show]/total_var}")
    print(f"  Total target-free variance (top {n_targetfree_show}): {100*np.sum(targetfree_var[:n_targetfree_show])/total_var:.2f}%")
    
    print(f"\nAxis Orthogonality:")
    print(f"  SF·sin: {dot_target_ax[0,1]:.6f}")
    print(f"  SF·cos: {dot_target_ax[0,2]:.6f}")
    print(f"  sin·cos: {dot_target_ax[1,2]:.6f}")

    print(f"\nTarget Signal Correlations:")
    print(f"  SF axis with SF target: r={target_signal[0,0,0]:.4f}, p={target_signal[0,0,1]:.4e}")
    print(f"  SF axis with sin target: r={target_signal[0,1,0]:.4f}, p={target_signal[0,1,1]:.4e}")
    print(f"  SF axis with cos target: r={target_signal[0,2,0]:.4f}, p={target_signal[0,2,1]:.4e}")
    print(f"  sin axis with sin target: r={target_signal[1,1,0]:.4f}, p={target_signal[1,1,1]:.4e}")
    print(f"  sin axis with cos target: r={target_signal[1,2,0]:.4f}, p={target_signal[1,2,1]:.4e}")
    print(f"  cos axis with sin target: r={target_signal[2,1,0]:.4f}, p={target_signal[2,1,1]:.4e}")
    print(f"  cos axis with cos target: r={target_signal[2,2,0]:.4f}, p={target_signal[2,2,1]:.4e}")
    
    # Compute normalized dsPCA metrics for cross-dataset comparison
    # Note: Variance percentages are already normalized by total_var, so they're comparable
    # Orthogonality (dot product) is also normalized (unit vectors), so it's comparable
    print(f"\n--- dsPCA Metrics (for cross-dataset comparison) ---")
    print(f"  Dataset: N_used={n_neurons_used} (of N_total={n_neurons}) neurons, {len(condition_keys)} conditions")
    print(f"  Target variance % (Ori+SF): Neuron-count INDEPENDENT (normalized by total variance)")
    print(f"  Orthogonality (Ori·SF): Neuron-count INDEPENDENT (unit vectors)")
    print(f"  Correlation coefficients: Neuron-count INDEPENDENT (Pearson r)")
    print(f"  --> All dsPCA metrics shown are comparable across datasets with different N")
    print(f"-----------------------------------------------------")
    
    # Plot 1: Variance explained by each axis (SF, sin, cos, target-free)
    fig, ax = plt.subplots(figsize=(10, 5))

    n_targetfree_plot = min(10, len(targetfree_var))
    x_positions = np.arange(3 + n_targetfree_plot)

    ax.bar([0], [sf_pct], color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.5, label='SF')
    ax.bar([1, 2], 100 * target_var[1:3] / total_var, color=['#d62728', '#ff7f0e'], alpha=0.8, edgecolor='black', linewidth=1.5, label='Orientation (sin/cos)')

    ax.bar(np.arange(3, 3 + n_targetfree_plot),
           100 * targetfree_var[:n_targetfree_plot] / total_var,
           color='gray', alpha=0.6, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(['SF', 'sin', 'cos'] + [f'TF{i+1}' for i in range(n_targetfree_plot)])
    ax.set_ylabel('% Variance Explained (N-independent)', fontsize=12)
    ax.set_xlabel('dsPCA Axes', fontsize=12)
    ax.set_title(f'dsPCA Variance Decomposition (N_used={n_neurons_used}, N_total={n_neurons})\n(Target vs Target-Free Subspaces)', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for i, var_val in enumerate(target_var):
        ax.text(i, 100*var_val/total_var + 0.5, f'{100*var_val/total_var:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    total_target_pct = 100 * np.sum(target_var) / total_var
    total_targetfree_pct = 100 * np.sum(targetfree_var[:n_targetfree_plot]) / total_var
    textstr = f'Target (SF+sin+cos): {total_target_pct:.1f}%\nTarget-free (top {n_targetfree_plot}): {total_targetfree_pct:.1f}%\nSF·sin: {dot_target_ax[0,1]:.4f}\nSF·cos: {dot_target_ax[0,2]:.4f}\nsin·cos: {dot_target_ax[1,2]:.4f}\n\n✓ All metrics N-independent'
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=1))

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'dspca_variance_explained.svg'), format='svg', bbox_inches='tight')
    plt.close()

    # Build orientation recovery from circular targets
    ori_from_targets = np.arctan2(targets_dspca_raw[:, 1], targets_dspca_raw[:, 2]) / 2
    ori_from_targets = np.rad2deg(ori_from_targets)
    ori_from_targets = np.mod(ori_from_targets, 180)
    unique_orientations = np.unique(np.round(ori_from_targets / 30.0) * 30.0)
    mapped_sfs = [convert_sf_value(sf) for sf in np.unique([sf for _, sf in condition_keys]) if convert_sf_value(sf) > 0]
    mapped_sfs = np.sort(np.unique(mapped_sfs))
    log_ticks = [np.log10(sf) for sf in mapped_sfs]
    tick_labels = [f"{sf:.2f}" for sf in mapped_sfs]

    # Precompute orientation-sorted data
    ori_colors = plt.cm.turbo(np.linspace(0, 1, len(unique_orientations) + 1)[:-1])
    orientation_data = []
    for ori_idx, ori in enumerate(unique_orientations):
        mask = np.abs(ori_from_targets - ori) < 1
        if np.sum(mask) > 0:
            proj_sf = projection_target[mask, 0]
            proj_sin = projection_target[mask, 1]
            proj_cos = projection_target[mask, 2]
            sort_idx = np.argsort(proj_sf)
            orientation_data.append({
                "ori_idx": ori_idx,
                "ori": ori,
                "sf": proj_sf[sort_idx],
                "sin": proj_sin[sort_idx],
                "cos": proj_cos[sort_idx],
            })

    # 12-panel target subspace visualization
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(3, 4)

    # Subplot 1: SF on Z, connect points per orientation
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    for entry in orientation_data:
        sf_vals = entry["sf"]
        proj_sin = entry["sin"]
        proj_cos = entry["cos"]

        nonzero = sf_vals != 0
        sf_norm = np.zeros_like(sf_vals)
        if np.any(nonzero):
            min_sf = sf_vals[nonzero].min()
            max_sf = sf_vals[nonzero].max()
            sf_norm[nonzero] = (sf_vals[nonzero] - min_sf) / (max_sf - min_sf + 1e-10)
        marker_sizes = 25 + 120 * sf_norm

        _, unique_idx = np.unique(sf_vals, return_index=True)
        for k in range(len(unique_idx) - 1):
            i = unique_idx[k]
            j = unique_idx[k+1]
            ax1.plot([proj_sin[i], proj_sin[j]], [proj_cos[i], proj_cos[j]], [sf_vals[i], sf_vals[j]],
                     '-', color=ori_colors[entry["ori_idx"]], alpha=0.4, linewidth=1.5)

        ax1.scatter(proj_sin, proj_cos, sf_vals,
                   c=[ori_colors[entry["ori_idx"]]], s=marker_sizes, alpha=0.7,
                   edgecolors='black', linewidths=0.5, label=f'{entry["ori"]:.0f}°')

    ax1.set_xlabel('sin(2θ)', fontsize=10)
    ax1.set_ylabel('cos(2θ)', fontsize=10)
    ax1.set_zlabel('SF (log10 mapped)', fontsize=10)
    ax1.set_title('Target Subspace (3D)\n[sin(2θ), cos(2θ), SF_log]', fontsize=11, fontweight='bold')
    ax1.set_zticks(log_ticks)
    ax1.set_zticklabels(tick_labels)
    ax1.legend(fontsize=7, ncol=2, loc='upper left', bbox_to_anchor=(0, 1))
    ax1.view_init(elev=20, azim=45)

    # Subplot 2: rotated view (+90° azimuth)
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    for entry in orientation_data:
        sf_vals = entry["sf"]
        proj_sin = entry["sin"]
        proj_cos = entry["cos"]
        nonzero = sf_vals != 0
        sf_norm = np.zeros_like(sf_vals)
        if np.any(nonzero):
            min_sf = sf_vals[nonzero].min()
            max_sf = sf_vals[nonzero].max()
            sf_norm[nonzero] = (sf_vals[nonzero] - min_sf) / (max_sf - min_sf + 1e-10)
        marker_sizes = 25 + 120 * sf_norm
        _, unique_idx = np.unique(sf_vals, return_index=True)
        for k in range(len(unique_idx) - 1):
            i = unique_idx[k]
            j = unique_idx[k+1]
            ax2.plot([proj_sin[i], proj_sin[j]], [proj_cos[i], proj_cos[j]], [sf_vals[i], sf_vals[j]],
                     '-', color=ori_colors[entry["ori_idx"]], alpha=0.4, linewidth=1.5)
        ax2.scatter(proj_sin, proj_cos, sf_vals,
                   c=[ori_colors[entry["ori_idx"]]], s=marker_sizes, alpha=0.7,
                   edgecolors='black', linewidths=0.5, label=f'{entry["ori"]:.0f}°')
    ax2.set_xlabel('sin(2θ)', fontsize=10)
    ax2.set_ylabel('cos(2θ)', fontsize=10)
    ax2.set_zlabel('SF (log10 mapped)', fontsize=10)
    ax2.set_title('Target Subspace (3D, rotated)', fontsize=11, fontweight='bold')
    ax2.set_zticks(log_ticks)
    ax2.set_zticklabels(tick_labels)
    ax2.legend(fontsize=7, ncol=2, loc='upper left', bbox_to_anchor=(0, 1))
    ax2.view_init(elev=20, azim=135)

    # Subplot 3: Mixed subspace (sin, cos, +1 target-free PC)
    if projection_targetfree.shape[1] >= 1:
        ax3 = fig.add_subplot(gs[0, 2], projection='3d')
        for ori_idx, ori in enumerate(unique_orientations):
            mask = np.abs(ori_from_targets - ori) < 1
            if np.sum(mask) > 0:
                sf_vals_log = np.where(mapped_sf_targets[mask] > 0, np.log10(mapped_sf_targets[mask]), 0.0)
                sort_idx = np.argsort(sf_vals_log)
                proj_sf_sorted = projection_target[mask, 0][sort_idx]
                proj_sin_sorted = projection_target[mask, 1][sort_idx]
                proj_cos_sorted = projection_target[mask, 2][sort_idx]
                proj_tf1_sorted = projection_targetfree[mask, 0][sort_idx]
                sf_sorted = sf_vals_log[sort_idx]
                _, unique_idx = np.unique(sf_sorted, return_index=True)
                for k in range(len(unique_idx) - 1):
                    i = unique_idx[k]; j = unique_idx[k+1]
                    ax3.plot([proj_sin_sorted[i], proj_sin_sorted[j]],
                             [proj_cos_sorted[i], proj_cos_sorted[j]],
                             [proj_tf1_sorted[i], proj_tf1_sorted[j]],
                             '-', color=ori_colors[ori_idx], alpha=0.4, linewidth=1.5)
                sf_log_sorted = sf_vals_log[sort_idx]
                nonzero = sf_log_sorted != 0
                sf_norm = np.zeros_like(sf_log_sorted)
                if np.any(nonzero):
                    min_sf = sf_log_sorted[nonzero].min()
                    max_sf = sf_log_sorted[nonzero].max()
                    sf_norm[nonzero] = (sf_log_sorted[nonzero] - min_sf) / (max_sf - min_sf + 1e-10)
                marker_sizes = 20 + 100 * sf_norm
                ax3.scatter(proj_sin_sorted, proj_cos_sorted, proj_tf1_sorted,
                           c=[ori_colors[ori_idx]], s=marker_sizes, alpha=0.7,
                           edgecolors='black', linewidths=0.5, label=f'{ori:.0f}°')
        tf_var_pct = 100*targetfree_var[0]/total_var if len(targetfree_var) > 0 else 0
        ax3.set_xlabel('sin(2θ)', fontsize=10)
        ax3.set_ylabel('cos(2θ)', fontsize=10)
        ax3.set_zlabel(f'Target-Free PC1\n({tf_var_pct:.1f}%)', fontsize=10)
        ax3.set_title('Mixed Subspace\n[sin, cos, +1 PC]', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=7, ncol=2, loc='upper left', bbox_to_anchor=(0, 1))
        ax3.view_init(elev=20, azim=45)

    # Subplot 4: Orientation plane (sin vs cos) sized by log-SF
    ax4 = fig.add_subplot(gs[0, 3])
    for ori_idx, ori in enumerate(unique_orientations):
        mask = np.abs(ori_from_targets - ori) < 1
        if np.sum(mask) > 0:
            proj_sin = projection_target[mask, 1]
            proj_cos = projection_target[mask, 2]
            sf_log = np.where(mapped_sf_targets[mask] > 0, np.log10(mapped_sf_targets[mask]), 0.0)
            sort_idx = np.argsort(sf_log)
            proj_sin = proj_sin[sort_idx]
            proj_cos = proj_cos[sort_idx]
            sf_log = sf_log[sort_idx]
            nonzero = sf_log != 0
            sf_norm = np.zeros_like(sf_log)
            if np.any(nonzero):
                min_sf = sf_log[nonzero].min()
                max_sf = sf_log[nonzero].max()
                sf_norm[nonzero] = (sf_log[nonzero] - min_sf) / (max_sf - min_sf + 1e-10)
            marker_sizes = 30 + 200 * sf_norm
            _, unique_idx = np.unique(sf_log, return_index=True)
            for k in range(len(unique_idx) - 1):
                i = unique_idx[k]; j = unique_idx[k+1]
                ax4.plot([proj_sin[i], proj_sin[j]], [proj_cos[i], proj_cos[j]],
                        '-', color=ori_colors[ori_idx], alpha=0.4, linewidth=1.5)
            ax4.scatter(proj_sin, proj_cos, c=[ori_colors[ori_idx]],
                       s=marker_sizes, alpha=0.6, label=f'{ori:.0f}°',
                       edgecolors='black', linewidths=0.5)
    circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', color='gray', alpha=0.2, linewidth=1)
    ax4.add_patch(circle)
    ax4.set_xlabel('sin(2θ)', fontsize=10)
    ax4.set_ylabel('cos(2θ)', fontsize=10)
    ax4.set_title('Orientation Plane (2D)\n[sin(2θ) vs cos(2θ)]', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=7, ncol=2, loc='best')
    ax4.set_aspect('equal')
    ax4.grid(alpha=0.3)

    # Subplot 5: 2D sin vs SF_log
    ax5 = fig.add_subplot(gs[1, 0])
    for entry in orientation_data:
        sf_vals = entry["sf"]
        proj_sin = entry["sin"]
        ax5.scatter(proj_sin, sf_vals, c=[ori_colors[entry["ori_idx"]]], s=50, alpha=0.7,
                   edgecolors='black', linewidths=0.5, label=f'{entry["ori"]:.0f}°')
        _, unique_idx = np.unique(sf_vals, return_index=True)
        for k in range(len(unique_idx) - 1):
            i = unique_idx[k]; j = unique_idx[k+1]
            ax5.plot([proj_sin[i], proj_sin[j]], [sf_vals[i], sf_vals[j]],
                    '-', color=ori_colors[entry["ori_idx"]], alpha=0.4, linewidth=1.0)
    ax5.set_xlabel('sin(2θ)', fontsize=10)
    ax5.set_ylabel('SF (log10 mapped)', fontsize=10)
    ax5.set_yticks(log_ticks)
    ax5.set_yticklabels(tick_labels)
    ax5.set_title('2D View: sin vs SF', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=7, ncol=2, loc='upper left', bbox_to_anchor=(0, 1))

    # Subplot 6: 2D cos vs SF_log
    ax6 = fig.add_subplot(gs[1, 1])
    for entry in orientation_data:
        sf_vals = entry["sf"]
        proj_cos = entry["cos"]
        ax6.scatter(proj_cos, sf_vals, c=[ori_colors[entry["ori_idx"]]], s=50, alpha=0.7,
                   edgecolors='black', linewidths=0.5, label=f'{entry["ori"]:.0f}°')
        _, unique_idx = np.unique(sf_vals, return_index=True)
        for k in range(len(unique_idx) - 1):
            i = unique_idx[k]; j = unique_idx[k+1]
            ax6.plot([proj_cos[i], proj_cos[j]], [sf_vals[i], sf_vals[j]],
                    '-', color=ori_colors[entry["ori_idx"]], alpha=0.4, linewidth=1.0)
    ax6.set_xlabel('cos(2θ)', fontsize=10)
    ax6.set_ylabel('SF (log10 mapped)', fontsize=10)
    ax6.set_yticks(log_ticks)
    ax6.set_yticklabels(tick_labels)
    ax6.set_title('2D View: cos vs SF', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=7, ncol=2, loc='upper left', bbox_to_anchor=(0, 1))

    # Subplots 7–12: per-SF orientation plane (cos vs sin)
    per_sf_axes = [fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3]),
                   fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]),
                   fig.add_subplot(gs[2, 2]), fig.add_subplot(gs[2, 3])]
    for ax_sf, sf_val in zip(per_sf_axes, mapped_sfs):
        mask = np.isclose(mapped_sf_targets, sf_val, rtol=0.0, atol=1e-9)
        if np.sum(mask) == 0:
            ax_sf.set_visible(False)
            continue
        proj_sin = projection_target[mask, 1]
        proj_cos = projection_target[mask, 2]
        ori_vals = ori_from_targets[mask]
        ori_bins = np.round(ori_vals / 30.0) * 30.0
        point_colors = []
        for ob in ori_bins:
            idx = int(np.argmin(np.abs(unique_orientations - ob)))
            point_colors.append(ori_colors[idx])
        order = np.argsort(ori_bins)
        proj_sin_sorted = proj_sin[order]; proj_cos_sorted = proj_cos[order]; ori_bins_sorted = ori_bins[order]
        _, unique_idx = np.unique(ori_bins_sorted, return_index=True)
        for k in range(len(unique_idx) - 1):
            i = unique_idx[k]; j = unique_idx[k+1]
            ax_sf.plot([proj_sin_sorted[i], proj_sin_sorted[j]], [proj_cos_sorted[i], proj_cos_sorted[j]],
                      '-', color='gray', alpha=0.4, linewidth=1.0)
        if len(unique_idx) > 1:
            i = unique_idx[-1]; j = unique_idx[0]
            ax_sf.plot([proj_sin_sorted[i], proj_sin_sorted[j]], [proj_cos_sorted[i], proj_cos_sorted[j]],
                      '-', color='gray', alpha=0.4, linewidth=1.0)
        ax_sf.scatter(proj_sin, proj_cos, c=point_colors, s=50, alpha=0.7,
                     edgecolors='black', linewidths=0.5)
        ax_sf.set_xlabel('sin(2θ)', fontsize=9)
        ax_sf.set_ylabel('cos(2θ)', fontsize=9)
        ax_sf.set_title(f'SF {sf_val:.2f}', fontsize=10)
        ax_sf.grid(alpha=0.2)

    fig.suptitle('dsPCA Target Subspace (Circular Basis)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'dspca_target_subspace_3d.svg'), format='svg', bbox_inches='tight')
    plt.close()
    
    # ========== Distance analyses per SF ==========
    # Distance from COM (6 subplots for SFs)
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
    axes1 = axes1.flatten()
    for i, sf_val in enumerate(mapped_sfs):
        ax = axes1[i]
        mask = np.isclose(mapped_sf_targets, sf_val, rtol=0.0, atol=1e-9)
        if not np.any(mask):
            ax.set_visible(False)
            continue
        sin_vals = projection_target[mask, 1]
        cos_vals = projection_target[mask, 2]
        ori_vals = ori_from_targets[mask]
        # Compute COM and distances
        com = np.array([np.nanmean(sin_vals), np.nanmean(cos_vals)])
        dist = np.sqrt((sin_vals - com[0])**2 + (cos_vals - com[1])**2)
        # Sort by orientation
        order = np.argsort(ori_vals)
        ax.plot(ori_vals[order], dist[order], '-o', color='black', linewidth=2, markersize=6)
        ax.set_xlabel('Orientation (deg)', fontsize=10)
        ax.set_ylabel('Distance to COM', fontsize=10)
        ax.set_title(f'SF {sf_val:.2f}', fontsize=11)
        ax.grid(alpha=0.3)
    fig1.suptitle('Distance to COM vs Orientation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'dspca_distance_to_com_by_sf.svg'), format='svg', bbox_inches='tight')
    plt.close()

    # Adjacent orientation distances (6 subplots for SFs)
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    axes2 = axes2.flatten()
    for i, sf_val in enumerate(mapped_sfs):
        ax = axes2[i]
        mask = np.isclose(mapped_sf_targets, sf_val, rtol=0.0, atol=1e-9)
        if not np.any(mask):
            ax.set_visible(False)
            continue
        sin_vals = projection_target[mask, 1]
        cos_vals = projection_target[mask, 2]
        ori_vals = ori_from_targets[mask]
        # Sort by orientation and wrap to include 0–150 connection
        order = np.argsort(ori_vals)
        sin_sorted = sin_vals[order]
        cos_sorted = cos_vals[order]
        ori_sorted = ori_vals[order]
        # Append wrap-around to close loop
        sin_wrap = np.append(sin_sorted, sin_sorted[0])
        cos_wrap = np.append(cos_sorted, cos_sorted[0])
        # Compute distances between consecutive points
        dists = np.sqrt(np.diff(sin_wrap)**2 + np.diff(cos_wrap)**2)
        ax.plot(np.arange(len(dists)), dists, '-o', color='black', linewidth=2, markersize=6)
        ax.set_xlabel('Adjacent orientation index', fontsize=10)
        ax.set_ylabel('Distance between adjacent orientations', fontsize=10)
        ax.set_title(f'SF {sf_val:.2f}', fontsize=11)
        ax.grid(alpha=0.3)
    fig2.suptitle('Adjacent Orientation Distance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'dspca_adjacent_orientation_distance_by_sf.svg'), format='svg', bbox_inches='tight')
    plt.close()
    
    # Plot 3: 3D visualization of target-free subspace (top 3 dims)
    if projection_targetfree.shape[1] >= 3:
        fig = plt.figure(figsize=(12, 5))
        
        # Subplot 1: Colored by orientation
        ax1 = fig.add_subplot(121, projection='3d')
    for ori_idx, ori in enumerate(unique_orientations):
        mask = np.abs(ori_from_targets - ori) < 1
        if np.sum(mask) > 0:
            ax1.scatter(projection_targetfree[mask, 0], projection_targetfree[mask, 1],
                        projection_targetfree[mask, 2], c=[ori_colors[ori_idx]],
                        s=50, alpha=0.7, edgecolors='black', linewidths=0.5, label=f'{ori:.0f}°')

    ax1.set_xlabel(f'Target-Free PC1\n({100*targetfree_var[0]/total_var:.1f}%)', fontsize=10)
    ax1.set_ylabel(f'Target-Free PC2\n({100*targetfree_var[1]/total_var:.1f}%)', fontsize=10)
    ax1.set_zlabel(f'Target-Free PC3\n({100*targetfree_var[2]/total_var:.1f}%)', fontsize=10)
    ax1.set_title('Target-Free Subspace\n(colored by Ori)', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=7, ncol=2, loc='upper left', bbox_to_anchor=(0, 1))
    ax1.view_init(elev=20, azim=45)

    # Subplot 2: Colored by SF (mapped)
    ax2 = fig.add_subplot(122, projection='3d')
    sf_color_map = {sf: plt.cm.viridis(i / max(len(mapped_sfs), 1)) for i, sf in enumerate(mapped_sfs)}
    for sf in mapped_sfs:
        mask = np.isclose(mapped_sf_targets, sf, rtol=0.0, atol=1e-9)
        if np.sum(mask) > 0:
            ax2.scatter(projection_targetfree[mask, 0], projection_targetfree[mask, 1],
                        projection_targetfree[mask, 2], c=[sf_color_map[sf]],
                        s=50, alpha=0.7, edgecolors='black', linewidths=0.5, label=f'SF {sf:.2f}')

    ax2.set_xlabel(f'Target-Free PC1\n({100*targetfree_var[0]/total_var:.1f}%)', fontsize=10)
    ax2.set_ylabel(f'Target-Free PC2\n({100*targetfree_var[1]/total_var:.1f}%)', fontsize=10)
    ax2.set_zlabel(f'Target-Free PC3\n({100*targetfree_var[2]/total_var:.1f}%)', fontsize=10)
    ax2.set_title('Target-Free Subspace\n(colored by SF)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=7, ncol=1, loc='upper left', bbox_to_anchor=(0, 1))
    ax2.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'dspca_targetfree_subspace_3d.svg'), format='svg', bbox_inches='tight')
    plt.close()
    
    # Plot 4: Correlation heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Target subspace correlations
    ax = axes[0]
    target_corr_vals = target_signal[:, :, 0]  # Shape: (n_target_axes, n_targets)
    im1 = ax.imshow(target_corr_vals, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['SF', 'sin', 'cos'])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['SF Axis', 'sin Axis', 'cos Axis'])
    ax.set_title('Target Subspace\nCorrelations with Task Variables', fontsize=11, fontweight='bold')
    
    # Add correlation values on heatmap
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{target_corr_vals[i, j]:.3f}\n(p={target_signal[i,j,1]:.1e})',
                    ha="center", va="center", color="black", fontsize=9, fontweight='bold')
    
    plt.colorbar(im1, ax=ax, label='Pearson r')
    
    # Target-free subspace correlations (top 10 dims)
    ax = axes[1]
    n_show = min(10, targetfree_signal.shape[0])
    targetfree_corr_vals = targetfree_signal[:n_show, :, 0]  # Shape: (n_dims, n_targets)
    im2 = ax.imshow(targetfree_corr_vals, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['SF', 'sin', 'cos'])
    ax.set_yticks(range(n_show))
    ax.set_yticklabels([f'PC{i+1}' for i in range(n_show)])
    ax.set_title(f'Target-Free Subspace (Top {n_show})\nCorrelations with Task Variables', fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=ax, label='Pearson r')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'dspca_correlation_heatmaps.svg'), format='svg', bbox_inches='tight')
    plt.close()
    
    # Save dsPCA results
    np.savez(
        os.path.join(save_path, 'dspca_results.npz'),
        projection_target=projection_target,
        projection_targetfree=projection_targetfree,
        ax_targets=ax_targets,
        ax_targetfree=ax_targetfree,
        target_var=target_var,
        targetfree_var=targetfree_var,
        total_var=total_var,
        targets=targets_dspca_raw,
        targets_fit=targets_dspca_fit,
        targets_fit_stats=np.stack([mu, sigma], axis=0),
        target_signal=target_signal,
        targetfree_signal=targetfree_signal,
        dot_target_ax=dot_target_ax,
        condition_orientations=np.array([ori for ori, sf in condition_keys]),
		condition_spatialfreqs=mapped_sf_targets,
		n_neurons_total=n_neurons,
		n_neurons_used=n_neurons_used,
		neuron_subset_idx=neuron_subset_idx if neuron_subset_idx is not None else np.array([], dtype=int),
    )

    print(f"\ndsPCA analysis complete. Results saved to {save_path}")
    print("="*60)
    
    # Print summary of neuron-count dependencies for user reference
    print("\n" + "="*70)
    print("CROSS-DATASET COMPARISON GUIDE: Neuron Count Dependencies")
    print("="*70)
    print("\n✓ NEURON-COUNT INDEPENDENT (directly comparable):")
    print("  PCA Metrics:")
    print("    - Power-law exponent α (eigenvalue decay rate)")
    print("    - Explained variance ratios (normalized by total variance)")
    print("    - Normalized PR (PR/N) - now saved in NPZ")
    print("    - Effective dimensionality % - now saved in NPZ")
    print("    - Knee ratio (Knee_PC/N) - now saved in NPZ")
    print("  dsPCA Metrics:")
    print("    - All variance percentages (normalized by total variance)")
    print("    - Orthogonality (Ori·SF dot product, unit vectors)")
    print("    - Correlation coefficients (Pearson r)")
    print("  Correlation Metrics:")
    print("    - Signal/noise correlation values (Pearson r)")
    print("\n✗ NEURON-COUNT DEPENDENT (NOT directly comparable):")
    print("    - Raw participation ratio (PR) - scales with N")
    print("    - Raw knee location (PC index) - scales with dimensionality")
    print("    - Raw eigenvalues - scale with N and data magnitude")
    print("    - Absolute variance values - scale with data magnitude")
    print("\nRECOMMENDATION: Use normalized metrics (PR/N, knee ratio, variance %,")
    print("α, correlation coefficients) when comparing across datasets!")
    print("="*70 + "\n")

# # ─────────────────────────────────────────────
# # WRITE EXCEL FILES
# # ─────────────────────────────────────────────
# df_resp = pd.DataFrame(responsive_summary_rows)
# for col in TARGET_SF_COLS:
#     if col not in df_resp.columns:
#         df_resp[col] = np.nan

# df_tuning = pd.DataFrame(tuning_summary_rows)

# excel_out = os.path.join(RESULTS_ROOT, "Grating_task_results.xlsx")
# with pd.ExcelWriter(excel_out) as writer:   # no engine arg, avoids xlsxwriter dependency
#     df_resp.to_excel(writer,   sheet_name="Responsive_frac_by_SF", index=False)
#     df_tuning.to_excel(writer, sheet_name="Tuning_curves",        index=False)

# print("\nAll done. Excel file saved at:", excel_out)
