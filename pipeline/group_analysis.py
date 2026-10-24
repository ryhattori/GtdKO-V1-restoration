from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import textwrap
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from scipy.stats import chi2, mannwhitneyu, ttest_1samp, ttest_ind, ttest_rel, wilcoxon, zscore
from statsmodels.formula.api import mixedlm
from statsmodels.stats.anova import AnovaRM
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


from dspca.dspca import dsPCA
from dspca.utils import adjust_lightness
HAS_DSPCA = True


# Silence known, non-actionable RuntimeWarnings that occur in degenerate cases.
# - statsmodels RM-ANOVA can warn when residual variance is 0
# - scipy Wilcoxon can warn when all paired differences are 0
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"statsmodels\.stats\.anova")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"scipy\.stats\._wilcoxon")

# MixedLM frequently emits non-actionable warnings when the estimated random-intercept
# variance collapses to ~0 (boundary/singular RE covariance). We still want the model
# (or its fixed-effect fallback) to run without flooding stdout.
warnings.filterwarnings(
	"ignore",
	category=UserWarning,
	module=r"statsmodels\.regression\.mixed_linear_model",
	message=r".*Random effects covariance is singular.*",
)
warnings.filterwarnings(
	"ignore",
	category=UserWarning,
	module=r"statsmodels\.regression\.mixed_linear_model",
	message=r".*random effects covariance matrix is singular.*",
)
warnings.filterwarnings(
	"ignore",
	category=ConvergenceWarning,
	module=r"statsmodels\.regression\.mixed_linear_model",
)
warnings.filterwarnings(
	"ignore",
	category=UserWarning,
	module=r"statsmodels\.regression\.mixed_linear_model",
	message=r".*boundary of the parameter space.*",
)

# Some optimization failures are reported by a shared statsmodels base module.
# These are very common for MixedLM in small/degenerate datasets and are not
# actionable in this pipeline (we already try multiple optimizers + fallback).
warnings.filterwarnings(
	"ignore",
	category=ConvergenceWarning,
	module=r"statsmodels\.base\.model",
	message=r"Maximum Likelihood optimization failed to converge\. Check mle_retvals",
)


_PANEL_NUMBER_RE = re.compile(r"(?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)(?P<star>\*)?")


# ---------------------------------------------------------------------------
# Paths and group definitions
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
RESULTS_ROOT = config.RESULTS_ROOT
GROUPED_PLOTS_DIR = os.path.join(RESULTS_ROOT, "group_figures")
os.makedirs(GROUPED_PLOTS_DIR, exist_ok=True)

IMAGE_ROOT = config.RAW_IMAGING_ROOT
PROC_IMAGE_ROOT = config.SUITE2P_ROOT

# ---------------------------------------------------------------------------
# Stats options (user-editable)
# ---------------------------------------------------------------------------
# If False, interaction terms are not computed/reported for omnibus RM-ANOVA/ART
# (e.g., Training×SF, Training×Ori, Training×N). Main-effect p-values are still
# computed as before.
RM_ANOVA_INCLUDE_INTERACTION_TERMS = True
ART_INCLUDE_INTERACTION_TERMS = True

# If False, LME models are fit without fixed-effect interaction terms.
LME_INCLUDE_INTERACTION_TERMS = True

# Optional: also compute LME stats using R's lme4 via rpy2 (if installed).
# If rpy2/R/lme4 are not available, these values will show as NA.
LME4_ENABLE = True

# Optional: also compute ART (aligned rank transform) omnibus tests using R's
# ARTool via rpy2 (if installed). Labels in panels:
# - ART-Raov / ART3-Raov : ARTool::art + Error(subject/...) (ANOVA-style repeated measures)
# - ART-Rlmer / ART3-Rlmer : ARTool::art + (1|subject) (mixed model)
ART_R_ENABLE = True

# Parallelism (user-editable)
# Notes:
# - Uses process-based parallelism (safe with heavy NumPy + file I/O).
# - Requires running this file as a script (protected by __main__).
USE_PARALLEL = False
PARALLEL_MAX_WORKERS: Optional[int] = None

# Target spatial frequency values as used in the single-session script
TARGET_SF_VALUES = np.array([0.0000, 7.0184, 14.7623, 29.8831, 59.9448, 119.9789, 240.0023])

# Decoding analyses are repeated for multiple neuron counts.
DECODING_CELL_COUNTS = [10, 20, 30, 40, 50, 60, 70, 80]
DECODING_DEFAULT_N = 80
DECODING_BASE_SEED = 42
RIDGE_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

SHUFFLED_DECODING_NULL_ITERATIONS = 100
SHUFFLED_DECODING_NULL_CACHE_VERSION = 3
SHUFFLED_DECODING_NULL_MAX_WORKERS: Optional[int] = None

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

RESPONSIVE_ORIENTATION_EXCLUDED_MAPPED_SF = 0.64

def convert_sf_value(sf: float) -> float:
	"""Convert original SF to desired value using nearest match with tolerance."""
	if sf <= 0:
		return 0.0
	# Use tolerance to match float originals
	for k, v in ORIG_SF_MAP.items():
		if abs(sf - k) < 1e-3:
			return v
	# Fallback: choose closest key
	closest_k = min(ORIG_SF_MAP.keys(), key=lambda x: abs(sf - x))
	return ORIG_SF_MAP[closest_k]

GROUPS: Dict[str, List[str]] = {
	"WT_BEFORE": [
		r"250423\VD017\Left\170um",
		r"250424\VD018\Left\190um",
		r"250424\VD019\Left\120um",
		r"250625\VD020\Left\170 um",
		r"250724\VD022\Left",
	],
	"WT_AFTER": [
		r"250716\VD017\Left",
		r"250717\VD018\Right",
		r"250717\VD019\Left",
		r"250707\VD020\Left",
		r"250912\VD022\Left",
	],
	"RESCUED_BEFORE": [
		r"250424\VD024\Right\150um",
		# r"250701\VD029\Left",	# removed due to training failure
		r"250701\VD031\Left\170",
		r"250715\VD032\Left",
		r"250814\VD033\Left",
		r"260223\VD034\Left",
	],
	"RESCUED_AFTER": [
		r"250716\VD024\Right",
		# r"250812\VD029\Left",	# removed due to training failure
		r"250812\VD031\Left",
		r"251115\VD032\Left2",
		r"251115\VD033\Left",
		r"260320\VD034\Left",
	],
	"BLIND": [
		r"250425\VD026\Left\170um",
		r"250515\VD027\Left\150um",
		r"250715\VD025\Right",
	],
}

# Mouse pairing for pre/post training comparisons
MOUSE_PAIRS = {
	"WT": [
		(r"250423\VD017\Left\170um", r"250716\VD017\Left"),
		(r"250424\VD018\Left\190um", r"250717\VD018\Right"),
		(r"250424\VD019\Left\120um", r"250717\VD019\Left"),
		(r"250625\VD020\Left\170 um", r"250707\VD020\Left"),
		(r"250724\VD022\Left", r"250912\VD022\Left"),
	],
	"RESCUED": [
		(r"250424\VD024\Right\150um", r"250716\VD024\Right"),
		# r"250701\VD029\Left", r"250812\VD029\Left",	# removed due to training failure
		(r"250701\VD031\Left\170", r"250812\VD031\Left"),
		(r"250715\VD032\Left", r"251115\VD032\Left2"),
		(r"250814\VD033\Left", r"251115\VD033\Left"),
		(r"260223\VD034\Left", r"260320\VD034\Left"),
	],
}

GROUP_COLORS = {
	"WT_BEFORE": "#9ecae1",      # lighter blue
	"WT_AFTER": "#08306b",       # deep navy blue
	"RESCUED_BEFORE": "#fcae91",  # light salmon
	"RESCUED_AFTER": "#a50f15",   # dark crimson
	"BLIND": "#7f7f7f",          # gray
}

GROUP_LABELS = {
	"WT_BEFORE": "WT pre-training",
	"WT_AFTER": "WT post-training",
	"RESCUED_BEFORE": "Rescued pre-training",
	"RESCUED_AFTER": "Rescued post-training",
	"BLIND": "Blind",
}


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


def _fit_ridge_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float) -> np.ndarray:
	"""Fit a standardized ridge decoder and predict held-out targets."""
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)
	model = Ridge(alpha=float(alpha))
	model.fit(X_train_scaled, y_train)
	return model.predict(X_test_scaled)


def _run_fixed_linear_decoder(
	X: np.ndarray,
	y: np.ndarray,
	*,
	alpha: float,
	outer_cv,
	scorer,
) -> float:
	"""Fixed-alpha ridge decoder returning pooled out-of-fold score."""
	if X.shape[0] < 4 or not np.isfinite(alpha):
		return float('nan')
	y_all_pred: List[np.ndarray] = []
	y_all_true: List[np.ndarray] = []
	for train_idx, test_idx in outer_cv.split(X):
		try:
			y_pred = _fit_ridge_predict(X[train_idx], y[train_idx], X[test_idx], alpha)
		except Exception:
			y_pred = None
		if y_pred is not None:
			y_all_pred.append(np.asarray(y_pred, dtype=float))
			y_all_true.append(np.asarray(y[test_idx], dtype=float))
	if not y_all_pred or not y_all_true:
		return float('nan')
	try:
		return float(scorer(np.concatenate(y_all_true, axis=0), np.concatenate(y_all_pred, axis=0)))
	except Exception:
		return float('nan')


def _sf_regression_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	"""Pearson correlation score for continuous log-SF decoding."""
	return _safe_pearsonr(y_true, y_pred)


def _orientation_regression_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	"""Circular correlation score for continuous orientation decoding."""
	true_phase = _orientation_phase_from_targets(y_true)
	pred_phase = _orientation_phase_from_targets(y_pred)
	return _safe_circular_corr(true_phase, pred_phase)


def _make_neuron_subsets(n_total: int, n_sub: int, *, seed: int) -> List[np.ndarray]:
	"""Create subsets of size n_sub so every neuron is used at least once."""
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


def _null_cache_path() -> str:
	return os.path.join(
		RESULTS_ROOT,
		f"rescued_pre_shuffled_continuous_null_iter{int(SHUFFLED_DECODING_NULL_ITERATIONS)}_v{int(SHUFFLED_DECODING_NULL_CACHE_VERSION)}.npz",
	)


def _session_null_cache_dir() -> str:
	path = os.path.join(RESULTS_ROOT, '_shuffled_null_session_cache')
	os.makedirs(path, exist_ok=True)
	return path


def _session_null_cache_path(session_rel: str, *, n_iter: int) -> str:
	slug = re.sub(r'[^A-Za-z0-9._-]+', '_', str(session_rel)).strip('_')
	if not slug:
		slug = f"session_{int(zlib.crc32(str(session_rel).encode('utf-8'))) & 0xffffffff:08x}"
	return os.path.join(
		_session_null_cache_dir(),
		f"{slug}_iter{int(n_iter)}_v{int(SHUFFLED_DECODING_NULL_CACHE_VERSION)}.npz",
	)


def _load_session_null_cache_file(cache_path: str, *, n_iter: int) -> Optional[Dict[str, np.ndarray]]:
	if not os.path.exists(cache_path):
		return None
	try:
		cached = np.load(cache_path, allow_pickle=False)
		if (
			'int_version' not in cached.files
			or int(cached['int_version']) != int(SHUFFLED_DECODING_NULL_CACHE_VERSION)
			or 'n_iterations' not in cached.files
			or int(cached['n_iterations']) != int(n_iter)
		):
			return None
		return {
			'orientation_by_sf': np.asarray(cached['orientation_by_sf'], dtype=float),
			'sf_by_orientation': np.asarray(cached['sf_by_orientation'], dtype=float),
			'orientations': np.asarray(cached['orientations'], dtype=float),
		}
	except Exception:
		return None


@lru_cache(maxsize=None)
def _load_or_compute_session_shuffled_null_cache(session_rel: str) -> Dict[str, np.ndarray]:
	"""Load one session's shuffled-null cache, computing it on demand if needed."""
	cache_path = _session_null_cache_path(session_rel, n_iter=int(SHUFFLED_DECODING_NULL_ITERATIONS))
	cached = _load_session_null_cache_file(cache_path, n_iter=int(SHUFFLED_DECODING_NULL_ITERATIONS))
	if cached is None:
		_save_session_null_cache_file(
			session_rel,
			n_iter=int(SHUFFLED_DECODING_NULL_ITERATIONS),
			output_path=cache_path,
		)
		cached = _load_session_null_cache_file(cache_path, n_iter=int(SHUFFLED_DECODING_NULL_ITERATIONS))
	if cached is None:
		raise RuntimeError(f"Unable to load shuffled-null cache for session {session_rel}")
	return cached


def _save_session_null_cache_file(session_rel: str, *, n_iter: int, output_path: str) -> None:
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	ori_null, sf_null, orientations = _compute_session_shuffled_continuous_null(session_rel, n_iter=int(n_iter))
	tmp_path = f"{output_path}.tmp"
	with open(tmp_path, 'wb') as f:
		np.savez_compressed(
			f,
			int_version=np.array(int(SHUFFLED_DECODING_NULL_CACHE_VERSION), dtype=int),
			n_iterations=np.array(int(n_iter), dtype=int),
			orientation_by_sf=ori_null,
			sf_by_orientation=sf_null,
			orientations=np.asarray(orientations, dtype=float),
		)
	os.replace(tmp_path, output_path)


def _resolve_shuffled_null_max_workers(n_sessions: int) -> int:
	requested = SHUFFLED_DECODING_NULL_MAX_WORKERS
	if requested is None:
		requested = max(1, (os.cpu_count() or 1) - 1)
	try:
		requested_int = int(requested)
	except Exception:
		requested_int = 1
	return max(1, min(int(n_sessions), requested_int))


def _run_session_null_worker_subprocess(session_rel: str, *, n_iter: int, output_path: str) -> str:
	cmd = [
		sys.executable,
		os.path.abspath(__file__),
		'--shuffled-null-worker',
		str(session_rel),
		'--null-iter',
		str(int(n_iter)),
		'--null-output',
		output_path,
	]
	env = os.environ.copy()
	for var_name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
		env[var_name] = '1'
	completed = subprocess.run(
		cmd,
		cwd=RESULTS_ROOT,
		capture_output=True,
		text=True,
		env=env,
	)
	if completed.returncode != 0:
		stdout_tail = (completed.stdout or '').strip()[-4000:]
		stderr_tail = (completed.stderr or '').strip()[-4000:]
		raise RuntimeError(
			f"Shuffled-null worker failed for {session_rel} (exit {completed.returncode})\n"
			f"stdout:\n{stdout_tail}\n"
			f"stderr:\n{stderr_tail}"
		)
	return output_path


@lru_cache(maxsize=None)
def _load_session_continuous_decoder_inputs(session_rel: str) -> Dict[str, object]:
	"""Rebuild the per-session continuous decoding inputs from raw data."""
	data_path = os.path.join(IMAGE_ROOT, session_rel)
	suite2p_path = os.path.join(PROC_IMAGE_ROOT, session_rel, 'suite2p', 'plane0')
	session_dir = os.path.join(RESULTS_ROOT, session_rel)

	json_files = glob.glob(os.path.join(data_path, '*.json'))
	if len(json_files) == 0:
		raise FileNotFoundError(f"No JSON file found for {session_rel}")
	with open(json_files[0], 'r') as f:
		json_data = [json.loads(line) for line in f][-1]

	try:
		scanimage_frame_path = glob.glob(os.path.join(data_path, '*Scanimage_Arm1_Frame*.csv'))[0]
		stimulus_time_path = glob.glob(os.path.join(data_path, '*DataUnity_TrialStart*.csv'))[0]
		unity_frame_path = glob.glob(os.path.join(data_path, '*DataUnity_Frame*.csv'))[0]
	except IndexError as exc:
		raise FileNotFoundError(f"Missing Bonsai CSV files for {session_rel}") from exc

	scanimage_frame_time_all = np.genfromtxt(scanimage_frame_path, delimiter=',', comments='//', usecols=0)
	scanimage_frame_boolean = np.genfromtxt(scanimage_frame_path, delimiter=',', comments='//', usecols=1, dtype=str)
	scanimage_frame_onset_bool = np.zeros_like(scanimage_frame_boolean, dtype=int)
	scanimage_frame_onset_bool[1:] = ((scanimage_frame_boolean[1:] == 'True') & (scanimage_frame_boolean[:-1] == 'False')).astype(int)
	scanimage_frame_onset = scanimage_frame_time_all[scanimage_frame_onset_bool == 1]

	stimulus_time_all = np.genfromtxt(stimulus_time_path, delimiter=',', comments='//', usecols=0)
	stimulus_time_bool = np.genfromtxt(stimulus_time_path, delimiter=',', comments='//', usecols=1, dtype=str)
	stimulus_onset_bool = np.zeros_like(stimulus_time_bool, dtype=int)
	stimulus_onset_bool[1:] = ((stimulus_time_bool[1:] == 'True') & (stimulus_time_bool[:-1] == 'False')).astype(int)
	stimulus_onset = stimulus_time_all[stimulus_onset_bool == 1]

	unity_frame_time_all = np.genfromtxt(unity_frame_path, delimiter=',', comments='//', usecols=0)
	unity_frame_boolean = np.genfromtxt(unity_frame_path, delimiter=',', comments='//', usecols=1, dtype=str)
	unity_frame_onset_bool = np.zeros_like(unity_frame_boolean, dtype=int)
	unity_frame_onset_bool[1:] = ((unity_frame_boolean[1:] == 'True') & (unity_frame_boolean[:-1] == 'False')).astype(int)
	unity_frame_onset = unity_frame_time_all[unity_frame_onset_bool == 1]

	if getattr(stimulus_onset, 'size', len(stimulus_onset)) == 0 and unity_frame_onset.size > 0:
		stimulus_onset = unity_frame_onset.copy()
		while stimulus_onset.size >= 2 and (np.diff(stimulus_onset)[-1] > 500):
			stimulus_onset = stimulus_onset[:-1]

	iscell = np.load(os.path.join(suite2p_path, 'iscell.npy'))
	spks = np.load(os.path.join(suite2p_path, 'spks.npy'))
	spks_cell = spks[iscell[:, 0] == 1]
	spks_cell_zscored = zscore(spks_cell, axis=1)
	valid_neurons = ~(np.isnan(spks_cell_zscored).any(axis=1) | (np.var(spks_cell, axis=1) < 1e-10))
	spks_cell_zscored = spks_cell_zscored[valid_neurons]

	mean_dt = np.mean(np.diff(scanimage_frame_onset))
	negative_window = int(round(50 / mean_dt))
	positive_window = int(round(250 / mean_dt))
	n_frames = spks_cell_zscored.shape[1]

	trial_slices: List[np.ndarray] = []
	kept_trial_indices: List[int] = []
	for trial_i, onset in enumerate(stimulus_onset):
		trial_idx = int(np.argmin(np.abs(scanimage_frame_onset - onset)))
		start_idx = trial_idx - negative_window
		end_idx = trial_idx + positive_window + 1
		if start_idx < 0 or end_idx > n_frames:
			continue
		trial_slices.append(spks_cell_zscored[:, start_idx:end_idx])
		kept_trial_indices.append(trial_i)
	if not trial_slices:
		raise ValueError(f"No valid trial windows for {session_rel}")

	kept_trial_indices_arr = np.array(kept_trial_indices, dtype=int)
	spks_tensor = np.stack(trial_slices, axis=-1)
	time_axis = (np.arange(spks_tensor.shape[1]) - negative_window) * mean_dt
	pre_idx = np.where(time_axis < 0)[0]
	response_idx = np.where((time_axis >= 0) & (time_axis <= 250))[0]

	orientation_history = np.asarray(json_data['orientation_history'], dtype=float)
	spatialfreq_history = np.asarray(json_data['spatialfreq_history'], dtype=float)
	unique_orientations = np.unique(orientation_history)
	unique_spatialfreqs = np.unique(spatialfreq_history)

	index_map = {orig_idx: kept_idx for kept_idx, orig_idx in enumerate(kept_trial_indices_arr)}

	def map_to_kept(idxs: np.ndarray) -> np.ndarray:
		return np.array([index_map[i] for i in np.asarray(idxs, dtype=int) if i in index_map], dtype=int)

	baseline_trial_inds = map_to_kept(np.where(spatialfreq_history == unique_spatialfreqs[0])[0])
	if baseline_trial_inds.size == 0:
		raise ValueError(f"No valid baseline trials for {session_rel}")

	response_per_trial = np.nanmean(spks_tensor[:, response_idx, :], axis=1)
	baseline_mean = np.nanmean(response_per_trial[:, baseline_trial_inds], axis=1, keepdims=True)
	delta_per_trial = response_per_trial - baseline_mean
	mean_heatmap = np.nanmean(spks_tensor, axis=2)
	mean_activity_trace = np.nanmean(spks_tensor, axis=(0, 2))
	if response_idx.size > 0:
		sort_mean_activity = np.nanmean(response_per_trial, axis=1)
	else:
		sort_mean_activity = np.full(mean_heatmap.shape[0], np.nan, dtype=float)

	mapped_sf_kept = np.array([convert_sf_value(sf) for sf in spatialfreq_history[kept_trial_indices_arr]], dtype=float)
	log_sf_kept = np.full(mapped_sf_kept.shape, np.nan, dtype=float)
	nonzero_mapped_sf = mapped_sf_kept > 0
	log_sf_kept[nonzero_mapped_sf] = np.log2(mapped_sf_kept[nonzero_mapped_sf])

	trial_inds_by_sf: List[np.ndarray] = []
	ori_targets_by_sf: List[Optional[np.ndarray]] = []
	for sf in unique_spatialfreqs:
		trial_idx = map_to_kept(np.where(spatialfreq_history == sf)[0])
		trial_inds_by_sf.append(trial_idx)
		try:
			sf_val = float(sf)
		except Exception:
			sf_val = np.nan
		if (not np.isfinite(sf_val)) or abs(sf_val) < 1e-12 or trial_idx.size < 4:
			ori_targets_by_sf.append(None)
			continue
		y_ori_deg = np.asarray(orientation_history[kept_trial_indices_arr[trial_idx]], dtype=float)
		if np.unique(y_ori_deg).size < 2:
			ori_targets_by_sf.append(None)
			continue
		ori_targets_by_sf.append(_orientation_targets_from_deg(y_ori_deg))

	trial_inds_by_ori: List[np.ndarray] = []
	log_sf_targets_by_ori: List[Optional[np.ndarray]] = []
	trial_inds_valid_by_ori: List[np.ndarray] = []
	for ori in unique_orientations:
		trial_idx = map_to_kept(np.where(orientation_history == ori)[0])
		trial_inds_by_ori.append(trial_idx)
		if trial_idx.size < 4:
			trial_inds_valid_by_ori.append(np.array([], dtype=int))
			log_sf_targets_by_ori.append(None)
			continue
		y_log_sf = log_sf_kept[trial_idx]
		valid_mask = np.isfinite(y_log_sf)
		if np.sum(valid_mask) < 4:
			trial_inds_valid_by_ori.append(np.array([], dtype=int))
			log_sf_targets_by_ori.append(None)
			continue
		trial_idx_valid = trial_idx[valid_mask]
		y_log_sf_valid = y_log_sf[valid_mask]
		if np.unique(y_log_sf_valid).size < 2:
			trial_inds_valid_by_ori.append(np.array([], dtype=int))
			log_sf_targets_by_ori.append(None)
			continue
		trial_inds_valid_by_ori.append(trial_idx_valid)
		log_sf_targets_by_ori.append(y_log_sf_valid[:, None])

	decoding_cellcount = safe_load_npz(os.path.join(session_dir, 'decoding_cellcount_results.npz'))
	if decoding_cellcount is None:
		raise FileNotFoundError(f"Missing decoding_cellcount_results.npz for {session_rel}")
	orientation_best_alpha = get_from_npz(decoding_cellcount, ['orientation_continuous_best_alpha_by_sf'])
	sf_best_alpha = get_from_npz(decoding_cellcount, ['sf_continuous_best_alpha_by_orientation'])
	if orientation_best_alpha is None or sf_best_alpha is None:
		raise KeyError(f"Missing continuous best-alpha arrays for {session_rel}")

	subsets_by_count: List[List[np.ndarray]] = []
	for n_sub in DECODING_CELL_COUNTS:
		seed_i = int(DECODING_BASE_SEED) ^ int(n_sub * 1009) ^ int(zlib.crc32(str(session_dir).encode('utf-8')))
		subsets_by_count.append(_make_neuron_subsets(int(delta_per_trial.shape[0]), int(n_sub), seed=seed_i))

	return {
		'delta_per_trial': delta_per_trial,
		'time_axis': np.asarray(time_axis, dtype=float),
		'mean_heatmap': np.asarray(mean_heatmap, dtype=float),
		'mean_activity_trace': np.asarray(mean_activity_trace, dtype=float),
		'sort_mean_activity': np.asarray(sort_mean_activity, dtype=float),
		'unique_spatialfreqs': np.asarray(unique_spatialfreqs, dtype=float),
		'unique_orientations': np.asarray(unique_orientations, dtype=float),
		'ori_targets_by_sf': ori_targets_by_sf,
		'trial_inds_by_sf': trial_inds_by_sf,
		'log_sf_targets_by_ori': log_sf_targets_by_ori,
		'trial_inds_valid_by_ori': trial_inds_valid_by_ori,
		'orientation_best_alpha': np.asarray(orientation_best_alpha, dtype=float),
		'sf_best_alpha': np.asarray(sf_best_alpha, dtype=float),
		'subsets_by_count': subsets_by_count,
	}


def plot_grouped_neural_activity_heatmap_and_mean_activity() -> None:
	"""Plot pooled-neuron heatmaps and mean traces for Blind/WT/Rescued groups."""
	group_order = ["BLIND", "WT_BEFORE", "WT_AFTER", "RESCUED_BEFORE", "RESCUED_AFTER"]
	group_panels: List[Dict[str, object]] = []

	def _safe_nanmean(arr: np.ndarray, axis: int) -> np.ndarray:
		arr = np.asarray(arr, dtype=float)
		finite = np.isfinite(arr)
		sum_arr = np.sum(np.where(finite, arr, 0.0), axis=axis)
		count_arr = np.sum(finite, axis=axis)
		out = sum_arr / np.maximum(count_arr, 1)
		return np.where(count_arr > 0, out, np.nan)

	def _interp_trace_to_ref(trace: np.ndarray, src_time: np.ndarray, ref_time: np.ndarray) -> np.ndarray:
		trace = np.asarray(trace, dtype=float).ravel()
		src_time = np.asarray(src_time, dtype=float).ravel()
		ref_time = np.asarray(ref_time, dtype=float).ravel()
		mask = np.isfinite(trace) & np.isfinite(src_time)
		if np.sum(mask) == 0:
			return np.full(ref_time.shape, np.nan, dtype=float)
		if np.sum(mask) == 1:
			return np.full(ref_time.shape, float(trace[mask][0]), dtype=float)
		trace_valid = trace[mask]
		time_valid = src_time[mask]
		return np.interp(ref_time, time_valid, trace_valid, left=float(trace_valid[0]), right=float(trace_valid[-1]))

	def _interp_rows_to_ref(mat: np.ndarray, src_time: np.ndarray, ref_time: np.ndarray) -> np.ndarray:
		mat = np.asarray(mat, dtype=float)
		if mat.ndim != 2:
			return np.full((0, ref_time.size), np.nan, dtype=float)
		out = np.full((mat.shape[0], ref_time.size), np.nan, dtype=float)
		for row_idx in range(mat.shape[0]):
			out[row_idx, :] = _interp_trace_to_ref(mat[row_idx, :], src_time, ref_time)
		return out

	def _fill_heatmap_nans(mat: np.ndarray) -> np.ndarray:
		out = np.asarray(mat, dtype=float).copy()
		if out.ndim != 2 or out.size == 0:
			return out
		nan_mask = ~np.isfinite(out)
		if not np.any(nan_mask):
			return out
		col_fill = _safe_nanmean(out, axis=0)
		rows, cols = np.where(nan_mask)
		out[rows, cols] = col_fill[cols]
		still_nan = ~np.isfinite(out)
		if np.any(still_nan):
			row_fill = _safe_nanmean(out, axis=1)
			rows, cols = np.where(still_nan)
			out[rows, cols] = row_fill[rows]
		out[~np.isfinite(out)] = 0.0
		return out

	for group_name in group_order:
		sessions = GROUPS.get(group_name, [])
		if not sessions:
			continue
		ref_time_axis: Optional[np.ndarray] = None
		pooled_heatmap_rows: List[np.ndarray] = []
		pooled_sort_scores: List[np.ndarray] = []
		group_mean_traces: List[np.ndarray] = []
		for session_rel in sessions:
			inputs = _load_session_continuous_decoder_inputs(session_rel)
			time_axis = np.asarray(inputs.get('time_axis'), dtype=float)
			mean_heatmap = np.asarray(inputs.get('mean_heatmap'), dtype=float)
			mean_trace = np.asarray(inputs.get('mean_activity_trace'), dtype=float)
			sort_mean_activity = np.asarray(inputs.get('sort_mean_activity'), dtype=float).ravel()
			if mean_heatmap.ndim != 2 or mean_trace.ndim != 1 or sort_mean_activity.ndim != 1:
				continue
			if mean_heatmap.shape[0] != sort_mean_activity.size:
				continue
			if ref_time_axis is None:
				ref_time_axis = np.asarray(time_axis, dtype=float)
			if ref_time_axis is None or ref_time_axis.size == 0:
				continue
			if time_axis.shape != ref_time_axis.shape or not np.allclose(time_axis, ref_time_axis, atol=1e-6, rtol=1e-6):
				mean_heatmap_use = _interp_rows_to_ref(mean_heatmap, time_axis, ref_time_axis)
				mean_trace_use = _interp_trace_to_ref(mean_trace, time_axis, ref_time_axis)
			else:
				mean_heatmap_use = mean_heatmap
				mean_trace_use = mean_trace
			valid_neuron_mask = np.any(np.isfinite(mean_heatmap_use), axis=1)
			if not np.any(valid_neuron_mask):
				continue
			pooled_heatmap_rows.append(np.asarray(mean_heatmap_use[valid_neuron_mask, :], dtype=float))
			pooled_sort_scores.append(np.asarray(sort_mean_activity[valid_neuron_mask], dtype=float))
			group_mean_traces.append(np.asarray(mean_trace_use, dtype=float))

		if ref_time_axis is None or not pooled_heatmap_rows:
			continue
		pooled_heatmap = np.vstack(pooled_heatmap_rows)
		pooled_sort = np.concatenate(pooled_sort_scores)
		valid_sort = np.where(np.isfinite(pooled_sort))[0]
		invalid_sort = np.where(~np.isfinite(pooled_sort))[0]
		sort_order = np.concatenate([
			valid_sort[np.argsort(-pooled_sort[valid_sort])],
			invalid_sort,
		])
		pooled_heatmap = _fill_heatmap_nans(pooled_heatmap[sort_order, :])
		heat_time_mask = np.asarray(ref_time_axis, dtype=float) >= 0
		if not np.any(heat_time_mask):
			continue
		heatmap_time_axis = np.asarray(ref_time_axis[heat_time_mask], dtype=float)
		pooled_heatmap = np.asarray(pooled_heatmap[:, heat_time_mask], dtype=float)
		group_trace_mean = _interp_trace_to_ref(_safe_nanmean(np.vstack(group_mean_traces), axis=0), ref_time_axis, ref_time_axis)
		group_panels.append({
			'group_name': group_name,
			'heatmap_time_axis': heatmap_time_axis,
			'time_axis': ref_time_axis,
			'heatmap': pooled_heatmap,
			'mean_trace': group_trace_mean,
			'n_neurons': int(pooled_heatmap.shape[0]),
		})

	if not group_panels:
		print('[skip] grouped neural activity heatmap due to missing pooled data')
		return

	vmin, vmax = -0.02, 0.05

	all_trace_vals = np.concatenate([
		panel['mean_trace'][np.isfinite(panel['mean_trace'])] for panel in group_panels if np.any(np.isfinite(panel['mean_trace']))
	]) if any(np.any(np.isfinite(panel['mean_trace'])) for panel in group_panels) else np.array([], dtype=float)
	trace_ylim: Optional[Tuple[float, float]] = None
	if all_trace_vals.size:
		trace_min = float(np.nanmin(all_trace_vals))
		trace_max = float(np.nanmax(all_trace_vals))
		if np.isfinite(trace_min) and np.isfinite(trace_max):
			pad = max(0.05 * (trace_max - trace_min), 0.05)
			trace_ylim = (trace_min - pad, trace_max + pad)

	n_cols = len(group_panels)
	fig = plt.figure(figsize=(1.92 * n_cols + 0.6, 3.84))
	gs = fig.add_gridspec(
		2,
		n_cols + 1,
		height_ratios=[1.68, 1.2],
		width_ratios=([1.0] * n_cols) + [0.08],
		wspace=0.32,
		hspace=0.34,
	)
	axes = np.empty((2, n_cols), dtype=object)
	for col_idx in range(n_cols):
		axes[0, col_idx] = fig.add_subplot(gs[0, col_idx])
		axes[1, col_idx] = fig.add_subplot(gs[1, col_idx])
	cax = fig.add_subplot(gs[0, n_cols])
	fig.add_subplot(gs[1, n_cols]).axis('off')

	last_im = None
	for col_idx, panel in enumerate(group_panels):
		ax_heat = axes[0, col_idx]
		ax_trace = axes[1, col_idx]
		heatmap_time_axis = np.asarray(panel['heatmap_time_axis'], dtype=float)
		time_axis = np.asarray(panel['time_axis'], dtype=float)
		heatmap = np.asarray(panel['heatmap'], dtype=float)
		mean_trace = np.asarray(panel['mean_trace'], dtype=float)
		group_name = str(panel['group_name'])
		n_neurons = int(panel['n_neurons'])

		last_im = ax_heat.imshow(
			heatmap,
			aspect='auto',
			cmap='viridis',
			origin='upper',
			extent=[float(heatmap_time_axis[0]), float(heatmap_time_axis[-1]), float(n_neurons), 0.0],
			vmin=vmin,
			vmax=vmax,
		)
		ax_heat.set_xlim(float(heatmap_time_axis[0]), float(heatmap_time_axis[-1]))
		ax_heat.set_title(f"{GROUP_LABELS.get(group_name, group_name)}\n(n={n_neurons} pooled neurons)")
		ax_heat.set_xlabel('Time (ms)')
		if col_idx == 0:
			ax_heat.set_ylabel('Neuron index (sorted)')
		else:
			ax_heat.set_ylabel('')

		ax_trace.plot(time_axis, mean_trace, color='blue', linewidth=2.0)
		ax_trace.axvline(x=0.0, color='red', linestyle='--', linewidth=1.5)
		ax_trace.axvline(x=250.0, color='green', linestyle='--', linewidth=1.5)
		ax_trace.set_xlabel('Time (ms)')
		if col_idx == 0:
			ax_trace.set_ylabel('Mean activity')
		else:
			ax_trace.set_ylabel('')
		if trace_ylim is not None:
			ax_trace.set_ylim(*trace_ylim)

	if last_im is not None:
		cbar = fig.colorbar(last_im, cax=cax)
		cbar.set_label('Average activity')

	fig.suptitle('Pooled neural activity heatmaps and mean activity by group', fontsize=14, fontweight='bold')
	fig.subplots_adjust(left=0.06, right=0.96, bottom=0.09, top=0.88)
	outfile = os.path.join(GROUPED_PLOTS_DIR, 'neural_activity_heatmap_and_mean_activity_groups.svg')
	fig.savefig(outfile, format='svg')
	plt.close(fig)
	print(f"  [saved] {os.path.basename(outfile)}")


def _compute_session_shuffled_continuous_null(session_rel: str, *, n_iter: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Compute shuffled-label continuous decoding null curves for one session."""
	inputs = _load_session_continuous_decoder_inputs(session_rel)
	delta_per_trial = np.asarray(inputs['delta_per_trial'], dtype=float)
	unique_spatialfreqs = np.asarray(inputs['unique_spatialfreqs'], dtype=float)
	unique_orientations = np.asarray(inputs['unique_orientations'], dtype=float)
	ori_targets_by_sf = list(inputs['ori_targets_by_sf'])
	trial_inds_by_sf = list(inputs['trial_inds_by_sf'])
	log_sf_targets_by_ori = list(inputs['log_sf_targets_by_ori'])
	trial_inds_valid_by_ori = list(inputs['trial_inds_valid_by_ori'])
	orientation_best_alpha = np.asarray(inputs['orientation_best_alpha'], dtype=float)
	sf_best_alpha = np.asarray(inputs['sf_best_alpha'], dtype=float)
	subsets_by_count = list(inputs['subsets_by_count'])
	ori_deg_by_sf = [None if y is None else np.mod(np.rad2deg(0.5 * _orientation_phase_from_targets(y)), 180.0) for y in ori_targets_by_sf]
	subsets_by_count_for_null = [subsets[:1] if len(subsets) > 1 else subsets for subsets in subsets_by_count]

	ori_null = np.full((int(n_iter), len(DECODING_CELL_COUNTS), len(TARGET_SF_VALUES)), np.nan, dtype=float)
	sf_null = np.full((int(n_iter), len(DECODING_CELL_COUNTS), len(unique_orientations)), np.nan, dtype=float)
	rng = np.random.default_rng(int(137) ^ int(zlib.crc32(str(session_rel).encode('utf-8'))))

	for iter_idx in range(int(n_iter)):
		shuffled_ori_targets: List[Optional[np.ndarray]] = []
		for ori_deg in ori_deg_by_sf:
			if ori_deg is None:
				shuffled_ori_targets.append(None)
				continue
			shuffled_ori_targets.append(_orientation_targets_from_deg(rng.permutation(ori_deg)))
		shuffled_log_sf_targets = [None if y is None else rng.permutation(np.asarray(y, dtype=float).ravel())[:, None] for y in log_sf_targets_by_ori]

		for count_idx, subsets in enumerate(subsets_by_count_for_null):
			sf_repeats = np.full((len(subsets), unique_orientations.size), np.nan, dtype=float)
			ori_repeats_all = np.full((len(subsets), unique_spatialfreqs.size), np.nan, dtype=float)
			for rep_idx, neuron_subset in enumerate(subsets):
				for sf_idx, trial_idx in enumerate(trial_inds_by_sf):
					y_shuf = shuffled_ori_targets[sf_idx]
					if y_shuf is None or trial_idx.size < 4:
						continue
					X = delta_per_trial[neuron_subset][:, trial_idx].T
					alpha = orientation_best_alpha[count_idx, rep_idx, sf_idx] if (
						count_idx < orientation_best_alpha.shape[0]
						and rep_idx < orientation_best_alpha.shape[1]
						and sf_idx < orientation_best_alpha.shape[2]
					) else np.nan
					outer_cv = KFold(n_splits=min(5, X.shape[0]), shuffle=True, random_state=62)
					ori_repeats_all[rep_idx, sf_idx] = _run_fixed_linear_decoder(
						X,
						y_shuf,
						alpha=alpha,
						outer_cv=outer_cv,
						scorer=_orientation_regression_score,
					)
				for ori_idx, trial_idx_valid in enumerate(trial_inds_valid_by_ori):
					y_shuf = shuffled_log_sf_targets[ori_idx]
					if y_shuf is None or trial_idx_valid.size < 4:
						continue
					X = delta_per_trial[neuron_subset][:, trial_idx_valid].T
					alpha = sf_best_alpha[count_idx, rep_idx, ori_idx] if (
						count_idx < sf_best_alpha.shape[0]
						and rep_idx < sf_best_alpha.shape[1]
						and ori_idx < sf_best_alpha.shape[2]
					) else np.nan
					outer_cv = KFold(n_splits=min(5, X.shape[0]), shuffle=True, random_state=72)
					sf_repeats[rep_idx, ori_idx] = _run_fixed_linear_decoder(
						X,
						y_shuf,
						alpha=alpha,
						outer_cv=outer_cv,
						scorer=_sf_regression_score,
					)
			with warnings.catch_warnings():
				warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
				ori_null[iter_idx, count_idx, :] = align_to_targets(np.nanmean(ori_repeats_all, axis=0), unique_spatialfreqs)
				sf_null[iter_idx, count_idx, :] = np.nanmean(sf_repeats, axis=0)

	return ori_null, sf_null, unique_orientations


@lru_cache(maxsize=1)
def _load_or_compute_rescued_pre_shuffled_null_cache() -> Dict[str, np.ndarray]:
	"""Load or compute the Rescued-pre shuffled-label null cache used by grouped plots."""
	cache_path = _null_cache_path()
	if os.path.exists(cache_path):
		try:
			cached = np.load(cache_path, allow_pickle=False)
			if (
				'int_version' in cached.files
				and int(cached['int_version']) == int(SHUFFLED_DECODING_NULL_CACHE_VERSION)
				and 'n_iterations' in cached.files
				and int(cached['n_iterations']) == int(SHUFFLED_DECODING_NULL_ITERATIONS)
			):
				return {
					'orientation_by_sf': np.asarray(cached['orientation_by_sf'], dtype=float),
					'sf_by_orientation': np.asarray(cached['sf_by_orientation'], dtype=float),
					'orientations': np.asarray(cached['orientations'], dtype=float),
					'cell_counts': np.asarray(cached['cell_counts'], dtype=int),
				}
		except Exception:
			pass

	sessions = list(GROUPS.get('RESCUED_BEFORE', []))
	ori_session: List[np.ndarray] = []
	sf_session: List[np.ndarray] = []
	orient_ref: Optional[np.ndarray] = None
	results_by_session: Dict[str, Dict[str, np.ndarray]] = {}
	missing_sessions: List[Tuple[str, str]] = []
	for session_rel in sessions:
		session_cache_path = _session_null_cache_path(session_rel, n_iter=int(SHUFFLED_DECODING_NULL_ITERATIONS))
		cached_session = _load_session_null_cache_file(session_cache_path, n_iter=int(SHUFFLED_DECODING_NULL_ITERATIONS))
		if cached_session is None:
			missing_sessions.append((session_rel, session_cache_path))
		else:
			results_by_session[session_rel] = cached_session

	if missing_sessions:
		max_workers = _resolve_shuffled_null_max_workers(len(missing_sessions))
		if max_workers > 1:
			future_map = {}
			with ThreadPoolExecutor(max_workers=max_workers) as executor:
				for session_rel, session_cache_path in missing_sessions:
					print(f"[null] shuffled continuous decoding: {session_rel}")
					future = executor.submit(
						_run_session_null_worker_subprocess,
						session_rel,
						n_iter=int(SHUFFLED_DECODING_NULL_ITERATIONS),
						output_path=session_cache_path,
					)
					future_map[future] = (session_rel, session_cache_path)
				for future in as_completed(future_map):
					session_rel, _session_cache_path = future_map[future]
					future.result()
					print(f"[null] finished: {session_rel}")
		else:
			for session_rel, session_cache_path in missing_sessions:
				print(f"[null] shuffled continuous decoding: {session_rel}")
				_run_session_null_worker_subprocess(
					session_rel,
					n_iter=int(SHUFFLED_DECODING_NULL_ITERATIONS),
					output_path=session_cache_path,
				)
				print(f"[null] finished: {session_rel}")

	for session_rel, session_cache_path in missing_sessions:
		cached_session = _load_session_null_cache_file(session_cache_path, n_iter=int(SHUFFLED_DECODING_NULL_ITERATIONS))
		if cached_session is None:
			raise RuntimeError(f"Missing shuffled-null session cache after worker completed: {session_rel}")
		results_by_session[session_rel] = cached_session

	for session_rel in sessions:
		result = results_by_session.get(session_rel)
		if result is None:
			raise RuntimeError(f"Missing shuffled-null result for {session_rel}")
		ori_null = np.asarray(result['orientation_by_sf'], dtype=float)
		sf_null = np.asarray(result['sf_by_orientation'], dtype=float)
		orientations = np.asarray(result['orientations'], dtype=float)
		if orient_ref is None:
			orient_ref = orientations
		elif not np.array_equal(orient_ref, orientations):
			raise ValueError('Orientation mismatch across Rescued pre sessions while building null cache')
		ori_session.append(ori_null)
		sf_session.append(sf_null)

	if not ori_session or not sf_session or orient_ref is None:
		raise RuntimeError('Unable to build Rescued-pre shuffled decoding null cache')

	with warnings.catch_warnings():
		warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
		orientation_by_sf = np.nanmean(np.stack(ori_session, axis=0), axis=0)
		sf_by_orientation = np.nanmean(np.stack(sf_session, axis=0), axis=0)
	np.savez_compressed(
		cache_path,
		int_version=np.array(int(SHUFFLED_DECODING_NULL_CACHE_VERSION), dtype=int),
		n_iterations=np.array(int(SHUFFLED_DECODING_NULL_ITERATIONS), dtype=int),
		cell_counts=np.asarray(DECODING_CELL_COUNTS, dtype=int),
		orientation_by_sf=orientation_by_sf,
		sf_by_orientation=sf_by_orientation,
		orientations=orient_ref,
	)
	return {
		'orientation_by_sf': orientation_by_sf,
		'sf_by_orientation': sf_by_orientation,
		'orientations': orient_ref,
		'cell_counts': np.asarray(DECODING_CELL_COUNTS, dtype=int),
	}


def _get_decoder_null_distribution(metric_key: str) -> Optional[np.ndarray]:
	"""Return shuffled-label null samples for a grouped continuous decoding metric."""
	cache = _load_or_compute_rescued_pre_shuffled_null_cache()
	cell_counts = np.asarray(cache['cell_counts'], dtype=int)
	default_idx = int(np.where(cell_counts == int(DECODING_DEFAULT_N))[0][0]) if int(DECODING_DEFAULT_N) in cell_counts else -1
	if metric_key == 'decoding_accuracy':
		return np.asarray(cache['orientation_by_sf'][:, default_idx, :], dtype=float) if default_idx >= 0 else None
	if metric_key.startswith('decoding_accuracy_n'):
		try:
			n_val = int(metric_key.split('_n', 1)[1])
		except Exception:
			return None
		if n_val not in cell_counts:
			return None
		return np.asarray(cache['orientation_by_sf'][:, int(np.where(cell_counts == n_val)[0][0]), :], dtype=float)
	if metric_key == 'sf_decoding_by_ori':
		return np.asarray(cache['sf_by_orientation'][:, default_idx, :], dtype=float) if default_idx >= 0 else None
	if metric_key.startswith('sf_decoding_by_ori_n'):
		try:
			n_val = int(metric_key.split('_n', 1)[1])
		except Exception:
			return None
		if n_val not in cell_counts:
			return None
		return np.asarray(cache['sf_by_orientation'][:, int(np.where(cell_counts == n_val)[0][0]), :], dtype=float)
	return None


def _decoder_null_band(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""Return a 95% confidence interval around the shuffled-label mean."""
	with warnings.catch_warnings():
		warnings.filterwarnings('ignore', category=RuntimeWarning)
		samples = np.asarray(samples, dtype=float)
		mean = np.nanmean(samples, axis=0)
		n = np.sum(np.isfinite(samples), axis=0)
		std = np.nanstd(samples, axis=0, ddof=1)
		sem = np.divide(std, np.sqrt(n), out=np.full_like(mean, np.nan, dtype=float), where=n > 1)
		half_width = 1.96 * sem
		low = mean - half_width
		high = mean + half_width
	return np.asarray(low, dtype=float), np.asarray(high, dtype=float)


def _empirical_above_chance_pvalues(observed: np.ndarray, null_samples: np.ndarray) -> Dict[int, Optional[float]]:
	"""One-sided empirical p-values: observed > shuffled-label null."""
	obs = np.asarray(observed, dtype=float).ravel()
	null = np.asarray(null_samples, dtype=float)
	results: Dict[int, Optional[float]] = {}
	for idx in range(obs.size):
		if idx >= null.shape[1] or not np.isfinite(obs[idx]):
			results[idx] = None
			continue
		null_col = null[:, idx]
		null_col = null_col[np.isfinite(null_col)]
		if null_col.size == 0:
			results[idx] = None
			continue
		results[idx] = float((1.0 + np.sum(null_col >= float(obs[idx]))) / float(null_col.size + 1))
	return results


def _format_sig_value_list(values: List[str]) -> str:
	return ', '.join(values) if values else 'none'


def _format_sig_mapped_sf_from_q(q_by_idx: Dict[int, Optional[float]]) -> str:
	items: List[str] = []
	for sf_idx in range(1, len(TARGET_SF_VALUES)):
		q = q_by_idx.get(sf_idx, None)
		if q is None or not np.isfinite(q) or q >= 0.05:
			continue
		items.append(f"{convert_sf_value(float(TARGET_SF_VALUES[sf_idx])):.2f}")
	return _format_sig_value_list(items)


def _format_sig_orientation_from_q(orientations: np.ndarray, q_by_idx: Dict[int, Optional[float]]) -> str:
	items: List[str] = []
	for idx, ori in enumerate(np.asarray(orientations, dtype=float).ravel()):
		q = q_by_idx.get(idx, None)
		if q is None or not np.isfinite(q) or q >= 0.05:
			continue
		items.append(f"{float(ori):.0f}")
	return _format_sig_value_list(items)


def _format_sig_cellcounts_from_q(cell_counts: List[int], q_by_idx: Dict[int, Optional[float]]) -> str:
	items: List[str] = []
	for idx, n_val in enumerate(cell_counts):
		q = q_by_idx.get(idx, None)
		if q is None or not np.isfinite(q) or q >= 0.05:
			continue
		items.append(str(int(n_val)))
	return _format_sig_value_list(items)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _make_colored_stats_vbox(lines: List[str], fontsize: float, *, line_sep: float = 1.0) -> VPacker:
	"""Create a multi-line OffsetBox where numeric values < 0.05 are colored red.

	Notes:
	- Only numeric substrings are colored; surrounding labels remain black.
	- Numbers immediately following 'SF=' or 'Ori=' are treated as labels and kept black.
	"""
	line_boxes: List[HPacker] = []
	for line in lines:
		children: List[TextArea] = []
		last = 0
		for m in _PANEL_NUMBER_RE.finditer(line):
			start, end = m.span()
			if start > last:
				children.append(TextArea(
					line[last:start],
					textprops={"fontsize": fontsize, "color": "black"},
				))
			raw = line[start:end]
			# Determine whether this number is a label (SF/Ori) or a p/q value.
			prefix3 = line[max(0, start - 3):start]
			prefix4 = line[max(0, start - 4):start]
			exempt = (prefix3 in ("SF=", "sf=") or prefix4 in ("Ori=", "ori="))
			color = "black"
			if not exempt:
				try:
					val = float(m.group("num"))
				except Exception:
					val = None
				# Only color p/q-like values (typically within [0, 1]).
				if val is not None and np.isfinite(val) and 0.0 <= val <= 1.0 and val < 0.05:
					color = "red"
			children.append(TextArea(
				raw,
				textprops={"fontsize": fontsize, "color": color},
			))
			last = end
		if last < len(line):
			children.append(TextArea(
				line[last:],
				textprops={"fontsize": fontsize, "color": "black"},
			))
		line_boxes.append(HPacker(children=children, align="baseline", pad=0.0, sep=0.0))

	return VPacker(children=line_boxes, align="left", pad=0.0, sep=float(line_sep))


def _add_colored_stats_panel_to_figure(
	fig: plt.Figure,
	lines: List[str],
	*,
	x: float = 0.985,
	y: float = 0.50,
	fontsize: float = 6.8,
	n_cols: int = 1,
	line_sep: float = 1.0,
) -> None:
	"""Add a right-margin stats panel to a figure (anchored to right edge).

	If the panel is very tall, consider passing n_cols=2 (or higher) to
	reduce height by flowing the lines into multiple columns.
	"""
	n_cols = int(max(1, n_cols))
	if n_cols == 1:
		content_box = _make_colored_stats_vbox(lines, fontsize=fontsize, line_sep=line_sep)
	else:
		# Split lines into columns (roughly equal number of lines per column).
		chunks: List[List[str]] = [[] for _ in range(n_cols)]
		for i, line in enumerate(lines):
			chunks[i % n_cols].append(line)
		col_boxes = [_make_colored_stats_vbox(chunk, fontsize=fontsize, line_sep=line_sep) for chunk in chunks]
		content_box = HPacker(children=col_boxes, align="top", pad=0.0, sep=10.0)

	# Use top anchoring when y is near the top, otherwise center vertically.
	box_align = (1.0, 1.0) if float(y) >= 0.85 else (1.0, 0.5)
	ab = AnnotationBbox(
		content_box,
		(x, y),
		xycoords=fig.transFigure,
		box_alignment=box_align,
		frameon=True,
		bboxprops=dict(boxstyle="round", facecolor="white", alpha=0.7),
	)
	# Allow rendering beyond the nominal figure clip if needed.
	ab.set_clip_on(False)
	fig.add_artist(ab)


def _add_colored_stats_box_to_axes(ax: plt.Axes, lines: List[str], *, x: float = 0.98, y: float = 0.98, fontsize: float = 7.0) -> None:
	"""Add a small stats box inside an axes (top-right by default)."""
	vbox = _make_colored_stats_vbox(lines, fontsize=fontsize)
	ab = AnnotationBbox(
		vbox,
		(x, y),
		xycoords=ax.transAxes,
		box_alignment=(1.0, 1.0),
		frameon=True,
		bboxprops=dict(boxstyle="round", facecolor="white", alpha=0.7),
	)
	ax.add_artist(ab)


def _add_colored_stats_box_outside_axes(ax: plt.Axes, lines: List[str], *, x: float = 1.02, y: float = 0.98, fontsize: float = 7.0) -> None:
	"""Add a small stats box outside an axes (to the right)."""
	vbox = _make_colored_stats_vbox(lines, fontsize=fontsize)
	ab = AnnotationBbox(
		vbox,
		(x, y),
		xycoords=ax.transAxes,
		box_alignment=(0.0, 1.0),
		frameon=True,
		bboxprops=dict(boxstyle="round", facecolor="white", alpha=0.7),
	)
	# Ensure it can render outside axes bounds
	ab.set_clip_on(False)
	ax.add_artist(ab)


def _wrap_stats_panel_lines(lines: List[str], *, max_chars: int = 95) -> List[str]:
	"""Wrap/soft-split long stats-panel lines to avoid overlap with plots.

	We prefer splitting on ' | ' boundaries (used throughout the panel strings),
	falling back to textwrap for other long lines.
	"""
	out: List[str] = []
	for line in lines:
		line = str(line)
		if len(line) <= max_chars:
			out.append(line)
			continue
		if " | " in line:
			parts = line.split(" | ")
			cur = parts[0]
			for part in parts[1:]:
				candidate = f"{cur} | {part}"
				if len(candidate) <= max_chars:
					cur = candidate
				else:
					out.append(cur)
					cur = part
			out.append(cur)
			continue
		out.extend(textwrap.wrap(
			line,
			width=max_chars,
			break_long_words=False,
			break_on_hyphens=False,
		))
	return out


def _bh_fdr_correct(p_by_key: Dict[object, Optional[float]]) -> Dict[object, Optional[float]]:
	"""Benjamini–Hochberg FDR correction (q-values).

	Returns a dict with the same keys as input, mapping to adjusted p-values (q-values).
	Keys with None/non-finite p-values remain None.
	"""
	items: List[Tuple[object, float]] = []
	for k, p in p_by_key.items():
		if p is None:
			continue
		try:
			pf = float(p)
		except Exception:
			continue
		if np.isfinite(pf):
			items.append((k, pf))

	if not items:
		return {k: None for k in p_by_key.keys()}

	items_sorted = sorted(items, key=lambda t: t[1])
	m = len(items_sorted)
	q_vals = [min(1.0, (p * m) / (i + 1)) for i, (_, p) in enumerate(items_sorted)]
	# Enforce monotonicity of BH q-values
	for i in range(m - 2, -1, -1):
		q_vals[i] = min(q_vals[i], q_vals[i + 1])

	q_by_key: Dict[object, Optional[float]] = {k: None for k in p_by_key.keys()}
	for (k, _), q in zip(items_sorted, q_vals):
		q_by_key[k] = float(q)
	return q_by_key


def _safe_wilcoxon_p(x: np.ndarray, y: np.ndarray, **kwargs) -> Optional[float]:
	"""Wilcoxon signed-rank p-value with guard rails.

	Avoids scipy RuntimeWarnings in degenerate cases (e.g., all paired differences are 0).
	Returns:
	- 1.0 if all finite paired differences are exactly 0
	- None if insufficient finite data
	- otherwise the Wilcoxon p-value
	"""
	xv = np.asarray(x, dtype=float)
	yv = np.asarray(y, dtype=float)
	if xv.shape != yv.shape:
		return None
	mask = np.isfinite(xv) & np.isfinite(yv)
	xv = xv[mask]
	yv = yv[mask]
	if xv.size == 0:
		return None
	d = yv - xv
	if np.all(d == 0):
		return 1.0
	try:
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=RuntimeWarning)
			res = wilcoxon(xv, yv, **kwargs)
		p = float(res.pvalue) if res is not None else np.nan
		return float(p) if np.isfinite(p) else None
	except Exception:
		return None


def _safe_onesample_ttest_p(x: np.ndarray, popmean: float, *, alternative: str = 'two-sided') -> Optional[float]:
	"""One-sample t-test p-value with fallback for older SciPy signatures."""
	xv = np.asarray(x, dtype=float)
	xv = xv[np.isfinite(xv)]
	if xv.size < 2:
		return None
	try:
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=RuntimeWarning)
			try:
				res = ttest_1samp(xv, float(popmean), nan_policy='omit', alternative=alternative)
			except TypeError:
				res = ttest_1samp(xv, float(popmean), nan_policy='omit')
		p = float(res.pvalue) if res is not None else np.nan
		if not np.isfinite(p):
			return None
		if alternative in {'greater', 'less'} and 'alternative' not in getattr(ttest_1samp, '__code__', type('obj', (), {'co_varnames': ()})) .co_varnames:
			stat = float(res.statistic) if res is not None and np.isfinite(res.statistic) else np.nan
			if not np.isfinite(stat):
				return None
			if alternative == 'greater':
				p = (p / 2.0) if stat > 0 else (1.0 - p / 2.0)
			else:
				p = (p / 2.0) if stat < 0 else (1.0 - p / 2.0)
		return float(p)
	except Exception:
		return None


def _safe_onesample_wilcoxon_p(x: np.ndarray, center: float, *, alternative: str = 'two-sided') -> Optional[float]:
	"""One-sample Wilcoxon signed-rank p-value against a scalar center."""
	xv = np.asarray(x, dtype=float)
	xv = xv[np.isfinite(xv)]
	if xv.size < 2:
		return None
	d = xv - float(center)
	if np.all(d == 0):
		return 1.0
	try:
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=RuntimeWarning)
			res = wilcoxon(d, zero_method='wilcox', alternative=alternative)
		p = float(res.pvalue) if res is not None else np.nan
		return float(p) if np.isfinite(p) else None
	except Exception:
		return None


def _asterisks_from_p(p: float) -> str:
	if p < 0.001:
		return "***"
	if p < 0.01:
		return "**"
	return "*"


def fit_circle_kasa(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
	"""Algebraic least-squares circle fit.

	Returns (cx, cy, r). Returns (nan, nan, nan) if fit fails.
	"""
	x = np.asarray(x, dtype=float)
	y = np.asarray(y, dtype=float)
	finite = np.isfinite(x) & np.isfinite(y)
	x = x[finite]
	y = y[finite]
	if x.size < 3:
		return (np.nan, np.nan, np.nan)

	A = np.column_stack([x, y, np.ones_like(x)])
	b = -(x**2 + y**2)
	try:
		D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
		cx = -D / 2.0
		cy = -E / 2.0
		r2 = (D**2 + E**2) / 4.0 - F
		if not np.isfinite(r2) or r2 <= 0:
			return (np.nan, np.nan, np.nan)
		r = float(np.sqrt(r2))
		return (float(cx), float(cy), r)
	except Exception:
		return (np.nan, np.nan, np.nan)


def circle_geometry_metrics(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
	"""Compute circle-related geometry metrics for a set of points.

	Metrics:
	- radial_cv: std(r_i)/mean(r_i) around fitted center
	- circle_rmse: RMS(r_i - r_fit)
	- circle_rmse_norm: circle_rmse / r_fit
	- angle_spacing_rmse_norm: RMS(Δθ_i - 2π/n) / (2π/n)
	"""
	cx, cy, r_fit = fit_circle_kasa(x, y)
	if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(r_fit) and r_fit > 0):
		return {
			"cx": np.nan,
			"cy": np.nan,
			"r_fit": np.nan,
			"radial_cv": np.nan,
			"circle_rmse": np.nan,
			"circle_rmse_norm": np.nan,
			"angle_spacing_rmse_norm": np.nan,
		}

	x = np.asarray(x, dtype=float)
	y = np.asarray(y, dtype=float)
	finite = np.isfinite(x) & np.isfinite(y)
	x = x[finite]
	y = y[finite]
	if x.size < 3:
		return {
			"cx": cx,
			"cy": cy,
			"r_fit": r_fit,
			"radial_cv": np.nan,
			"circle_rmse": np.nan,
			"circle_rmse_norm": np.nan,
			"angle_spacing_rmse_norm": np.nan,
		}

	r_i = np.sqrt((x - cx)**2 + (y - cy)**2)
	mean_r = float(np.nanmean(r_i))
	if not np.isfinite(mean_r) or mean_r <= 0 or np.sum(np.isfinite(r_i)) < 2:
		radial_cv = np.nan
	else:
		radial_cv = float(np.nanstd(r_i, ddof=1) / mean_r)

	res = r_i - r_fit
	circle_rmse = float(np.sqrt(np.nanmean(res**2))) if np.any(np.isfinite(res)) else np.nan
	circle_rmse_norm = float(circle_rmse / r_fit) if (np.isfinite(circle_rmse) and r_fit > 0) else np.nan

	# Angular spacing deviation around fitted center
	angles = np.mod(np.arctan2(y - cy, x - cx), 2 * np.pi)
	angles = angles[np.isfinite(angles)]
	angles = np.sort(angles)
	if angles.size >= 3:
		diffs = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
		ideal = 2 * np.pi / angles.size
		angle_spacing_rmse_norm = float(np.sqrt(np.mean((diffs - ideal) ** 2)) / ideal) if ideal > 0 else np.nan
	else:
		angle_spacing_rmse_norm = np.nan

	return {
		"cx": cx,
		"cy": cy,
		"r_fit": r_fit,
		"radial_cv": radial_cv,
		"circle_rmse": circle_rmse,
		"circle_rmse_norm": circle_rmse_norm,
		"angle_spacing_rmse_norm": angle_spacing_rmse_norm,
	}

def safe_load_npy(path: str) -> Optional[np.ndarray]:
	"""Load an .npy file if it exists; otherwise return None."""

	if not os.path.exists(path):
		print(f"[missing] {path}")
		return None
	try:
		return np.load(path, allow_pickle=True)
	except Exception as exc:  # pragma: no cover - defensive
		print(f"[error] could not load {path}: {exc}")
		return None


def safe_load_npz(path: str) -> Optional[np.lib.npyio.NpzFile]:
	"""Load an .npz file safely."""

	if not os.path.exists(path):
		print(f"[missing] {path}")
		return None
	try:
		return np.load(path, allow_pickle=True)
	except Exception as exc:  # pragma: no cover - defensive
		print(f"[error] could not load {path}: {exc}")
		return None


def safe_load_first_existing_npy(paths: List[str]) -> Optional[np.ndarray]:
	"""Load the first existing .npy file from a list of candidate paths."""
	for path in paths:
		if os.path.exists(path):
			return safe_load_npy(path)
	if paths:
		print(f"[missing] {paths[0]}")
	return None


def get_from_npz(npz: Optional[np.lib.npyio.NpzFile], candidates: List[str]) -> Optional[np.ndarray]:
	"""Return first existing key from candidates inside an npz file."""

	if npz is None:
		return None
	for key in candidates:
		if key in npz.files:
			return npz[key]
	return None


def align_to_targets(values: np.ndarray, session_sfs: np.ndarray) -> np.ndarray:
	"""Align a 1D vector defined over session_sfs onto TARGET_SF_VALUES.

	Unknown or mismatched SF entries are filled with NaN. Uses nearest
	matching within a small tolerance.
	"""

	out = np.full_like(TARGET_SF_VALUES, np.nan, dtype=float)
	if session_sfs is None or values is None:
		return out

	session_sfs = np.asarray(session_sfs, dtype=float)
	values = np.asarray(values, dtype=float)

	for idx, target in enumerate(TARGET_SF_VALUES):
		# Use tight tolerance to avoid accidental cross-matching.
		diffs = np.abs(session_sfs - target)
		if np.any(diffs < 1e-3):
			match_idx = int(np.argmin(diffs))
			if match_idx < values.shape[-1]:
				out[idx] = values[match_idx]
	return out


def align_rows_to_targets(matrix: np.ndarray, session_sfs: np.ndarray) -> np.ndarray:
	"""Align a (n_sf, n_feat) matrix along its first axis to TARGET_SF_VALUES."""

	if matrix is None or session_sfs is None:
		return np.full((len(TARGET_SF_VALUES), matrix.shape[1] if matrix is not None else 0), np.nan, dtype=float)

	out = np.full((len(TARGET_SF_VALUES), matrix.shape[1]), np.nan, dtype=float)
	session_sfs = np.asarray(session_sfs, dtype=float)

	for idx, target in enumerate(TARGET_SF_VALUES):
		diffs = np.abs(session_sfs - target)
		if np.any(diffs < 1e-3):
			sf_idx = int(np.argmin(diffs))
			if sf_idx < matrix.shape[0]:
				out[idx] = matrix[sf_idx]
	return out


def align_matrix_to_targets(matrix: np.ndarray, session_sfs: np.ndarray) -> np.ndarray:
	"""Align a (n_items, n_sf) matrix to TARGET_SF_VALUES along sf axis."""

	if matrix is None or session_sfs is None:
		return np.full((matrix.shape[0] if matrix is not None else 0, len(TARGET_SF_VALUES)), np.nan)

	out = np.full((matrix.shape[0], len(TARGET_SF_VALUES)), np.nan, dtype=float)
	session_sfs = np.asarray(session_sfs, dtype=float)

	for idx, target in enumerate(TARGET_SF_VALUES):
		diffs = np.abs(session_sfs - target)
		if np.any(diffs < 1e-3):
			sf_idx = int(np.argmin(diffs))
			if sf_idx < matrix.shape[1]:
				out[:, idx] = matrix[:, sf_idx]
	return out


def nan_mean_sem(stack: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
	"""Compute nanmean and SEM along axis."""

	stack = np.asarray(stack, dtype=float)
	# Treat non-finite values as missing.
	x = np.where(np.isfinite(stack), stack, np.nan)
	count = np.sum(~np.isnan(x), axis=axis)

	# Mean without triggering "Mean of empty slice" warnings.
	sum_x = np.nansum(x, axis=axis)
	mean = sum_x / np.maximum(count, 1)
	mean = np.where(count == 0, np.nan, mean)

	# SEM with ddof=1, but only where count>=2 (otherwise undefined).
	mean_expanded = np.expand_dims(mean, axis=axis)
	ss = np.nansum((x - mean_expanded) ** 2, axis=axis)
	var = ss / np.maximum(count - 1, 1)
	var = np.where(count >= 2, var, np.nan)
	sem = np.sqrt(var) / np.sqrt(np.maximum(count, 1))
	sem = np.where(count >= 2, sem, np.nan)
	return mean, sem


def to_nan_array(value: Optional[np.ndarray]) -> np.ndarray:
	"""Convert value to float array, defaulting to NaN if None."""

	if value is None:
		return np.array(np.nan, dtype=float)
	return np.array(value, dtype=float)


def has_any_finite(arr: np.ndarray) -> bool:
	"""Check if array has at least one finite value."""

	return np.isfinite(arr).any()


def plot_with_error(x: np.ndarray, y_mean: np.ndarray, y_sem: np.ndarray, label: str, color: str, ax: plt.Axes, marker: str = "o") -> None:
	"""Line plot with SEM error bars (no shading)."""

	fmt = f"-{marker}" if marker else "-"
	ax.errorbar(x, y_mean, yerr=y_sem, fmt=fmt, capsize=3, color=color, label=label)


def compute_paired_pvalues(metric_key: str) -> Dict[str, Optional[float]]:
	"""Compute Wilcoxon signed-rank p-values for pre vs post training within WT and Rescued groups separately.
	
	Returns dict with 'WT' and 'RESCUED' keys, each containing p-value or None if insufficient data.
	"""

	results: Dict[str, Optional[float]] = {"WT": None, "RESCUED": None}

	for condition in ["WT", "RESCUED"]:
		pre_values = []
		post_values = []
		
		pairs = MOUSE_PAIRS.get(condition, [])
		
		for session_before, session_after in pairs:
			# Load metric values
			session_before_summary = prepare_session_summaries(session_before)
			session_after_summary = prepare_session_summaries(session_after)

			val_before = session_before_summary[metric_key]
			val_after = session_after_summary[metric_key]

			if val_before is None or val_after is None:
				continue

			# Convert to scalar if needed
			if np.ndim(val_before) > 0:
				val_before = np.nanmean(val_before)
			if np.ndim(val_after) > 0:
				val_after = np.nanmean(val_after)

			if np.isfinite(val_before) and np.isfinite(val_after):
				pre_values.append(float(val_before))
				post_values.append(float(val_after))

		if len(pre_values) < 2:
			continue

		pre_values = np.array(pre_values)
		post_values = np.array(post_values)

		try:
			pval = _safe_wilcoxon_p(pre_values, post_values)
			results[condition] = float(pval) if (pval is not None and np.isfinite(pval)) else None
		except Exception:
			pass

	return results


def _posthoc_training_pvalues_within_each_sf_rm(metric_key: str, pairs: List[Tuple[str, str]]) -> Dict[int, Optional[float]]:
	"""Post-hoc simple effects: test Training (Pre vs Post) within each SF.

	Returns dict keyed by sf_idx (index into TARGET_SF_VALUES).
	Uses AnovaRM with within=['training'] on paired mouse data.
	"""
	results: Dict[int, Optional[float]] = {}
	per_sf_rows: Dict[int, List[Dict[str, object]]] = {i: [] for i in range(len(TARGET_SF_VALUES))}
	for session_before, session_after in pairs:
		s_before = prepare_session_summaries(session_before)
		s_after = prepare_session_summaries(session_after)
		v_before = s_before.get(metric_key)
		v_after = s_after.get(metric_key)
		if v_before is None or v_after is None:
			continue
		v_before = np.asarray(v_before, dtype=float)
		v_after = np.asarray(v_after, dtype=float)
		if v_before.ndim == 0:
			v_before = np.full(len(TARGET_SF_VALUES), float(v_before))
			v_after = np.full(len(TARGET_SF_VALUES), float(v_after))
		if v_before.ndim != 1 or v_after.ndim != 1:
			continue
		min_len = min(len(TARGET_SF_VALUES), v_before.size, v_after.size)
		subject = _mouse_id_from_session_rel(session_before)
		for sf_idx in range(1, min_len):
			vb = float(v_before[sf_idx])
			va = float(v_after[sf_idx])
			if np.isfinite(vb) and np.isfinite(va):
				per_sf_rows[sf_idx].append({"subject": subject, "training": "Pre", "value": vb})
				per_sf_rows[sf_idx].append({"subject": subject, "training": "Post", "value": va})

	for sf_idx, rows in per_sf_rows.items():
		if sf_idx == 0 or len(rows) == 0:
			continue
		df = pd.DataFrame(rows)
		grid = df.pivot_table(index='subject', columns=['training'], values='value', aggfunc='mean')
		grid = grid.dropna(axis=0, how='any')
		if grid.shape[0] < 2:
			results[sf_idx] = None
			continue
		balanced = pd.DataFrame({
			"subject": np.repeat(grid.index.values, 2),
			"training": np.tile(['Pre', 'Post'], grid.shape[0]),
			"value": np.r_[grid['Pre'].values, grid['Post'].values],
		})
		try:
			res = AnovaRM(balanced, depvar='value', subject='subject', within=['training']).fit()
			p = res.anova_table.loc['training', 'Pr > F'] if 'training' in res.anova_table.index else np.nan
			results[sf_idx] = float(p) if np.isfinite(p) else None
		except Exception:
			results[sf_idx] = None

	return results


def _posthoc_training_pvalues_within_each_sf_art(metric_key: str, pairs: List[Tuple[str, str]]) -> Dict[int, Optional[float]]:
	"""Post-hoc simple effects: ART-style Training test within each SF.

	Implementation: subject-center values, rank, then AnovaRM within=['training'].
	"""
	results: Dict[int, Optional[float]] = {}
	per_sf_rows: Dict[int, List[Dict[str, object]]] = {i: [] for i in range(len(TARGET_SF_VALUES))}
	for session_before, session_after in pairs:
		s_before = prepare_session_summaries(session_before)
		s_after = prepare_session_summaries(session_after)
		v_before = s_before.get(metric_key)
		v_after = s_after.get(metric_key)
		if v_before is None or v_after is None:
			continue
		v_before = np.asarray(v_before, dtype=float)
		v_after = np.asarray(v_after, dtype=float)
		if v_before.ndim == 0:
			v_before = np.full(len(TARGET_SF_VALUES), float(v_before))
			v_after = np.full(len(TARGET_SF_VALUES), float(v_after))
		if v_before.ndim != 1 or v_after.ndim != 1:
			continue
		min_len = min(len(TARGET_SF_VALUES), v_before.size, v_after.size)
		subject = _mouse_id_from_session_rel(session_before)
		for sf_idx in range(1, min_len):
			vb = float(v_before[sf_idx])
			va = float(v_after[sf_idx])
			if np.isfinite(vb) and np.isfinite(va):
				per_sf_rows[sf_idx].append({"subject": subject, "training": "Pre", "value": vb})
				per_sf_rows[sf_idx].append({"subject": subject, "training": "Post", "value": va})

	for sf_idx, rows in per_sf_rows.items():
		if sf_idx == 0 or len(rows) == 0:
			continue
		df = pd.DataFrame(rows)
		grid = df.pivot_table(index='subject', columns=['training'], values='value', aggfunc='mean')
		grid = grid.dropna(axis=0, how='any')
		if grid.shape[0] < 2:
			results[sf_idx] = None
			continue
		balanced = pd.DataFrame({
			"subject": np.repeat(grid.index.values, 2),
			"training": np.tile(['Pre', 'Post'], grid.shape[0]),
			"value": np.r_[grid['Pre'].values, grid['Post'].values],
		})
		try:
			balanced['value_centered'] = balanced['value'] - balanced.groupby('subject')['value'].transform('mean')
			balanced['rank'] = balanced['value_centered'].rank(method='average')
			res = AnovaRM(balanced, depvar='rank', subject='subject', within=['training']).fit()
			p = res.anova_table.loc['training', 'Pr > F'] if 'training' in res.anova_table.index else np.nan
			results[sf_idx] = float(p) if np.isfinite(p) else None
		except Exception:
			results[sf_idx] = None

	return results


def _posthoc_training_pvalues_within_each_sf_ttest(metric_key: str, pairs: List[Tuple[str, str]]) -> Dict[int, Optional[float]]:
	"""Post-hoc simple effects: paired t-test (Pre vs Post) within each SF.

	Returns dict keyed by sf_idx (index into TARGET_SF_VALUES).
	"""
	results: Dict[int, Optional[float]] = {}
	per_sf_rows: Dict[int, List[Dict[str, object]]] = {i: [] for i in range(len(TARGET_SF_VALUES))}
	for session_before, session_after in pairs:
		s_before = prepare_session_summaries(session_before)
		s_after = prepare_session_summaries(session_after)
		v_before = s_before.get(metric_key)
		v_after = s_after.get(metric_key)
		if v_before is None or v_after is None:
			continue
		v_before = np.asarray(v_before, dtype=float)
		v_after = np.asarray(v_after, dtype=float)
		if v_before.ndim == 0:
			v_before = np.full(len(TARGET_SF_VALUES), float(v_before))
			v_after = np.full(len(TARGET_SF_VALUES), float(v_after))
		if v_before.ndim != 1 or v_after.ndim != 1:
			continue
		min_len = min(len(TARGET_SF_VALUES), v_before.size, v_after.size)
		subject = _mouse_id_from_session_rel(session_before)
		for sf_idx in range(1, min_len):
			vb = float(v_before[sf_idx])
			va = float(v_after[sf_idx])
			if np.isfinite(vb) and np.isfinite(va):
				per_sf_rows[sf_idx].append({"subject": subject, "training": "Pre", "value": vb})
				per_sf_rows[sf_idx].append({"subject": subject, "training": "Post", "value": va})

	for sf_idx, rows in per_sf_rows.items():
		if sf_idx == 0 or len(rows) == 0:
			continue
		df = pd.DataFrame(rows)
		grid = df.pivot_table(index='subject', columns=['training'], values='value', aggfunc='mean')
		grid = grid.dropna(axis=0, how='any')
		if grid.shape[0] < 2:
			results[sf_idx] = None
			continue
		try:
			stat = ttest_rel(grid['Pre'].values, grid['Post'].values, nan_policy='omit')
			p = float(stat.pvalue) if stat is not None else np.nan
			results[sf_idx] = float(p) if np.isfinite(p) else None
		except Exception:
			results[sf_idx] = None

	return results


def _posthoc_training_pvalues_within_each_sf_wilcoxon(metric_key: str, pairs: List[Tuple[str, str]]) -> Dict[int, Optional[float]]:
	"""Post-hoc simple effects: Wilcoxon signed-rank (Pre vs Post) within each SF.

	Returns dict keyed by sf_idx (index into TARGET_SF_VALUES).
	"""
	results: Dict[int, Optional[float]] = {}
	per_sf_rows: Dict[int, List[Dict[str, object]]] = {i: [] for i in range(len(TARGET_SF_VALUES))}
	for session_before, session_after in pairs:
		s_before = prepare_session_summaries(session_before)
		s_after = prepare_session_summaries(session_after)
		v_before = s_before.get(metric_key)
		v_after = s_after.get(metric_key)
		if v_before is None or v_after is None:
			continue
		v_before = np.asarray(v_before, dtype=float)
		v_after = np.asarray(v_after, dtype=float)
		if v_before.ndim == 0:
			v_before = np.full(len(TARGET_SF_VALUES), float(v_before))
			v_after = np.full(len(TARGET_SF_VALUES), float(v_after))
		if v_before.ndim != 1 or v_after.ndim != 1:
			continue
		min_len = min(len(TARGET_SF_VALUES), v_before.size, v_after.size)
		subject = _mouse_id_from_session_rel(session_before)
		for sf_idx in range(1, min_len):
			vb = float(v_before[sf_idx])
			va = float(v_after[sf_idx])
			if np.isfinite(vb) and np.isfinite(va):
				per_sf_rows[sf_idx].append({"subject": subject, "training": "Pre", "value": vb})
				per_sf_rows[sf_idx].append({"subject": subject, "training": "Post", "value": va})

	for sf_idx, rows in per_sf_rows.items():
		if sf_idx == 0 or len(rows) == 0:
			continue
		df = pd.DataFrame(rows)
		grid = df.pivot_table(index='subject', columns=['training'], values='value', aggfunc='mean')
		grid = grid.dropna(axis=0, how='any')
		if grid.shape[0] < 2:
			results[sf_idx] = None
			continue
		try:
			p = _safe_wilcoxon_p(grid['Pre'].values, grid['Post'].values, zero_method='wilcox', alternative='two-sided')
			results[sf_idx] = float(p) if (p is not None and np.isfinite(p)) else None
		except Exception:
			results[sf_idx] = None

	return results


def _posthoc_training_pvalues_within_each_orientation_rm(metric_key: str, pairs: List[Tuple[str, str]]) -> Dict[int, Optional[float]]:
	"""Post-hoc simple effects: test Training (Pre vs Post) within each orientation index."""
	results: Dict[int, Optional[float]] = {}
	# Determine max orientation length over available sessions
	max_len = 0
	for session_before, _ in pairs:
		oris = prepare_session_summaries(session_before).get('orientations')
		if oris is not None:
			max_len = max(max_len, len(oris))
	if max_len < 2:
		return results

	per_ori_rows: Dict[int, List[Dict[str, object]]] = {i: [] for i in range(max_len)}
	for session_before, session_after in pairs:
		s_before = prepare_session_summaries(session_before)
		s_after = prepare_session_summaries(session_after)
		v_before = s_before.get(metric_key)
		v_after = s_after.get(metric_key)
		oris = s_before.get('orientations')
		if v_before is None or v_after is None or oris is None:
			continue
		v_before = np.asarray(v_before, dtype=float)
		v_after = np.asarray(v_after, dtype=float)
		if v_before.ndim != 1 or v_after.ndim != 1:
			continue
		min_len = min(len(oris), v_before.size, v_after.size)
		subject = _mouse_id_from_session_rel(session_before)
		for oi in range(min_len):
			vb = float(v_before[oi])
			va = float(v_after[oi])
			if np.isfinite(vb) and np.isfinite(va):
				per_ori_rows[oi].append({"subject": subject, "training": "Pre", "value": vb})
				per_ori_rows[oi].append({"subject": subject, "training": "Post", "value": va})

	for oi, rows in per_ori_rows.items():
		if len(rows) == 0:
			continue
		df = pd.DataFrame(rows)
		grid = df.pivot_table(index='subject', columns=['training'], values='value', aggfunc='mean')
		grid = grid.dropna(axis=0, how='any')
		if grid.shape[0] < 2:
			results[oi] = None
			continue
		balanced = pd.DataFrame({
			"subject": np.repeat(grid.index.values, 2),
			"training": np.tile(['Pre', 'Post'], grid.shape[0]),
			"value": np.r_[grid['Pre'].values, grid['Post'].values],
		})
		try:
			res = AnovaRM(balanced, depvar='value', subject='subject', within=['training']).fit()
			p = res.anova_table.loc['training', 'Pr > F'] if 'training' in res.anova_table.index else np.nan
			results[oi] = float(p) if np.isfinite(p) else None
		except Exception:
			results[oi] = None
	return results


def _posthoc_training_pvalues_within_each_orientation_art(metric_key: str, pairs: List[Tuple[str, str]]) -> Dict[int, Optional[float]]:
	"""Post-hoc simple effects: ART-style Training test within each orientation index."""
	results: Dict[int, Optional[float]] = {}
	max_len = 0
	for session_before, _ in pairs:
		oris = prepare_session_summaries(session_before).get('orientations')
		if oris is not None:
			max_len = max(max_len, len(oris))
	if max_len < 2:
		return results

	per_ori_rows: Dict[int, List[Dict[str, object]]] = {i: [] for i in range(max_len)}
	for session_before, session_after in pairs:
		s_before = prepare_session_summaries(session_before)
		s_after = prepare_session_summaries(session_after)
		v_before = s_before.get(metric_key)
		v_after = s_after.get(metric_key)
		oris = s_before.get('orientations')
		if v_before is None or v_after is None or oris is None:
			continue
		v_before = np.asarray(v_before, dtype=float)
		v_after = np.asarray(v_after, dtype=float)
		if v_before.ndim != 1 or v_after.ndim != 1:
			continue
		min_len = min(len(oris), v_before.size, v_after.size)
		subject = _mouse_id_from_session_rel(session_before)
		for oi in range(min_len):
			vb = float(v_before[oi])
			va = float(v_after[oi])
			if np.isfinite(vb) and np.isfinite(va):
				per_ori_rows[oi].append({"subject": subject, "training": "Pre", "value": vb})
				per_ori_rows[oi].append({"subject": subject, "training": "Post", "value": va})

	for oi, rows in per_ori_rows.items():
		if len(rows) == 0:
			continue
		df = pd.DataFrame(rows)
		grid = df.pivot_table(index='subject', columns=['training'], values='value', aggfunc='mean')
		grid = grid.dropna(axis=0, how='any')
		if grid.shape[0] < 2:
			results[oi] = None
			continue
		balanced = pd.DataFrame({
			"subject": np.repeat(grid.index.values, 2),
			"training": np.tile(['Pre', 'Post'], grid.shape[0]),
			"value": np.r_[grid['Pre'].values, grid['Post'].values],
		})
		try:
			balanced['value_centered'] = balanced['value'] - balanced.groupby('subject')['value'].transform('mean')
			balanced['rank'] = balanced['value_centered'].rank(method='average')
			res = AnovaRM(balanced, depvar='rank', subject='subject', within=['training']).fit()
			p = res.anova_table.loc['training', 'Pr > F'] if 'training' in res.anova_table.index else np.nan
			results[oi] = float(p) if np.isfinite(p) else None
		except Exception:
			results[oi] = None
	return results


def _posthoc_training_pvalues_within_each_orientation_ttest(metric_key: str, pairs: List[Tuple[str, str]]) -> Dict[int, Optional[float]]:
	"""Post-hoc simple effects: paired t-test (Pre vs Post) within each orientation index."""
	results: Dict[int, Optional[float]] = {}
	max_len = 0
	for session_before, _ in pairs:
		oris = prepare_session_summaries(session_before).get('orientations')
		if oris is not None:
			max_len = max(max_len, len(oris))
	if max_len < 2:
		return results

	per_ori_rows: Dict[int, List[Dict[str, object]]] = {i: [] for i in range(max_len)}
	for session_before, session_after in pairs:
		s_before = prepare_session_summaries(session_before)
		s_after = prepare_session_summaries(session_after)
		v_before = s_before.get(metric_key)
		v_after = s_after.get(metric_key)
		oris = s_before.get('orientations')
		if v_before is None or v_after is None or oris is None:
			continue
		v_before = np.asarray(v_before, dtype=float)
		v_after = np.asarray(v_after, dtype=float)
		if v_before.ndim != 1 or v_after.ndim != 1:
			continue
		min_len = min(len(oris), v_before.size, v_after.size)
		subject = _mouse_id_from_session_rel(session_before)
		for oi in range(min_len):
			vb = float(v_before[oi])
			va = float(v_after[oi])
			if np.isfinite(vb) and np.isfinite(va):
				per_ori_rows[oi].append({"subject": subject, "training": "Pre", "value": vb})
				per_ori_rows[oi].append({"subject": subject, "training": "Post", "value": va})

	for oi, rows in per_ori_rows.items():
		if len(rows) == 0:
			continue
		df = pd.DataFrame(rows)
		grid = df.pivot_table(index='subject', columns=['training'], values='value', aggfunc='mean')
		grid = grid.dropna(axis=0, how='any')
		if grid.shape[0] < 2:
			results[oi] = None
			continue
		try:
			stat = ttest_rel(grid['Pre'].values, grid['Post'].values, nan_policy='omit')
			p = float(stat.pvalue) if stat is not None else np.nan
			results[oi] = float(p) if np.isfinite(p) else None
		except Exception:
			results[oi] = None
	return results


def _posthoc_training_pvalues_within_each_orientation_wilcoxon(metric_key: str, pairs: List[Tuple[str, str]]) -> Dict[int, Optional[float]]:
	"""Post-hoc simple effects: Wilcoxon signed-rank (Pre vs Post) within each orientation index."""
	results: Dict[int, Optional[float]] = {}
	max_len = 0
	for session_before, _ in pairs:
		oris = prepare_session_summaries(session_before).get('orientations')
		if oris is not None:
			max_len = max(max_len, len(oris))
	if max_len < 2:
		return results

	per_ori_rows: Dict[int, List[Dict[str, object]]] = {i: [] for i in range(max_len)}
	for session_before, session_after in pairs:
		s_before = prepare_session_summaries(session_before)
		s_after = prepare_session_summaries(session_after)
		v_before = s_before.get(metric_key)
		v_after = s_after.get(metric_key)
		oris = s_before.get('orientations')
		if v_before is None or v_after is None or oris is None:
			continue
		v_before = np.asarray(v_before, dtype=float)
		v_after = np.asarray(v_after, dtype=float)
		if v_before.ndim != 1 or v_after.ndim != 1:
			continue
		min_len = min(len(oris), v_before.size, v_after.size)
		subject = _mouse_id_from_session_rel(session_before)
		for oi in range(min_len):
			vb = float(v_before[oi])
			va = float(v_after[oi])
			if np.isfinite(vb) and np.isfinite(va):
				per_ori_rows[oi].append({"subject": subject, "training": "Pre", "value": vb})
				per_ori_rows[oi].append({"subject": subject, "training": "Post", "value": va})

	for oi, rows in per_ori_rows.items():
		if len(rows) == 0:
			continue
		df = pd.DataFrame(rows)
		grid = df.pivot_table(index='subject', columns=['training'], values='value', aggfunc='mean')
		grid = grid.dropna(axis=0, how='any')
		if grid.shape[0] < 2:
			results[oi] = None
			continue
		try:
			p = _safe_wilcoxon_p(grid['Pre'].values, grid['Post'].values, zero_method='wilcox', alternative='two-sided')
			results[oi] = float(p) if (p is not None and np.isfinite(p)) else None
		except Exception:
			results[oi] = None
	return results


def compute_rmanova_pvalues(metric_key: str) -> Dict[str, Dict[str, Optional[float]]]:
	"""Compute two-way repeated measures ANOVA for each group (WT, RESCUED).
	
	Tests main effects of Training and SF; optionally Training×SF interaction.

	SF is treated as a within-subject (repeated-measures) factor.
	Returns dict structure: {group: {"training_p": p, "sf_p": p, "interaction_p": p}}
	"""
	results: Dict[str, Dict[str, Optional[float]]] = {
		"WT": {"training_p": None, "sf_p": None, "interaction_p": None},
		"RESCUED": {"training_p": None, "sf_p": None, "interaction_p": None},
	}

	for condition in ["WT", "RESCUED"]:
		pairs = MOUSE_PAIRS.get(condition, [])
		rows: List[Dict[str, object]] = []
		for session_before, session_after in pairs:
			s_before = prepare_session_summaries(session_before)
			s_after = prepare_session_summaries(session_after)
			val_before = s_before.get(metric_key)
			val_after = s_after.get(metric_key)
			if val_before is None or val_after is None:
				continue
			val_before = np.asarray(val_before, dtype=float)
			val_after = np.asarray(val_after, dtype=float)
			if val_before.ndim == 0:
				val_before = np.full(len(TARGET_SF_VALUES), float(val_before))
				val_after = np.full(len(TARGET_SF_VALUES), float(val_after))
			if val_before.ndim != 1 or val_after.ndim != 1:
				continue
			min_len = min(len(TARGET_SF_VALUES), val_before.size, val_after.size)
			if min_len <= 2:
				continue
			subject = _mouse_id_from_session_rel(session_before)
			for sf_idx in range(1, min_len):
				sf = str(float(TARGET_SF_VALUES[sf_idx]))
				vb = float(val_before[sf_idx])
				va = float(val_after[sf_idx])
				if np.isfinite(vb) and np.isfinite(va):
					rows.append({"subject": subject, "training": "Pre", "sf": sf, "value": vb})
					rows.append({"subject": subject, "training": "Post", "sf": sf, "value": va})

		df = pd.DataFrame(rows)
		if df.empty:
			continue

		# Enforce balanced grid: keep SF levels present (both training) for all subjects.
		required_cols = df.pivot_table(index='subject', columns=['training', 'sf'], values='value', aggfunc='mean')
		if required_cols.empty:
			continue
		# SF intersection across subjects
		all_sfs = sorted({c[1] for c in required_cols.columns})
		if len(all_sfs) < 2:
			continue
		good_sfs: List[str] = []
		for sf in all_sfs:
			if (('Pre', sf) in required_cols.columns) and (('Post', sf) in required_cols.columns):
				col = required_cols[[('Pre', sf), ('Post', sf)]]
				if not col.isna().any().any():
					good_sfs.append(sf)
		if len(good_sfs) < 2:
			continue
		df = df[df['sf'].isin(good_sfs)].copy()
		# Keep only subjects with full grid on the reduced SF set
		grid = df.pivot_table(index='subject', columns=['training', 'sf'], values='value', aggfunc='mean')
		grid = grid.dropna(axis=0, how='any')
		if grid.shape[0] < 2:
			continue
		# Rebuild long df from the balanced grid (avoids duplicates)
		balanced_rows: List[Dict[str, object]] = []
		for subject in grid.index:
			for training in ['Pre', 'Post']:
				for sf in good_sfs:
					balanced_rows.append({
						"subject": subject,
						"training": training,
						"sf": sf,
						"value": float(grid.loc[subject, (training, sf)]),
					})
		df_bal = pd.DataFrame(balanced_rows)
		try:
			if RM_ANOVA_INCLUDE_INTERACTION_TERMS:
				res = AnovaRM(df_bal, depvar='value', subject='subject', within=['training', 'sf']).fit()
				tab = res.anova_table
				p_train = tab.loc['training', 'Pr > F'] if 'training' in tab.index else np.nan
				p_sf = tab.loc['sf', 'Pr > F'] if 'sf' in tab.index else np.nan
				p_int = tab.loc['training:sf', 'Pr > F'] if 'training:sf' in tab.index else np.nan
			else:
				# Reduced (additive) model: estimate each main effect by collapsing over the other factor.
				df_train = df_bal.groupby(['subject', 'training'], as_index=False)['value'].mean()
				res_train = AnovaRM(df_train, depvar='value', subject='subject', within=['training']).fit()
				tab_train = res_train.anova_table
				p_train = tab_train.loc['training', 'Pr > F'] if 'training' in tab_train.index else np.nan

				df_sf = df_bal.groupby(['subject', 'sf'], as_index=False)['value'].mean()
				res_sf = AnovaRM(df_sf, depvar='value', subject='subject', within=['sf']).fit()
				tab_sf = res_sf.anova_table
				p_sf = tab_sf.loc['sf', 'Pr > F'] if 'sf' in tab_sf.index else np.nan
				p_int = np.nan
			results[condition] = {
				"training_p": float(p_train) if np.isfinite(p_train) else None,
				"sf_p": float(p_sf) if np.isfinite(p_sf) else None,
				"interaction_p": float(p_int) if np.isfinite(p_int) else None,
			}
		except Exception as exc:
			print(f"[debug] RM-ANOVA computation for {condition}/{metric_key}: {type(exc).__name__}")
			continue

	return results


def compute_rmanova_pvalues_orientation(metric_key: str) -> Dict[str, Dict[str, Optional[float]]]:
	"""Compute two-way repeated measures ANOVA for orientation-based metrics (WT, RESCUED).
	
	Tests main effects of Training and Orientation; optionally Training×Orientation interaction.
	Returns dict structure: {group: {"training_p": p, "orientation_p": p, "interaction_p": p}}
	"""
	
	results: Dict[str, Dict[str, Optional[float]]] = {
		"WT": {"training_p": None, "orientation_p": None, "interaction_p": None},
		"RESCUED": {"training_p": None, "orientation_p": None, "interaction_p": None},
	}
	
	for condition in ["WT", "RESCUED"]:
		pairs = MOUSE_PAIRS.get(condition, [])
		rows: List[Dict[str, object]] = []
		for session_before, session_after in pairs:
			s_before = prepare_session_summaries(session_before)
			s_after = prepare_session_summaries(session_after)
			val_before = s_before.get(metric_key)
			val_after = s_after.get(metric_key)
			orientations = s_before.get("orientations")
			if val_before is None or val_after is None or orientations is None:
				continue
			val_before = np.asarray(val_before, dtype=float)
			val_after = np.asarray(val_after, dtype=float)
			if val_before.ndim != 1 or val_after.ndim != 1:
				continue
			min_len = min(len(orientations), val_before.size, val_after.size)
			if min_len < 2:
				continue
			subject = _mouse_id_from_session_rel(session_before)
			for ori_idx in range(min_len):
				vb = float(val_before[ori_idx])
				va = float(val_after[ori_idx])
				if np.isfinite(vb) and np.isfinite(va):
					rows.append({"subject": subject, "training": "Pre", "ori": str(int(ori_idx)), "value": vb})
					rows.append({"subject": subject, "training": "Post", "ori": str(int(ori_idx)), "value": va})

		df = pd.DataFrame(rows)
		if df.empty:
			continue
		grid = df.pivot_table(index='subject', columns=['training', 'ori'], values='value', aggfunc='mean')
		grid = grid.dropna(axis=0, how='any')
		if grid.shape[0] < 2:
			continue
		oris = sorted({c[1] for c in grid.columns})
		balanced_rows: List[Dict[str, object]] = []
		for subject in grid.index:
			for training in ['Pre', 'Post']:
				for ori in oris:
					balanced_rows.append({
						"subject": subject,
						"training": training,
						"ori": ori,
						"value": float(grid.loc[subject, (training, ori)]),
					})
		df_bal = pd.DataFrame(balanced_rows)
		try:
			if RM_ANOVA_INCLUDE_INTERACTION_TERMS:
				res = AnovaRM(df_bal, depvar='value', subject='subject', within=['training', 'ori']).fit()
				tab = res.anova_table
				p_train = tab.loc['training', 'Pr > F'] if 'training' in tab.index else np.nan
				p_ori = tab.loc['ori', 'Pr > F'] if 'ori' in tab.index else np.nan
				p_int = tab.loc['training:ori', 'Pr > F'] if 'training:ori' in tab.index else np.nan
			else:
				df_train = df_bal.groupby(['subject', 'training'], as_index=False)['value'].mean()
				res_train = AnovaRM(df_train, depvar='value', subject='subject', within=['training']).fit()
				tab_train = res_train.anova_table
				p_train = tab_train.loc['training', 'Pr > F'] if 'training' in tab_train.index else np.nan

				df_ori = df_bal.groupby(['subject', 'ori'], as_index=False)['value'].mean()
				res_ori = AnovaRM(df_ori, depvar='value', subject='subject', within=['ori']).fit()
				tab_ori = res_ori.anova_table
				p_ori = tab_ori.loc['ori', 'Pr > F'] if 'ori' in tab_ori.index else np.nan
				p_int = np.nan
			results[condition] = {
				"training_p": float(p_train) if np.isfinite(p_train) else None,
				"orientation_p": float(p_ori) if np.isfinite(p_ori) else None,
				"interaction_p": float(p_int) if np.isfinite(p_int) else None,
			}
		except Exception as exc:
			print(f"[debug] RM-ANOVA orientation computation for {condition}/{metric_key}: {type(exc).__name__}")
			continue
	
	return results


def _aligned_rank_transform(
	df: pd.DataFrame,
	factor_a: str,
	factor_b: str,
	subject: str,
	value_col: str,
	*,
	include_interaction: bool = True,
) -> Dict[str, pd.Series]:
	"""Perform Aligned Rank Transform (ART) for two-way design.

	Returns dict with aligned+ranked series for 'A', 'B', and optionally 'AxB'.

	Assumes balanced repeated-measures: each subject has observations across levels of A and B.
	"""
	# Centering by grand mean
	grand_mean = df[value_col].mean()

	# Compute marginal means
	mean_a = df.groupby(factor_a, observed=False)[value_col].mean()
	mean_b = df.groupby(factor_b, observed=False)[value_col].mean()
	mean_ab = df.groupby([factor_a, factor_b], observed=False)[value_col].mean()

	if include_interaction:
		# Align for A: remove other effects (B and interaction)
		def align_a(row):
			return (row[value_col]
			        - mean_b[row[factor_b]]
			        - (mean_ab[row[factor_a], row[factor_b]] - mean_a[row[factor_a]] - mean_b[row[factor_b]] - grand_mean)
			        )
		# Align for B: remove A and interaction
		def align_b(row):
			return (row[value_col]
			        - mean_a[row[factor_a]]
			        - (mean_ab[row[factor_a], row[factor_b]] - mean_a[row[factor_a]] - mean_b[row[factor_b]] - grand_mean)
			        )
	else:
		# Reduced (additive) model: align using only the other main effect.
		def align_a(row):
			return row[value_col] - mean_b[row[factor_b]]
		def align_b(row):
			return row[value_col] - mean_a[row[factor_a]]
	aligned_a = df.apply(align_a, axis=1)
	aligned_b = df.apply(align_b, axis=1)

	# Rank within full dataset (ART ranks across all aligned values per effect)
	rank_a = aligned_a.rank(method="average")
	rank_b = aligned_b.rank(method="average")

	out: Dict[str, pd.Series] = {"A": rank_a, "B": rank_b}
	if include_interaction:
		# Align for interaction: remove A and B main effects
		def align_inter(row):
			return (row[value_col]
			        - mean_a[row[factor_a]]
			        - mean_b[row[factor_b]]
			        + grand_mean)
		aligned_inter = df.apply(align_inter, axis=1)
		rank_inter = aligned_inter.rank(method="average")
		out["AxB"] = rank_inter
	return out


def _balance_long_rm_df(
	df: pd.DataFrame,
	*,
	subject: str,
	within: List[str],
	value_col: str,
) -> pd.DataFrame:
	"""Return a balanced long-form dataframe for repeated-measures ANOVA.

	Approach:
	- Pivot to a subject×(within-level-combo) grid
	- Drop any within-level combos with missing data in any subject
	- Drop subjects with any remaining missing
	- Reconstruct a balanced long-form table

	This intentionally favors correctness/stability over maximizing retained data.
	"""
	if df.empty:
		return df
	if not within:
		return df
	grid = df.pivot_table(index=subject, columns=within, values=value_col, aggfunc='mean')
	if grid.empty:
		return pd.DataFrame(columns=[subject, *within, value_col])

	grid2 = grid.dropna(axis=1, how='any')
	grid2 = grid2.dropna(axis=0, how='any')
	if grid2.shape[0] < 2:
		return pd.DataFrame(columns=[subject, *within, value_col])

	cols = grid2.columns
	if isinstance(cols, pd.MultiIndex):
		if cols.nlevels != len(within):
			return pd.DataFrame(columns=[subject, *within, value_col])
		for li, factor in enumerate(within):
			levels = sorted(set(cols.get_level_values(li)))
			if len(levels) < 2:
				return pd.DataFrame(columns=[subject, *within, value_col])
	else:
		if len(within) != 1:
			return pd.DataFrame(columns=[subject, *within, value_col])
		levels = sorted(set(cols))
		if len(levels) < 2:
			return pd.DataFrame(columns=[subject, *within, value_col])

	rows: List[Dict[str, object]] = []
	for subj in grid2.index:
		if isinstance(cols, pd.MultiIndex):
			for combo in cols:
				row: Dict[str, object] = {subject: subj, value_col: float(grid2.loc[subj, combo])}
				for i, factor in enumerate(within):
					row[factor] = combo[i]
				rows.append(row)
		else:
			for level in cols:
				rows.append({subject: subj, within[0]: level, value_col: float(grid2.loc[subj, level])})

	out = pd.DataFrame(rows)
	return out


def _rmanova_3way_pvalues(
	df: pd.DataFrame,
	*,
	subject: str,
	factor_a: str,
	factor_b: str,
	factor_c: str,
	value_col: str,
	include_interactions: bool,
) -> Dict[str, Optional[float]]:
	"""Three-way within-subject RM-ANOVA p-values.

	Returns mapping keyed by effect term names as used by statsmodels AnovaRM:
	- A, B, C
	- A:B, A:C, B:C
	- A:B:C

	If include_interactions=False, returns only main-effect p-values (interactions are None)
	computed by collapsing across the other factors (one-way RM-ANOVA per factor).
	"""
	terms = [
		factor_a,
		factor_b,
		factor_c,
		f"{factor_a}:{factor_b}",
		f"{factor_a}:{factor_c}",
		f"{factor_b}:{factor_c}",
		f"{factor_a}:{factor_b}:{factor_c}",
	]
	out: Dict[str, Optional[float]] = {t: None for t in terms}
	if df.empty:
		return out

	try:
		if include_interactions:
			res = AnovaRM(df, depvar=value_col, subject=subject, within=[factor_a, factor_b, factor_c]).fit()
			tab = res.anova_table
			for t in terms:
				if t in tab.index:
					p = tab.loc[t, 'Pr > F']
					out[t] = float(p) if (p is not None and np.isfinite(p)) else None
			return out

		# Reduced additive: main effects only by collapsing across other factors
		df_a = df.groupby([subject, factor_a], as_index=False)[value_col].mean()
		out[factor_a] = float(
			AnovaRM(df_a, depvar=value_col, subject=subject, within=[factor_a]).fit().anova_table.loc[factor_a, 'Pr > F']
		)
		df_b = df.groupby([subject, factor_b], as_index=False)[value_col].mean()
		out[factor_b] = float(
			AnovaRM(df_b, depvar=value_col, subject=subject, within=[factor_b]).fit().anova_table.loc[factor_b, 'Pr > F']
		)
		df_c = df.groupby([subject, factor_c], as_index=False)[value_col].mean()
		out[factor_c] = float(
			AnovaRM(df_c, depvar=value_col, subject=subject, within=[factor_c]).fit().anova_table.loc[factor_c, 'Pr > F']
		)
		# Interactions intentionally suppressed in reduced model
		return out
	except Exception:
		return out


def _aligned_rank_transform_3way(
	df: pd.DataFrame,
	*,
	factor_a: str,
	factor_b: str,
	factor_c: str,
	value_col: str,
	include_interactions: bool = True,
) -> Dict[str, pd.Series]:
	"""Aligned Rank Transform (ART) for a three-way design.

	Returns ranked aligned responses for:
	- 'A', 'B', 'C'
	- 'AxB', 'AxC', 'BxC'
	- 'AxBxC'

	If include_interactions=False, returns only {'A','B','C'} aligned by an additive model.
	"""
	grand_mean = df[value_col].mean()
	mean_a = df.groupby(factor_a, observed=False)[value_col].mean()
	mean_b = df.groupby(factor_b, observed=False)[value_col].mean()
	mean_c = df.groupby(factor_c, observed=False)[value_col].mean()
	mean_ab = df.groupby([factor_a, factor_b], observed=False)[value_col].mean()
	mean_ac = df.groupby([factor_a, factor_c], observed=False)[value_col].mean()
	mean_bc = df.groupby([factor_b, factor_c], observed=False)[value_col].mean()
	mean_abc = df.groupby([factor_a, factor_b, factor_c], observed=False)[value_col].mean()

	def eff_a(row) -> float:
		return float(mean_a[row[factor_a]] - grand_mean)
	def eff_b(row) -> float:
		return float(mean_b[row[factor_b]] - grand_mean)
	def eff_c(row) -> float:
		return float(mean_c[row[factor_c]] - grand_mean)
	def eff_ab(row) -> float:
		key = (row[factor_a], row[factor_b])
		return float(mean_ab[key] - mean_a[row[factor_a]] - mean_b[row[factor_b]] + grand_mean)
	def eff_ac(row) -> float:
		key = (row[factor_a], row[factor_c])
		return float(mean_ac[key] - mean_a[row[factor_a]] - mean_c[row[factor_c]] + grand_mean)
	def eff_bc(row) -> float:
		key = (row[factor_b], row[factor_c])
		return float(mean_bc[key] - mean_b[row[factor_b]] - mean_c[row[factor_c]] + grand_mean)
	def eff_abc(row) -> float:
		key = (row[factor_a], row[factor_b], row[factor_c])
		return float(
			mean_abc[key]
			- mean_ab[(row[factor_a], row[factor_b])]
			- mean_ac[(row[factor_a], row[factor_c])]
			- mean_bc[(row[factor_b], row[factor_c])]
			+ mean_a[row[factor_a]]
			+ mean_b[row[factor_b]]
			+ mean_c[row[factor_c]]
			- grand_mean
		)

	if include_interactions:
		def align_a(row) -> float:
			return float(row[value_col] - (eff_b(row) + eff_c(row) + eff_ab(row) + eff_ac(row) + eff_bc(row) + eff_abc(row)) + grand_mean)
		def align_b(row) -> float:
			return float(row[value_col] - (eff_a(row) + eff_c(row) + eff_ab(row) + eff_ac(row) + eff_bc(row) + eff_abc(row)) + grand_mean)
		def align_c(row) -> float:
			return float(row[value_col] - (eff_a(row) + eff_b(row) + eff_ab(row) + eff_ac(row) + eff_bc(row) + eff_abc(row)) + grand_mean)
		def align_ab(row) -> float:
			return float(row[value_col] - (eff_a(row) + eff_b(row) + eff_c(row) + eff_ac(row) + eff_bc(row) + eff_abc(row)) + grand_mean)
		def align_ac(row) -> float:
			return float(row[value_col] - (eff_a(row) + eff_b(row) + eff_c(row) + eff_ab(row) + eff_bc(row) + eff_abc(row)) + grand_mean)
		def align_bc(row) -> float:
			return float(row[value_col] - (eff_a(row) + eff_b(row) + eff_c(row) + eff_ab(row) + eff_ac(row) + eff_abc(row)) + grand_mean)
		def align_abc(row) -> float:
			return float(row[value_col] - (eff_a(row) + eff_b(row) + eff_c(row) + eff_ab(row) + eff_ac(row) + eff_bc(row)) + grand_mean)
		aligned_a = df.apply(align_a, axis=1)
		aligned_b = df.apply(align_b, axis=1)
		aligned_c = df.apply(align_c, axis=1)
		aligned_ab = df.apply(align_ab, axis=1)
		aligned_ac = df.apply(align_ac, axis=1)
		aligned_bc = df.apply(align_bc, axis=1)
		aligned_abc = df.apply(align_abc, axis=1)
		return {
			"A": aligned_a.rank(method='average'),
			"B": aligned_b.rank(method='average'),
			"C": aligned_c.rank(method='average'),
			"AxB": aligned_ab.rank(method='average'),
			"AxC": aligned_ac.rank(method='average'),
			"BxC": aligned_bc.rank(method='average'),
			"AxBxC": aligned_abc.rank(method='average'),
		}

	# Reduced (additive) model: align using only the other main effects.
	def align_a_add(row) -> float:
		return float(row[value_col] - eff_b(row) - eff_c(row))
	def align_b_add(row) -> float:
		return float(row[value_col] - eff_a(row) - eff_c(row))
	def align_c_add(row) -> float:
		return float(row[value_col] - eff_a(row) - eff_b(row))
	aligned_a = df.apply(align_a_add, axis=1)
	aligned_b = df.apply(align_b_add, axis=1)
	aligned_c = df.apply(align_c_add, axis=1)
	return {
		"A": aligned_a.rank(method='average'),
		"B": aligned_b.rank(method='average'),
		"C": aligned_c.rank(method='average'),
	}


def _art_3way_pvalues(
	df: pd.DataFrame,
	*,
	subject: str,
	factor_a: str,
	factor_b: str,
	factor_c: str,
	value_col: str,
	include_interactions: bool,
) -> Dict[str, Optional[float]]:
	"""Three-way ART p-values using AnovaRM on aligned ranks."""
	terms = [
		factor_a,
		factor_b,
		factor_c,
		f"{factor_a}:{factor_b}",
		f"{factor_a}:{factor_c}",
		f"{factor_b}:{factor_c}",
		f"{factor_a}:{factor_b}:{factor_c}",
	]
	out: Dict[str, Optional[float]] = {t: None for t in terms}
	if df.empty:
		return out
	try:
		art = _aligned_rank_transform_3way(
			df,
			factor_a=factor_a,
			factor_b=factor_b,
			factor_c=factor_c,
			value_col=value_col,
			include_interactions=include_interactions,
		)

		df2 = df.copy()
		df2['rank_A'] = art['A']
		df2['rank_B'] = art['B']
		df2['rank_C'] = art['C']

		if include_interactions:
			df2['rank_AB'] = art.get('AxB')
			df2['rank_AC'] = art.get('AxC')
			df2['rank_BC'] = art.get('BxC')
			df2['rank_ABC'] = art.get('AxBxC')

			within = [factor_a, factor_b, factor_c]
			tab_a = AnovaRM(df2, depvar='rank_A', subject=subject, within=within).fit().anova_table
			tab_b = AnovaRM(df2, depvar='rank_B', subject=subject, within=within).fit().anova_table
			tab_c = AnovaRM(df2, depvar='rank_C', subject=subject, within=within).fit().anova_table
			tab_ab = AnovaRM(df2, depvar='rank_AB', subject=subject, within=within).fit().anova_table
			tab_ac = AnovaRM(df2, depvar='rank_AC', subject=subject, within=within).fit().anova_table
			tab_bc = AnovaRM(df2, depvar='rank_BC', subject=subject, within=within).fit().anova_table
			tab_abc = AnovaRM(df2, depvar='rank_ABC', subject=subject, within=within).fit().anova_table

			def safe_get(tab: pd.DataFrame, term: str) -> Optional[float]:
				if term in tab.index:
					p = tab.loc[term, 'Pr > F']
					return float(p) if (p is not None and np.isfinite(p)) else None
				return None

			out[factor_a] = safe_get(tab_a, factor_a)
			out[factor_b] = safe_get(tab_b, factor_b)
			out[factor_c] = safe_get(tab_c, factor_c)
			out[f"{factor_a}:{factor_b}"] = safe_get(tab_ab, f"{factor_a}:{factor_b}")
			out[f"{factor_a}:{factor_c}"] = safe_get(tab_ac, f"{factor_a}:{factor_c}")
			out[f"{factor_b}:{factor_c}"] = safe_get(tab_bc, f"{factor_b}:{factor_c}")
			out[f"{factor_a}:{factor_b}:{factor_c}"] = safe_get(tab_abc, f"{factor_a}:{factor_b}:{factor_c}")
			return out

		# Reduced additive: one-way RM-ANOVA on ranks after collapsing across other factors
		df_a = df2.groupby([subject, factor_a], as_index=False)['rank_A'].mean()
		out[factor_a] = float(AnovaRM(df_a, depvar='rank_A', subject=subject, within=[factor_a]).fit().anova_table.loc[factor_a, 'Pr > F'])
		df_b = df2.groupby([subject, factor_b], as_index=False)['rank_B'].mean()
		out[factor_b] = float(AnovaRM(df_b, depvar='rank_B', subject=subject, within=[factor_b]).fit().anova_table.loc[factor_b, 'Pr > F'])
		df_c = df2.groupby([subject, factor_c], as_index=False)['rank_C'].mean()
		out[factor_c] = float(AnovaRM(df_c, depvar='rank_C', subject=subject, within=[factor_c]).fit().anova_table.loc[factor_c, 'Pr > F'])
		return out
	except Exception:
		return out


def _compute_rm_art_3way_from_rows(
	rows: List[Dict[str, object]],
	*,
	subject: str,
	factor_a: str,
	factor_b: str,
	factor_c: str,
	value_col: str = 'value',
	center_by_subject: bool = True,
) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]]]:
	"""Convenience: balance rows then compute 3-way RM and ART p-values."""
	df = pd.DataFrame(rows)
	if df.empty:
		terms = [
			factor_a,
			factor_b,
			factor_c,
			f"{factor_a}:{factor_b}",
			f"{factor_a}:{factor_c}",
			f"{factor_b}:{factor_c}",
			f"{factor_a}:{factor_b}:{factor_c}",
		]
		empty = {t: None for t in terms}
		return empty, empty.copy()

	df_bal = _balance_long_rm_df(df, subject=subject, within=[factor_a, factor_b, factor_c], value_col=value_col)
	if df_bal.empty:
		terms = [
			factor_a,
			factor_b,
			factor_c,
			f"{factor_a}:{factor_b}",
			f"{factor_a}:{factor_c}",
			f"{factor_b}:{factor_c}",
			f"{factor_a}:{factor_b}:{factor_c}",
		]
		empty = {t: None for t in terms}
		return empty, empty.copy()

	if center_by_subject:
		df_bal['value_centered'] = df_bal[value_col] - df_bal.groupby(subject)[value_col].transform('mean')
		val_col = 'value_centered'
	else:
		val_col = value_col

	for col in [factor_a, factor_b, factor_c]:
		df_bal[col] = df_bal[col].astype('category')

	rm = _rmanova_3way_pvalues(
		df_bal,
		subject=subject,
		factor_a=factor_a,
		factor_b=factor_b,
		factor_c=factor_c,
		value_col=val_col,
		include_interactions=RM_ANOVA_INCLUDE_INTERACTION_TERMS,
	)
	art = _art_3way_pvalues(
		df_bal,
		subject=subject,
		factor_a=factor_a,
		factor_b=factor_b,
		factor_c=factor_c,
		value_col=val_col,
		include_interactions=ART_INCLUDE_INTERACTION_TERMS,
	)
	return rm, art


def _safe_fit_mixedlm(*, df: pd.DataFrame, formula: str, group_col: str) -> Optional[object]:
	"""Fit a random-intercept MixedLM using ML (reml=False); return result or None.

	Important: callers that do LRTs must NOT mix MixedLM and OLS results.
	This function therefore does *not* auto-fallback to OLS.
	"""
	if df.empty or df[group_col].nunique() < 2:
		return None
	try:
		md = mixedlm(formula, df, groups=df[group_col], re_formula="1")
		for method in ("lbfgs", "bfgs", "cg", "powell", "nm"):
			try:
				return md.fit(reml=False, method=method, disp=False, maxiter=2000)
			except Exception:
				continue
	except Exception:
		return None
	return None


def _lme_lr_p(full_res: object, reduced_res: object) -> Optional[float]:
	"""Likelihood-ratio p-value between nested MixedLM models."""
	try:
		ll_full = float(full_res.llf)
		ll_red = float(reduced_res.llf)
		k_full = int(getattr(full_res.model, "exog", np.empty((0, 0))).shape[1])
		k_red = int(getattr(reduced_res.model, "exog", np.empty((0, 0))).shape[1])
		df_diff = k_full - k_red
		if df_diff <= 0:
			return None
		lr = 2.0 * (ll_full - ll_red)
		if not np.isfinite(lr):
			return None
		# Numerical noise can very slightly violate ll_full>=ll_red.
		if lr < 0 and lr > -1e-6:
			lr = 0.0
		if lr < 0:
			return None
		p = float(chi2.sf(lr, df_diff))
		return p if np.isfinite(p) else None
	except Exception:
		return None


def _lme_2way_lr_pvalues_from_rows(
	rows: List[Dict[str, object]],
	*,
	subject: str,
	factor_a: str,
	factor_b: str,
	value_col: str = "value",
	include_interactions: bool,
) -> Dict[str, Optional[float]]:
	"""LME omnibus p-values for a 2-way within-subject design via LRT."""
	out: Dict[str, Optional[float]] = {"a": None, "b": None, "a:b": None}
	df = pd.DataFrame(rows)
	if df.empty:
		return out
	if value_col not in df.columns:
		return out
	# Drop missing
	df = df[[subject, factor_a, factor_b, value_col]].copy()
	df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
	df = df.dropna(axis=0, how='any')
	if df.empty:
		return out

	# Balance to a complete repeated-measures grid. This substantially improves
	# MixedLM stability for categorical interaction models.
	df = _balance_long_rm_df(df, subject=subject, within=[factor_a, factor_b], value_col=value_col)
	if df.empty:
		return out
	for col in [factor_a, factor_b, subject]:
		df[col] = df[col].astype('category')

	if include_interactions:
		formula_full = f"{value_col} ~ C({factor_a}) * C({factor_b})"
		formula_add = f"{value_col} ~ C({factor_a}) + C({factor_b})"
	else:
		formula_full = f"{value_col} ~ C({factor_a}) + C({factor_b})"
		formula_add = formula_full

	# First try: consistent MixedLM family (LRT)
	full_res = _safe_fit_mixedlm(df=df, formula=formula_full, group_col=subject)
	if full_res is not None:
		res_no_a = _safe_fit_mixedlm(df=df, formula=f"{value_col} ~ C({factor_b})" + (f"" if not include_interactions else ""), group_col=subject)
		res_no_b = _safe_fit_mixedlm(df=df, formula=f"{value_col} ~ C({factor_a})" + (f"" if not include_interactions else ""), group_col=subject)
		add_res = _safe_fit_mixedlm(df=df, formula=formula_add, group_col=subject) if include_interactions else None
		need_ok = (res_no_a is not None) and (res_no_b is not None) and ((add_res is not None) if include_interactions else True)
		if need_ok:
			out["a"] = _lme_lr_p(full_res, res_no_a)
			out["b"] = _lme_lr_p(full_res, res_no_b)
			out["a:b"] = _lme_lr_p(full_res, add_res) if include_interactions and add_res is not None else None
			return out

	# No fallback beyond MixedLM: return what we have (likely NAs downstream).
	return out


def _lme_3way_lr_pvalues_from_rows(
	rows: List[Dict[str, object]],
	*,
	subject: str,
	factor_a: str,
	factor_b: str,
	factor_c: str,
	value_col: str = "value",
	include_interactions: bool,
) -> Dict[str, Optional[float]]:
	"""LME omnibus p-values for a 3-way within-subject design via LRT.

	For interactions, we use a hierarchical strategy:
	- 3-way: compare full (all) vs model with only 2-way interactions
	- 2-way: compare the 2-way-interactions model vs the same model with that interaction removed
	"""
	terms = [
		factor_a,
		factor_b,
		factor_c,
		f"{factor_a}:{factor_b}",
		f"{factor_a}:{factor_c}",
		f"{factor_b}:{factor_c}",
		f"{factor_a}:{factor_b}:{factor_c}",
	]
	out: Dict[str, Optional[float]] = {t: None for t in terms}
	df = pd.DataFrame(rows)
	if df.empty:
		return out
	if value_col not in df.columns:
		return out
	keep_cols = [subject, factor_a, factor_b, factor_c, value_col]
	df = df[keep_cols].copy()
	df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
	df = df.dropna(axis=0, how='any')
	if df.empty:
		return out

	# Balance to a complete repeated-measures grid to avoid rank-deficient designs.
	df = _balance_long_rm_df(df, subject=subject, within=[factor_a, factor_b, factor_c], value_col=value_col)
	if df.empty:
		return out
	for col in [factor_a, factor_b, factor_c, subject]:
		df[col] = df[col].astype('category')

	if include_interactions:
		formula_full = f"{value_col} ~ C({factor_a}) * C({factor_b}) * C({factor_c})"
		formula_2way = (
			f"{value_col} ~ C({factor_a}) + C({factor_b}) + C({factor_c})"
			f" + C({factor_a}):C({factor_b}) + C({factor_a}):C({factor_c}) + C({factor_b}):C({factor_c})"
		)
	else:
		formula_full = f"{value_col} ~ C({factor_a}) + C({factor_b}) + C({factor_c})"
		formula_2way = formula_full

	# First try: consistent MixedLM family (LRT)
	full_res = _safe_fit_mixedlm(df=df, formula=formula_full, group_col=subject)
	if full_res is not None:
		res_no_a = _safe_fit_mixedlm(df=df, formula=f"{value_col} ~ C({factor_b}) * C({factor_c})" if include_interactions else f"{value_col} ~ C({factor_b}) + C({factor_c})", group_col=subject)
		res_no_b = _safe_fit_mixedlm(df=df, formula=f"{value_col} ~ C({factor_a}) * C({factor_c})" if include_interactions else f"{value_col} ~ C({factor_a}) + C({factor_c})", group_col=subject)
		res_no_c = _safe_fit_mixedlm(df=df, formula=f"{value_col} ~ C({factor_a}) * C({factor_b})" if include_interactions else f"{value_col} ~ C({factor_a}) + C({factor_b})", group_col=subject)
		if res_no_a is not None:
			out[factor_a] = _lme_lr_p(full_res, res_no_a)
		if res_no_b is not None:
			out[factor_b] = _lme_lr_p(full_res, res_no_b)
		if res_no_c is not None:
			out[factor_c] = _lme_lr_p(full_res, res_no_c)

		if not include_interactions:
			return out

		res_2way = _safe_fit_mixedlm(df=df, formula=formula_2way, group_col=subject)
		res_drop_ab = _safe_fit_mixedlm(
			df=df,
			formula=(
				f"{value_col} ~ C({factor_a}) + C({factor_b}) + C({factor_c})"
				f" + C({factor_a}):C({factor_c}) + C({factor_b}):C({factor_c})"
			),
			group_col=subject,
		)
		res_drop_ac = _safe_fit_mixedlm(
			df=df,
			formula=(
				f"{value_col} ~ C({factor_a}) + C({factor_b}) + C({factor_c})"
				f" + C({factor_a}):C({factor_b}) + C({factor_b}):C({factor_c})"
			),
			group_col=subject,
		)
		res_drop_bc = _safe_fit_mixedlm(
			df=df,
			formula=(
				f"{value_col} ~ C({factor_a}) + C({factor_b}) + C({factor_c})"
				f" + C({factor_a}):C({factor_b}) + C({factor_a}):C({factor_c})"
			),
			group_col=subject,
		)
		need_ok = (res_2way is not None) and (res_drop_ab is not None) and (res_drop_ac is not None) and (res_drop_bc is not None)
		if need_ok:
			out[f"{factor_a}:{factor_b}:{factor_c}"] = _lme_lr_p(full_res, res_2way)
			out[f"{factor_a}:{factor_b}"] = _lme_lr_p(res_2way, res_drop_ab)
			out[f"{factor_a}:{factor_c}"] = _lme_lr_p(res_2way, res_drop_ac)
			out[f"{factor_b}:{factor_c}"] = _lme_lr_p(res_2way, res_drop_bc)
			return out

	# No fallback beyond MixedLM: return what we have (likely NAs downstream).
	return out


def _try_import_lme4():
	"""Best-effort import of rpy2 + lme4; returns a tuple or None.

	We keep this optional because many environments won't have R+rpy2 installed.
	"""
	try:
		# rpy2 on Windows should use ABI mode; letting it try "ANY" can emit a noisy
		# ImportError message before it falls back to ABI.
		if os.name == 'nt':
			os.environ.setdefault('RPY2_CFFI_MODE', 'ABI')
		import rpy2.robjects as ro  # type: ignore[import-not-found]
		from rpy2.robjects import pandas2ri  # type: ignore[import-not-found]
		from rpy2.robjects.conversion import localconverter  # type: ignore[import-not-found]
		from rpy2.robjects.packages import importr  # type: ignore[import-not-found]
		s_stats = importr('stats')
		lme4 = importr('lme4')
		base = importr('base')
		# Silence noisy singular-fit warnings that lme4 often emits for random-intercept
		# models with near-zero variance (common in small / perfectly balanced designs).
		try:
			ro.r('options(lme4.warnSingular = FALSE)')
		except Exception:
			pass
		return ro, pandas2ri, localconverter, s_stats, lme4, base
	except Exception:
		return None


def _try_import_artool():
	"""Best-effort import of rpy2 + ARTool; returns a tuple or None."""
	if not ART_R_ENABLE:
		return None
	try:
		if os.name == 'nt':
			os.environ.setdefault('RPY2_CFFI_MODE', 'ABI')
		import rpy2.robjects as ro  # type: ignore[import-not-found]
		from rpy2.robjects import pandas2ri  # type: ignore[import-not-found]
		from rpy2.robjects.conversion import localconverter  # type: ignore[import-not-found]
		from rpy2.robjects.packages import importr  # type: ignore[import-not-found]
		base = importr('base')
		s_stats = importr('stats')
		# ARTool uses lme4 for mixed models; ensure it's available.
		_ = importr('lme4')
		# Also silence lme4 singular-fit spam in ARTool mixed-model paths.
		try:
			ro.r('options(lme4.warnSingular = FALSE)')
		except Exception:
			pass
		artool = importr('ARTool')
		return ro, pandas2ri, localconverter, base, s_stats, artool
	except Exception:
		return None


def _artool_anova_pvalues(
	*,
	df: pd.DataFrame,
	formula_rhs: str,
) -> Dict[str, Optional[float]]:
	"""Run ARTool::art + anova and return {term_name: pvalue}.

	formula_rhs: everything after '~', e.g.
	- "training*sf + Error(subject/(training*sf))" (aov-style repeated measures)
	- "training*sf + (1|subject)" (mixed model)
	"""
	out: Dict[str, Optional[float]] = {}
	imp = _try_import_artool()
	if imp is None:
		return out
	ro, pandas2ri, localconverter, base, s_stats, artool = imp
	if df.empty:
		return out

	try:
		df_r = df.copy()
		if 'value' not in df_r.columns:
			return out
		# Coerce factors explicitly.
		for c in df_r.columns:
			if c == 'value':
				continue
			if df_r[c].dtype == object or str(df_r[c].dtype).startswith('category'):
				df_r[c] = df_r[c].astype(str)
		with localconverter(ro.default_converter + pandas2ri.converter):
			r_df = ro.conversion.py2rpy(df_r)

		# Ensure non-numeric columns are treated as factors.
		make_factors = ro.r(
			'function(d){ for(n in names(d)) if(n != "value") d[[n]] <- as.factor(d[[n]]); d }'
		)
		r_df = make_factors(r_df)
		fml = ro.Formula(f"value ~ {formula_rhs}")

		# Run art()+anova() and extract p-values entirely in R.
		# This avoids rpy2 edge-cases with list-of-tables returned by Error(...)
		# and is more robust across ARTool versions.
		run_and_extract = ro.r(
			"""
			function(fml, d){
			  get_tbl <- function(tbl){
			    if(is.null(tbl)) return(NULL)
			    if(!is.data.frame(tbl) && !is.matrix(tbl)) return(NULL)
			    cn <- colnames(tbl)
			    term <- NULL
			    if(!is.null(cn) && any(cn == 'Term')) {
			      term <- as.character(tbl[, 'Term'])
			    } else {
			      term <- rownames(tbl)
			      if(is.null(term)) term <- as.character(seq_len(nrow(tbl)))
			    }
			    pcol <- NULL
			    if(!is.null(cn) && any(cn == 'Pr(>F)')) pcol <- 'Pr(>F)'
			    if(is.null(pcol) && !is.null(cn) && any(cn == 'Pr(>Chisq)')) pcol <- 'Pr(>Chisq)'
			    if(is.null(pcol)) return(NULL)
			    pv <- suppressWarnings(as.numeric(tbl[, pcol]))
			    data.frame(term=term, p=pv, stringsAsFactors=FALSE)
			  }
			  extract <- function(an){
			    acc <- list()
			    if(is.data.frame(an) || is.matrix(an)) {
			      dt <- get_tbl(an)
			      if(!is.null(dt)) acc[[length(acc)+1]] <- dt
			    } else if(is.list(an)) {
			      for(i in seq_along(an)) {
			        dt <- get_tbl(an[[i]])
			        if(!is.null(dt)) acc[[length(acc)+1]] <- dt
			      }
			    }
			    if(length(acc) == 0) return(data.frame(term=character(0), p=numeric(0), stringsAsFactors=FALSE))
			    d2 <- do.call(rbind, acc)
			    d2 <- d2[!is.na(d2$p), , drop=FALSE]
			    if(nrow(d2) == 0) return(data.frame(term=character(0), p=numeric(0), stringsAsFactors=FALSE))
			    d2 <- d2[!duplicated(d2$term), , drop=FALSE]
			    d2
			  }
			  tryCatch({
			    m <- suppressWarnings(suppressMessages(ARTool::art(fml, data=d)))
			    an <- suppressWarnings(suppressMessages(stats::anova(m)))
			    extract(an)
			  }, error=function(e){
			    data.frame(term=character(0), p=numeric(0), stringsAsFactors=FALSE)
			  })
			}
			"""
		)
		p_df = run_and_extract(fml, r_df)
		with localconverter(ro.default_converter + pandas2ri.converter):
			p_pd = ro.conversion.rpy2py(p_df)
		if p_pd is None or len(p_pd) == 0:
			return out
		for _, row in p_pd.iterrows():
			term = str(row.get('term'))
			try:
				pv = float(row.get('p'))
			except Exception:
				pv = np.nan
			out[term] = float(pv) if np.isfinite(pv) else None
		return out
	except Exception:
		return out


def _art_r_2way_pvalues_from_rows(
	rows: List[Dict[str, object]],
	*,
	subject: str,
	factor_a: str,
	factor_b: str,
	value_col: str = 'value',
	include_interactions: bool,
) -> Dict[str, Optional[float]]:
	"""ART-Rlmer omnibus p-values for a 2-way within-subject design.

	Returns keys: a, b, a:b
	"""
	out: Dict[str, Optional[float]] = {'a': None, 'b': None, 'a:b': None}
	if not ART_R_ENABLE:
		return out
	df = pd.DataFrame(rows)
	if df.empty or value_col not in df.columns:
		return out
	df = df[[subject, factor_a, factor_b, value_col]].copy()
	df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
	df = df.dropna(axis=0, how='any')
	if df.empty:
		return out
	# Balance repeated-measures grid
	df = _balance_long_rm_df(df, subject=subject, within=[factor_a, factor_b], value_col=value_col)
	if df.empty:
		return out
	df = df.rename(columns={value_col: 'value'})
	# Back-compat: this is the lmer-style variant (mixed model).
	if include_interactions:
		rhs = f"{factor_a}*{factor_b} + (1|{subject})"
	else:
		rhs = f"{factor_a}+{factor_b} + (1|{subject})"
	pvals = _artool_anova_pvalues(df=df, formula_rhs=rhs)
	out['a'] = pvals.get(factor_a)
	out['b'] = pvals.get(factor_b)
	out['a:b'] = pvals.get(f"{factor_a}:{factor_b}") if include_interactions else None
	return out


def _art_raov_2way_pvalues_from_rows(
	rows: List[Dict[str, object]],
	*,
	subject: str,
	factor_a: str,
	factor_b: str,
	value_col: str = 'value',
	include_interactions: bool,
) -> Dict[str, Optional[float]]:
	"""ART-Raov omnibus p-values for a 2-way within-subject design.

	Uses ARTool::art with an Error(subject/...) term to request ANOVA-style repeated-measures tests.
	Returns keys: a, b, a:b
	"""
	out: Dict[str, Optional[float]] = {'a': None, 'b': None, 'a:b': None}
	if not ART_R_ENABLE:
		return out
	df = pd.DataFrame(rows)
	if df.empty or value_col not in df.columns:
		return out
	df = df[[subject, factor_a, factor_b, value_col]].copy()
	df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
	df = df.dropna(axis=0, how='any')
	if df.empty:
		return out
	df = _balance_long_rm_df(df, subject=subject, within=[factor_a, factor_b], value_col=value_col)
	if df.empty:
		return out
	df = df.rename(columns={value_col: 'value'})
	fixed = f"{factor_a}*{factor_b}" if include_interactions else f"{factor_a}+{factor_b}"
	# Keep the full within-subject error structure regardless of whether we test interactions.
	err = f"Error({subject}/({factor_a}*{factor_b}))"
	pvals = _artool_anova_pvalues(df=df, formula_rhs=f"{fixed} + {err}")
	out['a'] = pvals.get(factor_a)
	out['b'] = pvals.get(factor_b)
	out['a:b'] = pvals.get(f"{factor_a}:{factor_b}") if include_interactions else None
	return out


def _art_r_3way_pvalues_from_rows(
	rows: List[Dict[str, object]],
	*,
	subject: str,
	factor_a: str,
	factor_b: str,
	factor_c: str,
	value_col: str = 'value',
	include_interactions: bool,
) -> Dict[str, Optional[float]]:
	"""ART3-Rlmer omnibus p-values for a 3-way within-subject design.

	Back-compat: this is the lmer-style variant (mixed model).
	"""
	terms = [
		factor_a,
		factor_b,
		factor_c,
		f"{factor_a}:{factor_b}",
		f"{factor_a}:{factor_c}",
		f"{factor_b}:{factor_c}",
		f"{factor_a}:{factor_b}:{factor_c}",
	]
	out: Dict[str, Optional[float]] = {t: None for t in terms}
	if not ART_R_ENABLE:
		return out
	df = pd.DataFrame(rows)
	if df.empty or value_col not in df.columns:
		return out
	df = df[[subject, factor_a, factor_b, factor_c, value_col]].copy()
	df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
	df = df.dropna(axis=0, how='any')
	if df.empty:
		return out
	# Balance repeated-measures grid
	df = _balance_long_rm_df(df, subject=subject, within=[factor_a, factor_b, factor_c], value_col=value_col)
	if df.empty:
		return out
	df = df.rename(columns={value_col: 'value'})
	if include_interactions:
		rhs = f"{factor_a}*{factor_b}*{factor_c} + (1|{subject})"
	else:
		rhs = f"{factor_a}+{factor_b}+{factor_c} + (1|{subject})"
	pvals = _artool_anova_pvalues(df=df, formula_rhs=rhs)
	for t in terms:
		out[t] = pvals.get(t)
	if not include_interactions:
		out[f"{factor_a}:{factor_b}"] = None
		out[f"{factor_a}:{factor_c}"] = None
		out[f"{factor_b}:{factor_c}"] = None
		out[f"{factor_a}:{factor_b}:{factor_c}"] = None
	return out


def _art_raov_3way_pvalues_from_rows(
	rows: List[Dict[str, object]],
	*,
	subject: str,
	factor_a: str,
	factor_b: str,
	factor_c: str,
	value_col: str = 'value',
	include_interactions: bool,
) -> Dict[str, Optional[float]]:
	"""ART3-Raov omnibus p-values for a 3-way within-subject design.

	Uses ARTool::art with an Error(subject/...) term to request ANOVA-style repeated-measures tests.
	"""
	terms = [
		factor_a,
		factor_b,
		factor_c,
		f"{factor_a}:{factor_b}",
		f"{factor_a}:{factor_c}",
		f"{factor_b}:{factor_c}",
		f"{factor_a}:{factor_b}:{factor_c}",
	]
	out: Dict[str, Optional[float]] = {t: None for t in terms}
	if not ART_R_ENABLE:
		return out
	df = pd.DataFrame(rows)
	if df.empty or value_col not in df.columns:
		return out
	df = df[[subject, factor_a, factor_b, factor_c, value_col]].copy()
	df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
	df = df.dropna(axis=0, how='any')
	if df.empty:
		return out
	df = _balance_long_rm_df(df, subject=subject, within=[factor_a, factor_b, factor_c], value_col=value_col)
	if df.empty:
		return out
	df = df.rename(columns={value_col: 'value'})
	fixed = f"{factor_a}*{factor_b}*{factor_c}" if include_interactions else f"{factor_a}+{factor_b}+{factor_c}"
	err = f"Error({subject}/({factor_a}*{factor_b}*{factor_c}))"
	pvals = _artool_anova_pvalues(df=df, formula_rhs=f"{fixed} + {err}")
	for t in terms:
		out[t] = pvals.get(t)
	if not include_interactions:
		out[f"{factor_a}:{factor_b}"] = None
		out[f"{factor_a}:{factor_c}"] = None
		out[f"{factor_b}:{factor_c}"] = None
		out[f"{factor_a}:{factor_b}:{factor_c}"] = None
	return out


def compute_art_r_pvalues_sf(metric_key: str) -> Dict[str, Dict[str, Optional[float]]]:
	"""Compute ART-Rlmer omnibus p-values for SF metrics (Training×SF)."""
	results: Dict[str, Dict[str, Optional[float]]] = {
		"WT": {"training_p": None, "sf_p": None, "interaction_p": None},
		"RESCUED": {"training_p": None, "sf_p": None, "interaction_p": None},
	}
	if not ART_R_ENABLE:
		return results
	for condition in ["WT", "RESCUED"]:
		pairs = MOUSE_PAIRS.get(condition, [])
		rows: List[Dict[str, object]] = []
		for session_before, session_after in pairs:
			s_before = prepare_session_summaries(session_before)
			s_after = prepare_session_summaries(session_after)
			v_before = s_before.get(metric_key)
			v_after = s_after.get(metric_key)
			if v_before is None or v_after is None:
				continue
			v_before = np.asarray(v_before, dtype=float)
			v_after = np.asarray(v_after, dtype=float)
			if v_before.ndim == 0:
				v_before = np.full(len(TARGET_SF_VALUES), float(v_before))
				v_after = np.full(len(TARGET_SF_VALUES), float(v_after))
			if v_before.ndim != 1 or v_after.ndim != 1:
				continue
			min_len = min(len(TARGET_SF_VALUES), v_before.size, v_after.size)
			if min_len <= 2:
				continue
			subj = _mouse_id_from_session_rel(session_before)
			for sf_idx in range(1, min_len):
				sf = str(float(TARGET_SF_VALUES[sf_idx]))
				vb = float(v_before[sf_idx])
				va = float(v_after[sf_idx])
				if np.isfinite(vb) and np.isfinite(va):
					rows.append({"subject": subj, "training": "Pre", "sf": sf, "value": vb})
					rows.append({"subject": subj, "training": "Post", "sf": sf, "value": va})
		p = _art_r_2way_pvalues_from_rows(
			rows,
			subject='subject',
			factor_a='training',
			factor_b='sf',
			value_col='value',
			include_interactions=ART_INCLUDE_INTERACTION_TERMS,
		)
		results[condition] = {
			"training_p": p.get('a'),
			"sf_p": p.get('b'),
			"interaction_p": p.get('a:b'),
		}
	return results


def compute_art_r_pvalues_orientation(metric_key: str) -> Dict[str, Dict[str, Optional[float]]]:
	"""Compute ART-Rlmer omnibus p-values for orientation metrics (Training×Ori)."""
	results: Dict[str, Dict[str, Optional[float]]] = {
		"WT": {"training_p": None, "orientation_p": None, "interaction_p": None},
		"RESCUED": {"training_p": None, "orientation_p": None, "interaction_p": None},
	}
	if not ART_R_ENABLE:
		return results
	for condition in ["WT", "RESCUED"]:
		pairs = MOUSE_PAIRS.get(condition, [])
		rows: List[Dict[str, object]] = []
		for session_before, session_after in pairs:
			s_before = prepare_session_summaries(session_before)
			s_after = prepare_session_summaries(session_after)
			v_before = s_before.get(metric_key)
			v_after = s_after.get(metric_key)
			if v_before is None or v_after is None:
				continue
			v_before = np.asarray(v_before, dtype=float)
			v_after = np.asarray(v_after, dtype=float)
			if v_before.ndim != 1 or v_after.ndim != 1:
				continue
			min_len = min(v_before.size, v_after.size)
			if min_len < 2:
				continue
			subj = _mouse_id_from_session_rel(session_before)
			for oi in range(min_len):
				vb = float(v_before[oi])
				va = float(v_after[oi])
				if np.isfinite(vb) and np.isfinite(va):
					rows.append({"subject": subj, "training": "Pre", "ori": str(int(oi)), "value": vb})
					rows.append({"subject": subj, "training": "Post", "ori": str(int(oi)), "value": va})
		p = _art_r_2way_pvalues_from_rows(
			rows,
			subject='subject',
			factor_a='training',
			factor_b='ori',
			value_col='value',
			include_interactions=ART_INCLUDE_INTERACTION_TERMS,
		)
		results[condition] = {
			"training_p": p.get('a'),
			"orientation_p": p.get('b'),
			"interaction_p": p.get('a:b'),
		}
	return results


def compute_art_raov_pvalues_sf(metric_key: str) -> Dict[str, Dict[str, Optional[float]]]:
	"""Compute ART-Raov omnibus p-values for SF metrics (Training×SF)."""
	results: Dict[str, Dict[str, Optional[float]]] = {
		"WT": {"training_p": None, "sf_p": None, "interaction_p": None},
		"RESCUED": {"training_p": None, "sf_p": None, "interaction_p": None},
	}
	if not ART_R_ENABLE:
		return results
	for condition in ["WT", "RESCUED"]:
		pairs = MOUSE_PAIRS.get(condition, [])
		rows: List[Dict[str, object]] = []
		for session_before, session_after in pairs:
			s_before = prepare_session_summaries(session_before)
			s_after = prepare_session_summaries(session_after)
			v_before = s_before.get(metric_key)
			v_after = s_after.get(metric_key)
			if v_before is None or v_after is None:
				continue
			v_before = np.asarray(v_before, dtype=float)
			v_after = np.asarray(v_after, dtype=float)
			if v_before.ndim == 0:
				v_before = np.full(len(TARGET_SF_VALUES), float(v_before))
				v_after = np.full(len(TARGET_SF_VALUES), float(v_after))
			if v_before.ndim != 1 or v_after.ndim != 1:
				continue
			min_len = min(len(TARGET_SF_VALUES), v_before.size, v_after.size)
			if min_len <= 2:
				continue
			subj = _mouse_id_from_session_rel(session_before)
			for sf_idx in range(1, min_len):
				sf = str(float(TARGET_SF_VALUES[sf_idx]))
				vb = float(v_before[sf_idx])
				va = float(v_after[sf_idx])
				if np.isfinite(vb) and np.isfinite(va):
					rows.append({"subject": subj, "training": "Pre", "sf": sf, "value": vb})
					rows.append({"subject": subj, "training": "Post", "sf": sf, "value": va})
		p = _art_raov_2way_pvalues_from_rows(
			rows,
			subject='subject',
			factor_a='training',
			factor_b='sf',
			value_col='value',
			include_interactions=ART_INCLUDE_INTERACTION_TERMS,
		)
		results[condition] = {
			"training_p": p.get('a'),
			"sf_p": p.get('b'),
			"interaction_p": p.get('a:b'),
		}
	return results


def compute_art_raov_pvalues_orientation(metric_key: str) -> Dict[str, Dict[str, Optional[float]]]:
	"""Compute ART-Raov omnibus p-values for orientation metrics (Training×Ori)."""
	results: Dict[str, Dict[str, Optional[float]]] = {
		"WT": {"training_p": None, "orientation_p": None, "interaction_p": None},
		"RESCUED": {"training_p": None, "orientation_p": None, "interaction_p": None},
	}
	if not ART_R_ENABLE:
		return results
	for condition in ["WT", "RESCUED"]:
		pairs = MOUSE_PAIRS.get(condition, [])
		rows: List[Dict[str, object]] = []
		for session_before, session_after in pairs:
			s_before = prepare_session_summaries(session_before)
			s_after = prepare_session_summaries(session_after)
			v_before = s_before.get(metric_key)
			v_after = s_after.get(metric_key)
			if v_before is None or v_after is None:
				continue
			v_before = np.asarray(v_before, dtype=float)
			v_after = np.asarray(v_after, dtype=float)
			if v_before.ndim != 1 or v_after.ndim != 1:
				continue
			min_len = min(v_before.size, v_after.size)
			if min_len < 2:
				continue
			subj = _mouse_id_from_session_rel(session_before)
			for oi in range(min_len):
				vb = float(v_before[oi])
				va = float(v_after[oi])
				if np.isfinite(vb) and np.isfinite(va):
					rows.append({"subject": subj, "training": "Pre", "ori": str(int(oi)), "value": vb})
					rows.append({"subject": subj, "training": "Post", "ori": str(int(oi)), "value": va})
		p = _art_raov_2way_pvalues_from_rows(
			rows,
			subject='subject',
			factor_a='training',
			factor_b='ori',
			value_col='value',
			include_interactions=ART_INCLUDE_INTERACTION_TERMS,
		)
		results[condition] = {
			"training_p": p.get('a'),
			"orientation_p": p.get('b'),
			"interaction_p": p.get('a:b'),
		}
	return results


def _lme4_lrt_p(
	*,
	df: pd.DataFrame,
	full_fixed: str,
	reduced_fixed: str,
	subject: str,
) -> Optional[float]:
	"""Likelihood-ratio p-value using lme4::lmer (REML=FALSE).

	full_fixed / reduced_fixed are fixed-effect formulas WITHOUT random effects.
	Random intercept (1|subject) is appended.
	"""
	imp = _try_import_lme4()
	if imp is None:
		return None
	ro, pandas2ri, localconverter, s_stats, lme4, base = imp
	if df.empty or subject not in df.columns or df[subject].nunique() < 2:
		return None
	try:
		df_r = df.copy()
		# Ensure factors are treated as categorical on the R side.
		# (R >= 4.0 defaults stringsAsFactors=FALSE, so we coerce explicitly.)
		for c in df_r.columns:
			if c == 'value':
				continue
			if df_r[c].dtype == object or str(df_r[c].dtype).startswith('category'):
				df_r[c] = df_r[c].astype(str)
		with localconverter(ro.default_converter + pandas2ri.converter):
			r_df = ro.conversion.py2rpy(df_r)
		make_factors = ro.r(
			'function(d){ for(n in names(d)) if(n != "value") d[[n]] <- as.factor(d[[n]]); d }'
		)
		r_df = make_factors(r_df)
		full_formula = ro.Formula(f"value ~ {full_fixed} + (1|{subject})")
		reduced_formula = ro.Formula(f"value ~ {reduced_fixed} + (1|{subject})")
		# Suppress common lme4 warnings (e.g., singular fit) to avoid console spam.
		quiet_lmer = ro.r(
			'function(fml, d, ctrl){ withCallingHandlers(lme4::lmer(fml, data=d, REML=FALSE, control=ctrl), '
			'warning=function(w) invokeRestart("muffleWarning"), '
			'message=function(m) invokeRestart("muffleMessage")) }'
		)
		try:
			ctrl = ro.r('lme4::lmerControl(check.conv.singular = "ignore")')
		except Exception:
			ctrl = ro.r('lme4::lmerControl()')
		m_full = quiet_lmer(full_formula, r_df, ctrl)
		m_red = quiet_lmer(reduced_formula, r_df, ctrl)

		# Avoid stats::anova(m_red, m_full) here: under rpy2 it can emit
		# "failed to find model names" repeatedly. Compute LRT directly.
		ll_full = s_stats.logLik(m_full)
		ll_red = s_stats.logLik(m_red)
		try:
			llf = float(base.as_numeric(ll_full)[0])
			llr = float(base.as_numeric(ll_red)[0])
		except Exception:
			return None
		try:
			dff = int(base.as_integer(base.attr(ll_full, 'df'))[0])
			dfr = int(base.as_integer(base.attr(ll_red, 'df'))[0])
		except Exception:
			# Fallback if df attribute isn't available
			return None
		df_diff = int(dff - dfr)
		if df_diff <= 0:
			return None
		stat = 2.0 * (llf - llr)
		if not np.isfinite(stat):
			return None
		# Numerical guard: nested models should give non-negative stat.
		stat = float(max(0.0, stat))
		p = float(base.as_numeric(s_stats.pchisq(stat, df=df_diff, lower_tail=False))[0])
		return p if np.isfinite(p) else None
	except Exception:
		return None


def _lme4_2way_lrt_pvalues_from_rows(
	rows: List[Dict[str, object]],
	*,
	subject: str,
	factor_a: str,
	factor_b: str,
	value_col: str = 'value',
	include_interactions: bool,
) -> Dict[str, Optional[float]]:
	"""2-way lme4 LRT omnibus p-values (A, B, A:B)."""
	out: Dict[str, Optional[float]] = {'a': None, 'b': None, 'a:b': None}
	if not LME4_ENABLE:
		return out
	df = pd.DataFrame(rows)
	if df.empty or value_col not in df.columns:
		return out
	df = df[[subject, factor_a, factor_b, value_col]].copy()
	df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
	df = df.dropna(axis=0, how='any')
	if df.empty:
		return out
	# Balance repeated-measures grid
	df = _balance_long_rm_df(df, subject=subject, within=[factor_a, factor_b], value_col=value_col)
	if df.empty:
		return out
	df = df.rename(columns={value_col: 'value'})
	if include_interactions:
		full = f"{factor_a}*{factor_b}"
		add = f"{factor_a}+{factor_b}"
	else:
		full = f"{factor_a}+{factor_b}"
		add = full

	# Main effects tested by dropping that factor from the full fixed effects.
	out['a'] = _lme4_lrt_p(df=df, full_fixed=full, reduced_fixed=f"{factor_b}", subject=subject)
	out['b'] = _lme4_lrt_p(df=df, full_fixed=full, reduced_fixed=f"{factor_a}", subject=subject)
	if include_interactions:
		out['a:b'] = _lme4_lrt_p(df=df, full_fixed=full, reduced_fixed=add, subject=subject)
	return out


def _lme4_3way_lrt_pvalues_from_rows(
	rows: List[Dict[str, object]],
	*,
	subject: str,
	factor_a: str,
	factor_b: str,
	factor_c: str,
	value_col: str = 'value',
	include_interactions: bool,
) -> Dict[str, Optional[float]]:
	"""3-way lme4 LRT omnibus p-values for main effects + interactions."""
	terms = [
		factor_a,
		factor_b,
		factor_c,
		f"{factor_a}:{factor_b}",
		f"{factor_a}:{factor_c}",
		f"{factor_b}:{factor_c}",
		f"{factor_a}:{factor_b}:{factor_c}",
	]
	out: Dict[str, Optional[float]] = {t: None for t in terms}
	if not LME4_ENABLE:
		return out
	df = pd.DataFrame(rows)
	if df.empty or value_col not in df.columns:
		return out
	df = df[[subject, factor_a, factor_b, factor_c, value_col]].copy()
	df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
	df = df.dropna(axis=0, how='any')
	if df.empty:
		return out
	# Balance repeated-measures grid
	df = _balance_long_rm_df(df, subject=subject, within=[factor_a, factor_b, factor_c], value_col=value_col)
	if df.empty:
		return out
	df = df.rename(columns={value_col: 'value'})

	if include_interactions:
		full = f"{factor_a}*{factor_b}*{factor_c}"
		m2 = (
			f"{factor_a}+{factor_b}+{factor_c}"
			f"+{factor_a}:{factor_b}+{factor_a}:{factor_c}+{factor_b}:{factor_c}"
		)
	else:
		full = f"{factor_a}+{factor_b}+{factor_c}"
		m2 = full

	# Main effects: drop factor and all interactions containing it.
	if include_interactions:
		out[factor_a] = _lme4_lrt_p(df=df, full_fixed=full, reduced_fixed=f"{factor_b}*{factor_c}", subject=subject)
		out[factor_b] = _lme4_lrt_p(df=df, full_fixed=full, reduced_fixed=f"{factor_a}*{factor_c}", subject=subject)
		out[factor_c] = _lme4_lrt_p(df=df, full_fixed=full, reduced_fixed=f"{factor_a}*{factor_b}", subject=subject)
	else:
		out[factor_a] = _lme4_lrt_p(df=df, full_fixed=full, reduced_fixed=f"{factor_b}+{factor_c}", subject=subject)
		out[factor_b] = _lme4_lrt_p(df=df, full_fixed=full, reduced_fixed=f"{factor_a}+{factor_c}", subject=subject)
		out[factor_c] = _lme4_lrt_p(df=df, full_fixed=full, reduced_fixed=f"{factor_a}+{factor_b}", subject=subject)

	if not include_interactions:
		return out

	# 3-way interaction (full vs 2-way-only)
	out[f"{factor_a}:{factor_b}:{factor_c}"] = _lme4_lrt_p(df=df, full_fixed=full, reduced_fixed=m2, subject=subject)

	# 2-way interactions within the 2-way model
	out[f"{factor_a}:{factor_b}"] = _lme4_lrt_p(
		df=df,
		full_fixed=m2,
		reduced_fixed=(
			f"{factor_a}+{factor_b}+{factor_c}"
			f"+{factor_a}:{factor_c}+{factor_b}:{factor_c}"
		),
		subject=subject,
	)
	out[f"{factor_a}:{factor_c}"] = _lme4_lrt_p(
		df=df,
		full_fixed=m2,
		reduced_fixed=(
			f"{factor_a}+{factor_b}+{factor_c}"
			f"+{factor_a}:{factor_b}+{factor_b}:{factor_c}"
		),
		subject=subject,
	)
	out[f"{factor_b}:{factor_c}"] = _lme4_lrt_p(
		df=df,
		full_fixed=m2,
		reduced_fixed=(
			f"{factor_a}+{factor_b}+{factor_c}"
			f"+{factor_a}:{factor_b}+{factor_a}:{factor_c}"
		),
		subject=subject,
	)

	return out


def compute_lme4_pvalues_sf(metric_key: str) -> Dict[str, Dict[str, Optional[float]]]:
	"""Compute lme4-based LME stats for SF metrics (Training×SF)."""
	results: Dict[str, Dict[str, Optional[float]]] = {
		"WT": {"training_p": None, "sf_p": None, "interaction_p": None},
		"RESCUED": {"training_p": None, "sf_p": None, "interaction_p": None},
	}
	if not LME4_ENABLE:
		return results
	for condition in ["WT", "RESCUED"]:
		pairs = MOUSE_PAIRS.get(condition, [])
		rows: List[Dict[str, object]] = []
		for session_before, session_after in pairs:
			s_before = prepare_session_summaries(session_before)
			s_after = prepare_session_summaries(session_after)
			v_before = s_before.get(metric_key)
			v_after = s_after.get(metric_key)
			if v_before is None or v_after is None:
				continue
			v_before = np.asarray(v_before, dtype=float)
			v_after = np.asarray(v_after, dtype=float)
			if v_before.ndim == 0:
				v_before = np.full(len(TARGET_SF_VALUES), float(v_before))
				v_after = np.full(len(TARGET_SF_VALUES), float(v_after))
			if v_before.ndim != 1 or v_after.ndim != 1:
				continue
			min_len = min(len(TARGET_SF_VALUES), v_before.size, v_after.size)
			if min_len <= 2:
				continue
			subj = _mouse_id_from_session_rel(session_before)
			for sf_idx in range(1, min_len):
				sf = str(float(TARGET_SF_VALUES[sf_idx]))
				vb = float(v_before[sf_idx])
				va = float(v_after[sf_idx])
				if np.isfinite(vb) and np.isfinite(va):
					rows.append({"subject": subj, "training": "Pre", "sf": sf, "value": vb})
					rows.append({"subject": subj, "training": "Post", "sf": sf, "value": va})
		p = _lme4_2way_lrt_pvalues_from_rows(
			rows,
			subject='subject',
			factor_a='training',
			factor_b='sf',
			value_col='value',
			include_interactions=LME_INCLUDE_INTERACTION_TERMS,
		)
		results[condition] = {
			"training_p": p.get('a'),
			"sf_p": p.get('b'),
			"interaction_p": p.get('a:b'),
		}
	return results


def compute_lme4_pvalues_orientation(metric_key: str) -> Dict[str, Dict[str, Optional[float]]]:
	"""Compute lme4-based LME stats for orientation metrics (Training×Ori)."""
	results: Dict[str, Dict[str, Optional[float]]] = {
		"WT": {"training_p": None, "orientation_p": None, "interaction_p": None},
		"RESCUED": {"training_p": None, "orientation_p": None, "interaction_p": None},
	}
	if not LME4_ENABLE:
		return results
	for condition in ["WT", "RESCUED"]:
		pairs = MOUSE_PAIRS.get(condition, [])
		rows: List[Dict[str, object]] = []
		for session_before, session_after in pairs:
			s_before = prepare_session_summaries(session_before)
			s_after = prepare_session_summaries(session_after)
			v_before = s_before.get(metric_key)
			v_after = s_after.get(metric_key)
			if v_before is None or v_after is None:
				continue
			v_before = np.asarray(v_before, dtype=float)
			v_after = np.asarray(v_after, dtype=float)
			if v_before.ndim != 1 or v_after.ndim != 1:
				continue
			min_len = min(v_before.size, v_after.size)
			if min_len < 2:
				continue
			subj = _mouse_id_from_session_rel(session_before)
			for oi in range(min_len):
				vb = float(v_before[oi])
				va = float(v_after[oi])
				if np.isfinite(vb) and np.isfinite(va):
					rows.append({"subject": subj, "training": "Pre", "ori": str(int(oi)), "value": vb})
					rows.append({"subject": subj, "training": "Post", "ori": str(int(oi)), "value": va})
		p = _lme4_2way_lrt_pvalues_from_rows(
			rows,
			subject='subject',
			factor_a='training',
			factor_b='ori',
			value_col='value',
			include_interactions=LME_INCLUDE_INTERACTION_TERMS,
		)
		results[condition] = {
			"training_p": p.get('a'),
			"orientation_p": p.get('b'),
			"interaction_p": p.get('a:b'),
		}
	return results


def compute_lme_pvalues_sf(metric_key: str) -> Dict[str, Dict[str, Optional[float]]]:
	"""Compute MixedLM (LRT) stats for SF-based metrics (Training × SF)."""
	results: Dict[str, Dict[str, Optional[float]]] = {
		"WT": {"training_p": None, "sf_p": None, "interaction_p": None},
		"RESCUED": {"training_p": None, "sf_p": None, "interaction_p": None},
	}
	for condition in ["WT", "RESCUED"]:
		pairs = MOUSE_PAIRS.get(condition, [])
		rows: List[Dict[str, object]] = []
		for session_before, session_after in pairs:
			s_before = prepare_session_summaries(session_before)
			s_after = prepare_session_summaries(session_after)
			v_before = s_before.get(metric_key)
			v_after = s_after.get(metric_key)
			if v_before is None or v_after is None:
				continue
			v_before = np.asarray(v_before, dtype=float)
			v_after = np.asarray(v_after, dtype=float)
			if v_before.ndim == 0:
				v_before = np.full(len(TARGET_SF_VALUES), float(v_before))
				v_after = np.full(len(TARGET_SF_VALUES), float(v_after))
			if v_before.ndim != 1 or v_after.ndim != 1:
				continue
			min_len = min(len(TARGET_SF_VALUES), v_before.size, v_after.size)
			if min_len <= 2:
				continue
			subj = _mouse_id_from_session_rel(session_before)
			for sf_idx in range(1, min_len):
				sf = str(float(TARGET_SF_VALUES[sf_idx]))
				vb = float(v_before[sf_idx])
				va = float(v_after[sf_idx])
				if np.isfinite(vb) and np.isfinite(va):
					rows.append({"subject": subj, "training": "Pre", "sf": sf, "value": vb})
					rows.append({"subject": subj, "training": "Post", "sf": sf, "value": va})
		p = _lme_2way_lr_pvalues_from_rows(
			rows,
			subject='subject',
			factor_a='training',
			factor_b='sf',
			value_col='value',
			include_interactions=LME_INCLUDE_INTERACTION_TERMS,
		)
		results[condition] = {
			"training_p": p.get('a'),
			"sf_p": p.get('b'),
			"interaction_p": p.get('a:b'),
		}
	return results


def compute_lme_pvalues_orientation(metric_key: str) -> Dict[str, Dict[str, Optional[float]]]:
	"""Compute MixedLM (LRT) stats for orientation-based metrics (Training × Ori)."""
	results: Dict[str, Dict[str, Optional[float]]] = {
		"WT": {"training_p": None, "orientation_p": None, "interaction_p": None},
		"RESCUED": {"training_p": None, "orientation_p": None, "interaction_p": None},
	}
	for condition in ["WT", "RESCUED"]:
		pairs = MOUSE_PAIRS.get(condition, [])
		rows: List[Dict[str, object]] = []
		for session_before, session_after in pairs:
			s_before = prepare_session_summaries(session_before)
			s_after = prepare_session_summaries(session_after)
			v_before = s_before.get(metric_key)
			v_after = s_after.get(metric_key)
			if v_before is None or v_after is None:
				continue
			v_before = np.asarray(v_before, dtype=float)
			v_after = np.asarray(v_after, dtype=float)
			if v_before.ndim == 0:
				# Fall back to 1D vector using known orientation count if available
				oris = s_before.get('orientations')
				n_ori = int(np.asarray(oris).size) if oris is not None else 0
				if n_ori <= 1:
					continue
				v_before = np.full(n_ori, float(v_before))
				v_after = np.full(n_ori, float(v_after))
			if v_before.ndim != 1 or v_after.ndim != 1:
				continue
			min_len = min(v_before.size, v_after.size)
			if min_len < 2:
				continue
			subj = _mouse_id_from_session_rel(session_before)
			for ori_idx in range(min_len):
				vb = float(v_before[ori_idx])
				va = float(v_after[ori_idx])
				if np.isfinite(vb) and np.isfinite(va):
					rows.append({"subject": subj, "training": "Pre", "ori": str(int(ori_idx)), "value": vb})
					rows.append({"subject": subj, "training": "Post", "ori": str(int(ori_idx)), "value": va})
		p = _lme_2way_lr_pvalues_from_rows(
			rows,
			subject='subject',
			factor_a='training',
			factor_b='ori',
			value_col='value',
			include_interactions=LME_INCLUDE_INTERACTION_TERMS,
		)
		results[condition] = {
			"training_p": p.get('a'),
			"orientation_p": p.get('b'),
			"interaction_p": p.get('a:b'),
		}
	return results


def compute_art_pvalues_sf(metric_key: str) -> Dict[str, Dict[str, Optional[float]]]:
	"""Compute ART non-parametric two-way stats for SF-based metrics (Training × SF).

	Returns dict: {group: {"training_p": p, "sf_p": p, "interaction_p": p}}
	"""
	results: Dict[str, Dict[str, Optional[float]]] = {
		"WT": {"training_p": None, "sf_p": None, "interaction_p": None},
		"RESCUED": {"training_p": None, "sf_p": None, "interaction_p": None},
	}

	for condition in ["WT", "RESCUED"]:
		pairs = MOUSE_PAIRS.get(condition, [])
		rows: List[Dict[str, object]] = []
		for session_before, session_after in pairs:
			s_before = prepare_session_summaries(session_before)
			s_after = prepare_session_summaries(session_after)
			v_before = s_before.get(metric_key)
			v_after = s_after.get(metric_key)
			if v_before is None or v_after is None:
				continue
			v_before = np.asarray(v_before, dtype=float)
			v_after = np.asarray(v_after, dtype=float)
			if v_before.ndim == 0:
				v_before = np.full(len(TARGET_SF_VALUES), float(v_before))
				v_after = np.full(len(TARGET_SF_VALUES), float(v_after))
			if v_before.ndim != 1 or v_after.ndim != 1:
				continue
			min_len = min(len(TARGET_SF_VALUES), v_before.size, v_after.size)
			if min_len <= 2:
				continue
			subject = _mouse_id_from_session_rel(session_before)
			for sf_idx in range(1, min_len):
				sf = str(float(TARGET_SF_VALUES[sf_idx]))
				vb = float(v_before[sf_idx])
				va = float(v_after[sf_idx])
				if np.isfinite(vb) and np.isfinite(va):
					rows.append({"subject": subject, "training": "Pre", "sf": sf, "value": vb})
					rows.append({"subject": subject, "training": "Post", "sf": sf, "value": va})

		df = pd.DataFrame(rows)
		if df.empty:
			continue
		grid = df.pivot_table(index='subject', columns=['training', 'sf'], values='value', aggfunc='mean')
		# Same SF intersection logic as RM-ANOVA
		all_sfs = sorted({c[1] for c in grid.columns})
		good_sfs: List[str] = []
		for sf in all_sfs:
			if (('Pre', sf) in grid.columns) and (('Post', sf) in grid.columns):
				col = grid[[('Pre', sf), ('Post', sf)]]
				if not col.isna().any().any():
					good_sfs.append(sf)
		if len(good_sfs) < 2:
			continue
		df = df[df['sf'].isin(good_sfs)].copy()
		grid2 = df.pivot_table(index='subject', columns=['training', 'sf'], values='value', aggfunc='mean')
		grid2 = grid2.dropna(axis=0, how='any')
		if grid2.shape[0] < 2:
			continue
		balanced_rows: List[Dict[str, object]] = []
		for subject in grid2.index:
			for training in ['Pre', 'Post']:
				for sf in good_sfs:
					balanced_rows.append({
						"subject": subject,
						"training": training,
						"sf": sf,
						"value": float(grid2.loc[subject, (training, sf)]),
					})
		df_bal = pd.DataFrame(balanced_rows)
		try:
			# Remove subject main effect
			df_bal['value_centered'] = df_bal['value'] - df_bal.groupby('subject')['value'].transform('mean')
			df_bal['training'] = df_bal['training'].astype('category')
			df_bal['sf'] = df_bal['sf'].astype('category')
			art = _aligned_rank_transform(
				df_bal,
				factor_a='training',
				factor_b='sf',
				subject='subject',
				value_col='value_centered',
				include_interaction=ART_INCLUDE_INTERACTION_TERMS,
			)
			df_bal['rank_A'] = art['A']
			df_bal['rank_B'] = art['B']
			p_int = np.nan
			if ART_INCLUDE_INTERACTION_TERMS:
				# ART: analyze each aligned-rank response with full model; take effect-specific p
				p_train = AnovaRM(df_bal, depvar='rank_A', subject='subject', within=['training', 'sf']).fit().anova_table.loc['training', 'Pr > F']
				p_sf = AnovaRM(df_bal, depvar='rank_B', subject='subject', within=['training', 'sf']).fit().anova_table.loc['sf', 'Pr > F']
				if 'AxB' in art:
					df_bal['rank_Inter'] = art['AxB']
					p_int = AnovaRM(df_bal, depvar='rank_Inter', subject='subject', within=['training', 'sf']).fit().anova_table.loc['training:sf', 'Pr > F']
			else:
				# Reduced model: test each main effect by collapsing ranks over the other factor.
				df_a = df_bal.groupby(['subject', 'training'], as_index=False)['rank_A'].mean()
				p_train = AnovaRM(df_a, depvar='rank_A', subject='subject', within=['training']).fit().anova_table.loc['training', 'Pr > F']
				df_b = df_bal.groupby(['subject', 'sf'], as_index=False)['rank_B'].mean()
				p_sf = AnovaRM(df_b, depvar='rank_B', subject='subject', within=['sf']).fit().anova_table.loc['sf', 'Pr > F']
			results[condition] = {
				"training_p": float(p_train) if np.isfinite(p_train) else None,
				"sf_p": float(p_sf) if np.isfinite(p_sf) else None,
				"interaction_p": float(p_int) if np.isfinite(p_int) else None,
			}
		except Exception as exc:
			print(f"[debug] ART SF failed for {condition}/{metric_key}: {type(exc).__name__}")
			continue

	return results


def compute_art_pvalues_orientation(metric_key: str) -> Dict[str, Dict[str, Optional[float]]]:
	"""Compute ART non-parametric two-way stats for orientation-based metrics (Training × Orientation)."""
	results: Dict[str, Dict[str, Optional[float]]] = {
		"WT": {"training_p": None, "orientation_p": None, "interaction_p": None},
		"RESCUED": {"training_p": None, "orientation_p": None, "interaction_p": None},
	}
	for condition in ["WT", "RESCUED"]:
		pairs = MOUSE_PAIRS.get(condition, [])
		rows: List[Dict[str, object]] = []
		for session_before, session_after in pairs:
			s_before = prepare_session_summaries(session_before)
			s_after = prepare_session_summaries(session_after)
			vals_b = s_before.get(metric_key)
			vals_a = s_after.get(metric_key)
			oris = s_before.get("orientations")
			if vals_b is None or vals_a is None or oris is None:
				continue
			vals_b = np.asarray(vals_b, dtype=float)
			vals_a = np.asarray(vals_a, dtype=float)
			if vals_b.ndim != 1 or vals_a.ndim != 1:
				continue
			min_len = min(len(oris), vals_b.size, vals_a.size)
			if min_len < 2:
				continue
			subject = _mouse_id_from_session_rel(session_before)
			for oi in range(min_len):
				vb = float(vals_b[oi])
				va = float(vals_a[oi])
				if np.isfinite(vb) and np.isfinite(va):
					rows.append({"subject": subject, "training": "Pre", "ori": str(int(oi)), "value": vb})
					rows.append({"subject": subject, "training": "Post", "ori": str(int(oi)), "value": va})
		df = pd.DataFrame(rows)
		if df.empty:
			continue
		grid = df.pivot_table(index='subject', columns=['training', 'ori'], values='value', aggfunc='mean')
		grid = grid.dropna(axis=0, how='any')
		if grid.shape[0] < 2:
			continue
		oris = sorted({c[1] for c in grid.columns})
		balanced_rows: List[Dict[str, object]] = []
		for subject in grid.index:
			for training in ['Pre', 'Post']:
				for ori in oris:
					balanced_rows.append({
						"subject": subject,
						"training": training,
						"ori": ori,
						"value": float(grid.loc[subject, (training, ori)]),
					})
		df_bal = pd.DataFrame(balanced_rows)
		try:
			df_bal['value_centered'] = df_bal['value'] - df_bal.groupby('subject')['value'].transform('mean')
			df_bal['training'] = df_bal['training'].astype('category')
			df_bal['ori'] = df_bal['ori'].astype('category')
			art = _aligned_rank_transform(
				df_bal,
				factor_a='training',
				factor_b='ori',
				subject='subject',
				value_col='value_centered',
				include_interaction=ART_INCLUDE_INTERACTION_TERMS,
			)
			df_bal['rank_A'] = art['A']
			df_bal['rank_B'] = art['B']
			p_int = np.nan
			if ART_INCLUDE_INTERACTION_TERMS:
				p_train = AnovaRM(df_bal, depvar='rank_A', subject='subject', within=['training', 'ori']).fit().anova_table.loc['training', 'Pr > F']
				p_ori = AnovaRM(df_bal, depvar='rank_B', subject='subject', within=['training', 'ori']).fit().anova_table.loc['ori', 'Pr > F']
				if 'AxB' in art:
					df_bal['rank_Inter'] = art['AxB']
					p_int = AnovaRM(df_bal, depvar='rank_Inter', subject='subject', within=['training', 'ori']).fit().anova_table.loc['training:ori', 'Pr > F']
			else:
				df_a = df_bal.groupby(['subject', 'training'], as_index=False)['rank_A'].mean()
				p_train = AnovaRM(df_a, depvar='rank_A', subject='subject', within=['training']).fit().anova_table.loc['training', 'Pr > F']
				df_b = df_bal.groupby(['subject', 'ori'], as_index=False)['rank_B'].mean()
				p_ori = AnovaRM(df_b, depvar='rank_B', subject='subject', within=['ori']).fit().anova_table.loc['ori', 'Pr > F']
			results[condition] = {
				"training_p": float(p_train) if np.isfinite(p_train) else None,
				"orientation_p": float(p_ori) if np.isfinite(p_ori) else None,
				"interaction_p": float(p_int) if np.isfinite(p_int) else None,
			}
		except Exception as exc:
			print(f"[debug] ART Ori failed for {condition}/{metric_key}: {type(exc).__name__}")
			continue
	return results


# ---------------------------------------------------------------------------
# Pseudopopulation analysis helpers
# ---------------------------------------------------------------------------

# For fair Before vs After comparisons, subsample the same number of neurons per session.
PSEUDOPOP_SUBSAMPLE_N_NEURONS_PER_SESSION = 80
PSEUDOPOP_SUBSAMPLE_SEED = 12345


def _rng_for_session(session_rel: str, *, base_seed: int = PSEUDOPOP_SUBSAMPLE_SEED) -> np.random.Generator:
	"""Deterministic RNG per session (stable across runs).

	Uses CRC32 of session path to derive a per-session seed.
	"""
	seed = int(base_seed) ^ int(zlib.crc32(session_rel.encode("utf-8")))
	return np.random.default_rng(seed)


def _subsample_neurons_matrix_per_session(condition_means: np.ndarray, session_rel: str, *, n_select: int = PSEUDOPOP_SUBSAMPLE_N_NEURONS_PER_SESSION) -> np.ndarray:
	"""Subsample up to n_select neurons (rows) from a (neurons, conditions) matrix.

	If fewer than n_select valid neurons exist, uses all valid neurons.
	"""
	cm = np.asarray(condition_means, dtype=float)
	if cm.ndim != 2 or cm.shape[0] == 0:
		return cm
	valid = np.isfinite(cm).any(axis=1)
	candidates = np.where(valid)[0]
	if candidates.size == 0:
		candidates = np.arange(cm.shape[0])

	n_use = int(min(int(n_select), int(candidates.size)))
	if n_use <= 0:
		return cm[:0, :]
	if candidates.size <= n_use:
		chosen = candidates
	else:
		rng = _rng_for_session(session_rel)
		chosen = rng.choice(candidates, size=n_use, replace=False)
	chosen = np.sort(np.asarray(chosen, dtype=int))
	return cm[chosen, :]

def participation_ratio(eigvals: np.ndarray) -> float:
	"""Compute participation ratio from eigenvalues."""
	eigvals = np.asarray(eigvals, dtype=float)
	eigvals = eigvals[eigvals > 0]
	if eigvals.size == 0:
		return np.nan
	return (np.sum(eigvals) ** 2) / np.sum(eigvals ** 2)


def load_orientation_tuning_matrix(session_rel: str) -> Optional[np.ndarray]:
	"""Load orientation_tuning.npy and return as (neurons, conditions) matrix.
	
	Orientation tuning has shape (n_neurons, n_orientations, n_sf).
	Reshape to (n_neurons, n_orientations * n_sf) treating each (ori, sf) as a condition.
	"""
	session_dir = os.path.join(RESULTS_ROOT, session_rel)
	tuning_path = os.path.join(session_dir, "orientation_tuning.npy")
	
	if not os.path.exists(tuning_path):
		print(f"[warn] Missing orientation_tuning.npy for {session_rel}")
		return None
	
	try:
		tuning = np.load(tuning_path, allow_pickle=True)  # (neurons, ori, sf)
		if tuning.ndim != 3:
			print(f"[warn] Unexpected tuning shape {tuning.shape} for {session_rel}")
			return None
		
		# Reshape to (neurons, conditions) where conditions = ori × sf
		n_neurons, n_ori, n_sf = tuning.shape
		condition_means = tuning.reshape(n_neurons, n_ori * n_sf)
		return condition_means
		
	except Exception as exc:
		print(f"[error] Failed to load {tuning_path}: {exc}")
		return None


def load_orientation_tuning_matrix_cv(session_rel: str, *, fold: int) -> Optional[np.ndarray]:
	"""Load aggregated CV train/test tuning and return as (neurons, conditions) matrix.

	When present, this prefers the newer orientation_tuning_cv10.npz bundle and reduces the
	per-fold train/test means into a single weighted train (fold=1) or held-out (fold=2)
	condition matrix. Legacy orientation_tuning_cv_fold{fold}.npy files remain supported.
	"""
	session_dir = os.path.join(RESULTS_ROOT, session_rel)
	bundle_path = os.path.join(session_dir, "orientation_tuning_cv10.npz")
	if os.path.exists(bundle_path):
		try:
			with np.load(bundle_path, allow_pickle=True) as bundle:
				train_means = np.asarray(bundle["train_means_by_fold"], dtype=float)
				test_means = np.asarray(bundle["test_means_by_fold"], dtype=float)
				train_counts = np.asarray(bundle["train_counts_by_fold"], dtype=float)
				test_counts = np.asarray(bundle["test_counts_by_fold"], dtype=float)
			if train_means.ndim != 4 or test_means.ndim != 4 or train_counts.ndim != 3 or test_counts.ndim != 3:
				print(f"[warn] Unexpected CV bundle shapes in {bundle_path}")
				return None
			n_folds, n_neurons, n_ori, n_sf = train_means.shape
			train_flat = train_means.reshape(n_folds, n_neurons, n_ori * n_sf)
			test_flat = test_means.reshape(n_folds, n_neurons, n_ori * n_sf)
			train_counts_flat = train_counts.reshape(n_folds, n_ori * n_sf)
			test_counts_flat = test_counts.reshape(n_folds, n_ori * n_sf)
			if int(fold) == 1:
				weighted_sum = np.nansum(train_flat * train_counts_flat[:, None, :], axis=0)
				weight_total = np.sum(train_counts_flat, axis=0)
			elif int(fold) == 2:
				weighted_sum = np.nansum(test_flat * test_counts_flat[:, None, :], axis=0)
				weight_total = np.sum(test_counts_flat, axis=0)
			else:
				print(f"[warn] Unsupported CV aggregate fold={fold} for {session_rel}")
				return None
			return np.divide(
				weighted_sum,
				weight_total[None, :],
				out=np.full((n_neurons, n_ori * n_sf), np.nan, dtype=float),
				where=weight_total[None, :] > 0,
			)
		except Exception as exc:
			print(f"[error] Failed to load {bundle_path}: {exc}")
			return None

	tuning_path = os.path.join(session_dir, f"orientation_tuning_cv_fold{int(fold)}.npy")
	if not os.path.exists(tuning_path):
		print(f"[warn] Missing orientation_tuning_cv_fold{int(fold)}.npy for {session_rel}")
		return None
	try:
		tuning = np.load(tuning_path, allow_pickle=True)  # (neurons, ori, sf)
		if tuning.ndim != 3:
			print(f"[warn] Unexpected CV tuning shape {tuning.shape} for {session_rel}")
			return None
		n_neurons, n_ori, n_sf = tuning.shape
		condition_means = tuning.reshape(n_neurons, n_ori * n_sf)
		return condition_means
	except Exception as exc:
		print(f"[error] Failed to load {tuning_path}: {exc}")
		return None


def load_orientation_tuning_cv_folds(session_rel: str) -> Optional[Dict[str, np.ndarray]]:
	"""Load per-fold CV tuning matrices for pseudopopulation 10-fold dsPCA."""
	session_dir = os.path.join(RESULTS_ROOT, session_rel)
	bundle_path = os.path.join(session_dir, "orientation_tuning_cv10.npz")
	if os.path.exists(bundle_path):
		try:
			with np.load(bundle_path, allow_pickle=True) as bundle:
				train_means = np.asarray(bundle["train_means_by_fold"], dtype=float)
				test_means = np.asarray(bundle["test_means_by_fold"], dtype=float)
				train_counts = np.asarray(bundle["train_counts_by_fold"], dtype=float)
				test_counts = np.asarray(bundle["test_counts_by_fold"], dtype=float)
				if "n_folds" in bundle.files:
					n_folds = int(np.asarray(bundle["n_folds"]).item())
				else:
					n_folds = int(train_means.shape[0])
			if train_means.ndim != 4 or test_means.ndim != 4 or train_counts.ndim != 3 or test_counts.ndim != 3:
				print(f"[warn] Unexpected CV bundle shapes in {bundle_path}")
				return None
			n_folds_arr, n_neurons, n_ori, n_sf = train_means.shape
			if n_folds_arr != n_folds:
				n_folds = int(n_folds_arr)
			return {
				"train_means_by_fold": train_means.reshape(n_folds, n_neurons, n_ori * n_sf),
				"test_means_by_fold": test_means.reshape(n_folds, n_neurons, n_ori * n_sf),
				"train_counts_by_fold": train_counts.reshape(n_folds, n_ori * n_sf),
				"test_counts_by_fold": test_counts.reshape(n_folds, n_ori * n_sf),
				"n_folds": np.array([n_folds], dtype=int),
			}
		except Exception as exc:
			print(f"[error] Failed to load {bundle_path}: {exc}")
			return None

	cm_train = load_orientation_tuning_matrix_cv(session_rel, fold=1)
	cm_test = load_orientation_tuning_matrix_cv(session_rel, fold=2)
	if cm_train is None or cm_test is None:
		return None
	if cm_train.shape != cm_test.shape:
		print(f"[warn] Legacy CV fold shape mismatch in {session_rel}: {cm_train.shape} vs {cm_test.shape}")
		return None
	n_conditions = int(cm_train.shape[1])
	one_counts = np.ones((1, n_conditions), dtype=float)
	return {
		"train_means_by_fold": cm_train[None, :, :],
		"test_means_by_fold": cm_test[None, :, :],
		"train_counts_by_fold": one_counts,
		"test_counts_by_fold": one_counts,
		"n_folds": np.array([1], dtype=int),
	}


def build_pseudopopulation(group_sessions: List[str], *, subsample_neurons: bool = True) -> Tuple[Optional[np.ndarray], int]:
	"""Build pseudopopulation by stacking neurons from all sessions in a group.
	
	Returns:
		(condition_means, n_conditions) where condition_means is (N_total_neurons, n_conditions)
		or (None, 0) if insufficient data.
	"""
	all_neurons = []
	n_conditions_ref = None
	
	for session in group_sessions:
		condition_means = load_orientation_tuning_matrix(session)
		if condition_means is None:
			continue

		# Subsample neurons per session so Before/After comparisons use matched N.
		if subsample_neurons:
			condition_means = _subsample_neurons_matrix_per_session(condition_means, session)
		
		if n_conditions_ref is None:
			n_conditions_ref = condition_means.shape[1]
		elif condition_means.shape[1] != n_conditions_ref:
			print(f"[warn] Condition count mismatch in {session}: expected {n_conditions_ref}, got {condition_means.shape[1]}")
			continue
		
		all_neurons.append(condition_means)
	
	if len(all_neurons) == 0:
		return None, 0
	
	pseudopop = np.vstack(all_neurons)  # (N_total, n_conditions)
	print(f"  Built pseudopopulation: {pseudopop.shape[0]} neurons, {pseudopop.shape[1]} conditions")
	return pseudopop, n_conditions_ref


def build_pseudopopulation_cv50(
	group_sessions: List[str],
	*,
	subsample_neurons: bool = True,
	n_select: int = PSEUDOPOP_SUBSAMPLE_N_NEURONS_PER_SESSION,
	) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:
	"""Build matched full/train/test pseudopopulations from per-fold CV tuning matrices.

	Returns:
		(full_means, train_means_by_fold, test_means_by_fold, test_counts_by_fold, n_conditions)
	where full_means is (N_total_neurons, n_conditions), train/test folds are
	(n_folds, N_total_neurons, n_conditions), and test counts are (n_folds, n_conditions).
	"""
	all_full = []
	all_train = []
	all_test = []
	test_counts_sum = None
	n_conditions_ref = None
	n_folds_ref = None

	for session in group_sessions:
		cm_full = load_orientation_tuning_matrix(session)
		cv_bundle = load_orientation_tuning_cv_folds(session)
		if cm_full is None or cv_bundle is None:
			continue

		train_folds = np.asarray(cv_bundle["train_means_by_fold"], dtype=float)
		test_folds = np.asarray(cv_bundle["test_means_by_fold"], dtype=float)
		test_counts = np.asarray(cv_bundle["test_counts_by_fold"], dtype=float)
		full_arr = np.asarray(cm_full, dtype=float)
		if train_folds.ndim != 3 or test_folds.ndim != 3 or test_counts.ndim != 2:
			print(f"[warn] Unexpected CV fold bundle shapes in {session}")
			continue
		if train_folds.shape != test_folds.shape:
			print(f"[warn] CV fold shape mismatch in {session}: {train_folds.shape} vs {test_folds.shape}")
			continue
		if full_arr.shape[0] != train_folds.shape[1] or full_arr.shape[1] != train_folds.shape[2]:
			print(f"[warn] Full/CV condition mismatch in {session}: {full_arr.shape} vs {train_folds.shape}")
			continue

		# Subsample the same neurons in full/train/test arrays for this session.
		if subsample_neurons:
			if full_arr.ndim != 2 or full_arr.shape[0] == 0:
				continue
			valid_full = np.isfinite(full_arr).any(axis=1)
			valid_train = np.isfinite(train_folds).any(axis=(0, 2))
			valid_test = np.isfinite(test_folds).any(axis=(0, 2))
			candidates = np.where(valid_full & valid_train & valid_test)[0]
			if candidates.size == 0:
				candidates = np.arange(full_arr.shape[0])
			n_use = int(min(int(n_select), int(candidates.size)))
			if candidates.size <= n_use:
				chosen = candidates
			else:
				rng = _rng_for_session(session)
				chosen = rng.choice(candidates, size=n_use, replace=False)
			chosen = np.sort(np.asarray(chosen, dtype=int))
			full_arr = full_arr[chosen, :]
			train_folds = train_folds[:, chosen, :]
			test_folds = test_folds[:, chosen, :]

		if n_conditions_ref is None:
			n_conditions_ref = int(full_arr.shape[1])
		elif int(full_arr.shape[1]) != int(n_conditions_ref):
			print(f"[warn] Condition count mismatch in {session}: expected {n_conditions_ref}, got {full_arr.shape[1]}")
			continue

		if n_folds_ref is None:
			n_folds_ref = int(train_folds.shape[0])
		elif int(train_folds.shape[0]) != int(n_folds_ref):
			print(f"[warn] Fold count mismatch in {session}: expected {n_folds_ref}, got {train_folds.shape[0]}")
			continue

		all_full.append(full_arr)
		all_train.append(train_folds)
		all_test.append(test_folds)
		if test_counts_sum is None:
			test_counts_sum = np.array(test_counts, copy=True)
		else:
			test_counts_sum = test_counts_sum + test_counts

	if len(all_train) == 0:
		return None, None, None, None, 0

	full_pseudopop = np.vstack(all_full)
	train_pseudopop = np.concatenate(all_train, axis=1)
	test_pseudopop = np.concatenate(all_test, axis=1)
	print(
		f"  Built CV pseudopopulation: {full_pseudopop.shape[0]} neurons, "
		f"{full_pseudopop.shape[1]} conditions, {train_pseudopop.shape[0]} folds"
	)
	return full_pseudopop, train_pseudopop, test_pseudopop, test_counts_sum, int(n_conditions_ref)


def _project_with_axes(data_cond_by_neuron: np.ndarray, axes: np.ndarray) -> np.ndarray:
	"""Project (conditions, neurons) data using dsPCA axes, handling axis orientation."""
	X = np.asarray(data_cond_by_neuron, dtype=float)
	A = np.asarray(axes, dtype=float)
	if X.ndim != 2 or A.ndim != 2:
		raise ValueError(f"Invalid shapes for projection: data {X.shape}, axes {A.shape}")
	if A.shape[0] == X.shape[1]:
		return X @ A
	if A.shape[1] == X.shape[1]:
		return X @ A.T
	raise ValueError(f"Axes shape {A.shape} not compatible with data {X.shape}")


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


def _orthogonality_sf_vs_ori(ax_targets: np.ndarray) -> float:
	"""Compute max |dot| between SF axis and the two orientation axes in neuron space."""
	A = np.asarray(ax_targets, dtype=float)
	if A.ndim != 2:
		return float("nan")
	# Expect 3 target axes: [SF, sin, cos] in some order matching targets columns.
	if A.shape[1] < 3 and A.shape[0] < 3:
		return float("nan")
	# Put into (neurons, components)
	if A.shape[0] < A.shape[1]:
		# Might already be (components, neurons)
		if A.shape[1] >= 3:
			A_nc = A.T
		else:
			A_nc = A
	else:
		A_nc = A
	if A_nc.shape[1] < 3:
		return float("nan")
	# Normalize
	def _unit(v: np.ndarray) -> np.ndarray:
		n = float(np.linalg.norm(v))
		return v / (n + 1e-12)
	sf = _unit(A_nc[:, 0])
	ori1 = _unit(A_nc[:, 1])
	ori2 = _unit(A_nc[:, 2])
	return float(max(abs(float(np.dot(sf, ori1))), abs(float(np.dot(sf, ori2)))))


def _pseudopopulation_dspca_make_plots(
	*,
	results: Dict[str, Dict[str, object]],
	group_names: List[str],
	group_labels_short: List[str],
	group_colors_pca: List[str],
	mapped_sfs_linear: List[float],
	out_tag: str,
) -> None:
	"""Shared plotting/saving for pseudopopulation dsPCA runs."""
	if not results:
		return

	# Save results
	np.savez(
		os.path.join(GROUPED_PLOTS_DIR, f'pseudopopulation_dspca_results{out_tag}.npz'),
		**{f"{name}_{key}": val for name, data in results.items()
		   for key, val in data.items() if key not in ['projection_target', 'projection_targetfree', 'ax_targets', 'targets', 'dspca']}
	)

	# Plot 1: Target variance comparison (Ori and SF with 2D orientation subspace)
	fig, ax = plt.subplots(figsize=(10, 6))
	x_pos = np.arange(len(results))
	width = 0.35
	var_ori_vals = [results[gn]['var_ori_pct'] for gn in group_names if gn in results]
	var_sf_vals = [results[gn]['var_sf_pct'] for gn in group_names if gn in results]
	labels = [group_labels_short[i] for i, gn in enumerate(group_names) if gn in results]
	colors_present = [group_colors_pca[i] for i, gn in enumerate(group_names) if gn in results]
	bars1 = ax.bar(x_pos - width/2, var_ori_vals, width, label='Orientation (2D subspace)',
			  color=colors_present, alpha=0.6, edgecolor='black', linewidth=1)
	bars2 = ax.bar(x_pos + width/2, var_sf_vals, width, label='SF (linear)',
			  color=colors_present, alpha=0.9, edgecolor='black', linewidth=1)
	ax.set_xticks(x_pos)
	ax.set_xticklabels(labels, rotation=15, ha='right')
	ax.set_ylabel('% Variance Explained', fontsize=12)
	ax.set_title('Pseudopopulation dsPCA: Target Axis Variance\n(2D Orientation [sin, cos] vs SF)', fontsize=13, fontweight='bold')
	ax.legend()
	ax.grid(axis='y', alpha=0.3)
	for bars in [bars1, bars2]:
		for bar in bars:
			height = bar.get_height()
			if np.isfinite(height):
				ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
					   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
	plt.tight_layout()
	plt.savefig(os.path.join(GROUPED_PLOTS_DIR, f'pseudopop_dspca_target_variance_comparison{out_tag}.svg'), format='svg')
	plt.close()
	print(f"  [saved] pseudopop_dspca_target_variance_comparison{out_tag}.svg")

	# Plot 2: Orthogonality comparison
	fig, ax = plt.subplots(figsize=(8, 5))
	orth_vals = [results[gn]['orthogonality'] for gn in group_names if gn in results]
	bars = ax.bar(x_pos, np.abs(orth_vals), color=colors_present, alpha=0.8, edgecolor='black', linewidth=1.5)
	ax.set_xticks(x_pos)
	ax.set_xticklabels(labels, rotation=15, ha='right')
	ax.set_ylabel('|Ori·SF| (Orthogonality)', fontsize=12)
	ax.set_title('Pseudopopulation dsPCA: Axis Orthogonality\n(Lower = more demixed)', fontsize=13, fontweight='bold')
	ax.grid(axis='y', alpha=0.3)
	for bar, orth_val in zip(bars, orth_vals):
		if np.isfinite(orth_val):
			ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(np.abs(orth_vals))*0.02,
				   f'{orth_val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
	plt.tight_layout()
	plt.savefig(os.path.join(GROUPED_PLOTS_DIR, f'pseudopop_dspca_orthogonality_comparison{out_tag}.svg'), format='svg')
	plt.close()
	print(f"  [saved] pseudopop_dspca_orthogonality_comparison{out_tag}.svg")

	# Plot 3: Total target vs target-free variance
	fig, ax = plt.subplots(figsize=(10, 6))
	var_target_vals = [results[gn]['var_target_pct'] for gn in group_names if gn in results]
	var_targetfree_vals = [results[gn]['var_targetfree_top10_pct'] for gn in group_names if gn in results]
	bars1 = ax.bar(x_pos - width/2, var_target_vals, width, label='Target (Ori+SF)',
			  color=colors_present, alpha=0.9, edgecolor='black', linewidth=1.5)
	bars2 = ax.bar(x_pos + width/2, var_targetfree_vals, width, label='Target-free (top 10)',
			  color='gray', alpha=0.6, edgecolor='black', linewidth=1)
	ax.set_xticks(x_pos)
	ax.set_xticklabels(labels, rotation=15, ha='right')
	ax.set_ylabel('% Variance Explained', fontsize=12)
	ax.set_title('Pseudopopulation dsPCA: Target vs Target-Free Variance', fontsize=13, fontweight='bold')
	ax.legend()
	ax.grid(axis='y', alpha=0.3)
	for bars in [bars1, bars2]:
		for bar in bars:
			height = bar.get_height()
			if np.isfinite(height):
				ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
					   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
	plt.tight_layout()
	plt.savefig(os.path.join(GROUPED_PLOTS_DIR, f'pseudopop_dspca_variance_decomposition{out_tag}.svg'), format='svg')
	plt.close()
	print(f"  [saved] pseudopop_dspca_variance_decomposition{out_tag}.svg")

	# Delegate the remaining detailed per-group plots to the existing inline code by
	# running a minimal version of the per-group plotting block below.
	# NOTE: This function is placed above run_pseudopopulation_dspca, so it can reuse
	# the same plotting logic without duplicating it.
	# The detailed plots are produced by calling _pseudopopulation_dspca_make_detailed_plots.
	_pseudopopulation_dspca_make_detailed_plots(
		results=results,
		group_names=group_names,
		mapped_sfs_linear=mapped_sfs_linear,
		out_tag=out_tag,
	)


def _pseudopopulation_dspca_make_detailed_plots(
	*,
	results: Dict[str, Dict[str, object]],
	group_names: List[str],
	mapped_sfs_linear: List[float],
	out_tag: str,
) -> None:
	"""Generate the detailed per-group dsPCA figures (3D subspaces and per-SF panels)."""
	# ========== 3D dsPCA Subspace Visualizations ==========
	# Create detailed subspace plots for each group
	for group_name in group_names:
		if group_name not in results:
			continue

		targets = np.asarray(results[group_name]['targets'], dtype=float)
		projection_target = np.asarray(results[group_name]['projection_target'], dtype=float)
		projection_targetfree = np.asarray(results[group_name]['projection_targetfree'], dtype=float)
		target_var = np.asarray(results[group_name]['target_var'], dtype=float)
		targetfree_var = np.asarray(results[group_name]['targetfree_var'], dtype=float)
		total_var = float(results[group_name]['total_var'])

		mapped_sf_targets = results[group_name].get('mapped_sf_targets', None)
		if mapped_sf_targets is None:
			mapped_sf_targets = np.where(np.abs(targets[:, 0]) > 0, 2 ** targets[:, 0], 0.0)
		mapped_sf_targets = np.asarray(mapped_sf_targets, dtype=float)
		sf_proj_scores = np.asarray(projection_target[:, 0], dtype=float)

		ori_from_targets = np.arctan2(targets[:, 1], targets[:, 2]) / 2
		ori_from_targets = np.rad2deg(ori_from_targets)
		ori_from_targets = np.mod(ori_from_targets, 180)
		unique_orientations = np.unique(np.round(ori_from_targets / 30.0) * 30.0)

		# Use the dsPCA SF-axis projection score so all three displayed axes are in projection space.
		mapped_sfs: List[float] = []
		sf_ticks: List[float] = []
		tick_labels: List[str] = []
		for sf_val in mapped_sfs_linear:
			mask_sf = np.isclose(mapped_sf_targets, float(sf_val), atol=1e-12)
			proj_vals = sf_proj_scores[mask_sf]
			proj_vals = proj_vals[np.isfinite(proj_vals)]
			if proj_vals.size == 0:
				continue
			mapped_sfs.append(float(sf_val))
		finite_sf_proj = sf_proj_scores[np.isfinite(sf_proj_scores)]
		if finite_sf_proj.size > 0:
			n_display_ticks = max(2, min(6, int(finite_sf_proj.size)))
			z_min = float(np.nanmin(finite_sf_proj))
			z_max = float(np.nanmax(finite_sf_proj))
			if np.isfinite(z_min) and np.isfinite(z_max):
				if abs(z_max - z_min) < 1e-12:
					sf_ticks = [z_min]
				else:
					sf_ticks = list(np.linspace(z_min, z_max, n_display_ticks))
				tick_labels = [f"{tick:.2f}" for tick in sf_ticks]

		ref_com_sin: Optional[float] = None
		ref_com_cos: Optional[float] = None
		ori_mean_sin: List[float] = []
		ori_mean_cos: List[float] = []
		for ori in unique_orientations:
			mask_ori = np.abs(ori_from_targets - ori) < 1
			if np.sum(mask_ori) == 0:
				continue
			m_sin = float(np.nanmean(projection_target[mask_ori, 1]))
			m_cos = float(np.nanmean(projection_target[mask_ori, 2]))
			if np.isfinite(m_sin) and np.isfinite(m_cos):
				ori_mean_sin.append(m_sin)
				ori_mean_cos.append(m_cos)
		if ori_mean_sin and ori_mean_cos:
			ref_com_sin = float(np.mean(ori_mean_sin))
			ref_com_cos = float(np.mean(ori_mean_cos))

		ori_colors = plt.cm.turbo(np.linspace(0, 1, len(unique_orientations) + 1)[:-1])
		orientation_data = []
		for ori_idx, ori in enumerate(unique_orientations):
			mask = (np.abs(ori_from_targets - ori) < 1) & np.isfinite(sf_proj_scores) & np.isfinite(mapped_sf_targets) & (mapped_sf_targets > 0)
			if np.sum(mask) > 0:
				proj_sf = sf_proj_scores[mask]
				mapped_sf = mapped_sf_targets[mask]
				proj_sin = projection_target[mask, 1]
				proj_cos = projection_target[mask, 2]
				sort_idx = np.argsort(mapped_sf)
				orientation_data.append({
					"ori_idx": ori_idx,
					"ori": ori,
					"sf_proj": proj_sf[sort_idx],
					"sf_mapped": mapped_sf[sort_idx],
					"sin": proj_sin[sort_idx],
					"cos": proj_cos[sort_idx],
				})

			def _marker_sizes_from_sf_categories(sf_values: np.ndarray, *, min_size: float, max_size: float) -> np.ndarray:
				sf_values = np.asarray(sf_values, dtype=float).ravel()
				if sf_values.size == 0:
					return np.array([], dtype=float)
				positive_levels = np.array(sorted([float(sf) for sf in mapped_sfs_linear if float(sf) > 0]), dtype=float)
				if positive_levels.size == 0:
					return np.full(sf_values.shape, max_size, dtype=float)
				if positive_levels.size == 1:
					sizes = np.full(sf_values.shape, min_size, dtype=float)
					sizes[np.isfinite(sf_values) & (sf_values > 0)] = max_size
					return sizes
				size_levels = np.linspace(float(max_size), float(min_size), positive_levels.size)
				sizes = np.full(sf_values.shape, float(min_size), dtype=float)
				matched = np.zeros(sf_values.shape, dtype=bool)
				for level_idx, level in enumerate(positive_levels):
					mask_level = np.isclose(sf_values, level, atol=1e-12)
					sizes[mask_level] = size_levels[level_idx]
					matched |= mask_level
				unmatched_positive = np.isfinite(sf_values) & (sf_values > 0) & (~matched)
				if np.any(unmatched_positive):
					nearest_idx = np.argmin(np.abs(sf_values[unmatched_positive, None] - positive_levels[None, :]), axis=1)
					sizes[unmatched_positive] = size_levels[nearest_idx]
				return sizes

		fig = plt.figure(figsize=(24, 14))
		gs = fig.add_gridspec(3, 4)

		ax1 = fig.add_subplot(gs[0, 0], projection='3d')
		for entry in orientation_data:
			sf_vals = entry["sf_proj"]
			sf_mapped = entry["sf_mapped"]
			proj_sin = entry["sin"]
			proj_cos = entry["cos"]
			if ref_com_sin is not None and ref_com_cos is not None:
				proj_sin = proj_sin - ref_com_sin
				proj_cos = proj_cos - ref_com_cos
			marker_sizes = _marker_sizes_from_sf_categories(sf_mapped, min_size=5.0, max_size=200.0)
			_, unique_idx = np.unique(sf_vals, return_index=True)
			for k in range(len(unique_idx) - 1):
				i = unique_idx[k]
				j = unique_idx[k+1]
				ax1.plot([proj_sin[i], proj_sin[j]],
				         [proj_cos[i], proj_cos[j]],
				         [sf_vals[i], sf_vals[j]],
				         '-', color=ori_colors[entry["ori_idx"]], alpha=0.4, linewidth=1.5)
			ax1.scatter(proj_sin, proj_cos, sf_vals,
			           c=[ori_colors[entry["ori_idx"]]], s=marker_sizes, alpha=0.7,
			           label=f'{entry["ori"]:.0f}°', edgecolors='black', linewidths=0.5)
		ax1.set_xlabel('sin(2θ)', fontsize=10)
		ax1.set_ylabel('cos(2θ)', fontsize=10)
		ax1.set_zlabel('dsPCA SF-axis projection', fontsize=10)
		ax1.set_title('Target Subspace (3D)\n[sin(2θ), cos(2θ), SF projection]', fontsize=11, fontweight='bold')
		ax1.set_zticks(sf_ticks)
		ax1.set_zticklabels(tick_labels)
		ax1.legend(fontsize=7, ncol=2, loc='upper left', bbox_to_anchor=(0, 1))
		ax1.view_init(elev=20, azim=45)
		ax1.set_xlim(-1.6, 1.6)
		ax1.set_ylim(-1.6, 1.6)

		# Subplot 2: same data, rotated view (+90° azimuth)
		ax2 = fig.add_subplot(gs[0, 1], projection='3d')
		for entry in orientation_data:
			sf_vals = entry["sf_proj"]
			sf_mapped = entry["sf_mapped"]
			proj_sin = entry["sin"]
			proj_cos = entry["cos"]
			if ref_com_sin is not None and ref_com_cos is not None:
				proj_sin = proj_sin - ref_com_sin
				proj_cos = proj_cos - ref_com_cos

			marker_sizes = _marker_sizes_from_sf_categories(sf_mapped, min_size=5.0, max_size=200.0)

			_, unique_idx = np.unique(sf_vals, return_index=True)
			for k in range(len(unique_idx) - 1):
				i = unique_idx[k]
				j = unique_idx[k+1]
				ax2.plot([proj_sin[i], proj_sin[j]],
				         [proj_cos[i], proj_cos[j]],
				         [sf_vals[i], sf_vals[j]],
				         '-', color=ori_colors[entry["ori_idx"]], alpha=0.4, linewidth=1.5)

			ax2.scatter(proj_sin, proj_cos, sf_vals,
			           c=[ori_colors[entry["ori_idx"]]], s=marker_sizes, alpha=0.7,
			           label=f'{entry["ori"]:.0f}°', edgecolors='black', linewidths=0.5)

		ax2.set_xlabel('sin(2θ)', fontsize=10)
		ax2.set_ylabel('cos(2θ)', fontsize=10)
		ax2.set_zlabel('dsPCA SF-axis projection', fontsize=10)
		ax2.set_title('Target Subspace (3D, rotated)', fontsize=11, fontweight='bold')
		ax2.set_zticks(sf_ticks)
		ax2.set_zticklabels(tick_labels)
		ax2.legend(fontsize=7, ncol=2, loc='upper left', bbox_to_anchor=(0, 1))
		ax2.view_init(elev=20, azim=135)
		ax2.zaxis._axinfo['juggled'] = (1, 2, 0)
		ax2.set_xlim(-1.6, 1.6)
		ax2.set_ylim(-1.6, 1.6)
		
		# Subplot 3: Mixed subspace (sin, cos, and 1st target-free PC)
		if projection_targetfree.shape[1] >= 1:
			ax3 = fig.add_subplot(gs[0, 2], projection='3d')
			
			for ori_idx, ori in enumerate(unique_orientations):
				mask = (np.abs(ori_from_targets - ori) < 1) & np.isfinite(mapped_sf_targets) & (mapped_sf_targets > 0)
				if np.sum(mask) > 0:
					sf_mapped = np.asarray(mapped_sf_targets[mask], dtype=float)
					sort_idx = np.argsort(sf_mapped)
					proj_sf_sorted = projection_target[mask, 0][sort_idx]
					proj_sin_sorted = projection_target[mask, 1][sort_idx]
					proj_cos_sorted = projection_target[mask, 2][sort_idx]
					proj_tf1_sorted = projection_targetfree[mask, 0][sort_idx]
					if ref_com_sin is not None and ref_com_cos is not None:
						proj_sin_sorted = proj_sin_sorted - ref_com_sin
						proj_cos_sorted = proj_cos_sorted - ref_com_cos
					
					# Plot line segments between consecutive unique SFs
					sf_sorted = sf_mapped[sort_idx]
					_, unique_idx = np.unique(sf_sorted, return_index=True)
					for k in range(len(unique_idx) - 1):
						i = unique_idx[k]
						j = unique_idx[k+1]
						ax3.plot([proj_sin_sorted[i], proj_sin_sorted[j]],
							     [proj_cos_sorted[i], proj_cos_sorted[j]],
							     [proj_tf1_sorted[i], proj_tf1_sorted[j]],
							     '-', color=ori_colors[ori_idx], alpha=0.4, linewidth=1.5)
					
					marker_sizes = _marker_sizes_from_sf_categories(sf_sorted, min_size=5.0, max_size=200.0)
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
			ax3.set_xlim(-1.6, 1.6)
			ax3.set_ylim(-1.6, 1.6)
		
		# Subplot 4: 2D projection onto sin(2θ) vs cos(2θ) orientation plane
		ax4 = fig.add_subplot(gs[0, 3])
		
		for ori_idx, ori in enumerate(unique_orientations):
			mask = np.abs(ori_from_targets - ori) < 1
			if np.sum(mask) > 0:
				proj_sin = projection_target[mask, 1]
				proj_cos = projection_target[mask, 2]
				if ref_com_sin is not None and ref_com_cos is not None:
					proj_sin = proj_sin - ref_com_sin
					proj_cos = proj_cos - ref_com_cos
				sf_log = targets[mask, 0]
				sort_idx = np.argsort(sf_log)
				proj_sin = proj_sin[sort_idx]
				proj_cos = proj_cos[sort_idx]
				sf_log = sf_log[sort_idx]

				# Marker size from log-SF; connect points in SF order
				nonzero = sf_log != 0
				sf_norm = np.zeros_like(sf_log)
				if np.any(nonzero):
					min_sf = sf_log[nonzero].min()
					max_sf = sf_log[nonzero].max()
					sf_norm[nonzero] = (sf_log[nonzero] - min_sf) / (max_sf - min_sf + 1e-10)
				marker_sizes = np.full_like(sf_log, 30.0, dtype=float)
				marker_sizes[nonzero] = 30.0 + 200.0 * (1.0 - sf_norm[nonzero])

				_, unique_idx = np.unique(sf_log, return_index=True)
				for k in range(len(unique_idx) - 1):
					i = unique_idx[k]
					j = unique_idx[k+1]
					ax4.plot([proj_sin[i], proj_sin[j]], [proj_cos[i], proj_cos[j]],
					        '-', color=ori_colors[ori_idx], alpha=0.4, linewidth=1.5)

				ax4.scatter(proj_sin, proj_cos, c=[ori_colors[ori_idx]], 
				       s=marker_sizes, alpha=0.6, label=f'{ori:.0f}°', 
				       edgecolors='black', linewidths=0.5)

		ax4.set_xlim(-1.6, 1.6)
		ax4.set_ylim(-1.6, 1.6)

		# Subplot 5: 2D (x=sin(2θ), y=SF_log)
		ax5 = fig.add_subplot(gs[1, 0])
		for entry in orientation_data:
			sf_vals = entry["sf_proj"]
			proj_sin = entry["sin"]
			ax5.scatter(proj_sin, sf_vals,
			           c=[ori_colors[entry["ori_idx"]]], s=50, alpha=0.7,
			           edgecolors='black', linewidths=0.5, label=f'{entry["ori"]:.0f}°')
			_, unique_idx = np.unique(sf_vals, return_index=True)
			for k in range(len(unique_idx) - 1):
				i = unique_idx[k]
				j = unique_idx[k+1]
				ax5.plot([proj_sin[i], proj_sin[j]], [sf_vals[i], sf_vals[j]],
				        '-', color=ori_colors[entry["ori_idx"]], alpha=0.4, linewidth=1.0)
		ax5.set_xlabel('sin(2θ)', fontsize=10)
		ax5.set_ylabel('dsPCA SF-axis projection', fontsize=10)
		ax5.set_yticks(sf_ticks)
		ax5.set_yticklabels(tick_labels)
		ax5.set_title('2D View: sin vs SF', fontsize=11, fontweight='bold')
		ax5.legend(fontsize=7, ncol=2, loc='upper left', bbox_to_anchor=(0, 1))

		# Subplot 6: 2D (x=cos(2θ), y=SF_log)
		ax6 = fig.add_subplot(gs[1, 1])
		for entry in orientation_data:
			sf_vals = entry["sf_proj"]
			proj_cos = entry["cos"]
			ax6.scatter(proj_cos, sf_vals,
			           c=[ori_colors[entry["ori_idx"]]], s=50, alpha=0.7,
			           edgecolors='black', linewidths=0.5, label=f'{entry["ori"]:.0f}°')
			_, unique_idx = np.unique(sf_vals, return_index=True)
			for k in range(len(unique_idx) - 1):
				i = unique_idx[k]
				j = unique_idx[k+1]
				ax6.plot([proj_cos[i], proj_cos[j]], [sf_vals[i], sf_vals[j]],
				        '-', color=ori_colors[entry["ori_idx"]], alpha=0.4, linewidth=1.0)
		ax6.set_xlabel('cos(2θ)', fontsize=10)
		ax6.set_ylabel('dsPCA SF-axis projection', fontsize=10)
		ax6.set_yticks(sf_ticks)
		ax6.set_yticklabels(tick_labels)
		ax6.set_title('2D View: cos vs SF', fontsize=11, fontweight='bold')
		ax6.legend(fontsize=7, ncol=2, loc='upper left', bbox_to_anchor=(0, 1))

		# Subplots 7–12: per-SF orientation plane (cos vs sin)
		per_sf_axes = [fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3]),
				   fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]),
				   fig.add_subplot(gs[2, 2]), fig.add_subplot(gs[2, 3])]
		for ax_idx, (ax_sf, sf_val) in enumerate(zip(per_sf_axes, mapped_sfs)):
			mask = np.isclose(mapped_sf_targets, float(sf_val), atol=1e-12)
			if np.sum(mask) == 0:
				ax_sf.set_visible(False)
				continue
			proj_sin = projection_target[mask, 1]
			proj_cos = projection_target[mask, 2]
			ori_vals = ori_from_targets[mask]
			# Quantize orientations to bins used in coloring
			ori_bins = np.round(ori_vals / 30.0) * 30.0
			# Map each point to its color
			point_colors = []
			for ob in ori_bins:
				idx = int(np.argmin(np.abs(unique_orientations - ob)))
				point_colors.append(ori_colors[idx])
			# Sort by orientation for connecting lines
			order = np.argsort(ori_bins)
			proj_sin_sorted = proj_sin[order]
			proj_cos_sorted = proj_cos[order]
			ori_bins_sorted = ori_bins[order]
			# Connect consecutive unique orientations, then wrap 0°↔150°
			_, unique_idx = np.unique(ori_bins_sorted, return_index=True)
			for k in range(len(unique_idx) - 1):
				i = unique_idx[k]
				j = unique_idx[k+1]
				ax_sf.plot([proj_sin_sorted[i], proj_sin_sorted[j]],
				          [proj_cos_sorted[i], proj_cos_sorted[j]],
				          '-', color='gray', alpha=0.4, linewidth=1.0)
			# Wrap-around connection (last to first) to connect 0° and 150°
			if len(unique_idx) > 1:
				i = unique_idx[-1]
				j = unique_idx[0]
				ax_sf.plot([proj_sin_sorted[i], proj_sin_sorted[j]],
				          [proj_cos_sorted[i], proj_cos_sorted[j]],
				          '-', color='gray', alpha=0.4, linewidth=1.0)
			# Scatter points colored by orientation
			ax_sf.scatter(proj_sin, proj_cos, c=point_colors, s=50, alpha=0.7,
			            edgecolors='black', linewidths=0.5)
			ax_sf.set_xlabel('sin(2θ)', fontsize=9)
			ax_sf.set_ylabel('cos(2θ)', fontsize=9)
			ax_sf.set_title(f'SF {sf_val:.2f}', fontsize=10)
			ax_sf.grid(alpha=0.2)
		
		# Draw circle to show circular orientation geometry
		circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', color='gray', alpha=0.2, linewidth=1)
		ax4.add_patch(circle)
		
		ax4.set_xlabel('sin(2θ)', fontsize=10)
		ax4.set_ylabel('cos(2θ)', fontsize=10)
		ax4.set_title('Orientation Plane (2D)\n[sin(2θ) vs cos(2θ)]', fontsize=11, fontweight='bold')
		ax4.legend(fontsize=7, ncol=2, loc='best')
		ax4.set_aspect('equal')
		ax4.grid(alpha=0.3)
		
		fig.suptitle(f'{GROUP_LABELS[group_name]}: dsPCA Target Subspace (Circular Basis)', fontsize=14, fontweight='bold', y=1.02)
		plt.tight_layout()
		plt.savefig(os.path.join(GROUPED_PLOTS_DIR, f'pseudopop_dspca_target_subspace_3d_{group_name}{out_tag}.svg'), 
		           format='svg', bbox_inches='tight')
		plt.close()
		print(f"  [saved] pseudopop_dspca_target_subspace_3d_{group_name}{out_tag}.svg")
		
		# Target-free subspace visualization (top 3 dims if available)
		if projection_targetfree.shape[1] >= 3:
			fig = plt.figure(figsize=(18, 5))
			
			# Subplot 1: Colored by orientation
			ax1_tf = fig.add_subplot(131, projection='3d')
			
			for ori_idx, ori in enumerate(unique_orientations):
				mask = np.abs(ori_from_targets - ori) < 1
				if np.sum(mask) > 0:
					# Sort by SF to ensure proper line order (no wrap-around)
					sf_vals = targets[mask, 2]
					sort_idx = np.argsort(sf_vals)
					masked_targetfree = projection_targetfree[mask]
					# Plot line segments (no wrap-around) using unique SF order
					sorted_targetfree = masked_targetfree[sort_idx]
					sf_sorted = sf_vals[sort_idx]
					_, unique_idx = np.unique(sf_sorted, return_index=True)
					for k in range(len(unique_idx) - 1):
						i = unique_idx[k]
						j = unique_idx[k+1]
						ax1_tf.plot([sorted_targetfree[i, 0], sorted_targetfree[j, 0]],
						         [sorted_targetfree[i, 1], sorted_targetfree[j, 1]],
						         [sorted_targetfree[i, 2], sorted_targetfree[j, 2]],
						         '-', color=ori_colors[ori_idx], alpha=0.4, linewidth=1.5)
					# Then scatter on top
					ax1_tf.scatter(sorted_targetfree[:, 0], sorted_targetfree[:, 1], 
					            sorted_targetfree[:, 2], c=[ori_colors[ori_idx]], 
					            s=50, alpha=0.7, edgecolors='black', linewidths=0.5, label=f'{ori:.0f}°')
			
			tf_var_pcts = [100*v/total_var for v in targetfree_var[:3]]
			ax1_tf.set_xlabel(f'TF-PC1 ({tf_var_pcts[0]:.1f}%)', fontsize=10)
			ax1_tf.set_ylabel(f'TF-PC2 ({tf_var_pcts[1]:.1f}%)', fontsize=10)
			ax1_tf.set_zlabel(f'TF-PC3 ({tf_var_pcts[2]:.1f}%)', fontsize=10)
			ax1_tf.set_title('Target-Free Subspace\n(Orientation-coded)', fontsize=11, fontweight='bold')
			ax1_tf.legend(fontsize=8, ncol=2, loc='upper left', bbox_to_anchor=(0, 1))
			ax1_tf.view_init(elev=20, azim=45)
			
			# Subplot 2: Colored by SF
			ax2_tf = fig.add_subplot(132, projection='3d')
			
			# Build SF color map excluding SF==0 (using mapped linear SFs)
			nonzero_sfs = [sf for sf in mapped_sfs if sf > 0]
			sf_color_map = {sf: plt.cm.viridis(i / max(len(nonzero_sfs), 1)) for i, sf in enumerate(nonzero_sfs)}
			
			for sf in nonzero_sfs:
				mask = np.isclose(mapped_sf_targets, float(sf), atol=1e-12)
				if np.sum(mask) > 0:
					ax2_tf.scatter(projection_targetfree[mask, 0], projection_targetfree[mask, 1], 
					            projection_targetfree[mask, 2], c=[sf_color_map[sf]], 
					            s=50, alpha=0.7, edgecolors='black', linewidths=0.5, label=f'SF {sf:.2f}')
			
			ax2_tf.set_xlabel(f'TF-PC1 ({tf_var_pcts[0]:.1f}%)', fontsize=10)
			ax2_tf.set_ylabel(f'TF-PC2 ({tf_var_pcts[1]:.1f}%)', fontsize=10)
			ax2_tf.set_zlabel(f'TF-PC3 ({tf_var_pcts[2]:.1f}%)', fontsize=10)
			ax2_tf.set_title('Target-Free Subspace\n(SF-coded)', fontsize=11, fontweight='bold')
			ax2_tf.legend(fontsize=8, ncol=1, loc='upper left', bbox_to_anchor=(0, 1))
			ax2_tf.view_init(elev=20, azim=45)
			
			# Subplot 3: Orientation-coded, marker size reflects SF (higher SF = smaller marker)
			ax3_tf = fig.add_subplot(133, projection='3d')
			for ori_idx, ori in enumerate(unique_orientations):
				mask = (np.abs(ori_from_targets - ori) < 1) & np.isfinite(mapped_sf_targets) & (mapped_sf_targets > 0)
				if np.sum(mask) > 0:
					sf_mapped = np.asarray(mapped_sf_targets[mask], dtype=float)
					masked_targetfree = np.asarray(projection_targetfree[mask], dtype=float)
					sort_idx = np.argsort(sf_mapped)
					sorted_targetfree = masked_targetfree[sort_idx]
					sf_sorted = sf_mapped[sort_idx]
					marker_sizes = _marker_sizes_from_sf_categories(sf_sorted, min_size=5.0, max_size=200.0)
					_, unique_idx = np.unique(sf_sorted, return_index=True)
					for k in range(len(unique_idx) - 1):
						i = unique_idx[k]
						j = unique_idx[k + 1]
						ax3_tf.plot([sorted_targetfree[i, 0], sorted_targetfree[j, 0]],
						           [sorted_targetfree[i, 1], sorted_targetfree[j, 1]],
						           [sorted_targetfree[i, 2], sorted_targetfree[j, 2]],
						           '-', color=ori_colors[ori_idx], alpha=0.4, linewidth=1.5)
					ax3_tf.scatter(sorted_targetfree[:, 0], sorted_targetfree[:, 1],
					            sorted_targetfree[:, 2], c=[ori_colors[ori_idx]],
					            s=marker_sizes, alpha=0.7, edgecolors='black', linewidths=0.5,
					            label=f'{ori:.0f}°')

			ax3_tf.set_xlabel(f'TF-PC1 ({tf_var_pcts[0]:.1f}%)', fontsize=10)
			ax3_tf.set_ylabel(f'TF-PC2 ({tf_var_pcts[1]:.1f}%)', fontsize=10)
			ax3_tf.set_zlabel(f'TF-PC3 ({tf_var_pcts[2]:.1f}%)', fontsize=10)
			ax3_tf.set_title('Target-Free Subspace\n(color=orientation, size=SF)', fontsize=11, fontweight='bold')
			ax3_tf.legend(fontsize=8, ncol=2, loc='upper left', bbox_to_anchor=(0, 1))
			ax3_tf.view_init(elev=20, azim=45)
			
			fig.suptitle(f'{GROUP_LABELS[group_name]}: dsPCA Target-Free Subspace (Circular Basis)', fontsize=14, fontweight='bold', y=1.02)
			plt.tight_layout()
			plt.savefig(os.path.join(GROUPED_PLOTS_DIR, f'pseudopop_dspca_targetfree_subspace_3d_{group_name}{out_tag}.svg'), 
			           format='svg', bbox_inches='tight')
			plt.close()
			print(f"  [saved] pseudopop_dspca_targetfree_subspace_3d_{group_name}{out_tag}.svg")


def _pseudopopulation_dspca_make_superimposed_and_distance_plots(
	*,
	results: Dict[str, Dict[str, object]],
	mapped_sfs_linear: List[float],
	out_tag: str,
) -> None:
	"""Generate per-SF superimposed orientation-plane figures and distance analyses.

	This is shared by standard and CV dsPCA runs.
	"""
	# ========== Combined Before/After per-SF orientation-plane figures (WT, RESCUED) ==========
	group_sets = {
		"WT": ("WT_BEFORE", "WT_AFTER"),
		"RESCUED": ("RESCUED_BEFORE", "RESCUED_AFTER"),
	}
	for grp_label, (gn_before, gn_after) in group_sets.items():
		if gn_before not in results or gn_after not in results:
			continue
		before_targets = np.asarray(results[gn_before]['targets'], dtype=float)
		after_targets = np.asarray(results[gn_after]['targets'], dtype=float)
		before_mapped_sf = results[gn_before].get('mapped_sf_targets', None)
		after_mapped_sf = results[gn_after].get('mapped_sf_targets', None)
		before_proj = np.asarray(results[gn_before]['projection_target'], dtype=float)
		after_proj = np.asarray(results[gn_after]['projection_target'], dtype=float)
		if before_mapped_sf is None:
			before_mapped_sf = np.where(np.abs(before_targets[:, 0]) > 0, 2 ** before_targets[:, 0], 0.0)
		if after_mapped_sf is None:
			after_mapped_sf = np.where(np.abs(after_targets[:, 0]) > 0, 2 ** after_targets[:, 0], 0.0)
		before_mapped_sf = np.asarray(before_mapped_sf, dtype=float)
		after_mapped_sf = np.asarray(after_mapped_sf, dtype=float)
		ori_before = np.mod(np.rad2deg(np.arctan2(before_targets[:, 1], before_targets[:, 2]) / 2), 180)
		ori_after = np.mod(np.rad2deg(np.arctan2(after_targets[:, 1], after_targets[:, 2]) / 2), 180)
		bins_before = np.unique(np.round(ori_before / 30.0) * 30.0)
		bins_after = np.unique(np.round(ori_after / 30.0) * 30.0)
		all_bins = np.unique(np.concatenate([bins_before, bins_after]))
		palette = plt.cm.turbo(np.linspace(0, 1, len(all_bins) + 1)[:-1])
		
		fig, axes = plt.subplots(2, 3, figsize=(15, 10))
		axes = axes.flatten()
		for i, sf_val in enumerate(mapped_sfs_linear):
			ax = axes[i]
			mask_b = np.isclose(before_mapped_sf, float(sf_val), atol=1e-12)
			mask_a = np.isclose(after_mapped_sf, float(sf_val), atol=1e-12)
			if not (np.any(mask_b) or np.any(mask_a)):
				ax.set_visible(False)
				continue
			# Before points
			sin_b = before_proj[mask_b, 1]
			cos_b = before_proj[mask_b, 2]
			ob = ori_before[mask_b]
			ob_bins = np.round(ob / 30.0) * 30.0
			colors_b = [palette[int(np.argmin(np.abs(all_bins - x)))] for x in ob_bins]
			# After points
			sin_a = after_proj[mask_a, 1]
			cos_a = after_proj[mask_a, 2]
			oa = ori_after[mask_a]
			oa_bins = np.round(oa / 30.0) * 30.0
			colors_a = [palette[int(np.argmin(np.abs(all_bins - x)))] for x in oa_bins]
			# Align centers of mass: shift both to put COM at origin; After additionally matched to Before COM
			com_b = np.array([np.nanmean(sin_b) if sin_b.size else 0.0, np.nanmean(cos_b) if cos_b.size else 0.0])
			com_a_raw = np.array([np.nanmean(sin_a) if sin_a.size else 0.0, np.nanmean(cos_a) if cos_a.size else 0.0])
			# First shift After to Before COM, then shift both so COM is at origin
			sin_a = sin_a - (com_a_raw[0] - com_b[0])
			cos_a = cos_a - (com_a_raw[1] - com_b[1])
			sin_b = sin_b - com_b[0]
			cos_b = cos_b - com_b[1]
			sin_a = sin_a - com_b[0]
			cos_a = cos_a - com_b[1]
			# Scatter
			ax.scatter(sin_b, cos_b, c=colors_b, s=50, alpha=0.7,
			          edgecolors='black', linewidths=0.5, label=f'{grp_label} Before')
			ax.scatter(sin_a, cos_a, c=colors_a, s=60, alpha=0.7,
			          edgecolors='black', linewidths=0.8, label=f'{grp_label} After')
			# Connect per-orientation order (sorted by orientation bins)
			order_b = np.argsort(ob_bins)
			order_a = np.argsort(oa_bins)
			sin_b_sorted = sin_b[order_b]
			cos_b_sorted = cos_b[order_b]
			sin_a_sorted = sin_a[order_a]
			cos_a_sorted = cos_a[order_a]
			# Wrap-aware connections
			wrap_b = np.append(order_b, order_b[0]) if len(order_b) > 0 else order_b
			wrap_a = np.append(order_a, order_a[0]) if len(order_a) > 0 else order_a
			for idx in range(len(wrap_b) - 1):
				i0, i1 = wrap_b[idx], wrap_b[idx + 1]
				ax.plot([sin_b[i0], sin_b[i1]], [cos_b[i0], cos_b[i1]], '-', color='black', alpha=0.5, linewidth=1.2)
			for idx in range(len(wrap_a) - 1):
				i0, i1 = wrap_a[idx], wrap_a[idx + 1]
				ax.plot([sin_a[i0], sin_a[i1]], [cos_a[i0], cos_a[i1]], '-', color='red', alpha=0.5, linewidth=1.2)
			ax.set_xlabel('sin(2θ)')
			ax.set_ylabel('cos(2θ)')
			ax.set_title(f'SF {sf_val:.2f}')
			ax.set_aspect('equal')
			ax.set_xlim([-1.5, 1.5])
			ax.set_ylim([-1.5, 1.5])
			ax.axhline(0, color='gray', alpha=0.2, linewidth=0.8)
			ax.axvline(0, color='gray', alpha=0.2, linewidth=0.8)
			ax.grid(alpha=0.3)
		# One legend for the whole figure
		handles = [
			plt.Line2D([0], [0], color='black', lw=1.5, label=f'{grp_label} Before'),
			plt.Line2D([0], [0], color='red', lw=1.5, label=f'{grp_label} After'),
		]
		fig.legend(handles=handles, loc='upper right')
		fig.suptitle(f'{grp_label}: Before vs After (COM at origin) per SF', fontsize=14, fontweight='bold')
		plt.tight_layout()
		out_name = f'pseudopop_dspca_orient_plane_by_sf_superimposed_{grp_label}{out_tag}.svg'
		plt.savefig(os.path.join(GROUPED_PLOTS_DIR, out_name), format='svg', bbox_inches='tight')
		plt.close(fig)
		print(f"  [saved] {out_name}")

		# ========== Distance analyses per SF (distance to COM; adjacent orientation distance) ==========
		fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
		axes1 = axes1.flatten()
		for i, sf_val in enumerate(mapped_sfs_linear):
			ax = axes1[i]
			mask_b = np.isclose(before_mapped_sf, float(sf_val), atol=1e-12)
			mask_a = np.isclose(after_mapped_sf, float(sf_val), atol=1e-12)
			if not (np.any(mask_b) or np.any(mask_a)):
				ax.set_visible(False)
				continue
			sin_b = before_proj[mask_b, 1]
			cos_b = before_proj[mask_b, 2]
			sin_a = after_proj[mask_a, 1]
			cos_a = after_proj[mask_a, 2]
			ob = ori_before[mask_b]
			ob_bins = np.round(ob / 30.0) * 30.0
			oa = ori_after[mask_a]
			oa_bins = np.round(oa / 30.0) * 30.0
			# COMs
			com_b = np.array([np.nanmean(sin_b) if sin_b.size else 0.0, np.nanmean(cos_b) if cos_b.size else 0.0])
			com_a = np.array([np.nanmean(sin_a) if sin_a.size else 0.0, np.nanmean(cos_a) if cos_a.size else 0.0])
			dist_b = np.sqrt((sin_b - com_b[0]) ** 2 + (cos_b - com_b[1]) ** 2) if sin_b.size else np.array([])
			dist_a = np.sqrt((sin_a - com_a[0]) ** 2 + (cos_a - com_a[1]) ** 2) if sin_a.size else np.array([])
			# Sort by orientation bins for lines
			order_b = np.argsort(ob_bins)
			order_a = np.argsort(oa_bins)
			if dist_b.size:
				ax.plot(ob_bins[order_b], dist_b[order_b], '-o', color='black', label='Before')
			if dist_a.size:
				ax.plot(oa_bins[order_a], dist_a[order_a], '-o', color='red', label='After')
			ax.set_xlabel('Orientation (deg)')
			ax.set_ylabel('Distance to COM')
			ax.set_title(f'SF {sf_val:.2f}')
			ax.grid(alpha=0.3)
		fig1.suptitle(f'{grp_label}: Distance to COM vs Orientation', fontsize=14, fontweight='bold')
		fig1.legend(loc='upper right')
		plt.tight_layout()
		out1 = f'pseudopop_dspca_distance_to_com_by_sf_{grp_label}{out_tag}.svg'
		plt.savefig(os.path.join(GROUPED_PLOTS_DIR, out1), format='svg', bbox_inches='tight')
		plt.close(fig1)
		print(f"  [saved] {out1}")

		fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
		axes2 = axes2.flatten()
		for i, sf_val in enumerate(mapped_sfs_linear):
			ax = axes2[i]
			mask_b = np.isclose(before_mapped_sf, float(sf_val), atol=1e-12)
			mask_a = np.isclose(after_mapped_sf, float(sf_val), atol=1e-12)
			if not (np.any(mask_b) or np.any(mask_a)):
				ax.set_visible(False)
				continue
			sin_b = before_proj[mask_b, 1]
			cos_b = before_proj[mask_b, 2]
			sin_a = after_proj[mask_a, 1]
			cos_a = after_proj[mask_a, 2]
			ob = ori_before[mask_b]
			ob_bins = np.round(ob / 30.0) * 30.0
			oa = ori_after[mask_a]
			oa_bins = np.round(oa / 30.0) * 30.0
			# Sort and wrap
			order_b = np.argsort(ob_bins)
			order_a = np.argsort(oa_bins)
			ob_bins_sorted = ob_bins[order_b]
			oa_bins_sorted = oa_bins[order_a]
			sin_b_sorted = sin_b[order_b]
			cos_b_sorted = cos_b[order_b]
			sin_a_sorted = sin_a[order_a]
			cos_a_sorted = cos_a[order_a]
			# Append wrap to close 0–150
			if len(ob_bins_sorted) > 1:
				ob_bins_wrap = np.append(ob_bins_sorted, ob_bins_sorted[0] + 180)
				sin_b_wrap = np.append(sin_b_sorted, sin_b_sorted[0])
				cos_b_wrap = np.append(cos_b_sorted, cos_b_sorted[0])
				dists_b = np.sqrt(np.diff(sin_b_wrap) ** 2 + np.diff(cos_b_wrap) ** 2)
				ax.plot(np.arange(len(dists_b)), dists_b, '-o', color='black', label='Before')
			if len(oa_bins_sorted) > 1:
				oa_bins_wrap = np.append(oa_bins_sorted, oa_bins_sorted[0] + 180)
				sin_a_wrap = np.append(sin_a_sorted, sin_a_sorted[0])
				cos_a_wrap = np.append(cos_a_sorted, cos_a_sorted[0])
				dists_a = np.sqrt(np.diff(sin_a_wrap) ** 2 + np.diff(cos_a_wrap) ** 2)
				ax.plot(np.arange(len(dists_a)), dists_a, '-o', color='red', label='After')
			ax.set_xlabel('Adjacent orientation index')
			ax.set_ylabel('Distance between adjacent orientations')
			ax.set_title(f'SF {sf_val:.2f}')
			ax.grid(alpha=0.3)
		fig2.suptitle(f'{grp_label}: Adjacent Orientation Distance', fontsize=14, fontweight='bold')
		fig2.legend(loc='upper right')
		plt.tight_layout()
		out2 = f'pseudopop_dspca_adjacent_orientation_distance_by_sf_{grp_label}{out_tag}.svg'
		plt.savefig(os.path.join(GROUPED_PLOTS_DIR, out2), format='svg', bbox_inches='tight')
		plt.close(fig2)
		print(f"  [saved] {out2}")

	# Additional superimposed figure: WT Before (blue) + Rescued Before (black) + Rescued After (red)
	if ("WT_BEFORE" in results) and ("RESCUED_BEFORE" in results) and ("RESCUED_AFTER" in results):
		wt_targets = np.asarray(results["WT_BEFORE"]['targets'], dtype=float)
		wt_proj = np.asarray(results["WT_BEFORE"]['projection_target'], dtype=float)
		wt_msf = results["WT_BEFORE"].get('mapped_sf_targets', None)
		rb_targets = np.asarray(results["RESCUED_BEFORE"]['targets'], dtype=float)
		rb_proj = np.asarray(results["RESCUED_BEFORE"]['projection_target'], dtype=float)
		rb_msf = results["RESCUED_BEFORE"].get('mapped_sf_targets', None)
		ra_targets = np.asarray(results["RESCUED_AFTER"]['targets'], dtype=float)
		ra_proj = np.asarray(results["RESCUED_AFTER"]['projection_target'], dtype=float)
		ra_msf = results["RESCUED_AFTER"].get('mapped_sf_targets', None)
		if wt_msf is None:
			wt_msf = np.where(np.abs(wt_targets[:, 0]) > 0, 2 ** wt_targets[:, 0], 0.0)
		if rb_msf is None:
			rb_msf = np.where(np.abs(rb_targets[:, 0]) > 0, 2 ** rb_targets[:, 0], 0.0)
		if ra_msf is None:
			ra_msf = np.where(np.abs(ra_targets[:, 0]) > 0, 2 ** ra_targets[:, 0], 0.0)
		wt_msf = np.asarray(wt_msf, dtype=float)
		rb_msf = np.asarray(rb_msf, dtype=float)
		ra_msf = np.asarray(ra_msf, dtype=float)

		wt_ori = np.mod(np.rad2deg(np.arctan2(wt_targets[:, 1], wt_targets[:, 2]) / 2), 180)
		rb_ori = np.mod(np.rad2deg(np.arctan2(rb_targets[:, 1], rb_targets[:, 2]) / 2), 180)
		ra_ori = np.mod(np.rad2deg(np.arctan2(ra_targets[:, 1], ra_targets[:, 2]) / 2), 180)
		bins_wt = np.unique(np.round(wt_ori / 30.0) * 30.0)
		bins_rb = np.unique(np.round(rb_ori / 30.0) * 30.0)
		bins_ra = np.unique(np.round(ra_ori / 30.0) * 30.0)
		all_bins = np.unique(np.concatenate([bins_wt, bins_rb, bins_ra]))
		palette = plt.cm.turbo(np.linspace(0, 1, len(all_bins) + 1)[:-1])

		fig3, axes3 = plt.subplots(2, 3, figsize=(15, 10))
		axes3 = axes3.flatten()
		panel_sfs = list(mapped_sfs_linear[:len(axes3)])
		for i, ax in enumerate(axes3):
			if i >= len(panel_sfs):
				ax.set_visible(False)
				continue
			sf_val = float(panel_sfs[i])
			mask_wt = np.isclose(wt_msf, sf_val, atol=1e-12)
			mask_rb = np.isclose(rb_msf, sf_val, atol=1e-12)
			mask_ra = np.isclose(ra_msf, sf_val, atol=1e-12)
			if not (np.any(mask_wt) or np.any(mask_rb) or np.any(mask_ra)):
				ax.set_visible(False)
				continue

			sin_wt = wt_proj[mask_wt, 1]
			cos_wt = wt_proj[mask_wt, 2]
			ori_wt_sf = wt_ori[mask_wt]
			bins_wt_sf = np.round(ori_wt_sf / 30.0) * 30.0
			colors_wt = [palette[int(np.argmin(np.abs(all_bins - x)))] for x in bins_wt_sf]

			sin_rb = rb_proj[mask_rb, 1]
			cos_rb = rb_proj[mask_rb, 2]
			ori_rb_sf = rb_ori[mask_rb]
			bins_rb_sf = np.round(ori_rb_sf / 30.0) * 30.0
			colors_rb = [palette[int(np.argmin(np.abs(all_bins - x)))] for x in bins_rb_sf]

			sin_ra = ra_proj[mask_ra, 1]
			cos_ra = ra_proj[mask_ra, 2]
			ori_ra_sf = ra_ori[mask_ra]
			bins_ra_sf = np.round(ori_ra_sf / 30.0) * 30.0
			colors_ra = [palette[int(np.argmin(np.abs(all_bins - x)))] for x in bins_ra_sf]

			# COM-align each condition set (plot-only): shift each group's COM to origin
			com_rb = np.array([np.nanmean(sin_rb) if sin_rb.size else 0.0, np.nanmean(cos_rb) if cos_rb.size else 0.0])
			com_ra = np.array([np.nanmean(sin_ra) if sin_ra.size else 0.0, np.nanmean(cos_ra) if cos_ra.size else 0.0])
			com_wt = np.array([np.nanmean(sin_wt) if sin_wt.size else 0.0, np.nanmean(cos_wt) if cos_wt.size else 0.0])
			sin_rb = sin_rb - com_rb[0]
			cos_rb = cos_rb - com_rb[1]
			sin_ra = sin_ra - com_ra[0]
			cos_ra = cos_ra - com_ra[1]
			sin_wt = sin_wt - com_wt[0]
			cos_wt = cos_wt - com_wt[1]

			# Scatter
			if sin_wt.size:
				ax.scatter(sin_wt, cos_wt, c=colors_wt, s=40, alpha=0.55, edgecolors='black', linewidths=0.5)
			if sin_rb.size:
				ax.scatter(sin_rb, cos_rb, c=colors_rb, s=50, alpha=0.70, edgecolors='black', linewidths=0.5)
			if sin_ra.size:
				ax.scatter(sin_ra, cos_ra, c=colors_ra, s=60, alpha=0.70, edgecolors='black', linewidths=0.8)

			# Lines (wrap-aware)
			if bins_wt_sf.size:
				order = np.argsort(bins_wt_sf)
				wrap = np.append(order, order[0]) if len(order) > 0 else order
				for idx in range(len(wrap) - 1):
					i0, i1 = wrap[idx], wrap[idx + 1]
					ax.plot([sin_wt[i0], sin_wt[i1]], [cos_wt[i0], cos_wt[i1]], '-', color='blue', alpha=0.55, linewidth=1.2)
			if bins_rb_sf.size:
				order = np.argsort(bins_rb_sf)
				wrap = np.append(order, order[0]) if len(order) > 0 else order
				for idx in range(len(wrap) - 1):
					i0, i1 = wrap[idx], wrap[idx + 1]
					ax.plot([sin_rb[i0], sin_rb[i1]], [cos_rb[i0], cos_rb[i1]], '-', color='black', alpha=0.55, linewidth=1.2)
			if bins_ra_sf.size:
				order = np.argsort(bins_ra_sf)
				wrap = np.append(order, order[0]) if len(order) > 0 else order
				for idx in range(len(wrap) - 1):
					i0, i1 = wrap[idx], wrap[idx + 1]
					ax.plot([sin_ra[i0], sin_ra[i1]], [cos_ra[i0], cos_ra[i1]], '-', color='red', alpha=0.55, linewidth=1.2)

			ax.set_xlabel('sin(2θ)')
			ax.set_ylabel('cos(2θ)')
			ax.set_title(f'SF {sf_val:.2f}')
			ax.set_aspect('equal')
			ax.set_xlim([-1.5, 1.5])
			ax.set_ylim([-1.5, 1.5])
			ax.axhline(0, color='gray', alpha=0.2, linewidth=0.8)
			ax.axvline(0, color='gray', alpha=0.2, linewidth=0.8)
			ax.grid(alpha=0.3)

		handles = [
			plt.Line2D([0], [0], color='black', lw=1.5, label='Rescued Before'),
			plt.Line2D([0], [0], color='red', lw=1.5, label='Rescued After'),
			plt.Line2D([0], [0], color='blue', lw=1.5, label='WT Before'),
		]
		fig3.legend(handles=handles, loc='upper right')
		fig3.suptitle('Superimposed: WT Before vs Rescued Before/After (COM aligned)', fontsize=14, fontweight='bold')
		plt.tight_layout()
		out_extra = f'pseudopop_dspca_orient_plane_by_sf_superimposed_RESCUED_with_WT_BEFORE{out_tag}.svg'
		plt.savefig(os.path.join(GROUPED_PLOTS_DIR, out_extra), format='svg', bbox_inches='tight')
		plt.close(fig3)
		print(f"  [saved] {out_extra}")



# ---------------------------------------------------------------------------
# Session loading
# ---------------------------------------------------------------------------
def _responsive_fraction_by_orientation_cache_path(session_rel: str) -> str:
	return os.path.join(RESULTS_ROOT, session_rel, 'responsive_fraction_by_orientation_excluding_064.npz')


@lru_cache(maxsize=None)
def _load_or_compute_responsive_fraction_by_orientation_excluding_064(session_rel: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
	"""Return per-orientation responsive fractions excluding mapped SF=0.64 trials.

	Fractions are computed per session by testing each neuron's response on all
	trials of a given orientation against the shared SF=0 baseline, while dropping
	the highest mapped spatial-frequency condition (0.64).
	"""
	cache_path = _responsive_fraction_by_orientation_cache_path(session_rel)
	if os.path.exists(cache_path):
		try:
			with np.load(cache_path) as cached:
				orientations = np.asarray(cached['orientations'], dtype=float)
				fractions = np.asarray(cached['responsive_fraction_by_orientation'], dtype=float)
			if orientations.ndim == 1 and fractions.ndim == 1 and orientations.size == fractions.size:
				return orientations, fractions
		except Exception:
			pass

	data_path = os.path.join(IMAGE_ROOT, session_rel)
	suite2p_path = os.path.join(PROC_IMAGE_ROOT, session_rel, 'suite2p', 'plane0')
	json_files = glob.glob(os.path.join(data_path, '*.json'))
	if not json_files:
		print(f"[warn] No JSON file found for {session_rel}; skipping responsive fraction by orientation")
		return None

	try:
		with open(json_files[0], 'r') as f:
			json_data = [json.loads(line) for line in f][-1]
		scanimage_frame_path = glob.glob(os.path.join(data_path, '*Scanimage_Arm1_Frame*.csv'))[0]
		stimulus_time_path = glob.glob(os.path.join(data_path, '*DataUnity_TrialStart*.csv'))[0]
		unity_frame_path = glob.glob(os.path.join(data_path, '*DataUnity_Frame*.csv'))[0]
		iscell = np.load(os.path.join(suite2p_path, 'iscell.npy'))
		spks = np.load(os.path.join(suite2p_path, 'spks.npy'))
	except Exception as exc:
		print(f"[warn] Failed loading raw session inputs for {session_rel}: {exc}")
		return None

	scanimage_frame_time_all = np.genfromtxt(scanimage_frame_path, delimiter=',', comments='//', usecols=0)
	scanimage_frame_boolean = np.genfromtxt(scanimage_frame_path, delimiter=',', comments='//', usecols=1, dtype=str)
	scanimage_frame_onset_bool = np.zeros_like(scanimage_frame_boolean, dtype=int)
	scanimage_frame_onset_bool[1:] = ((scanimage_frame_boolean[1:] == 'True') & (scanimage_frame_boolean[:-1] == 'False')).astype(int)
	scanimage_frame_onset = scanimage_frame_time_all[scanimage_frame_onset_bool == 1]

	stimulus_time_all = np.genfromtxt(stimulus_time_path, delimiter=',', comments='//', usecols=0)
	stimulus_time_bool = np.genfromtxt(stimulus_time_path, delimiter=',', comments='//', usecols=1, dtype=str)
	stimulus_onset_bool = np.zeros_like(stimulus_time_bool, dtype=int)
	stimulus_onset_bool[1:] = ((stimulus_time_bool[1:] == 'True') & (stimulus_time_bool[:-1] == 'False')).astype(int)
	stimulus_onset = stimulus_time_all[stimulus_onset_bool == 1]

	unity_frame_time_all = np.genfromtxt(unity_frame_path, delimiter=',', comments='//', usecols=0)
	unity_frame_boolean = np.genfromtxt(unity_frame_path, delimiter=',', comments='//', usecols=1, dtype=str)
	unity_frame_onset_bool = np.zeros_like(unity_frame_boolean, dtype=int)
	unity_frame_onset_bool[1:] = ((unity_frame_boolean[1:] == 'True') & (unity_frame_boolean[:-1] == 'False')).astype(int)
	unity_frame_onset = unity_frame_time_all[unity_frame_onset_bool == 1]
	if stimulus_onset.size == 0 and unity_frame_onset.size > 0:
		stimulus_onset = unity_frame_onset.copy()
		while stimulus_onset.size >= 2 and np.diff(stimulus_onset)[-1] > 500:
			stimulus_onset = stimulus_onset[:-1]

	spks_cell = np.asarray(spks[iscell[:, 0] == 1], dtype=float)
	spks_cell_zscored = zscore(spks_cell, axis=1)
	has_nan = np.isnan(spks_cell_zscored).any(axis=1)
	neuron_variance = np.var(spks_cell, axis=1)
	has_zero_var = neuron_variance < 1e-10
	valid_neurons = ~(has_nan | has_zero_var)
	spks_cell_zscored = spks_cell_zscored[valid_neurons]
	if spks_cell_zscored.size == 0:
		return None

	mean_dt = float(np.mean(np.diff(scanimage_frame_onset)))
	negative_window = int(round(50 / mean_dt))
	positive_window = int(round(250 / mean_dt))
	window_length = negative_window + positive_window + 1
	n_frames = int(spks_cell_zscored.shape[1])

	trial_slices: List[np.ndarray] = []
	kept_trial_indices: List[int] = []
	for trial_i, onset in enumerate(stimulus_onset):
		trial_idx = int(np.argmin(np.abs(scanimage_frame_onset - onset)))
		start_idx = trial_idx - negative_window
		end_idx = trial_idx + positive_window + 1
		if start_idx < 0 or end_idx > n_frames:
			continue
		trial_slices.append(spks_cell_zscored[:, start_idx:end_idx])
		kept_trial_indices.append(int(trial_i))
	if not trial_slices:
		return None

	kept_trial_indices_arr = np.asarray(kept_trial_indices, dtype=int)
	spks_tensor = np.stack(trial_slices, axis=-1)
	time_axis = (np.arange(window_length) - negative_window) * mean_dt
	response_idx = np.where((time_axis >= 0) & (time_axis <= 250))[0]
	response_per_trial = np.nanmean(spks_tensor[:, response_idx, :], axis=1)

	orientation_history = np.asarray(json_data['orientation_history'], dtype=float)
	spatialfreq_history = np.asarray(json_data['spatialfreq_history'], dtype=float)
	unique_orientations = np.unique(orientation_history)
	unique_spatialfreqs = np.unique(spatialfreq_history)
	if unique_orientations.size == 0 or unique_spatialfreqs.size == 0:
		return None

	index_map = {orig_idx: kept_idx for kept_idx, orig_idx in enumerate(kept_trial_indices_arr)}

	def map_to_kept(idxs: np.ndarray) -> np.ndarray:
		return np.asarray([index_map[i] for i in np.asarray(idxs, dtype=int) if int(i) in index_map], dtype=int)

	baseline_trial_inds = map_to_kept(np.where(spatialfreq_history == unique_spatialfreqs[0])[0])
	if baseline_trial_inds.size == 0:
		return unique_orientations.astype(float), np.full(unique_orientations.shape, np.nan, dtype=float)

	baseline_activity = response_per_trial[:, baseline_trial_inds]
	baseline_mean = np.nanmean(baseline_activity, axis=1)
	mapped_sf_history = np.asarray([convert_sf_value(float(sf)) for sf in spatialfreq_history], dtype=float)
	fractions = np.full(unique_orientations.shape, np.nan, dtype=float)
	n_neurons = int(response_per_trial.shape[0])

	for ori_idx, ori in enumerate(unique_orientations):
		trial_inds_all = np.where(
			(orientation_history == ori)
			& (mapped_sf_history > 0)
			& (~np.isclose(mapped_sf_history, float(RESPONSIVE_ORIENTATION_EXCLUDED_MAPPED_SF), atol=1e-6, rtol=1e-6))
		)[0]
		trial_inds = map_to_kept(trial_inds_all)
		if trial_inds.size == 0:
			continue

		response_activity = response_per_trial[:, trial_inds]
		response_mean = np.nanmean(response_activity, axis=1)
		p_values = np.ones(n_neurons, dtype=float)
		for neuron_idx in range(n_neurons):
			base_vals = baseline_activity[neuron_idx]
			resp_vals = response_activity[neuron_idx]
			base_vals = base_vals[np.isfinite(base_vals)]
			resp_vals = resp_vals[np.isfinite(resp_vals)]
			n_pair = min(base_vals.size, resp_vals.size)
			if n_pair < 2:
				continue
			try:
				_, p_val = ttest_rel(resp_vals[:n_pair], base_vals[:n_pair])
			except Exception:
				p_val = np.nan
			p_values[neuron_idx] = p_val if np.isfinite(p_val) else 1.0

		responsive_mask = (p_values < 0.05) & (response_mean > baseline_mean)
		fractions[ori_idx] = float(np.sum(responsive_mask) / n_neurons)

	os.makedirs(os.path.dirname(cache_path), exist_ok=True)
	tmp_path = f"{cache_path}.tmp"
	with open(tmp_path, 'wb') as f:
		np.savez(
			f,
			orientations=unique_orientations.astype(float),
			responsive_fraction_by_orientation=fractions,
			excluded_mapped_sf=float(RESPONSIVE_ORIENTATION_EXCLUDED_MAPPED_SF),
		)
	os.replace(tmp_path, cache_path)
	return unique_orientations.astype(float), fractions


def load_session(session_rel: str) -> Dict[str, Optional[np.ndarray]]:
	"""Load relevant metrics for one session."""

	session_dir = os.path.join(RESULTS_ROOT, session_rel)
	metrics: Dict[str, Optional[np.ndarray]] = {}

	# Load unique spatial frequencies from orientation_decoding_metadata
	orientation_metadata = safe_load_npz(os.path.join(session_dir, "orientation_decoding_metadata.npz"))
	session_sfs = np.array(orientation_metadata["unique_spatialfreqs"], dtype=float) if (orientation_metadata is not None and "unique_spatialfreqs" in orientation_metadata.files) else None

	metrics["session_sfs"] = session_sfs
	metrics["responsive_fraction_by_sf"] = safe_load_npy(os.path.join(session_dir, "responsive_fraction_by_sf.npy"))
	metrics["responsive_neurons_by_sf"] = safe_load_npy(os.path.join(session_dir, "responsive_neurons_by_sf.npy"))
	metrics["osi_spf"] = safe_load_npy(os.path.join(session_dir, "osi_spf.npy"))
	metrics["osi_diff"] = safe_load_npy(os.path.join(session_dir, "osi_diff.npy"))
	metrics["orientation_decoding_accuracy"] = safe_load_first_existing_npy([
		os.path.join(session_dir, "orientation_continuous_circular_corr_repeats_by_sf.npy"),
		os.path.join(session_dir, "orientation_decoding_circular_correlation_repeats_by_sf.npy"),
		os.path.join(session_dir, "orientation_decoding_accuracy_repeats_by_sf.npy"),
	])
	metrics["noise_corr_mean_by_sf"] = safe_load_npy(os.path.join(session_dir, "noise_corr_mean_by_sf.npy"))
	metrics["signal_corr_mean_by_sf"] = safe_load_npy(os.path.join(session_dir, "signal_corr_mean_by_sf.npy"))
	metrics["spatialfreq_tuning"] = safe_load_npy(os.path.join(session_dir, "spatialfreq_tuning.npy"))
	metrics["decoder_orientation_accuracy_by_sf"] = safe_load_npy(os.path.join(session_dir, "decoder_orientation_accuracy_by_sf.npy"))
	metrics["decoder_fraction_relative_orientation_by_sf"] = safe_load_npy(os.path.join(session_dir, "decoder_fraction_relative_orientation_by_sf.npy"))
	metrics["decoding_based_osi_by_sf"] = safe_load_npy(os.path.join(session_dir, "decoding_based_osi_by_sf.npy"))
	metrics["sf_decoding_accuracy_by_orientation"] = safe_load_first_existing_npy([
		os.path.join(session_dir, "sf_continuous_pearson_by_orientation.npy"),
		os.path.join(session_dir, "sf_decoding_pearsonr_by_orientation.npy"),
		os.path.join(session_dir, "sf_decoding_accuracy_by_orientation.npy"),
	])
	metrics["decoding_cellcount"] = safe_load_npz(os.path.join(session_dir, "decoding_cellcount_results.npz"))
	metrics["signal_minus_noise_corr_by_sf"] = safe_load_npy(os.path.join(session_dir, "signal_minus_noise_corr_by_sf.npy"))
	metrics["signal_corr_mean_by_orientation"] = safe_load_npy(os.path.join(session_dir, "signal_corr_mean_by_orientation.npy"))
	metrics["noise_corr_mean_by_orientation"] = safe_load_npy(os.path.join(session_dir, "noise_corr_mean_by_orientation.npy"))
	metrics["signal_minus_noise_corr_by_orientation"] = safe_load_npy(os.path.join(session_dir, "signal_minus_noise_corr_by_orientation.npy"))
	metrics["orientation_metadata"] = orientation_metadata
	metrics["pca_dim_summary"] = safe_load_npz(os.path.join(session_dir, "pca_dimensionality_summary.npz"))
	metrics["dspca_results"] = safe_load_npz(os.path.join(session_dir, "dspca_results.npz"))
	return metrics


def prepare_session_summaries(session_rel: str) -> Dict[str, np.ndarray]:
	"""Compute aligned per-session summaries needed for group plots."""

	data = load_session(session_rel)
	session_sfs = data.get("session_sfs")
	orientations = None
	meta = data.get("orientation_metadata")
	if meta is not None and "unique_orientations" in meta.files:
		orientations = np.array(meta["unique_orientations"], dtype=float)
	unique_spatialfreqs_meta = np.array(meta["unique_spatialfreqs"], dtype=float) if (meta is not None and "unique_spatialfreqs" in meta.files) else None
	responsive_fraction_by_orientation_non064 = None
	responsive_orientation_metric = _load_or_compute_responsive_fraction_by_orientation_excluding_064(session_rel)
	if responsive_orientation_metric is not None:
		responsive_orientations, responsive_fraction_by_orientation_non064 = responsive_orientation_metric
		if orientations is None:
			orientations = responsive_orientations
		elif not np.array_equal(np.asarray(orientations, dtype=float), np.asarray(responsive_orientations, dtype=float)):
			print(f"[warn] Orientation mismatch in cached responsive-by-orientation metric for {session_rel}; ignoring metric")
			responsive_fraction_by_orientation_non064 = None

	# Responsive fraction (per SF)
	resp_frac = data.get("responsive_fraction_by_sf")
	resp_frac_aligned = align_to_targets(resp_frac, session_sfs)

	# OSI vector sum and diff: mean over responsive neurons per SF
	responsive_mask = data.get("responsive_neurons_by_sf")
	osi_spf = data.get("osi_spf")
	osi_diff = data.get("osi_diff")

	def mean_osi(osi_matrix: Optional[np.ndarray]) -> np.ndarray:
		if osi_matrix is None or responsive_mask is None or session_sfs is None:
			return np.full_like(TARGET_SF_VALUES, np.nan, dtype=float)
		means = []
		for sf_idx in range(responsive_mask.shape[0]):
			mask = responsive_mask[sf_idx].astype(bool)
			vals = osi_matrix[:, sf_idx]
			means.append(np.nanmean(vals[mask]) if np.any(mask) else np.nan)
		means = np.array(means, dtype=float)
		return align_to_targets(means, session_sfs)

	osi_spf_mean = mean_osi(osi_spf)
	osi_diff_mean = mean_osi(osi_diff)

	# Orientation decoding accuracy per SF (mean over repeats)
	decoding_cellcount = data.get("decoding_cellcount")
	decoding_aligned = None
	decoding_based_osi = None
	rel_frac_aligned = None
	dec_ori_acc_aligned = None
	sf_decoding_by_ori = None
	per_n_metrics: Dict[str, np.ndarray] = {}

	# If available, prefer the per-cell-count outputs from decoding_cellcount_results.npz
	if decoding_cellcount is not None and "cell_counts" in decoding_cellcount.files:
		try:
			cc = np.array(decoding_cellcount["cell_counts"], dtype=int)
			dec_sf_by_n = get_from_npz(
				decoding_cellcount,
				["orientation_continuous_circular_corr_by_sf", "orientation_decoding_accuracy_by_sf"],
			)
			sf_by_ori_by_n = get_from_npz(
				decoding_cellcount,
				["sf_continuous_pearson_by_orientation", "sf_decoding_accuracy_by_orientation"],
			)
			# Build per-N metric keys used for grouped plots
			for n in DECODING_CELL_COUNTS:
				if n not in cc:
					continue
				idx = int(np.where(cc == int(n))[0][0])
				# Orientation decoding accuracy per SF
				if dec_sf_by_n is None:
					continue
				dec_sf = dec_sf_by_n[idx]
				dec_sf_aligned = align_to_targets(dec_sf, session_sfs)
				# Decoder-based OSI per SF
				osi_sf = decoding_cellcount["decoding_based_osi_by_sf"][idx]
				osi_sf_aligned = align_to_targets(osi_sf, session_sfs)
				# Decoder fractions over relative orientations (SF x bins)
				rel_sf = decoding_cellcount["decoder_fraction_relative_orientation_by_sf"][idx]
				rel_sf_aligned = align_rows_to_targets(rel_sf, session_sfs)
				# Decoder orientation accuracy per SF (SF x orientation)
				acc_sf = decoding_cellcount["decoder_orientation_accuracy_by_sf"][idx]
				acc_sf_aligned = align_rows_to_targets(acc_sf, session_sfs)
				# SF decoding accuracy by orientation (vector)
				if sf_by_ori_by_n is None:
					continue
				sf_by_ori = sf_by_ori_by_n[idx]

				# Store per-N keys
				# NOTE: existing plotting/stats functions treat each metric_key independently.
				#       We keep defaults below (N=DECODING_DEFAULT_N) to preserve old behavior.
				# 1D over SF
				# - decoding accuracy
				# - decoding-based OSI
				# 2D over SF x bins/orientation
				# - rel_frac
				# - dec_ori_acc
				# 1D over orientation
				# - sf_decoding_by_ori
				key_suffix = f"_n{int(n)}"
				per_n_metrics[f"decoding_accuracy{key_suffix}"] = dec_sf_aligned
				per_n_metrics[f"decoding_based_osi{key_suffix}"] = osi_sf_aligned
				per_n_metrics[f"rel_frac{key_suffix}"] = rel_sf_aligned
				per_n_metrics[f"dec_ori_acc{key_suffix}"] = acc_sf_aligned
				per_n_metrics[f"sf_decoding_by_ori{key_suffix}"] = np.array(sf_by_ori, dtype=float)

			# Set defaults to N=DECODING_DEFAULT_N (if present)
			if int(DECODING_DEFAULT_N) in cc:
				idx0 = int(np.where(cc == int(DECODING_DEFAULT_N))[0][0])
				if dec_sf_by_n is not None:
					decoding_aligned = align_to_targets(dec_sf_by_n[idx0], session_sfs)
				decoding_based_osi = align_to_targets(decoding_cellcount["decoding_based_osi_by_sf"][idx0], session_sfs)
				rel_frac_aligned = align_rows_to_targets(decoding_cellcount["decoder_fraction_relative_orientation_by_sf"][idx0], session_sfs)
				dec_ori_acc_aligned = align_rows_to_targets(decoding_cellcount["decoder_orientation_accuracy_by_sf"][idx0], session_sfs)
				if sf_by_ori_by_n is not None:
					sf_decoding_by_ori = np.array(sf_by_ori_by_n[idx0], dtype=float)
		except Exception:
			decoding_aligned = None

	# Fallback to legacy single-N outputs if decoding_cellcount_results.npz is missing
	if decoding_aligned is None:
		decoding = data.get("orientation_decoding_accuracy")
		if decoding is not None:
			if decoding.ndim == 2:
				decoding = np.nanmean(decoding, axis=0)
			decoding_aligned = align_to_targets(decoding, session_sfs)
		else:
			decoding_aligned = np.full_like(TARGET_SF_VALUES, np.nan, dtype=float)

	# Noise and signal correlations
	noise_corr = align_to_targets(data.get("noise_corr_mean_by_sf"), session_sfs)
	signal_corr = align_to_targets(data.get("signal_corr_mean_by_sf"), session_sfs)
	signal_minus_noise_sf = align_to_targets(data.get("signal_minus_noise_corr_by_sf"), session_sfs)

	# Decoding-based OSI (per SF)
	if decoding_based_osi is None:
		decoding_based_osi = align_to_targets(data.get("decoding_based_osi_by_sf"), session_sfs)

	# Decoder outputs over relative orientations (rows: SF, cols: rel bins)
	if rel_frac_aligned is None:
		rel_frac = data.get("decoder_fraction_relative_orientation_by_sf")
		rel_frac_aligned = align_rows_to_targets(rel_frac, session_sfs)

	# Decoder orientation accuracy per SF (rows: SF, cols: orientations)
	if dec_ori_acc_aligned is None:
		dec_ori_acc = data.get("decoder_orientation_accuracy_by_sf")
		dec_ori_acc_aligned = align_rows_to_targets(dec_ori_acc, session_sfs)

	# SF decoding accuracy by orientation (vector over orientations)
	if sf_decoding_by_ori is None:
		sf_decoding_by_ori = data.get("sf_decoding_accuracy_by_orientation")

	# Signal/noise by orientation (vector over orientations)
	signal_corr_by_ori = data.get("signal_corr_mean_by_orientation")
	noise_corr_by_ori = data.get("noise_corr_mean_by_orientation")
	signal_minus_noise_by_ori = data.get("signal_minus_noise_corr_by_orientation")

	# Participation ratio and power-law alpha
	pca_npz = data.get("pca_dim_summary")
	pr_by_sf_raw = get_from_npz(pca_npz, ["participation_ratio_by_sf", "pr_by_sf"])
	pr_by_sf = align_to_targets(pr_by_sf_raw, session_sfs)
	pr_overall = to_nan_array(get_from_npz(pca_npz, ["participation_ratio_all", "participation_ratio_overall", "pr_overall"]))
	alpha_powerlaw = to_nan_array(get_from_npz(pca_npz, ["powerlaw_alpha", "alpha_powerlaw", "alpha"]))

	# dsPCA variance explained (extract scalars from target_var array)
	dspca_npz = data.get("dspca_results")
	target_var = get_from_npz(dspca_npz, ["target_var"])
	if target_var is not None and len(target_var) >= 2:
		var_ori = float(target_var[0]) if np.isfinite(target_var[0]) else np.nan
		var_sf = float(target_var[1]) if np.isfinite(target_var[1]) else np.nan
	else:
		var_ori = np.nan
		var_sf = np.nan

	# Build return dict; include per-N keys if decoding_cellcount_results.npz was available.
	out: Dict[str, np.ndarray] = {
		"responsive_fraction_by_sf": resp_frac_aligned,
		"responsive_fraction_by_orientation_non064": (
			np.asarray(responsive_fraction_by_orientation_non064, dtype=float)
			if responsive_fraction_by_orientation_non064 is not None
			else np.full(len(orientations), np.nan, dtype=float) if orientations is not None else None
		),
		"osi_spf": osi_spf_mean,
		"osi_diff": osi_diff_mean,
		"decoding_accuracy": decoding_aligned,
		"noise_corr": noise_corr,
		"signal_corr": signal_corr,
		"signal_minus_noise_sf": signal_minus_noise_sf,
		"decoding_based_osi": decoding_based_osi,
		"rel_frac": rel_frac_aligned,
		"dec_ori_acc": dec_ori_acc_aligned,
		"orientations": orientations,
		"sf_decoding_by_ori": sf_decoding_by_ori,
		"signal_corr_by_ori": signal_corr_by_ori,
		"noise_corr_by_ori": noise_corr_by_ori,
		"signal_minus_noise_by_ori": signal_minus_noise_by_ori,
		"pr_by_sf": pr_by_sf,
		"pr_overall": pr_overall,
		"alpha_powerlaw": alpha_powerlaw,
		"var_ori": var_ori,
		"var_sf": var_sf,
		"unique_spatialfreqs_meta": unique_spatialfreqs_meta,
	}

	# Attach per-cell-count decoding keys, if present.
	# Ensure all requested per-N keys exist to avoid KeyErrors in aggregation.
	# Sessions missing decoding_cellcount_results.npz will contribute NaNs for those keys.
	for n in DECODING_CELL_COUNTS:
		suffix = f"_n{int(n)}"
		per_n_metrics.setdefault(f"decoding_accuracy{suffix}", np.full_like(TARGET_SF_VALUES, np.nan, dtype=float))
		per_n_metrics.setdefault(f"decoding_based_osi{suffix}", np.full_like(TARGET_SF_VALUES, np.nan, dtype=float))
		per_n_metrics.setdefault(f"rel_frac{suffix}", np.full((len(TARGET_SF_VALUES), 6), np.nan, dtype=float))
		per_n_metrics.setdefault(f"dec_ori_acc{suffix}", np.full((len(TARGET_SF_VALUES), 6), np.nan, dtype=float))
		per_n_metrics.setdefault(f"sf_decoding_by_ori{suffix}", np.full((6,), np.nan, dtype=float))

	out.update(per_n_metrics)

	return out


# ---------------------------------------------------------------------------
# Aggregation and plotting
# ---------------------------------------------------------------------------
def aggregate_group(metric_key: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
	"""Return mean and SEM per group for the requested metric."""

	results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
	for group_name, sessions in sorted(GROUPS.items()):
		aligned = []
		for session in sessions:
			session_summary = prepare_session_summaries(session)
			aligned.append(session_summary[metric_key])
		stack = np.stack(aligned, axis=0)
		mean, sem = nan_mean_sem(stack, axis=0)
		results[group_name] = (mean, sem)
	return results


def aggregate_group_matrix(metric_key: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
	"""Aggregate a matrix-valued metric with consistent shapes across sessions."""

	results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
	for group_name, sessions in GROUPS.items():
		aligned = []
		ref_shape: Optional[Tuple[int, ...]] = None
		for session in sessions:
			session_summary = prepare_session_summaries(session)
			arr = session_summary[metric_key]
			if arr is None or arr.size == 0:
				continue
			if ref_shape is None:
				ref_shape = arr.shape
			elif arr.shape != ref_shape:
				print(f"[warn] Shape mismatch for {metric_key} in group {group_name}; skipping session {session}")
				continue
			aligned.append(arr)
		if len(aligned) == 0:
			continue
		stack = np.stack(aligned, axis=0)
		mean, sem = nan_mean_sem(stack, axis=0)
		results[group_name] = (mean, sem)
	return results


def aggregate_group_orientation(metric_key: str) -> Tuple[Optional[np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
	"""Aggregate orientation-vector metrics ensuring consistent orientations."""

	orientations_ref: Optional[np.ndarray] = None
	results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

	for group_name, sessions in GROUPS.items():
		aligned = []
		group_orientations: Optional[np.ndarray] = None
		for session in sessions:
			session_summary = prepare_session_summaries(session)
			if session_summary["orientations"] is None:
				continue
			if group_orientations is None:
				group_orientations = session_summary["orientations"]
			elif not np.array_equal(group_orientations, session_summary["orientations"]):
				print(f"[warn] Orientation mismatch inside group {group_name}; skipping orientation metrics for this group")
				group_orientations = None
				break
			aligned.append(session_summary[metric_key])

		if group_orientations is None or len(aligned) == 0:
			continue

		if orientations_ref is None:
			orientations_ref = group_orientations
		elif not np.array_equal(orientations_ref, group_orientations):
			print("[warn] Orientation mismatch across groups; skipping orientation plots")
			return None, {}

		stack = np.stack(aligned, axis=0)
		mean, sem = nan_mean_sem(stack, axis=0)
		results[group_name] = (mean, sem)

	return orientations_ref, results


def get_orientation_reference() -> Optional[np.ndarray]:
	"""Find a consistent orientation array across all sessions, else None."""

	orient_ref: Optional[np.ndarray] = None
	for sessions in GROUPS.values():
		for session in sessions:
			summary = prepare_session_summaries(session)
			if summary["orientations"] is None:
				continue
			if orient_ref is None:
				orient_ref = summary["orientations"]
			elif not np.array_equal(orient_ref, summary["orientations"]):
				print("[warn] orientation mismatch across sessions; chance lines skipped")
				return None
	return orient_ref


def get_nonzero_sf_count() -> Optional[int]:
	"""Return count of non-baseline SF classes if consistent across sessions."""

	count_ref: Optional[int] = None
	for sessions in GROUPS.values():
		for session in sessions:
			data = load_session(session)
			meta = data.get("orientation_metadata")
			if meta is None or "unique_spatialfreqs" not in meta.files:
				continue
			sf_vals = np.array(meta["unique_spatialfreqs"], dtype=float)
			nonzero = np.sum(sf_vals != 0)
			if count_ref is None:
				count_ref = nonzero
			elif count_ref != nonzero:
				print("[warn] spatial frequency class count mismatch; chance lines skipped for SF decoding")
				return None
	return count_ref


def _plot_blind_individual_traces_by_sf(ax, metric_key: str, x: np.ndarray, blind_color: Optional[str] = None) -> None:
	"""Overlay Blind-session individual traces for SF-aligned scalar metrics."""
	blind_color = blind_color if blind_color is not None else GROUP_COLORS.get('BLIND', '#7f7f7f')
	blind_label = GROUP_LABELS.get('BLIND', 'Blind')
	plotted_label = False
	for session in GROUPS.get('BLIND', []):
		try:
			summary = prepare_session_summaries(session)
		except Exception:
			continue
		vals = summary.get(metric_key)
		if vals is None:
			continue
		arr = np.asarray(vals, dtype=float)
		if arr.ndim != 1 or arr.size <= 1:
			continue
		y = arr[1:]
		if y.size != len(x) or not np.any(np.isfinite(y)):
			continue
		ax.plot(
			x,
			y,
			'-',
			color=blind_color,
			linewidth=1.0,
			label=blind_label if not plotted_label else None,
		)
		plotted_label = True


def _plot_blind_individual_traces_by_orientation(ax, metric_key: str, x: np.ndarray, blind_color: Optional[str] = None) -> None:
	"""Overlay Blind-session individual traces for orientation-aligned metrics."""
	blind_color = blind_color if blind_color is not None else GROUP_COLORS.get('BLIND', '#7f7f7f')
	blind_label = GROUP_LABELS.get('BLIND', 'Blind')
	x_arr = np.asarray(x, dtype=float).ravel()
	plotted_label = False
	for session in GROUPS.get('BLIND', []):
		try:
			summary = prepare_session_summaries(session)
		except Exception:
			continue
		orientations = summary.get('orientations')
		vals = summary.get(metric_key)
		if orientations is None or vals is None:
			continue
		orientations_arr = np.asarray(orientations, dtype=float).ravel()
		arr = np.asarray(vals, dtype=float).ravel()
		if orientations_arr.shape != x_arr.shape or arr.shape != x_arr.shape:
			continue
		if not np.array_equal(orientations_arr, x_arr):
			continue
		if not np.any(np.isfinite(arr)):
			continue
		ax.plot(
			x_arr,
			arr,
			'-',
			color=blind_color,
			linewidth=1.0,
			label=blind_label if not plotted_label else None,
		)
		plotted_label = True


def _collect_session_metric_matrix_by_orientation(session_list: List[str], metric_key: str) -> np.ndarray:
	"""Collect a subject-by-orientation matrix for an orientation-aligned metric."""
	rows: List[np.ndarray] = []
	orient_ref: Optional[np.ndarray] = None
	for session in session_list:
		try:
			summary = prepare_session_summaries(session)
		except Exception:
			continue
		orientations = summary.get('orientations')
		vals = summary.get(metric_key)
		if orientations is None or vals is None:
			continue
		orient_arr = np.asarray(orientations, dtype=float).ravel()
		val_arr = np.asarray(vals, dtype=float).ravel()
		if orient_arr.shape != val_arr.shape or orient_arr.size == 0:
			continue
		if orient_ref is None:
			orient_ref = orient_arr
		elif not np.array_equal(orient_ref, orient_arr):
			continue
		rows.append(val_arr)
	if not rows:
		return np.empty((0, 0), dtype=float)
	return np.vstack(rows)


def _collect_blind_curves_over_cell_counts(metric_base: str, sf_idx: int, cell_counts: List[int]) -> List[np.ndarray]:
	"""Collect Blind-session curves across neuron counts for one SF index."""
	blind_curves: List[np.ndarray] = []
	for s_blind in GROUPS.get('BLIND', []):
		try:
			summary_b = prepare_session_summaries(s_blind)
		except Exception:
			continue
		y_b = np.full(len(cell_counts), np.nan, dtype=float)
		for j, n in enumerate(cell_counts):
			v_b = _get_metric_for_n(summary_b, metric_base, n)
			if v_b is None:
				continue
			try:
				v_b_arr = np.asarray(v_b, dtype=float)
				if v_b_arr.ndim != 1 or sf_idx >= v_b_arr.size:
					continue
				y_b[j] = float(v_b_arr[sf_idx])
			except Exception:
				continue
		if np.any(np.isfinite(y_b)):
			blind_curves.append(y_b)
	return blind_curves


def plot_metric(metric_key: str, title: str, ylabel: str, filename: str, ylim: Optional[Tuple[float, float]] = None, chance: Optional[float] = None, blind_as_individual: bool = False, blind_color_override: Optional[str] = None, background_band: Optional[Tuple[float, float]] = None, background_band_color: str = '#d9d9d9', exclude_groups: Optional[List[str]] = None, chance_null_metric_key: Optional[str] = None) -> None:
	"""Generic grouped line plot against spatial frequency with statistical annotations."""

	stats = aggregate_group(metric_key)
	if not stats:
		print(f"[skip] No data for {metric_key}")
		return
	exclude = set(exclude_groups or [])
	group_plot_order = [
		"WT_BEFORE",
		"WT_AFTER",
		"RESCUED_BEFORE",
		"RESCUED_AFTER",
		"BLIND",
	]
	ordered_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
	for group_name in group_plot_order:
		if group_name in stats and group_name not in exclude:
			ordered_stats[group_name] = stats[group_name]
	for group_name, value in stats.items():
		if group_name not in ordered_stats and group_name not in exclude:
			ordered_stats[group_name] = value
	stats = ordered_stats

	# Check if any data is finite
	has_data = False
	for mean, sem in stats.values():
		if has_any_finite(mean):
			has_data = True
			break
	if not has_data:
		print(f"[skip] {metric_key} has no finite data")
		return

	# Wider figure so the right-margin stats panel doesn't shrink the plotting area.
	fig, ax = plt.subplots(figsize=(12, 5))
	x = TARGET_SF_VALUES[1:]  # skip baseline 0 sf for plotting
	chance_samples = _get_decoder_null_distribution(chance_null_metric_key) if chance_null_metric_key else None
	if chance_samples is not None:
		low, high = _decoder_null_band(chance_samples)
		ax.fill_between(
			x,
			low[1:],
			high[1:],
			color='#d9d9d9',
			alpha=1.0,
			zorder=0,
		)
	if background_band is not None:
		band_y0, band_y1 = background_band
		ax.axhspan(float(band_y0), float(band_y1), facecolor=background_band_color, edgecolor='none', zorder=0)

	markers = ["o", "s", "^", "D", "v"]
	for idx, (group_name, (mean, sem)) in enumerate(stats.items()):
		if blind_as_individual and group_name == "BLIND":
			continue
		color = blind_color_override if (group_name == "BLIND" and blind_color_override is not None) else GROUP_COLORS[group_name]
		plot_with_error(x, mean[1:], sem[1:], GROUP_LABELS[group_name], color, ax, marker=markers[idx % len(markers)])
	if blind_as_individual:
		_plot_blind_individual_traces_by_sf(ax, metric_key, x, blind_color=blind_color_override)

	if chance is not None:
		ax.axhline(y=chance, color="k", linestyle="--", linewidth=1, label="chance")
	elif ylim is not None and ylim[0] < 0 < ylim[1]:
		ax.axhline(y=0.0, color="k", linestyle="--", linewidth=1, alpha=0.7)

	ax.set_xscale("log")
	ax.set_xlabel("Spatial frequency (log scale)")
	ax.set_ylabel(ylabel)
	ax.set_title(title)
	ax.legend()
	if ylim is not None:
		ax.set_ylim(*ylim)
	ax.grid(True, which="both", linestyle="--", alpha=0.4)

	# Post-hoc simple effects (Training within each SF), pooled across all paired mice.
	# Draw stars using axes-fraction y so they stay visible even if ylim changes.
	pairs_wt = MOUSE_PAIRS.get("WT", [])
	pairs_resc = MOUSE_PAIRS.get("RESCUED", [])
	posthoc_rm_wt = _posthoc_training_pvalues_within_each_sf_rm(metric_key, pairs_wt)
	posthoc_rm_resc = _posthoc_training_pvalues_within_each_sf_rm(metric_key, pairs_resc)
	posthoc_rm_wt_fdr = _bh_fdr_correct(posthoc_rm_wt)
	posthoc_rm_resc_fdr = _bh_fdr_correct(posthoc_rm_resc)

	# Numeric p/q summary panel (in the figure margin, outside the axes)
	posthoc_art_wt = _posthoc_training_pvalues_within_each_sf_art(metric_key, pairs_wt)
	posthoc_art_resc = _posthoc_training_pvalues_within_each_sf_art(metric_key, pairs_resc)
	posthoc_art_wt_fdr = _bh_fdr_correct(posthoc_art_wt)
	posthoc_art_resc_fdr = _bh_fdr_correct(posthoc_art_resc)
	posthoc_t_wt = _posthoc_training_pvalues_within_each_sf_ttest(metric_key, pairs_wt)
	posthoc_t_resc = _posthoc_training_pvalues_within_each_sf_ttest(metric_key, pairs_resc)
	posthoc_t_wt_fdr = _bh_fdr_correct(posthoc_t_wt)
	posthoc_t_resc_fdr = _bh_fdr_correct(posthoc_t_resc)
	posthoc_w_wt = _posthoc_training_pvalues_within_each_sf_wilcoxon(metric_key, pairs_wt)
	posthoc_w_resc = _posthoc_training_pvalues_within_each_sf_wilcoxon(metric_key, pairs_resc)
	posthoc_w_wt_fdr = _bh_fdr_correct(posthoc_w_wt)
	posthoc_w_resc_fdr = _bh_fdr_correct(posthoc_w_resc)

	def _fmt_val(v: Optional[float]) -> str:
		if v is None or not np.isfinite(v):
			return "NA"
		vv = float(v)
		return f"{vv:.3f}" + ("*" if vv < 0.05 else "")

	def _fmt_pq(p: Optional[float], q: Optional[float]) -> str:
		return f"{_fmt_val(p)}/{_fmt_val(q)}"

	# Omnibus (two-way) test results: move these out of the axes into the margin panel.
	pvals = compute_paired_pvalues(metric_key)
	rmanova_pvals = compute_rmanova_pvalues(metric_key)
	art_pvals = compute_art_pvalues_sf(metric_key)
	art_raov_pvals = compute_art_raov_pvalues_sf(metric_key)
	art_rlmer_pvals = compute_art_r_pvalues_sf(metric_key)
	lme_pvals = compute_lme_pvalues_sf(metric_key)
	lme4_pvals = compute_lme4_pvalues_sf(metric_key)

	panel_lines: List[str] = [
		"Omnibus tests (p) (* means <0.05)",
	]
	if RM_ANOVA_INCLUDE_INTERACTION_TERMS:
		panel_lines.append(
			(
				"WT: paired={paired}  RM train={rm_tr}  RM train×SF={rm_int}"
			).format(
				paired=_fmt_val(pvals.get("WT")),
				rm_tr=_fmt_val(rmanova_pvals.get("WT", {}).get("training_p")),
				rm_int=_fmt_val(rmanova_pvals.get("WT", {}).get("interaction_p")),
			)
		)
		panel_lines.append(
			(
				"Resc: paired={paired}  RM train={rm_tr}  RM train×SF={rm_int}"
			).format(
				paired=_fmt_val(pvals.get("RESCUED")),
				rm_tr=_fmt_val(rmanova_pvals.get("RESCUED", {}).get("training_p")),
				rm_int=_fmt_val(rmanova_pvals.get("RESCUED", {}).get("interaction_p")),
			)
		)
	else:
		panel_lines.append(
			(
				"WT: paired={paired}  RM train={rm_tr}"
			).format(
				paired=_fmt_val(pvals.get("WT")),
				rm_tr=_fmt_val(rmanova_pvals.get("WT", {}).get("training_p")),
			)
		)
		panel_lines.append(
			(
				"Resc: paired={paired}  RM train={rm_tr}"
			).format(
				paired=_fmt_val(pvals.get("RESCUED")),
				rm_tr=_fmt_val(rmanova_pvals.get("RESCUED", {}).get("training_p")),
			)
		)

	if ART_INCLUDE_INTERACTION_TERMS:
		panel_lines.append(
			(
				"WT: ART train={art_tr}  ART SF={art_sf}  ART int={art_int}"
			).format(
				art_tr=_fmt_val(art_pvals.get("WT", {}).get("training_p")),
				art_sf=_fmt_val(art_pvals.get("WT", {}).get("sf_p")),
				art_int=_fmt_val(art_pvals.get("WT", {}).get("interaction_p")),
			)
		)
		panel_lines.append(
			(
				"Resc: ART train={art_tr}  ART SF={art_sf}  ART int={art_int}"
			).format(
				art_tr=_fmt_val(art_pvals.get("RESCUED", {}).get("training_p")),
				art_sf=_fmt_val(art_pvals.get("RESCUED", {}).get("sf_p")),
				art_int=_fmt_val(art_pvals.get("RESCUED", {}).get("interaction_p")),
			)
		)
		panel_lines.append(
			(
				"WT: ART-Raov train={art_tr}  ART-Raov SF={art_sf}  ART-Raov int={art_int}"
			).format(
				art_tr=_fmt_val(art_raov_pvals.get("WT", {}).get("training_p")),
				art_sf=_fmt_val(art_raov_pvals.get("WT", {}).get("sf_p")),
				art_int=_fmt_val(art_raov_pvals.get("WT", {}).get("interaction_p")),
			)
		)
		panel_lines.append(
			(
				"Resc: ART-Raov train={art_tr}  ART-Raov SF={art_sf}  ART-Raov int={art_int}"
			).format(
				art_tr=_fmt_val(art_raov_pvals.get("RESCUED", {}).get("training_p")),
				art_sf=_fmt_val(art_raov_pvals.get("RESCUED", {}).get("sf_p")),
				art_int=_fmt_val(art_raov_pvals.get("RESCUED", {}).get("interaction_p")),
			)
		)
		panel_lines.append(
			(
				"WT: ART-Rlmer train={art_tr}  ART-Rlmer SF={art_sf}  ART-Rlmer int={art_int}"
			).format(
				art_tr=_fmt_val(art_rlmer_pvals.get("WT", {}).get("training_p")),
				art_sf=_fmt_val(art_rlmer_pvals.get("WT", {}).get("sf_p")),
				art_int=_fmt_val(art_rlmer_pvals.get("WT", {}).get("interaction_p")),
			)
		)
		panel_lines.append(
			(
				"Resc: ART-Rlmer train={art_tr}  ART-Rlmer SF={art_sf}  ART-Rlmer int={art_int}"
			).format(
				art_tr=_fmt_val(art_rlmer_pvals.get("RESCUED", {}).get("training_p")),
				art_sf=_fmt_val(art_rlmer_pvals.get("RESCUED", {}).get("sf_p")),
				art_int=_fmt_val(art_rlmer_pvals.get("RESCUED", {}).get("interaction_p")),
			)
		)
	else:
		panel_lines.append(
			(
				"WT: ART train={art_tr}  ART SF={art_sf}"
			).format(
				art_tr=_fmt_val(art_pvals.get("WT", {}).get("training_p")),
				art_sf=_fmt_val(art_pvals.get("WT", {}).get("sf_p")),
			)
		)
		panel_lines.append(
			(
				"Resc: ART train={art_tr}  ART SF={art_sf}"
			).format(
				art_tr=_fmt_val(art_pvals.get("RESCUED", {}).get("training_p")),
				art_sf=_fmt_val(art_pvals.get("RESCUED", {}).get("sf_p")),
			)
		)
		panel_lines.append(
			(
				"WT: ART-Raov train={art_tr}  ART-Raov SF={art_sf}"
			).format(
				art_tr=_fmt_val(art_raov_pvals.get("WT", {}).get("training_p")),
				art_sf=_fmt_val(art_raov_pvals.get("WT", {}).get("sf_p")),
			)
		)
		panel_lines.append(
			(
				"Resc: ART-Raov train={art_tr}  ART-Raov SF={art_sf}"
			).format(
				art_tr=_fmt_val(art_raov_pvals.get("RESCUED", {}).get("training_p")),
				art_sf=_fmt_val(art_raov_pvals.get("RESCUED", {}).get("sf_p")),
			)
		)
		panel_lines.append(
			(
				"WT: ART-Rlmer train={art_tr}  ART-Rlmer SF={art_sf}"
			).format(
				art_tr=_fmt_val(art_rlmer_pvals.get("WT", {}).get("training_p")),
				art_sf=_fmt_val(art_rlmer_pvals.get("WT", {}).get("sf_p")),
			)
		)
		panel_lines.append(
			(
				"Resc: ART-Rlmer train={art_tr}  ART-Rlmer SF={art_sf}"
			).format(
				art_tr=_fmt_val(art_rlmer_pvals.get("RESCUED", {}).get("training_p")),
				art_sf=_fmt_val(art_rlmer_pvals.get("RESCUED", {}).get("sf_p")),
			)
		)

	# LME (MixedLM) omnibus p-values (LRT)
	if LME_INCLUDE_INTERACTION_TERMS:
		panel_lines.append(
			(
				"WT: LME train={lme_tr}  LME SF={lme_sf}  LME int={lme_int}"
			).format(
				lme_tr=_fmt_val(lme_pvals.get("WT", {}).get("training_p")),
				lme_sf=_fmt_val(lme_pvals.get("WT", {}).get("sf_p")),
				lme_int=_fmt_val(lme_pvals.get("WT", {}).get("interaction_p")),
			)
		)
		panel_lines.append(
			(
				"Resc: LME train={lme_tr}  LME SF={lme_sf}  LME int={lme_int}"
			).format(
				lme_tr=_fmt_val(lme_pvals.get("RESCUED", {}).get("training_p")),
				lme_sf=_fmt_val(lme_pvals.get("RESCUED", {}).get("sf_p")),
				lme_int=_fmt_val(lme_pvals.get("RESCUED", {}).get("interaction_p")),
			)
		)
		panel_lines.append(
			(
				"WT: LME-lme4 train={lme_tr}  LME-lme4 SF={lme_sf}  LME-lme4 int={lme_int}"
			).format(
				lme_tr=_fmt_val(lme4_pvals.get("WT", {}).get("training_p")),
				lme_sf=_fmt_val(lme4_pvals.get("WT", {}).get("sf_p")),
				lme_int=_fmt_val(lme4_pvals.get("WT", {}).get("interaction_p")),
			)
		)
		panel_lines.append(
			(
				"Resc: LME-lme4 train={lme_tr}  LME-lme4 SF={lme_sf}  LME-lme4 int={lme_int}"
			).format(
				lme_tr=_fmt_val(lme4_pvals.get("RESCUED", {}).get("training_p")),
				lme_sf=_fmt_val(lme4_pvals.get("RESCUED", {}).get("sf_p")),
				lme_int=_fmt_val(lme4_pvals.get("RESCUED", {}).get("interaction_p")),
			)
		)
	else:
		panel_lines.append(
			(
				"WT: LME train={lme_tr}  LME SF={lme_sf}"
			).format(
				lme_tr=_fmt_val(lme_pvals.get("WT", {}).get("training_p")),
				lme_sf=_fmt_val(lme_pvals.get("WT", {}).get("sf_p")),
			)
		)
		panel_lines.append(
			(
				"Resc: LME train={lme_tr}  LME SF={lme_sf}"
			).format(
				lme_tr=_fmt_val(lme_pvals.get("RESCUED", {}).get("training_p")),
				lme_sf=_fmt_val(lme_pvals.get("RESCUED", {}).get("sf_p")),
			)
		)
		panel_lines.append(
			(
				"WT: LME-lme4 train={lme_tr}  LME-lme4 SF={lme_sf}"
			).format(
				lme_tr=_fmt_val(lme4_pvals.get("WT", {}).get("training_p")),
				lme_sf=_fmt_val(lme4_pvals.get("WT", {}).get("sf_p")),
			)
		)
		panel_lines.append(
			(
				"Resc: LME-lme4 train={lme_tr}  LME-lme4 SF={lme_sf}"
			).format(
				lme_tr=_fmt_val(lme4_pvals.get("RESCUED", {}).get("training_p")),
				lme_sf=_fmt_val(lme4_pvals.get("RESCUED", {}).get("sf_p")),
			)
		)

	panel_lines.extend([
		"",
		"Posthoc train@SF: p/q (BH)  (* means <0.05)",
		"WT: RM  ART  t  W   (each is p/q)",
		"Resc: RM  ART  t  W",
	])
	for sf_idx in range(1, len(TARGET_SF_VALUES)):
		sf_mapped = convert_sf_value(float(TARGET_SF_VALUES[sf_idx]))
		wt = "  ".join([
			_fmt_pq(posthoc_rm_wt.get(sf_idx), posthoc_rm_wt_fdr.get(sf_idx)),
			_fmt_pq(posthoc_art_wt.get(sf_idx), posthoc_art_wt_fdr.get(sf_idx)),
			_fmt_pq(posthoc_t_wt.get(sf_idx), posthoc_t_wt_fdr.get(sf_idx)),
			_fmt_pq(posthoc_w_wt.get(sf_idx), posthoc_w_wt_fdr.get(sf_idx)),
		])
		re = "  ".join([
			_fmt_pq(posthoc_rm_resc.get(sf_idx), posthoc_rm_resc_fdr.get(sf_idx)),
			_fmt_pq(posthoc_art_resc.get(sf_idx), posthoc_art_resc_fdr.get(sf_idx)),
			_fmt_pq(posthoc_t_resc.get(sf_idx), posthoc_t_resc_fdr.get(sf_idx)),
			_fmt_pq(posthoc_w_resc.get(sf_idx), posthoc_w_resc_fdr.get(sf_idx)),
		])
		panel_lines.append(f"SF={sf_mapped:.2f} WT  : {wt}")
		panel_lines.append(f"     Resc: {re}")

	if chance_samples is not None:
		panel_lines.extend([
			"",
			"Above shuffled chance@SF: sig q<0.05",
		])
		for group_name in ["WT_BEFORE", "WT_AFTER", "RESCUED_BEFORE", "RESCUED_AFTER"]:
			if group_name not in stats:
				continue
			obs_mean = np.asarray(stats[group_name][0], dtype=float)
			p_emp = _empirical_above_chance_pvalues(obs_mean[1:], chance_samples[:, 1:])
			q_emp = _bh_fdr_correct(p_emp)
			q_with_baseline = {0: None}
			for idx0, q_val in q_emp.items():
				q_with_baseline[int(idx0) + 1] = q_val
			panel_lines.append(f"{GROUP_LABELS.get(group_name, group_name)}: {_format_sig_mapped_sf_from_q(q_with_baseline)}")

	if metric_key == "responsive_fraction_by_sf" and filename == "responsive_fraction_by_sf_groups.svg":
		mapped_levels = [convert_sf_value(float(v)) for v in TARGET_SF_VALUES[1:]]
		wt_pre_sessions = [s_pre for s_pre, _ in MOUSE_PAIRS.get("WT", [])]
		wt_post_sessions = [s_post for _, s_post in MOUSE_PAIRS.get("WT", [])]
		resc_pre_sessions = [s_pre for s_pre, _ in MOUSE_PAIRS.get("RESCUED", [])]
		resc_post_sessions = [s_post for _, s_post in MOUSE_PAIRS.get("RESCUED", [])]
		panel_lines.extend([
			"",
			*_build_cross_group_scalar_metric_stats_lines(
				metric_key=metric_key,
				group_a_sessions=wt_pre_sessions,
				group_b_sessions=resc_pre_sessions,
				mapped_levels=mapped_levels,
				label_a="WT pre",
				label_b="Rescued pre",
			),
			"",
			*_build_cross_group_scalar_metric_stats_lines(
				metric_key=metric_key,
				group_a_sessions=wt_pre_sessions,
				group_b_sessions=resc_post_sessions,
				mapped_levels=mapped_levels,
				label_a="WT pre",
				label_b="Rescued post",
			),
			"",
			*_build_above_chance_scalar_metric_stats_lines(
				metric_key=metric_key,
				sessions=wt_pre_sessions,
				label="WT pre",
				chance_value=0.05,
				mapped_levels=mapped_levels,
			),
			"",
			*_build_above_chance_scalar_metric_stats_lines(
				metric_key=metric_key,
				sessions=wt_post_sessions,
				label="WT post",
				chance_value=0.05,
				mapped_levels=mapped_levels,
			),
			"",
			*_build_above_chance_scalar_metric_stats_lines(
				metric_key=metric_key,
				sessions=resc_pre_sessions,
				label="Rescued pre",
				chance_value=0.05,
				mapped_levels=mapped_levels,
			),
			"",
			*_build_above_chance_scalar_metric_stats_lines(
				metric_key=metric_key,
				sessions=resc_post_sessions,
				label="Rescued post",
				chance_value=0.05,
				mapped_levels=mapped_levels,
			),
		])

	# Give the panel enough room without shrinking the plotting area.
	panel_right = 0.67
	panel_fontsize = 6.8
	panel_lines_to_draw = panel_lines
	panel_line_sep = 1.0
	panel_y = 0.50
	if metric_key == "responsive_fraction_by_sf" and filename == "responsive_fraction_by_sf_groups.svg":
		panel_right = 0.58
		panel_fontsize = 4.6
		panel_lines_to_draw = _wrap_stats_panel_lines(panel_lines, max_chars=74)
		panel_line_sep = 0.2
		panel_y = 0.985
	fig.subplots_adjust(right=panel_right)
	_add_colored_stats_panel_to_figure(fig, panel_lines_to_draw, x=0.985, y=panel_y, fontsize=panel_fontsize, line_sep=panel_line_sep)
	star_transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
	# Offset in log-x to separate groups visually
	x_offset_factor = 1.035
	for sf_idx in range(1, len(TARGET_SF_VALUES)):
		x_i = float(TARGET_SF_VALUES[sf_idx])
		x_wt = x_i / x_offset_factor
		x_resc = x_i * x_offset_factor

		p_wt = posthoc_rm_wt.get(sf_idx, None)
		q_wt = posthoc_rm_wt_fdr.get(sf_idx, None)
		p_resc = posthoc_rm_resc.get(sf_idx, None)
		q_resc = posthoc_rm_resc_fdr.get(sf_idx, None)

		# Uncorrected stars (higher, y=0.96): left=WT, right=Rescued
		if p_wt is not None and np.isfinite(p_wt) and p_wt < 0.05:
			ax.text(x_wt, 0.96, _asterisks_from_p(float(p_wt)), transform=star_transform,
				ha='center', va='top', fontsize=10, fontweight='bold')
		if p_resc is not None and np.isfinite(p_resc) and p_resc < 0.05:
			ax.text(x_resc, 0.96, _asterisks_from_p(float(p_resc)), transform=star_transform,
				ha='center', va='top', fontsize=10, fontweight='bold')

		# FDR(BH) stars (lower, y=0.91): left=WT, right=Rescued
		if q_wt is not None and np.isfinite(q_wt) and q_wt < 0.05:
			ax.text(x_wt, 0.91, _asterisks_from_p(float(q_wt)), transform=star_transform,
				ha='center', va='top', fontsize=9, fontweight='bold')
		if q_resc is not None and np.isfinite(q_resc) and q_resc < 0.05:
			ax.text(x_resc, 0.91, _asterisks_from_p(float(q_resc)), transform=star_transform,
				ha='center', va='top', fontsize=9, fontweight='bold')
	
	# Keep space for the right-margin panel.
	fig.tight_layout(rect=(0.0, 0.0, panel_right, 1.0))
	out_path = os.path.join(GROUPED_PLOTS_DIR, filename)
	plt.savefig(out_path, format="svg")
	plt.close(fig)
	print(f"[saved] {out_path}")


def plot_matrix_over_sf(metric_key: str, axis_labels: List[str], title: str, ylabel: str, filename: str, chance: Optional[float] = None) -> None:
	"""Plot multiple subplots (one per second-axis label) vs SF for each group."""

	stats = aggregate_group_matrix(metric_key)
	if not stats:
		print(f"[skip] No data for {metric_key}")
		return

	# Determine feature count from first group's mean
	first_group = next(iter(stats.values()))
	n_feat = first_group[0].shape[1]
	if n_feat == 0:
		print(f"[skip] {metric_key} has zero features")
		return

	# Check if any data is finite
	has_data = False
	for mean, sem in stats.values():
		if np.isfinite(mean).any():
			has_data = True
			break
	if not has_data:
		print(f"[skip] {metric_key} has no finite data")
		return

	if len(axis_labels) < n_feat:
		axis_labels = axis_labels + [f"Feat {i+1}" for i in range(len(axis_labels), n_feat)]
	elif len(axis_labels) > n_feat:
		axis_labels = axis_labels[:n_feat]

	n_cols = 3
	n_rows = int(np.ceil(n_feat / n_cols))
	# Slightly wider canvas with a reserved right margin for omnibus 3-way stats.
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.9 * n_cols, 3.2 * n_rows), sharex=True, sharey=False)
	axes = np.ravel(axes)
	x = TARGET_SF_VALUES[1:]
	markers = ["o", "s", "^", "D", "v"]

	for axis_idx, label in enumerate(axis_labels):
		ax = axes[axis_idx]
		for idx, (group_name, (mean, sem)) in enumerate(stats.items()):
			plot_with_error(x, mean[1:, axis_idx], sem[1:, axis_idx], GROUP_LABELS[group_name], GROUP_COLORS[group_name], ax, marker=markers[idx % len(markers)])
		if chance is not None:
			ax.axhline(y=chance, color="k", linestyle="--", linewidth=1, label="chance" if axis_idx == 0 else None)
		ax.set_xscale("log")
		ax.set_title(label)
		ax.grid(True, which="both", linestyle="--", alpha=0.4)
		if axis_idx % n_cols == 0:
			ax.set_ylabel(ylabel)
	for ax in axes:
		ax.set_xlabel("Spatial frequency (log scale)")

	# Omnibus 3-way stats panel: Training × SF × Feature
	panel_lines: List[str] = []
	for condition in ["WT", "RESCUED"]:
		pairs = MOUSE_PAIRS.get(condition, [])
		if not pairs:
			continue
		rows: List[Dict[str, object]] = []
		for session_before, session_after in pairs:
			s_before = prepare_session_summaries(session_before)
			s_after = prepare_session_summaries(session_after)
			vb = s_before.get(metric_key)
			va = s_after.get(metric_key)
			if vb is None or va is None:
				continue
			vb = np.asarray(vb, dtype=float)
			va = np.asarray(va, dtype=float)
			if vb.ndim != 2 or va.ndim != 2:
				continue
			n_sf = min(vb.shape[0], va.shape[0], len(TARGET_SF_VALUES))
			n_bins = min(vb.shape[1], va.shape[1], n_feat)
			if n_sf < 2 or n_bins < 2:
				continue
			subj = _mouse_id_from_session_rel(session_before)
			for sf_idx in range(1, n_sf):
				sf = str(float(TARGET_SF_VALUES[sf_idx]))
				for bin_idx in range(n_bins):
					feat = str(int(bin_idx))
					b = float(vb[sf_idx, bin_idx])
					a = float(va[sf_idx, bin_idx])
					if np.isfinite(b) and np.isfinite(a):
						rows.append({"subject": subj, "training": "Pre", "sf": sf, "feat": feat, "value": b})
						rows.append({"subject": subj, "training": "Post", "sf": sf, "feat": feat, "value": a})

		rm3, art3 = _compute_rm_art_3way_from_rows(
			rows,
			subject='subject',
			factor_a='training',
			factor_b='sf',
			factor_c='feat',
			value_col='value',
			center_by_subject=True,
		)
		art3_raov = _art_raov_3way_pvalues_from_rows(
			rows,
			subject='subject',
			factor_a='training',
			factor_b='sf',
			factor_c='feat',
			value_col='value',
			include_interactions=ART_INCLUDE_INTERACTION_TERMS,
		)
		art3_rlmer = _art_r_3way_pvalues_from_rows(
			rows,
			subject='subject',
			factor_a='training',
			factor_b='sf',
			factor_c='feat',
			value_col='value',
			include_interactions=ART_INCLUDE_INTERACTION_TERMS,
		)
		lme3 = _lme_3way_lr_pvalues_from_rows(
			rows,
			subject='subject',
			factor_a='training',
			factor_b='sf',
			factor_c='feat',
			value_col='value',
			include_interactions=LME_INCLUDE_INTERACTION_TERMS,
		)
		panel_lines.append(f"{condition} RM3 (t,sf,f)")
		panel_lines.append(f"  main: t={rm3['training']!s} sf={rm3['sf']!s} f={rm3['feat']!s}")
		panel_lines.append(
			f"  int: t:sf={rm3['training:sf']!s} t:f={rm3['training:feat']!s} sf:f={rm3['sf:feat']!s} t:sf:f={rm3['training:sf:feat']!s}"
		)
		panel_lines.append(f"{condition} ART3 (t,sf,f)")
		panel_lines.append(f"  main: t={art3['training']!s} sf={art3['sf']!s} f={art3['feat']!s}")
		panel_lines.append(
			f"  int: t:sf={art3['training:sf']!s} t:f={art3['training:feat']!s} sf:f={art3['sf:feat']!s} t:sf:f={art3['training:sf:feat']!s}"
		)
		panel_lines.append(f"{condition} ART3-Raov (t,sf,f)")
		panel_lines.append(f"  main: t={art3_raov.get('training')!s} sf={art3_raov.get('sf')!s} f={art3_raov.get('feat')!s}")
		panel_lines.append(
			f"  int: t:sf={art3_raov.get('training:sf')!s} t:f={art3_raov.get('training:feat')!s} sf:f={art3_raov.get('sf:feat')!s} t:sf:f={art3_raov.get('training:sf:feat')!s}"
		)
		panel_lines.append(f"{condition} ART3-Rlmer (t,sf,f)")
		panel_lines.append(f"  main: t={art3_rlmer.get('training')!s} sf={art3_rlmer.get('sf')!s} f={art3_rlmer.get('feat')!s}")
		panel_lines.append(
			f"  int: t:sf={art3_rlmer.get('training:sf')!s} t:f={art3_rlmer.get('training:feat')!s} sf:f={art3_rlmer.get('sf:feat')!s} t:sf:f={art3_rlmer.get('training:sf:feat')!s}"
		)
		panel_lines.append(f"{condition} LME3 (t,sf,f)")
		panel_lines.append(f"  main: t={lme3['training']!s} sf={lme3['sf']!s} f={lme3['feat']!s}")
		panel_lines.append(
			f"  int: t:sf={lme3['training:sf']!s} t:f={lme3['training:feat']!s} sf:f={lme3['sf:feat']!s} t:sf:f={lme3['training:sf:feat']!s}"
		)
		panel_lines.append("-")

	if panel_lines and panel_lines[-1] == "-":
		panel_lines = panel_lines[:-1]
	if panel_lines:
		_add_colored_stats_panel_to_figure(fig, panel_lines, x=0.985, y=0.50, fontsize=5.4)

	# Leave a right margin for the stats panel.
	plt.tight_layout(rect=[0.0, 0.0, 0.70, 1.0])
	out_path = os.path.join(GROUPED_PLOTS_DIR, filename)
	plt.savefig(out_path, format="svg")
	plt.close(fig)
	print(f"[saved] {out_path}")


def _posthoc_training_pvalues_within_each_sf_and_bin_rm_matrix(metric_key: str, pairs: List[Tuple[str, str]]) -> Dict[int, Dict[int, Optional[float]]]:
	"""Post-hoc simple effects: Training (Pre vs Post) within each SF×bin (matrix metrics).

	Returns nested dict: {sf_idx: {bin_idx: p}} using AnovaRM within=['training'].
	"""
	per_cell_rows: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
	for session_before, session_after in pairs:
		s_before = prepare_session_summaries(session_before)
		s_after = prepare_session_summaries(session_after)
		vb = s_before.get(metric_key)
		va = s_after.get(metric_key)
		if vb is None or va is None:
			continue
		vb = np.asarray(vb, dtype=float)
		va = np.asarray(va, dtype=float)
		if vb.ndim != 2 or va.ndim != 2:
			continue
		n_sf = min(vb.shape[0], va.shape[0], len(TARGET_SF_VALUES))
		n_bins = min(vb.shape[1], va.shape[1])
		if n_sf < 2 or n_bins < 1:
			continue
		subject = _mouse_id_from_session_rel(session_before)
		for sf_idx in range(1, n_sf):
			for bin_idx in range(n_bins):
				b = float(vb[sf_idx, bin_idx])
				a = float(va[sf_idx, bin_idx])
				if np.isfinite(b) and np.isfinite(a):
					per_cell_rows.setdefault((sf_idx, bin_idx), []).extend([
						{"subject": subject, "training": "Pre", "value": b},
						{"subject": subject, "training": "Post", "value": a},
					])

	results: Dict[int, Dict[int, Optional[float]]] = {}
	for (sf_idx, bin_idx), rows in per_cell_rows.items():
		df = pd.DataFrame(rows)
		if df.empty:
			continue
		grid = df.pivot_table(index='subject', columns=['training'], values='value', aggfunc='mean')
		grid = grid.dropna(axis=0, how='any')
		if grid.shape[0] < 2:
			results.setdefault(sf_idx, {})[bin_idx] = None
			continue
		balanced = pd.DataFrame({
			"subject": np.repeat(grid.index.values, 2),
			"training": np.tile(['Pre', 'Post'], grid.shape[0]),
			"value": np.r_[grid['Pre'].values, grid['Post'].values],
		})
		try:
			res = AnovaRM(balanced, depvar='value', subject='subject', within=['training']).fit()
			p = res.anova_table.loc['training', 'Pr > F'] if 'training' in res.anova_table.index else np.nan
			results.setdefault(sf_idx, {})[bin_idx] = float(p) if np.isfinite(p) else None
		except Exception:
			results.setdefault(sf_idx, {})[bin_idx] = None

	return results


def _posthoc_training_pvalues_within_each_sf_and_bin_art_matrix(metric_key: str, pairs: List[Tuple[str, str]]) -> Dict[int, Dict[int, Optional[float]]]:
	"""Post-hoc simple effects: ART-style Training within each SF×bin (matrix metrics).

	Implementation: subject-center values, rank, then AnovaRM within=['training'].
	"""
	per_cell_rows: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
	for session_before, session_after in pairs:
		s_before = prepare_session_summaries(session_before)
		s_after = prepare_session_summaries(session_after)
		vb = s_before.get(metric_key)
		va = s_after.get(metric_key)
		if vb is None or va is None:
			continue
		vb = np.asarray(vb, dtype=float)
		va = np.asarray(va, dtype=float)
		if vb.ndim != 2 or va.ndim != 2:
			continue
		n_sf = min(vb.shape[0], va.shape[0], len(TARGET_SF_VALUES))
		n_bins = min(vb.shape[1], va.shape[1])
		if n_sf < 2 or n_bins < 1:
			continue
		subject = _mouse_id_from_session_rel(session_before)
		for sf_idx in range(1, n_sf):
			for bin_idx in range(n_bins):
				b = float(vb[sf_idx, bin_idx])
				a = float(va[sf_idx, bin_idx])
				if np.isfinite(b) and np.isfinite(a):
					per_cell_rows.setdefault((sf_idx, bin_idx), []).extend([
						{"subject": subject, "training": "Pre", "value": b},
						{"subject": subject, "training": "Post", "value": a},
					])

	results: Dict[int, Dict[int, Optional[float]]] = {}
	for (sf_idx, bin_idx), rows in per_cell_rows.items():
		df = pd.DataFrame(rows)
		if df.empty:
			continue
		grid = df.pivot_table(index='subject', columns=['training'], values='value', aggfunc='mean')
		grid = grid.dropna(axis=0, how='any')
		if grid.shape[0] < 2:
			results.setdefault(sf_idx, {})[bin_idx] = None
			continue
		balanced = pd.DataFrame({
			"subject": np.repeat(grid.index.values, 2),
			"training": np.tile(['Pre', 'Post'], grid.shape[0]),
			"value": np.r_[grid['Pre'].values, grid['Post'].values],
		})
		try:
			balanced['value_centered'] = balanced['value'] - balanced.groupby('subject')['value'].transform('mean')
			balanced['rank'] = balanced['value_centered'].rank(method='average')
			res = AnovaRM(balanced, depvar='rank', subject='subject', within=['training']).fit()
			p = res.anova_table.loc['training', 'Pr > F'] if 'training' in res.anova_table.index else np.nan
			results.setdefault(sf_idx, {})[bin_idx] = float(p) if np.isfinite(p) else None
		except Exception:
			results.setdefault(sf_idx, {})[bin_idx] = None

	return results


def _posthoc_training_pvalues_within_each_sf_and_bin_ttest_matrix(metric_key: str, pairs: List[Tuple[str, str]]) -> Dict[int, Dict[int, Optional[float]]]:
	"""Post-hoc simple effects: paired t-test within each SF×bin (matrix metrics)."""
	per_cell: Dict[Tuple[int, int], List[Tuple[str, float, float]]] = {}
	for session_before, session_after in pairs:
		s_before = prepare_session_summaries(session_before)
		s_after = prepare_session_summaries(session_after)
		vb = s_before.get(metric_key)
		va = s_after.get(metric_key)
		if vb is None or va is None:
			continue
		vb = np.asarray(vb, dtype=float)
		va = np.asarray(va, dtype=float)
		if vb.ndim != 2 or va.ndim != 2:
			continue
		n_sf = min(vb.shape[0], va.shape[0], len(TARGET_SF_VALUES))
		n_bins = min(vb.shape[1], va.shape[1])
		if n_sf < 2 or n_bins < 1:
			continue
		subject = _mouse_id_from_session_rel(session_before)
		for sf_idx in range(1, n_sf):
			for bin_idx in range(n_bins):
				b = float(vb[sf_idx, bin_idx])
				a = float(va[sf_idx, bin_idx])
				if np.isfinite(b) and np.isfinite(a):
					per_cell.setdefault((sf_idx, bin_idx), []).append((subject, b, a))

	results: Dict[int, Dict[int, Optional[float]]] = {}
	for (sf_idx, bin_idx), recs in per_cell.items():
		subjects = sorted({s for s, _, _ in recs})
		if len(subjects) < 2:
			results.setdefault(sf_idx, {})[bin_idx] = None
			continue
		vals_b: List[float] = []
		vals_a: List[float] = []
		for subject in subjects:
			b = next(b for s, b, _ in recs if s == subject)
			a = next(a for s, _, a in recs if s == subject)
			vals_b.append(float(b))
			vals_a.append(float(a))
		try:
			stat = ttest_rel(np.asarray(vals_b, dtype=float), np.asarray(vals_a, dtype=float), nan_policy='omit')
			p = float(stat.pvalue) if stat is not None else np.nan
			results.setdefault(sf_idx, {})[bin_idx] = float(p) if np.isfinite(p) else None
		except Exception:
			results.setdefault(sf_idx, {})[bin_idx] = None

	return results


def _posthoc_training_pvalues_within_each_sf_and_bin_wilcoxon_matrix(metric_key: str, pairs: List[Tuple[str, str]]) -> Dict[int, Dict[int, Optional[float]]]:
	"""Post-hoc simple effects: Wilcoxon signed-rank within each SF×bin (matrix metrics)."""
	per_cell: Dict[Tuple[int, int], List[Tuple[str, float, float]]] = {}
	for session_before, session_after in pairs:
		s_before = prepare_session_summaries(session_before)
		s_after = prepare_session_summaries(session_after)
		vb = s_before.get(metric_key)
		va = s_after.get(metric_key)
		if vb is None or va is None:
			continue
		vb = np.asarray(vb, dtype=float)
		va = np.asarray(va, dtype=float)
		if vb.ndim != 2 or va.ndim != 2:
			continue
		n_sf = min(vb.shape[0], va.shape[0], len(TARGET_SF_VALUES))
		n_bins = min(vb.shape[1], va.shape[1])
		if n_sf < 2 or n_bins < 1:
			continue
		subject = _mouse_id_from_session_rel(session_before)
		for sf_idx in range(1, n_sf):
			for bin_idx in range(n_bins):
				b = float(vb[sf_idx, bin_idx])
				a = float(va[sf_idx, bin_idx])
				if np.isfinite(b) and np.isfinite(a):
					per_cell.setdefault((sf_idx, bin_idx), []).append((subject, b, a))

	results: Dict[int, Dict[int, Optional[float]]] = {}
	for (sf_idx, bin_idx), recs in per_cell.items():
		subjects = sorted({s for s, _, _ in recs})
		if len(subjects) < 2:
			results.setdefault(sf_idx, {})[bin_idx] = None
			continue
		vals_b: List[float] = []
		vals_a: List[float] = []
		for subject in subjects:
			b = next(b for s, b, _ in recs if s == subject)
			a = next(a for s, _, a in recs if s == subject)
			vals_b.append(float(b))
			vals_a.append(float(a))
		try:
			p = _safe_wilcoxon_p(vals_b, vals_a, zero_method='wilcox', alternative='two-sided')
			results.setdefault(sf_idx, {})[bin_idx] = float(p) if (p is not None and np.isfinite(p)) else None
		except Exception:
			results.setdefault(sf_idx, {})[bin_idx] = None

	return results


def plot_orientation_metric(metric_key: str, title: str, ylabel: str, filename: str, ylim: Optional[Tuple[float, float]] = None, chance: Optional[float] = None, blind_as_individual: bool = False, blind_color_override: Optional[str] = None, background_band: Optional[Tuple[float, float]] = None, background_band_color: str = '#d9d9d9', exclude_groups: Optional[List[str]] = None, chance_null_metric_key: Optional[str] = None) -> None:
	"""Plot orientation-vector metrics if orientations are consistent across sessions."""

	orientations, stats = aggregate_group_orientation(metric_key)
	if orientations is None or not stats:
		print(f"[skip] {metric_key} due to missing or inconsistent orientations")
		return
	exclude = set(exclude_groups or [])
	group_plot_order = [
		"WT_BEFORE",
		"WT_AFTER",
		"RESCUED_BEFORE",
		"RESCUED_AFTER",
		"BLIND",
	]
	ordered_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
	for group_name in group_plot_order:
		if group_name in stats and group_name not in exclude:
			ordered_stats[group_name] = stats[group_name]
	for group_name, value in stats.items():
		if group_name not in ordered_stats and group_name not in exclude:
			ordered_stats[group_name] = value
	stats = ordered_stats

	# Check if any data is finite
	has_data = False
	for mean, sem in stats.values():
		if np.isfinite(mean).any():
			has_data = True
			break
	if not has_data:
		print(f"[skip] {metric_key} has no finite data")
		return

	# Wider figure so the right-margin stats panel doesn't shrink the plotting area.
	fig, ax = plt.subplots(figsize=(11, 5))
	markers = ["o", "s", "^", "D", "v"]
	x = orientations
	chance_samples = _get_decoder_null_distribution(chance_null_metric_key) if chance_null_metric_key else None
	if chance_samples is not None:
		cache = _load_or_compute_rescued_pre_shuffled_null_cache()
		chance_orientations = np.asarray(cache['orientations'], dtype=float)
		if np.array_equal(chance_orientations, np.asarray(orientations, dtype=float)):
			low, high = _decoder_null_band(chance_samples)
			ax.fill_between(
				x,
				low,
				high,
				color='#d9d9d9',
				alpha=1.0,
				zorder=0,
			)
		else:
			chance_samples = None
	if background_band is not None:
		band_y0, band_y1 = background_band
		ax.axhspan(float(band_y0), float(band_y1), facecolor=background_band_color, edgecolor='none', zorder=0)
	for idx, (group_name, (mean, sem)) in enumerate(stats.items()):
		if blind_as_individual and group_name == "BLIND":
			continue
		plot_with_error(x, mean, sem, GROUP_LABELS[group_name], GROUP_COLORS[group_name], ax, marker=markers[idx % len(markers)])
	if blind_as_individual:
		_plot_blind_individual_traces_by_orientation(ax, metric_key, x, blind_color=blind_color_override)

	if chance is not None:
		ax.axhline(y=chance, color="k", linestyle="--", linewidth=1, label="chance")
	elif "decoding" in metric_key:
		ax.axhline(y=0.0, color="k", linestyle="--", linewidth=1, alpha=0.7)

	ax.set_xticks(x)
	ax.set_xticklabels([f"{float(v):.0f}" for v in x])
	ax.set_xlabel("Orientation (deg)")
	ax.set_ylabel(ylabel)
	ax.set_title(title)
	if ylim is not None:
		ax.set_ylim(*ylim)
	ax.grid(True, linestyle="--", alpha=0.4)
	ax.legend()

	# Stars: post-hoc Training within each orientation, separately for WT and Rescued
	pairs_wt = MOUSE_PAIRS.get("WT", [])
	pairs_resc = MOUSE_PAIRS.get("RESCUED", [])
	posthoc_rm_wt = _posthoc_training_pvalues_within_each_orientation_rm(metric_key, pairs_wt)
	posthoc_rm_resc = _posthoc_training_pvalues_within_each_orientation_rm(metric_key, pairs_resc)
	posthoc_rm_wt_fdr = _bh_fdr_correct(posthoc_rm_wt)
	posthoc_rm_resc_fdr = _bh_fdr_correct(posthoc_rm_resc)

	# Numeric p/q summary panel (in the figure margin, outside the axes)
	posthoc_art_wt = _posthoc_training_pvalues_within_each_orientation_art(metric_key, pairs_wt)
	posthoc_art_resc = _posthoc_training_pvalues_within_each_orientation_art(metric_key, pairs_resc)
	posthoc_art_wt_fdr = _bh_fdr_correct(posthoc_art_wt)
	posthoc_art_resc_fdr = _bh_fdr_correct(posthoc_art_resc)
	posthoc_t_wt = _posthoc_training_pvalues_within_each_orientation_ttest(metric_key, pairs_wt)
	posthoc_t_resc = _posthoc_training_pvalues_within_each_orientation_ttest(metric_key, pairs_resc)
	posthoc_t_wt_fdr = _bh_fdr_correct(posthoc_t_wt)
	posthoc_t_resc_fdr = _bh_fdr_correct(posthoc_t_resc)
	posthoc_w_wt = _posthoc_training_pvalues_within_each_orientation_wilcoxon(metric_key, pairs_wt)
	posthoc_w_resc = _posthoc_training_pvalues_within_each_orientation_wilcoxon(metric_key, pairs_resc)
	posthoc_w_wt_fdr = _bh_fdr_correct(posthoc_w_wt)
	posthoc_w_resc_fdr = _bh_fdr_correct(posthoc_w_resc)

	def _fmt_val(v: Optional[float]) -> str:
		if v is None or not np.isfinite(v):
			return "NA"
		vv = float(v)
		return f"{vv:.3f}" + ("*" if vv < 0.05 else "")

	def _fmt_pq(p: Optional[float], q: Optional[float]) -> str:
		return f"{_fmt_val(p)}/{_fmt_val(q)}"

	# Omnibus (two-way) test results: move these out of the axes into the margin panel.
	pvals = compute_paired_pvalues(metric_key)
	rmanova_pvals = compute_rmanova_pvalues_orientation(metric_key)
	art_pvals = compute_art_pvalues_orientation(metric_key)
	art_raov_pvals = compute_art_raov_pvalues_orientation(metric_key)
	art_rlmer_pvals = compute_art_r_pvalues_orientation(metric_key)
	lme_pvals = compute_lme_pvalues_orientation(metric_key)
	lme4_pvals = compute_lme4_pvalues_orientation(metric_key)

	panel_lines: List[str] = [
		"Omnibus tests (p) (* means <0.05)",
	]
	if RM_ANOVA_INCLUDE_INTERACTION_TERMS:
		panel_lines.append(
			(
				"WT: paired={paired}  RM train={rm_tr}  RM train×Ori={rm_int}"
			).format(
				paired=_fmt_val(pvals.get("WT")),
				rm_tr=_fmt_val(rmanova_pvals.get("WT", {}).get("training_p")),
				rm_int=_fmt_val(rmanova_pvals.get("WT", {}).get("interaction_p")),
			)
		)
		panel_lines.append(
			(
				"Resc: paired={paired}  RM train={rm_tr}  RM train×Ori={rm_int}"
			).format(
				paired=_fmt_val(pvals.get("RESCUED")),
				rm_tr=_fmt_val(rmanova_pvals.get("RESCUED", {}).get("training_p")),
				rm_int=_fmt_val(rmanova_pvals.get("RESCUED", {}).get("interaction_p")),
			)
		)
	else:
		panel_lines.append(
			(
				"WT: paired={paired}  RM train={rm_tr}"
			).format(
				paired=_fmt_val(pvals.get("WT")),
				rm_tr=_fmt_val(rmanova_pvals.get("WT", {}).get("training_p")),
			)
		)
		panel_lines.append(
			(
				"Resc: paired={paired}  RM train={rm_tr}"
			).format(
				paired=_fmt_val(pvals.get("RESCUED")),
				rm_tr=_fmt_val(rmanova_pvals.get("RESCUED", {}).get("training_p")),
			)
		)

	if ART_INCLUDE_INTERACTION_TERMS:
		panel_lines.append(
			(
				"WT: ART train={art_tr}  ART Ori={art_ori}  ART int={art_int}"
			).format(
				art_tr=_fmt_val(art_pvals.get("WT", {}).get("training_p")),
				art_ori=_fmt_val(art_pvals.get("WT", {}).get("orientation_p")),
				art_int=_fmt_val(art_pvals.get("WT", {}).get("interaction_p")),
			)
		)
		panel_lines.append(
			(
				"Resc: ART train={art_tr}  ART Ori={art_ori}  ART int={art_int}"
			).format(
				art_tr=_fmt_val(art_pvals.get("RESCUED", {}).get("training_p")),
				art_ori=_fmt_val(art_pvals.get("RESCUED", {}).get("orientation_p")),
				art_int=_fmt_val(art_pvals.get("RESCUED", {}).get("interaction_p")),
			)
		)
		panel_lines.append(
			(
				"WT: ART-Raov train={art_tr}  ART-Raov Ori={art_ori}  ART-Raov int={art_int}"
			).format(
				art_tr=_fmt_val(art_raov_pvals.get("WT", {}).get("training_p")),
				art_ori=_fmt_val(art_raov_pvals.get("WT", {}).get("orientation_p")),
				art_int=_fmt_val(art_raov_pvals.get("WT", {}).get("interaction_p")),
			)
		)
		panel_lines.append(
			(
				"Resc: ART-Raov train={art_tr}  ART-Raov Ori={art_ori}  ART-Raov int={art_int}"
			).format(
				art_tr=_fmt_val(art_raov_pvals.get("RESCUED", {}).get("training_p")),
				art_ori=_fmt_val(art_raov_pvals.get("RESCUED", {}).get("orientation_p")),
				art_int=_fmt_val(art_raov_pvals.get("RESCUED", {}).get("interaction_p")),
			)
		)
		panel_lines.append(
			(
				"WT: ART-Rlmer train={art_tr}  ART-Rlmer Ori={art_ori}  ART-Rlmer int={art_int}"
			).format(
				art_tr=_fmt_val(art_rlmer_pvals.get("WT", {}).get("training_p")),
				art_ori=_fmt_val(art_rlmer_pvals.get("WT", {}).get("orientation_p")),
				art_int=_fmt_val(art_rlmer_pvals.get("WT", {}).get("interaction_p")),
			)
		)
		panel_lines.append(
			(
				"Resc: ART-Rlmer train={art_tr}  ART-Rlmer Ori={art_ori}  ART-Rlmer int={art_int}"
			).format(
				art_tr=_fmt_val(art_rlmer_pvals.get("RESCUED", {}).get("training_p")),
				art_ori=_fmt_val(art_rlmer_pvals.get("RESCUED", {}).get("orientation_p")),
				art_int=_fmt_val(art_rlmer_pvals.get("RESCUED", {}).get("interaction_p")),
			)
		)
	else:
		panel_lines.append(
			(
				"WT: ART train={art_tr}  ART Ori={art_ori}"
			).format(
				art_tr=_fmt_val(art_pvals.get("WT", {}).get("training_p")),
				art_ori=_fmt_val(art_pvals.get("WT", {}).get("orientation_p")),
			)
		)
		panel_lines.append(
			(
				"Resc: ART train={art_tr}  ART Ori={art_ori}"
			).format(
				art_tr=_fmt_val(art_pvals.get("RESCUED", {}).get("training_p")),
				art_ori=_fmt_val(art_pvals.get("RESCUED", {}).get("orientation_p")),
			)
		)
		panel_lines.append(
			(
				"WT: ART-Raov train={art_tr}  ART-Raov Ori={art_ori}"
			).format(
				art_tr=_fmt_val(art_raov_pvals.get("WT", {}).get("training_p")),
				art_ori=_fmt_val(art_raov_pvals.get("WT", {}).get("orientation_p")),
			)
		)
		panel_lines.append(
			(
				"Resc: ART-Raov train={art_tr}  ART-Raov Ori={art_ori}"
			).format(
				art_tr=_fmt_val(art_raov_pvals.get("RESCUED", {}).get("training_p")),
				art_ori=_fmt_val(art_raov_pvals.get("RESCUED", {}).get("orientation_p")),
			)
		)
		panel_lines.append(
			(
				"WT: ART-Rlmer train={art_tr}  ART-Rlmer Ori={art_ori}"
			).format(
				art_tr=_fmt_val(art_rlmer_pvals.get("WT", {}).get("training_p")),
				art_ori=_fmt_val(art_rlmer_pvals.get("WT", {}).get("orientation_p")),
			)
		)
		panel_lines.append(
			(
				"Resc: ART-Rlmer train={art_tr}  ART-Rlmer Ori={art_ori}"
			).format(
				art_tr=_fmt_val(art_rlmer_pvals.get("RESCUED", {}).get("training_p")),
				art_ori=_fmt_val(art_rlmer_pvals.get("RESCUED", {}).get("orientation_p")),
			)
		)

	# LME (MixedLM) omnibus p-values (LRT)
	if LME_INCLUDE_INTERACTION_TERMS:
		panel_lines.append(
			(
				"WT: LME train={lme_tr}  LME Ori={lme_ori}  LME int={lme_int}"
			).format(
				lme_tr=_fmt_val(lme_pvals.get("WT", {}).get("training_p")),
				lme_ori=_fmt_val(lme_pvals.get("WT", {}).get("orientation_p")),
				lme_int=_fmt_val(lme_pvals.get("WT", {}).get("interaction_p")),
			)
		)
		panel_lines.append(
			(
				"Resc: LME train={lme_tr}  LME Ori={lme_ori}  LME int={lme_int}"
			).format(
				lme_tr=_fmt_val(lme_pvals.get("RESCUED", {}).get("training_p")),
				lme_ori=_fmt_val(lme_pvals.get("RESCUED", {}).get("orientation_p")),
				lme_int=_fmt_val(lme_pvals.get("RESCUED", {}).get("interaction_p")),
			)
		)
		panel_lines.append(
			(
				"WT: LME-lme4 train={lme_tr}  LME-lme4 Ori={lme_ori}  LME-lme4 int={lme_int}"
			).format(
				lme_tr=_fmt_val(lme4_pvals.get("WT", {}).get("training_p")),
				lme_ori=_fmt_val(lme4_pvals.get("WT", {}).get("orientation_p")),
				lme_int=_fmt_val(lme4_pvals.get("WT", {}).get("interaction_p")),
			)
		)
		panel_lines.append(
			(
				"Resc: LME-lme4 train={lme_tr}  LME-lme4 Ori={lme_ori}  LME-lme4 int={lme_int}"
			).format(
				lme_tr=_fmt_val(lme4_pvals.get("RESCUED", {}).get("training_p")),
				lme_ori=_fmt_val(lme4_pvals.get("RESCUED", {}).get("orientation_p")),
				lme_int=_fmt_val(lme4_pvals.get("RESCUED", {}).get("interaction_p")),
			)
		)
	else:
		panel_lines.append(
			(
				"WT: LME train={lme_tr}  LME Ori={lme_ori}"
			).format(
				lme_tr=_fmt_val(lme_pvals.get("WT", {}).get("training_p")),
				lme_ori=_fmt_val(lme_pvals.get("WT", {}).get("orientation_p")),
			)
		)
		panel_lines.append(
			(
				"Resc: LME train={lme_tr}  LME Ori={lme_ori}"
			).format(
				lme_tr=_fmt_val(lme_pvals.get("RESCUED", {}).get("training_p")),
				lme_ori=_fmt_val(lme_pvals.get("RESCUED", {}).get("orientation_p")),
			)
		)
		panel_lines.append(
			(
				"WT: LME-lme4 train={lme_tr}  LME-lme4 Ori={lme_ori}"
			).format(
				lme_tr=_fmt_val(lme4_pvals.get("WT", {}).get("training_p")),
				lme_ori=_fmt_val(lme4_pvals.get("WT", {}).get("orientation_p")),
			)
		)
		panel_lines.append(
			(
				"Resc: LME-lme4 train={lme_tr}  LME-lme4 Ori={lme_ori}"
			).format(
				lme_tr=_fmt_val(lme4_pvals.get("RESCUED", {}).get("training_p")),
				lme_ori=_fmt_val(lme4_pvals.get("RESCUED", {}).get("orientation_p")),
			)
		)

	if chance_samples is not None:
		panel_lines.extend([
			"",
			"Above shuffled chance@Ori: sig q<0.05",
		])
		for group_name in ["WT_BEFORE", "WT_AFTER", "RESCUED_BEFORE", "RESCUED_AFTER"]:
			if group_name not in stats:
				continue
			obs_mean = np.asarray(stats[group_name][0], dtype=float)
			p_emp = _empirical_above_chance_pvalues(obs_mean, chance_samples)
			q_emp = _bh_fdr_correct(p_emp)
			panel_lines.append(f"{GROUP_LABELS.get(group_name, group_name)}: {_format_sig_orientation_from_q(orientations, q_emp)}")

	if metric_key == "responsive_fraction_by_orientation_non064" and filename == "responsive_fraction_by_orientation_excluding_064_groups.svg":
		wt_pre_sessions = [s_pre for s_pre, _ in MOUSE_PAIRS.get("WT", [])]
		wt_post_sessions = [s_post for _, s_post in MOUSE_PAIRS.get("WT", [])]
		resc_post_sessions = [s_post for _, s_post in MOUSE_PAIRS.get("RESCUED", [])]
		panel_lines.extend([
			"",
			*_build_cross_group_art_rlmer_orientation_stats_lines(
				group_a=_collect_session_metric_matrix_by_orientation(wt_pre_sessions, metric_key),
				group_b=_collect_session_metric_matrix_by_orientation(resc_post_sessions, metric_key),
				orientations=orientations,
				label_a="WT pre",
				label_b="Rescued post",
			),
			"",
			*_build_cross_group_art_rlmer_orientation_stats_lines(
				group_a=_collect_session_metric_matrix_by_orientation(wt_post_sessions, metric_key),
				group_b=_collect_session_metric_matrix_by_orientation(resc_post_sessions, metric_key),
				orientations=orientations,
				label_a="WT post",
				label_b="Rescued post",
			),
		])

	panel_lines.extend([
		"",
		"Posthoc train@Ori: p/q (BH)  (* means <0.05)",
		"WT: RM  ART  t  W   (each is p/q)",
		"Resc: RM  ART  t  W",
	])
	for oi in range(len(orientations)):
		ori_val = float(orientations[oi])
		wt = "  ".join([
			_fmt_pq(posthoc_rm_wt.get(oi), posthoc_rm_wt_fdr.get(oi)),
			_fmt_pq(posthoc_art_wt.get(oi), posthoc_art_wt_fdr.get(oi)),
			_fmt_pq(posthoc_t_wt.get(oi), posthoc_t_wt_fdr.get(oi)),
			_fmt_pq(posthoc_w_wt.get(oi), posthoc_w_wt_fdr.get(oi)),
		])
		re = "  ".join([
			_fmt_pq(posthoc_rm_resc.get(oi), posthoc_rm_resc_fdr.get(oi)),
			_fmt_pq(posthoc_art_resc.get(oi), posthoc_art_resc_fdr.get(oi)),
			_fmt_pq(posthoc_t_resc.get(oi), posthoc_t_resc_fdr.get(oi)),
			_fmt_pq(posthoc_w_resc.get(oi), posthoc_w_resc_fdr.get(oi)),
		])
		panel_lines.append(f"Ori={ori_val:.0f}° WT  : {wt}")
		panel_lines.append(f"     Resc: {re}")

	panel_right = 0.67
	panel_fontsize = 6.8
	panel_lines_to_draw = panel_lines
	panel_line_sep = 1.0
	panel_y = 0.50
	if metric_key == "responsive_fraction_by_orientation_non064" and filename == "responsive_fraction_by_orientation_excluding_064_groups.svg":
		panel_right = 0.58
		panel_fontsize = 4.8
		panel_lines_to_draw = _wrap_stats_panel_lines(panel_lines, max_chars=72)
		panel_line_sep = 0.2
		panel_y = 0.985
	fig.subplots_adjust(right=panel_right)
	_add_colored_stats_panel_to_figure(fig, panel_lines_to_draw, x=0.985, y=panel_y, fontsize=panel_fontsize, line_sep=panel_line_sep)
	star_transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
	if len(x) >= 2 and np.isfinite(x[1] - x[0]):
		dx = float(x[1] - x[0]) * 0.12
	else:
		dx = 2.0
	for oi in range(len(x)):
		x_i = float(x[oi])
		x_wt = x_i - dx
		x_resc = x_i + dx
		p_wt = posthoc_rm_wt.get(oi, None)
		q_wt = posthoc_rm_wt_fdr.get(oi, None)
		p_resc = posthoc_rm_resc.get(oi, None)
		q_resc = posthoc_rm_resc_fdr.get(oi, None)
		# Uncorrected (higher)
		if p_wt is not None and np.isfinite(p_wt) and p_wt < 0.05:
			ax.text(x_wt, 0.96, _asterisks_from_p(float(p_wt)), transform=star_transform,
				ha='center', va='top', fontsize=10, fontweight='bold')
		if p_resc is not None and np.isfinite(p_resc) and p_resc < 0.05:
			ax.text(x_resc, 0.96, _asterisks_from_p(float(p_resc)), transform=star_transform,
				ha='center', va='top', fontsize=10, fontweight='bold')
		# FDR(BH) (lower)
		if q_wt is not None and np.isfinite(q_wt) and q_wt < 0.05:
			ax.text(x_wt, 0.91, _asterisks_from_p(float(q_wt)), transform=star_transform,
				ha='center', va='top', fontsize=9, fontweight='bold')
		if q_resc is not None and np.isfinite(q_resc) and q_resc < 0.05:
			ax.text(x_resc, 0.91, _asterisks_from_p(float(q_resc)), transform=star_transform,
				ha='center', va='top', fontsize=9, fontweight='bold')
	
	# Keep space for the right-margin panel.
	fig.tight_layout(rect=(0.0, 0.0, panel_right, 1.0))
	out_path = os.path.join(GROUPED_PLOTS_DIR, filename)
	plt.savefig(out_path, format="svg")
	plt.close(fig)
	print(f"[saved] {out_path}")


def _training_pvalues_within_each_sf_for_matrix_metric_rm(metric_key: str, pairs: List[Tuple[str, str]]) -> Dict[int, Optional[float]]:
	"""Per-SF Training p-values for matrix metrics using RM-ANOVA within each SF.

	Within each SF, uses a two-way within-subject model: training × bin, and returns p(training).
	"""
	per_sf: Dict[int, List[Tuple[str, np.ndarray, np.ndarray]]] = {}
	for session_before, session_after in pairs:
		s_before = prepare_session_summaries(session_before)
		s_after = prepare_session_summaries(session_after)
		vb = s_before.get(metric_key)
		va = s_after.get(metric_key)
		if vb is None or va is None:
			continue
		vb = np.asarray(vb, dtype=float)
		va = np.asarray(va, dtype=float)
		if vb.ndim != 2 or va.ndim != 2:
			continue
		n_sf = min(vb.shape[0], va.shape[0], len(TARGET_SF_VALUES))
		if n_sf < 1:
			continue
		subject = _mouse_id_from_session_rel(session_before)
		for sf_idx in range(n_sf):
			per_sf.setdefault(sf_idx, []).append((subject, vb[sf_idx, :], va[sf_idx, :]))

	results: Dict[int, Optional[float]] = {}
	for sf_idx, records in per_sf.items():
		# Determine bins available for all subjects with finite values in both trainings.
		subjects = sorted({s for s, _, _ in records})
		if len(subjects) < 2:
			results[sf_idx] = None
			continue
		by_subj: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
		min_bins = None
		for subject, vb, va in records:
			vb = np.asarray(vb, dtype=float).ravel()
			va = np.asarray(va, dtype=float).ravel()
			b = min(vb.size, va.size)
			min_bins = b if min_bins is None else min(min_bins, b)
			by_subj[subject] = (vb, va)
		if min_bins is None or min_bins < 2:
			results[sf_idx] = None
			continue

		good_bins: List[int] = []
		for bi in range(int(min_bins)):
			ok = True
			for subject in subjects:
				vb, va = by_subj[subject]
				if not (np.isfinite(vb[bi]) and np.isfinite(va[bi])):
					ok = False
					break
			if ok:
				good_bins.append(bi)
		if len(good_bins) < 2:
			results[sf_idx] = None
			continue

		rows: List[Dict[str, object]] = []
		for subject in subjects:
			vb, va = by_subj[subject]
			for bi in good_bins:
				rows.append({"subject": subject, "training": "Pre", "bin": str(int(bi)), "value": float(vb[bi])})
				rows.append({"subject": subject, "training": "Post", "bin": str(int(bi)), "value": float(va[bi])})
		df = pd.DataFrame(rows)
		try:
			res = AnovaRM(df, depvar='value', subject='subject', within=['training', 'bin']).fit()
			p = res.anova_table.loc['training', 'Pr > F'] if 'training' in res.anova_table.index else np.nan
			results[sf_idx] = float(p) if np.isfinite(p) else None
		except Exception:
			results[sf_idx] = None
	return results


def _training_pvalues_within_each_sf_for_matrix_metric_art(metric_key: str, pairs: List[Tuple[str, str]]) -> Dict[int, Optional[float]]:
	"""Per-SF Training p-values for matrix metrics using ART within each SF."""
	# Reuse RM collection and balancing logic by reconstructing long df per SF.
	per_sf: Dict[int, List[Tuple[str, np.ndarray, np.ndarray]]] = {}
	for session_before, session_after in pairs:
		s_before = prepare_session_summaries(session_before)
		s_after = prepare_session_summaries(session_after)
		vb = s_before.get(metric_key)
		va = s_after.get(metric_key)
		if vb is None or va is None:
			continue
		vb = np.asarray(vb, dtype=float)
		va = np.asarray(va, dtype=float)
		if vb.ndim != 2 or va.ndim != 2:
			continue
		n_sf = min(vb.shape[0], va.shape[0], len(TARGET_SF_VALUES))
		if n_sf < 1:
			continue
		subject = _mouse_id_from_session_rel(session_before)
		for sf_idx in range(n_sf):
			per_sf.setdefault(sf_idx, []).append((subject, vb[sf_idx, :], va[sf_idx, :]))

	results: Dict[int, Optional[float]] = {}
	for sf_idx, records in per_sf.items():
		subjects = sorted({s for s, _, _ in records})
		if len(subjects) < 2:
			results[sf_idx] = None
			continue
		by_subj: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
		min_bins = None
		for subject, vb, va in records:
			vb = np.asarray(vb, dtype=float).ravel()
			va = np.asarray(va, dtype=float).ravel()
			b = min(vb.size, va.size)
			min_bins = b if min_bins is None else min(min_bins, b)
			by_subj[subject] = (vb, va)
		if min_bins is None or min_bins < 2:
			results[sf_idx] = None
			continue

		good_bins: List[int] = []
		for bi in range(int(min_bins)):
			ok = True
			for subject in subjects:
				vb, va = by_subj[subject]
				if not (np.isfinite(vb[bi]) and np.isfinite(va[bi])):
					ok = False
					break
			if ok:
				good_bins.append(bi)
		if len(good_bins) < 2:
			results[sf_idx] = None
			continue

		rows: List[Dict[str, object]] = []
		for subject in subjects:
			vb, va = by_subj[subject]
			for bi in good_bins:
				rows.append({"subject": subject, "training": "Pre", "bin": str(int(bi)), "value": float(vb[bi])})
				rows.append({"subject": subject, "training": "Post", "bin": str(int(bi)), "value": float(va[bi])})
		df = pd.DataFrame(rows)
		try:
			# Center within subject, ART align+rank for training effect while keeping bin factor.
			df['value_centered'] = df['value'] - df.groupby('subject')['value'].transform('mean')
			df['training'] = df['training'].astype('category')
			df['bin'] = df['bin'].astype('category')
			art = _aligned_rank_transform(df, factor_a='training', factor_b='bin', subject='subject', value_col='value_centered')
			df['rank_A'] = art['A']
			res = AnovaRM(df, depvar='rank_A', subject='subject', within=['training', 'bin']).fit()
			p = res.anova_table.loc['training', 'Pr > F'] if 'training' in res.anova_table.index else np.nan
			results[sf_idx] = float(p) if np.isfinite(p) else None
		except Exception:
			results[sf_idx] = None
	return results


def _training_pvalues_within_each_sf_for_matrix_metric_ttest(metric_key: str, pairs: List[Tuple[str, str]]) -> Dict[int, Optional[float]]:
	"""Per-SF paired t-test p-values on subject mean across bins."""
	per_sf: Dict[int, List[Tuple[str, np.ndarray, np.ndarray]]] = {}
	for session_before, session_after in pairs:
		s_before = prepare_session_summaries(session_before)
		s_after = prepare_session_summaries(session_after)
		vb = s_before.get(metric_key)
		va = s_after.get(metric_key)
		if vb is None or va is None:
			continue
		vb = np.asarray(vb, dtype=float)
		va = np.asarray(va, dtype=float)
		if vb.ndim != 2 or va.ndim != 2:
			continue
		n_sf = min(vb.shape[0], va.shape[0], len(TARGET_SF_VALUES))
		subject = _mouse_id_from_session_rel(session_before)
		for sf_idx in range(n_sf):
			per_sf.setdefault(sf_idx, []).append((subject, vb[sf_idx, :], va[sf_idx, :]))

	results: Dict[int, Optional[float]] = {}
	for sf_idx, records in per_sf.items():
		subjects = sorted({s for s, _, _ in records})
		if len(subjects) < 2:
			results[sf_idx] = None
			continue
		vals_b: List[float] = []
		vals_a: List[float] = []
		for subject in subjects:
			# Take the first occurrence per subject.
			vb = next(vb for s, vb, _ in records if s == subject)
			va = next(va for s, _, va in records if s == subject)
			vb = np.asarray(vb, dtype=float).ravel()
			va = np.asarray(va, dtype=float).ravel()
			b = min(vb.size, va.size)
			if b < 1:
				continue
			vb = vb[:b]
			va = va[:b]
			mask = np.isfinite(vb) & np.isfinite(va)
			if mask.sum() < 1:
				continue
			vals_b.append(float(np.nanmean(vb[mask])))
			vals_a.append(float(np.nanmean(va[mask])))
		if len(vals_b) < 2:
			results[sf_idx] = None
			continue
		try:
			p = float(ttest_rel(vals_b, vals_a, nan_policy='omit').pvalue)
			results[sf_idx] = p if np.isfinite(p) else None
		except Exception:
			results[sf_idx] = None
	return results


def _training_pvalues_within_each_sf_for_matrix_metric_wilcoxon(metric_key: str, pairs: List[Tuple[str, str]]) -> Dict[int, Optional[float]]:
	"""Per-SF Wilcoxon signed-rank p-values on subject mean across bins."""
	per_sf: Dict[int, List[Tuple[str, np.ndarray, np.ndarray]]] = {}
	for session_before, session_after in pairs:
		s_before = prepare_session_summaries(session_before)
		s_after = prepare_session_summaries(session_after)
		vb = s_before.get(metric_key)
		va = s_after.get(metric_key)
		if vb is None or va is None:
			continue
		vb = np.asarray(vb, dtype=float)
		va = np.asarray(va, dtype=float)
		if vb.ndim != 2 or va.ndim != 2:
			continue
		n_sf = min(vb.shape[0], va.shape[0], len(TARGET_SF_VALUES))
		subject = _mouse_id_from_session_rel(session_before)
		for sf_idx in range(n_sf):
			per_sf.setdefault(sf_idx, []).append((subject, vb[sf_idx, :], va[sf_idx, :]))

	results: Dict[int, Optional[float]] = {}
	for sf_idx, records in per_sf.items():
		subjects = sorted({s for s, _, _ in records})
		if len(subjects) < 2:
			results[sf_idx] = None
			continue
		vals_b: List[float] = []
		vals_a: List[float] = []
		for subject in subjects:
			vb = next(vb for s, vb, _ in records if s == subject)
			va = next(va for s, _, va in records if s == subject)
			vb = np.asarray(vb, dtype=float).ravel()
			va = np.asarray(va, dtype=float).ravel()
			b = min(vb.size, va.size)
			if b < 1:
				continue
			vb = vb[:b]
			va = va[:b]
			mask = np.isfinite(vb) & np.isfinite(va)
			if mask.sum() < 1:
				continue
			vals_b.append(float(np.nanmean(vb[mask])))
			vals_a.append(float(np.nanmean(va[mask])))
		if len(vals_b) < 2:
			results[sf_idx] = None
			continue
		try:
			p = _safe_wilcoxon_p(vals_b, vals_a)
			results[sf_idx] = float(p) if (p is not None and np.isfinite(p)) else None
		except Exception:
			results[sf_idx] = None
	return results


def plot_per_sf_bins(metric_key: str, x_values: List[float], x_labels: List[str], title: str, ylabel: str, filename: str, chance: Optional[float] = None, x_axis_label: str = "Relative orientation (deg)", *, add_stats_boxes: bool = False) -> None:
	"""Plot per-SF subplots with categorical/ordered bins on x-axis."""

	stats = aggregate_group_matrix(metric_key)
	if not stats:
		print(f"[skip] No data for {metric_key}")
		return

	first_group = next(iter(stats.values()))
	n_sf, n_bins = first_group[0].shape
	if n_bins == 0:
		print(f"[skip] {metric_key} has zero bins")
		return

	# Check if any data is finite
	has_data = False
	for mean, sem in stats.values():
		if np.isfinite(mean).any():
			has_data = True
			break
	if not has_data:
		print(f"[skip] {metric_key} has no finite data")
		return

	# Get target SFs
	sf_indices = [i for i in range(len(TARGET_SF_VALUES))]
	n_cols = 3
	n_rows = (len(sf_indices) + n_cols - 1) // n_cols
	fig_width = 15
	if add_stats_boxes and metric_key == "rel_frac":
		# Extra width for the right-margin bin-level posthoc panel.
		fig_width = 20
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, 5*n_rows))
	if n_rows == 1 and n_cols == 1:
		axes = np.array([axes])
	else:
		axes = axes.flatten()

	# Optional per-SF stats boxes (Training pre vs post within each SF), shown per subplot.
	if add_stats_boxes:
		pairs_wt = MOUSE_PAIRS.get("WT", [])
		pairs_resc = MOUSE_PAIRS.get("RESCUED", [])
		p_rm_wt = _training_pvalues_within_each_sf_for_matrix_metric_rm(metric_key, pairs_wt)
		p_rm_re = _training_pvalues_within_each_sf_for_matrix_metric_rm(metric_key, pairs_resc)
		p_art_wt = _training_pvalues_within_each_sf_for_matrix_metric_art(metric_key, pairs_wt)
		p_art_re = _training_pvalues_within_each_sf_for_matrix_metric_art(metric_key, pairs_resc)
		p_t_wt = _training_pvalues_within_each_sf_for_matrix_metric_ttest(metric_key, pairs_wt)
		p_t_re = _training_pvalues_within_each_sf_for_matrix_metric_ttest(metric_key, pairs_resc)
		p_w_wt = _training_pvalues_within_each_sf_for_matrix_metric_wilcoxon(metric_key, pairs_wt)
		p_w_re = _training_pvalues_within_each_sf_for_matrix_metric_wilcoxon(metric_key, pairs_resc)

		q_rm_wt = _bh_fdr_correct(p_rm_wt)
		q_rm_re = _bh_fdr_correct(p_rm_re)
		q_art_wt = _bh_fdr_correct(p_art_wt)
		q_art_re = _bh_fdr_correct(p_art_re)
		q_t_wt = _bh_fdr_correct(p_t_wt)
		q_t_re = _bh_fdr_correct(p_t_re)
		q_w_wt = _bh_fdr_correct(p_w_wt)
		q_w_re = _bh_fdr_correct(p_w_re)

		def _fmt_p(v: Optional[float]) -> str:
			if v is None or not np.isfinite(v):
				return "NA"
			vv = float(v)
			return f"{vv:.3f}" + ("*" if vv < 0.05 else "")

		def _fmt_pq(p: Optional[float], q: Optional[float]) -> str:
			return f"{_fmt_p(p)}/{_fmt_p(q)}"

		# Extra: for rel_frac only, compute posthoc Training within each SF×bin.
		if metric_key == "rel_frac":
			p_bins_rm_wt = _posthoc_training_pvalues_within_each_sf_and_bin_rm_matrix(metric_key, pairs_wt)
			p_bins_rm_re = _posthoc_training_pvalues_within_each_sf_and_bin_rm_matrix(metric_key, pairs_resc)
			p_bins_art_wt = _posthoc_training_pvalues_within_each_sf_and_bin_art_matrix(metric_key, pairs_wt)
			p_bins_art_re = _posthoc_training_pvalues_within_each_sf_and_bin_art_matrix(metric_key, pairs_resc)
			p_bins_t_wt = _posthoc_training_pvalues_within_each_sf_and_bin_ttest_matrix(metric_key, pairs_wt)
			p_bins_t_re = _posthoc_training_pvalues_within_each_sf_and_bin_ttest_matrix(metric_key, pairs_resc)
			p_bins_w_wt = _posthoc_training_pvalues_within_each_sf_and_bin_wilcoxon_matrix(metric_key, pairs_wt)
			p_bins_w_re = _posthoc_training_pvalues_within_each_sf_and_bin_wilcoxon_matrix(metric_key, pairs_resc)

			def _bh_fdr_nested(p_nested: Dict[int, Dict[int, Optional[float]]]) -> Dict[int, Dict[int, Optional[float]]]:
				out: Dict[int, Dict[int, Optional[float]]] = {}
				for sf_i, by_bin in p_nested.items():
					out[sf_i] = _bh_fdr_correct(by_bin)
				return out

			q_bins_rm_wt = _bh_fdr_nested(p_bins_rm_wt)
			q_bins_rm_re = _bh_fdr_nested(p_bins_rm_re)
			q_bins_art_wt = _bh_fdr_nested(p_bins_art_wt)
			q_bins_art_re = _bh_fdr_nested(p_bins_art_re)
			q_bins_t_wt = _bh_fdr_nested(p_bins_t_wt)
			q_bins_t_re = _bh_fdr_nested(p_bins_t_re)
			q_bins_w_wt = _bh_fdr_nested(p_bins_w_wt)
			q_bins_w_re = _bh_fdr_nested(p_bins_w_re)

			# Also compute BH-FDR across ALL SF×bin cells (global family), per group+test.
			def _bh_fdr_nested_global(p_nested: Dict[int, Dict[int, Optional[float]]], *, sf_start_idx: int = 1) -> Dict[int, Dict[int, Optional[float]]]:
				flat: Dict[Tuple[int, int], Optional[float]] = {}
				for sf_i, by_bin in p_nested.items():
					if sf_i < sf_start_idx:
						continue
					for bi, pv in by_bin.items():
						flat[(sf_i, bi)] = pv
				q_flat = _bh_fdr_correct(flat)
				out: Dict[int, Dict[int, Optional[float]]] = {}
				for (sf_i, bi), qv in q_flat.items():
					out.setdefault(sf_i, {})[bi] = qv
				return out

			q_bins_rm_wt_all = _bh_fdr_nested_global(p_bins_rm_wt)
			q_bins_rm_re_all = _bh_fdr_nested_global(p_bins_rm_re)
			q_bins_art_wt_all = _bh_fdr_nested_global(p_bins_art_wt)
			q_bins_art_re_all = _bh_fdr_nested_global(p_bins_art_re)
			q_bins_t_wt_all = _bh_fdr_nested_global(p_bins_t_wt)
			q_bins_t_re_all = _bh_fdr_nested_global(p_bins_t_re)
			q_bins_w_wt_all = _bh_fdr_nested_global(p_bins_w_wt)
			q_bins_w_re_all = _bh_fdr_nested_global(p_bins_w_re)
		else:
			p_bins_rm_wt = p_bins_rm_re = p_bins_art_wt = p_bins_art_re = p_bins_t_wt = p_bins_t_re = p_bins_w_wt = p_bins_w_re = {}
			q_bins_rm_wt = q_bins_rm_re = q_bins_art_wt = q_bins_art_re = q_bins_t_wt = q_bins_t_re = q_bins_w_wt = q_bins_w_re = {}
			q_bins_rm_wt_all = q_bins_rm_re_all = q_bins_art_wt_all = q_bins_art_re_all = q_bins_t_wt_all = q_bins_t_re_all = q_bins_w_wt_all = q_bins_w_re_all = {}
	else:
		p_rm_wt = p_rm_re = p_art_wt = p_art_re = p_t_wt = p_t_re = p_w_wt = p_w_re = {}
		q_rm_wt = q_rm_re = q_art_wt = q_art_re = q_t_wt = q_t_re = q_w_wt = q_w_re = {}
		_fmt_p = None  # type: ignore
		_fmt_pq = None  # type: ignore
		p_bins_rm_wt = p_bins_rm_re = p_bins_art_wt = p_bins_art_re = p_bins_t_wt = p_bins_t_re = p_bins_w_wt = p_bins_w_re = {}
		q_bins_rm_wt = q_bins_rm_re = q_bins_art_wt = q_bins_art_re = q_bins_t_wt = q_bins_t_re = q_bins_w_wt = q_bins_w_re = {}
		q_bins_rm_wt_all = q_bins_rm_re_all = q_bins_art_wt_all = q_bins_art_re_all = q_bins_t_wt_all = q_bins_t_re_all = q_bins_w_wt_all = q_bins_w_re_all = {}

	for plot_idx, sf_idx in enumerate(sf_indices):
		ax = axes[plot_idx]
		sf = TARGET_SF_VALUES[sf_idx]
		markers = ["o", "s", "^", "D", "v"]
		for gidx, (group_name, (mean, sem)) in enumerate(stats.items()):
			if mean.ndim == 2 and sf_idx < mean.shape[0]:
				plot_with_error(x_values, mean[sf_idx, :], sem[sf_idx, :], GROUP_LABELS[group_name], GROUP_COLORS[group_name], ax, marker=markers[gidx % len(markers)])

		if chance is not None:
			ax.axhline(y=chance, color="k", linestyle="--", linewidth=1, label="chance")

		ax.set_title(f"SF = {sf:.2f} cyc/deg")
		ax.set_xticks(x_values)
		ax.set_xticklabels(x_labels)
		ax.grid(True, linestyle="--", alpha=0.4)
		if plot_idx % n_cols == 0:
			ax.set_ylabel(ylabel)

		if add_stats_boxes and _fmt_p is not None and _fmt_pq is not None:
			lines = [
				f"Train@SF={sf:.2f} (p/q BH)",
				(
					"WT: RM={rm}  ART={art}  t={t}  W={w}"
				).format(
					rm=_fmt_pq(p_rm_wt.get(sf_idx), q_rm_wt.get(sf_idx)),
					art=_fmt_pq(p_art_wt.get(sf_idx), q_art_wt.get(sf_idx)),
					t=_fmt_pq(p_t_wt.get(sf_idx), q_t_wt.get(sf_idx)),
					w=_fmt_pq(p_w_wt.get(sf_idx), q_w_wt.get(sf_idx)),
				),
				(
					"Resc: RM={rm}  ART={art}  t={t}  W={w}"
				).format(
					rm=_fmt_pq(p_rm_re.get(sf_idx), q_rm_re.get(sf_idx)),
					art=_fmt_pq(p_art_re.get(sf_idx), q_art_re.get(sf_idx)),
					t=_fmt_pq(p_t_re.get(sf_idx), q_t_re.get(sf_idx)),
					w=_fmt_pq(p_w_re.get(sf_idx), q_w_re.get(sf_idx)),
				),
			]
			_add_colored_stats_box_to_axes(ax, lines, x=0.98, y=0.98, fontsize=7.0)

	for ax in axes[len(sf_indices):]:
		ax.axis('off')
	for ax in axes:
		if ax.has_data():
			ax.set_xlabel(x_axis_label)

	# For rel_frac per-SF figure, add a right-margin panel with posthoc Training within each SF×bin.
	if add_stats_boxes and metric_key == "rel_frac" and _fmt_pq is not None:
		# Stable, non-numeric bin labels to avoid accidental coloring of labels.
		bin_labels_safe = ["neg60", "neg30", "zero", "pos30", "pos60", "pos90"]
		n_bins_safe = min(n_bins, len(bin_labels_safe))

		def _fmt_pqq(p: Optional[float], q_sf: Optional[float], q_all: Optional[float]) -> str:
			# Display p / q(within-SF) / q(global across all SF×bin)
			return f"{_fmt_p(p)}/{_fmt_p(q_sf)}/{_fmt_p(q_all)}"

		def _fmt_bins(p_by_bin: Dict[int, Optional[float]], q_by_bin: Dict[int, Optional[float]], q_all_by_bin: Dict[int, Optional[float]]) -> str:
			parts: List[str] = []
			for bi in range(n_bins_safe):
				parts.append(f"{bin_labels_safe[bi]}={_fmt_pqq(p_by_bin.get(bi), q_by_bin.get(bi), q_all_by_bin.get(bi))}")
			return "  ".join(parts)

		panel_lines: List[str] = [
			"Posthoc: Training within each SF×bin (p/qSF/qALL BH)  (* means <0.05)",
			"(qSF: BH within-SF across bins; qALL: BH across all SF×bin cells)",
			"",
		]
		for sf_idx in range(1, len(TARGET_SF_VALUES)):
			sf_mapped = convert_sf_value(float(TARGET_SF_VALUES[sf_idx]))
			panel_lines.append(f"SF={sf_mapped:.2f}")
			panel_lines.append("WT RM : " + _fmt_bins(p_bins_rm_wt.get(sf_idx, {}), q_bins_rm_wt.get(sf_idx, {}), q_bins_rm_wt_all.get(sf_idx, {})))
			panel_lines.append("WT ART: " + _fmt_bins(p_bins_art_wt.get(sf_idx, {}), q_bins_art_wt.get(sf_idx, {}), q_bins_art_wt_all.get(sf_idx, {})))
			panel_lines.append("WT t  : " + _fmt_bins(p_bins_t_wt.get(sf_idx, {}), q_bins_t_wt.get(sf_idx, {}), q_bins_t_wt_all.get(sf_idx, {})))
			panel_lines.append("WT W  : " + _fmt_bins(p_bins_w_wt.get(sf_idx, {}), q_bins_w_wt.get(sf_idx, {}), q_bins_w_wt_all.get(sf_idx, {})))
			panel_lines.append("Resc RM : " + _fmt_bins(p_bins_rm_re.get(sf_idx, {}), q_bins_rm_re.get(sf_idx, {}), q_bins_rm_re_all.get(sf_idx, {})))
			panel_lines.append("Resc ART: " + _fmt_bins(p_bins_art_re.get(sf_idx, {}), q_bins_art_re.get(sf_idx, {}), q_bins_art_re_all.get(sf_idx, {})))
			panel_lines.append("Resc t  : " + _fmt_bins(p_bins_t_re.get(sf_idx, {}), q_bins_t_re.get(sf_idx, {}), q_bins_t_re_all.get(sf_idx, {})))
			panel_lines.append("Resc W  : " + _fmt_bins(p_bins_w_re.get(sf_idx, {}), q_bins_w_re.get(sf_idx, {}), q_bins_w_re_all.get(sf_idx, {})))
			panel_lines.append("")

		panel_right = 0.68
		fig.subplots_adjust(right=panel_right)
		_add_colored_stats_panel_to_figure(fig, panel_lines, x=0.985, y=0.50, fontsize=6.0)
		fig.tight_layout(rect=(0.0, 0.0, panel_right, 1.0))
	else:
		plt.tight_layout()
	out_path = os.path.join(GROUPED_PLOTS_DIR, filename)
	plt.savefig(out_path, format="svg")
	plt.close(fig)
	print(f"[saved] {out_path}")


def plot_scalar_bar(metric_key: str, title: str, ylabel: str, filename: str) -> None:
	"""Bar plot for scalar metrics (one value per session) with paired connecting lines and p-value."""

	stats = aggregate_group(metric_key)
	if not stats:
		print(f"[skip] No data for {metric_key}")
		return

	# Check if any data is finite
	has_data = False
	for mean, sem in stats.values():
		if np.isfinite(mean).any() if np.ndim(mean) > 0 else np.isfinite(mean):
			has_data = True
			break
	if not has_data:
		print(f"[skip] {metric_key} has no finite data")
		return

	groups = list(stats.keys())
	means = [stats[g][0] for g in groups]
	sems = [stats[g][1] for g in groups]
	means = np.array([m if np.ndim(m)==0 else np.nanmean(m) for m in means], dtype=float)
	sems = np.array([s if np.ndim(s)==0 else np.nanmean(s) for s in sems], dtype=float)

	# Collect individual session values for paired plotting
	session_values = {}
	for group_name, sessions in GROUPS.items():
		session_values[group_name] = {}
		for session in sessions:
			session_summary = prepare_session_summaries(session)
			val = session_summary[metric_key]
			if val is None:
				continue
			# Convert to scalar if needed
			if np.ndim(val) > 0:
				val = np.nanmean(val)
			if np.isfinite(val):
				session_values[group_name][session] = float(val)

	x = np.arange(len(groups))
	fig, ax = plt.subplots(figsize=(8, 5))
	ax.bar(x, means, yerr=sems, color=[GROUP_COLORS[g] for g in groups], alpha=0.5, capsize=4, zorder=1)

	# Plot individual session points with slight jitter and connecting lines for pairs
	group_to_x = {g: i for i, g in enumerate(groups)}
	np.random.seed(42)
	jitter_amount = 0.15

	# Draw connecting lines for paired mice
	for condition, pairs in MOUSE_PAIRS.items():
		before_group = f"{condition}_BEFORE"
		after_group = f"{condition}_AFTER"
		
		if before_group not in group_to_x or after_group not in group_to_x:
			continue
		
		x_before = group_to_x[before_group]
		x_after = group_to_x[after_group]
		
		for session_before, session_after in pairs:
			val_before = session_values.get(before_group, {}).get(session_before)
			val_after = session_values.get(after_group, {}).get(session_after)
			
			if val_before is not None and val_after is not None:
				# Same jitter for both points
				jitter = np.random.uniform(-jitter_amount, jitter_amount)
				ax.plot([x_before + jitter, x_after + jitter], [val_before, val_after], 
				        color='gray', alpha=0.5, linewidth=1, zorder=2)
				ax.scatter([x_before + jitter], [val_before], 
				          color=GROUP_COLORS[before_group], s=40, alpha=0.8, zorder=3, edgecolors='black', linewidths=0.5)
				ax.scatter([x_after + jitter], [val_after], 
				          color=GROUP_COLORS[after_group], s=40, alpha=0.8, zorder=3, edgecolors='black', linewidths=0.5)

	# Plot unpaired sessions (BLIND group or any unpaired sessions)
	for group_name in groups:
		if group_name == "BLIND":
			x_pos = group_to_x[group_name]
			for session, val in session_values.get(group_name, {}).items():
				jitter = np.random.uniform(-jitter_amount, jitter_amount)
				ax.scatter([x_pos + jitter], [val], 
				          color=GROUP_COLORS[group_name], s=40, alpha=0.8, zorder=3, edgecolors='black', linewidths=0.5)

	ax.set_xticks(x)
	ax.set_xticklabels([GROUP_LABELS[g] for g in groups], rotation=30, ha="right")
	ax.set_ylabel(ylabel)
	ax.set_title(title)
	ax.grid(True, axis="y", linestyle="--", alpha=0.4)
	
	# Add p-value annotations for WT and Rescued groups separately
	pvals = compute_paired_pvalues(metric_key)
	p_text_parts = []
	if pvals["WT"] is not None:
		p_wt = pvals["WT"]
		sig_str_wt = f"WT: p={p_wt:.4f}"
		if p_wt < 0.001:
			sig_str_wt += " ***"
		elif p_wt < 0.01:
			sig_str_wt += " **"
		elif p_wt < 0.05:
			sig_str_wt += " *"
		p_text_parts.append(sig_str_wt)
	
	if pvals["RESCUED"] is not None:
		p_rescued = pvals["RESCUED"]
		sig_str_rescued = f"Rescued: p={p_rescued:.4f}"
		if p_rescued < 0.001:
			sig_str_rescued += " ***"
		elif p_rescued < 0.01:
			sig_str_rescued += " **"
		elif p_rescued < 0.05:
			sig_str_rescued += " *"
		p_text_parts.append(sig_str_rescued)
	
	if p_text_parts:
		p_text = "\n".join(p_text_parts)
		ax.text(0.5, 0.95, p_text, transform=ax.transAxes, 
		        fontsize=9, verticalalignment='top', horizontalalignment='center',
		        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
	
	plt.tight_layout()
	out_path = os.path.join(GROUPED_PLOTS_DIR, filename)
	plt.savefig(out_path, format="svg")
	plt.close(fig)
	print(f"[saved] {out_path}")



# ---------------------------------------------------------------------------
# Pseudopopulation PCA and dsPCA analysis
# ---------------------------------------------------------------------------
def run_pseudopopulation_pca():
	"""Run PCA on pseudopopulations for each group and create comparison plots."""
	print("\n" + "="*70)
	print("Running Pseudopopulation PCA Analysis")
	print("="*70)
	
	group_names = ["WT_BEFORE", "WT_AFTER", "RESCUED_BEFORE", "RESCUED_AFTER"]
	group_labels_short = ["WT pre", "WT post", "Rescued pre", "Rescued post"]
	group_colors_pca = ["#9ecae1", "#08306b", "#fcae91", "#a50f15"]
	
	results = {}
	
	for group_name in group_names:
		print(f"\n--- {group_name} ---")
		sessions = GROUPS.get(group_name, [])
		if len(sessions) == 0:
			print(f"  No sessions for {group_name}")
			continue
		
		pseudopop, n_conditions = build_pseudopopulation(sessions)
		if pseudopop is None or pseudopop.shape[0] < 2:
			print(f"  Insufficient neurons for {group_name}")
			continue
		
		n_neurons = pseudopop.shape[0]
		
		# Remove neurons that are ALL NaN (never responsive)
		valid_neurons = ~np.all(np.isnan(pseudopop), axis=1)
		pseudopop_valid = pseudopop[valid_neurons]
		
		if pseudopop_valid.shape[0] < 2:
			print(f"  Insufficient valid neurons for {group_name}")
			print(f"    (All {n_neurons} neurons are all-NaN)")
			continue
		
		print(f"    Kept {pseudopop_valid.shape[0]} / {n_neurons} neurons (removed all-NaN neurons)")
		
		# NaN represents "no response" in baseline-subtracted data, treat as 0
		# This preserves selective neurons that respond to some but not all conditions
		pseudopop_filled = np.nan_to_num(pseudopop_valid, nan=0.0)
		
		# Remove neurons with zero variance (constant across all conditions)
		var_per_neuron = np.var(pseudopop_filled, axis=1)
		valid_var = var_per_neuron > 1e-10
		pseudopop_clean = pseudopop_filled[valid_var]
		
		if pseudopop_clean.shape[0] < 2:
			print(f"  Insufficient neurons with variance for {group_name}")
			print(f"    (Only {pseudopop_clean.shape[0]} neurons have non-zero variance)")
			continue
		
		print(f"    Final: {pseudopop_clean.shape[0]} neurons with variance, {pseudopop_clean.shape[1]} conditions")
		
		# Center data (mean across conditions per neuron)
		pseudopop_centered = pseudopop_clean - np.mean(pseudopop_clean, axis=1, keepdims=True)
		
		# Final check for NaN (should not happen but be safe)
		if np.any(np.isnan(pseudopop_centered)):
			print(f"  Error: NaN detected after cleaning for {group_name}, skipping")
			continue
		
		# Run PCA
		pca = PCA(svd_solver='full')
		pca.fit(pseudopop_centered.T)  # (conditions, neurons)
		
		eigvals = pca.explained_variance_
		evr = pca.explained_variance_ratio_
		cum_evr = np.cumsum(evr)
		pr = participation_ratio(eigvals)
		
		# Knee at 80% variance
		knee_idx = np.where(cum_evr >= 0.8)[0]
		knee_pc = int(knee_idx[0] + 1) if knee_idx.size > 0 else np.nan
		
		# Power-law fit
		eig_pos = eigvals[eigvals > 0]
		if eig_pos.size >= 2:
			ranks = np.arange(1, eig_pos.size + 1)
			slope, intercept = np.polyfit(np.log(ranks), np.log(eig_pos), 1)
			alpha = -slope
		else:
			alpha = np.nan
		
		# Normalized metrics
		pr_norm = pr / n_neurons if n_neurons > 0 else np.nan
		knee_ratio = knee_pc / n_neurons if np.isfinite(knee_pc) and n_neurons > 0 else np.nan
		eff_dim_pct = pr_norm * 100 if np.isfinite(pr_norm) else np.nan
		
		print(f"  N neurons: {n_neurons}")
		print(f"  Participation Ratio: {pr:.2f} (norm={pr_norm:.4f})")
		print(f"  Effective Dimensionality: {eff_dim_pct:.2f}%")
		print(f"  Knee PC: {knee_pc} (ratio={knee_ratio:.4f})")
		print(f"  Power-law α: {alpha:.4f}")
		
		results[group_name] = {
			'eigvals': eigvals,
			'evr': evr,
			'cum_evr': cum_evr,
			'pr': pr,
			'pr_norm': pr_norm,
			'knee_pc': knee_pc,
			'knee_ratio': knee_ratio,
			'alpha': alpha,
			'eff_dim_pct': eff_dim_pct,
			'n_neurons': n_neurons,
			'pca': pca,
			'pseudopop_centered': pseudopop_centered,
		}
	
	# Save results
	np.savez(
		os.path.join(GROUPED_PLOTS_DIR, 'pseudopopulation_pca_results.npz'),
		**{f"{name}_{key}": val for name, data in results.items() 
		   for key, val in data.items() if key not in ['pca', 'pseudopop_centered']}
	)
	
	# Plot 1: Participation ratio comparison
	if results:
		fig, ax = plt.subplots(figsize=(8, 5))
		x_pos = np.arange(len(results))
		pr_vals = [results[gn]['pr'] for gn in group_names if gn in results]
		pr_norm_vals = [results[gn]['pr_norm'] for gn in group_names if gn in results]
		n_neurons_vals = [results[gn]['n_neurons'] for gn in group_names if gn in results]
		labels = [group_labels_short[i] for i, gn in enumerate(group_names) if gn in results]
		colors = [group_colors_pca[i] for i, gn in enumerate(group_names) if gn in results]
		
		bars = ax.bar(x_pos, pr_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
		ax.set_xticks(x_pos)
		ax.set_xticklabels(labels, rotation=15, ha='right')
		ax.set_ylabel('Participation Ratio', fontsize=12)
		ax.set_title('Pseudopopulation Dimensionality Comparison\n(Participation Ratio)', fontsize=13, fontweight='bold')
		ax.grid(axis='y', alpha=0.3)
		
		# Add value labels
		for i, (bar, pr_val, pr_norm, n_neur) in enumerate(zip(bars, pr_vals, pr_norm_vals, n_neurons_vals)):
			ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(pr_vals)*0.02,
				   f'{pr_val:.1f}\n({pr_norm:.3f})\nN={n_neur}',
				   ha='center', va='bottom', fontsize=9, fontweight='bold')
		
		plt.tight_layout()
		plt.savefig(os.path.join(GROUPED_PLOTS_DIR, 'pseudopop_participation_ratio_comparison.svg'), format='svg')
		plt.close()
		print("  [saved] pseudopop_participation_ratio_comparison.svg")
	
	# Plot 2: Power-law alpha comparison
	if results:
		fig, ax = plt.subplots(figsize=(8, 5))
		alpha_vals = [results[gn]['alpha'] for gn in group_names if gn in results]
		
		bars = ax.bar(x_pos, alpha_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
		ax.set_xticks(x_pos)
		ax.set_xticklabels(labels, rotation=15, ha='right')
		ax.set_ylabel('Power-law α', fontsize=12)
		ax.set_title('Pseudopopulation Eigenvalue Decay\n(Power-law Exponent)', fontsize=13, fontweight='bold')
		ax.grid(axis='y', alpha=0.3)
		
		for i, (bar, alpha_val) in enumerate(zip(bars, alpha_vals)):
			if np.isfinite(alpha_val):
				ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(alpha_vals)*0.02,
					   f'{alpha_val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
		
		plt.tight_layout()
		plt.savefig(os.path.join(GROUPED_PLOTS_DIR, 'pseudopop_powerlaw_alpha_comparison.svg'), format='svg')
		plt.close()
		print("  [saved] pseudopop_powerlaw_alpha_comparison.svg")
	
	# Plot 3: Scree plots overlay
	if results:
		fig, ax = plt.subplots(figsize=(10, 6))
		for i, gn in enumerate(group_names):
			if gn not in results:
				continue
			evr = results[gn]['evr']
			n_show = min(50, len(evr))
			ax.plot(np.arange(1, n_show+1), evr[:n_show], '-o', 
				   color=group_colors_pca[i], label=group_labels_short[i], alpha=0.7, markersize=4)
		
		ax.set_xlabel('PC Index', fontsize=12)
		ax.set_ylabel('Explained Variance Ratio', fontsize=12)
		ax.set_title('Pseudopopulation Scree Plots (Top 50 PCs)', fontsize=13, fontweight='bold')
		ax.legend()
		ax.grid(alpha=0.3)
		plt.tight_layout()
		plt.savefig(os.path.join(GROUPED_PLOTS_DIR, 'pseudopop_scree_plots_comparison.svg'), format='svg')
		plt.close()
		print("  [saved] pseudopop_scree_plots_comparison.svg")
	
	# Plot 4: Cumulative variance
	if results:
		fig, ax = plt.subplots(figsize=(10, 6))
		for i, gn in enumerate(group_names):
			if gn not in results:
				continue
			cum_evr = results[gn]['cum_evr']
			n_show = min(100, len(cum_evr))
			ax.plot(np.arange(1, n_show+1), cum_evr[:n_show], '-', 
				   color=group_colors_pca[i], label=group_labels_short[i], alpha=0.7, linewidth=2)
			# Mark 80% threshold
			knee = results[gn]['knee_pc']
			if np.isfinite(knee) and knee <= n_show:
				ax.axvline(knee, color=group_colors_pca[i], linestyle='--', alpha=0.4, linewidth=1)
		
		ax.axhline(0.8, color='red', linestyle=':', alpha=0.5, linewidth=2, label='80% threshold')
		ax.set_xlabel('PC Index', fontsize=12)
		ax.set_ylabel('Cumulative Explained Variance', fontsize=12)
		ax.set_title('Pseudopopulation Cumulative Variance (Top 100 PCs)', fontsize=13, fontweight='bold')
		ax.legend()
		ax.grid(alpha=0.3)
		plt.tight_layout()
		plt.savefig(os.path.join(GROUPED_PLOTS_DIR, 'pseudopop_cumulative_variance_comparison.svg'), format='svg')
		plt.close()
		print("  [saved] pseudopop_cumulative_variance_comparison.svg")
	
	# ========== 3D Manifold Visualizations ==========
	# Get reference orientation and SF values to label conditions
	first_session = GROUPS["WT_BEFORE"][0] if "WT_BEFORE" in GROUPS and GROUPS["WT_BEFORE"] else None
	if first_session is not None:
		session_dir = os.path.join(RESULTS_ROOT, first_session)
		orientation_meta = safe_load_npz(os.path.join(session_dir, "orientation_decoding_metadata.npz"))
		spatialfreq_hist = safe_load_npy(os.path.join(session_dir, "spatialfreq_history_reconciled.npy"))
		
		if orientation_meta is not None and "unique_orientations" in orientation_meta.files:
			unique_oris = np.array(orientation_meta["unique_orientations"], dtype=float)
		else:
			unique_oris = None
		
		if spatialfreq_hist is not None:
			unique_sfs = np.unique(spatialfreq_hist)
		else:
			unique_sfs = None
		
		if unique_oris is not None and unique_sfs is not None:
			# Create condition labels: (n_orientations * n_sf) conditions
			n_ori = len(unique_oris)
			n_sf = len(unique_sfs)
			condition_oris = np.repeat(unique_oris, n_sf)
			condition_sfs = np.tile(unique_sfs, n_ori)
			
			# Create 3D manifold plots for each group
			for group_name in group_names:
				if group_name not in results:
					continue
				
				pca_obj = results[group_name]['pca']
				pseudopop_centered = results[group_name]['pseudopop_centered']
				
				# Project onto top 3 PCs
				if pca_obj.n_components_ >= 3:
					from mpl_toolkits.mplot3d import Axes3D
					
					pca_3d = PCA(n_components=3, svd_solver='full')
					pc_projections = pca_3d.fit_transform(pseudopop_centered.T)  # (n_conditions, 3)
					evr_3d = pca_3d.explained_variance_ratio_
					
					# Plot 1: 3D colored by orientation
					fig = plt.figure(figsize=(10, 8))
					ax = fig.add_subplot(111, projection='3d')
					
					ori_colors = plt.cm.turbo(np.linspace(0, 1, len(unique_oris) + 1)[:-1])
					
					for ori_idx, ori in enumerate(unique_oris):
						mask = condition_oris == ori
						ax.plot(pc_projections[mask, 0], pc_projections[mask, 1], pc_projections[mask, 2],
							   'o-', color=ori_colors[ori_idx], label=f'{ori:.0f}°', markersize=6, alpha=0.7)
					
					ax.set_xlabel(f'PC1 ({evr_3d[0]*100:.1f}%)', fontsize=11)
					ax.set_ylabel(f'PC2 ({evr_3d[1]*100:.1f}%)', fontsize=11)
					ax.set_zlabel(f'PC3 ({evr_3d[2]*100:.1f}%)', fontsize=11)
					ax.set_title(f'{GROUP_LABELS[group_name]}: Population manifold\n(Orientation coding, trajectories across SF)', fontsize=12, fontweight='bold')
					ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1, fontsize=8, title='Orientation')
					plt.tight_layout()
					plt.savefig(os.path.join(GROUPED_PLOTS_DIR, f'pseudopop_pca_manifold_3d_orientation_{group_name}.svg'), 
							   format='svg', bbox_inches='tight')
					plt.close()
					print(f"  [saved] pseudopop_pca_manifold_3d_orientation_{group_name}.svg")
					
					# Plot 2: 3D colored by spatial frequency
					fig = plt.figure(figsize=(10, 8))
					ax = fig.add_subplot(111, projection='3d')
					
					unique_sfs_sorted = np.sort(unique_sfs)
					sf_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_sfs_sorted)))
					sf_color_map = {sf: sf_colors[i] for i, sf in enumerate(unique_sfs_sorted)}
					
					for sf in unique_sfs_sorted:
						mask = condition_sfs == sf
						ax.plot(pc_projections[mask, 0], pc_projections[mask, 1], pc_projections[mask, 2],
							   'o-', color=sf_color_map[sf], label=f'SF {sf:.2f}', markersize=6, alpha=0.7)
					
					ax.set_xlabel(f'PC1 ({evr_3d[0]*100:.1f}%)', fontsize=11)
					ax.set_ylabel(f'PC2 ({evr_3d[1]*100:.1f}%)', fontsize=11)
					ax.set_zlabel(f'PC3 ({evr_3d[2]*100:.1f}%)', fontsize=11)
					ax.set_title(f'{GROUP_LABELS[group_name]}: Population manifold\n(SF coding, trajectories across orientation)', fontsize=12, fontweight='bold')
					ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1, fontsize=8, title='SF')
					plt.tight_layout()
					plt.savefig(os.path.join(GROUPED_PLOTS_DIR, f'pseudopop_pca_manifold_3d_sf_{group_name}.svg'), 
							   format='svg', bbox_inches='tight')
					plt.close()
					print(f"  [saved] pseudopop_pca_manifold_3d_sf_{group_name}.svg")
					
					# Plot 3: 3D combined (orientation as color, SF as marker size)
					fig = plt.figure(figsize=(10, 8))
					ax = fig.add_subplot(111, projection='3d')
					
					sf_norm = (condition_sfs - condition_sfs.min()) / (condition_sfs.max() - condition_sfs.min() + 1e-10)
					marker_sizes = 20 + 80 * sf_norm
					
					for ori_idx, ori in enumerate(unique_oris):
						mask = condition_oris == ori
						# Sort by SF to ensure proper line order (no wrap-around)
						sf_vals = condition_sfs[mask]
						sort_idx = np.argsort(sf_vals)
						masked_projections = pc_projections[mask]
						masked_sizes = marker_sizes[mask]
						# Plot line segments (no wrap-around by using sorted order once)
						sorted_proj = masked_projections[sort_idx]
						for i in range(len(sorted_proj) - 1):
							ax.plot([sorted_proj[i, 0], sorted_proj[i+1, 0]],
								   [sorted_proj[i, 1], sorted_proj[i+1, 1]],
								   [sorted_proj[i, 2], sorted_proj[i+1, 2]],
								   '-', color=ori_colors[ori_idx], alpha=0.4, linewidth=1.5)
						# Then scatter on top
						ax.scatter(sorted_proj[:, 0], sorted_proj[:, 1], sorted_proj[:, 2],
								  c=[ori_colors[ori_idx]], s=masked_sizes[sort_idx], alpha=0.7, 
								  label=f'{ori:.0f}°', edgecolors='black', linewidths=0.5)
					
					ax.set_xlabel(f'PC1 ({evr_3d[0]*100:.1f}%)', fontsize=11)
					ax.set_ylabel(f'PC2 ({evr_3d[1]*100:.1f}%)', fontsize=11)
					ax.set_zlabel(f'PC3 ({evr_3d[2]*100:.1f}%)', fontsize=11)
					ax.set_title(f'{GROUP_LABELS[group_name]}: Population manifold\n(color=orientation, size=SF)', fontsize=12, fontweight='bold')
					ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1, fontsize=8, title='Orientation')
					
					ax.text2D(0.02, 0.98, 'Marker size ∝ Spatial Frequency', transform=ax.transAxes,
							 fontsize=9, verticalalignment='top',
							 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
					
					plt.tight_layout()
					plt.savefig(os.path.join(GROUPED_PLOTS_DIR, f'pseudopop_pca_manifold_3d_combined_{group_name}.svg'), 
							   format='svg', bbox_inches='tight')
					plt.close()
					print(f"  [saved] pseudopop_pca_manifold_3d_combined_{group_name}.svg")
					
					# Plot 4: 2D projection (PC1 vs PC2)
					fig, ax = plt.subplots(figsize=(10, 8))
					
					for ori_idx, ori in enumerate(unique_oris):
						mask = condition_oris == ori
						# Sort by SF to ensure proper line order (no wrap-around)
						sf_vals = condition_sfs[mask]
						sort_idx = np.argsort(sf_vals)
						masked_projections = pc_projections[mask]
						masked_sizes = marker_sizes[mask]
						# Plot line segments (no wrap-around by using sorted order once)
						sorted_proj = masked_projections[sort_idx]
						for i in range(len(sorted_proj) - 1):
							ax.plot([sorted_proj[i, 0], sorted_proj[i+1, 0]],
								   [sorted_proj[i, 1], sorted_proj[i+1, 1]],
								   '-', color=ori_colors[ori_idx], alpha=0.4, linewidth=1.5)
						# Then scatter on top
						ax.scatter(sorted_proj[:, 0], sorted_proj[:, 1],
								  c=[ori_colors[ori_idx]], s=masked_sizes[sort_idx], alpha=0.7, 
								  label=f'{ori:.0f}°', edgecolors='black', linewidths=0.5)
					
					ax.set_xlabel(f'PC1 ({evr_3d[0]*100:.1f}%)', fontsize=12)
					ax.set_ylabel(f'PC2 ({evr_3d[1]*100:.1f}%)', fontsize=12)
					ax.set_title(f'{GROUP_LABELS[group_name]}: Population manifold (2D)\ncolor=orientation, size=SF', fontsize=13, fontweight='bold')
					ax.legend(loc='best', ncol=2, fontsize=9, title='Orientation')
					ax.grid(alpha=0.3)
					ax.axis('equal')
					
					# Add text annotation
					ax.text(0.02, 0.98, 'Marker size ∝ Spatial Frequency', transform=ax.transAxes,
							 fontsize=9, verticalalignment='top',
							 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
					
					plt.tight_layout()
					plt.savefig(os.path.join(GROUPED_PLOTS_DIR, f'pseudopop_pca_manifold_2d_{group_name}.svg'), 
							   format='svg', bbox_inches='tight')
					plt.close()
					print(f"  [saved] pseudopop_pca_manifold_2d_{group_name}.svg")
	
	print("\nPseudopopulation PCA analysis complete.")
	print("="*70)
	return results


def run_pseudopopulation_dspca(*, subsample_neurons: bool = True, out_tag: str = ""):
	"""Run dsPCA on pseudopopulations for each group and create comparison plots."""
	if not HAS_DSPCA:
		print("\n[skip] dsPCA not installed. Run: pip install dspca")
		return {}
	
	print("\n" + "="*70)
	print("Running Pseudopopulation dsPCA Analysis")
	print("="*70)
	
	group_names = ["WT_BEFORE", "WT_AFTER", "RESCUED_BEFORE", "RESCUED_AFTER"]
	group_labels_short = ["WT pre", "WT post", "Rescued pre", "Rescued post"]
	group_colors_pca = ["#9ecae1", "#08306b", "#fcae91", "#a50f15"]
	
	# Get reference orientation and SF values (assume consistent across sessions)
	first_session = GROUPS["WT_BEFORE"][0] if "WT_BEFORE" in GROUPS and GROUPS["WT_BEFORE"] else None
	if first_session is None:
		print("  No reference session found")
		return {}
	
	meta = safe_load_npz(os.path.join(RESULTS_ROOT, first_session, "orientation_decoding_metadata.npz"))
	if meta is None or "unique_orientations" not in meta.files or "unique_spatialfreqs" not in meta.files:
		print("  Could not load orientation/SF metadata")
		return {}
	
	unique_orientations = np.array(meta["unique_orientations"], dtype=float)
	unique_spatialfreqs = np.array(meta["unique_spatialfreqs"], dtype=float)
	n_ori = len(unique_orientations)
	n_sf = len(unique_spatialfreqs)
	
	# Build target array with circular orientation basis: [SF_code, sin(2θ), cos(2θ)]
	# - sin(2θ) and cos(2θ) preserve 180° periodicity of orientation coding
	# - SF is log-normalized then z-scored across SF levels (excluding SF<=0).
	#   This makes dsPCA results invariant to log base (log2 vs log10) up to numerical noise.
	mapped_sfs_by_level: List[float] = [float(convert_sf_value(sf)) for sf in unique_spatialfreqs]
	# Linear SF levels for labeling/panels (exclude baseline/invalid)
	mapped_sfs_linear: List[float] = sorted([float(msf) for msf in mapped_sfs_by_level if float(msf) > 0])
	raw_log_sf_by_level: List[float] = [float(np.log2(msf)) if msf > 0 else 0.0 for msf in mapped_sfs_by_level]
	nonzero_raw = np.array([v for v, msf in zip(raw_log_sf_by_level, mapped_sfs_by_level) if msf > 0], dtype=float)
	if nonzero_raw.size:
		sf_mu = float(np.mean(nonzero_raw))
		sf_sigma = float(np.std(nonzero_raw, ddof=0))
	else:
		sf_mu = 0.0
		sf_sigma = 1.0
	if not np.isfinite(sf_sigma) or sf_sigma <= 0:
		sf_sigma = 1.0

	sf_code_by_level: List[float] = []
	for msf, raw_log in zip(mapped_sfs_by_level, raw_log_sf_by_level):
		if msf > 0:
			sf_code_by_level.append(float((raw_log - sf_mu) / sf_sigma))
		else:
			# Keep baseline / invalid SFs at 0 in code space.
			sf_code_by_level.append(0.0)

	# Build condition-wise targets (same ordering as pseudopopulation conditions)
	# and keep the mapped SF (non-log) per condition for slicing/labeling.
	mapped_sf_targets: List[float] = []
	targets = []
	for ori in unique_orientations:
		ori_rad = np.deg2rad(2 * ori)  # 2× frequency for 180° periodicity
		for sf_idx, _sf in enumerate(unique_spatialfreqs):
			mapped_sf_targets.append(float(mapped_sfs_by_level[sf_idx]))
			targets.append([sf_code_by_level[sf_idx], np.sin(ori_rad), np.cos(ori_rad)])
	targets = np.array(targets, dtype=float)  # (n_conditions, 3): [SF_code, sin(2θ), cos(2θ)]
	mapped_sf_targets_arr = np.array(mapped_sf_targets, dtype=float)

	def _standardize_targets_for_fit(targets_raw: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
		"""Z-score each regressor column for dsPCA fitting.

		Important: plotting/labeling should continue to use `targets_raw` (unstandardized) to
		avoid distorting derived orientation labels.
		"""
		t = np.asarray(targets_raw, dtype=float)
		t_fit = t.copy()
		mu = np.nanmean(t_fit, axis=0)
		sigma = np.nanstd(t_fit, axis=0, ddof=0)
		sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)
		t_fit = (t_fit - mu) / sigma
		return t_fit, {"mu": mu, "sigma": sigma}

	targets_fit, targets_fit_stats = _standardize_targets_for_fit(targets)

	# Helper for plotting: map SF_code -> mapped SF value for titles/ticks.
	sf_code_to_mapped_sf: Dict[float, float] = {
		float(code): float(msf)
		for code, msf in zip(sf_code_by_level, mapped_sfs_by_level)
		if float(msf) > 0
	}
	# Ordered SF levels for panels (mapped_sf, sf_code)
	sf_levels_for_panels: List[Tuple[float, float]] = [
		(float(msf), float(code))
		for msf, code in sorted(zip(mapped_sfs_by_level, sf_code_by_level), key=lambda t: float(t[0]))
		if float(msf) > 0
	]

	results = {}
	for group_name in group_names:
		print(f"\n--- {group_name} ---")
		sessions = GROUPS.get(group_name, [])
		if len(sessions) == 0:
			print(f"  No sessions for {group_name}")
			continue

		pseudopop, n_conditions = build_pseudopopulation(sessions, subsample_neurons=subsample_neurons)
		if pseudopop is None or pseudopop.shape[0] < 2:
			print(f"  Insufficient neurons for {group_name}")
			continue

		if pseudopop.shape[1] != targets.shape[0]:
			print(f"  Condition mismatch: pseudopop has {pseudopop.shape[1]}, targets has {targets.shape[0]}")
			continue

		n_neurons = pseudopop.shape[0]

		# Remove neurons that are ALL NaN (never responsive)
		valid_neurons = ~np.all(np.isnan(pseudopop), axis=1)
		pseudopop_valid = pseudopop[valid_neurons]
		if pseudopop_valid.shape[0] < 2:
			print(f"  Insufficient valid neurons for {group_name}")
			print(f"    (All {n_neurons} neurons are all-NaN)")
			continue
		print(f"    Kept {pseudopop_valid.shape[0]} / {n_neurons} neurons (removed all-NaN neurons)")

		# NaN represents "no response" in baseline-subtracted data, treat as 0
		pseudopop_filled = np.nan_to_num(pseudopop_valid, nan=0.0)

		# Remove neurons with zero variance
		var_per_neuron = np.var(pseudopop_filled, axis=1)
		valid_var = var_per_neuron > 1e-10
		pseudopop_clean = pseudopop_filled[valid_var]
		if pseudopop_clean.shape[0] < 2:
			print(f"  Insufficient neurons with variance for {group_name}")
			print(f"    (Only {pseudopop_clean.shape[0]} neurons have non-zero variance)")
			continue
		print(f"    Final: {pseudopop_clean.shape[0]} neurons with variance, {pseudopop_clean.shape[1]} conditions")

		# Transpose for dsPCA: (conditions, neurons)
		data_dspca = pseudopop_clean.T

		print(f"  Data shape: {data_dspca.shape} (conditions × neurons)")
		print(f"  Targets shape: {targets.shape} (should match conditions)")

		if data_dspca.shape[0] != targets.shape[0]:
			print(f"  Error: Data/target shape mismatch: {data_dspca.shape[0]} vs {targets.shape[0]}")
			continue

		try:
			projection_target, projection_targetfree, ax_targets, ax_targetfree, \
			target_signal, targetfree_signal, target_var, targetfree_var, total_var, dot_target_ax = \
				dsPCA(data=data_dspca, targets=targets_fit)

			var_sf = float(target_var[0])
			var_ori_sin = float(target_var[1])
			var_ori_cos = float(target_var[2])
			var_ori_total = float(np.sqrt(var_ori_sin**2 + var_ori_cos**2))

			var_ori_pct = 100 * var_ori_total / total_var
			var_sf_pct = 100 * var_sf / total_var
			var_target_pct = 100 * np.sum(target_var) / total_var
			var_targetfree_top10_pct = 100 * np.sum(targetfree_var[:min(10, len(targetfree_var))]) / total_var

			# Keep the prior placeholder (axes are constructed orthogonal in target space)
			ori_mag = np.sqrt(var_ori_sin**2 + var_ori_cos**2) + 1e-10
			ori_unit_x = var_ori_sin / ori_mag
			ori_unit_y = var_ori_cos / ori_mag
			sf_unit = 1.0
			orthogonality = abs(sf_unit * 0 + np.sqrt(ori_unit_x**2 + ori_unit_y**2))

			print(f"  Orientation subspace variance (sqrt(sin^2+cos^2)): {var_ori_pct:.2f}%")
			print(f"  SF axis variance: {var_sf_pct:.2f}%")
			print(f"  Total target variance: {var_target_pct:.2f}%")
			print(f"  Orthogonality (Ori vs SF): {orthogonality:.6f}")

			results[group_name] = {
				'var_ori_pct': var_ori_pct,
				'var_sf_pct': var_sf_pct,
				'var_target_pct': var_target_pct,
				'var_targetfree_top10_pct': var_targetfree_top10_pct,
				'orthogonality': orthogonality,
				'n_neurons': n_neurons,
				'target_var': target_var,
				'targetfree_var': targetfree_var,
				'total_var': total_var,
				'projection_target': projection_target,
				'projection_targetfree': projection_targetfree,
				'ax_targets': ax_targets,
				'targets': targets,
				'targets_fit': targets_fit,
				'targets_fit_stats': targets_fit_stats,
				'mapped_sf_targets': mapped_sf_targets_arr,
				'dspca': None,
			}

		except Exception as exc:
			print(f"  dsPCA failed: {type(exc).__name__}: {exc}")
			continue
	
	_pseudopopulation_dspca_make_plots(
		results=results,
		group_names=group_names,
		group_labels_short=group_labels_short,
		group_colors_pca=group_colors_pca,
		mapped_sfs_linear=mapped_sfs_linear,
		out_tag=out_tag,
	)
	_pseudopopulation_dspca_make_detailed_plots(
		results=results,
		group_names=group_names,
		mapped_sfs_linear=mapped_sfs_linear,
		out_tag=out_tag,
	)

	# Additional superimposed + distance analysis figures (shared with CV run)
	_pseudopopulation_dspca_make_superimposed_and_distance_plots(
		results=results,
		mapped_sfs_linear=mapped_sfs_linear,
		out_tag=out_tag,
	)

	print("\nPseudopopulation dsPCA analysis complete.")
	print("="*70)
	return results


def run_pseudopopulation_dspca_cv50(*, subsample_neurons: bool = True, out_tag: str = "_10fcv") -> Dict[str, Dict[str, object]]:
	"""Cross-validated dsPCA: fit each train split and aggregate held-out folds.

	Uses per-session 10-fold CV bundles when available and aligns held-out projections into
	a single reference basis before plotting.
	"""
	if not HAS_DSPCA:
		print("\n[skip] dsPCA not installed. Run: pip install dspca")
		return {}

	print("\n" + "="*70)
	print("Running Pseudopopulation dsPCA Analysis (10-fold CV)")
	print("="*70)

	group_names = ["WT_BEFORE", "WT_AFTER", "RESCUED_BEFORE", "RESCUED_AFTER"]
	group_labels_short = ["WT pre", "WT post", "Rescued pre", "Rescued post"]
	group_colors_pca = ["#9ecae1", "#08306b", "#fcae91", "#a50f15"]

	first_session = GROUPS["WT_BEFORE"][0] if "WT_BEFORE" in GROUPS and GROUPS["WT_BEFORE"] else None
	if first_session is None:
		print("  No reference session found")
		return {}
	meta = safe_load_npz(os.path.join(RESULTS_ROOT, first_session, "orientation_decoding_metadata.npz"))
	if meta is None or "unique_orientations" not in meta.files or "unique_spatialfreqs" not in meta.files:
		print("  Could not load orientation/SF metadata")
		return {}

	unique_orientations = np.array(meta["unique_orientations"], dtype=float)
	unique_spatialfreqs = np.array(meta["unique_spatialfreqs"], dtype=float)

	mapped_sfs_by_level: List[float] = [float(convert_sf_value(sf)) for sf in unique_spatialfreqs]
	mapped_sfs_linear: List[float] = sorted([float(msf) for msf in mapped_sfs_by_level if float(msf) > 0])
	raw_log_sf_by_level: List[float] = [float(np.log2(msf)) if msf > 0 else 0.0 for msf in mapped_sfs_by_level]
	nonzero_raw = np.array([v for v, msf in zip(raw_log_sf_by_level, mapped_sfs_by_level) if msf > 0], dtype=float)
	if nonzero_raw.size:
		sf_mu = float(np.mean(nonzero_raw))
		sf_sigma = float(np.std(nonzero_raw, ddof=0))
	else:
		sf_mu = 0.0
		sf_sigma = 1.0
	if not np.isfinite(sf_sigma) or sf_sigma <= 0:
		sf_sigma = 1.0

	sf_code_by_level: List[float] = []
	for msf, raw_log in zip(mapped_sfs_by_level, raw_log_sf_by_level):
		if msf > 0:
			sf_code_by_level.append(float((raw_log - sf_mu) / sf_sigma))
		else:
			sf_code_by_level.append(0.0)

	mapped_sf_targets: List[float] = []
	targets = []
	for ori in unique_orientations:
		ori_rad = np.deg2rad(2 * ori)
		for sf_idx, _sf in enumerate(unique_spatialfreqs):
			mapped_sf_targets.append(float(mapped_sfs_by_level[sf_idx]))
			targets.append([sf_code_by_level[sf_idx], np.sin(ori_rad), np.cos(ori_rad)])
	targets = np.array(targets, dtype=float)
	mapped_sf_targets_arr = np.array(mapped_sf_targets, dtype=float)

	# Z-score all target regressors for fitting (keep `targets` raw for labels/masks)
	mu = np.nanmean(targets, axis=0)
	sigma = np.nanstd(targets, axis=0, ddof=0)
	sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)
	targets_fit = (targets - mu) / sigma
	targets_fit_stats = {"mu": mu, "sigma": sigma}

	results: Dict[str, Dict[str, object]] = {}
	for group_name in group_names:
		print(f"\n--- {group_name} ---")
		sessions = GROUPS.get(group_name, [])
		if len(sessions) == 0:
			print(f"  No sessions for {group_name}")
			continue

		full_pop, train_pop_folds, test_pop_folds, test_counts_folds, n_conditions = build_pseudopopulation_cv50(
			sessions,
			subsample_neurons=subsample_neurons,
		)
		if (
			full_pop is None
			or train_pop_folds is None
			or test_pop_folds is None
			or test_counts_folds is None
			or full_pop.shape[0] < 2
		):
			print(f"  Insufficient neurons for {group_name}")
			continue
		if full_pop.shape[1] != targets.shape[0] or train_pop_folds.shape[2] != targets.shape[0] or test_pop_folds.shape[2] != targets.shape[0]:
			print(
				f"  Condition mismatch: full/train/test have "
				f"{full_pop.shape[1]}/{train_pop_folds.shape[2]}/{test_pop_folds.shape[2]}, targets has {targets.shape[0]}"
			)
			continue

		full = np.asarray(full_pop, dtype=float)
		train_folds = np.asarray(train_pop_folds, dtype=float)
		test_folds = np.asarray(test_pop_folds, dtype=float)
		fold_counts = np.asarray(test_counts_folds, dtype=float)
		n_folds = int(train_folds.shape[0])

		valid_neurons_full = np.isfinite(full).any(axis=1)
		valid_neurons_train = np.isfinite(train_folds).any(axis=(0, 2))
		valid_neurons_test = np.isfinite(test_folds).any(axis=(0, 2))
		valid_neurons = valid_neurons_full & valid_neurons_train & valid_neurons_test
		full = full[valid_neurons]
		train_folds = train_folds[:, valid_neurons, :]
		test_folds = test_folds[:, valid_neurons, :]
		print(f"    Kept {full.shape[0]} / {full_pop.shape[0]} neurons (matched across folds)")

		full_filled = np.nan_to_num(full, nan=0.0)
		var_per_neuron = np.var(full_filled, axis=1)
		valid_var = var_per_neuron > 1e-10
		full_clean = full_filled[valid_var]
		train_clean_folds = np.nan_to_num(train_folds[:, valid_var, :], nan=0.0)
		test_clean_folds = np.nan_to_num(test_folds[:, valid_var, :], nan=0.0)
		if full_clean.shape[0] < 2:
			print(f"  Insufficient neurons with variance for {group_name}")
			continue
		print(f"    Final: {full_clean.shape[0]} neurons with variance, {full_clean.shape[1]} conditions, {n_folds} folds")

		X_ref = full_clean.T

		try:
			proj_t_ref, proj_tf_ref, ax_targets_ref, ax_targetfree_ref, *_rest_ref = dsPCA(data=X_ref, targets=targets_fit)
			proj_t_test_folds = []
			proj_tf_test_folds = []
			proj_t_train_folds = []
			proj_tf_train_folds = []

			for fold_idx in range(n_folds):
				train_fold = train_clean_folds[fold_idx]
				test_fold = test_clean_folds[fold_idx]
				X_train = train_fold.T
				X_test = test_fold.T
				fold_center = np.nanmean(X_train, axis=0, keepdims=True)
				proj_t_train_fold, proj_tf_train_fold, ax_targets_fold, ax_targetfree_fold, *_rest_fold = dsPCA(
					data=X_train,
					targets=targets_fit,
				)
				proj_t_test_fold = _project_with_axes(X_test - fold_center, ax_targets_fold)
				proj_tf_test_fold = _project_with_axes(X_test - fold_center, ax_targetfree_fold)
				proj_t_train_folds.append(
					_align_target_projection_to_reference(proj_t_train_fold, ax_targets_fold, ax_targets_ref)
				)
				proj_tf_train_folds.append(
					_align_component_projection_signs(proj_tf_train_fold, ax_targetfree_fold, ax_targetfree_ref)
				)
				proj_t_test_folds.append(
					_align_target_projection_to_reference(proj_t_test_fold, ax_targets_fold, ax_targets_ref)
				)
				proj_tf_test_folds.append(
					_align_component_projection_signs(proj_tf_test_fold, ax_targetfree_fold, ax_targetfree_ref)
				)

			proj_t_train_folds_arr = np.stack(proj_t_train_folds, axis=0)
			proj_tf_train_folds_arr = np.stack(proj_tf_train_folds, axis=0)
			proj_t_test_folds_arr = np.stack(proj_t_test_folds, axis=0)
			proj_tf_test_folds_arr = np.stack(proj_tf_test_folds, axis=0)
			proj_t_test = np.divide(
				np.nansum(proj_t_test_folds_arr * fold_counts[:, :, None], axis=0),
				np.sum(fold_counts, axis=0)[:, None],
				out=np.full_like(proj_t_test_folds_arr[0], np.nan, dtype=float),
				where=np.sum(fold_counts, axis=0)[:, None] > 0,
			)
			proj_tf_test = np.divide(
				np.nansum(proj_tf_test_folds_arr * fold_counts[:, :, None], axis=0),
				np.sum(fold_counts, axis=0)[:, None],
				out=np.full_like(proj_tf_test_folds_arr[0], np.nan, dtype=float),
				where=np.sum(fold_counts, axis=0)[:, None] > 0,
			)
			proj_t_train = np.divide(
				np.nansum(proj_t_train_folds_arr * fold_counts[:, :, None], axis=0),
				np.sum(fold_counts, axis=0)[:, None],
				out=np.full_like(proj_t_train_folds_arr[0], np.nan, dtype=float),
				where=np.sum(fold_counts, axis=0)[:, None] > 0,
			)
			proj_tf_train = np.divide(
				np.nansum(proj_tf_train_folds_arr * fold_counts[:, :, None], axis=0),
				np.sum(fold_counts, axis=0)[:, None],
				out=np.full_like(proj_tf_train_folds_arr[0], np.nan, dtype=float),
				where=np.sum(fold_counts, axis=0)[:, None] > 0,
			)

			target_var_test = np.var(proj_t_test, axis=0, ddof=0)
			targetfree_var_test = np.var(proj_tf_test, axis=0, ddof=0)
			total_var_test = float(np.sum(np.var(X_ref, axis=0, ddof=0)))
			var_sf = float(target_var_test[0])
			var_ori_sin = float(target_var_test[1])
			var_ori_cos = float(target_var_test[2])
			var_ori_total = float(np.sqrt(var_ori_sin**2 + var_ori_cos**2))
			var_ori_pct = 100 * var_ori_total / total_var_test if total_var_test > 0 else float('nan')
			var_sf_pct = 100 * var_sf / total_var_test if total_var_test > 0 else float('nan')
			var_target_pct = 100 * float(np.sum(target_var_test)) / total_var_test if total_var_test > 0 else float('nan')
			var_targetfree_top10_pct = 100 * float(np.sum(targetfree_var_test[:min(10, len(targetfree_var_test))])) / total_var_test if total_var_test > 0 else float('nan')
			orthogonality = _orthogonality_sf_vs_ori(ax_targets_ref)

			print(f"  [TEST] Orientation subspace variance (sqrt(sin^2+cos^2)): {var_ori_pct:.2f}%")
			print(f"  [TEST] SF axis variance: {var_sf_pct:.2f}%")
			print(f"  [TEST] Total target variance: {var_target_pct:.2f}%")
			print(f"  [REFERENCE axes] Orthogonality (SF vs Ori): {orthogonality:.6f}")

			results[group_name] = {
				'var_ori_pct': var_ori_pct,
				'var_sf_pct': var_sf_pct,
				'var_target_pct': var_target_pct,
				'var_targetfree_top10_pct': var_targetfree_top10_pct,
				'orthogonality': orthogonality,
				'n_neurons': int(full_clean.shape[0]),
				'target_var': target_var_test,
				'targetfree_var': targetfree_var_test,
				'total_var': total_var_test,
				'projection_target': proj_t_test,
				'projection_targetfree': proj_tf_test,
				'projection_target_train': proj_t_train,
				'projection_targetfree_train': proj_tf_train,
				'projection_target_test_folds': proj_t_test_folds_arr,
				'projection_targetfree_test_folds': proj_tf_test_folds_arr,
				'fold_test_counts': fold_counts,
				'ax_targets': ax_targets_ref,
				'targets': targets,
				'targets_fit': targets_fit,
				'targets_fit_stats': targets_fit_stats,
				'mapped_sf_targets': mapped_sf_targets_arr,
				'dspca': None,
			}
		except Exception as exc:
			print(f"  CV dsPCA failed: {type(exc).__name__}: {exc}")
			continue

	_pseudopopulation_dspca_make_plots(
		results=results,
		group_names=group_names,
		group_labels_short=group_labels_short,
		group_colors_pca=group_colors_pca,
		mapped_sfs_linear=mapped_sfs_linear,
		out_tag=out_tag,
	)
	_pseudopopulation_dspca_make_detailed_plots(
		results=results,
		group_names=group_names,
		mapped_sfs_linear=mapped_sfs_linear,
		out_tag=out_tag,
	)
	_pseudopopulation_dspca_make_superimposed_and_distance_plots(
		results=results,
		mapped_sfs_linear=mapped_sfs_linear,
		out_tag=out_tag,
	)
	print("\nPseudopopulation dsPCA 10-fold CV analysis complete.")
	print("="*70)
	return results


# ---------------------------------------------------------------------------
# dsPCA distance metric comparison plots (group-level)
# ---------------------------------------------------------------------------
def load_session_distance_metrics(session_rel: str) -> Optional[Dict[str, np.ndarray]]:
	"""Load per-SF and per-orientation distance metrics from a session's dsPCA results."""
	session_dir = os.path.join(RESULTS_ROOT, session_rel)
	dspca_results_path = os.path.join(session_dir, 'dspca_results.npz')
	
	if not os.path.exists(dspca_results_path):
		return None

	try:
		npz = np.load(dspca_results_path, allow_pickle=True)
		metrics = {}
		
		# Load distance metrics if available
		if 'projection_target' in npz.files:
			proj_target = npz['projection_target']
			targets = npz.get('targets', None)
			
			if targets is not None and proj_target.shape[0] == targets.shape[0]:
				# Recover unique orientations and SFs
				ori_from_tgt = np.rad2deg(np.arctan2(targets[:, 1], targets[:, 2]) / 2)
				ori_from_tgt = np.mod(ori_from_tgt, 180)
				
				# Quantize to 30° bins for consistency
				ori_quantized = np.round(ori_from_tgt / 30) * 30
				ori_quantized = np.mod(ori_quantized, 180)
				
				unique_ori_quant = np.unique(ori_quantized)
				sf_log_vals = targets[:, 0]
				unique_sf_log = np.unique(sf_log_vals[sf_log_vals != 0])  # Exclude baseline
				
				# Compute distance-to-COM for each SF
				dist_to_com_by_sf = {}
				for sf_log in unique_sf_log:
					mask = np.abs(sf_log_vals - sf_log) < 1e-6
					if np.any(mask):
						sin_vals = proj_target[mask, 1]
						cos_vals = proj_target[mask, 2]
						com = np.array([np.nanmean(sin_vals), np.nanmean(cos_vals)])
						dist = np.sqrt((sin_vals - com[0])**2 + (cos_vals - com[1])**2)
						ori_sorted = ori_quantized[mask]
						sort_idx = np.argsort(ori_sorted)
						dist_to_com_by_sf[sf_log] = dist[sort_idx]
				
				# Compute adjacent orientation distances for each SF
				adj_dist_by_sf = {}
				adj_dist_cv_by_sf = {}
				circle_rmse_norm_by_sf = {}
				radial_cv_by_sf = {}
				angle_spacing_rmse_norm_by_sf = {}
				for sf_log in unique_sf_log:
					mask = np.abs(sf_log_vals - sf_log) < 1e-6
					if np.any(mask):
						sin_vals = proj_target[mask, 1]
						cos_vals = proj_target[mask, 2]
						ori_sf = ori_quantized[mask]

						# Sort by binned orientation and collapse duplicates (robustness)
						sort_idx = np.argsort(ori_sf)
						ori_sorted = ori_sf[sort_idx]
						sin_sorted = sin_vals[sort_idx]
						cos_sorted = cos_vals[sort_idx]
						unique_bins = np.unique(ori_sorted)
						if unique_bins.size >= 2 and unique_bins.size != ori_sorted.size:
							sin_u = []
							cos_u = []
							for ob in unique_bins:
								m = ori_sorted == ob
								sin_u.append(np.nanmean(sin_sorted[m]))
								cos_u.append(np.nanmean(cos_sorted[m]))
							sin_sorted = np.asarray(sin_u)
							cos_sorted = np.asarray(cos_u)
						# Wrap around
						sin_wrap = np.append(sin_sorted, sin_sorted[0])
						cos_wrap = np.append(cos_sorted, cos_sorted[0])
						dists = np.sqrt(np.diff(sin_wrap)**2 + np.diff(cos_wrap)**2)
						adj_dist_by_sf[sf_log] = dists
						mean_d = float(np.nanmean(dists)) if dists.size else np.nan
						if not np.isfinite(mean_d) or mean_d <= 0 or np.sum(np.isfinite(dists)) < 2:
							adj_dist_cv_by_sf[sf_log] = np.nan
						else:
							adj_dist_cv_by_sf[sf_log] = float(np.nanstd(dists, ddof=1) / mean_d)

						geom = circle_geometry_metrics(sin_sorted, cos_sorted)
						radial_cv_by_sf[sf_log] = float(geom["radial_cv"]) if "radial_cv" in geom else np.nan
						circle_rmse_norm_by_sf[sf_log] = float(geom["circle_rmse_norm"]) if "circle_rmse_norm" in geom else np.nan
						angle_spacing_rmse_norm_by_sf[sf_log] = float(geom["angle_spacing_rmse_norm"]) if "angle_spacing_rmse_norm" in geom else np.nan
				
				metrics['dist_to_com_by_sf'] = dist_to_com_by_sf
				metrics['adj_dist_by_sf'] = adj_dist_by_sf
				metrics['adj_dist_cv_by_sf'] = adj_dist_cv_by_sf
				metrics['circle_rmse_norm_by_sf'] = circle_rmse_norm_by_sf
				metrics['radial_cv_by_sf'] = radial_cv_by_sf
				metrics['angle_spacing_rmse_norm_by_sf'] = angle_spacing_rmse_norm_by_sf
				metrics['unique_sf_log'] = unique_sf_log
				metrics['unique_ori_quant'] = unique_ori_quant
		
		npz.close()
		return metrics if metrics else None
	except Exception as e:
		print(f"    Error loading distance metrics from {session_rel}: {e}")
		return None


def _convex_hull_area_2d(points_xy: np.ndarray) -> float:
	"""Area of the convex hull of points in 2D.

	Uses SciPy ConvexHull when available; falls back to a simple angle-sorted polygon area.
	Returns NaN if area cannot be computed.
	"""
	pts = np.asarray(points_xy, dtype=float)
	pts = pts[np.all(np.isfinite(pts), axis=1)]
	if pts.shape[0] < 3:
		return float('nan')
	# Remove exact duplicates to avoid hull issues
	pts = np.unique(pts, axis=0)
	if pts.shape[0] < 3:
		return float('nan')
	try:
		from scipy.spatial import ConvexHull  # local import
		hull = ConvexHull(pts)
		return float(hull.volume)  # in 2D, `volume` is area
	except Exception:
		# Fallback: sort by polar angle around centroid and compute shoelace area
		c = np.mean(pts, axis=0)
		ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
		order = np.argsort(ang)
		poly = pts[order]
		x = poly[:, 0]
		y = poly[:, 1]
		return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def compute_orientation_area_by_sf_from_dspca_npz(dspca_npz: "np.lib.npyio.NpzFile") -> Dict[float, float]:
	"""Compute orientation-plane area per SF from per-session dsPCA results.

	Area is computed in the (sin-axis, cos-axis) dsPCA target subspace (projection_target[:,1:3]).
	For each SF, the 6 orientation points (0..150°) define a polygon; we quantify its spread using
	convex hull area.
	"""
	if dspca_npz is None:
		return {}
	if ("condition_orientations" not in dspca_npz.files) or ("condition_spatialfreqs" not in dspca_npz.files):
		return {}
	# Prefer held-out CV projections if present.
	if "projection_target_test" in dspca_npz.files:
		proj = np.asarray(dspca_npz["projection_target_test"], dtype=float)
	elif "projection_target" in dspca_npz.files:
		proj = np.asarray(dspca_npz["projection_target"], dtype=float)
	else:
		return {}
	oris = np.asarray(dspca_npz["condition_orientations"], dtype=float)
	sfs = np.asarray(dspca_npz["condition_spatialfreqs"], dtype=float)
	if proj.ndim != 2 or proj.shape[1] < 3:
		return {}

	# Quantize orientations to nearest 30° bin for robustness.
	oris_q = np.mod(np.round(oris / 30.0) * 30.0, 180.0)
	unique_sfs = np.unique(sfs[np.isfinite(sfs) & (sfs > 0)])
	area_by_sf: Dict[float, float] = {}
	for sf in unique_sfs:
		mask_sf = np.isfinite(sfs) & np.isclose(sfs, float(sf), atol=1e-12)
		if not np.any(mask_sf):
			continue
		pts_by_ori: List[np.ndarray] = []
		for ori in np.arange(0, 180, 30, dtype=float):
			mask = mask_sf & np.isclose(oris_q, ori, atol=1e-6)
			if not np.any(mask):
				continue
			# Average if duplicates exist
			xy = np.nanmean(proj[mask, 1:3], axis=0)
			pts_by_ori.append(np.asarray(xy, dtype=float))
		pts = np.vstack(pts_by_ori) if len(pts_by_ori) else np.empty((0, 2), dtype=float)
		area_by_sf[float(sf)] = _convex_hull_area_2d(pts)
	return area_by_sf


def _compute_mean_adjacent_distance_by_sf_from_dspca_npz(dspca_npz: "np.lib.npyio.NpzFile") -> Dict[float, float]:
	"""Compute mean adjacent-orientation distance per SF from dsPCA results.

	For each SF, orientation-plane points are ordered in 30-degree bins, wrapped to
	close the loop, and reduced to a single scalar by averaging adjacent edge lengths.
	"""
	if dspca_npz is None:
		return {}
	if ("condition_orientations" not in dspca_npz.files) or ("condition_spatialfreqs" not in dspca_npz.files):
		return {}
	if "projection_target_test" in dspca_npz.files:
		proj = np.asarray(dspca_npz["projection_target_test"], dtype=float)
	elif "projection_target" in dspca_npz.files:
		proj = np.asarray(dspca_npz["projection_target"], dtype=float)
	else:
		return {}
	oris = np.asarray(dspca_npz["condition_orientations"], dtype=float)
	sfs = np.asarray(dspca_npz["condition_spatialfreqs"], dtype=float)
	if proj.ndim != 2 or proj.shape[1] < 3:
		return {}

	oris_q = np.mod(np.round(oris / 30.0) * 30.0, 180.0)
	unique_sfs = np.unique(sfs[np.isfinite(sfs) & (sfs > 0)])
	mean_adj_by_sf: Dict[float, float] = {}
	for sf in unique_sfs:
		mask_sf = np.isfinite(sfs) & np.isclose(sfs, float(sf), atol=1e-12)
		if not np.any(mask_sf):
			continue
		pts_by_ori: List[np.ndarray] = []
		for ori in np.arange(0, 180, 30, dtype=float):
			mask = mask_sf & np.isclose(oris_q, ori, atol=1e-6)
			if not np.any(mask):
				continue
			xy = np.nanmean(proj[mask, 1:3], axis=0)
			pts_by_ori.append(np.asarray(xy, dtype=float))
		if len(pts_by_ori) < 2:
			mean_adj_by_sf[float(sf)] = np.nan
			continue
		pts = np.vstack(pts_by_ori)
		x_wrap = np.append(pts[:, 0], pts[0, 0])
		y_wrap = np.append(pts[:, 1], pts[0, 1])
		dists = np.sqrt(np.diff(x_wrap) ** 2 + np.diff(y_wrap) ** 2)
		mean_adj_by_sf[float(sf)] = float(np.nanmean(dists)) if np.any(np.isfinite(dists)) else np.nan
	return mean_adj_by_sf


def _compute_std_adjacent_distance_by_sf_from_dspca_npz(dspca_npz: "np.lib.npyio.NpzFile") -> Dict[float, float]:
	"""Compute SD of adjacent-orientation distances per SF from dsPCA results.

	For each SF, orientation-plane points are ordered in 30-degree bins, wrapped to
	close the loop, and the standard deviation across adjacent edge lengths is returned.
	"""
	if dspca_npz is None:
		return {}
	if ("condition_orientations" not in dspca_npz.files) or ("condition_spatialfreqs" not in dspca_npz.files):
		return {}
	if "projection_target_test" in dspca_npz.files:
		proj = np.asarray(dspca_npz["projection_target_test"], dtype=float)
	elif "projection_target" in dspca_npz.files:
		proj = np.asarray(dspca_npz["projection_target"], dtype=float)
	else:
		return {}
	oris = np.asarray(dspca_npz["condition_orientations"], dtype=float)
	sfs = np.asarray(dspca_npz["condition_spatialfreqs"], dtype=float)
	if proj.ndim != 2 or proj.shape[1] < 3:
		return {}

	oris_q = np.mod(np.round(oris / 30.0) * 30.0, 180.0)
	unique_sfs = np.unique(sfs[np.isfinite(sfs) & (sfs > 0)])
	std_adj_by_sf: Dict[float, float] = {}
	for sf in unique_sfs:
		mask_sf = np.isfinite(sfs) & np.isclose(sfs, float(sf), atol=1e-12)
		if not np.any(mask_sf):
			continue
		pts_by_ori: List[np.ndarray] = []
		for ori in np.arange(0, 180, 30, dtype=float):
			mask = mask_sf & np.isclose(oris_q, ori, atol=1e-6)
			if not np.any(mask):
				continue
			xy = np.nanmean(proj[mask, 1:3], axis=0)
			pts_by_ori.append(np.asarray(xy, dtype=float))
		if len(pts_by_ori) < 2:
			std_adj_by_sf[float(sf)] = np.nan
			continue
		pts = np.vstack(pts_by_ori)
		x_wrap = np.append(pts[:, 0], pts[0, 0])
		y_wrap = np.append(pts[:, 1], pts[0, 1])
		dists = np.sqrt(np.diff(x_wrap) ** 2 + np.diff(y_wrap) ** 2)
		n_valid = int(np.sum(np.isfinite(dists)))
		std_adj_by_sf[float(sf)] = float(np.nanstd(dists, ddof=1)) if n_valid >= 2 else np.nan
	return std_adj_by_sf


def _compute_adjacent_distance_cv_by_sf_from_dspca_npz(dspca_npz: "np.lib.npyio.NpzFile") -> Dict[float, float]:
	"""Compute CV of adjacent-orientation distances per SF from dsPCA results."""
	if dspca_npz is None:
		return {}
	if ("condition_orientations" not in dspca_npz.files) or ("condition_spatialfreqs" not in dspca_npz.files):
		return {}
	if "projection_target_test" in dspca_npz.files:
		proj = np.asarray(dspca_npz["projection_target_test"], dtype=float)
	elif "projection_target" in dspca_npz.files:
		proj = np.asarray(dspca_npz["projection_target"], dtype=float)
	else:
		return {}
	oris = np.asarray(dspca_npz["condition_orientations"], dtype=float)
	sfs = np.asarray(dspca_npz["condition_spatialfreqs"], dtype=float)
	if proj.ndim != 2 or proj.shape[1] < 3:
		return {}

	oris_q = np.mod(np.round(oris / 30.0) * 30.0, 180.0)
	unique_sfs = np.unique(sfs[np.isfinite(sfs) & (sfs > 0)])
	adj_cv_by_sf: Dict[float, float] = {}
	for sf in unique_sfs:
		mask_sf = np.isfinite(sfs) & np.isclose(sfs, float(sf), atol=1e-12)
		if not np.any(mask_sf):
			continue
		pts_by_ori: List[np.ndarray] = []
		for ori in np.arange(0, 180, 30, dtype=float):
			mask = mask_sf & np.isclose(oris_q, ori, atol=1e-6)
			if not np.any(mask):
				continue
			xy = np.nanmean(proj[mask, 1:3], axis=0)
			pts_by_ori.append(np.asarray(xy, dtype=float))
		if len(pts_by_ori) < 2:
			adj_cv_by_sf[float(sf)] = np.nan
			continue
		pts = np.vstack(pts_by_ori)
		x_wrap = np.append(pts[:, 0], pts[0, 0])
		y_wrap = np.append(pts[:, 1], pts[0, 1])
		dists = np.sqrt(np.diff(x_wrap) ** 2 + np.diff(y_wrap) ** 2)
		mean_d = float(np.nanmean(dists)) if np.any(np.isfinite(dists)) else np.nan
		n_valid = int(np.sum(np.isfinite(dists)))
		if not np.isfinite(mean_d) or mean_d <= 0 or n_valid < 2:
			adj_cv_by_sf[float(sf)] = np.nan
		else:
			adj_cv_by_sf[float(sf)] = float(np.nanstd(dists, ddof=1) / mean_d)
	return adj_cv_by_sf


def _compute_angle_spacing_rmse_norm_by_sf_from_dspca_npz(dspca_npz: "np.lib.npyio.NpzFile") -> Dict[float, float]:
	"""Compute normalized angle-spacing RMSE per SF from dsPCA results."""
	if dspca_npz is None:
		return {}
	if ("condition_orientations" not in dspca_npz.files) or ("condition_spatialfreqs" not in dspca_npz.files):
		return {}
	if "projection_target_test" in dspca_npz.files:
		proj = np.asarray(dspca_npz["projection_target_test"], dtype=float)
	elif "projection_target" in dspca_npz.files:
		proj = np.asarray(dspca_npz["projection_target"], dtype=float)
	else:
		return {}
	oris = np.asarray(dspca_npz["condition_orientations"], dtype=float)
	sfs = np.asarray(dspca_npz["condition_spatialfreqs"], dtype=float)
	if proj.ndim != 2 or proj.shape[1] < 3:
		return {}

	oris_q = np.mod(np.round(oris / 30.0) * 30.0, 180.0)
	unique_sfs = np.unique(sfs[np.isfinite(sfs) & (sfs > 0)])
	angle_rmse_by_sf: Dict[float, float] = {}
	for sf in unique_sfs:
		mask_sf = np.isfinite(sfs) & np.isclose(sfs, float(sf), atol=1e-12)
		if not np.any(mask_sf):
			continue
		pts_by_ori: List[np.ndarray] = []
		for ori in np.arange(0, 180, 30, dtype=float):
			mask = mask_sf & np.isclose(oris_q, ori, atol=1e-6)
			if not np.any(mask):
				continue
			xy = np.nanmean(proj[mask, 1:3], axis=0)
			pts_by_ori.append(np.asarray(xy, dtype=float))
		if len(pts_by_ori) < 3:
			angle_rmse_by_sf[float(sf)] = np.nan
			continue
		pts = np.vstack(pts_by_ori)
		geom = circle_geometry_metrics(pts[:, 0], pts[:, 1])
		angle_rmse_by_sf[float(sf)] = float(geom.get("angle_spacing_rmse_norm", np.nan))
	return angle_rmse_by_sf


def _load_session_orientation_plane_points(
	session_rel: str,
	*,
	cv50: bool,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
	"""Load per-condition dsPCA orientation-plane points for one session.

	Returns:
		(oris_q, sfs, plane_xy)
	where:
	- oris_q is quantized orientation in degrees (0..150 step 30)
	- sfs is mapped spatial frequency per condition (0.02..0.64)
	- plane_xy is (n_conditions, 2) = projection_target[:, 1:3]

	Uses held-out CV projections when cv50=True and available.
	"""
	session_dir = os.path.join(RESULTS_ROOT, session_rel)
	fn = "dspca_results_cv50.npz" if cv50 else "dspca_results.npz"
	npz = safe_load_npz(os.path.join(session_dir, fn))
	if npz is None:
		return None
	try:
		if ("condition_orientations" not in npz.files) or ("condition_spatialfreqs" not in npz.files):
			npz.close()
			return None
		if cv50 and ("projection_target_test" in npz.files):
			proj = np.asarray(npz["projection_target_test"], dtype=float)
		elif "projection_target" in npz.files:
			proj = np.asarray(npz["projection_target"], dtype=float)
		else:
			npz.close()
			return None
		oris = np.asarray(npz["condition_orientations"], dtype=float)
		sfs = np.asarray(npz["condition_spatialfreqs"], dtype=float)
	finally:
		try:
			npz.close()
		except Exception:
			pass

	if proj.ndim != 2 or proj.shape[1] < 3:
		return None
	plane_xy = np.asarray(proj[:, 1:3], dtype=float)
	oris_q = np.mod(np.round(oris / 30.0) * 30.0, 180.0)
	return oris_q, sfs, plane_xy


def _session_points_by_sf_ori(
	session_rel: str,
	*,
	cv50: bool,
	mapped_levels: List[float],
	ori_levels: np.ndarray,
) -> Optional[Dict[float, Dict[float, np.ndarray]]]:
	"""Return per-session centered points dict: sf -> ori -> (x,y)."""
	loaded = _load_session_orientation_plane_points(session_rel, cv50=cv50)
	if loaded is None:
		return None
	oris_q, sfs, xy = loaded
	out: Dict[float, Dict[float, np.ndarray]] = {}
	for sf in mapped_levels:
		mask_sf = np.isfinite(sfs) & np.isclose(sfs, float(sf), atol=1e-12)
		if not np.any(mask_sf):
			continue
		pts: Dict[float, np.ndarray] = {}
		for ori in ori_levels:
			mask = mask_sf & np.isclose(oris_q, float(ori), atol=1e-6)
			if not np.any(mask):
				continue
			pts[float(ori)] = np.nanmean(xy[mask], axis=0)
		if not pts:
			continue
		# Center this session's SF polygon at origin to avoid translation smearing.
		stack = np.vstack([v for v in pts.values() if v is not None and np.all(np.isfinite(v))])
		if stack.size:
			com = np.nanmean(stack, axis=0)
		else:
			com = np.array([0.0, 0.0], dtype=float)
		for ori in list(pts.keys()):
			pts[ori] = np.asarray(pts[ori], dtype=float) - com
		out[float(sf)] = pts
	return out if out else None


def _session_com_xy_by_sf(
	session_rel: str,
	*,
	cv50: bool,
	mapped_levels: List[float],
	ori_levels: np.ndarray,
) -> np.ndarray:
	"""Return per-SF COM coordinates using pre-alignment orientation-plane points."""
	loaded = _load_session_orientation_plane_points(session_rel, cv50=cv50)
	out = np.full((len(mapped_levels), 2), np.nan, dtype=float)
	if loaded is None:
		return out
	oris_q, sfs, xy = loaded
	for sf_i, sf in enumerate(mapped_levels):
		mask_sf = np.isfinite(sfs) & np.isclose(sfs, float(sf), atol=1e-12)
		if not np.any(mask_sf):
			continue
		pts: List[np.ndarray] = []
		for ori in ori_levels:
			mask = mask_sf & np.isclose(oris_q, float(ori), atol=1e-6)
			if not np.any(mask):
				continue
			pts.append(np.nanmean(xy[mask], axis=0))
		if not pts:
			continue
		stack = np.vstack([np.asarray(v, dtype=float) for v in pts if v is not None and np.all(np.isfinite(v))])
		if stack.size == 0:
			continue
		com = np.nanmean(stack, axis=0)
		if np.all(np.isfinite(com)):
			out[sf_i, :] = np.asarray(com, dtype=float)
	return out


def _session_com_distance_from_origin_by_sf(
	session_rel: str,
	*,
	cv50: bool,
	mapped_levels: List[float],
	ori_levels: np.ndarray,
) -> np.ndarray:
	"""Return per-SF COM distance from origin using pre-alignment coordinates."""
	coords = _session_com_xy_by_sf(
		session_rel,
		cv50=cv50,
		mapped_levels=mapped_levels,
		ori_levels=ori_levels,
	)
	out = np.full(len(mapped_levels), np.nan, dtype=float)
	finite = np.all(np.isfinite(coords), axis=1)
	if np.any(finite):
		out[finite] = np.sqrt(np.sum(np.square(coords[finite, :]), axis=1))
	return out


def _session_mean_sf_axis_by_sf(
	session_rel: str,
	*,
	cv50: bool,
	mapped_levels: List[float],
) -> np.ndarray:
	"""Return per-SF mean dsPCA SF-axis projection averaged across all orientations."""
	session_dir = os.path.join(RESULTS_ROOT, session_rel)
	fn = "dspca_results_cv50.npz" if cv50 else "dspca_results.npz"
	npz = safe_load_npz(os.path.join(session_dir, fn))
	out = np.full(len(mapped_levels), np.nan, dtype=float)
	if npz is None:
		return out
	try:
		if "condition_spatialfreqs" not in npz.files:
			return out
		if cv50 and ("projection_target_test" in npz.files):
			proj = np.asarray(npz["projection_target_test"], dtype=float)
		elif "projection_target" in npz.files:
			proj = np.asarray(npz["projection_target"], dtype=float)
		else:
			return out
		sfs = np.asarray(npz["condition_spatialfreqs"], dtype=float)
	finally:
		try:
			npz.close()
		except Exception:
			pass

	if proj.ndim != 2 or proj.shape[1] < 1:
		return out
	for sf_i, sf in enumerate(mapped_levels):
		mask_sf = np.isfinite(sfs) & np.isclose(sfs, float(sf), atol=1e-12)
		if not np.any(mask_sf):
			continue
		vals = np.asarray(proj[mask_sf, 0], dtype=float)
		vals = vals[np.isfinite(vals)]
		if vals.size:
			out[sf_i] = float(np.nanmean(vals))
	return out


def _session_adjacent_sf_axis_distance_by_pair(
	session_rel: str,
	*,
	cv50: bool,
	pair_levels: List[Tuple[float, float]],
) -> np.ndarray:
	"""Return adjacent-SF separations along the dsPCA SF axis for one session."""
	endpoint_levels = sorted({float(v) for pair in pair_levels for v in pair})
	mean_by_sf = _session_mean_sf_axis_by_sf(
		session_rel,
		cv50=cv50,
		mapped_levels=endpoint_levels,
	)
	level_to_idx = {float(sf): idx for idx, sf in enumerate(endpoint_levels)}
	out = np.full(len(pair_levels), np.nan, dtype=float)
	for pair_i, (sf_lo, sf_hi) in enumerate(pair_levels):
		idx_lo = level_to_idx.get(float(sf_lo))
		idx_hi = level_to_idx.get(float(sf_hi))
		if idx_lo is None or idx_hi is None:
			continue
		val_lo = mean_by_sf[idx_lo]
		val_hi = mean_by_sf[idx_hi]
		if np.isfinite(val_lo) and np.isfinite(val_hi):
			out[pair_i] = float(abs(val_hi - val_lo))
	return out


def _plot_sessionavg_dspca_orient_plane_by_sf_overlay(
	*,
	before_sessions: List[str],
	after_sessions: List[str],
	before_label: str,
	after_label: str,
	figure_title: str,
	before_line_color: str,
	after_line_color: str,
	middle_sessions: Optional[List[str]] = None,
	middle_label: Optional[str] = None,
	middle_line_color: Optional[str] = None,
	before_edge_color: Optional[str] = None,
	after_edge_color: Optional[str] = None,
	middle_edge_color: Optional[str] = None,
	cv50: bool = False,
	outfile: Optional[str] = None,
) -> None:
	"""Plot a session-averaged dsPCA orientation-plane overlay for two or three session sets."""
	if not before_sessions or not after_sessions:
		print(f"[skip] Missing sessions for {before_label} vs {after_label}")
		return
	if middle_sessions is not None and not middle_sessions:
		print(f"[skip] Missing sessions for {middle_label}")
		return

	# Use mapped SF levels for subplots (0.02..0.64)
	mapped_levels = [float(convert_sf_value(float(sf))) for sf in TARGET_SF_VALUES if float(sf) > 0]
	mapped_levels = sorted(list({sf for sf in mapped_levels if sf > 0}))
	ori_levels = np.arange(0, 180, 30, dtype=float)

	# Orientation palette
	p_palette = plt.cm.turbo(np.linspace(0, 1, len(ori_levels) + 1)[:-1])
	ori_to_color = {float(ori): p_palette[i] for i, ori in enumerate(ori_levels)}
	before_edge_color = before_edge_color or before_line_color
	after_edge_color = after_edge_color or after_line_color
	if middle_sessions is not None:
		middle_line_color = middle_line_color or 'black'
		middle_edge_color = middle_edge_color or middle_line_color

	# Collect per-session points
	before_pts_by_sess: List[Tuple[str, Dict[float, Dict[float, np.ndarray]]]] = []
	for session_rel in before_sessions:
		pts = _session_points_by_sf_ori(session_rel, cv50=cv50, mapped_levels=mapped_levels, ori_levels=ori_levels)
		if pts is not None:
			before_pts_by_sess.append((session_rel, pts))
	after_pts_by_sess: List[Tuple[str, Dict[float, Dict[float, np.ndarray]]]] = []
	for session_rel in after_sessions:
		pts = _session_points_by_sf_ori(session_rel, cv50=cv50, mapped_levels=mapped_levels, ori_levels=ori_levels)
		if pts is not None:
			after_pts_by_sess.append((session_rel, pts))
	middle_pts_by_sess: List[Tuple[str, Dict[float, Dict[float, np.ndarray]]]] = []
	if middle_sessions is not None:
		for session_rel in middle_sessions:
			pts = _session_points_by_sf_ori(session_rel, cv50=cv50, mapped_levels=mapped_levels, ori_levels=ori_levels)
			if pts is not None:
				middle_pts_by_sess.append((session_rel, pts))

	if not before_pts_by_sess or not after_pts_by_sess:
		note = "10-fold CV" if cv50 else "standard"
		print(f"[skip] No usable per-session dsPCA results for {before_label} vs {after_label} ({note})")
		return
	if middle_sessions is not None and not middle_pts_by_sess:
		note = "10-fold CV" if cv50 else "standard"
		print(f"[skip] No usable per-session dsPCA results for {middle_label} ({note})")
		return

	fig, axes = plt.subplots(2, 3, figsize=(15, 10))
	axes = axes.flatten()

	for i, sf_val in enumerate(mapped_levels[: len(axes)]):
		ax = axes[i]
		ax.set_title(f"SF {sf_val:.2f}")
		ax.set_xlabel('sin(2θ)')
		ax.set_ylabel('cos(2θ)')
		ax.set_aspect('equal')
		ax.set_xlim([-0.7, 0.7])
		ax.set_ylim([-0.7, 0.7])
		ax.axhline(0, color='gray', alpha=0.2, linewidth=0.8)
		ax.axvline(0, color='gray', alpha=0.2, linewidth=0.8)
		ax.grid(alpha=0.3)

		# Build session stacks: (n_sessions, n_ori, 2)
		b_stack = np.full((len(before_pts_by_sess), len(ori_levels), 2), np.nan, dtype=float)
		for si, (_session_rel, per_sf) in enumerate(before_pts_by_sess):
			pts = per_sf.get(float(sf_val), {})
			for oi, ori in enumerate(ori_levels):
				if float(ori) in pts and pts[float(ori)] is not None:
					b_stack[si, oi, :] = np.asarray(pts[float(ori)], dtype=float)
		a_stack = np.full((len(after_pts_by_sess), len(ori_levels), 2), np.nan, dtype=float)
		for si, (_session_rel, per_sf) in enumerate(after_pts_by_sess):
			pts = per_sf.get(float(sf_val), {})
			for oi, ori in enumerate(ori_levels):
				if float(ori) in pts and pts[float(ori)] is not None:
					a_stack[si, oi, :] = np.asarray(pts[float(ori)], dtype=float)
		m_stack = None
		if middle_sessions is not None:
			m_stack = np.full((len(middle_pts_by_sess), len(ori_levels), 2), np.nan, dtype=float)
			for si, (_session_rel, per_sf) in enumerate(middle_pts_by_sess):
				pts = per_sf.get(float(sf_val), {})
				for oi, ori in enumerate(ori_levels):
					if float(ori) in pts and pts[float(ori)] is not None:
						m_stack[si, oi, :] = np.asarray(pts[float(ori)], dtype=float)

		# Mean/SEM per orientation
		b_mean = np.nanmean(b_stack, axis=0)
		b_sem = np.nanstd(b_stack, axis=0, ddof=1) / np.sqrt(np.maximum(np.sum(np.isfinite(b_stack[..., 0]), axis=0), 1))[:, None]
		a_mean = np.nanmean(a_stack, axis=0)
		a_sem = np.nanstd(a_stack, axis=0, ddof=1) / np.sqrt(np.maximum(np.sum(np.isfinite(a_stack[..., 0]), axis=0), 1))[:, None]
		m_mean = None
		m_sem = None
		if m_stack is not None:
			m_mean = np.nanmean(m_stack, axis=0)
			m_sem = np.nanstd(m_stack, axis=0, ddof=1) / np.sqrt(np.maximum(np.sum(np.isfinite(m_stack[..., 0]), axis=0), 1))[:, None]

		# Plot SEM circles + mean points
		for oi, ori in enumerate(ori_levels):
			c = ori_to_color[float(ori)]
			if np.all(np.isfinite(b_mean[oi])):
				r = float(np.sqrt(np.nansum(np.square(b_sem[oi])))) if np.all(np.isfinite(b_sem[oi])) else np.nan
				if np.isfinite(r) and r > 0:
					ax.add_patch(Circle(tuple(b_mean[oi]), radius=r, facecolor=c, edgecolor=before_edge_color, linewidth=0.6, alpha=0.16))
				ax.scatter([b_mean[oi, 0]], [b_mean[oi, 1]], s=55, c=[c], edgecolors=before_edge_color, linewidths=0.8, zorder=3)
			if np.all(np.isfinite(a_mean[oi])):
				r = float(np.sqrt(np.nansum(np.square(a_sem[oi])))) if np.all(np.isfinite(a_sem[oi])) else np.nan
				if np.isfinite(r) and r > 0:
					ax.add_patch(Circle(tuple(a_mean[oi]), radius=r, facecolor=c, edgecolor=after_edge_color, linewidth=0.8, alpha=0.10))
				ax.scatter([a_mean[oi, 0]], [a_mean[oi, 1]], s=65, c=[c], edgecolors=after_edge_color, linewidths=1.0, zorder=4)
			if m_mean is not None and np.all(np.isfinite(m_mean[oi])):
				r = float(np.sqrt(np.nansum(np.square(m_sem[oi])))) if np.all(np.isfinite(m_sem[oi])) else np.nan
				if np.isfinite(r) and r > 0:
					ax.add_patch(Circle(tuple(m_mean[oi]), radius=r, facecolor=c, edgecolor=middle_edge_color, linewidth=0.7, alpha=0.13))
				ax.scatter([m_mean[oi, 0]], [m_mean[oi, 1]], s=60, c=[c], edgecolors=middle_edge_color, linewidths=0.9, zorder=4)

		def _connect(mean_xy: np.ndarray, *, color: str) -> None:
			valid = np.all(np.isfinite(mean_xy), axis=1)
			idx = np.where(valid)[0]
			if idx.size < 2:
				return
			for k in range(idx.size - 1):
				i0, i1 = idx[k], idx[k + 1]
				ax.plot([mean_xy[i0, 0], mean_xy[i1, 0]], [mean_xy[i0, 1], mean_xy[i1, 1]], '-', color=color, alpha=0.55, linewidth=1.2)
			if idx.size == len(ori_levels):
				ax.plot([mean_xy[idx[-1], 0], mean_xy[idx[0], 0]], [mean_xy[idx[-1], 1], mean_xy[idx[0], 1]], '-', color=color, alpha=0.55, linewidth=1.2)

		_connect(b_mean, color=before_line_color)
		_connect(a_mean, color=after_line_color)
		if m_mean is not None:
			_connect(m_mean, color=middle_line_color)

	for k in range(len(mapped_levels[: len(axes)]), len(axes)):
		axes[k].axis('off')

	handles = [
		plt.Line2D([0], [0], color=before_line_color, lw=1.5, label=before_label),
	]
	if middle_sessions is not None:
		handles.append(plt.Line2D([0], [0], color=middle_line_color, lw=1.5, label=middle_label))
	handles.extend([
		plt.Line2D([0], [0], color=after_line_color, lw=1.5, label=after_label),
	])
	fig.legend(handles=handles, loc='upper right')
	mode = "10-fold CV" if cv50 else "standard"
	fig.suptitle(f'{figure_title} ({mode})', fontsize=14, fontweight='bold')
	plt.tight_layout()
	if outfile is None:
		print(f"[skip] No output filename specified for {before_label} vs {after_label}")
		plt.close(fig)
		return
	plt.savefig(os.path.join(GROUPED_PLOTS_DIR, outfile), format='svg', bbox_inches='tight')
	plt.close(fig)
	print(f"  [saved] {outfile}")


def plot_sessionavg_dspca_com_distance_from_origin_by_sf_all_groups(
	*,
	cv50: bool = False,
	outfile: Optional[str] = None,
) -> None:
	"""Plot session-mean pre-aligned COM distance summaries across all four groups."""
	mapped_levels = [float(convert_sf_value(float(sf))) for sf in TARGET_SF_VALUES if float(sf) > 0]
	mapped_levels = sorted(list({sf for sf in mapped_levels if sf > 0}))
	ori_levels = np.arange(0, 180, 30, dtype=float)
	if not mapped_levels:
		print("[skip] No nonzero SF levels available for COM-distance summary")
		return

	group_specs = [
		("WT_BEFORE", GROUP_LABELS.get("WT_BEFORE", "WT pre-training"), GROUP_COLORS.get("WT_BEFORE", "#9ecae1"), "o"),
		("WT_AFTER", GROUP_LABELS.get("WT_AFTER", "WT post-training"), GROUP_COLORS.get("WT_AFTER", "#08306b"), "o"),
		("RESCUED_BEFORE", GROUP_LABELS.get("RESCUED_BEFORE", "Rescued pre-training"), GROUP_COLORS.get("RESCUED_BEFORE", "#fcae91"), "s"),
		("RESCUED_AFTER", GROUP_LABELS.get("RESCUED_AFTER", "Rescued post-training"), GROUP_COLORS.get("RESCUED_AFTER", "#a50f15"), "s"),
	]
	ref_idx_064 = int(np.argmin(np.abs(np.asarray(mapped_levels, dtype=float) - 0.64)))
	ref_idx_002 = int(np.argmin(np.abs(np.asarray(mapped_levels, dtype=float) - 0.02)))
	ref_idx_008 = int(np.argmin(np.abs(np.asarray(mapped_levels, dtype=float) - 0.08)))

	fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True)
	ax_origin, ax_ref_064, ax_ref_002, ax_ref_008, ax_signed_064, ax_signed_002, ax_signed_008, ax_unused = axes.flatten()

	def _dist_stack_from_ref(coords_stack: np.ndarray, ref_idx: int) -> np.ndarray:
		out = np.full((coords_stack.shape[0], len(mapped_levels)), np.nan, dtype=float)
		for si in range(coords_stack.shape[0]):
			ref_xy = coords_stack[si, ref_idx, :]
			if not np.all(np.isfinite(ref_xy)):
				continue
			deltas = coords_stack[si, :, :] - ref_xy[None, :]
			finite_delta = np.all(np.isfinite(deltas), axis=1)
			if np.any(finite_delta):
				out[si, finite_delta] = np.sqrt(np.sum(np.square(deltas[finite_delta, :]), axis=1))
			out[si, ref_idx] = 0.0
		return out

	def _signed_proj_stack_from_ref(coords_stack: np.ndarray, ref_idx: int, *, axis_lo_idx: int, axis_hi_idx: int) -> np.ndarray:
		out = np.full((coords_stack.shape[0], len(mapped_levels)), np.nan, dtype=float)
		for si in range(coords_stack.shape[0]):
			lo_xy = coords_stack[si, axis_lo_idx, :]
			hi_xy = coords_stack[si, axis_hi_idx, :]
			ref_xy = coords_stack[si, ref_idx, :]
			if not (np.all(np.isfinite(lo_xy)) and np.all(np.isfinite(hi_xy)) and np.all(np.isfinite(ref_xy))):
				continue
			axis_vec = hi_xy - lo_xy
			axis_norm = float(np.sqrt(np.sum(np.square(axis_vec))))
			if not np.isfinite(axis_norm) or axis_norm <= 1e-12:
				continue
			unit_vec = axis_vec / axis_norm
			deltas = coords_stack[si, :, :] - ref_xy[None, :]
			finite_delta = np.all(np.isfinite(deltas), axis=1)
			if np.any(finite_delta):
				out[si, finite_delta] = deltas[finite_delta, :] @ unit_vec
			out[si, ref_idx] = 0.0
		return out

	has_data = False
	for group_name, label, color, marker in group_specs:
		sessions = list(GROUPS.get(group_name, []))
		if not sessions:
			continue
		coords_stack = np.stack([
			_session_com_xy_by_sf(
				session_rel,
				cv50=cv50,
				mapped_levels=mapped_levels,
				ori_levels=ori_levels,
			)
			for session_rel in sessions
		], axis=0)
		origin_stack = np.full((coords_stack.shape[0], len(mapped_levels)), np.nan, dtype=float)
		finite_origin = np.all(np.isfinite(coords_stack), axis=2)
		if np.any(finite_origin):
			origin_stack[finite_origin] = np.sqrt(np.sum(np.square(coords_stack[finite_origin]), axis=1))
		ref_stack_064 = _dist_stack_from_ref(coords_stack, ref_idx_064)
		ref_stack_002 = _dist_stack_from_ref(coords_stack, ref_idx_002)
		ref_stack_008 = _dist_stack_from_ref(coords_stack, ref_idx_008)
		signed_stack_064 = _signed_proj_stack_from_ref(coords_stack, ref_idx_064, axis_lo_idx=ref_idx_002, axis_hi_idx=ref_idx_064)
		signed_stack_002 = _signed_proj_stack_from_ref(coords_stack, ref_idx_002, axis_lo_idx=ref_idx_002, axis_hi_idx=ref_idx_064)
		signed_stack_008 = _signed_proj_stack_from_ref(coords_stack, ref_idx_008, axis_lo_idx=ref_idx_002, axis_hi_idx=ref_idx_064)

		mean_origin, sem_origin = nan_mean_sem(origin_stack, axis=0)
		mean_ref_064, sem_ref_064 = nan_mean_sem(ref_stack_064, axis=0)
		mean_ref_002, sem_ref_002 = nan_mean_sem(ref_stack_002, axis=0)
		mean_ref_008, sem_ref_008 = nan_mean_sem(ref_stack_008, axis=0)
		mean_signed_064, sem_signed_064 = nan_mean_sem(signed_stack_064, axis=0)
		mean_signed_002, sem_signed_002 = nan_mean_sem(signed_stack_002, axis=0)
		mean_signed_008, sem_signed_008 = nan_mean_sem(signed_stack_008, axis=0)
		if not (
			np.any(np.isfinite(mean_origin))
			or np.any(np.isfinite(mean_ref_064))
			or np.any(np.isfinite(mean_ref_002))
			or np.any(np.isfinite(mean_ref_008))
			or np.any(np.isfinite(mean_signed_064))
			or np.any(np.isfinite(mean_signed_002))
			or np.any(np.isfinite(mean_signed_008))
		):
			continue
		has_data = True
		ax_origin.errorbar(
			mapped_levels,
			mean_origin,
			yerr=sem_origin,
			fmt=f'-{marker}',
			capsize=4,
			color=color,
			label=label,
			linewidth=2,
			markersize=6,
			alpha=0.9,
		)
		ax_ref_064.errorbar(
			mapped_levels,
			mean_ref_064,
			yerr=sem_ref_064,
			fmt=f'-{marker}',
			capsize=4,
			color=color,
			label=label,
			linewidth=2,
			markersize=6,
			alpha=0.9,
		)
		ax_ref_002.errorbar(
			mapped_levels,
			mean_ref_002,
			yerr=sem_ref_002,
			fmt=f'-{marker}',
			capsize=4,
			color=color,
			label=label,
			linewidth=2,
			markersize=6,
			alpha=0.9,
		)
		ax_ref_008.errorbar(
			mapped_levels,
			mean_ref_008,
			yerr=sem_ref_008,
			fmt=f'-{marker}',
			capsize=4,
			color=color,
			label=label,
			linewidth=2,
			markersize=6,
			alpha=0.9,
		)
		ax_signed_064.errorbar(
			mapped_levels,
			mean_signed_064,
			yerr=sem_signed_064,
			fmt=f'-{marker}',
			capsize=4,
			color=color,
			label=label,
			linewidth=2,
			markersize=6,
			alpha=0.9,
		)
		ax_signed_002.errorbar(
			mapped_levels,
			mean_signed_002,
			yerr=sem_signed_002,
			fmt=f'-{marker}',
			capsize=4,
			color=color,
			label=label,
			linewidth=2,
			markersize=6,
			alpha=0.9,
		)
		ax_signed_008.errorbar(
			mapped_levels,
			mean_signed_008,
			yerr=sem_signed_008,
			fmt=f'-{marker}',
			capsize=4,
			color=color,
			label=label,
			linewidth=2,
			markersize=6,
			alpha=0.9,
		)

	if not has_data:
		plt.close(fig)
		note = "10-fold CV" if cv50 else "standard"
		print(f"[skip] No usable per-session COM-distance data ({note})")
		return

	for ax in axes.flatten()[:-1]:
		ax.set_xscale('log')
		ax.set_xticks(mapped_levels)
		ax.set_xticklabels([f"{sf:.2f}" for sf in mapped_levels], rotation=45)
		ax.set_xlabel('Spatial frequency')
		ax.grid(alpha=0.3)
	ax_unused.axis('off')
	ax_origin.set_ylabel('XY distance')
	ax_origin.set_title('COM distance from origin')
	ax_ref_064.set_ylabel('XY distance from COM at SF=0.64')
	ax_ref_064.set_title('COM distance from COM at SF=0.64')
	ax_ref_002.set_ylabel('XY distance from COM at SF=0.02')
	ax_ref_002.set_title('COM distance from COM at SF=0.02')
	ax_ref_008.set_ylabel('XY distance from COM at SF=0.08')
	ax_ref_008.set_title('COM distance from COM at SF=0.08')
	ax_signed_064.set_ylabel('Signed projection')
	ax_signed_064.set_title('Signed projection from SF=0.64')
	ax_signed_002.set_ylabel('Signed projection')
	ax_signed_002.set_title('Signed projection from SF=0.02')
	ax_signed_008.set_ylabel('Signed projection')
	ax_signed_008.set_title('Signed projection from SF=0.08')
	mode = "10-fold CV" if cv50 else "standard"
	fig.suptitle(f'Session-averaged dsPCA COM-distance summaries by SF ({mode})')
	fig.legend(loc='upper center', ncol=4)
	plt.tight_layout(rect=[0, 0, 1, 0.93])
	if outfile is None:
		suffix = "_10fcv" if cv50 else ""
		outfile = f'sessionavg_dspca_com_distance_from_origin_by_sf_all_groups{suffix}.svg'
	plt.savefig(os.path.join(GROUPED_PLOTS_DIR, outfile), format='svg', bbox_inches='tight')
	plt.close(fig)
	print(f"  [saved] {outfile}")


def plot_sessionavg_dspca_orient_plane_by_sf_superimposed(
	*,
	group_label: str,
	cv50: bool = False,
	outfile: Optional[str] = None,
) -> None:
	"""Session-averaged version of the pseudopop 'orient plane by SF superimposed' plot.

	Computes mean (across sessions) dsPCA orientation-plane points per (SF, ori),
	connects adjacent orientations, and shows SEM as orientation-colored circles.
	"""
	group_sets = {
		"WT": ("WT_BEFORE", "WT_AFTER"),
		"RESCUED": ("RESCUED_BEFORE", "RESCUED_AFTER"),
	}
	if group_label not in group_sets:
		print(f"[skip] Unknown group_label={group_label}")
		return
	gn_before, gn_after = group_sets[group_label]
	before_sessions = list(GROUPS.get(gn_before, []))
	after_sessions = list(GROUPS.get(gn_after, []))
	if not before_sessions or not after_sessions:
		print(f"[skip] Missing sessions for {group_label}")
		return

	_plot_sessionavg_dspca_orient_plane_by_sf_overlay(
		before_sessions=before_sessions,
		after_sessions=after_sessions,
		before_label=f'{group_label} Before (session mean)',
		after_label=f'{group_label} After (session mean)',
		figure_title=f'{group_label}: Session-averaged dsPCA orientation plane per SF',
		before_line_color='black',
		after_line_color='red',
		cv50=cv50,
		outfile=outfile,
	)


def plot_sessionavg_dspca_adjacent_sf_axis_distance_all_groups(
	*,
	cv50: bool = False,
	outfile: Optional[str] = None,
) -> None:
	"""Plot mean±SEM adjacent-SF separations along the dsPCA SF axis for all four groups."""
	# Use the four adjacent pairs explicitly requested by the user.
	pair_levels: List[Tuple[float, float]] = [
		(0.02, 0.04),
		(0.04, 0.08),
		(0.08, 0.16),
		(0.16, 0.32),
		(0.32, 0.64),
	]
	pair_labels = [f"{lo:.2f}-{hi:.2f}" for lo, hi in pair_levels]
	group_specs = [
		("WT_BEFORE", GROUP_LABELS.get("WT_BEFORE", "WT pre-training"), GROUP_COLORS.get("WT_BEFORE", "#9ecae1"), "o"),
		("WT_AFTER", GROUP_LABELS.get("WT_AFTER", "WT post-training"), GROUP_COLORS.get("WT_AFTER", "#08306b"), "o"),
		("RESCUED_BEFORE", GROUP_LABELS.get("RESCUED_BEFORE", "Rescued pre-training"), GROUP_COLORS.get("RESCUED_BEFORE", "#fcae91"), "s"),
		("RESCUED_AFTER", GROUP_LABELS.get("RESCUED_AFTER", "Rescued post-training"), GROUP_COLORS.get("RESCUED_AFTER", "#a50f15"), "s"),
	]
	x = np.arange(len(pair_levels), dtype=float)
	fig, ax = plt.subplots(figsize=(9, 5))
	has_data = False

	for group_name, label, color, marker in group_specs:
		sessions = list(GROUPS.get(group_name, []))
		if not sessions:
			continue
		stack = np.stack([
			_session_adjacent_sf_axis_distance_by_pair(
				session_rel,
				cv50=cv50,
				pair_levels=pair_levels,
			)
			for session_rel in sessions
		], axis=0)
		mean_v, sem_v = nan_mean_sem(stack, axis=0)
		if not np.any(np.isfinite(mean_v)):
			continue
		has_data = True
		ax.errorbar(
			x,
			mean_v,
			yerr=sem_v,
			fmt=f'-{marker}',
			capsize=4,
			color=color,
			label=label,
			linewidth=2,
			markersize=6,
			alpha=0.9,
		)

	if not has_data:
		plt.close(fig)
		note = "10-fold CV" if cv50 else "standard"
		print(f"[skip] No usable adjacent SF-axis separation data ({note})")
		return

	ax.set_xticks(x)
	ax.set_xticklabels(pair_labels, rotation=45, ha='right')
	ax.set_xlabel('Adjacent SF pair')
	ax.set_ylabel('Distance')
	ax.set_title(
		'Adjacent SF separations along dsPCA SF axis'
		+ (' (10-fold CV)' if cv50 else '')
	)
	ax.grid(alpha=0.3)
	ax.legend(fontsize=9)
	plt.tight_layout()
	if outfile is None:
		suffix = "_10fcv" if cv50 else ""
		outfile = f'sessionavg_dspca_adjacent_sf_axis_distance_all_groups{suffix}.svg'
	plt.savefig(os.path.join(GROUPED_PLOTS_DIR, outfile), format='svg', bbox_inches='tight')
	plt.close(fig)
	print(f"  [saved] {outfile}")


def plot_sessionavg_dspca_orient_plane_by_sf_superimposed_rescued_after_with_wt_before(
	*,
	cv50: bool = False,
	outfile: Optional[str] = None,
) -> None:
	"""Session-averaged dsPCA orientation-plane overlay for WT before plus rescued before/after."""
	before_sessions = list(GROUPS.get("WT_BEFORE", []))
	middle_sessions = list(GROUPS.get("RESCUED_BEFORE", []))
	after_sessions = list(GROUPS.get("RESCUED_AFTER", []))
	if outfile is None:
		suffix = "_10fcv" if cv50 else ""
		outfile = f'sessionavg_dspca_orient_plane_by_sf_superimposed_RESCUED_with_WT_BEFORE{suffix}.svg'
	_plot_sessionavg_dspca_orient_plane_by_sf_overlay(
		before_sessions=before_sessions,
		middle_sessions=middle_sessions,
		after_sessions=after_sessions,
		before_label='WT Before (session mean)',
		middle_label='Rescued Before (session mean)',
		after_label='Rescued After (session mean)',
		figure_title='WT Before vs Rescued Before/After: Session-averaged dsPCA orientation plane per SF',
		before_line_color='blue',
		middle_line_color='black',
		after_line_color='red',
		cv50=cv50,
		outfile=outfile,
	)



def plot_orientation_area_by_sf_paired(
	*,
	out_file: str = "dspca_orientation_area_by_sf_paired.svg",
	dspca_filename: str = "dspca_results.npz",
	title_suffix: str = "",
) -> None:
	"""Plot mean±SEM orientation-plane area vs SF, paired Before/After (WT and RESCUED).

	Uses per-session dsPCA results saved as RESULTS_ROOT/<session>/<dspca_filename>.
	"""
	# Use mapped SF levels for x-axis (0.02..0.64)
	mapped_levels = [float(convert_sf_value(float(sf))) for sf in TARGET_SF_VALUES if float(sf) > 0]
	mapped_levels = sorted(list({sf for sf in mapped_levels if sf > 0}))
	if not mapped_levels:
		print("[skip] No nonzero SF levels available for orientation area plot")
		return

	def _load_area_vec(session_rel: str) -> np.ndarray:
		path = os.path.join(RESULTS_ROOT, session_rel, dspca_filename)
		if not os.path.exists(path):
			return np.full(len(mapped_levels), np.nan, dtype=float)
		npz = safe_load_npz(path)
		if npz is None:
			return np.full(len(mapped_levels), np.nan, dtype=float)
		area_map = compute_orientation_area_by_sf_from_dspca_npz(npz)
		out = np.full(len(mapped_levels), np.nan, dtype=float)
		for i, sf in enumerate(mapped_levels):
			# tolerate float rounding
			best = None
			best_err = float('inf')
			for k, v in area_map.items():
				err = abs(float(k) - float(sf))
				if err < best_err:
					best_err = err
					best = v
			out[i] = float(best) if (best is not None and best_err < 1e-6) else np.nan
		return out

	def _paired_matrix(pairs: List[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
		before_list = []
		after_list = []
		for s_before, s_after in pairs:
			before_list.append(_load_area_vec(s_before))
			after_list.append(_load_area_vec(s_after))
		return np.vstack(before_list) if before_list else np.empty((0, len(mapped_levels))), np.vstack(after_list) if after_list else np.empty((0, len(mapped_levels)))

	def _nansem(a: np.ndarray, axis: int = 0) -> np.ndarray:
		# Avoid ddof warnings when n<2 (SEM undefined); return NaN for those entries.
		a = np.asarray(a, dtype=float)
		valid_n = np.sum(np.isfinite(a), axis=axis)
		out = np.full(a.shape[1 - axis], np.nan, dtype=float) if a.ndim == 2 else np.nan
		try:
			# axis is expected to be 0 here (per-SF SEM across mice)
			if a.ndim != 2 or axis != 0:
				with np.errstate(invalid='ignore', divide='ignore'):
					return np.nanstd(a, axis=axis, ddof=1) / np.sqrt(np.maximum(valid_n, 1))
			for j in range(a.shape[1]):
				n = int(valid_n[j])
				if n >= 2:
					out[j] = float(np.nanstd(a[:, j], ddof=1) / np.sqrt(n))
			return out
		except Exception:
			with np.errstate(invalid='ignore', divide='ignore'):
				return np.nanstd(a, axis=axis, ddof=1) / np.sqrt(np.maximum(valid_n, 1))

	# Build paired matrices
	pairs_wt = MOUSE_PAIRS.get("WT", [])
	pairs_resc = MOUSE_PAIRS.get("RESCUED", [])
	wt_b, wt_a = _paired_matrix(pairs_wt)
	rb_b, rb_a = _paired_matrix(pairs_resc)

	fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
	panels = [
		("WT", wt_b, wt_a, axes[0], "WT pre-training", "WT post-training", GROUP_COLORS.get("WT_BEFORE", "#9ecae1"), GROUP_COLORS.get("WT_AFTER", "#08306b")),
		("RESCUED", rb_b, rb_a, axes[1], "Rescued pre-training", "Rescued post-training", GROUP_COLORS.get("RESCUED_BEFORE", "#fcae91"), GROUP_COLORS.get("RESCUED_AFTER", "#a50f15")),
	]

	print("\n" + "=" * 70)
	print("dsPCA ORIENTATION AREA (convex hull) | Paired Before/After")
	print("=" * 70)

	for label, bmat, amat, ax, name_b, name_a, col_b, col_a in panels:
		ax.set_xscale('log')
		x = np.array(mapped_levels, dtype=float)
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=RuntimeWarning)
			mean_b = np.nanmean(bmat, axis=0) if bmat.size else np.full_like(x, np.nan)
			mean_a = np.nanmean(amat, axis=0) if amat.size else np.full_like(x, np.nan)
		sem_b = _nansem(bmat, axis=0) if bmat.size else np.full_like(x, np.nan)
		sem_a = _nansem(amat, axis=0) if amat.size else np.full_like(x, np.nan)

		# Plot with error bars (mean ± SEM)
		ax.errorbar(x, mean_b, yerr=sem_b, fmt='-o', capsize=3, color=col_b, ecolor=col_b, linewidth=2, markersize=6, label=name_b, alpha=0.9)
		ax.errorbar(x, mean_a, yerr=sem_a, fmt='-o', capsize=3, color=col_a, ecolor=col_a, linewidth=2, markersize=6, label=name_a, alpha=0.9)
		title = f"{label}: Orientation-plane area vs SF"
		if title_suffix:
			title += f" {title_suffix}"
		ax.set_title(title)
		ax.set_xlabel("Spatial frequency")
		ax.set_xticks(x)
		ax.set_xticklabels([f"{v:.2f}" for v in x], rotation=45, ha='right')
		ax.grid(alpha=0.3)
		ax.legend(fontsize=9)

		# Paired stats per SF + overall stats box (includes ART-Rlmer if available)
		overall_t_p: Optional[float] = None
		overall_w_p: Optional[float] = None
		art_rlmer_train_p: Optional[float] = None
		art_rlmer_sf_p: Optional[float] = None
		art_rlmer_int_p: Optional[float] = None
		try:
			from scipy.stats import ttest_rel, wilcoxon
			print(f"\n[{label}] paired stats per SF (Before vs After):")
			for i, sf in enumerate(mapped_levels):
				b = bmat[:, i] if bmat.size else np.array([], dtype=float)
				a = amat[:, i] if amat.size else np.array([], dtype=float)
				mask = np.isfinite(b) & np.isfinite(a)
				if np.sum(mask) < 2:
					print(f"  SF {sf:.2f}: n={int(np.sum(mask))} (skip)")
					continue
				t_res = ttest_rel(a[mask], b[mask], nan_policy='omit')
				# Wilcoxon may fail on ties/zeros; catch
				try:
					w_res = wilcoxon(a[mask], b[mask], zero_method='wilcox')
					w_p = float(w_res.pvalue)
				except Exception:
					w_p = float('nan')
				print(f"  SF {sf:.2f}: n={int(np.sum(mask))} | t-test p={float(t_res.pvalue):.4g} | wilcoxon p={w_p:.4g}")
			# Overall paired test on mean area across SFs
			if bmat.size and amat.size:
				with warnings.catch_warnings():
					warnings.filterwarnings("ignore", category=RuntimeWarning)
					b_mean = np.nanmean(bmat, axis=1)
					a_mean = np.nanmean(amat, axis=1)
				mask = np.isfinite(b_mean) & np.isfinite(a_mean)
				if np.sum(mask) >= 2:
					t_overall = ttest_rel(a_mean[mask], b_mean[mask], nan_policy='omit')
					overall_t_p = float(t_overall.pvalue) if np.isfinite(t_overall.pvalue) else None
					try:
						w_overall = wilcoxon(a_mean[mask], b_mean[mask], zero_method='wilcox')
						w_overall_p = float(w_overall.pvalue)
						overall_w_p = w_overall_p if np.isfinite(w_overall_p) else None
					except Exception:
						w_overall_p = float('nan')
					print(f"  OVERALL mean(SF): n={int(np.sum(mask))} | t-test p={float(t_overall.pvalue):.4g} | wilcoxon p={w_overall_p:.4g}")

				# ART-Rlmer omnibus (Training×SF) if available
				try:
					rows: List[Dict[str, object]] = []
					# best-effort subject IDs: use mouse IDs from pairs when possible
					pair_list = pairs_wt if label == "WT" else pairs_resc
					subject_ids = [p[0].split('\\')[1] if ('\\' in p[0]) else f"S{i+1}" for i, p in enumerate(pair_list)]
					for mi, subj in enumerate(subject_ids[: bmat.shape[0]]):
						for si, sfv in enumerate(mapped_levels):
							bv = float(bmat[mi, si])
							av = float(amat[mi, si])
							if np.isfinite(bv):
								rows.append({"subject": subj, "training": "Before", "sf": f"{sfv:.2f}", "value": bv})
							if np.isfinite(av):
								rows.append({"subject": subj, "training": "After", "sf": f"{sfv:.2f}", "value": av})
					p_art = _art_r_2way_pvalues_from_rows(
						rows,
						subject='subject',
						factor_a='training',
						factor_b='sf',
						value_col='value',
						include_interactions=ART_INCLUDE_INTERACTION_TERMS,
					)
					# _art_r_2way_pvalues_from_rows returns {'a','b','a:b'}
					art_rlmer_train_p = p_art.get('a')
					art_rlmer_sf_p = p_art.get('b')
					art_rlmer_int_p = p_art.get('a:b') if ART_INCLUDE_INTERACTION_TERMS else None
				except Exception:
					pass
		except Exception as exc:
			print(f"  [warn] stats failed for {label}: {type(exc).__name__}: {exc}")

		stat_lines: List[str] = []
		stat_lines.append(f"Paired t (mean SF) p={_fmt_p_short(overall_t_p)}")
		stat_lines.append(f"Wilcoxon (mean SF) p={_fmt_p_short(overall_w_p)}")
		stat_lines.append(f"ART-Rlmer train p={_fmt_p_short(art_rlmer_train_p)}")
		stat_lines.append(f"ART-Rlmer SF p={_fmt_p_short(art_rlmer_sf_p)}")
		if ART_INCLUDE_INTERACTION_TERMS:
			stat_lines.append(f"ART-Rlmer train×SF p={_fmt_p_short(art_rlmer_int_p)}")
		_add_colored_stats_box_outside_axes(ax, stat_lines, x=1.02, y=0.98, fontsize=7.0)

	# Combined panel with all four curves.
	ax = axes[2]
	x = np.array(mapped_levels, dtype=float)
	combined_specs = [
		(wt_b, "WT pre-training", GROUP_COLORS.get("WT_BEFORE", "#9ecae1"), 'o'),
		(wt_a, "WT post-training", GROUP_COLORS.get("WT_AFTER", "#08306b"), 'o'),
		(rb_b, "Rescued pre-training", GROUP_COLORS.get("RESCUED_BEFORE", "#fcae91"), 's'),
		(rb_a, "Rescued post-training", GROUP_COLORS.get("RESCUED_AFTER", "#a50f15"), 's'),
	]
	ax.set_xscale('log')
	for mat, name, color, marker in combined_specs:
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=RuntimeWarning)
			mean_v = np.nanmean(mat, axis=0) if mat.size else np.full_like(x, np.nan)
		sem_v = _nansem(mat, axis=0) if mat.size else np.full_like(x, np.nan)
		ax.errorbar(x, mean_v, yerr=sem_v, fmt=f'-{marker}', capsize=3, color=color, ecolor=color, linewidth=2, markersize=6, label=name, alpha=0.9)
	ax.set_title(f"All groups{' ' + title_suffix if title_suffix else ''}")
	ax.set_xlabel("Spatial frequency")
	ax.set_xticks(x)
	ax.set_xticklabels([f"{v:.2f}" for v in x], rotation=45, ha='right')
	ax.grid(alpha=0.3)
	ax.legend(fontsize=8)

	cross_lines = _build_cross_group_art_rlmer_stats_lines(
		wt_before=wt_b,
		rescued_after=rb_a,
		mapped_levels=mapped_levels,
		label_a="WT pre",
		label_b="Rescued post",
	)
	_add_colored_stats_box_to_axes(ax, cross_lines, x=0.98, y=0.98, fontsize=7.0)

	axes[0].set_ylabel("Orientation-plane area (convex hull)")
	supt = "dsPCA orientation-plane area (polygon spread) vs SF | mean±SEM (paired)"
	if title_suffix:
		supt += f" {title_suffix}"
	fig.suptitle(supt)
	plt.tight_layout(rect=[0, 0, 0.95, 1])
	plt.savefig(os.path.join(GROUPED_PLOTS_DIR, out_file), format='svg', bbox_inches='tight')
	plt.close(fig)
	print(f"  [saved] {out_file}")


def plot_orientation_area_by_sf_paired_cv50(*, out_file: str = "dspca_orientation_area_by_sf_paired_10fcv.svg") -> None:
	"""10-fold CV variant: aggregate held-out dsPCA projections and summarize polygon area."""
	plot_orientation_area_by_sf_paired(
		out_file=out_file,
		dspca_filename="dspca_results_cv50.npz",
		title_suffix="(10-fold CV)",
	)


def plot_adjacent_mean_distance_by_sf_paired(
	*,
	out_file: str = "dspca_adjacent_mean_distance_by_sf_paired.svg",
	dspca_filename: str = "dspca_results.npz",
	title_suffix: str = "",
) -> None:
	"""Plot mean adjacent-orientation distance vs SF, paired Before/After (WT and RESCUED).

	For each mouse/session and SF, the six adjacent edge lengths in the dsPCA orientation
	plane are averaged first; the figure then shows mean±SEM of that scalar across mice.
	"""
	mapped_levels = [float(convert_sf_value(float(sf))) for sf in TARGET_SF_VALUES if float(sf) > 0]
	mapped_levels = sorted(list({sf for sf in mapped_levels if sf > 0}))
	if not mapped_levels:
		print("[skip] No nonzero SF levels available for adjacent-distance summary plot")
		return

	def _align_metric_map(metric_map: Dict[float, float]) -> np.ndarray:
		out = np.full(len(mapped_levels), np.nan, dtype=float)
		for i, sf in enumerate(mapped_levels):
			best = None
			best_err = float('inf')
			for k, v in metric_map.items():
				err = abs(float(k) - float(sf))
				if err < best_err:
					best_err = err
					best = v
			out[i] = float(best) if (best is not None and best_err < 1e-6) else np.nan
		return out

	def _load_mean_adj_vec(session_rel: str) -> np.ndarray:
		path = os.path.join(RESULTS_ROOT, session_rel, dspca_filename)
		if not os.path.exists(path):
			return np.full(len(mapped_levels), np.nan, dtype=float)
		npz = safe_load_npz(path)
		if npz is None:
			return np.full(len(mapped_levels), np.nan, dtype=float)
		try:
			mean_adj_map = _compute_mean_adjacent_distance_by_sf_from_dspca_npz(npz)
		finally:
			try:
				npz.close()
			except Exception:
				pass
		return _align_metric_map(mean_adj_map)

	def _load_std_adj_vec(session_rel: str) -> np.ndarray:
		path = os.path.join(RESULTS_ROOT, session_rel, dspca_filename)
		if not os.path.exists(path):
			return np.full(len(mapped_levels), np.nan, dtype=float)
		npz = safe_load_npz(path)
		if npz is None:
			return np.full(len(mapped_levels), np.nan, dtype=float)
		try:
			std_adj_map = _compute_std_adjacent_distance_by_sf_from_dspca_npz(npz)
		finally:
			try:
				npz.close()
			except Exception:
				pass
		return _align_metric_map(std_adj_map)

	def _load_adj_cv_vec(session_rel: str) -> np.ndarray:
		path = os.path.join(RESULTS_ROOT, session_rel, dspca_filename)
		if not os.path.exists(path):
			return np.full(len(mapped_levels), np.nan, dtype=float)
		npz = safe_load_npz(path)
		if npz is None:
			return np.full(len(mapped_levels), np.nan, dtype=float)
		try:
			adj_cv_map = _compute_adjacent_distance_cv_by_sf_from_dspca_npz(npz)
		finally:
			try:
				npz.close()
			except Exception:
				pass
		return _align_metric_map(adj_cv_map)

	def _load_angle_spacing_rmse_vec(session_rel: str) -> np.ndarray:
		path = os.path.join(RESULTS_ROOT, session_rel, dspca_filename)
		if not os.path.exists(path):
			return np.full(len(mapped_levels), np.nan, dtype=float)
		npz = safe_load_npz(path)
		if npz is None:
			return np.full(len(mapped_levels), np.nan, dtype=float)
		try:
			angle_map = _compute_angle_spacing_rmse_norm_by_sf_from_dspca_npz(npz)
		finally:
			try:
				npz.close()
			except Exception:
				pass
		return _align_metric_map(angle_map)

	def _paired_matrix(pairs: List[Tuple[str, str]], loader) -> Tuple[np.ndarray, np.ndarray]:
		before_list = []
		after_list = []
		for s_before, s_after in pairs:
			before_list.append(loader(s_before))
			after_list.append(loader(s_after))
		return np.vstack(before_list) if before_list else np.empty((0, len(mapped_levels))), np.vstack(after_list) if after_list else np.empty((0, len(mapped_levels)))

	def _nansem(a: np.ndarray, axis: int = 0) -> np.ndarray:
		a = np.asarray(a, dtype=float)
		valid_n = np.sum(np.isfinite(a), axis=axis)
		out = np.full(a.shape[1 - axis], np.nan, dtype=float) if a.ndim == 2 else np.nan
		try:
			if a.ndim != 2 or axis != 0:
				with np.errstate(invalid='ignore', divide='ignore'):
					return np.nanstd(a, axis=axis, ddof=1) / np.sqrt(np.maximum(valid_n, 1))
			for j in range(a.shape[1]):
				n = int(valid_n[j])
				if n >= 2:
					out[j] = float(np.nanstd(a[:, j], ddof=1) / np.sqrt(n))
			return out
		except Exception:
			with np.errstate(invalid='ignore', divide='ignore'):
				return np.nanstd(a, axis=axis, ddof=1) / np.sqrt(np.maximum(valid_n, 1))

	pairs_wt = MOUSE_PAIRS.get("WT", [])
	pairs_resc = MOUSE_PAIRS.get("RESCUED", [])
	wt_b, wt_a = _paired_matrix(pairs_wt, _load_mean_adj_vec)
	rb_b, rb_a = _paired_matrix(pairs_resc, _load_mean_adj_vec)
	wt_b_std, wt_a_std = _paired_matrix(pairs_wt, _load_std_adj_vec)
	rb_b_std, rb_a_std = _paired_matrix(pairs_resc, _load_std_adj_vec)
	wt_b_cv, wt_a_cv = _paired_matrix(pairs_wt, _load_adj_cv_vec)
	rb_b_cv, rb_a_cv = _paired_matrix(pairs_resc, _load_adj_cv_vec)
	wt_b_ang, wt_a_ang = _paired_matrix(pairs_wt, _load_angle_spacing_rmse_vec)
	rb_b_ang, rb_a_ang = _paired_matrix(pairs_resc, _load_angle_spacing_rmse_vec)

	fig, axes = plt.subplots(4, 3, figsize=(20, 20), sharex='col')

	def _plot_row(
		row_axes,
		wt_before_mat: np.ndarray,
		wt_after_mat: np.ndarray,
		resc_before_mat: np.ndarray,
		resc_after_mat: np.ndarray,
		*,
		metric_title: str,
		ylabel: str,
		print_header: str,
	) -> None:
		panels = [
			("WT", wt_before_mat, wt_after_mat, row_axes[0], "WT pre-training", "WT post-training", GROUP_COLORS.get("WT_BEFORE", "#9ecae1"), GROUP_COLORS.get("WT_AFTER", "#08306b")),
			("RESCUED", resc_before_mat, resc_after_mat, row_axes[1], "Rescued pre-training", "Rescued post-training", GROUP_COLORS.get("RESCUED_BEFORE", "#fcae91"), GROUP_COLORS.get("RESCUED_AFTER", "#a50f15")),
		]

		print("\n" + "=" * 70)
		print(print_header)
		print("=" * 70)

		for label, bmat, amat, ax, name_b, name_a, col_b, col_a in panels:
			ax.set_xscale('log')
			x = np.array(mapped_levels, dtype=float)
			with warnings.catch_warnings():
				warnings.filterwarnings("ignore", category=RuntimeWarning)
				mean_b = np.nanmean(bmat, axis=0) if bmat.size else np.full_like(x, np.nan)
				mean_a = np.nanmean(amat, axis=0) if amat.size else np.full_like(x, np.nan)
			sem_b = _nansem(bmat, axis=0) if bmat.size else np.full_like(x, np.nan)
			sem_a = _nansem(amat, axis=0) if amat.size else np.full_like(x, np.nan)

			ax.errorbar(x, mean_b, yerr=sem_b, fmt='-o', capsize=3, color=col_b, ecolor=col_b, linewidth=2, markersize=6, label=name_b, alpha=0.9)
			ax.errorbar(x, mean_a, yerr=sem_a, fmt='-o', capsize=3, color=col_a, ecolor=col_a, linewidth=2, markersize=6, label=name_a, alpha=0.9)
			title = f"{label}: {metric_title} vs SF"
			if title_suffix:
				title += f" {title_suffix}"
			ax.set_title(title)
			ax.set_xlabel("Spatial frequency")
			ax.set_xticks(x)
			ax.set_xticklabels([f"{v:.2f}" for v in x], rotation=45, ha='right')
			ax.grid(alpha=0.3)
			ax.legend(fontsize=9)

			overall_t_p: Optional[float] = None
			overall_w_p: Optional[float] = None
			art_rlmer_train_p: Optional[float] = None
			art_rlmer_sf_p: Optional[float] = None
			art_rlmer_int_p: Optional[float] = None
			try:
				from scipy.stats import ttest_rel, wilcoxon
				print(f"\n[{label}] paired stats per SF (Before vs After):")
				for i, sf in enumerate(mapped_levels):
					b = bmat[:, i] if bmat.size else np.array([], dtype=float)
					a = amat[:, i] if amat.size else np.array([], dtype=float)
					mask = np.isfinite(b) & np.isfinite(a)
					if np.sum(mask) < 2:
						print(f"  SF {sf:.2f}: n={int(np.sum(mask))} (skip)")
						continue
					t_res = ttest_rel(a[mask], b[mask], nan_policy='omit')
					try:
						w_res = wilcoxon(a[mask], b[mask], zero_method='wilcox')
						w_p = float(w_res.pvalue)
					except Exception:
						w_p = float('nan')
					print(f"  SF {sf:.2f}: n={int(np.sum(mask))} | t-test p={float(t_res.pvalue):.4g} | wilcoxon p={w_p:.4g}")
				if bmat.size and amat.size:
					with warnings.catch_warnings():
						warnings.filterwarnings("ignore", category=RuntimeWarning)
						b_mean = np.nanmean(bmat, axis=1)
						a_mean = np.nanmean(amat, axis=1)
					mask = np.isfinite(b_mean) & np.isfinite(a_mean)
					if np.sum(mask) >= 2:
						t_overall = ttest_rel(a_mean[mask], b_mean[mask], nan_policy='omit')
						overall_t_p = float(t_overall.pvalue) if np.isfinite(t_overall.pvalue) else None
						try:
							w_overall = wilcoxon(a_mean[mask], b_mean[mask], zero_method='wilcox')
							w_overall_p = float(w_overall.pvalue)
							overall_w_p = w_overall_p if np.isfinite(w_overall_p) else None
						except Exception:
							w_overall_p = float('nan')
						print(f"  OVERALL mean(SF): n={int(np.sum(mask))} | t-test p={float(t_overall.pvalue):.4g} | wilcoxon p={w_overall_p:.4g}")

					try:
						rows: List[Dict[str, object]] = []
						pair_list = pairs_wt if label == "WT" else pairs_resc
						subject_ids = [p[0].split('\\')[1] if ('\\' in p[0]) else f"S{i+1}" for i, p in enumerate(pair_list)]
						for mi, subj in enumerate(subject_ids[: bmat.shape[0]]):
							for si, sfv in enumerate(mapped_levels):
								bv = float(bmat[mi, si])
								av = float(amat[mi, si])
								if np.isfinite(bv):
									rows.append({"subject": subj, "training": "Before", "sf": f"{sfv:.2f}", "value": bv})
								if np.isfinite(av):
									rows.append({"subject": subj, "training": "After", "sf": f"{sfv:.2f}", "value": av})
						p_art = _art_r_2way_pvalues_from_rows(
							rows,
							subject='subject',
							factor_a='training',
							factor_b='sf',
							value_col='value',
							include_interactions=ART_INCLUDE_INTERACTION_TERMS,
						)
						art_rlmer_train_p = p_art.get('a')
						art_rlmer_sf_p = p_art.get('b')
						art_rlmer_int_p = p_art.get('a:b') if ART_INCLUDE_INTERACTION_TERMS else None
					except Exception:
						pass
			except Exception as exc:
				print(f"  [warn] stats failed for {label}: {type(exc).__name__}: {exc}")

			stat_lines: List[str] = []
			stat_lines.append(f"Paired t (mean SF) p={_fmt_p_short(overall_t_p)}")
			stat_lines.append(f"Wilcoxon (mean SF) p={_fmt_p_short(overall_w_p)}")
			stat_lines.append(f"ART-Rlmer train p={_fmt_p_short(art_rlmer_train_p)}")
			stat_lines.append(f"ART-Rlmer SF p={_fmt_p_short(art_rlmer_sf_p)}")
			if ART_INCLUDE_INTERACTION_TERMS:
				stat_lines.append(f"ART-Rlmer train×SF p={_fmt_p_short(art_rlmer_int_p)}")
			_add_colored_stats_box_outside_axes(ax, stat_lines, x=1.02, y=0.98, fontsize=7.0)

		ax = row_axes[2]
		x = np.array(mapped_levels, dtype=float)
		combined_specs = [
			(wt_before_mat, "WT pre-training", GROUP_COLORS.get("WT_BEFORE", "#9ecae1"), 'o'),
			(wt_after_mat, "WT post-training", GROUP_COLORS.get("WT_AFTER", "#08306b"), 'o'),
			(resc_before_mat, "Rescued pre-training", GROUP_COLORS.get("RESCUED_BEFORE", "#fcae91"), 's'),
			(resc_after_mat, "Rescued post-training", GROUP_COLORS.get("RESCUED_AFTER", "#a50f15"), 's'),
		]
		ax.set_xscale('log')
		for mat, name, color, marker in combined_specs:
			with warnings.catch_warnings():
				warnings.filterwarnings("ignore", category=RuntimeWarning)
				mean_v = np.nanmean(mat, axis=0) if mat.size else np.full_like(x, np.nan)
			sem_v = _nansem(mat, axis=0) if mat.size else np.full_like(x, np.nan)
			ax.errorbar(x, mean_v, yerr=sem_v, fmt=f'-{marker}', capsize=3, color=color, ecolor=color, linewidth=2, markersize=6, label=name, alpha=0.9)
		ax.set_title(f"All groups: {metric_title}{' ' + title_suffix if title_suffix else ''}")
		ax.set_xlabel("Spatial frequency")
		ax.set_xticks(x)
		ax.set_xticklabels([f"{v:.2f}" for v in x], rotation=45, ha='right')
		ax.grid(alpha=0.3)
		ax.legend(fontsize=8)

		cross_lines = _build_cross_group_art_rlmer_stats_lines(
			wt_before=wt_before_mat,
			rescued_after=resc_after_mat,
			mapped_levels=mapped_levels,
			label_a="WT pre",
			label_b="Rescued post",
		)
		_add_colored_stats_box_to_axes(ax, cross_lines, x=0.98, y=0.98, fontsize=7.0)

		row_axes[0].set_ylabel(ylabel)

	_plot_row(
		axes[0],
		wt_b,
		wt_a,
		rb_b,
		rb_a,
		metric_title="Mean adjacent distance",
		ylabel="Mean adjacent orientation distance",
		print_header="dsPCA MEAN ADJACENT-ORIENTATION DISTANCE | Paired Before/After",
	)
	_plot_row(
		axes[1],
		wt_b_std,
		wt_a_std,
		rb_b_std,
		rb_a_std,
		metric_title="SD of adjacent distance",
		ylabel="SD across adjacent orientation pairs",
		print_header="dsPCA SD OF ADJACENT-ORIENTATION DISTANCE | Paired Before/After",
	)
	_plot_row(
		axes[2],
		wt_b_cv,
		wt_a_cv,
		rb_b_cv,
		rb_a_cv,
		metric_title="Adjacent-edge CV",
		ylabel="Adjacent-edge CV (std/mean)",
		print_header="dsPCA ADJACENT-EDGE CV | Paired Before/After",
	)
	_plot_row(
		axes[3],
		wt_b_ang,
		wt_a_ang,
		rb_b_ang,
		rb_a_ang,
		metric_title="Angle-spacing RMSE norm",
		ylabel="Angle-spacing RMSE / ideal",
		print_header="dsPCA ANGLE-SPACING RMSE NORM | Paired Before/After",
	)
	supt = "dsPCA orientation-plane adjacent-distance summaries vs SF | mean±SEM (paired)"
	if title_suffix:
		supt += f" {title_suffix}"
	fig.suptitle(supt)
	plt.tight_layout(rect=[0, 0, 0.95, 1])
	plt.savefig(os.path.join(GROUPED_PLOTS_DIR, out_file), format='svg', bbox_inches='tight')
	plt.close(fig)
	print(f"  [saved] {out_file}")


def plot_adjacent_mean_distance_by_sf_paired_cv50(
	*,
	out_file: str = "dspca_adjacent_mean_distance_by_sf_paired_10fcv.svg",
) -> None:
	"""10-fold CV variant: aggregate held-out dsPCA projections and summarize adjacent distances."""
	plot_adjacent_mean_distance_by_sf_paired(
		out_file=out_file,
		dspca_filename="dspca_results_cv50.npz",
		title_suffix="(10-fold CV)",
	)


def aggregate_distance_metrics_by_group(metric_type: str) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
	"""Aggregate distance metrics (dist_to_com or adj_dist) for each group by training condition.
	
	Returns: {group: {condition: (mean, sem)}} where:
	- group is "WT" or "RESCUED"
	- condition is "BEFORE" or "AFTER"
	- mean/sem are arrays aligned to SFs or orientation indices
	"""
	results = {}
	
	for group_name in ["WT_BEFORE", "WT_AFTER", "RESCUED_BEFORE", "RESCUED_AFTER"]:
		sessions = GROUPS.get(group_name, [])
		if not sessions:
			continue
		
		group = group_name.split("_")[0]  # "WT" or "RESCUED"
		condition = group_name.split("_")[1]  # "BEFORE" or "AFTER"
		
		if group not in results:
			results[group] = {}
		
		# Load metrics from all sessions
		all_metrics = []
		for session in sessions:
			metrics = load_session_distance_metrics(session)
			if metrics:
				all_metrics.append(metrics)
		
		if not all_metrics:
			continue
		
		# Aggregate based on metric type
		if metric_type == "dist_to_com":
			# Stack distance-to-COM across sessions for consistent SFs
			# Get reference SFs
			ref_sf_logs = sorted(all_metrics[0].get('unique_sf_log', []))
			
			per_sf_distances = {sf_log: [] for sf_log in ref_sf_logs}
			for metrics in all_metrics:
				dist_dict = metrics.get('dist_to_com_by_sf', {})
				for sf_log in ref_sf_logs:
					if sf_log in dist_dict:
						per_sf_distances[sf_log].append(dist_dict[sf_log])
			
			# Compute mean±SEM per SF
			mean_by_sf = []
			sem_by_sf = []
			valid_sfs = []
			for sf_log in ref_sf_logs:
				distances = per_sf_distances[sf_log]
				if distances:
					# Stack and align
					min_len = min(len(d) for d in distances)
					aligned = np.array([d[:min_len] for d in distances])
					mean_by_sf.append(np.nanmean(aligned, axis=0))
					sem_by_sf.append(np.nanstd(aligned, axis=0, ddof=1) / np.sqrt(aligned.shape[0]))
					valid_sfs.append(sf_log)
			
			if mean_by_sf:
				results[group][condition] = (np.array(mean_by_sf), np.array(sem_by_sf), np.array(valid_sfs))
		
		elif metric_type == "adj_dist":
			# Stack adjacent distances across sessions
			ref_sf_logs = sorted(all_metrics[0].get('unique_sf_log', []))
			
			per_sf_adj_dists = {sf_log: [] for sf_log in ref_sf_logs}
			for metrics in all_metrics:
				adj_dict = metrics.get('adj_dist_by_sf', {})
				for sf_log in ref_sf_logs:
					if sf_log in adj_dict:
						per_sf_adj_dists[sf_log].append(adj_dict[sf_log])
			
			# Compute mean±SEM per SF
			mean_by_sf = []
			sem_by_sf = []
			valid_sfs = []
			for sf_log in ref_sf_logs:
				adj_distances = per_sf_adj_dists[sf_log]
				if adj_distances:
					# Stack and align
					min_len = min(len(d) for d in adj_distances)
					aligned = np.array([d[:min_len] for d in adj_distances])
					mean_by_sf.append(np.nanmean(aligned, axis=0))
					sem_by_sf.append(np.nanstd(aligned, axis=0, ddof=1) / np.sqrt(aligned.shape[0]))
					valid_sfs.append(sf_log)
			
			if mean_by_sf:
				results[group][condition] = (np.array(mean_by_sf), np.array(sem_by_sf), np.array(valid_sfs))

		elif metric_type == "adj_cv":
			# Stack adjacent-distance CV (edge-length coefficient of variation) across sessions
			ref_sf_logs = sorted(all_metrics[0].get('unique_sf_log', []))

			per_sf_cvs = {sf_log: [] for sf_log in ref_sf_logs}
			for metrics in all_metrics:
				cv_dict = metrics.get('adj_dist_cv_by_sf', {})
				for sf_log in ref_sf_logs:
					if sf_log in cv_dict and np.isfinite(cv_dict[sf_log]):
						per_sf_cvs[sf_log].append(float(cv_dict[sf_log]))

			mean_by_sf = []
			sem_by_sf = []
			valid_sfs = []
			for sf_log in ref_sf_logs:
				vals = np.array(per_sf_cvs[sf_log], dtype=float)
				vals = vals[np.isfinite(vals)]
				if vals.size >= 1:
					mean_by_sf.append(float(np.nanmean(vals)))
					sem_by_sf.append(float(np.nanstd(vals, ddof=1) / np.sqrt(vals.size)) if vals.size >= 2 else np.nan)
					valid_sfs.append(sf_log)

			if mean_by_sf:
				results[group][condition] = (np.array(mean_by_sf), np.array(sem_by_sf), np.array(valid_sfs))

		elif metric_type in {"radial_cv", "circle_rmse_norm", "angle_spacing_rmse_norm"}:
			# Stack scalar-per-SF circle geometry metrics across sessions
			ref_sf_logs = sorted(all_metrics[0].get('unique_sf_log', []))
			key_map = {
				"radial_cv": "radial_cv_by_sf",
				"circle_rmse_norm": "circle_rmse_norm_by_sf",
				"angle_spacing_rmse_norm": "angle_spacing_rmse_norm_by_sf",
			}
			metric_dict_key = key_map[metric_type]

			per_sf_vals = {sf_log: [] for sf_log in ref_sf_logs}
			for metrics in all_metrics:
				val_dict = metrics.get(metric_dict_key, {})
				for sf_log in ref_sf_logs:
					if sf_log in val_dict and np.isfinite(val_dict[sf_log]):
						per_sf_vals[sf_log].append(float(val_dict[sf_log]))

			mean_by_sf = []
			sem_by_sf = []
			valid_sfs = []
			for sf_log in ref_sf_logs:
				vals = np.array(per_sf_vals[sf_log], dtype=float)
				vals = vals[np.isfinite(vals)]
				if vals.size >= 1:
					mean_by_sf.append(float(np.nanmean(vals)))
					sem_by_sf.append(float(np.nanstd(vals, ddof=1) / np.sqrt(vals.size)) if vals.size >= 2 else np.nan)
					valid_sfs.append(sf_log)

			if mean_by_sf:
				results[group][condition] = (np.array(mean_by_sf), np.array(sem_by_sf), np.array(valid_sfs))
	
	return results


def compute_rmanova_distance_to_com(group: str) -> Optional[Dict[str, Optional[float]]]:
	"""Compute within-mouse RM-ANOVA for distance-to-COM with SF as repeated measure.

	Uses per-mouse per-SF mean across orientation bins from dsPCA distance curves.
	Model: distance_mean ~ training * sf (both within-subject).
	"""
	return compute_rmanova_training_sf_from_vector_distance_metric(group, metric_dict_key="dist_to_com_by_sf")


def compute_rmanova_distance_to_com_per_sf(group: str) -> Dict[float, Optional[float]]:
	"""Per-SF RM-ANOVA p-values for training effect on distance-to-COM curves.

	Model: value ~ training * ori_bin, within-subject (mouse).
	Returns {sf_log: p_training}.
	"""
	# Legacy API: return the (SF-repeated) training p-value for each SF.
	res = compute_rmanova_training_sf_from_vector_distance_metric(group, metric_dict_key="dist_to_com_by_sf")
	p = res.get('training_p', None) if res else None
	# Use SF keys observed in the paired data
	keys: List[float] = sorted(list(_collect_vector_metric_records(group, 'dist_to_com_by_sf', 'ori_bin').keys()))
	return {k: p for k in keys}


def compute_art_distance_to_com(group: str) -> Optional[Dict[str, Optional[float]]]:
	"""Compute ART for distance-to-COM with SF as repeated measure.

	Uses per-mouse per-SF mean across orientation bins.
	"""
	return compute_art_training_sf_from_vector_distance_metric(group, metric_dict_key="dist_to_com_by_sf")


def compute_art_distance_to_com_per_sf(group: str) -> Dict[float, Optional[float]]:
	"""Per-SF ART p-values for training effect on distance-to-COM curves.

	Uses ART ranks for Training in a two-way within-subject design (training × ori_bin).
	Returns {sf_log: p_training}.
	"""
	# Legacy API: return the (SF-repeated) training p-value for each SF.
	res = compute_art_training_sf_from_vector_distance_metric(group, metric_dict_key="dist_to_com_by_sf")
	p = res.get('training_p', None) if res else None
	keys: List[float] = sorted(list(_collect_vector_metric_records(group, 'dist_to_com_by_sf', 'ori_bin').keys()))
	return {k: p for k in keys}


def compute_rmanova_adjacent_dist(group: str) -> Optional[Dict[str, Optional[float]]]:
	"""Compute within-mouse RM-ANOVA for adjacent distances with SF as repeated measure.

	Uses per-mouse per-SF mean across adjacent-pair bins.
	Model: distance_mean ~ training * sf (both within-subject).
	"""
	return compute_rmanova_training_sf_from_vector_distance_metric(group, metric_dict_key="adj_dist_by_sf")


def compute_rmanova_adjacent_dist_per_sf(group: str) -> Dict[float, Optional[float]]:
	"""Per-SF RM-ANOVA p-values for training effect on adjacent-distance curves.

	Model: value ~ training * adj_pair, within-subject (mouse).
	Returns {sf_log: p_training}.
	"""
	# Legacy API: return the (SF-repeated) training p-value for each SF.
	res = compute_rmanova_training_sf_from_vector_distance_metric(group, metric_dict_key="adj_dist_by_sf")
	p = res.get('training_p', None) if res else None
	keys: List[float] = sorted(list(_collect_vector_metric_records(group, 'adj_dist_by_sf', 'adj_pair').keys()))
	return {k: p for k in keys}


def compute_art_adjacent_dist(group: str) -> Optional[Dict[str, Optional[float]]]:
	"""Compute ART for adjacent orientation distances across SF and Training."""
	agg = aggregate_distance_metrics_by_group("adj_dist")
	if group not in agg:
		return {"training_p": None, "sf_p": None, "interaction_p": None}
	
	before_data = agg[group].get("BEFORE")
	after_data = agg[group].get("AFTER")
	
	if before_data is None or after_data is None:
		return {"training_p": None, "sf_p": None, "interaction_p": None}
	return compute_art_training_sf_from_vector_distance_metric(group, metric_dict_key="adj_dist_by_sf")


def compute_paired_adjacent_cv_per_sf(group: str) -> Dict[float, Optional[float]]:
	"""Compute per-SF paired Wilcoxon p-values for adjacent-edge CV (equidistance).

	Uses per-session values and mouse pairing (pre vs post) from MOUSE_PAIRS.
	Returns dict keyed by sf_log (float).
	"""
	results: Dict[float, Optional[float]] = {}
	pairs = MOUSE_PAIRS.get(group, [])
	if not pairs:
		return results

	# Collect paired values per SF.
	per_sf_pairs: Dict[float, List[Tuple[float, float]]] = {}
	for before_session, after_session in pairs:
		m_before = load_session_distance_metrics(before_session)
		m_after = load_session_distance_metrics(after_session)
		if not m_before or not m_after:
			continue
		cv_b = m_before.get('adj_dist_cv_by_sf', {})
		cv_a = m_after.get('adj_dist_cv_by_sf', {})
		if not cv_b or not cv_a:
			continue

		# Match SFs by rounding to avoid tiny float differences.
		keys_b = {round(float(k), 6): float(v) for k, v in cv_b.items()}
		keys_a = {round(float(k), 6): float(v) for k, v in cv_a.items()}
		common = set(keys_b.keys()) & set(keys_a.keys())
		for sf in common:
			vb = keys_b[sf]
			va = keys_a[sf]
			if not (np.isfinite(vb) and np.isfinite(va)):
				continue
			per_sf_pairs.setdefault(float(sf), []).append((vb, va))

	for sf, pairs_list in per_sf_pairs.items():
		if len(pairs_list) < 2:
			results[sf] = None
			continue
		before_vals = np.array([p[0] for p in pairs_list], dtype=float)
		after_vals = np.array([p[1] for p in pairs_list], dtype=float)
		try:
			pval = _safe_wilcoxon_p(before_vals, after_vals)
			results[sf] = float(pval) if (pval is not None and np.isfinite(pval)) else None
		except:
			results[sf] = None

	return results


def compute_paired_scalar_metric_per_sf(group: str, metric_dict_key: str) -> Dict[float, Optional[float]]:
	"""Compute per-SF paired Wilcoxon p-values for a scalar-per-SF metric.

	metric_dict_key must be one of the dicts returned by load_session_distance_metrics(), e.g.
	- 'radial_cv_by_sf'
	- 'circle_rmse_norm_by_sf'
	- 'angle_spacing_rmse_norm_by_sf'
	- 'adj_dist_cv_by_sf'
	"""
	results: Dict[float, Optional[float]] = {}
	pairs = MOUSE_PAIRS.get(group, [])
	if not pairs:
		return results

	per_sf_pairs: Dict[float, List[Tuple[float, float]]] = {}
	for before_session, after_session in pairs:
		m_before = load_session_distance_metrics(before_session)
		m_after = load_session_distance_metrics(after_session)
		if not m_before or not m_after:
			continue
		vals_b = m_before.get(metric_dict_key, {})
		vals_a = m_after.get(metric_dict_key, {})
		if not vals_b or not vals_a:
			continue

		keys_b = {round(float(k), 6): float(v) for k, v in vals_b.items()}
		keys_a = {round(float(k), 6): float(v) for k, v in vals_a.items()}
		common = set(keys_b.keys()) & set(keys_a.keys())
		for sf in common:
			vb = keys_b[sf]
			va = keys_a[sf]
			if not (np.isfinite(vb) and np.isfinite(va)):
				continue
			per_sf_pairs.setdefault(float(sf), []).append((vb, va))

	for sf, pairs_list in per_sf_pairs.items():
		if len(pairs_list) < 2:
			results[sf] = None
			continue
		before_vals = np.array([p[0] for p in pairs_list], dtype=float)
		after_vals = np.array([p[1] for p in pairs_list], dtype=float)
		try:
			pval = _safe_wilcoxon_p(before_vals, after_vals)
			results[sf] = float(pval) if (pval is not None and np.isfinite(pval)) else None
		except:
			results[sf] = None

	return results


def _mouse_id_from_session_rel(session_rel: str) -> str:
	parts = session_rel.split('\\')
	for part in parts:
		if part.startswith('VD'):
			return part
	return session_rel


def _collect_scalar_metric_records(group: str, metric_dict_key: str) -> Dict[float, List[Dict[str, object]]]:
	"""Collect per-SF long-form records for a scalar metric using mouse pairs.

	Returns {sf_log: [ {subject, training, value}, ... ]}.
	"""
	per_sf_records: Dict[float, List[Dict[str, object]]] = {}
	pairs = MOUSE_PAIRS.get(group, [])
	for before_session, after_session in pairs:
		m_before = load_session_distance_metrics(before_session)
		m_after = load_session_distance_metrics(after_session)
		if not m_before or not m_after:
			continue
		vals_b = m_before.get(metric_dict_key, {})
		vals_a = m_after.get(metric_dict_key, {})
		if not vals_b or not vals_a:
			continue

		subject = _mouse_id_from_session_rel(before_session)
		keys_b = {round(float(k), 6): float(v) for k, v in vals_b.items()}
		keys_a = {round(float(k), 6): float(v) for k, v in vals_a.items()}
		common = set(keys_b.keys()) & set(keys_a.keys())
		for sf in common:
			vb = keys_b[sf]
			va = keys_a[sf]
			if not (np.isfinite(vb) and np.isfinite(va)):
				continue
			sf_key = float(sf)
			per_sf_records.setdefault(sf_key, []).append({"subject": subject, "training": "before", "value": vb})
			per_sf_records.setdefault(sf_key, []).append({"subject": subject, "training": "after", "value": va})

	return per_sf_records


def _build_training_sf_long_df_from_scalar_dict(group: str, metric_dict_key: str) -> pd.DataFrame:
	"""Build balanced long-form dataframe for Training×SF within-subject analysis.

	Source: per-session dict metrics from load_session_distance_metrics().
	metric_dict_key: e.g. 'adj_dist_cv_by_sf', 'radial_cv_by_sf', ...
	
	Returns columns: subject, training, sf, value.
	Filters to SFs present for all subjects (both training levels) and subjects with full grid.
	"""
	per_sf_records = _collect_scalar_metric_records(group, metric_dict_key)
	rows: List[Dict[str, object]] = []
	for sf, recs in per_sf_records.items():
		for r in recs:
			rows.append({"subject": r['subject'], "training": r['training'], "sf": str(float(sf)), "value": float(r['value'])})
	if not rows:
		return pd.DataFrame(columns=['subject', 'training', 'sf', 'value'])
	df = pd.DataFrame(rows)
	# Balance: keep only SFs complete across subjects.
	grid = df.pivot_table(index='subject', columns=['training', 'sf'], values='value', aggfunc='mean')
	if grid.empty:
		return pd.DataFrame(columns=['subject', 'training', 'sf', 'value'])
	all_sfs = sorted({c[1] for c in grid.columns})
	good_sfs: List[str] = []
	for sf in all_sfs:
		if (('before', sf) in grid.columns) and (('after', sf) in grid.columns):
			col = grid[[('before', sf), ('after', sf)]]
			if not col.isna().any().any():
				good_sfs.append(sf)
	if len(good_sfs) < 2:
		return pd.DataFrame(columns=['subject', 'training', 'sf', 'value'])
	df = df[df['sf'].isin(good_sfs)].copy()
	grid2 = df.pivot_table(index='subject', columns=['training', 'sf'], values='value', aggfunc='mean')
	grid2 = grid2.dropna(axis=0, how='any')
	if grid2.shape[0] < 2:
		return pd.DataFrame(columns=['subject', 'training', 'sf', 'value'])
	balanced_rows: List[Dict[str, object]] = []
	for subject in grid2.index:
		for training in ['before', 'after']:
			for sf in good_sfs:
				balanced_rows.append({
					"subject": subject,
					"training": training,
					"sf": sf,
					"value": float(grid2.loc[subject, (training, sf)]),
				})
	return pd.DataFrame(balanced_rows)


def compute_rmanova_training_sf_from_scalar_metric(group: str, metric_dict_key: str) -> Dict[str, Optional[float]]:
	"""RM-ANOVA (AnovaRM) for Training×SF using scalar-per-SF dict metrics."""
	df = _build_training_sf_long_df_from_scalar_dict(group, metric_dict_key)
	if df.empty:
		return {"training_p": None, "sf_p": None, "interaction_p": None}
	try:
		if RM_ANOVA_INCLUDE_INTERACTION_TERMS:
			res = AnovaRM(df, depvar='value', subject='subject', within=['training', 'sf']).fit()
			tab = res.anova_table
			p_train = tab.loc['training', 'Pr > F'] if 'training' in tab.index else np.nan
			p_sf = tab.loc['sf', 'Pr > F'] if 'sf' in tab.index else np.nan
			p_int = tab.loc['training:sf', 'Pr > F'] if 'training:sf' in tab.index else np.nan
		else:
			df_train = df.groupby(['subject', 'training'], as_index=False)['value'].mean()
			res_train = AnovaRM(df_train, depvar='value', subject='subject', within=['training']).fit()
			tab_train = res_train.anova_table
			p_train = tab_train.loc['training', 'Pr > F'] if 'training' in tab_train.index else np.nan

			df_sf = df.groupby(['subject', 'sf'], as_index=False)['value'].mean()
			res_sf = AnovaRM(df_sf, depvar='value', subject='subject', within=['sf']).fit()
			tab_sf = res_sf.anova_table
			p_sf = tab_sf.loc['sf', 'Pr > F'] if 'sf' in tab_sf.index else np.nan
			p_int = np.nan
		return {
			"training_p": float(p_train) if np.isfinite(p_train) else None,
			"sf_p": float(p_sf) if np.isfinite(p_sf) else None,
			"interaction_p": float(p_int) if np.isfinite(p_int) else None,
		}
	except Exception:
		return {"training_p": None, "sf_p": None, "interaction_p": None}


def compute_art_training_sf_from_scalar_metric(group: str, metric_dict_key: str) -> Dict[str, Optional[float]]:
	"""ART (aligned-rank) for Training×SF using scalar-per-SF dict metrics."""
	df = _build_training_sf_long_df_from_scalar_dict(group, metric_dict_key)
	if df.empty:
		return {"training_p": None, "sf_p": None, "interaction_p": None}
	try:
		# Remove subject main effect, then ART-align/rank
		df['value_centered'] = df['value'] - df.groupby('subject')['value'].transform('mean')
		df['training'] = df['training'].astype('category')
		df['sf'] = df['sf'].astype('category')
		art = _aligned_rank_transform(
			df,
			factor_a='training',
			factor_b='sf',
			subject='subject',
			value_col='value_centered',
			include_interaction=ART_INCLUDE_INTERACTION_TERMS,
		)
		df['rank_A'] = art['A']
		df['rank_B'] = art['B']
		p_int = np.nan
		if ART_INCLUDE_INTERACTION_TERMS:
			p_train = AnovaRM(df, depvar='rank_A', subject='subject', within=['training', 'sf']).fit().anova_table.loc['training', 'Pr > F']
			p_sf = AnovaRM(df, depvar='rank_B', subject='subject', within=['training', 'sf']).fit().anova_table.loc['sf', 'Pr > F']
			if 'AxB' in art:
				df['rank_Inter'] = art['AxB']
				p_int = AnovaRM(df, depvar='rank_Inter', subject='subject', within=['training', 'sf']).fit().anova_table.loc['training:sf', 'Pr > F']
		else:
			df_a = df.groupby(['subject', 'training'], as_index=False)['rank_A'].mean()
			p_train = AnovaRM(df_a, depvar='rank_A', subject='subject', within=['training']).fit().anova_table.loc['training', 'Pr > F']
			df_b = df.groupby(['subject', 'sf'], as_index=False)['rank_B'].mean()
			p_sf = AnovaRM(df_b, depvar='rank_B', subject='subject', within=['sf']).fit().anova_table.loc['sf', 'Pr > F']
		return {
			"training_p": float(p_train) if np.isfinite(p_train) else None,
			"sf_p": float(p_sf) if np.isfinite(p_sf) else None,
			"interaction_p": float(p_int) if np.isfinite(p_int) else None,
		}
	except Exception:
		return {"training_p": None, "sf_p": None, "interaction_p": None}


def compute_art_r_training_sf_from_scalar_metric(group: str, metric_dict_key: str) -> Dict[str, Optional[float]]:
	"""ART-Rlmer (R ARTool) omnibus p-values for Training×SF using scalar-per-SF dict metrics."""
	df = _build_training_sf_long_df_from_scalar_dict(group, metric_dict_key)
	if df.empty:
		return {"training_p": None, "sf_p": None, "interaction_p": None}
	try:
		p = _art_r_2way_pvalues_from_rows(
			df.to_dict('records'),
			subject='subject',
			factor_a='training',
			factor_b='sf',
			value_col='value',
			include_interactions=ART_INCLUDE_INTERACTION_TERMS,
		)
		return {
			# _art_r_2way_pvalues_from_rows returns {'a','b','a:b'} mapped to (training, sf, training:sf)
			"training_p": p.get('a'),
			"sf_p": p.get('b'),
			"interaction_p": p.get('a:b') if ART_INCLUDE_INTERACTION_TERMS else None,
		}
	except Exception:
		return {"training_p": None, "sf_p": None, "interaction_p": None}


def compute_art_raov_training_sf_from_scalar_metric(group: str, metric_dict_key: str) -> Dict[str, Optional[float]]:
	"""ART-Raov (R ARTool) omnibus p-values for Training×SF using scalar-per-SF dict metrics."""
	df = _build_training_sf_long_df_from_scalar_dict(group, metric_dict_key)
	if df.empty:
		return {"training_p": None, "sf_p": None, "interaction_p": None}
	try:
		p = _art_raov_2way_pvalues_from_rows(
			df.to_dict('records'),
			subject='subject',
			factor_a='training',
			factor_b='sf',
			value_col='value',
			include_interactions=ART_INCLUDE_INTERACTION_TERMS,
		)
		return {
			"training_p": p.get('a'),
			"sf_p": p.get('b'),
			"interaction_p": p.get('a:b'),
		}
	except Exception:
		return {"training_p": None, "sf_p": None, "interaction_p": None}


def _build_training_sf_long_df_from_vector_distance_metric(group: str, metric_dict_key: str) -> pd.DataFrame:
	"""Build balanced long-form df for Training×SF from vector distance metrics.

	metric_dict_key: 'dist_to_com_by_sf' or 'adj_dist_by_sf'
	Uses per-SF mean across bins as the scalar response.
	"""
	pairs = MOUSE_PAIRS.get(group, [])
	rows: List[Dict[str, object]] = []
	for before_session, after_session in pairs:
		m_before = load_session_distance_metrics(before_session)
		m_after = load_session_distance_metrics(after_session)
		if not m_before or not m_after:
			continue
		vals_b = m_before.get(metric_dict_key, {})
		vals_a = m_after.get(metric_dict_key, {})
		if not vals_b or not vals_a:
			continue
		subject = _mouse_id_from_session_rel(before_session)
		keys_b = {round(float(k), 6): np.asarray(v, dtype=float) for k, v in vals_b.items()}
		keys_a = {round(float(k), 6): np.asarray(v, dtype=float) for k, v in vals_a.items()}
		common = set(keys_b.keys()) & set(keys_a.keys())
		for sf in common:
			vb = keys_b[sf]
			va = keys_a[sf]
			if vb.ndim != 1 or va.ndim != 1:
				continue
			mb = float(np.nanmean(vb))
			ma = float(np.nanmean(va))
			if np.isfinite(mb) and np.isfinite(ma):
				rows.append({"subject": subject, "training": "before", "sf": str(float(sf)), "value": mb})
				rows.append({"subject": subject, "training": "after", "sf": str(float(sf)), "value": ma})
	if not rows:
		return pd.DataFrame(columns=['subject', 'training', 'sf', 'value'])
	df = pd.DataFrame(rows)
	grid = df.pivot_table(index='subject', columns=['training', 'sf'], values='value', aggfunc='mean')
	if grid.empty:
		return pd.DataFrame(columns=['subject', 'training', 'sf', 'value'])
	all_sfs = sorted({c[1] for c in grid.columns})
	good_sfs: List[str] = []
	for sf in all_sfs:
		if (('before', sf) in grid.columns) and (('after', sf) in grid.columns):
			col = grid[[('before', sf), ('after', sf)]]
			if not col.isna().any().any():
				good_sfs.append(sf)
	if len(good_sfs) < 2:
		return pd.DataFrame(columns=['subject', 'training', 'sf', 'value'])
	df = df[df['sf'].isin(good_sfs)].copy()
	grid2 = df.pivot_table(index='subject', columns=['training', 'sf'], values='value', aggfunc='mean')
	grid2 = grid2.dropna(axis=0, how='any')
	if grid2.shape[0] < 2:
		return pd.DataFrame(columns=['subject', 'training', 'sf', 'value'])
	balanced_rows: List[Dict[str, object]] = []
	for subject in grid2.index:
		for training in ['before', 'after']:
			for sf in good_sfs:
				balanced_rows.append({
					"subject": subject,
					"training": training,
					"sf": sf,
					"value": float(grid2.loc[subject, (training, sf)]),
				})
	return pd.DataFrame(balanced_rows)


def compute_rmanova_training_sf_from_vector_distance_metric(group: str, metric_dict_key: str) -> Dict[str, Optional[float]]:
	"""RM-ANOVA (AnovaRM) for Training×SF from vector distance metrics reduced to per-SF mean."""
	df = _build_training_sf_long_df_from_vector_distance_metric(group, metric_dict_key)
	if df.empty:
		return {"training_p": None, "sf_p": None, "interaction_p": None}
	try:
		if RM_ANOVA_INCLUDE_INTERACTION_TERMS:
			res = AnovaRM(df, depvar='value', subject='subject', within=['training', 'sf']).fit()
			tab = res.anova_table
			p_train = tab.loc['training', 'Pr > F'] if 'training' in tab.index else np.nan
			p_sf = tab.loc['sf', 'Pr > F'] if 'sf' in tab.index else np.nan
			p_int = tab.loc['training:sf', 'Pr > F'] if 'training:sf' in tab.index else np.nan
		else:
			df_train = df.groupby(['subject', 'training'], as_index=False)['value'].mean()
			res_train = AnovaRM(df_train, depvar='value', subject='subject', within=['training']).fit()
			tab_train = res_train.anova_table
			p_train = tab_train.loc['training', 'Pr > F'] if 'training' in tab_train.index else np.nan

			df_sf = df.groupby(['subject', 'sf'], as_index=False)['value'].mean()
			res_sf = AnovaRM(df_sf, depvar='value', subject='subject', within=['sf']).fit()
			tab_sf = res_sf.anova_table
			p_sf = tab_sf.loc['sf', 'Pr > F'] if 'sf' in tab_sf.index else np.nan
			p_int = np.nan
		return {
			"training_p": float(p_train) if np.isfinite(p_train) else None,
			"sf_p": float(p_sf) if np.isfinite(p_sf) else None,
			"interaction_p": float(p_int) if np.isfinite(p_int) else None,
		}
	except Exception:
		return {"training_p": None, "sf_p": None, "interaction_p": None}


def compute_art_training_sf_from_vector_distance_metric(group: str, metric_dict_key: str) -> Dict[str, Optional[float]]:
	"""ART (aligned-rank) for Training×SF from vector distance metrics reduced to per-SF mean."""
	df = _build_training_sf_long_df_from_vector_distance_metric(group, metric_dict_key)
	if df.empty:
		return {"training_p": None, "sf_p": None, "interaction_p": None}
	try:
		df['value_centered'] = df['value'] - df.groupby('subject')['value'].transform('mean')
		df['training'] = df['training'].astype('category')
		df['sf'] = df['sf'].astype('category')
		art = _aligned_rank_transform(
			df,
			factor_a='training',
			factor_b='sf',
			subject='subject',
			value_col='value_centered',
			include_interaction=ART_INCLUDE_INTERACTION_TERMS,
		)
		df['rank_A'] = art['A']
		df['rank_B'] = art['B']
		p_int = np.nan
		if ART_INCLUDE_INTERACTION_TERMS:
			p_train = AnovaRM(df, depvar='rank_A', subject='subject', within=['training', 'sf']).fit().anova_table.loc['training', 'Pr > F']
			p_sf = AnovaRM(df, depvar='rank_B', subject='subject', within=['training', 'sf']).fit().anova_table.loc['sf', 'Pr > F']
			if 'AxB' in art:
				df['rank_Inter'] = art['AxB']
				p_int = AnovaRM(df, depvar='rank_Inter', subject='subject', within=['training', 'sf']).fit().anova_table.loc['training:sf', 'Pr > F']
		else:
			df_a = df.groupby(['subject', 'training'], as_index=False)['rank_A'].mean()
			p_train = AnovaRM(df_a, depvar='rank_A', subject='subject', within=['training']).fit().anova_table.loc['training', 'Pr > F']
			df_b = df.groupby(['subject', 'sf'], as_index=False)['rank_B'].mean()
			p_sf = AnovaRM(df_b, depvar='rank_B', subject='subject', within=['sf']).fit().anova_table.loc['sf', 'Pr > F']
		return {
			"training_p": float(p_train) if np.isfinite(p_train) else None,
			"sf_p": float(p_sf) if np.isfinite(p_sf) else None,
			"interaction_p": float(p_int) if np.isfinite(p_int) else None,
		}
	except Exception:
		return {"training_p": None, "sf_p": None, "interaction_p": None}


def compute_art_r_training_sf_from_vector_distance_metric(group: str, metric_dict_key: str) -> Dict[str, Optional[float]]:
	"""ART-Rlmer (R ARTool) omnibus p-values for Training×SF from vector distance metrics reduced to per-SF mean."""
	df = _build_training_sf_long_df_from_vector_distance_metric(group, metric_dict_key)
	if df.empty:
		return {"training_p": None, "sf_p": None, "interaction_p": None}
	try:
		p = _art_r_2way_pvalues_from_rows(
			df.to_dict('records'),
			subject='subject',
			factor_a='training',
			factor_b='sf',
			value_col='value',
			include_interactions=ART_INCLUDE_INTERACTION_TERMS,
		)
		return {
			"training_p": p.get('training'),
			"sf_p": p.get('sf'),
			"interaction_p": p.get('training:sf'),
		}
	except Exception:
		return {"training_p": None, "sf_p": None, "interaction_p": None}


def compute_art_raov_training_sf_from_vector_distance_metric(group: str, metric_dict_key: str) -> Dict[str, Optional[float]]:
	"""ART-Raov (R ARTool) omnibus p-values for Training×SF from vector distance metrics reduced to per-SF mean."""
	df = _build_training_sf_long_df_from_vector_distance_metric(group, metric_dict_key)
	if df.empty:
		return {"training_p": None, "sf_p": None, "interaction_p": None}
	try:
		p = _art_raov_2way_pvalues_from_rows(
			df.to_dict('records'),
			subject='subject',
			factor_a='training',
			factor_b='sf',
			value_col='value',
			include_interactions=ART_INCLUDE_INTERACTION_TERMS,
		)
		return {
			"training_p": p.get('a'),
			"sf_p": p.get('b'),
			"interaction_p": p.get('a:b'),
		}
	except Exception:
		return {"training_p": None, "sf_p": None, "interaction_p": None}


def compute_posthoc_training_p_per_sf_scalar_dict_metric_rm(group: str, metric_dict_key: str) -> Dict[float, Optional[float]]:
	"""Post-hoc simple effects: Training (before vs after) within each SF.

	Applies to scalar-per-SF dict metrics from load_session_distance_metrics().
	Returns {sf_log: p_training}.
	"""
	results: Dict[float, Optional[float]] = {}
	per_sf_records = _collect_scalar_metric_records(group, metric_dict_key)
	for sf, records in per_sf_records.items():
		df = pd.DataFrame(records)
		if df.empty:
			results[sf] = None
			continue
		grid = df.pivot_table(index='subject', columns=['training'], values='value', aggfunc='mean')
		grid = grid.dropna(axis=0, how='any')
		if grid.shape[0] < 2:
			results[sf] = None
			continue
		balanced = pd.DataFrame({
			"subject": np.repeat(grid.index.values, 2),
			"training": np.tile(['before', 'after'], grid.shape[0]),
			"value": np.r_[grid['before'].values, grid['after'].values],
		})
		try:
			res = AnovaRM(balanced, depvar='value', subject='subject', within=['training']).fit()
			p = res.anova_table.loc['training', 'Pr > F'] if 'training' in res.anova_table.index else np.nan
			results[sf] = float(p) if np.isfinite(p) else None
		except Exception:
			results[sf] = None
	return results


def compute_posthoc_training_p_per_sf_scalar_dict_metric_art(group: str, metric_dict_key: str) -> Dict[float, Optional[float]]:
	"""Post-hoc simple effects: ART-style Training test within each SF for scalar dict metrics."""
	results: Dict[float, Optional[float]] = {}
	per_sf_records = _collect_scalar_metric_records(group, metric_dict_key)
	for sf, records in per_sf_records.items():
		df = pd.DataFrame(records)
		if df.empty:
			results[sf] = None
			continue
		grid = df.pivot_table(index='subject', columns=['training'], values='value', aggfunc='mean')
		grid = grid.dropna(axis=0, how='any')
		if grid.shape[0] < 2:
			results[sf] = None
			continue
		balanced = pd.DataFrame({
			"subject": np.repeat(grid.index.values, 2),
			"training": np.tile(['before', 'after'], grid.shape[0]),
			"value": np.r_[grid['before'].values, grid['after'].values],
		})
		try:
			balanced['value_centered'] = balanced['value'] - balanced.groupby('subject')['value'].transform('mean')
			balanced['rank'] = balanced['value_centered'].rank(method='average')
			res = AnovaRM(balanced, depvar='rank', subject='subject', within=['training']).fit()
			p = res.anova_table.loc['training', 'Pr > F'] if 'training' in res.anova_table.index else np.nan
			results[sf] = float(p) if np.isfinite(p) else None
		except Exception:
			results[sf] = None
	return results


def compute_posthoc_training_p_per_sf_vector_metric_rm(group: str, metric_dict_key: str, bin_col: str) -> Dict[float, Optional[float]]:
	"""Post-hoc simple effects: Training (before vs after) within each SF for vector metrics.

	Uses AnovaRM with within=['training', bin_col] and returns p(training).
	"""
	results: Dict[float, Optional[float]] = {}
	per_sf_records = _collect_vector_metric_records(group, metric_dict_key, bin_col)
	for sf, records in per_sf_records.items():
		df = pd.DataFrame(records)
		if df.empty:
			results[sf] = None
			continue
		# Require complete grid per subject (training × bin)
		grid = df.pivot_table(index='subject', columns=['training', bin_col], values='value', aggfunc='mean')
		grid = grid.dropna(axis=0, how='any')
		if grid.shape[0] < 2:
			results[sf] = None
			continue
		bins = sorted({c[1] for c in grid.columns})
		balanced_rows: List[Dict[str, object]] = []
		for subject in grid.index:
			for training in ['before', 'after']:
				for b in bins:
					balanced_rows.append({
						"subject": subject,
						"training": training,
						bin_col: b,
						"value": float(grid.loc[subject, (training, b)]),
					})
		df_bal = pd.DataFrame(balanced_rows)
		try:
			res = AnovaRM(df_bal, depvar='value', subject='subject', within=['training', bin_col]).fit()
			p = res.anova_table.loc['training', 'Pr > F'] if 'training' in res.anova_table.index else np.nan
			results[sf] = float(p) if np.isfinite(p) else None
		except Exception:
			results[sf] = None
	return results


def compute_posthoc_training_p_per_sf_vector_metric_art(group: str, metric_dict_key: str, bin_col: str) -> Dict[float, Optional[float]]:
	"""Post-hoc simple effects: ART-style Training test within each SF for vector metrics."""
	results: Dict[float, Optional[float]] = {}
	per_sf_records = _collect_vector_metric_records(group, metric_dict_key, bin_col)
	for sf, records in per_sf_records.items():
		df = pd.DataFrame(records)
		if df.empty:
			results[sf] = None
			continue
		grid = df.pivot_table(index='subject', columns=['training', bin_col], values='value', aggfunc='mean')
		grid = grid.dropna(axis=0, how='any')
		if grid.shape[0] < 2:
			results[sf] = None
			continue
		bins = sorted({c[1] for c in grid.columns})
		balanced_rows: List[Dict[str, object]] = []
		for subject in grid.index:
			for training in ['before', 'after']:
				for b in bins:
					balanced_rows.append({
						"subject": subject,
						"training": training,
						bin_col: b,
						"value": float(grid.loc[subject, (training, b)]),
					})
		df_bal = pd.DataFrame(balanced_rows)
		try:
			df_bal['value_centered'] = df_bal['value'] - df_bal.groupby('subject')['value'].transform('mean')
			df_bal['training'] = df_bal['training'].astype('category')
			df_bal[bin_col] = df_bal[bin_col].astype('category')
			art = _aligned_rank_transform(df_bal, factor_a='training', factor_b=bin_col, subject='subject', value_col='value_centered')
			df_bal['rank_A'] = art['A']
			res = AnovaRM(df_bal, depvar='rank_A', subject='subject', within=['training', bin_col]).fit()
			p = res.anova_table.loc['training', 'Pr > F'] if 'training' in res.anova_table.index else np.nan
			results[sf] = float(p) if np.isfinite(p) else None
		except Exception:
			results[sf] = None
	return results


def compute_rmanova_scalar_metric_per_sf(group: str, metric_dict_key: str) -> Dict[float, Optional[float]]:
	"""Legacy API: returns the Training p-value for each SF.

	P-value is computed from a single Training×SF within-subject model (SF repeated-measures).
	"""
	res = compute_rmanova_training_sf_from_scalar_metric(group, metric_dict_key)
	p = res.get('training_p', None) if res else None
	keys = sorted(list(_collect_scalar_metric_records(group, metric_dict_key).keys()))
	return {k: p for k in keys}


def compute_art_scalar_metric_per_sf(group: str, metric_dict_key: str) -> Dict[float, Optional[float]]:
	"""Legacy API: returns the Training p-value for each SF.

	P-value is computed from a single Training×SF within-subject ART model (SF repeated-measures).
	"""
	res = compute_art_training_sf_from_scalar_metric(group, metric_dict_key)
	p = res.get('training_p', None) if res else None
	keys = sorted(list(_collect_scalar_metric_records(group, metric_dict_key).keys()))
	return {k: p for k in keys}


def _collect_vector_metric_records(group: str, metric_dict_key: str, bin_col: str) -> Dict[float, List[Dict[str, object]]]:
	"""Collect per-SF long-form records for vector metrics using mouse pairs.

	Returns {sf_log: [ {subject, training, <bin_col>, value}, ... ]}.
	"""
	per_sf_records: Dict[float, List[Dict[str, object]]] = {}
	pairs = MOUSE_PAIRS.get(group, [])
	for before_session, after_session in pairs:
		m_before = load_session_distance_metrics(before_session)
		m_after = load_session_distance_metrics(after_session)
		if not m_before or not m_after:
			continue
		vals_b = m_before.get(metric_dict_key, {})
		vals_a = m_after.get(metric_dict_key, {})
		if not vals_b or not vals_a:
			continue

		subject = _mouse_id_from_session_rel(before_session)
		keys_b = {round(float(k), 6): np.asarray(v, dtype=float) for k, v in vals_b.items()}
		keys_a = {round(float(k), 6): np.asarray(v, dtype=float) for k, v in vals_a.items()}
		common = set(keys_b.keys()) & set(keys_a.keys())
		for sf in common:
			vb = keys_b[sf]
			va = keys_a[sf]
			if vb.ndim != 1 or va.ndim != 1:
				continue
			min_len = min(vb.size, va.size)
			if min_len < 2:
				continue
			vb = vb[:min_len]
			va = va[:min_len]
			for i in range(min_len):
				if np.isfinite(vb[i]):
					per_sf_records.setdefault(float(sf), []).append({"subject": subject, "training": "before", bin_col: i, "value": float(vb[i])})
				if np.isfinite(va[i]):
					per_sf_records.setdefault(float(sf), []).append({"subject": subject, "training": "after", bin_col: i, "value": float(va[i])})

	return per_sf_records


def compute_rmanova_vector_metric_training_p_per_sf(group: str, metric_dict_key: str, bin_col: str) -> Dict[float, Optional[float]]:
	"""Legacy API: returns the Training p-value for each SF.

	P-value is computed from a single Training×SF within-subject model on the per-SF mean across bins.
	"""
	res = compute_rmanova_training_sf_from_vector_distance_metric(group, metric_dict_key=metric_dict_key)
	p = res.get('training_p', None) if res else None
	keys = sorted(list(_collect_vector_metric_records(group, metric_dict_key, bin_col).keys()))
	return {k: p for k in keys}


def compute_art_vector_metric_training_p_per_sf(group: str, metric_dict_key: str, bin_col: str) -> Dict[float, Optional[float]]:
	"""Legacy API: returns the Training p-value for each SF.

	P-value is computed from a single Training×SF within-subject ART model on the per-SF mean across bins.
	"""
	res = compute_art_training_sf_from_vector_distance_metric(group, metric_dict_key=metric_dict_key)
	p = res.get('training_p', None) if res else None
	keys = sorted(list(_collect_vector_metric_records(group, metric_dict_key, bin_col).keys()))
	return {k: p for k in keys}


def compute_art_adjacent_dist_per_sf(group: str) -> Dict[float, Optional[float]]:
	"""Per-SF ART p-values for training effect on adjacent-distance curves.

	Uses ART ranks for Training in a two-way within-subject design (training × adj_pair).
	Returns {sf_log: p_training}.
	"""
	return compute_art_vector_metric_training_p_per_sf(group, metric_dict_key="adj_dist_by_sf", bin_col="adj_pair")


def _fmt_sig_sf_log_list(p_by_sf_log: Dict[float, Optional[float]]) -> str:
	items: List[str] = []
	for sf_log in sorted(p_by_sf_log.keys()):
		p = p_by_sf_log.get(sf_log, None)
		if p is None or not np.isfinite(p) or p >= 0.05:
			continue
		sf_mapped = 2 ** float(sf_log)
		items.append(f"{sf_mapped:.2f}(p={p:.3f})")
	return ", ".join(items) if items else "none"


def _fmt_sig_mapped_sf_list(p_by_sf: Dict[float, Optional[float]]) -> str:
	items: List[str] = []
	for sf in sorted(p_by_sf.keys()):
		p = p_by_sf.get(sf, None)
		if p is None or not np.isfinite(p) or p >= 0.05:
			continue
		items.append(f"{float(sf):.2f}(p={float(p):.3f})")
	return ", ".join(items) if items else "none"


def _collect_session_metric_matrix_by_sf(session_list: List[str], metric_key: str) -> np.ndarray:
	"""Collect aligned scalar-per-SF session values into a sessions×SF matrix (non-baseline only)."""
	n_sf = max(0, len(TARGET_SF_VALUES) - 1)
	rows: List[np.ndarray] = []
	for session in session_list:
		try:
			summary = prepare_session_summaries(session)
		except Exception:
			continue
		vals = summary.get(metric_key)
		if vals is None:
			continue
		arr = np.asarray(vals, dtype=float).ravel()
		if arr.size < (n_sf + 1):
			continue
		rows.append(np.asarray(arr[1:n_sf + 1], dtype=float))
	if not rows:
		return np.empty((0, n_sf), dtype=float)
	return np.vstack(rows)


def _build_cross_group_scalar_metric_stats_lines(
	*,
	metric_key: str,
	group_a_sessions: List[str],
	group_b_sessions: List[str],
	mapped_levels: List[float],
	label_a: str,
	label_b: str,
) -> List[str]:
	"""Cross-group ART-Rlmer stats for scalar-per-SF grouped metrics."""
	a = _collect_session_metric_matrix_by_sf(group_a_sessions, metric_key)
	b = _collect_session_metric_matrix_by_sf(group_b_sessions, metric_key)
	return _build_cross_group_art_rlmer_stats_lines(
		wt_before=a,
		rescued_after=b,
		mapped_levels=mapped_levels,
		label_a=label_a,
		label_b=label_b,
	)


def _build_above_chance_scalar_metric_stats_lines(
	*,
	metric_key: str,
	sessions: List[str],
	label: str,
	chance_value: float,
	mapped_levels: List[float],
) -> List[str]:
	"""One-sample per-SF tests against a scalar chance level for scalar grouped metrics."""
	mat = _collect_session_metric_matrix_by_sf(sessions, metric_key)
	p_t: Dict[float, Optional[float]] = {float(sf): None for sf in mapped_levels}
	p_w: Dict[float, Optional[float]] = {float(sf): None for sf in mapped_levels}
	for idx, sf in enumerate(mapped_levels):
		if idx >= mat.shape[1]:
			break
		vals = np.asarray(mat[:, idx], dtype=float)
		vals = vals[np.isfinite(vals)]
		if vals.size < 2:
			continue
		p_t[float(sf)] = _safe_onesample_ttest_p(vals, chance_value, alternative='greater')
		p_w[float(sf)] = _safe_onesample_wilcoxon_p(vals, chance_value, alternative='greater')
	q_t = _bh_fdr_correct(p_t)
	q_w = _bh_fdr_correct(p_w)
	lines: List[str] = [
		f"{label} > chance={chance_value:.2f} @SF (one-sided)",
		"t/W: p/q(BH)",
	]
	for sf in mapped_levels:
		lines.append(
			f"SF={float(sf):.2f} t={_fmt_p_short(p_t.get(float(sf)))}/{_fmt_p_short(q_t.get(float(sf)))} W={_fmt_p_short(p_w.get(float(sf)))}/{_fmt_p_short(q_w.get(float(sf)))}"
		)
	return lines


def _build_cross_group_art_rlmer_stats_lines(
	*,
	wt_before: np.ndarray,
	rescued_after: np.ndarray,
	mapped_levels: List[float],
	label_a: str,
	label_b: str,
) -> List[str]:
	"""Build ART-Rlmer cross-group stats lines for WT-pre vs Rescued-post.

	Treats mouse as a random intercept, `cohort` as a between-subject factor, and
	SF as a repeated factor. Also computes simple cohort effects within each SF.
	"""
	a = np.asarray(wt_before, dtype=float)
	b = np.asarray(rescued_after, dtype=float)
	lines: List[str] = [f"{label_a} vs {label_b}"]
	if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or a.shape[1] != len(mapped_levels):
		lines.append("ART-Rlmer cohort p=NA")
		lines.append("ART-Rlmer SF p=NA")
		lines.append("ART-Rlmer cohort×SF p=NA")
		lines.append("ART-Rlmer cohort@SF: none")
		return lines

	rows: List[Dict[str, object]] = []
	for mi in range(a.shape[0]):
		subj = f"WTpre_{mi+1}"
		for si, sfv in enumerate(mapped_levels):
			val = float(a[mi, si])
			if np.isfinite(val):
				rows.append({"subject": subj, "cohort": label_a, "sf": f"{sfv:.2f}", "value": val})
	for mi in range(b.shape[0]):
		subj = f"Rpost_{mi+1}"
		for si, sfv in enumerate(mapped_levels):
			val = float(b[mi, si])
			if np.isfinite(val):
				rows.append({"subject": subj, "cohort": label_b, "sf": f"{sfv:.2f}", "value": val})

	df = pd.DataFrame(rows)
	cohort_p: Optional[float] = None
	sf_p: Optional[float] = None
	interaction_p: Optional[float] = None
	per_sf_p: Dict[float, Optional[float]] = {float(sf): None for sf in mapped_levels}
	if not df.empty:
		# Keep only subjects with complete SF coverage.
		n_sf = len(mapped_levels)
		counts = df.groupby('subject', observed=False)['sf'].nunique()
		keep_subjects = counts[counts == n_sf].index
		df = df[df['subject'].isin(keep_subjects)].copy()
		if not df.empty:
			for col in ['subject', 'cohort', 'sf']:
				df[col] = df[col].astype('category')
			try:
				pvals = _artool_anova_pvalues(df=df, formula_rhs="cohort*sf + (1|subject)")
				cohort_p = pvals.get('cohort')
				sf_p = pvals.get('sf')
				interaction_p = pvals.get('cohort:sf')
			except Exception:
				pass
			for sfv in mapped_levels:
				df_sf = df[df['sf'] == f"{float(sfv):.2f}"][["subject", "cohort", "value"]].copy()
				if df_sf.empty or df_sf['cohort'].nunique() < 2 or df_sf['subject'].nunique() < 2:
					continue
				for col in ['subject', 'cohort']:
					df_sf[col] = df_sf[col].astype('category')
				try:
					pvals_sf = _artool_anova_pvalues(df=df_sf, formula_rhs="cohort + (1|subject)")
					per_sf_p[float(sfv)] = pvals_sf.get('cohort')
				except Exception:
					pass

	lines.append(f"ART-Rlmer cohort p={_fmt_p_short(cohort_p)}")
	lines.append(f"ART-Rlmer SF p={_fmt_p_short(sf_p)}")
	lines.append(f"ART-Rlmer cohort×SF p={_fmt_p_short(interaction_p)}")
	lines.append(f"ART-Rlmer cohort@SF: {_fmt_sig_mapped_sf_list(per_sf_p)}")
	return lines


def _build_cross_group_art_rlmer_orientation_stats_lines(
	*,
	group_a: np.ndarray,
	group_b: np.ndarray,
	orientations: np.ndarray,
	label_a: str,
	label_b: str,
) -> List[str]:
	"""Build ART-Rlmer cross-group stats lines for orientation-aligned grouped metrics."""
	a = np.asarray(group_a, dtype=float)
	b = np.asarray(group_b, dtype=float)
	orient_arr = np.asarray(orientations, dtype=float).ravel()
	lines: List[str] = [f"{label_a} vs {label_b}"]
	if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or a.shape[1] != orient_arr.size:
		lines.append("ART-Rlmer cohort p=NA")
		lines.append("ART-Rlmer Ori p=NA")
		lines.append("ART-Rlmer cohort×Ori p=NA")
		lines.append("ART-Rlmer cohort@Ori: none")
		return lines

	rows: List[Dict[str, object]] = []
	for mi in range(a.shape[0]):
		subj = f"A_{mi+1}"
		for oi, ori_val in enumerate(orient_arr):
			val = float(a[mi, oi])
			if np.isfinite(val):
				rows.append({"subject": subj, "cohort": label_a, "ori": f"{float(ori_val):.0f}", "value": val})
	for mi in range(b.shape[0]):
		subj = f"B_{mi+1}"
		for oi, ori_val in enumerate(orient_arr):
			val = float(b[mi, oi])
			if np.isfinite(val):
				rows.append({"subject": subj, "cohort": label_b, "ori": f"{float(ori_val):.0f}", "value": val})

	df = pd.DataFrame(rows)
	cohort_p: Optional[float] = None
	ori_p: Optional[float] = None
	interaction_p: Optional[float] = None
	per_ori_idx_p: Dict[int, Optional[float]] = {oi: None for oi in range(orient_arr.size)}
	if not df.empty:
		counts = df.groupby('subject', observed=False)['ori'].nunique()
		keep_subjects = counts[counts == orient_arr.size].index
		df = df[df['subject'].isin(keep_subjects)].copy()
		if not df.empty:
			for col in ['subject', 'cohort', 'ori']:
				df[col] = df[col].astype('category')
			try:
				pvals = _artool_anova_pvalues(df=df, formula_rhs="cohort*ori + (1|subject)")
				cohort_p = pvals.get('cohort')
				ori_p = pvals.get('ori')
				interaction_p = pvals.get('cohort:ori')
			except Exception:
				pass
			for oi, ori_val in enumerate(orient_arr):
				ori_key = f"{float(ori_val):.0f}"
				df_ori = df[df['ori'] == ori_key][['subject', 'cohort', 'value']].copy()
				if df_ori.empty or df_ori['cohort'].nunique() < 2 or df_ori['subject'].nunique() < 2:
					continue
				for col in ['subject', 'cohort']:
					df_ori[col] = df_ori[col].astype('category')
				try:
					pvals_ori = _artool_anova_pvalues(df=df_ori, formula_rhs="cohort + (1|subject)")
					per_ori_idx_p[oi] = pvals_ori.get('cohort')
				except Exception:
					pass

	lines.append(f"ART-Rlmer cohort p={_fmt_p_short(cohort_p)}")
	lines.append(f"ART-Rlmer Ori p={_fmt_p_short(ori_p)}")
	lines.append(f"ART-Rlmer cohort×Ori p={_fmt_p_short(interaction_p)}")
	lines.append(f"ART-Rlmer cohort@Ori: {_format_sig_orientation_from_q(orient_arr, per_ori_idx_p)}")
	return lines


def _fmt_p_short(p: Optional[float]) -> str:
	"""Format a p-value for small stats boxes."""
	if p is None or not np.isfinite(p):
		return "NA"
	return f"{float(p):.3f}"


def plot_distance_to_com_comparison():
	"""Plot distance to COM for Before vs After conditions by group, one subplot per SF."""
	agg = aggregate_distance_metrics_by_group("dist_to_com")
	
	if not agg:
		print("No distance-to-COM data available")
		return
	
	for group in ["WT", "RESCUED"]:
		if group not in agg:
			continue
		
		before_data = agg[group].get("BEFORE")
		after_data = agg[group].get("AFTER")
		
		if before_data is None or after_data is None:
			continue
		
		mean_before, sem_before, sfs_before = before_data
		mean_after, sem_after, sfs_after = after_data
		
		# Align SFs
		common_sfs = np.intersect1d(sfs_before, sfs_after)
		if len(common_sfs) == 0:
			continue
		
		idx_before = [np.where(sfs_before == sf)[0][0] for sf in common_sfs]
		idx_after = [np.where(sfs_after == sf)[0][0] for sf in common_sfs]
		
		# Create subplots: one per SF
		n_sfs = len(common_sfs)
		n_cols = min(3, n_sfs)
		n_rows = int(np.ceil(n_sfs / n_cols))
		fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
		axes = axes.flatten()
		
		color_before = "#9ecae1" if group == "WT" else "#fcae91"
		color_after = "#08306b" if group == "WT" else "#a50f15"
		
		# Training×SF (SF repeated-measures) p-values
		p_rm = compute_rmanova_distance_to_com(group)
		p_art = compute_art_distance_to_com(group)
		p_art_raov = compute_art_raov_training_sf_from_vector_distance_metric(group, metric_dict_key='dist_to_com_by_sf')
		p_art_rlmer = compute_art_r_training_sf_from_vector_distance_metric(group, metric_dict_key='dist_to_com_by_sf')
		# Post-hoc: Training within each SF
		posthoc_rm = compute_posthoc_training_p_per_sf_vector_metric_rm(group, metric_dict_key='dist_to_com_by_sf', bin_col='ori_bin')
		posthoc_art = compute_posthoc_training_p_per_sf_vector_metric_art(group, metric_dict_key='dist_to_com_by_sf', bin_col='ori_bin')
		posthoc_rm_r = {round(float(k), 6): v for k, v in posthoc_rm.items()}
		posthoc_art_r = {round(float(k), 6): v for k, v in posthoc_art.items()}
		
		# Plot each SF in its own subplot
		for sf_i, (sf_val, idx_b, idx_a) in enumerate(zip(common_sfs, idx_before, idx_after)):
			ax = axes[sf_i]
			
			mean_before_sf = mean_before[idx_b]
			sem_before_sf = sem_before[idx_b]
			mean_after_sf = mean_after[idx_a]
			sem_after_sf = sem_after[idx_a]
			
			ori_bins = np.arange(len(mean_before_sf))
			
			# Plot Before
			ax.errorbar(ori_bins, mean_before_sf, yerr=sem_before_sf, fmt='-o', capsize=3,
					color=color_before, linewidth=2, markersize=6, label='Before', alpha=0.8)
			
			# Plot After
			ax.errorbar(ori_bins, mean_after_sf, yerr=sem_after_sf, fmt='-s', capsize=3,
					color=color_after, linewidth=2, markersize=6, label='After', alpha=0.8)
			
			ax.set_xlabel('Orientation bin', fontsize=10)
			ax.set_ylabel('Distance to COM', fontsize=10)
			ax.set_title(f'SF={sf_val:.2f}', fontsize=11, fontweight='bold')
			ax.legend(fontsize=9)
			ax.grid(alpha=0.3)
			
			# Add group-level stats (same for all subplots). Always show ART-Rlmer even if NA.
			stat_lines: List[str] = []
			stat_lines.append(f"RM train p={_fmt_p_short((p_rm or {}).get('training_p'))}")
			stat_lines.append(f"RM train×SF p={_fmt_p_short((p_rm or {}).get('interaction_p'))}")
			stat_lines.append(f"ART train p={_fmt_p_short((p_art or {}).get('training_p'))}")
			stat_lines.append(f"ART train×SF p={_fmt_p_short((p_art or {}).get('interaction_p'))}")
			stat_lines.append(f"ART-Raov train p={_fmt_p_short((p_art_raov or {}).get('training_p'))}")
			stat_lines.append(f"ART-Raov train×SF p={_fmt_p_short((p_art_raov or {}).get('interaction_p'))}")
			stat_lines.append(f"ART-Rlmer train p={_fmt_p_short((p_art_rlmer or {}).get('training_p'))}")
			stat_lines.append(f"ART-Rlmer train×SF p={_fmt_p_short((p_art_rlmer or {}).get('interaction_p'))}")
			# Per-SF simple effects (Training within this SF)
			sf_key = round(float(sf_val), 6)
			p_sf_rm = posthoc_rm_r.get(sf_key, None)
			p_sf_art = posthoc_art_r.get(sf_key, None)
			stat_lines.append(f"RM train@SF p={_fmt_p_short(p_sf_rm)}")
			stat_lines.append(f"ART train@SF p={_fmt_p_short(p_sf_art)}")
			if stat_lines:
				ax.text(0.98, 0.98, "\n".join(stat_lines), transform=ax.transAxes, fontsize=8,
						verticalalignment='top', horizontalalignment='right',
						bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
		
		# Hide unused subplots
		for idx in range(n_sfs, len(axes)):
			axes[idx].axis('off')
		
		group_label = "WT" if group == "WT" else "Rescued"
		fig.suptitle(f'{group_label}: Distance to COM (Before vs After)', fontsize=14, fontweight='bold')
		
		plt.tight_layout(rect=[0, 0, 1, 0.97])
		filename = f'dspca_distance_to_com_before_after_{group}.svg'
		plt.savefig(os.path.join(GROUPED_PLOTS_DIR, filename), format='svg', bbox_inches='tight')
		plt.close()
		print(f"  [saved] {filename}")


def plot_adjacent_distance_comparison():
	"""Plot adjacent orientation distances for Before vs After conditions by group, one subplot per SF."""
	agg = aggregate_distance_metrics_by_group("adj_dist")
	
	if not agg:
		print("No adjacent distance data available")
		return
	
	for group in ["WT", "RESCUED"]:
		if group not in agg:
			continue
		
		before_data = agg[group].get("BEFORE")
		after_data = agg[group].get("AFTER")
		
		if before_data is None or after_data is None:
			continue
		
		mean_before, sem_before, sfs_before = before_data
		mean_after, sem_after, sfs_after = after_data
		
		# Align SFs
		common_sfs = np.intersect1d(sfs_before, sfs_after)
		if len(common_sfs) == 0:
			continue
		
		idx_before = [np.where(sfs_before == sf)[0][0] for sf in common_sfs]
		idx_after = [np.where(sfs_after == sf)[0][0] for sf in common_sfs]
		
		# Orientation pair labels
		ori_pairs = ['0-30', '30-60', '60-90', '90-120', '120-150', '150-0']
		
		# Create subplots: one per SF
		n_sfs = len(common_sfs)
		n_cols = min(3, n_sfs)
		n_rows = int(np.ceil(n_sfs / n_cols))
		fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
		axes = axes.flatten()
		
		color_before = "#9ecae1" if group == "WT" else "#fcae91"
		color_after = "#08306b" if group == "WT" else "#a50f15"
		
		# Training×SF (SF repeated-measures) p-values
		p_rm = compute_rmanova_adjacent_dist(group)
		p_art = compute_art_adjacent_dist(group)
		p_art_raov = compute_art_raov_training_sf_from_vector_distance_metric(group, metric_dict_key='adj_dist_by_sf')
		p_art_rlmer = compute_art_r_training_sf_from_vector_distance_metric(group, metric_dict_key='adj_dist_by_sf')
		# Post-hoc: Training within each SF
		posthoc_rm = compute_posthoc_training_p_per_sf_vector_metric_rm(group, metric_dict_key='adj_dist_by_sf', bin_col='adj_pair')
		posthoc_art = compute_posthoc_training_p_per_sf_vector_metric_art(group, metric_dict_key='adj_dist_by_sf', bin_col='adj_pair')
		posthoc_rm_r = {round(float(k), 6): v for k, v in posthoc_rm.items()}
		posthoc_art_r = {round(float(k), 6): v for k, v in posthoc_art.items()}
		
		# Plot each SF in its own subplot
		for sf_i, (sf_val, idx_b, idx_a) in enumerate(zip(common_sfs, idx_before, idx_after)):
			ax = axes[sf_i]
			
			mean_before_sf = mean_before[idx_b]
			sem_before_sf = sem_before[idx_b]
			mean_after_sf = mean_after[idx_a]
			sem_after_sf = sem_after[idx_a]
			
			bin_idx = np.arange(len(mean_before_sf))
			
			# Plot Before
			ax.errorbar(bin_idx, mean_before_sf, yerr=sem_before_sf, fmt='-o', capsize=3,
					color=color_before, linewidth=2, markersize=6, label='Before', alpha=0.8)
			
			# Plot After
			ax.errorbar(bin_idx, mean_after_sf, yerr=sem_after_sf, fmt='-s', capsize=3,
					color=color_after, linewidth=2, markersize=6, label='After', alpha=0.8)
			
			ax.set_xlabel('Adjacent orientation pair', fontsize=10)
			ax.set_ylabel('Distance', fontsize=10)
			ax.set_title(f'SF={sf_val:.2f}', fontsize=11, fontweight='bold')
			ax.set_xticks(bin_idx)
			ax.set_xticklabels(ori_pairs, rotation=45, ha='right', fontsize=8)
			ax.legend(fontsize=9)
			ax.grid(alpha=0.3)
			
			# Add group-level stats (same for all subplots). Always show ART-Rlmer even if NA.
			stat_lines: List[str] = []
			stat_lines.append(f"RM train p={_fmt_p_short((p_rm or {}).get('training_p'))}")
			stat_lines.append(f"RM train×SF p={_fmt_p_short((p_rm or {}).get('interaction_p'))}")
			stat_lines.append(f"ART train p={_fmt_p_short((p_art or {}).get('training_p'))}")
			stat_lines.append(f"ART train×SF p={_fmt_p_short((p_art or {}).get('interaction_p'))}")
			stat_lines.append(f"ART-Raov train p={_fmt_p_short((p_art_raov or {}).get('training_p'))}")
			stat_lines.append(f"ART-Raov train×SF p={_fmt_p_short((p_art_raov or {}).get('interaction_p'))}")
			stat_lines.append(f"ART-Rlmer train p={_fmt_p_short((p_art_rlmer or {}).get('training_p'))}")
			stat_lines.append(f"ART-Rlmer train×SF p={_fmt_p_short((p_art_rlmer or {}).get('interaction_p'))}")
			# Per-SF simple effects (Training within this SF)
			sf_key = round(float(sf_val), 6)
			p_sf_rm = posthoc_rm_r.get(sf_key, None)
			p_sf_art = posthoc_art_r.get(sf_key, None)
			stat_lines.append(f"RM train@SF p={_fmt_p_short(p_sf_rm)}")
			stat_lines.append(f"ART train@SF p={_fmt_p_short(p_sf_art)}")
			if stat_lines:
				ax.text(0.98, 0.98, "\n".join(stat_lines), transform=ax.transAxes, fontsize=8,
						verticalalignment='top', horizontalalignment='right',
						bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
		
		# Hide unused subplots
		for idx in range(n_sfs, len(axes)):
			axes[idx].axis('off')
		
		group_label = "WT" if group == "WT" else "Rescued"
		fig.suptitle(f'{group_label}: Adjacent Orientation Distance (Before vs After)', fontsize=14, fontweight='bold')
		
		plt.tight_layout(rect=[0, 0, 1, 0.97])
		filename = f'dspca_adjacent_orientation_distance_before_after_{group}.svg'
		plt.savefig(os.path.join(GROUPED_PLOTS_DIR, filename), format='svg', bbox_inches='tight')
		plt.close()
		print(f"  [saved] {filename}")


def plot_adjacent_cv_comparison():
	"""Plot per-SF edge-length CV (equidistance) Before vs After for each group."""
	agg = aggregate_distance_metrics_by_group("adj_cv")
	if not agg:
		print("No adjacent-distance CV data available")
		return

	for group in ["WT", "RESCUED"]:
		if group not in agg:
			continue
		before_data = agg[group].get("BEFORE")
		after_data = agg[group].get("AFTER")
		if before_data is None or after_data is None:
			continue

		mean_before, sem_before, sfs_before = before_data
		mean_after, sem_after, sfs_after = after_data
		common_sfs = np.intersect1d(sfs_before, sfs_after)
		if len(common_sfs) == 0:
			continue

		idx_before = [np.where(sfs_before == sf)[0][0] for sf in common_sfs]
		idx_after = [np.where(sfs_after == sf)[0][0] for sf in common_sfs]

		x_sf = 2 ** common_sfs  # convert log2(mapped_sf) back to mapped_sf
		yb = np.array([mean_before[i] for i in idx_before], dtype=float)
		ya = np.array([mean_after[i] for i in idx_after], dtype=float)
		sb = np.array([sem_before[i] for i in idx_before], dtype=float)
		sa = np.array([sem_after[i] for i in idx_after], dtype=float)

		color_before = "#9ecae1" if group == "WT" else "#fcae91"
		color_after = "#08306b" if group == "WT" else "#a50f15"

		fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
		ax.errorbar(x_sf, yb, yerr=sb, fmt='-o', capsize=3, color=color_before,
				linewidth=2, markersize=6, label='Before', alpha=0.85)
		ax.errorbar(x_sf, ya, yerr=sa, fmt='-s', capsize=3, color=color_after,
				linewidth=2, markersize=6, label='After', alpha=0.85)
		ax.set_xscale('log')
		ax.set_xlabel('Spatial frequency (mapped)', fontsize=11)
		ax.set_ylabel('Adjacent-edge CV (std/mean)', fontsize=11)
		ax.grid(alpha=0.3, which='both')
		ax.legend(fontsize=10)

		# Training×SF repeated-measures p-values
		p_rm = compute_rmanova_training_sf_from_scalar_metric(group, 'adj_dist_cv_by_sf')
		p_art = compute_art_training_sf_from_scalar_metric(group, 'adj_dist_cv_by_sf')
		p_art_raov = compute_art_raov_training_sf_from_scalar_metric(group, 'adj_dist_cv_by_sf')
		p_art_rlmer = compute_art_r_training_sf_from_scalar_metric(group, 'adj_dist_cv_by_sf')
		# Post-hoc: Training within each SF (only displayed if interaction suggests SF-dependent effects)
		posthoc_rm = compute_posthoc_training_p_per_sf_scalar_dict_metric_rm(group, metric_dict_key='adj_dist_cv_by_sf')
		posthoc_art = compute_posthoc_training_p_per_sf_scalar_dict_metric_art(group, metric_dict_key='adj_dist_cv_by_sf')

		# Stars per SF for this group (RM posthoc within each SF)
		posthoc_rm_fdr = _bh_fdr_correct(posthoc_rm)
		star_transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
		for sf_log in common_sfs:
			p = posthoc_rm.get(float(sf_log), None)
			q = posthoc_rm_fdr.get(float(sf_log), None)
			x_i = float(2 ** float(sf_log))
			if p is not None and np.isfinite(p) and p < 0.05:
				ax.text(x_i, 0.96, _asterisks_from_p(float(p)), transform=star_transform,
					ha='center', va='top', fontsize=10, fontweight='bold')
			if q is not None and np.isfinite(q) and q < 0.05:
				ax.text(x_i, 0.91, _asterisks_from_p(float(q)), transform=star_transform,
					ha='center', va='top', fontsize=9, fontweight='bold')
		lines: List[str] = []
		lines.append(f"RM train p={_fmt_p_short((p_rm or {}).get('training_p'))}")
		lines.append(f"RM train×SF p={_fmt_p_short((p_rm or {}).get('interaction_p'))}")
		lines.append(f"ART train p={_fmt_p_short((p_art or {}).get('training_p'))}")
		lines.append(f"ART train×SF p={_fmt_p_short((p_art or {}).get('interaction_p'))}")
		lines.append(f"ART-Raov train p={_fmt_p_short((p_art_raov or {}).get('training_p'))}")
		lines.append(f"ART-Raov train×SF p={_fmt_p_short((p_art_raov or {}).get('interaction_p'))}")
		lines.append(f"ART-Rlmer train p={_fmt_p_short((p_art_rlmer or {}).get('training_p'))}")
		lines.append(f"ART-Rlmer train×SF p={_fmt_p_short((p_art_rlmer or {}).get('interaction_p'))}")
		lines.append(f"simple train@SF (RM): {_fmt_sig_sf_log_list(posthoc_rm)}")
		lines.append(f"simple train@SF (ART): {_fmt_sig_sf_log_list(posthoc_art)}")
		if lines:
			ax.text(0.98, 0.98, "\n".join(lines), transform=ax.transAxes, fontsize=9,
					verticalalignment='top', horizontalalignment='right',
					bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

		group_label = "WT" if group == "WT" else "Rescued"
		fig.suptitle(f'{group_label}: dsPCA orientation-plane equidistance (edge CV)', fontsize=13, fontweight='bold')
		plt.tight_layout(rect=[0, 0, 1, 0.95])
		filename = f'dspca_adjacent_edge_cv_before_after_{group}.svg'
		plt.savefig(os.path.join(GROUPED_PLOTS_DIR, filename), format='svg', bbox_inches='tight')
		plt.close()
		print(f"  [saved] {filename}")


def _plot_scalar_metric_over_sf(metric_type: str, ylabel: str, title_suffix: str, out_prefix: str, p_metric_key: str) -> None:
	"""Internal helper: plot scalar-per-SF metric Before vs After for WT/RESCUED."""
	agg = aggregate_distance_metrics_by_group(metric_type)
	if not agg:
		print(f"No {metric_type} data available")
		return

	for group in ["WT", "RESCUED"]:
		if group not in agg:
			continue
		before_data = agg[group].get("BEFORE")
		after_data = agg[group].get("AFTER")
		if before_data is None or after_data is None:
			continue

		mean_before, sem_before, sfs_before = before_data
		mean_after, sem_after, sfs_after = after_data
		common_sfs = np.intersect1d(sfs_before, sfs_after)
		if len(common_sfs) == 0:
			continue

		idx_before = [np.where(sfs_before == sf)[0][0] for sf in common_sfs]
		idx_after = [np.where(sfs_after == sf)[0][0] for sf in common_sfs]

		x_sf = 2 ** common_sfs
		yb = np.array([mean_before[i] for i in idx_before], dtype=float)
		ya = np.array([mean_after[i] for i in idx_after], dtype=float)
		sb = np.array([sem_before[i] for i in idx_before], dtype=float)
		sa = np.array([sem_after[i] for i in idx_after], dtype=float)

		color_before = "#9ecae1" if group == "WT" else "#fcae91"
		color_after = "#08306b" if group == "WT" else "#a50f15"

		fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
		ax.errorbar(x_sf, yb, yerr=sb, fmt='-o', capsize=3, color=color_before,
				linewidth=2, markersize=6, label='Before', alpha=0.85)
		ax.errorbar(x_sf, ya, yerr=sa, fmt='-s', capsize=3, color=color_after,
				linewidth=2, markersize=6, label='After', alpha=0.85)
		ax.set_xscale('log')
		ax.set_xlabel('Spatial frequency (mapped)', fontsize=11)
		ax.set_ylabel(ylabel, fontsize=11)
		ax.grid(alpha=0.3, which='both')
		ax.legend(fontsize=10)

		p_rm = compute_rmanova_training_sf_from_scalar_metric(group, p_metric_key)
		p_art = compute_art_training_sf_from_scalar_metric(group, p_metric_key)
		p_art_raov = compute_art_raov_training_sf_from_scalar_metric(group, p_metric_key)
		p_art_rlmer = compute_art_r_training_sf_from_scalar_metric(group, p_metric_key)
		# Post-hoc: Training within each SF (only displayed if interaction suggests SF-dependent effects)
		posthoc_rm = compute_posthoc_training_p_per_sf_scalar_dict_metric_rm(group, metric_dict_key=p_metric_key)
		posthoc_art = compute_posthoc_training_p_per_sf_scalar_dict_metric_art(group, metric_dict_key=p_metric_key)

		# Stars per SF for this group (RM posthoc within each SF)
		posthoc_rm_fdr = _bh_fdr_correct(posthoc_rm)
		star_transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
		for sf_log in common_sfs:
			p = posthoc_rm.get(float(sf_log), None)
			q = posthoc_rm_fdr.get(float(sf_log), None)
			x_i = float(2 ** float(sf_log))
			if p is not None and np.isfinite(p) and p < 0.05:
				ax.text(x_i, 0.96, _asterisks_from_p(float(p)), transform=star_transform,
					ha='center', va='top', fontsize=10, fontweight='bold')
			if q is not None and np.isfinite(q) and q < 0.05:
				ax.text(x_i, 0.91, _asterisks_from_p(float(q)), transform=star_transform,
					ha='center', va='top', fontsize=9, fontweight='bold')
		lines: List[str] = []
		lines.append(f"RM train p={_fmt_p_short((p_rm or {}).get('training_p'))}")
		lines.append(f"RM train×SF p={_fmt_p_short((p_rm or {}).get('interaction_p'))}")
		lines.append(f"ART train p={_fmt_p_short((p_art or {}).get('training_p'))}")
		lines.append(f"ART train×SF p={_fmt_p_short((p_art or {}).get('interaction_p'))}")
		lines.append(f"ART-Raov train p={_fmt_p_short((p_art_raov or {}).get('training_p'))}")
		lines.append(f"ART-Raov train×SF p={_fmt_p_short((p_art_raov or {}).get('interaction_p'))}")
		lines.append(f"ART-Rlmer train p={_fmt_p_short((p_art_rlmer or {}).get('training_p'))}")
		lines.append(f"ART-Rlmer train×SF p={_fmt_p_short((p_art_rlmer or {}).get('interaction_p'))}")
		lines.append(f"simple train@SF (RM): {_fmt_sig_sf_log_list(posthoc_rm)}")
		lines.append(f"simple train@SF (ART): {_fmt_sig_sf_log_list(posthoc_art)}")
		if lines:
			ax.text(0.98, 0.98, "\n".join(lines), transform=ax.transAxes, fontsize=9,
					verticalalignment='top', horizontalalignment='right',
					bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

		group_label = "WT" if group == "WT" else "Rescued"
		fig.suptitle(f'{group_label}: {title_suffix} (Before vs After)', fontsize=13, fontweight='bold')
		plt.tight_layout(rect=[0, 0, 1, 0.95])
		filename = f'{out_prefix}_{group}.svg'
		plt.savefig(os.path.join(GROUPED_PLOTS_DIR, filename), format='svg', bbox_inches='tight')
		plt.close()
		print(f"  [saved] {filename}")


def plot_circle_radial_cv_comparison() -> None:
	"""Plot fitted-circle radial CV (std(r)/mean(r)) per SF, Before vs After."""
	_plot_scalar_metric_over_sf(
		metric_type="radial_cv",
		ylabel="Radial CV (std(r)/mean(r))",
		title_suffix="Circle-like radial consistency",
		out_prefix="dspca_circle_radial_cv_before_after",
		p_metric_key="radial_cv_by_sf",
	)


def plot_circle_fit_rmse_comparison() -> None:
	"""Plot normalized circle-fit RMSE per SF, Before vs After."""
	_plot_scalar_metric_over_sf(
		metric_type="circle_rmse_norm",
		ylabel="Circle-fit RMSE / r_fit",
		title_suffix="Circle-fit residual (normalized)",
		out_prefix="dspca_circle_fit_rmse_norm_before_after",
		p_metric_key="circle_rmse_norm_by_sf",
	)


def plot_circle_angle_spacing_comparison() -> None:
	"""Plot normalized angular-spacing deviation per SF, Before vs After."""
	_plot_scalar_metric_over_sf(
		metric_type="angle_spacing_rmse_norm",
		ylabel="Angle-spacing RMSE / ideal",
		title_suffix="Along-circle spacing regularity",
		out_prefix="dspca_circle_angle_spacing_rmse_norm_before_after",
		p_metric_key="angle_spacing_rmse_norm_by_sf",
	)


# ---------------------------------------------------------------------------
# Decoding-vs-cell-count grouped figures
# ---------------------------------------------------------------------------

def _get_metric_for_n(summary: Dict[str, np.ndarray], base_key: str, n: int) -> Optional[np.ndarray]:
	"""Fetch per-N metric if present; otherwise fall back to base_key."""
	k = f"{base_key}_n{int(n)}"
	if k in summary:
		return summary.get(k)
	return summary.get(base_key)


def _compute_training_x_n_stats(
	*,
	group: str,
	subjects: List[str],
	vals_pre: np.ndarray,
	vals_post: np.ndarray,
	cell_counts: List[int],
) -> Tuple[Dict[str, Optional[float]], Dict[int, Optional[float]], Dict[int, Optional[float]], Dict[int, Optional[float]], Dict[int, Optional[float]]]:
	"""Compute RM + ART p-values and paired posthoc (t/W) with BH-FDR across N.

	Returns:
	- omnibus: {"rm_train_p", "rm_n_p", "rm_inter_p", "art_train_p", "art_n_p", "art_inter_p"}
	- p_t_by_n, q_t_by_n, p_w_by_n, q_w_by_n
	"""
	# Per-N paired tests
	p_t: Dict[int, Optional[float]] = {int(n): None for n in cell_counts}
	p_w: Dict[int, Optional[float]] = {int(n): None for n in cell_counts}
	for j, n in enumerate(cell_counts):
		xb = np.asarray(vals_pre[:, j], dtype=float)
		xa = np.asarray(vals_post[:, j], dtype=float)
		mask = np.isfinite(xb) & np.isfinite(xa)
		if int(np.sum(mask)) < 2:
			continue
		try:
			p_t[int(n)] = float(ttest_rel(xb[mask], xa[mask], nan_policy='omit').pvalue)
		except Exception:
			p_t[int(n)] = None
		try:
			p = _safe_wilcoxon_p(xb[mask], xa[mask])
			p_w[int(n)] = float(p) if (p is not None and np.isfinite(p)) else None
		except Exception:
			p_w[int(n)] = None

	q_t = _bh_fdr_correct(p_t)
	q_w = _bh_fdr_correct(p_w)

	# Omnibus RM/ART
	omnibus: Dict[str, Optional[float]] = {
		"rm_train_p": None,
		"rm_n_p": None,
		"rm_inter_p": None,
		"art_train_p": None,
		"art_n_p": None,
		"art_inter_p": None,
		"art_r_train_p": None,
		"art_r_n_p": None,
		"art_r_inter_p": None,
		"lme_train_p": None,
		"lme_n_p": None,
		"lme_inter_p": None,
		"lme4_train_p": None,
		"lme4_n_p": None,
		"lme4_inter_p": None,
	}

	rows: List[Dict[str, object]] = []
	for i_sub, subj in enumerate(subjects):
		for j, n in enumerate(cell_counts):
			vb = float(vals_pre[i_sub, j])
			va = float(vals_post[i_sub, j])
			if np.isfinite(vb) and np.isfinite(va):
				rows.append({"subject": subj, "training": "Pre", "n": str(int(n)), "value": vb})
				rows.append({"subject": subj, "training": "Post", "n": str(int(n)), "value": va})

	df = pd.DataFrame(rows)
	if df.empty:
		return omnibus, p_t, q_t, p_w, q_w

	# Balance: keep only N levels with complete Pre+Post across all included subjects
	grid = df.pivot_table(index='subject', columns=['training', 'n'], values='value', aggfunc='mean')
	all_ns = sorted({c[1] for c in grid.columns})
	good_ns: List[str] = []
	for n in all_ns:
		if (('Pre', n) in grid.columns) and (('Post', n) in grid.columns):
			col = grid[[('Pre', n), ('Post', n)]]
			if not col.isna().any().any():
				good_ns.append(n)
	if len(good_ns) < 2:
		return omnibus, p_t, q_t, p_w, q_w

	df = df[df['n'].isin(good_ns)].copy()
	grid2 = df.pivot_table(index='subject', columns=['training', 'n'], values='value', aggfunc='mean')
	grid2 = grid2.dropna(axis=0, how='any')
	if grid2.shape[0] < 2:
		return omnibus, p_t, q_t, p_w, q_w

	balanced_rows: List[Dict[str, object]] = []
	for subj in grid2.index:
		for training in ['Pre', 'Post']:
			for n in good_ns:
				balanced_rows.append({
					"subject": subj,
					"training": training,
					"n": n,
					"value": float(grid2.loc[subj, (training, n)]),
				})
	df_bal = pd.DataFrame(balanced_rows)
	try:
		# Remove subject main effect
		df_bal['value_centered'] = df_bal['value'] - df_bal.groupby('subject')['value'].transform('mean')
		df_bal['training'] = df_bal['training'].astype('category')
		df_bal['n'] = df_bal['n'].astype('category')
		if RM_ANOVA_INCLUDE_INTERACTION_TERMS:
			anova_rm = AnovaRM(df_bal, depvar='value_centered', subject='subject', within=['training', 'n']).fit().anova_table
			omnibus['rm_train_p'] = float(anova_rm.loc['training', 'Pr > F']) if 'training' in anova_rm.index else None
			omnibus['rm_n_p'] = float(anova_rm.loc['n', 'Pr > F']) if 'n' in anova_rm.index else None
			omnibus['rm_inter_p'] = float(anova_rm.loc['training:n', 'Pr > F']) if 'training:n' in anova_rm.index else None
		else:
			df_train = df_bal.groupby(['subject', 'training'], as_index=False)['value_centered'].mean()
			omnibus['rm_train_p'] = float(
				AnovaRM(df_train, depvar='value_centered', subject='subject', within=['training']).fit().anova_table.loc['training', 'Pr > F']
			)
			df_n = df_bal.groupby(['subject', 'n'], as_index=False)['value_centered'].mean()
			omnibus['rm_n_p'] = float(
				AnovaRM(df_n, depvar='value_centered', subject='subject', within=['n']).fit().anova_table.loc['n', 'Pr > F']
			)
			omnibus['rm_inter_p'] = None
		art = _aligned_rank_transform(
			df_bal,
			factor_a='training',
			factor_b='n',
			subject='subject',
			value_col='value_centered',
			include_interaction=ART_INCLUDE_INTERACTION_TERMS,
		)
		df_bal['rank_A'] = art['A']
		df_bal['rank_B'] = art['B']
		p_int = np.nan
		if ART_INCLUDE_INTERACTION_TERMS:
			p_train = AnovaRM(df_bal, depvar='rank_A', subject='subject', within=['training', 'n']).fit().anova_table.loc['training', 'Pr > F']
			p_n = AnovaRM(df_bal, depvar='rank_B', subject='subject', within=['training', 'n']).fit().anova_table.loc['n', 'Pr > F']
			if 'AxB' in art:
				df_bal['rank_Inter'] = art['AxB']
				p_int = AnovaRM(df_bal, depvar='rank_Inter', subject='subject', within=['training', 'n']).fit().anova_table.loc['training:n', 'Pr > F']
		else:
			df_a = df_bal.groupby(['subject', 'training'], as_index=False)['rank_A'].mean()
			p_train = AnovaRM(df_a, depvar='rank_A', subject='subject', within=['training']).fit().anova_table.loc['training', 'Pr > F']
			df_b = df_bal.groupby(['subject', 'n'], as_index=False)['rank_B'].mean()
			p_n = AnovaRM(df_b, depvar='rank_B', subject='subject', within=['n']).fit().anova_table.loc['n', 'Pr > F']
		omnibus['art_train_p'] = float(p_train) if np.isfinite(p_train) else None
		omnibus['art_n_p'] = float(p_n) if np.isfinite(p_n) else None
		omnibus['art_inter_p'] = float(p_int) if np.isfinite(p_int) else None

		# ART-Raov / ART-Rlmer (2-way) omnibus (Training×N)
		art_raov2 = _art_raov_2way_pvalues_from_rows(
			balanced_rows,
			subject='subject',
			factor_a='training',
			factor_b='n',
			value_col='value',
			include_interactions=ART_INCLUDE_INTERACTION_TERMS,
		)
		art_rlmer2 = _art_r_2way_pvalues_from_rows(
			balanced_rows,
			subject='subject',
			factor_a='training',
			factor_b='n',
			value_col='value',
			include_interactions=ART_INCLUDE_INTERACTION_TERMS,
		)
		omnibus['art_raov_train_p'] = art_raov2.get('a')
		omnibus['art_raov_n_p'] = art_raov2.get('b')
		omnibus['art_raov_inter_p'] = art_raov2.get('a:b')
		omnibus['art_rlmer_train_p'] = art_rlmer2.get('a')
		omnibus['art_rlmer_n_p'] = art_rlmer2.get('b')
		omnibus['art_rlmer_inter_p'] = art_rlmer2.get('a:b')
		# Back-compat keys
		omnibus['art_r_train_p'] = omnibus.get('art_rlmer_train_p')
		omnibus['art_r_n_p'] = omnibus.get('art_rlmer_n_p')
		omnibus['art_r_inter_p'] = omnibus.get('art_rlmer_inter_p')

		# LME (2-way) omnibus (Training×N)
		lme2 = _lme_2way_lr_pvalues_from_rows(
			balanced_rows,
			subject='subject',
			factor_a='training',
			factor_b='n',
			value_col='value',
			include_interactions=LME_INCLUDE_INTERACTION_TERMS,
		)
		omnibus['lme_train_p'] = lme2.get('a')
		omnibus['lme_n_p'] = lme2.get('b')
		omnibus['lme_inter_p'] = lme2.get('a:b')

		# LME-lme4 (2-way) omnibus (Training×N)
		lme4_2 = _lme4_2way_lrt_pvalues_from_rows(
			balanced_rows,
			subject='subject',
			factor_a='training',
			factor_b='n',
			value_col='value',
			include_interactions=LME_INCLUDE_INTERACTION_TERMS,
		)
		omnibus['lme4_train_p'] = lme4_2.get('a')
		omnibus['lme4_n_p'] = lme4_2.get('b')
		omnibus['lme4_inter_p'] = lme4_2.get('a:b')
	except Exception as exc:
		print(f"[debug] Training×N omnibus failed for {group}: {type(exc).__name__}")

	return omnibus, p_t, q_t, p_w, q_w


def _compute_observed_vs_shuffled_x_n_art_stats(
	*,
	subjects: List[str],
	observed_vals: np.ndarray,
	shuffled_vals: np.ndarray,
	cell_counts: List[int],
) -> Dict[str, Optional[float]]:
	"""Compute ART omnibus stats for observed-vs-shuffled comparisons across neuron counts."""
	omnibus: Dict[str, Optional[float]] = {
		"art_source_p": None,
		"art_n_p": None,
		"art_inter_p": None,
		"art_raov_source_p": None,
		"art_raov_n_p": None,
		"art_raov_inter_p": None,
		"art_rlmer_source_p": None,
		"art_rlmer_n_p": None,
		"art_rlmer_inter_p": None,
	}
	rows: List[Dict[str, object]] = []
	for i_sub, subj in enumerate(subjects):
		for j, n in enumerate(cell_counts):
			v_obs = float(observed_vals[i_sub, j])
			v_shuf = float(shuffled_vals[i_sub, j])
			if np.isfinite(v_obs) and np.isfinite(v_shuf):
				rows.append({"subject": subj, "source": "Observed", "n": str(int(n)), "value": v_obs})
				rows.append({"subject": subj, "source": "Shuffled", "n": str(int(n)), "value": v_shuf})

	df = pd.DataFrame(rows)
	if df.empty:
		return omnibus

	grid = df.pivot_table(index='subject', columns=['source', 'n'], values='value', aggfunc='mean')
	all_ns = sorted({c[1] for c in grid.columns})
	good_ns: List[str] = []
	for n in all_ns:
		if (('Observed', n) in grid.columns) and (('Shuffled', n) in grid.columns):
			col = grid[[('Observed', n), ('Shuffled', n)]]
			if not col.isna().any().any():
				good_ns.append(n)
	if len(good_ns) < 2:
		return omnibus

	df = df[df['n'].isin(good_ns)].copy()
	grid2 = df.pivot_table(index='subject', columns=['source', 'n'], values='value', aggfunc='mean')
	grid2 = grid2.dropna(axis=0, how='any')
	if grid2.shape[0] < 2:
		return omnibus

	balanced_rows: List[Dict[str, object]] = []
	for subj in grid2.index:
		for source in ['Observed', 'Shuffled']:
			for n in good_ns:
				balanced_rows.append({
					"subject": subj,
					"source": source,
					"n": n,
					"value": float(grid2.loc[subj, (source, n)]),
				})
	df_bal = pd.DataFrame(balanced_rows)
	try:
		df_bal['value_centered'] = df_bal['value'] - df_bal.groupby('subject')['value'].transform('mean')
		df_bal['source'] = df_bal['source'].astype('category')
		df_bal['n'] = df_bal['n'].astype('category')
		art = _aligned_rank_transform(
			df_bal,
			factor_a='source',
			factor_b='n',
			subject='subject',
			value_col='value_centered',
			include_interaction=ART_INCLUDE_INTERACTION_TERMS,
		)
		df_bal['rank_A'] = art['A']
		df_bal['rank_B'] = art['B']
		p_int = np.nan
		if ART_INCLUDE_INTERACTION_TERMS:
			p_source = AnovaRM(df_bal, depvar='rank_A', subject='subject', within=['source', 'n']).fit().anova_table.loc['source', 'Pr > F']
			p_n = AnovaRM(df_bal, depvar='rank_B', subject='subject', within=['source', 'n']).fit().anova_table.loc['n', 'Pr > F']
			if 'AxB' in art:
				df_bal['rank_Inter'] = art['AxB']
				p_int = AnovaRM(df_bal, depvar='rank_Inter', subject='subject', within=['source', 'n']).fit().anova_table.loc['source:n', 'Pr > F']
		else:
			df_a = df_bal.groupby(['subject', 'source'], as_index=False)['rank_A'].mean()
			p_source = AnovaRM(df_a, depvar='rank_A', subject='subject', within=['source']).fit().anova_table.loc['source', 'Pr > F']
			df_b = df_bal.groupby(['subject', 'n'], as_index=False)['rank_B'].mean()
			p_n = AnovaRM(df_b, depvar='rank_B', subject='subject', within=['n']).fit().anova_table.loc['n', 'Pr > F']
		omnibus['art_source_p'] = float(p_source) if np.isfinite(p_source) else None
		omnibus['art_n_p'] = float(p_n) if np.isfinite(p_n) else None
		omnibus['art_inter_p'] = float(p_int) if np.isfinite(p_int) else None

		art_raov2 = _art_raov_2way_pvalues_from_rows(
			balanced_rows,
			subject='subject',
			factor_a='source',
			factor_b='n',
			value_col='value',
			include_interactions=ART_INCLUDE_INTERACTION_TERMS,
		)
		art_rlmer2 = _art_r_2way_pvalues_from_rows(
			balanced_rows,
			subject='subject',
			factor_a='source',
			factor_b='n',
			value_col='value',
			include_interactions=ART_INCLUDE_INTERACTION_TERMS,
		)
		omnibus['art_raov_source_p'] = art_raov2.get('a')
		omnibus['art_raov_n_p'] = art_raov2.get('b')
		omnibus['art_raov_inter_p'] = art_raov2.get('a:b')
		omnibus['art_rlmer_source_p'] = art_rlmer2.get('a')
		omnibus['art_rlmer_n_p'] = art_rlmer2.get('b')
		omnibus['art_rlmer_inter_p'] = art_rlmer2.get('a:b')
	except Exception as exc:
		print(f"[debug] Observed-vs-shuffled ART×N omnibus failed: {type(exc).__name__}")

	return omnibus


def plot_sf_decoding_accuracy_vs_cell_number() -> None:
	"""Plot continuous SF-decoding correlation (avg across orientations) vs neuron count."""
	cell_counts = [int(n) for n in DECODING_CELL_COUNTS]
	chance_cache = _load_or_compute_rescued_pre_shuffled_null_cache()
	with warnings.catch_warnings():
		warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
		chance_samples_cellcount = np.nanmean(np.asarray(chance_cache['sf_by_orientation'], dtype=float), axis=2)

	# For 3-way omnibus (Training×N×Orientation)
	rows_3way_by_group: Dict[str, List[Dict[str, object]]] = {}

	# Collect paired mouse data for WT and RESCUED
	data_by_group: Dict[str, Dict[str, object]] = {}
	for group in ["WT", "RESCUED"]:
		pairs = MOUSE_PAIRS.get(group, [])
		if not pairs:
			continue
		rows_3way: List[Dict[str, object]] = []
		subjects: List[str] = []
		vals_pre = np.full((len(pairs), len(cell_counts)), np.nan, dtype=float)
		vals_post = np.full((len(pairs), len(cell_counts)), np.nan, dtype=float)
		vals_shuf_pre = np.full((len(pairs), len(cell_counts)), np.nan, dtype=float)
		for i, (s_pre, s_post) in enumerate(pairs):
			summary_pre = prepare_session_summaries(s_pre)
			summary_post = prepare_session_summaries(s_post)
			subj = _mouse_id_from_session_rel(s_pre)
			subjects.append(subj)
			null_mean_by_n_ori = None
			if group == "RESCUED":
				session_null = _load_or_compute_session_shuffled_null_cache(s_pre)
				with warnings.catch_warnings():
					warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
					null_mean_by_n_ori = np.nanmean(np.asarray(session_null['sf_by_orientation'], dtype=float), axis=0)
			for j, n in enumerate(cell_counts):
				v_pre = _get_metric_for_n(summary_pre, 'sf_decoding_by_ori', n)
				v_post = _get_metric_for_n(summary_post, 'sf_decoding_by_ori', n)
				if v_pre is None or v_post is None:
					continue
				v_pre_arr = np.asarray(v_pre, dtype=float).ravel()
				v_post_arr = np.asarray(v_post, dtype=float).ravel()
				vals_pre[i, j] = float(np.nanmean(v_pre_arr))
				vals_post[i, j] = float(np.nanmean(v_post_arr))
				if group == "RESCUED" and null_mean_by_n_ori is not None and j < null_mean_by_n_ori.shape[0]:
					with warnings.catch_warnings():
						warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
						vals_shuf_pre[i, j] = float(np.nanmean(np.asarray(null_mean_by_n_ori[j], dtype=float).ravel()))

				# 3-way rows: Training×N×Orientation
				# Use index-based orientation bins to ensure consistent factor levels across mice.
				min_len = int(min(v_pre_arr.size, v_post_arr.size))
				if min_len < 2:
					continue
				for k in range(min_len):
					b = float(v_pre_arr[k])
					a = float(v_post_arr[k])
					if np.isfinite(b) and np.isfinite(a):
						rows_3way.append({"subject": subj, "training": "Pre", "n": str(int(n)), "ori": str(int(k)), "value": b})
						rows_3way.append({"subject": subj, "training": "Post", "n": str(int(n)), "ori": str(int(k)), "value": a})
		data_by_group[group] = {"subjects": subjects, "pre": vals_pre, "post": vals_post, "shuf_pre": vals_shuf_pre}
		rows_3way_by_group[group] = rows_3way

	# Wider canvas with reserved right margin for stats panel
	fig, ax = plt.subplots(figsize=(11.5, 5.6))
	low, high = _decoder_null_band(chance_samples_cellcount)
	ax.fill_between(cell_counts, low, high, color='#d9d9d9', alpha=1.0, zorder=0)

	# Plot group means ± SEM (Pre and Post)
	for group, group_info in data_by_group.items():
		vals_pre = np.asarray(group_info["pre"], dtype=float)
		vals_post = np.asarray(group_info["post"], dtype=float)
		m_pre = np.nanmean(vals_pre, axis=0)
		m_post = np.nanmean(vals_post, axis=0)
		n_pre = np.sum(np.isfinite(vals_pre), axis=0)
		n_post = np.sum(np.isfinite(vals_post), axis=0)
		sem_pre = np.nanstd(vals_pre, axis=0, ddof=1) / np.sqrt(np.maximum(n_pre, 1))
		sem_post = np.nanstd(vals_post, axis=0, ddof=1) / np.sqrt(np.maximum(n_post, 1))

		if group == "WT":
			c_pre = GROUP_COLORS["WT_BEFORE"]
			c_post = GROUP_COLORS["WT_AFTER"]
			lab_pre = "WT pre-training"
			lab_post = "WT post-training"
		else:
			c_pre = GROUP_COLORS["RESCUED_BEFORE"]
			c_post = GROUP_COLORS["RESCUED_AFTER"]
			lab_pre = "Rescued pre-training"
			lab_post = "Rescued post-training"

		ax.errorbar(cell_counts, m_pre, yerr=sem_pre, fmt='-o', capsize=3, color=c_pre, label=lab_pre)
		ax.errorbar(cell_counts, m_post, yerr=sem_post, fmt='-o', capsize=3, color=c_post, label=lab_post)

	ax.axhline(y=0.0, linestyle='--', color='red', linewidth=1.0, label='Null correlation')

	ax.set_xlabel('Number of neurons used for decoding')
	ax.set_ylabel('SF decoding Pearson r (avg across orientations)')
	ax.set_ylim(-1, 1)
	ax.set_title('Continuous SF decoding correlation vs neuron count')
	ax.legend(loc='lower right')
	# Leave a generous right margin so the stats panel never overlaps the axes.
	# (This particular panel is tall, so reserve extra space.)
	fig.tight_layout(rect=[0.0, 0.0, 0.56, 1.0])

	# Stats panel
	def _fmt_opt(v: Optional[float]) -> str:
		if v is None:
			return "NA"
		try:
			vf = float(v)
		except Exception:
			return "NA"
		if not np.isfinite(vf):
			return "NA"
		return f"{vf:.3g}"

	lines: List[str] = []
	for group, group_info in data_by_group.items():
		subjects = list(group_info["subjects"])
		vals_pre = np.asarray(group_info["pre"], dtype=float)
		vals_post = np.asarray(group_info["post"], dtype=float)
		vals_shuf_pre = np.asarray(group_info.get("shuf_pre"), dtype=float)
		# Omnibus 3-way (Training×N×Orientation)
		rm3, art3 = _compute_rm_art_3way_from_rows(
			rows_3way_by_group.get(group, []),
			subject='subject',
			factor_a='training',
			factor_b='n',
			factor_c='ori',
			value_col='value',
			center_by_subject=True,
		)
		art3_raov = _art_raov_3way_pvalues_from_rows(
			rows_3way_by_group.get(group, []),
			subject='subject',
			factor_a='training',
			factor_b='n',
			factor_c='ori',
			value_col='value',
			include_interactions=ART_INCLUDE_INTERACTION_TERMS,
		)
		art3_rlmer = _art_r_3way_pvalues_from_rows(
			rows_3way_by_group.get(group, []),
			subject='subject',
			factor_a='training',
			factor_b='n',
			factor_c='ori',
			value_col='value',
			include_interactions=ART_INCLUDE_INTERACTION_TERMS,
		)
		lme3 = _lme_3way_lr_pvalues_from_rows(
			rows_3way_by_group.get(group, []),
			subject='subject',
			factor_a='training',
			factor_b='n',
			factor_c='ori',
			value_col='value',
			include_interactions=LME_INCLUDE_INTERACTION_TERMS,
		)
		lme4_3 = _lme4_3way_lrt_pvalues_from_rows(
			rows_3way_by_group.get(group, []),
			subject='subject',
			factor_a='training',
			factor_b='n',
			factor_c='ori',
			value_col='value',
			include_interactions=LME_INCLUDE_INTERACTION_TERMS,
		)
		lines.append(f"{group} RM3 (t,n,ori): t={_fmt_opt(rm3['training'])} n={_fmt_opt(rm3['n'])} ori={_fmt_opt(rm3['ori'])}")
		lines.append(
			f"{group} RM3 ints: t:n={_fmt_opt(rm3['training:n'])} t:ori={_fmt_opt(rm3['training:ori'])} n:ori={_fmt_opt(rm3['n:ori'])} t:n:ori={_fmt_opt(rm3['training:n:ori'])}"
		)
		lines.append(f"{group} ART3 (t,n,ori): t={_fmt_opt(art3['training'])} n={_fmt_opt(art3['n'])} ori={_fmt_opt(art3['ori'])}")
		lines.append(
			f"{group} ART3 ints: t:n={_fmt_opt(art3['training:n'])} t:ori={_fmt_opt(art3['training:ori'])} n:ori={_fmt_opt(art3['n:ori'])} t:n:ori={_fmt_opt(art3['training:n:ori'])}"
		)
		lines.append(f"{group} ART3-Raov (t,n,ori): t={_fmt_opt(art3_raov['training'])} n={_fmt_opt(art3_raov['n'])} ori={_fmt_opt(art3_raov['ori'])}")
		lines.append(
			f"{group} ART3-Raov ints: t:n={_fmt_opt(art3_raov['training:n'])} t:ori={_fmt_opt(art3_raov['training:ori'])} n:ori={_fmt_opt(art3_raov['n:ori'])} t:n:ori={_fmt_opt(art3_raov['training:n:ori'])}"
		)
		lines.append(f"{group} ART3-Rlmer (t,n,ori): t={_fmt_opt(art3_rlmer['training'])} n={_fmt_opt(art3_rlmer['n'])} ori={_fmt_opt(art3_rlmer['ori'])}")
		lines.append(
			f"{group} ART3-Rlmer ints: t:n={_fmt_opt(art3_rlmer['training:n'])} t:ori={_fmt_opt(art3_rlmer['training:ori'])} n:ori={_fmt_opt(art3_rlmer['n:ori'])} t:n:ori={_fmt_opt(art3_rlmer['training:n:ori'])}"
		)
		lines.append(f"{group} LME3 (t,n,ori): t={_fmt_opt(lme3['training'])} n={_fmt_opt(lme3['n'])} ori={_fmt_opt(lme3['ori'])}")
		lines.append(
			f"{group} LME3 ints: t:n={_fmt_opt(lme3['training:n'])} t:ori={_fmt_opt(lme3['training:ori'])} n:ori={_fmt_opt(lme3['n:ori'])} t:n:ori={_fmt_opt(lme3['training:n:ori'])}"
		)
		lines.append(f"{group} LME3-lme4 (t,n,ori): t={_fmt_opt(lme4_3['training'])} n={_fmt_opt(lme4_3['n'])} ori={_fmt_opt(lme4_3['ori'])}")
		lines.append(
			f"{group} LME3-lme4 ints: t:n={_fmt_opt(lme4_3['training:n'])} t:ori={_fmt_opt(lme4_3['training:ori'])} n:ori={_fmt_opt(lme4_3['n:ori'])} t:n:ori={_fmt_opt(lme4_3['training:n:ori'])}"
		)
		omni, p_t, q_t, p_w, q_w = _compute_training_x_n_stats(
			group=group,
			subjects=subjects,
			vals_pre=vals_pre,
			vals_post=vals_post,
			cell_counts=cell_counts,
		)
		lines.append(f"{group} RM: train p={_fmt_opt(omni['rm_train_p'])} n p={_fmt_opt(omni['rm_n_p'])} int p={_fmt_opt(omni['rm_inter_p'])}")
		lines.append(f"{group} ART: train p={_fmt_opt(omni['art_train_p'])} n p={_fmt_opt(omni['art_n_p'])} int p={_fmt_opt(omni['art_inter_p'])}")
		lines.append(f"{group} ART-Raov: train p={_fmt_opt(omni.get('art_raov_train_p'))} n p={_fmt_opt(omni.get('art_raov_n_p'))} int p={_fmt_opt(omni.get('art_raov_inter_p'))}")
		lines.append(f"{group} ART-Rlmer: train p={_fmt_opt(omni.get('art_rlmer_train_p'))} n p={_fmt_opt(omni.get('art_rlmer_n_p'))} int p={_fmt_opt(omni.get('art_rlmer_inter_p'))}")
		lines.append(f"{group} LME: train p={_fmt_opt(omni['lme_train_p'])} n p={_fmt_opt(omni['lme_n_p'])} int p={_fmt_opt(omni['lme_inter_p'])}")
		lines.append(f"{group} LME-lme4: train p={_fmt_opt(omni['lme4_train_p'])} n p={_fmt_opt(omni['lme4_n_p'])} int p={_fmt_opt(omni['lme4_inter_p'])}")
		if group == "RESCUED" and vals_shuf_pre.size:
			omni_shuf = _compute_observed_vs_shuffled_x_n_art_stats(
				subjects=subjects,
				observed_vals=vals_pre,
				shuffled_vals=vals_shuf_pre,
				cell_counts=cell_counts,
			)
			lines.append(
				"ADDED Resc pre vs shuf: "
				f"ART src/n/i={_fmt_opt(omni_shuf['art_source_p'])}/{_fmt_opt(omni_shuf['art_n_p'])}/{_fmt_opt(omni_shuf['art_inter_p'])} | "
				f"ART-Raov src/n/i={_fmt_opt(omni_shuf['art_raov_source_p'])}/{_fmt_opt(omni_shuf['art_raov_n_p'])}/{_fmt_opt(omni_shuf['art_raov_inter_p'])} | "
				f"ART-Rlmer src/n/i={_fmt_opt(omni_shuf['art_rlmer_source_p'])}/{_fmt_opt(omni_shuf['art_rlmer_n_p'])}/{_fmt_opt(omni_shuf['art_rlmer_inter_p'])}"
			)
		for n in cell_counts:
			pt = p_t.get(int(n), None)
			qt = q_t.get(int(n), None)
			pw = p_w.get(int(n), None)
			qw = q_w.get(int(n), None)
			lines.append(
				f"{group} N={int(n)}: t p={pt if pt is not None else np.nan:.3g} q={qt if qt is not None else np.nan:.3g} | W p={pw if pw is not None else np.nan:.3g} q={qw if qw is not None else np.nan:.3g}"
			)
	lines.append("Above shuffled chance@N: sig q<0.05")
	for group_name, group_info in data_by_group.items():
		for line_label, mean_vals in [
			(f"{group_name} pre", np.nanmean(np.asarray(group_info["pre"], dtype=float), axis=0)),
			(f"{group_name} post", np.nanmean(np.asarray(group_info["post"], dtype=float), axis=0)),
		]:
			p_emp = _empirical_above_chance_pvalues(np.asarray(mean_vals, dtype=float), chance_samples_cellcount)
			q_emp = _bh_fdr_correct(p_emp)
			lines.append(f"{line_label}: {_format_sig_cellcounts_from_q(cell_counts, q_emp)}")
		lines.append("-")
	if lines and lines[-1] == "-":
		lines = lines[:-1]

	panel_lines_wrapped = _wrap_stats_panel_lines(lines, max_chars=118)
	fig_h_in = float(fig.get_size_inches()[1])
	available_pts = fig_h_in * 72.0 * 0.84
	fontsize = 5.2
	per_line_pts = max(1.0, fontsize * 2.2)
	max_lines_per_col = max(12, int(available_pts / per_line_pts))
	n_cols = int(np.ceil(len(panel_lines_wrapped) / float(max_lines_per_col)))
	n_cols = int(min(max(n_cols, 1), 4))
	line_sep = 0.25
	if n_cols >= 3:
		fontsize = 4.8
		line_sep = 0.1
	if n_cols >= 4:
		fontsize = 4.5
		line_sep = 0.0
	_add_colored_stats_panel_to_figure(
		fig,
		panel_lines_wrapped,
		x=0.995,
		y=0.99,
		fontsize=fontsize,
		n_cols=n_cols,
		line_sep=line_sep,
	)
	outfile = os.path.join(GROUPED_PLOTS_DIR, 'sf_decoding_accuracy_vs_cell_number_groups.svg')
	# Avoid bbox_inches='tight' here; it can collapse the reserved right margin.
	fig.savefig(outfile, format='svg')
	plt.close(fig)
	print(f"  [saved] {os.path.basename(outfile)}")


def plot_orientation_decoding_accuracy_vs_cell_number_mean_non240_sf() -> None:
	"""Plot orientation-decoding correlation vs neuron count averaged across non-240 SFs."""
	cell_counts = [int(n) for n in DECODING_CELL_COUNTS]
	nonzero_sf_indices = [i for i in range(1, len(TARGET_SF_VALUES))]
	combined_sf_indices = nonzero_sf_indices[:-1] if len(nonzero_sf_indices) > 1 else nonzero_sf_indices
	if not combined_sf_indices:
		raise ValueError("No nonzero SF indices available for combined orientation-decoding summary.")

	combined_sf_values = [float(TARGET_SF_VALUES[i]) for i in combined_sf_indices]
	combined_sf_label = ", ".join(f"{sf:.4g}" for sf in combined_sf_values)

	chance_cache = _load_or_compute_rescued_pre_shuffled_null_cache()
	with warnings.catch_warnings():
		warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
		chance_samples_cellcount = np.nanmean(
			np.asarray(chance_cache['orientation_by_sf'], dtype=float)[:, :, combined_sf_indices],
			axis=2,
		)

	data_by_group: Dict[str, Dict[str, object]] = {}
	for group in ["WT", "RESCUED"]:
		pairs = MOUSE_PAIRS.get(group, [])
		if not pairs:
			continue
		subjects: List[str] = []
		vals_pre = np.full((len(pairs), len(cell_counts)), np.nan, dtype=float)
		vals_post = np.full((len(pairs), len(cell_counts)), np.nan, dtype=float)
		vals_shuf_pre = np.full((len(pairs), len(cell_counts)), np.nan, dtype=float)
		for i, (s_pre, s_post) in enumerate(pairs):
			summary_pre = prepare_session_summaries(s_pre)
			summary_post = prepare_session_summaries(s_post)
			subj = _mouse_id_from_session_rel(s_pre)
			subjects.append(subj)
			null_mean_by_count_sf = None
			if group == "RESCUED":
				session_null = _load_or_compute_session_shuffled_null_cache(s_pre)
				with warnings.catch_warnings():
					warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
					null_mean_by_count_sf = np.nanmean(np.asarray(session_null['orientation_by_sf'], dtype=float), axis=0)
			for j, n in enumerate(cell_counts):
				v_pre = _get_metric_for_n(summary_pre, 'decoding_accuracy', n)
				v_post = _get_metric_for_n(summary_post, 'decoding_accuracy', n)
				if v_pre is None or v_post is None:
					continue
				v_pre_arr = np.asarray(v_pre, dtype=float).ravel()
				v_post_arr = np.asarray(v_post, dtype=float).ravel()
				valid_sf = [sf_idx for sf_idx in combined_sf_indices if sf_idx < v_pre_arr.size and sf_idx < v_post_arr.size]
				if not valid_sf:
					continue
				vals_pre[i, j] = float(np.nanmean(v_pre_arr[valid_sf]))
				vals_post[i, j] = float(np.nanmean(v_post_arr[valid_sf]))
				if group == "RESCUED" and null_mean_by_count_sf is not None and j < null_mean_by_count_sf.shape[0]:
					valid_null_sf = [sf_idx for sf_idx in combined_sf_indices if sf_idx < null_mean_by_count_sf.shape[1]]
					if valid_null_sf:
						with warnings.catch_warnings():
							warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
							vals_shuf_pre[i, j] = float(np.nanmean(np.asarray(null_mean_by_count_sf[j, valid_null_sf], dtype=float)))
		data_by_group[group] = {
			"subjects": subjects,
			"pre": vals_pre,
			"post": vals_post,
			"shuf_pre": vals_shuf_pre,
		}

	fig, ax = plt.subplots(figsize=(11.5, 5.6))
	low, high = _decoder_null_band(chance_samples_cellcount)
	ax.fill_between(cell_counts, low, high, color='#d9d9d9', alpha=1.0, zorder=0)

	for group, group_info in data_by_group.items():
		vals_pre = np.asarray(group_info["pre"], dtype=float)
		vals_post = np.asarray(group_info["post"], dtype=float)
		m_pre = np.nanmean(vals_pre, axis=0)
		m_post = np.nanmean(vals_post, axis=0)
		n_pre = np.sum(np.isfinite(vals_pre), axis=0)
		n_post = np.sum(np.isfinite(vals_post), axis=0)
		sem_pre = np.nanstd(vals_pre, axis=0, ddof=1) / np.sqrt(np.maximum(n_pre, 1))
		sem_post = np.nanstd(vals_post, axis=0, ddof=1) / np.sqrt(np.maximum(n_post, 1))

		if group == "WT":
			c_pre = GROUP_COLORS["WT_BEFORE"]
			c_post = GROUP_COLORS["WT_AFTER"]
			lab_pre = "WT pre-training"
			lab_post = "WT post-training"
		else:
			c_pre = GROUP_COLORS["RESCUED_BEFORE"]
			c_post = GROUP_COLORS["RESCUED_AFTER"]
			lab_pre = "Rescued pre-training"
			lab_post = "Rescued post-training"

		ax.errorbar(cell_counts, m_pre, yerr=sem_pre, fmt='-o', capsize=3, color=c_pre, label=lab_pre)
		ax.errorbar(cell_counts, m_post, yerr=sem_post, fmt='-o', capsize=3, color=c_post, label=lab_post)

	ax.axhline(y=0.0, linestyle='--', color='red', linewidth=1.0, label='Null correlation')
	ax.set_xlabel('Number of neurons used for decoding')
	ax.set_ylabel('Circular correlation')
	ax.set_ylim(-1, 1)
	ax.set_title('Continuous orientation decoding correlation vs neuron count | mean across SFs excluding 240')
	ax.legend(loc='lower right')
	fig.tight_layout(rect=[0.0, 0.0, 0.56, 1.0])

	def _fmt_opt(v: Optional[float]) -> str:
		if v is None:
			return "NA"
		try:
			vf = float(v)
		except Exception:
			return "NA"
		if not np.isfinite(vf):
			return "NA"
		return f"{vf:.3g}"

	lines: List[str] = [f"Mean SFs (exclude 240): {combined_sf_label}"]
	for group, group_info in data_by_group.items():
		subjects = list(group_info["subjects"])
		vals_pre = np.asarray(group_info["pre"], dtype=float)
		vals_post = np.asarray(group_info["post"], dtype=float)
		vals_shuf_pre = np.asarray(group_info.get("shuf_pre"), dtype=float)
		omni, p_t, q_t, p_w, q_w = _compute_training_x_n_stats(
			group=group,
			subjects=subjects,
			vals_pre=vals_pre,
			vals_post=vals_post,
			cell_counts=cell_counts,
		)
		lines.append(f"{group} RM: train p={_fmt_opt(omni['rm_train_p'])} n p={_fmt_opt(omni['rm_n_p'])} int p={_fmt_opt(omni['rm_inter_p'])}")
		lines.append(f"{group} ART: train p={_fmt_opt(omni['art_train_p'])} n p={_fmt_opt(omni['art_n_p'])} int p={_fmt_opt(omni['art_inter_p'])}")
		lines.append(f"{group} ART-Raov: train p={_fmt_opt(omni.get('art_raov_train_p'))} n p={_fmt_opt(omni.get('art_raov_n_p'))} int p={_fmt_opt(omni.get('art_raov_inter_p'))}")
		lines.append(f"{group} ART-Rlmer: train p={_fmt_opt(omni.get('art_rlmer_train_p'))} n p={_fmt_opt(omni.get('art_rlmer_n_p'))} int p={_fmt_opt(omni.get('art_rlmer_inter_p'))}")
		lines.append(f"{group} LME: train p={_fmt_opt(omni['lme_train_p'])} n p={_fmt_opt(omni['lme_n_p'])} int p={_fmt_opt(omni['lme_inter_p'])}")
		lines.append(f"{group} LME-lme4: train p={_fmt_opt(omni['lme4_train_p'])} n p={_fmt_opt(omni['lme4_n_p'])} int p={_fmt_opt(omni['lme4_inter_p'])}")
		if group == "RESCUED" and vals_shuf_pre.size:
			omni_shuf = _compute_observed_vs_shuffled_x_n_art_stats(
				subjects=subjects,
				observed_vals=vals_pre,
				shuffled_vals=vals_shuf_pre,
				cell_counts=cell_counts,
			)
			lines.append(
				"ADDED Resc pre vs shuf "
				f"ART src/n/i={_fmt_opt(omni_shuf['art_source_p'])}/{_fmt_opt(omni_shuf['art_n_p'])}/{_fmt_opt(omni_shuf['art_inter_p'])}"
			)
			lines.append(
				"ADDED Resc pre vs shuf "
				f"ART-Raov src/n/i={_fmt_opt(omni_shuf['art_raov_source_p'])}/{_fmt_opt(omni_shuf['art_raov_n_p'])}/{_fmt_opt(omni_shuf['art_raov_inter_p'])}"
			)
			lines.append(
				"ADDED Resc pre vs shuf "
				f"ART-Rlmer src/n/i={_fmt_opt(omni_shuf['art_rlmer_source_p'])}/{_fmt_opt(omni_shuf['art_rlmer_n_p'])}/{_fmt_opt(omni_shuf['art_rlmer_inter_p'])}"
			)
		for n in cell_counts:
			pt = p_t.get(int(n), None)
			qt = q_t.get(int(n), None)
			pw = p_w.get(int(n), None)
			qw = q_w.get(int(n), None)
			lines.append(
				f"{group} N={int(n)}: t p={pt if pt is not None else np.nan:.3g} q={qt if qt is not None else np.nan:.3g} | W p={pw if pw is not None else np.nan:.3g} q={qw if qw is not None else np.nan:.3g}"
			)
		lines.append("-")

	lines.append("Above shuffled chance@N: sig q<0.05")
	for group_name, group_info in data_by_group.items():
		for line_label, mean_vals in [
			(f"{group_name} pre", np.nanmean(np.asarray(group_info["pre"], dtype=float), axis=0)),
			(f"{group_name} post", np.nanmean(np.asarray(group_info["post"], dtype=float), axis=0)),
		]:
			p_emp = _empirical_above_chance_pvalues(np.asarray(mean_vals, dtype=float), chance_samples_cellcount)
			q_emp = _bh_fdr_correct(p_emp)
			lines.append(f"{line_label}: {_format_sig_cellcounts_from_q(cell_counts, q_emp)}")
		lines.append("-")
	if lines and lines[-1] == "-":
		lines = lines[:-1]

	panel_lines_wrapped = _wrap_stats_panel_lines(lines, max_chars=118)
	fig_h_in = float(fig.get_size_inches()[1])
	available_pts = fig_h_in * 72.0 * 0.84
	fontsize = 5.2
	per_line_pts = max(1.0, fontsize * 2.2)
	max_lines_per_col = max(12, int(available_pts / per_line_pts))
	n_cols = int(np.ceil(len(panel_lines_wrapped) / float(max_lines_per_col)))
	n_cols = int(min(max(n_cols, 1), 4))
	line_sep = 0.25
	if n_cols >= 3:
		fontsize = 4.8
		line_sep = 0.1
	if n_cols >= 4:
		fontsize = 4.5
		line_sep = 0.0
	_add_colored_stats_panel_to_figure(
		fig,
		panel_lines_wrapped,
		x=0.995,
		y=0.99,
		fontsize=fontsize,
		n_cols=n_cols,
		line_sep=line_sep,
	)
	outfile = os.path.join(GROUPED_PLOTS_DIR, 'orientation_decoding_accuracy_vs_cell_number_mean_non240_sf_groups.svg')
	fig.savefig(outfile, format='svg')
	plt.close(fig)
	print(f"  [saved] {os.path.basename(outfile)}")


def _plot_metric_vs_cell_number_per_sf(
	*,
	metric_base: str,
	title: str,
	ylabel: str,
	outfile: str,
	y_limits: Optional[Tuple[float, float]] = None,
	blind_as_individual: bool = False,
	include_blind: bool = True,
	chance_null_metric_base: Optional[str] = None,
) -> None:
	"""Create a 6-subplot figure (non-zero SFs) of metric vs neuron count.

	Stats are summarized in a single right-side panel (like other grouped plots).
	"""
	cell_counts = [int(n) for n in DECODING_CELL_COUNTS]
	# Nonzero SF indices (skip baseline SF=0)
	sf_indices = list(range(1, min(len(TARGET_SF_VALUES), 7)))
	if len(sf_indices) != 6:
		sf_indices = [i for i in sf_indices if i != 0][:6]

	# Larger canvas with reserved right margin for a global stats panel.
	# Wider figures reduce stats-panel wrapping (shorter, less likely to clip).
	fig, axes = plt.subplots(2, 3, figsize=(22.5, 9.2), sharey=True)
	axes = axes.ravel()
	panel_lines: List[str] = []
	rows_3way_by_group: Dict[str, List[Dict[str, object]]] = {"WT": [], "RESCUED": []}
	chance_samples_full = None
	if chance_null_metric_base == 'decoding_accuracy':
		chance_cache = _load_or_compute_rescued_pre_shuffled_null_cache()
		chance_samples_full = np.asarray(chance_cache['orientation_by_sf'], dtype=float)

	def _fmt_opt(v: Optional[float]) -> str:
		if v is None:
			return "NA"
		try:
			vf = float(v)
		except Exception:
			return "NA"
		if not np.isfinite(vf):
			return "NA"
		return f"{vf:.3g}"

	for ax_i, sf_idx in enumerate(sf_indices):
		ax = axes[ax_i]
		sf_val = float(TARGET_SF_VALUES[sf_idx])
		ax.set_title(f"SF={sf_val:.4g}")
		ax.set_xlabel('N neurons')
		ax.set_ylabel(ylabel)
		if y_limits is not None:
			ax.set_ylim(*y_limits)
			if y_limits[0] < 0 < y_limits[1]:
				ax.axhline(y=0.0, linestyle='--', color='red', linewidth=1.0, alpha=0.8)
		else:
			ax.set_ylim(bottom=0)
		if chance_samples_full is not None:
			low, high = _decoder_null_band(chance_samples_full[:, :, sf_idx])
			ax.fill_between(
				cell_counts,
				low,
				high,
				color='#d9d9d9',
				alpha=1.0,
				zorder=0,
			)

		# BLIND: plot individual animals or mean ± SEM (unpaired)
		blind_curves = _collect_blind_curves_over_cell_counts(metric_base, sf_idx, cell_counts) if include_blind else []

		if blind_curves:
			if blind_as_individual:
				blind_color = GROUP_COLORS.get('BLIND', '#7f7f7f')
				blind_label = GROUP_LABELS.get('BLIND', 'Blind')
				for i_curve, y_blind in enumerate(blind_curves):
					ax.plot(
						cell_counts,
						y_blind,
						'-',
						color=blind_color,
						linewidth=1.0,
						zorder=1,
						label=blind_label if ax_i == 0 and i_curve == 0 else None,
					)
			else:
				blind_stack = np.vstack(blind_curves)
				m_blind, sem_blind = nan_mean_sem(blind_stack, axis=0)
				ax.errorbar(
					cell_counts,
					m_blind,
					yerr=sem_blind,
					fmt='-o',
					capsize=2,
					color=GROUP_COLORS.get('BLIND', '#7f7f7f'),
					alpha=0.75,
					linewidth=1.5,
					zorder=1,
				)

		# Collect paired data per group for this SF
		stats_by_group: Dict[str, Dict[str, object]] = {}
		for group in ["WT", "RESCUED"]:
			pairs = MOUSE_PAIRS.get(group, [])
			if not pairs:
				continue
			subjects: List[str] = []
			vals_pre = np.full((len(pairs), len(cell_counts)), np.nan, dtype=float)
			vals_post = np.full((len(pairs), len(cell_counts)), np.nan, dtype=float)
			for i, (s_pre, s_post) in enumerate(pairs):
				summary_pre = prepare_session_summaries(s_pre)
				summary_post = prepare_session_summaries(s_post)
				subj = _mouse_id_from_session_rel(s_pre)
				subjects.append(subj)
				for j, n in enumerate(cell_counts):
					v_pre = _get_metric_for_n(summary_pre, metric_base, n)
					v_post = _get_metric_for_n(summary_post, metric_base, n)
					if v_pre is None or v_post is None:
						continue
					v_pre = np.asarray(v_pre, dtype=float)
					v_post = np.asarray(v_post, dtype=float)
					if v_pre.ndim != 1 or v_post.ndim != 1:
						continue
					if sf_idx >= v_pre.size or sf_idx >= v_post.size:
						continue
					b = float(v_pre[sf_idx])
					a = float(v_post[sf_idx])
					vals_pre[i, j] = b
					vals_post[i, j] = a
					# Accumulate global 3-way rows: Training×N×SF
					if np.isfinite(b) and np.isfinite(a):
						rows_3way_by_group.setdefault(group, []).append({
							"subject": subj,
							"training": "Pre",
							"n": str(int(n)),
							"sf": str(float(TARGET_SF_VALUES[sf_idx])),
							"value": b,
						})
						rows_3way_by_group.setdefault(group, []).append({
							"subject": subj,
							"training": "Post",
							"n": str(int(n)),
							"sf": str(float(TARGET_SF_VALUES[sf_idx])),
							"value": a,
						})

			# Plot mean ± SEM
			m_pre = np.nanmean(vals_pre, axis=0)
			m_post = np.nanmean(vals_post, axis=0)
			n_pre = np.sum(np.isfinite(vals_pre), axis=0)
			n_post = np.sum(np.isfinite(vals_post), axis=0)
			sem_pre = np.nanstd(vals_pre, axis=0, ddof=1) / np.sqrt(np.maximum(n_pre, 1))
			sem_post = np.nanstd(vals_post, axis=0, ddof=1) / np.sqrt(np.maximum(n_post, 1))

			if group == "WT":
				c_pre = GROUP_COLORS["WT_BEFORE"]
				c_post = GROUP_COLORS["WT_AFTER"]
			else:
				c_pre = GROUP_COLORS["RESCUED_BEFORE"]
				c_post = GROUP_COLORS["RESCUED_AFTER"]

			ax.errorbar(cell_counts, m_pre, yerr=sem_pre, fmt='-o', capsize=2, color=c_pre)
			ax.errorbar(cell_counts, m_post, yerr=sem_post, fmt='-o', capsize=2, color=c_post)

			# Stats: omnibus + paired posthoc for every N (p and BH-FDR q)
			omni, p_t, q_t, p_w, q_w = _compute_training_x_n_stats(
				group=group,
				subjects=subjects,
				vals_pre=vals_pre,
				vals_post=vals_post,
				cell_counts=cell_counts,
			)
			stats_by_group[group] = {
				"omni": omni,
				"p_t": p_t,
				"q_t": q_t,
				"p_w": p_w,
				"q_w": q_w,
				"mean_pre": m_pre,
				"mean_post": m_post,
				"subjects": subjects,
				"vals_pre": vals_pre,
			}
			if group == "RESCUED":
				vals_shuf_pre = np.full((len(pairs), len(cell_counts)), np.nan, dtype=float)
				for i, (s_pre, _s_post) in enumerate(pairs):
					session_null = _load_or_compute_session_shuffled_null_cache(s_pre)
					with warnings.catch_warnings():
						warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
						null_mean_by_count_sf = np.nanmean(np.asarray(session_null['orientation_by_sf'], dtype=float), axis=0)
					if sf_idx < null_mean_by_count_sf.shape[1]:
						vals_shuf_pre[i, :] = np.asarray(null_mean_by_count_sf[:, sf_idx], dtype=float)
				stats_by_group[group]["shuf_pre"] = vals_shuf_pre

		# Add this SF block to the global right-side panel (compact table format)
		if stats_by_group:
			panel_lines.append(f"SF={sf_val:.4g}")
			for group in ["WT", "RESCUED"]:
				if group not in stats_by_group:
					continue
				omni = stats_by_group[group]["omni"]
				panel_lines.append(
					f"{group} RM t/n/i={_fmt_opt(omni['rm_train_p'])}/{_fmt_opt(omni['rm_n_p'])}/{_fmt_opt(omni['rm_inter_p'])} | ART t/n/i={_fmt_opt(omni['art_train_p'])}/{_fmt_opt(omni['art_n_p'])}/{_fmt_opt(omni['art_inter_p'])} | ART-Raov t/n/i={_fmt_opt(omni.get('art_raov_train_p'))}/{_fmt_opt(omni.get('art_raov_n_p'))}/{_fmt_opt(omni.get('art_raov_inter_p'))} | ART-Rlmer t/n/i={_fmt_opt(omni.get('art_rlmer_train_p'))}/{_fmt_opt(omni.get('art_rlmer_n_p'))}/{_fmt_opt(omni.get('art_rlmer_inter_p'))} | LME t/n/i={_fmt_opt(omni['lme_train_p'])}/{_fmt_opt(omni['lme_n_p'])}/{_fmt_opt(omni['lme_inter_p'])} | LME-lme4 t/n/i={_fmt_opt(omni['lme4_train_p'])}/{_fmt_opt(omni['lme4_n_p'])}/{_fmt_opt(omni['lme4_inter_p'])}"
				)
			# Per-N paired tests: one row per N, show both groups if available
			for n in cell_counts:
				parts: List[str] = []
				for group in ["WT", "RESCUED"]:
					if group not in stats_by_group:
						continue
					p_t = stats_by_group[group]["p_t"].get(int(n), None)
					q_t = stats_by_group[group]["q_t"].get(int(n), None)
					p_w = stats_by_group[group]["p_w"].get(int(n), None)
					q_w = stats_by_group[group]["q_w"].get(int(n), None)
					parts.append(
						f"{group} t p/q={p_t if p_t is not None else np.nan:.3g}/{q_t if q_t is not None else np.nan:.3g} W p/q={p_w if p_w is not None else np.nan:.3g}/{q_w if q_w is not None else np.nan:.3g}"
					)
				if parts:
					panel_lines.append(f"N={int(n)}: " + " | ".join(parts))
			if chance_samples_full is not None:
				chance_parts: List[str] = []
				for line_label, mean_vals in [
					('WT pre', stats_by_group.get('WT', {}).get('mean_pre')),
					('WT post', stats_by_group.get('WT', {}).get('mean_post')),
					('Resc pre', stats_by_group.get('RESCUED', {}).get('mean_pre')),
					('Resc post', stats_by_group.get('RESCUED', {}).get('mean_post')),
				]:
					if mean_vals is None:
						continue
					p_emp = _empirical_above_chance_pvalues(np.asarray(mean_vals, dtype=float), chance_samples_full[:, :, sf_idx])
					q_emp = _bh_fdr_correct(p_emp)
					chance_parts.append(f"{line_label}={_format_sig_cellcounts_from_q(cell_counts, q_emp)}")
				if chance_parts:
					panel_lines.append("Chance q<0.05: " + " | ".join(chance_parts))
			resc_info = stats_by_group.get('RESCUED', None)
			if resc_info is not None and resc_info.get('shuf_pre') is not None:
				omni_shuf = _compute_observed_vs_shuffled_x_n_art_stats(
					subjects=list(resc_info.get('subjects', [])),
					observed_vals=np.asarray(resc_info.get('vals_pre'), dtype=float),
					shuffled_vals=np.asarray(resc_info.get('shuf_pre'), dtype=float),
					cell_counts=cell_counts,
				)
				panel_lines.append(
					"ADDED Resc pre vs shuf "
					f"ART src/n/i={_fmt_opt(omni_shuf['art_source_p'])}/{_fmt_opt(omni_shuf['art_n_p'])}/{_fmt_opt(omni_shuf['art_inter_p'])}"
				)
				panel_lines.append(
					"ADDED Resc pre vs shuf "
					f"ART-Raov src/n/i={_fmt_opt(omni_shuf['art_raov_source_p'])}/{_fmt_opt(omni_shuf['art_raov_n_p'])}/{_fmt_opt(omni_shuf['art_raov_inter_p'])}"
				)
				panel_lines.append(
					"ADDED Resc pre vs shuf "
					f"ART-Rlmer src/n/i={_fmt_opt(omni_shuf['art_rlmer_source_p'])}/{_fmt_opt(omni_shuf['art_rlmer_n_p'])}/{_fmt_opt(omni_shuf['art_rlmer_inter_p'])}"
				)
			panel_lines.append("-")

	# Hide any unused axes
	for k in range(len(sf_indices), len(axes)):
		axes[k].axis('off')

	# Global 3-way omnibus (Training×N×SF), one block per group
	global_lines: List[str] = []
	for group in ["WT", "RESCUED"]:
		rm3, art3 = _compute_rm_art_3way_from_rows(
			rows_3way_by_group.get(group, []),
			subject='subject',
			factor_a='training',
			factor_b='n',
			factor_c='sf',
			value_col='value',
			center_by_subject=True,
		)
		art3_raov = _art_raov_3way_pvalues_from_rows(
			rows_3way_by_group.get(group, []),
			subject='subject',
			factor_a='training',
			factor_b='n',
			factor_c='sf',
			value_col='value',
			include_interactions=ART_INCLUDE_INTERACTION_TERMS,
		)
		art3_rlmer = _art_r_3way_pvalues_from_rows(
			rows_3way_by_group.get(group, []),
			subject='subject',
			factor_a='training',
			factor_b='n',
			factor_c='sf',
			value_col='value',
			include_interactions=ART_INCLUDE_INTERACTION_TERMS,
		)
		lme3 = _lme_3way_lr_pvalues_from_rows(
			rows_3way_by_group.get(group, []),
			subject='subject',
			factor_a='training',
			factor_b='n',
			factor_c='sf',
			value_col='value',
			include_interactions=LME_INCLUDE_INTERACTION_TERMS,
		)
		lme4_3 = _lme4_3way_lrt_pvalues_from_rows(
			rows_3way_by_group.get(group, []),
			subject='subject',
			factor_a='training',
			factor_b='n',
			factor_c='sf',
			value_col='value',
			include_interactions=LME_INCLUDE_INTERACTION_TERMS,
		)
		global_lines.append(f"{group} RM3 (t,n,sf): t={_fmt_opt(rm3['training'])} n={_fmt_opt(rm3['n'])} sf={_fmt_opt(rm3['sf'])}")
		global_lines.append(
			f"{group} RM3 ints: t:n={_fmt_opt(rm3['training:n'])} t:sf={_fmt_opt(rm3['training:sf'])} n:sf={_fmt_opt(rm3['n:sf'])} t:n:sf={_fmt_opt(rm3['training:n:sf'])}"
		)
		global_lines.append(f"{group} ART3 (t,n,sf): t={_fmt_opt(art3['training'])} n={_fmt_opt(art3['n'])} sf={_fmt_opt(art3['sf'])}")
		global_lines.append(
			f"{group} ART3 ints: t:n={_fmt_opt(art3['training:n'])} t:sf={_fmt_opt(art3['training:sf'])} n:sf={_fmt_opt(art3['n:sf'])} t:n:sf={_fmt_opt(art3['training:n:sf'])}"
		)
		global_lines.append(f"{group} ART3-Raov (t,n,sf): t={_fmt_opt(art3_raov['training'])} n={_fmt_opt(art3_raov['n'])} sf={_fmt_opt(art3_raov['sf'])}")
		global_lines.append(
			f"{group} ART3-Raov ints: t:n={_fmt_opt(art3_raov['training:n'])} t:sf={_fmt_opt(art3_raov['training:sf'])} n:sf={_fmt_opt(art3_raov['n:sf'])} t:n:sf={_fmt_opt(art3_raov['training:n:sf'])}"
		)
		global_lines.append(f"{group} ART3-Rlmer (t,n,sf): t={_fmt_opt(art3_rlmer['training'])} n={_fmt_opt(art3_rlmer['n'])} sf={_fmt_opt(art3_rlmer['sf'])}")
		global_lines.append(
			f"{group} ART3-Rlmer ints: t:n={_fmt_opt(art3_rlmer['training:n'])} t:sf={_fmt_opt(art3_rlmer['training:sf'])} n:sf={_fmt_opt(art3_rlmer['n:sf'])} t:n:sf={_fmt_opt(art3_rlmer['training:n:sf'])}"
		)
		global_lines.append(f"{group} LME3 (t,n,sf): t={_fmt_opt(lme3['training'])} n={_fmt_opt(lme3['n'])} sf={_fmt_opt(lme3['sf'])}")
		global_lines.append(
			f"{group} LME3 ints: t:n={_fmt_opt(lme3['training:n'])} t:sf={_fmt_opt(lme3['training:sf'])} n:sf={_fmt_opt(lme3['n:sf'])} t:n:sf={_fmt_opt(lme3['training:n:sf'])}"
		)
		global_lines.append(f"{group} LME3-lme4 (t,n,sf): t={_fmt_opt(lme4_3['training'])} n={_fmt_opt(lme4_3['n'])} sf={_fmt_opt(lme4_3['sf'])}")
		global_lines.append(
			f"{group} LME3-lme4 ints: t:n={_fmt_opt(lme4_3['training:n'])} t:sf={_fmt_opt(lme4_3['training:sf'])} n:sf={_fmt_opt(lme4_3['n:sf'])} t:n:sf={_fmt_opt(lme4_3['training:n:sf'])}"
		)
		global_lines.append("-")
	if global_lines and global_lines[-1] == "-":
		global_lines = global_lines[:-1]
	if global_lines:
		panel_lines = global_lines + ["-"] + panel_lines

	# Global stats panel on the right
	if panel_lines and panel_lines[-1] == "-":
		panel_lines = panel_lines[:-1]
	if panel_lines:
		# Allow longer lines; we reserve more right margin for this panel.
		panel_lines_wrapped = _wrap_stats_panel_lines(panel_lines, max_chars=140)
		# Dynamically choose columns so the panel fits vertically.
		# Rough height model: ~1.25*fontsize points per line.
		fig_h_in = float(fig.get_size_inches()[1])
		available_pts = fig_h_in * 72.0 * 0.82
		fontsize = 5.2
		# Conservative estimate to avoid under-predicting panel height.
		per_line_pts = max(1.0, fontsize * 2.2)
		max_lines_per_col = max(12, int(available_pts / per_line_pts))
		n_cols = int(np.ceil(len(panel_lines_wrapped) / float(max_lines_per_col)))
		n_cols = int(min(max(n_cols, 1), 6))
		# If we still need many columns, slightly reduce font + line spacing.
		line_sep = 0.4
		if n_cols >= 3:
			fontsize = 4.8
			line_sep = 0.2
		if n_cols >= 4:
			fontsize = 4.5
			line_sep = 0.0

		# Keep the top visible; if anything clips, it will clip at the bottom only.
		panel_y = 0.99
		_add_colored_stats_panel_to_figure(
			fig,
			panel_lines_wrapped,
			x=0.995,
			y=panel_y,
			fontsize=fontsize,
			n_cols=n_cols,
			line_sep=line_sep,
		)

	fig.suptitle(title, fontsize=13, fontweight='bold')
	# Leave a generous right margin so the stats panel never overlaps the subplots.
	fig.tight_layout(rect=[0.0, 0.0, 0.42, 0.95])
	# Avoid bbox_inches='tight' here; it can collapse the reserved right margin.
	fig.savefig(os.path.join(GROUPED_PLOTS_DIR, outfile), format='svg')
	plt.close(fig)
	print(f"  [saved] {outfile}")


def plot_orientation_decoding_accuracy_vs_cell_number_per_sf() -> None:
	_plot_metric_vs_cell_number_per_sf(
		metric_base='decoding_accuracy',
		title='Continuous orientation decoding correlation vs neuron count (per SF)',
		ylabel='Circular correlation',
		outfile='orientation_decoding_accuracy_vs_cell_number_per_sf_groups.svg',
		y_limits=(-1, 1),
		include_blind=False,
		chance_null_metric_base='decoding_accuracy',
	)


def plot_decoding_based_osi_vs_cell_number_per_sf() -> None:
	_plot_metric_vs_cell_number_per_sf(
		metric_base='decoding_based_osi',
		title='Decoding-based OSI vs neuron count (per SF)',
		ylabel='Decoding-based OSI',
		outfile='decoding_based_osi_vs_cell_number_per_sf_groups.svg',
		y_limits=(0, 1),
	)


def _safe_mouse_id(s_pre: str, s_post: str) -> str:
	"""Derive mouse ID for filenames."""
	try:
		mid = _mouse_id_from_session_rel(s_pre)
		if mid:
			return str(mid)
	except Exception:
		pass
	# Fallback: try post session
	try:
		mid = _mouse_id_from_session_rel(s_post)
		if mid:
			return str(mid)
	except Exception:
		pass
	return "UNKNOWN"


def plot_mouse_sf_decoding_accuracy_vs_cell_number(s_pre: str, s_post: str) -> None:
	"""Per-mouse continuous SF decoding correlation (avg across orientations) vs neuron count."""
	cell_counts = [int(n) for n in DECODING_CELL_COUNTS]
	mouse_id = _safe_mouse_id(s_pre, s_post)

	summary_pre = prepare_session_summaries(s_pre)
	summary_post = prepare_session_summaries(s_post)

	y_pre = np.full(len(cell_counts), np.nan, dtype=float)
	y_post = np.full(len(cell_counts), np.nan, dtype=float)
	for j, n in enumerate(cell_counts):
		v_pre = _get_metric_for_n(summary_pre, 'sf_decoding_by_ori', n)
		v_post = _get_metric_for_n(summary_post, 'sf_decoding_by_ori', n)
		if v_pre is not None:
			y_pre[j] = float(np.nanmean(np.asarray(v_pre, dtype=float)))
		if v_post is not None:
			y_post[j] = float(np.nanmean(np.asarray(v_post, dtype=float)))

	fig, ax = plt.subplots(figsize=(11.5, 5.6))
	ax.plot(cell_counts, y_pre, '-o', color='tab:blue', label='Before')
	ax.plot(cell_counts, y_post, '-o', color='tab:orange', label='After')
	ax.axhline(y=0.0, linestyle='--', color='red', linewidth=1.0, label='Null correlation')
	ax.set_xlabel('Number of neurons used for decoding')
	ax.set_ylabel('SF decoding Pearson r (avg across orientations)')
	ax.set_ylim(-1, 1)
	ax.set_title(f'Continuous SF decoding correlation vs neuron count | {mouse_id}')
	ax.legend(loc='lower right')

	# Right-side info panel (single mouse: no across-mouse inferential stats)
	lines = [
		f"Mouse: {mouse_id}",
		f"Pre: {s_pre}",
		f"Post: {s_post}",
		"Note: single-mouse plot (no group-level p/q)",
	]
	_add_colored_stats_panel_to_figure(fig, lines, x=0.985, y=0.50, fontsize=6.2)
	fig.tight_layout(rect=[0.0, 0.0, 0.64, 1.0])

	outfile = os.path.join(GROUPED_PLOTS_DIR, f'sf_decoding_accuracy_vs_cell_number_{mouse_id}.svg')
	fig.savefig(outfile, format='svg')
	plt.close(fig)
	print(f"  [saved] {os.path.basename(outfile)}")


def _plot_mouse_metric_vs_cell_number_per_sf(*, s_pre: str, s_post: str, metric_base: str, title: str, ylabel: str, outfile_prefix: str, y_limits: Optional[Tuple[float, float]] = None) -> None:
	"""Per-mouse 6-subplot figure of a metric vs neuron count (per SF)."""
	cell_counts = [int(n) for n in DECODING_CELL_COUNTS]
	mouse_id = _safe_mouse_id(s_pre, s_post)
	# Nonzero SF indices (skip baseline SF=0)
	sf_indices = list(range(1, min(len(TARGET_SF_VALUES), 7)))
	if len(sf_indices) != 6:
		sf_indices = [i for i in sf_indices if i != 0][:6]

	summary_pre = prepare_session_summaries(s_pre)
	summary_post = prepare_session_summaries(s_post)

	fig, axes = plt.subplots(2, 3, figsize=(18.5, 9.2), sharey=True)
	axes = axes.ravel()

	for ax_i, sf_idx in enumerate(sf_indices):
		ax = axes[ax_i]
		sf_val = float(TARGET_SF_VALUES[sf_idx])
		ax.set_title(f"SF={sf_val:.4g}")
		ax.set_xlabel('N neurons')
		ax.set_ylabel(ylabel)
		if y_limits is not None:
			ax.set_ylim(*y_limits)
			if y_limits[0] < 0 < y_limits[1]:
				ax.axhline(y=0.0, linestyle='--', color='red', linewidth=1.0, alpha=0.8)
		else:
			ax.set_ylim(bottom=0)

		y_pre = np.full(len(cell_counts), np.nan, dtype=float)
		y_post = np.full(len(cell_counts), np.nan, dtype=float)
		for j, n in enumerate(cell_counts):
			v_pre = _get_metric_for_n(summary_pre, metric_base, n)
			v_post = _get_metric_for_n(summary_post, metric_base, n)
			if v_pre is None or v_post is None:
				continue
			v_pre = np.asarray(v_pre, dtype=float)
			v_post = np.asarray(v_post, dtype=float)
			if v_pre.ndim != 1 or v_post.ndim != 1:
				continue
			if sf_idx >= v_pre.size or sf_idx >= v_post.size:
				continue
			y_pre[j] = float(v_pre[sf_idx])
			y_post[j] = float(v_post[sf_idx])

		ax.plot(cell_counts, y_pre, '-o', color='tab:blue')
		ax.plot(cell_counts, y_post, '-o', color='tab:orange')

	# Hide any unused axes
	for k in range(len(sf_indices), len(axes)):
		axes[k].axis('off')

	# Global right-side info panel
	panel_lines = [
		f"Mouse: {mouse_id}",
		f"Pre: {s_pre}",
		f"Post: {s_post}",
		"Note: single-mouse plot (no group-level p/q)",
	]
	_add_colored_stats_panel_to_figure(fig, panel_lines, x=0.985, y=0.50, fontsize=6.2)
	fig.suptitle(f"{title} | {mouse_id}", fontsize=13, fontweight='bold')
	fig.tight_layout(rect=[0.0, 0.0, 0.62, 0.95])

	outfile = os.path.join(GROUPED_PLOTS_DIR, f"{outfile_prefix}_{mouse_id}.svg")
	fig.savefig(outfile, format='svg')
	plt.close(fig)
	print(f"  [saved] {os.path.basename(outfile)}")


def plot_mouse_orientation_decoding_accuracy_vs_cell_number_per_sf(s_pre: str, s_post: str) -> None:
	_plot_mouse_metric_vs_cell_number_per_sf(
		s_pre=s_pre,
		s_post=s_post,
		metric_base='decoding_accuracy',
		title='Continuous orientation decoding correlation vs neuron count (per SF)',
		ylabel='Circular correlation',
		outfile_prefix='orientation_decoding_accuracy_vs_cell_number_per_sf',
		y_limits=(-1, 1),
	)


def plot_mouse_decoding_based_osi_vs_cell_number_per_sf(s_pre: str, s_post: str) -> None:
	_plot_mouse_metric_vs_cell_number_per_sf(
		s_pre=s_pre,
		s_post=s_post,
		metric_base='decoding_based_osi',
		title='Decoding-based OSI vs neuron count (per SF)',
		ylabel='Decoding-based OSI',
		outfile_prefix='decoding_based_osi_vs_cell_number_per_sf',
	)


def plot_all_mouse_cellcount_figures() -> None:
	"""Generate 3 per-mouse decoding-vs-cell-count figures for every pre/post mouse pair."""
	print("\n" + "=" * 70)
	print("PER-MOUSE DECODING-vs-CELL-COUNT FIGURES")
	print("=" * 70)
	for group in ["WT", "RESCUED"]:
		pairs = MOUSE_PAIRS.get(group, [])
		for s_pre, s_post in pairs:
			plot_mouse_sf_decoding_accuracy_vs_cell_number(s_pre, s_post)
			plot_mouse_orientation_decoding_accuracy_vs_cell_number_per_sf(s_pre, s_post)
			plot_mouse_decoding_based_osi_vs_cell_number_per_sf(s_pre, s_post)


def _run_decoding_figures_for_n_cells(n_cells: int) -> None:
	"""Generate all figures in the per-n-cell loop.

	This is split out so it can be executed in a worker process when USE_PARALLEL=True.
	"""
	orientation_ref = get_orientation_reference()
	sf_class_count = get_nonzero_sf_count()
	rel_bins = [-60, -30, 0, 30, 60, 90]

	# Orientation decoding accuracy vs SF
	plot_metric(
		metric_key=f"decoding_accuracy_n{int(n_cells)}",
		title=f"Continuous orientation decoding correlation vs spatial frequency | N={int(n_cells)}",
		ylabel="Circular correlation",
		filename=f"orientation_decoding_accuracy_by_sf_groups_n{int(n_cells)}.svg",
		ylim=(-1, 1),
		exclude_groups=['BLIND'],
		chance_null_metric_key=f"decoding_accuracy_n{int(n_cells)}",
	)

	# Decoding-based OSI vs SF
	plot_metric(
		metric_key=f"decoding_based_osi_n{int(n_cells)}",
		title=f"Decoding-based OSI vs spatial frequency | N={int(n_cells)}",
		ylabel="Decoding-based OSI (vector sum)",
		filename=f"decoding_based_osi_by_sf_groups_n{int(n_cells)}.svg",
		ylim=(0, 1),
	)

	# Decoder output fractions by relative orientation (matrix over SF)
	plot_matrix_over_sf(
		metric_key=f"rel_frac_n{int(n_cells)}",
		axis_labels=["-60", "-30", "Correct", "+30", "+60", "+90"],
		title=f"Decoder output fractions by relative orientation | N={int(n_cells)}",
		ylabel="Fraction of trials",
		filename=f"decoder_output_fraction_relative_orientation_by_sf_groups_n{int(n_cells)}.svg",
	)

	# Decoder orientation accuracy by SF (matrix over SF)
	plot_matrix_over_sf(
		metric_key=f"dec_ori_acc_n{int(n_cells)}",
		axis_labels=["Ori 1", "Ori 2", "Ori 3", "Ori 4", "Ori 5", "Ori 6"],
		title=f"Decoder accuracy by orientation and SF | N={int(n_cells)}",
		ylabel="Accuracy",
		filename=f"decoder_orientation_accuracy_by_sf_groups_n{int(n_cells)}.svg",
		chance=(1.0 / len(orientation_ref)) if orientation_ref is not None else None,
	)

	# SF decoding accuracy by orientation
	plot_orientation_metric(
		metric_key=f"sf_decoding_by_ori_n{int(n_cells)}",
		title=f"Continuous SF decoding correlation vs orientation | N={int(n_cells)}",
		ylabel="SF decoding Pearson r",
		filename=f"sf_decoding_accuracy_by_orientation_groups_n{int(n_cells)}.svg",
		exclude_groups=['BLIND'],
		chance_null_metric_key=f"sf_decoding_by_ori_n{int(n_cells)}",
	)

	# Per-SF bins: decoder output fractions
	plot_per_sf_bins(
		metric_key=f"rel_frac_n{int(n_cells)}",
		x_values=rel_bins,
		x_labels=["-60", "-30", "0", "+30", "+60", "+90"],
		title=f"Decoder output fractions per SF | N={int(n_cells)}",
		ylabel="Fraction of trials",
		filename=f"decoder_output_fraction_relative_orientation_per_sf_groups_n{int(n_cells)}.svg",
		add_stats_boxes=True,
	)

	# Per-SF bins: decoder orientation accuracy
	plot_per_sf_bins(
		metric_key=f"dec_ori_acc_n{int(n_cells)}",
		x_values=list(orientation_ref) if orientation_ref is not None else list(range(6)),
		x_labels=[f"{o:.0f}" for o in (orientation_ref if orientation_ref is not None else list(range(6)))],
		title=f"Decoder orientation accuracy per SF | N={int(n_cells)}",
		ylabel="Accuracy",
		filename=f"decoder_accuracy_by_orientation_per_sf_groups_n{int(n_cells)}.svg",
		chance=(1.0 / len(orientation_ref)) if orientation_ref is not None else None,
		x_axis_label="Orientation (deg)",
	)


def run_all_analyses(*, use_parallel: bool = USE_PARALLEL, max_workers: Optional[int] = PARALLEL_MAX_WORKERS) -> None:
	"""Run all analyses/plots.

	Kept as a function so Windows multiprocessing can import this module safely.
	"""
	orientation_ref = get_orientation_reference()
	sf_class_count = get_nonzero_sf_count()

	plot_metric(
		metric_key="responsive_fraction_by_sf",
		title="Fraction of responsive neurons vs spatial frequency",
		ylabel="Responsive fraction",
		filename="responsive_fraction_by_sf_groups.svg",
		ylim=(0, 1),
		blind_as_individual=True,
		blind_color_override='black',
		background_band=(0.0, 0.05),
	)

	plot_orientation_metric(
		metric_key="responsive_fraction_by_orientation_non064",
		title="Fraction of visually responsive neurons vs orientation (excluding SF=0.64)",
		ylabel="Responsive fraction",
		filename="responsive_fraction_by_orientation_excluding_064_groups.svg",
		ylim=(0, 1),
		blind_as_individual=True,
		blind_color_override='black',
		background_band=(0.0, 0.05),
	)

	plot_grouped_neural_activity_heatmap_and_mean_activity()

	plot_metric(
		metric_key="decoding_accuracy",
		title="Continuous orientation decoding correlation vs spatial frequency",
		ylabel="Circular correlation",
		filename="orientation_decoding_accuracy_by_sf_groups.svg",
		ylim=(-1, 1),
		exclude_groups=['BLIND'],
		chance_null_metric_key='decoding_accuracy',
	)

	plot_metric(
		metric_key="osi_spf",
		title="OSI (vector sum) vs spatial frequency",
		ylabel="OSI (vector sum)",
		filename="osi_vector_sum_by_sf_groups.svg",
		ylim=(0, 1),
	)

	plot_metric(
		metric_key="osi_diff",
		title="OSI (difference method) vs spatial frequency",
		ylabel="OSI (difference)",
		filename="osi_difference_by_sf_groups.svg",
		ylim=(0, 1),
	)

	plot_metric(
		metric_key="noise_corr",
		title="Noise correlation vs spatial frequency",
		ylabel="Noise correlation",
		filename="noise_correlation_by_sf_groups.svg",
	)

	plot_metric(
		metric_key="signal_corr",
		title="Signal correlation vs spatial frequency",
		ylabel="Signal correlation",
		filename="signal_correlation_by_sf_groups.svg",
	)

	plot_metric(
		metric_key="signal_minus_noise_sf",
		title="Signal - Noise correlation vs spatial frequency",
		ylabel="Signal - Noise correlation",
		filename="signal_minus_noise_correlation_by_sf_groups.svg",
	)

	plot_metric(
		metric_key="decoding_based_osi",
		title="Decoding-based OSI vs spatial frequency",
		ylabel="Decoding-based OSI (vector sum)",
		filename="decoding_based_osi_by_sf_groups.svg",
		ylim=(0, 1),
	)

	# Decoder relative orientation output fractions (per SF, 6 bins: -60, -30, correct, +30, +60, +90)
	plot_matrix_over_sf(
		metric_key="rel_frac",
		axis_labels=["-60", "-30", "Correct", "+30", "+60", "+90"],
		title="Decoder output fractions by relative orientation",
		ylabel="Fraction of trials",
		filename="decoder_output_fraction_relative_orientation_by_sf_groups.svg",
	)

	# Decoder orientation accuracy per SF (per-orientation curves)
	plot_matrix_over_sf(
		metric_key="dec_ori_acc",
		axis_labels=["Ori 1", "Ori 2", "Ori 3", "Ori 4", "Ori 5", "Ori 6"],
		title="Decoder accuracy by orientation and SF",
		ylabel="Accuracy",
		filename="decoder_orientation_accuracy_by_sf_groups.svg",
		chance=(1.0 / len(orientation_ref)) if orientation_ref is not None else None,
	)

	# Signal/noise metrics by orientation (requires consistent orientations across sessions)
	plot_orientation_metric(
		metric_key="signal_corr_by_ori",
		title="Signal correlation vs orientation",
		ylabel="Signal correlation",
		filename="signal_correlation_by_orientation_groups.svg",
	)

	plot_orientation_metric(
		metric_key="noise_corr_by_ori",
		title="Noise correlation vs orientation",
		ylabel="Noise correlation",
		filename="noise_correlation_by_orientation_groups.svg",
	)

	plot_orientation_metric(
		metric_key="signal_minus_noise_by_ori",
		title="Signal - Noise correlation vs orientation",
		ylabel="Signal - Noise correlation",
		filename="signal_minus_noise_correlation_by_orientation_groups.svg",
	)

	plot_orientation_metric(
		metric_key="sf_decoding_by_ori",
		title="Continuous SF decoding correlation vs orientation",
		ylabel="SF decoding Pearson r",
		filename="sf_decoding_accuracy_by_orientation_groups.svg",
		exclude_groups=['BLIND'],
		chance_null_metric_key='sf_decoding_by_ori',
	)

	# Per-SF versions (relative orientation bins and orientation accuracies)
	rel_bins = [-60, -30, 0, 30, 60, 90]
	plot_per_sf_bins(
		metric_key="rel_frac",
		x_values=rel_bins,
		x_labels=["-60", "-30", "0", "+30", "+60", "+90"],
		title="Decoder output fractions per SF",
		ylabel="Fraction of trials",
		filename="decoder_output_fraction_relative_orientation_per_sf_groups.svg",
		add_stats_boxes=True,
	)

	plot_per_sf_bins(
		metric_key="dec_ori_acc",
		x_values=list(orientation_ref) if orientation_ref is not None else list(range(6)),
		x_labels=[f"{o:.0f}" for o in (orientation_ref if orientation_ref is not None else list(range(6)))],
		title="Decoder orientation accuracy per SF",
		ylabel="Accuracy",
		filename="decoder_accuracy_by_orientation_per_sf_groups.svg",
		chance=(1.0 / len(orientation_ref)) if orientation_ref is not None else None,
		x_axis_label="Orientation (deg)",
	)

	# Participation ratio and power-law alpha
	plot_metric(
		metric_key="pr_by_sf",
		title="Participation ratio vs spatial frequency",
		ylabel="Participation ratio",
		filename="pca_participation_ratio_by_sf_groups.svg",
	)

	plot_scalar_bar(
		metric_key="pr_overall",
		title="Participation ratio (overall)",
		ylabel="Participation ratio",
		filename="pca_participation_ratio_overall_groups.svg",
	)

	plot_scalar_bar(
		metric_key="alpha_powerlaw",
		title="Power-law alpha (overall)",
		ylabel="Alpha",
		filename="pca_alpha_overall_groups.svg",
	)

	# dsPCA variance explained
	plot_scalar_bar(
		metric_key="var_ori",
		title="dsPCA variance explained (orientation)",
		ylabel="Variance explained",
		filename="dspca_variance_explained_orientation_groups.svg",
	)

	plot_scalar_bar(
		metric_key="var_sf",
		title="dsPCA variance explained (spatial freq)",
		ylabel="Variance explained",
		filename="dspca_variance_explained_sf_groups.svg",
	)

	# ---------------------------------------------------------------------------
	# Additional decoding figures for each neuron count (10..80)
	# ---------------------------------------------------------------------------

	if use_parallel:
		print("\n" + "=" * 70)
		print(f"PARALLEL DECODING FIGURES | workers={max_workers or os.cpu_count()}")
		print("=" * 70)
		with ProcessPoolExecutor(max_workers=max_workers) as ex:
			futures = [ex.submit(_run_decoding_figures_for_n_cells, int(n)) for n in DECODING_CELL_COUNTS]
			for fut in as_completed(futures):
				fut.result()
	else:
		for n_cells in DECODING_CELL_COUNTS:
			_run_decoding_figures_for_n_cells(int(n_cells))

	# New summary figures that sweep neuron count
	plot_sf_decoding_accuracy_vs_cell_number()
	plot_orientation_decoding_accuracy_vs_cell_number_per_sf()
	plot_orientation_decoding_accuracy_vs_cell_number_mean_non240_sf()
	plot_decoding_based_osi_vs_cell_number_per_sf()

	# Per-mouse versions (3 figures per mouse)
	plot_all_mouse_cellcount_figures()

	# Run pseudopopulation analyses
	print("\n" + "="*70)
	print("PSEUDOPOPULATION ANALYSES")
	print("="*70)

	pseudopop_pca_results = run_pseudopopulation_pca()
	pseudopop_dspca_results = run_pseudopopulation_dspca(subsample_neurons=True, out_tag="")
	_ = run_pseudopopulation_dspca(subsample_neurons=False, out_tag="_all_neurons")
	# Cross-validated pseudopopulation dsPCA (10-fold; generates separate *_10fcv figures)
	_ = run_pseudopopulation_dspca_cv50(subsample_neurons=True, out_tag="_10fcv")
	_ = run_pseudopopulation_dspca_cv50(subsample_neurons=False, out_tag="_all_neurons_10fcv")

	# dsPCA distance metric comparison plots (Before vs After)
	print("\n" + "="*70)
	print("dsPCA DISTANCE METRIC COMPARISONS")
	print("="*70)
	plot_distance_to_com_comparison()
	plot_adjacent_distance_comparison()
	plot_adjacent_cv_comparison()
	plot_circle_radial_cv_comparison()
	plot_circle_fit_rmse_comparison()
	plot_circle_angle_spacing_comparison()
	plot_adjacent_mean_distance_by_sf_paired()
	plot_orientation_area_by_sf_paired()
	print("\n  Session-averaged dsPCA orient-plane plots")
	plot_sessionavg_dspca_adjacent_sf_axis_distance_all_groups(cv50=False)
	plot_sessionavg_dspca_com_distance_from_origin_by_sf_all_groups(cv50=False)
	plot_sessionavg_dspca_orient_plane_by_sf_superimposed(group_label="WT", cv50=False)
	plot_sessionavg_dspca_orient_plane_by_sf_superimposed(group_label="RESCUED", cv50=False)
	plot_sessionavg_dspca_orient_plane_by_sf_superimposed_rescued_after_with_wt_before(cv50=False)
	plot_adjacent_mean_distance_by_sf_paired_cv50()
	plot_orientation_area_by_sf_paired_cv50()
	plot_sessionavg_dspca_adjacent_sf_axis_distance_all_groups(cv50=True)
	plot_sessionavg_dspca_com_distance_from_origin_by_sf_all_groups(cv50=True)
	plot_sessionavg_dspca_orient_plane_by_sf_superimposed(group_label="WT", cv50=True)
	plot_sessionavg_dspca_orient_plane_by_sf_superimposed(group_label="RESCUED", cv50=True)
	plot_sessionavg_dspca_orient_plane_by_sf_superimposed_rescued_after_with_wt_before(cv50=True)

	print("\n" + "="*70)
	print("All analyses complete!")
	print("="*70)


def run_dspca_analyses_only(*, skip_cv50: bool = False) -> None:
	"""Run only dsPCA-related plots and pseudopop dsPCA analyses.

	Includes:
	- Grouped dsPCA variance explained bars (per-session outputs)
	- Pseudopopulation dsPCA (subsampled + all neurons)
	- Optional pseudopopulation dsPCA 10-fold CV (subsampled + all neurons)
	- dsPCA distance/geometry comparison figures
	"""
	print("\n" + "=" * 70)
	print("dsPCA-ONLY RUN")
	print("=" * 70)

	# Group-level dsPCA variance explained (from per-session saved metrics)
	plot_scalar_bar(
		metric_key="var_ori",
		title="dsPCA variance explained (orientation)",
		ylabel="Variance explained",
		filename="dspca_variance_explained_orientation_groups.svg",
	)
	plot_scalar_bar(
		metric_key="var_sf",
		title="dsPCA variance explained (spatial freq)",
		ylabel="Variance explained",
		filename="dspca_variance_explained_sf_groups.svg",
	)

	# Pseudopop dsPCA analyses
	print("\n" + "=" * 70)
	print("PSEUDOPOPULATION dsPCA")
	print("=" * 70)
	run_pseudopopulation_dspca(subsample_neurons=True, out_tag="")
	run_pseudopopulation_dspca(subsample_neurons=False, out_tag="_all_neurons")
	if not skip_cv50:
		run_pseudopopulation_dspca_cv50(subsample_neurons=True, out_tag="_10fcv")
		run_pseudopopulation_dspca_cv50(subsample_neurons=False, out_tag="_all_neurons_10fcv")

	# dsPCA distance/geometry metric comparisons
	print("\n" + "=" * 70)
	print("dsPCA METRIC COMPARISONS")
	print("=" * 70)
	plot_distance_to_com_comparison()
	plot_adjacent_distance_comparison()
	plot_adjacent_cv_comparison()
	plot_circle_radial_cv_comparison()
	plot_circle_fit_rmse_comparison()
	plot_circle_angle_spacing_comparison()
	plot_adjacent_mean_distance_by_sf_paired()
	plot_orientation_area_by_sf_paired()
	print("\n  Session-averaged dsPCA orient-plane plots")
	plot_sessionavg_dspca_adjacent_sf_axis_distance_all_groups(cv50=False)
	plot_sessionavg_dspca_com_distance_from_origin_by_sf_all_groups(cv50=False)
	plot_sessionavg_dspca_orient_plane_by_sf_superimposed(group_label="WT", cv50=False)
	plot_sessionavg_dspca_orient_plane_by_sf_superimposed(group_label="RESCUED", cv50=False)
	plot_sessionavg_dspca_orient_plane_by_sf_superimposed_rescued_after_with_wt_before(cv50=False)
	if not skip_cv50:
		plot_adjacent_mean_distance_by_sf_paired_cv50()
		plot_orientation_area_by_sf_paired_cv50()
		plot_sessionavg_dspca_adjacent_sf_axis_distance_all_groups(cv50=True)
		plot_sessionavg_dspca_com_distance_from_origin_by_sf_all_groups(cv50=True)
		plot_sessionavg_dspca_orient_plane_by_sf_superimposed(group_label="WT", cv50=True)
		plot_sessionavg_dspca_orient_plane_by_sf_superimposed(group_label="RESCUED", cv50=True)
		plot_sessionavg_dspca_orient_plane_by_sf_superimposed_rescued_after_with_wt_before(cv50=True)

	print("\n" + "=" * 70)
	print("dsPCA-only run complete!")
	print("=" * 70)


def _parse_cli_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Grouped analysis/plotting for grating task results.",
		add_help=True,
	)
	parser.add_argument(
		"--dspca-only",
		action="store_true",
		help="Run only dsPCA sections (variance bars, pseudopop dsPCA, dsPCA comparison plots).",
	)
	parser.add_argument(
		"--skip-cv50",
		action="store_true",
		help="Skip held-out CV pseudopop dsPCA and held-out CV summary plots.",
	)

	par_group = parser.add_mutually_exclusive_group()
	par_group.add_argument(
		"--parallel",
		action="store_true",
		help="Enable parallel decoding figures (only relevant for full run).",
	)
	par_group.add_argument(
		"--no-parallel",
		action="store_true",
		help="Disable parallel decoding figures (only relevant for full run).",
	)
	parser.add_argument(
		"--max-workers",
		type=int,
		default=None,
		help="Max workers for parallel decoding figures (only relevant for full run).",
	)
	parser.add_argument(
		"--null-workers",
		type=int,
		default=None,
		help="Max workers for shuffled-label null decoding cache generation.",
	)
	parser.add_argument(
		"--shuffled-null-worker",
		default=None,
		help=argparse.SUPPRESS,
	)
	parser.add_argument(
		"--null-iter",
		type=int,
		default=SHUFFLED_DECODING_NULL_ITERATIONS,
		help=argparse.SUPPRESS,
	)
	parser.add_argument(
		"--null-output",
		default=None,
		help=argparse.SUPPRESS,
	)

	# Notebook kernels inject argv entries like --f=<path>. Ignore unknowns.
	args, _unknown = parser.parse_known_args()
	return args


if __name__ == '__main__':
	_args = _parse_cli_args()
	if _args.null_workers is not None:
		SHUFFLED_DECODING_NULL_MAX_WORKERS = max(1, int(_args.null_workers))
	use_parallel = USE_PARALLEL
	if getattr(_args, "parallel", False):
		use_parallel = True
	elif getattr(_args, "no_parallel", False):
		use_parallel = False
	max_workers = _args.max_workers if (_args.max_workers is not None) else PARALLEL_MAX_WORKERS

	if _args.shuffled_null_worker:
		if not _args.null_output:
			raise ValueError('--null-output is required with --shuffled-null-worker')
		_save_session_null_cache_file(
			_args.shuffled_null_worker,
			n_iter=int(_args.null_iter),
			output_path=str(_args.null_output),
		)
	elif _args.dspca_only:
		run_dspca_analyses_only(skip_cv50=bool(_args.skip_cv50))
	else:
		run_all_analyses(use_parallel=use_parallel, max_workers=max_workers)

