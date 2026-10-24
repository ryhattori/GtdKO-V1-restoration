"""Data locations. Set the environment variables, or place the data under data/ in this repository.

V1_RAW_IMAGING  folder with one subfolder per imaging session (<date>/<mouse>/<hemisphere>[/<depth>]) containing the
                stimulus log files of that session
V1_SUITE2P      folder with the same session subfolders containing the Suite2p output (suite2p/plane0)
V1_RESULTS      folder where the per-session and group analyses write their results (created if absent)
V1_BEHAVIOR_ZIP zip file with one trial-by-trial xlsx file per mouse for the VR task
V1_TRIAL_CACHE  folder for cached single-trial responses (default: <V1_RESULTS>/_trial_cache)
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

RAW_IMAGING_ROOT = os.environ.get("V1_RAW_IMAGING", os.path.join(HERE, "data", "raw_imaging"))
SUITE2P_ROOT = os.environ.get("V1_SUITE2P", os.path.join(HERE, "data", "suite2p"))
RESULTS_ROOT = os.environ.get("V1_RESULTS", os.path.join(HERE, "results"))
BEHAVIOR_ZIP = os.environ.get("V1_BEHAVIOR_ZIP", os.path.join(HERE, "data", "behavior.zip"))
TRIAL_CACHE = os.environ.get("V1_TRIAL_CACHE", os.path.join(RESULTS_ROOT, "_trial_cache"))
