"""Figure 3M-N: trial table of the VR visual discrimination task.

Reads the per-mouse trial-by-trial xlsx files from the behavior zip (config.BEHAVIOR_ZIP), removes invalid trials,
re-indexes the kept trials and assigns the analysis windows (trials 1-400 and 401-1000).

Exclusion rule (applied identically to every mouse):
  1. Invalid trials: ChosenSide or CorrectPattern not in {1, 2}.
  2. Contingency mismatch: the rewarded pattern (CorrectPattern) or the stimulus set (PathPattern, where logged)
     differs from the contingency assigned to the mouse, defined as the modal value over its first 1000 trials.
Outputs: trials_clean.csv, exclusion_log.csv, mouse_epoch_summary.csv, mouse_level_values_ORIGINAL.csv
(values without exclusions, for comparison), export_order_check.csv.
"""
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "pipeline")]
import config  # noqa: E402
ZIP_PATH = config.BEHAVIOR_ZIP
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

WT_MICE = {"VD017", "VD018", "VD019", "VD020", "VD022"}
RESCUED_MICE = {"VD024", "VD031", "VD032", "VD033", "VD034"}

EPOCHS = [("early", 1, 400), ("late", 401, 1000)]
CONTINGENCY_REF_TRIALS = 1000


def mode_or_nan(s):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else np.nan


def load_mouse(zf, member):
    mouse = os.path.basename(member)[:5]
    df = pd.read_excel(io.BytesIO(zf.read(member)))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "TrialNumber"]).reset_index(drop=True)
    df.insert(0, "mouse", mouse)
    df.insert(1, "group", "WT" if mouse in WT_MICE else
              "Rescued" if mouse in RESCUED_MICE else "Unassigned")
    df["orig_idx"] = np.arange(1, len(df) + 1)
    df["session"] = mouse + "_" + df["Date"].dt.strftime("%Y-%m-%d")

    df["invalid"] = ~df["ChosenSide"].isin([1, 2]) | ~df["CorrectPattern"].isin([1, 2])

    ref = df[(df["orig_idx"] <= CONTINGENCY_REF_TRIALS) & ~df["invalid"]]
    assigned_pattern = mode_or_nan(ref["CorrectPattern"])
    assigned_set = mode_or_nan(ref["PathPattern"])
    df["assigned_pattern"] = assigned_pattern
    df["assigned_set"] = assigned_set

    wrong_pattern = df["CorrectPattern"] != assigned_pattern
    wrong_set = df["PathPattern"].notna() & (df["PathPattern"] != assigned_set) \
        if pd.notna(assigned_set) else pd.Series(False, index=df.index)
    df["excl_contingency"] = ~df["invalid"] & (wrong_pattern | wrong_set)
    df["keep"] = ~df["invalid"] & ~df["excl_contingency"]

    df["clean_idx"] = np.where(df["keep"], df["keep"].cumsum(), np.nan)
    df["epoch"] = pd.NA
    for name, lo, hi in EPOCHS:
        df.loc[(df["clean_idx"] >= lo) & (df["clean_idx"] <= hi), "epoch"] = name

    # correct side follows from the choice and its outcome
    df["corr_side"] = np.where(df["Correctness"] == 1, df["ChosenSide"], 3 - df["ChosenSide"])
    df.loc[df["invalid"], "corr_side"] = np.nan

    # previous kept trial within the same session (for history analyses)
    k = df[df["keep"]].copy()
    same_sess = k["session"].eq(k["session"].shift())
    df.loc[k.index, "prev_choice"] = k["ChosenSide"].shift().where(same_sess)
    df.loc[k.index, "prev_correct"] = k["Correctness"].shift().where(same_sess)
    df.loc[k.index, "prev_corr_side"] = k["corr_side"].shift().where(same_sess)
    return df


def main():
    with zipfile.ZipFile(ZIP_PATH) as zf:
        members = sorted(m for m in zf.namelist() if m.endswith("_Date_Trial_Correctness.xlsx"))
        trials = pd.concat([load_mouse(zf, m) for m in members], ignore_index=True)

    cols = ["mouse", "group", "session", "Date", "orig_idx", "clean_idx", "epoch", "keep",
            "invalid", "excl_contingency", "Correctness", "ChosenSide", "corr_side",
            "CorrectPattern", "PathPattern", "assigned_pattern", "assigned_set",
            "prev_choice", "prev_correct", "prev_corr_side"]
    trials[cols].to_csv(os.path.join(OUT_DIR, "trials_clean.csv"), index=False)

    # exclusion log: contiguous runs of excluded trials inside the analysis range
    log = []
    for mouse, d in trials.groupby("mouse"):
        ex = d[~d["keep"]]
        if ex.empty:
            continue
        runs = (ex["orig_idx"].diff() != 1).cumsum()
        for _, r in ex.groupby(runs):
            log.append({
                "mouse": mouse, "group": d["group"].iloc[0],
                "orig_trials": f"{r['orig_idx'].min()}-{r['orig_idx'].max()}",
                "n_trials": len(r),
                "reason": "invalid entry" if r["invalid"].all() else "contingency mismatch",
                "CorrectPattern": ",".join(map(str, sorted(r["CorrectPattern"].unique()))),
                "assigned_pattern": d["assigned_pattern"].iloc[0],
                "PathPattern": ",".join(map(str, r["PathPattern"].dropna().unique())) or "not logged",
                "dates": f"{r['Date'].min():%Y-%m-%d} to {r['Date'].max():%Y-%m-%d}",
                "within_first_1000_orig": bool(r["orig_idx"].min() <= 1000),
            })
    pd.DataFrame(log).to_csv(os.path.join(OUT_DIR, "exclusion_log.csv"), index=False)

    summ = (trials[trials["epoch"].notna()]
            .groupby(["group", "mouse", "epoch"])["Correctness"]
            .agg(n="size", correct="sum", p_correct="mean").reset_index())
    summ.to_csv(os.path.join(OUT_DIR, "mouse_epoch_summary.csv"), index=False)

    # values without trial exclusions (for comparison)
    orig = []
    for mouse, d in trials[trials["group"] != "Unassigned"].groupby("mouse"):
        c = d.sort_values("orig_idx")["Correctness"].to_numpy(float)
        orig.append({"group": d["group"].iloc[0], "mouse": mouse,
                     "early": c[0:400].mean(), "late": c[400:1000].mean()})
    pd.DataFrame(orig).to_csv(os.path.join(OUT_DIR, "mouse_level_values_ORIGINAL.csv"), index=False)

    # export-order check: same-day sessions whose trial numbers restart get
    # interleaved by the Date + TrialNumber sort (1, 1, 2, 2, ...)
    dup = trials[trials.duplicated(["mouse", "Date", "TrialNumber"], keep=False)]
    chk = (dup.groupby(["mouse", "Date"])
           .agg(dup_rows=("orig_idx", "size"), first_orig=("orig_idx", "min"), last_orig=("orig_idx", "max"),
                rows_in_fig3M_windows=("epoch", lambda s: int(s.notna().sum())))
           .reset_index())
    chk.to_csv(os.path.join(OUT_DIR, "export_order_check.csv"), index=False)
    print(pd.DataFrame(log).to_string(index=False))
    print()
    print(summ.pivot_table(index=["group", "mouse"], columns="epoch",
                           values=["p_correct", "n"]).round(3).to_string())


if __name__ == "__main__":
    main()
