"""Per-mouse learning evidence and the effective chance level of the VR task.

1. Trials to expert criterion: first trial at which accuracy over the preceding
   100 trials reaches 70% (kept trials only). Groups compared with the same
   exact two-sided permutation test used for Fig. 3M left (difference in means,
   all 252 relabellings of 10 mice into 5 + 5).
2. Task structure: how often the correct side repeats after an error vs. after a
   correct trial, and the accuracy a blind mouse could reach by exploiting it.
   With repeat probabilities p_err (after error) and p_cor (after correct), a
   blind mouse that simply alternates sides scores p_err / (p_err + p_cor);
   no strategy based on the previous trial does better.
3. Stimulus dependence controlling for side bias and choice history:
   per-mouse, per-epoch logistic regression
       P(choose side 1) ~ stimulus side + previous choice + previous choice x outcome
   (all coded +/-1). A positive stimulus weight means choices follow the
   grating pattern beyond what bias and history explain.
"""
import itertools
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW, CRITERION = 100, 0.70


def perm_test(a, b):
    """Exact two-sided permutation test on mean(a) - mean(b) over all relabellings."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled, obs = np.r_[a, b], a.mean() - b.mean()
    null = [pooled[list(ix)].mean() - np.delete(pooled, list(ix)).mean()
            for ix in itertools.combinations(range(len(pooled)), len(a))]
    return float(np.mean(np.abs(null) >= abs(obs) - 1e-12))

t = pd.read_csv(os.path.join(HERE, "trials_clean.csv"), low_memory=False)
t = t[t["keep"] & t["group"].isin(["WT", "Rescued", "Unassigned"])].sort_values(["mouse", "clean_idx"])

# ------------------------------------------------------------------ 1. trials to criterion
crit_rows = []
for mouse, d in t.groupby("mouse"):
    c = d["Correctness"].to_numpy(float)
    run = pd.Series(c).rolling(WINDOW).mean().to_numpy()
    hit = np.flatnonzero(run >= CRITERION)
    crit_rows.append({
        "mouse": mouse, "group": d["group"].iloc[0], "n_kept_trials": len(c),
        "trials_to_criterion": int(hit[0] + 1) if len(hit) else np.nan,
        "sessions_to_criterion": d["session"].iloc[:hit[0] + 1].nunique() if len(hit) else np.nan,
        "max_100trial_accuracy": np.nanmax(run),
    })
crit = pd.DataFrame(crit_rows)
wt_c = crit.loc[crit.group == "WT", "trials_to_criterion"].dropna()
rs_c = crit.loc[crit.group == "Rescued", "trials_to_criterion"].dropna()
p_crit = perm_test(rs_c, wt_c)
pd.DataFrame([{"test": "trials to criterion, Rescued - WT", "mean_rescued": rs_c.mean(),
               "mean_wt": wt_c.mean(), "diff": rs_c.mean() - wt_c.mean(), "p_perm_two_sided": p_crit}]
             ).to_csv(os.path.join(HERE, "criterion_test.csv"), index=False)

# ------------------------------------------------------------------ 2. task structure
def repeat_probs(d):
    d = d[d["prev_corr_side"].notna()]
    same = d["corr_side"] == d["prev_corr_side"]
    return same[d["prev_correct"] == 0].mean(), same[d["prev_correct"] == 1].mean(), len(d)

win = t[t["epoch"].notna()]
task_rows = []
for mouse, d in win.groupby("mouse"):
    p_err, p_cor, n = repeat_probs(d)
    task_rows.append({"mouse": mouse, "group": d["group"].iloc[0], "p_repeat_after_error": p_err,
                      "p_repeat_after_correct": p_cor, "blind_alternation_accuracy": p_err / (p_err + p_cor),
                      "p_switch_choice": (d["ChosenSide"] != d["prev_choice"])[d["prev_choice"].notna()].mean()})
task = pd.DataFrame(task_rows)
p_err_all, p_cor_all, _ = repeat_probs(win[win.group != "Unassigned"])
ceiling = p_err_all / (p_err_all + p_cor_all)

# ------------------------------------------------------------------ 3. stimulus weights
def pm(x):
    return np.where(x == 1, 1.0, -1.0)

reg_rows = []
for (mouse, epoch), d in win.groupby(["mouse", "epoch"]):
    d = d[d["prev_choice"].notna()]
    X = pd.DataFrame({"stimulus": pm(d["corr_side"]), "prev_choice": pm(d["prev_choice"]),
                      "win_stay_lose_shift": pm(d["prev_choice"]) * np.where(d["prev_correct"] == 1, 1, -1)},
                     index=d.index)
    y = (d["ChosenSide"] == 1).astype(float)
    fit = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    ci = fit.conf_int().loc["stimulus"]
    reg_rows.append({"mouse": mouse, "group": d["group"].iloc[0], "epoch": epoch, "n": len(d),
                     "accuracy": d["Correctness"].mean(), "stim_weight": fit.params["stimulus"],
                     "ci_lo": ci[0], "ci_hi": ci[1], "p": fit.pvalues["stimulus"],
                     "bias": fit.params["const"], "prev_choice_w": fit.params["prev_choice"],
                     "wsls_w": fit.params["win_stay_lose_shift"]})
reg = pd.DataFrame(reg_rows)

# ------------------------------------------------------------------ 4. criterion-window sensitivity
sens_rows = []
for w in (50, 100, 150, 200):
    ttc = {}
    for mouse, d in t[t.group.isin(["WT", "Rescued"])].groupby("mouse"):
        run = pd.Series(d["Correctness"].to_numpy(float)).rolling(w).mean().to_numpy()
        hit = np.flatnonzero(run >= CRITERION)
        ttc[mouse] = (d["group"].iloc[0], hit[0] + 1 if len(hit) else np.nan)
    wt_v = [v for g, v in ttc.values() if g == "WT"]
    rs_v = [v for g, v in ttc.values() if g == "Rescued"]
    sens_rows.append({"window": w, "WT": sorted(int(v) for v in wt_v), "Rescued": sorted(int(v) for v in rs_v),
                      "complete_separation": max(wt_v) < min(rs_v),
                      "p_exact_two_sided": perm_test(rs_v, wt_v)})
sens = pd.DataFrame(sens_rows)

# ------------------------------------------------------------------ 5. blind-agent null for the stimulus weight
# Simulate mice that cannot see: choices follow each mouse's fitted bias and
# history weights only, and the correct side follows the task's repeat rule.
rng = np.random.default_rng(1)
N_SIM = 200


def simulate_blind(n, b0, bc, bw, sessions=5):
    rows = []
    for _ in range(sessions):
        cs, prev_c, prev_r = rng.choice([1, -1]), None, None
        for _ in range(n // sessions):
            z = b0 if prev_c is None else b0 + bc * prev_c + bw * prev_c * prev_r
            ch = 1 if rng.random() < 1 / (1 + np.exp(-z)) else -1
            r = 1 if ch == cs else -1
            if prev_c is not None:
                rows.append((ch == 1, cs, prev_c, prev_c * prev_r, r == 1))
            prev_c, prev_r = ch, r
            cs = cs if rng.random() < (p_cor_all if r == 1 else p_err_all) else -cs
    a = np.array(rows, float)
    fit = sm.Logit(a[:, 0], sm.add_constant(a[:, 1:4])).fit(disp=0)
    return fit.params[1], fit.pvalues[1], a[:, 4].mean()


sim_rows = []
for r in reg[reg.epoch == "late"].itertuples():
    sims = np.array([simulate_blind(int(r.n), r.bias, r.prev_choice_w, r.wsls_w) for _ in range(N_SIM)])
    sim_rows.append({"mouse": r.mouse, "group": r.group, "observed_stim_weight": r.stim_weight,
                     "observed_accuracy": r.accuracy, "blind_weight_mean": sims[:, 0].mean(),
                     "blind_weight_2.5pct": np.percentile(sims[:, 0], 2.5),
                     "blind_weight_97.5pct": np.percentile(sims[:, 0], 97.5),
                     "blind_false_positive_rate": np.mean(sims[:, 1] < 0.05),
                     "blind_accuracy": sims[:, 2].mean()})
sim = pd.DataFrame(sim_rows)

# ------------------------------------------------------------------ outputs
sens.to_csv(os.path.join(HERE, "criterion_window_sensitivity.csv"), index=False)
sim.to_csv(os.path.join(HERE, "blind_agent_simulation.csv"), index=False)
crit.to_csv(os.path.join(HERE, "trials_to_criterion.csv"), index=False)
task.to_csv(os.path.join(HERE, "task_structure.csv"), index=False)
reg.to_csv(os.path.join(HERE, "stimulus_weights.csv"), index=False)

with open(os.path.join(HERE, "learning_and_chance_output.txt"), "w", encoding="utf-8") as fh:
    def say(s=""):
        print(s)
        fh.write(s + "\n")
    say("=== 1. Trials to criterion (100-trial running accuracy >= 70%) ===")
    say(crit.round(3).to_string(index=False))
    say(f"Rescued vs WT: mean {rs_c.mean():.0f} vs {wt_c.mean():.0f} trials; "
        f"exact two-sided permutation test p = {p_crit:.4f}")
    say()
    say("=== 2. Task structure (analysis windows) ===")
    say(task.round(3).to_string(index=False))
    say(f"Pooled: P(correct side repeats | error) = {p_err_all:.3f}; | correct = {p_cor_all:.3f}")
    say(f"=> a blind mouse that alternates sides would score ~{100 * ceiling:.1f}% correct; "
        f"pure side bias scores < 50%.")
    say()
    say("=== 3. History-controlled stimulus weight (logit units; >0 = choices follow the pattern) ===")
    say(reg.round(4).to_string(index=False))
    say()
    say("=== 4. Trials-to-criterion sensitivity to the running-window length ===")
    say(sens.round(4).to_string(index=False))
    say()
    say(f"=== 5. Blind-agent null for the stimulus weight ({N_SIM} simulated blind mice per real mouse, trials 401-1000) ===")
    say(sim.round(3).to_string(index=False))
