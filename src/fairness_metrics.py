"""
fairness_metrics.py
===================
Compute standard fairness metrics (DPD, EOD, DI) on the baseline ranker's
test-set predictions, with gap_reason as the protected attribute.

For each gap-reason variant we compute three metrics, comparing to the
control variant of the same person on the same JD:

    DPD (Demographic Parity Difference)
        = SR(control) - SR(variant)
        where SR is selection rate at threshold 0.5.
        Positive DPD = variant is selected less often than control = bias against variant.

    EOD (Equalized Odds Difference)
        = TPR(control on label=1) - TPR(variant on label=1)
        Positive EOD = qualified variant candidates get rejected more than qualified control = bias.
        This is the most policy-relevant metric — measures fairness AMONG QUALIFIED candidates.

    DI (Disparate Impact ratio)
        = SR(variant) / SR(control)
        US EEOC "four-fifths rule" treats DI < 0.80 as evidence of discrimination.
        DI = 1.0 means perfectly equal selection. DI < 0.80 is the legal red flag.

We also report bootstrapped 95% confidence intervals to indicate statistical
reliability of each metric.

OUTPUTS:
    data/processed/audit/baseline_fairness_metrics.csv
    outputs/08_fairness_metrics_log.txt

Run from repo root:
    python src/fairness_metrics.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DECISION_THRESHOLD = 0.5
BOOTSTRAP_N = 1000
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = REPO_ROOT / "data" / "processed"
AUDIT_DIR = PROC_DIR / "audit"
OUT_DIR = REPO_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

PRED_PATH = AUDIT_DIR / "baseline_predictions.csv"
METRICS_OUT = AUDIT_DIR / "baseline_fairness_metrics.csv"
LOG_PATH = OUT_DIR / "08_fairness_metrics_log.txt"

class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, msg):
        for s in self.streams:
            try: s.write(msg)
            except Exception: pass
    def flush(self):
        for s in self.streams:
            try: s.flush()
            except Exception: pass

log_file = open(LOG_PATH, "w")
sys.stdout = Tee(sys.__stdout__, log_file)

def section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

VARIANT_TYPES = ["no_reason", "caregiving", "health", "layoff", "education"]
rng = np.random.default_rng(RANDOM_SEED)

# -----------------------------------------------------------------------------
# Step 1: load
# -----------------------------------------------------------------------------
section("STEP 1: load baseline predictions")
preds = pd.read_csv(PRED_PATH)
print(f"  loaded {len(preds):,} prediction records")

# pivot so each row is (person_id, jd_id, label) with one column per variant
pivot = preds.pivot_table(
    index=["person_id", "jd_id", "label"],
    columns="variant_type",
    values="predicted_proba",
    aggfunc="first",
).reset_index()
print(f"  pivoted to {len(pivot):,} (person, JD) rows")

# binary decisions at threshold
for vt in ["control"] + VARIANT_TYPES:
    pivot[f"{vt}_pred"] = (pivot[vt] >= DECISION_THRESHOLD).astype(int)

# -----------------------------------------------------------------------------
# Step 2: metric computation helpers
# -----------------------------------------------------------------------------
def selection_rate(pred_array):
    """Fraction of predictions == 1."""
    return float(np.mean(pred_array)) if len(pred_array) else 0.0

def tpr(pred_array, label_array):
    """True positive rate: of true label=1, fraction predicted 1."""
    pos = label_array == 1
    if pos.sum() == 0: return 0.0
    return float(np.mean(pred_array[pos]))

def compute_metrics(ctrl_pred, var_pred, label):
    sr_ctrl = selection_rate(ctrl_pred)
    sr_var = selection_rate(var_pred)
    tpr_ctrl = tpr(ctrl_pred, label)
    tpr_var = tpr(var_pred, label)
    dpd = sr_ctrl - sr_var
    eod = tpr_ctrl - tpr_var
    di = sr_var / sr_ctrl if sr_ctrl > 0 else float("nan")
    return {
        "sr_control": sr_ctrl,
        "sr_variant": sr_var,
        "tpr_control": tpr_ctrl,
        "tpr_variant": tpr_var,
        "DPD": dpd,
        "EOD": eod,
        "DI": di,
    }

def bootstrap_ci(ctrl_pred, var_pred, label, metric_func, n=BOOTSTRAP_N, seed=42):
    """95% CI via paired bootstrap over (person, JD) pairs."""
    rng_local = np.random.default_rng(seed)
    n_total = len(ctrl_pred)
    estimates = []
    for _ in range(n):
        idx = rng_local.integers(0, n_total, size=n_total)
        m = metric_func(ctrl_pred[idx], var_pred[idx], label[idx])
        estimates.append(m)
    estimates = np.array(estimates)
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))

# -----------------------------------------------------------------------------
# Step 3: compute per-variant metrics
# -----------------------------------------------------------------------------
section(f"STEP 3: per-variant fairness metrics (threshold={DECISION_THRESHOLD})")

ctrl_pred = pivot["control_pred"].to_numpy()
label = pivot["label"].to_numpy()

results = []
for vt in VARIANT_TYPES:
    var_pred = pivot[f"{vt}_pred"].to_numpy()
    m = compute_metrics(ctrl_pred, var_pred, label)

    # bootstrap CIs for DPD and EOD
    dpd_lo, dpd_hi = bootstrap_ci(
        ctrl_pred, var_pred, label,
        lambda c, v, l: selection_rate(c) - selection_rate(v),
        seed=42,
    )
    eod_lo, eod_hi = bootstrap_ci(
        ctrl_pred, var_pred, label,
        lambda c, v, l: tpr(c, l) - tpr(v, l),
        seed=43,
    )

    m["DPD_ci_low"] = dpd_lo
    m["DPD_ci_high"] = dpd_hi
    m["EOD_ci_low"] = eod_lo
    m["EOD_ci_high"] = eod_hi
    m["variant_type"] = vt
    results.append(m)

results_df = pd.DataFrame(results)[
    ["variant_type", "sr_control", "sr_variant", "DPD", "DPD_ci_low", "DPD_ci_high",
     "tpr_control", "tpr_variant", "EOD", "EOD_ci_low", "EOD_ci_high", "DI"]
]
results_df = results_df.round(4)

# -----------------------------------------------------------------------------
# Step 4: print headline
# -----------------------------------------------------------------------------
section("STEP 4: HEADLINE TABLE")
print(f"\n  Selection rates (fraction selected at threshold {DECISION_THRESHOLD}):")
print(f"    control: {results_df['sr_control'].iloc[0]:.4f}\n")

print(f"  Per-variant fairness metrics (positive DPD/EOD = bias against variant):")
print(f"  {'variant':<12s} {'SR':>7s} {'DPD':>9s} {'DPD 95% CI':>20s} {'EOD':>9s} {'EOD 95% CI':>20s} {'DI':>7s}")
for _, r in results_df.iterrows():
    print(f"  {r['variant_type']:<12s} "
          f"{r['sr_variant']:>7.4f} "
          f"{r['DPD']:>+9.4f} "
          f"[{r['DPD_ci_low']:>+7.4f}, {r['DPD_ci_high']:>+7.4f}]  "
          f"{r['EOD']:>+9.4f} "
          f"[{r['EOD_ci_low']:>+7.4f}, {r['EOD_ci_high']:>+7.4f}]  "
          f"{r['DI']:>7.4f}")

# Disparate impact: which variants violate the four-fifths rule?
print(f"\n  Disparate Impact (DI) — '4/5 rule' threshold = 0.80:")
for _, r in results_df.iterrows():
    flag = "VIOLATES" if r["DI"] < 0.80 else "OK"
    print(f"    {r['variant_type']:<12s}  DI = {r['DI']:.4f}  {flag}")

# -----------------------------------------------------------------------------
# Step 5: interpretation
# -----------------------------------------------------------------------------
section("STEP 5: interpretation")
print(f"\n  DPD = how much LESS often a variant is selected vs control.")
print(f"  EOD = how much LESS often a QUALIFIED variant is recommended vs control.")
print(f"  DI  = ratio of selection rates. <0.80 = legal red flag.\n")

# rank variants by EOD (most policy-relevant metric)
ranked = results_df.sort_values("EOD", ascending=False)
print(f"  Variants ranked by EOD (most penalized first):")
for _, r in ranked.iterrows():
    sig = "*" if (r["EOD_ci_low"] > 0 or r["EOD_ci_high"] < 0) else " "
    print(f"    {sig} {r['variant_type']:<12s}  EOD={r['EOD']:>+.4f}  "
          f"95% CI: [{r['EOD_ci_low']:>+.4f}, {r['EOD_ci_high']:>+.4f}]")
print(f"\n  '*' = 95% CI excludes zero (statistically significant bias).")

# -----------------------------------------------------------------------------
# Step 6: write output CSV
# -----------------------------------------------------------------------------
section("STEP 6: save")
results_df.to_csv(METRICS_OUT, index=False)
print(f"  wrote {METRICS_OUT}")

section("DONE")
print(f"  metrics: {METRICS_OUT}")
print(f"  log:     {LOG_PATH}")
print(f"\n  next: figures.")

sys.stdout = sys.__stdout__
log_file.close()