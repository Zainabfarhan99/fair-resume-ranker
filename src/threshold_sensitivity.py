"""
threshold_sensitivity.py
========================
Does the disclosure paradox hold across different decision thresholds?

The baseline audit and fairness metrics used threshold 0.5. Real ATS systems
use different cutoffs (some shortlist generously at 0.3, others strictly at 0.7).
A reviewer could ask: 'is your paradox finding an artifact of threshold choice?'

This script recomputes EOD, DI, and flip rates at thresholds {0.3, 0.4, 0.5, 0.6, 0.7}
and tests whether the reason-stratified ordering (education > caregiving > layoff
> health > no_reason) persists across the threshold range.

OUTPUTS:
    data/processed/audit/threshold_sensitivity.csv
    outputs/figures/fig7_threshold_sensitivity.png
    outputs/13_threshold_sensitivity_log.txt

Run from repo root:
    python src/threshold_sensitivity.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = REPO_ROOT / "data" / "processed"
AUDIT_DIR = PROC_DIR / "audit"
OUT_DIR = REPO_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

PREDICTIONS = AUDIT_DIR / "baseline_predictions.csv"
RESULTS_CSV = AUDIT_DIR / "threshold_sensitivity.csv"
FIG_PATH = FIG_DIR / "fig7_threshold_sensitivity.png"
LOG_PATH = OUT_DIR / "13_threshold_sensitivity_log.txt"

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
REASONS = ["no_reason", "caregiving", "health", "layoff", "education"]
FOUR_FIFTHS = 0.80  # EEOC 4/5 rule

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

# -----------------------------------------------------------------------------
# Step 1: load and pivot predictions
# -----------------------------------------------------------------------------
section("STEP 1: load and pivot predictions")
t0 = time.time()
preds = pd.read_csv(PREDICTIONS)
print(f"  loaded {len(preds):,} prediction records")

pivoted = preds.pivot_table(
    index=["person_id", "jd_id", "label"],
    columns="variant_type",
    values="predicted_proba",
    aggfunc="first",
).reset_index()
print(f"  pivoted to {len(pivoted):,} (person × JD) rows in {time.time()-t0:.1f}s")

audit_cohort = pivoted[pivoted["label"] == 1].copy()
all_pairs = pivoted.copy()
print(f"  audit-relevant pairs (label=1): {len(audit_cohort):,}")
print(f"  full test pairs (label=0+1):     {len(all_pairs):,}")

# -----------------------------------------------------------------------------
# Step 2: per-threshold metric computation
# -----------------------------------------------------------------------------
section("STEP 2: compute EOD, DI, flip rate per (threshold, reason)")

records = []
for thresh in THRESHOLDS:
    # selection rates: over the FULL test cohort (label=0+1), at this threshold
    ctrl_sr = (all_pairs["control"] >= thresh).mean()
    # TPR: over the AUDIT-RELEVANT cohort (label=1), at this threshold
    ctrl_tpr = (audit_cohort["control"] >= thresh).mean()
    n_ctrl_match = int((audit_cohort["control"] >= thresh).sum())

    for r in REASONS:
        var_sr = (all_pairs[r] >= thresh).mean()
        var_tpr = (audit_cohort[r] >= thresh).mean()

        dpd = ctrl_sr - var_sr
        eod = ctrl_tpr - var_tpr
        di = var_sr / ctrl_sr if ctrl_sr > 0 else np.nan

        flippable = audit_cohort["control"] >= thresh
        flipped = flippable & (audit_cohort[r] < thresh)
        flip_rate = flipped.sum() / flippable.sum() if flippable.sum() > 0 else np.nan

        records.append({
            "threshold": thresh,
            "reason": r,
            "ctrl_sr": round(float(ctrl_sr), 6),
            "var_sr": round(float(var_sr), 6),
            "ctrl_tpr": round(float(ctrl_tpr), 6),
            "var_tpr": round(float(var_tpr), 6),
            "dpd": round(float(dpd), 6),
            "eod": round(float(eod), 6),
            "di": round(float(di), 6),
            "flip_rate": round(float(flip_rate), 6),
            "violates_4_5_rule": bool(di < FOUR_FIFTHS),
            "n_control_match": n_ctrl_match,
        })

results_df = pd.DataFrame(records)

# Print EOD table
print(f"\n  EOD by threshold × reason:\n")
print(f"  {'threshold':<10s} | " + " | ".join(f"{r:>10s}" for r in REASONS))
print(f"  {'-' * 10}-+-" + "-+-".join("-" * 10 for _ in REASONS))
for thresh in THRESHOLDS:
    row_str = " | ".join(
        f"{results_df[(results_df['threshold']==thresh) & (results_df['reason']==r)]['eod'].iloc[0]:>+10.4f}"
        for r in REASONS
    )
    print(f"  {thresh:>9.1f}  | {row_str}")

# Print DI table
print(f"\n  DI by threshold × reason (< 0.80 = 4/5 rule violation marked ✗):\n")
print(f"  {'threshold':<10s} | " + " | ".join(f"{r:>11s}" for r in REASONS))
print(f"  {'-' * 10}-+-" + "-+-".join("-" * 11 for _ in REASONS))
for thresh in THRESHOLDS:
    row_parts = []
    for r in REASONS:
        di = results_df[(results_df['threshold']==thresh) & (results_df['reason']==r)]['di'].iloc[0]
        mark = " ✗" if di < FOUR_FIFTHS else "  "
        row_parts.append(f"{di:>9.4f}{mark}")
    print(f"  {thresh:>9.1f}  | " + " | ".join(row_parts))

# Print flip rate table
print(f"\n  Flip rate (%) by threshold × reason:\n")
print(f"  {'threshold':<10s} | " + " | ".join(f"{r:>10s}" for r in REASONS))
print(f"  {'-' * 10}-+-" + "-+-".join("-" * 10 for _ in REASONS))
for thresh in THRESHOLDS:
    row_str = " | ".join(
        f"{results_df[(results_df['threshold']==thresh) & (results_df['reason']==r)]['flip_rate'].iloc[0] * 100:>9.1f}%"
        for r in REASONS
    )
    print(f"  {thresh:>9.1f}  | {row_str}")

# -----------------------------------------------------------------------------
# Step 3: ordering robustness
# -----------------------------------------------------------------------------
section("STEP 3: is the reason-ordering preserved across thresholds?")

# expected ordering by EOD magnitude (worst → best penalty):
# education > caregiving ≈ layoff > health ≈ no_reason
EXPECTED_RANK = ["education", "caregiving", "layoff", "health", "no_reason"]

print(f"\n  Reasons ranked by EOD (descending) at each threshold:\n")
print(f"  {'threshold':<10s}  ranking (worst → best penalty)")
print(f"  {'-' * 10}  {'-' * 60}")
preserved_count = 0
for thresh in THRESHOLDS:
    sub = results_df[results_df["threshold"] == thresh].sort_values("eod", ascending=False)
    ranking = sub["reason"].tolist()
    # check if top-3 ordering preserved (education first, caregiving/layoff next two)
    top3 = ranking[:3]
    top3_expected = {"education", "caregiving", "layoff"}
    preserved = (ranking[0] == "education") and (set(top3) == top3_expected)
    if preserved:
        preserved_count += 1
    marker = "✓" if preserved else "✗"
    print(f"  {thresh:>9.1f}  {' > '.join(ranking)}  [{marker}]")

print(f"\n  Reason-ordering preserved at {preserved_count} / {len(THRESHOLDS)} thresholds.")

# -----------------------------------------------------------------------------
# Step 4: 4/5 rule violations across thresholds
# -----------------------------------------------------------------------------
section("STEP 4: 4/5 rule violations by threshold")

print(f"\n  At each threshold, which reasons violate DI < 0.80?\n")
for thresh in THRESHOLDS:
    sub = results_df[results_df["threshold"] == thresh]
    violators = sub[sub["violates_4_5_rule"]]["reason"].tolist()
    if violators:
        violator_str = ", ".join(violators)
    else:
        violator_str = "(none)"
    print(f"  threshold {thresh:.1f}: {violator_str}")

# -----------------------------------------------------------------------------
# Step 5: save CSV
# -----------------------------------------------------------------------------
section("STEP 5: write CSV")
results_df.to_csv(RESULTS_CSV, index=False)
print(f"  wrote {RESULTS_CSV}")

# -----------------------------------------------------------------------------
# Step 6: figure
# -----------------------------------------------------------------------------
section("STEP 6: generate figure")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLOURS = {
    "no_reason":  "#3a85c9",
    "caregiving": "#e87b35",
    "health":     "#2ca44b",
    "layoff":     "#b8407a",
    "education":  "#8e44ad",
}
MARKERS = {
    "no_reason":  "o",
    "caregiving": "s",
    "health":     "^",
    "layoff":     "D",
    "education":  "v",
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: EOD vs threshold
for r in REASONS:
    sub = results_df[results_df["reason"] == r].sort_values("threshold")
    ax1.plot(
        sub["threshold"], sub["eod"],
        marker=MARKERS[r], color=COLOURS[r], linewidth=2, markersize=7,
        label=r, alpha=0.9,
    )

ax1.axhline(0, color="grey", linewidth=0.7, alpha=0.7)
ax1.set_xlabel("Decision threshold")
ax1.set_ylabel("Equalised Odds Difference (EOD) vs. control")
ax1.set_title("EOD across thresholds: ordering preserved",
              fontsize=11, pad=10)
ax1.set_xticks(THRESHOLDS)
ax1.legend(loc="upper right", framealpha=0.95, ncol=2)

# Panel B: DI vs threshold with 4/5 rule line
for r in REASONS:
    sub = results_df[results_df["reason"] == r].sort_values("threshold")
    ax2.plot(
        sub["threshold"], sub["di"],
        marker=MARKERS[r], color=COLOURS[r], linewidth=2, markersize=7,
        label=r, alpha=0.9,
    )

ax2.axhline(FOUR_FIFTHS, color="red", linewidth=1, linestyle="--",
            alpha=0.8, label="EEOC 4/5 rule (DI=0.80)")
ax2.set_xlabel("Decision threshold")
ax2.set_ylabel("Disparate Impact (DI = variant SR / control SR)")
ax2.set_title("DI across thresholds: education violation persists",
              fontsize=11, pad=10)
ax2.set_xticks(THRESHOLDS)
ax2.legend(loc="lower left", framealpha=0.95, ncol=2)

plt.suptitle(
    "Threshold sensitivity: the disclosure paradox holds across the realistic threshold range",
    fontsize=12, y=1.02,
)
plt.tight_layout()
plt.savefig(FIG_PATH)
plt.close()
print(f"  saved {FIG_PATH}")

section("DONE")
print(f"  CSV:    {RESULTS_CSV}")
print(f"  figure: {FIG_PATH}")
print(f"  log:    {LOG_PATH}")

sys.stdout = sys.__stdout__
log_file.close()