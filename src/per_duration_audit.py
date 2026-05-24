"""
per_duration_audit.py
=====================
Does the disclosure paradox depend on gap DURATION, or is it purely reason-driven?

We have baseline predictions for all (variant × JD) pairs already. Each candidate
has one gap_duration in {6, 12, 18, 24, 36, 48, 60} months, applied identically
across the 5 reason variants for that candidate. We now stratify the audit by
duration and ask: does EOD scale with duration, or is it stable across durations?

OUTPUTS:
    data/processed/audit/per_duration_metrics.csv
    outputs/figures/fig6_eod_by_duration.png
    outputs/12_per_duration_audit_log.txt

Run from repo root:
    python src/per_duration_audit.py
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
RESULTS_CSV = AUDIT_DIR / "per_duration_metrics.csv"
FIG_PATH = FIG_DIR / "fig6_eod_by_duration.png"
LOG_PATH = OUT_DIR / "12_per_duration_audit_log.txt"

DURATIONS = [6, 12, 18, 24, 36, 48, 60]
REASONS = ["no_reason", "caregiving", "health", "layoff", "education"]
THRESHOLD = 0.5

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
# Step 1: load predictions
# -----------------------------------------------------------------------------
section("STEP 1: load baseline predictions")
t0 = time.time()
preds = pd.read_csv(PREDICTIONS)
print(f"  loaded {len(preds):,} prediction records in {time.time()-t0:.1f}s")
print(f"  columns: {list(preds.columns)}")

# -----------------------------------------------------------------------------
# Step 2: pivot to wide format so each (person, JD) row has all 6 variants
# -----------------------------------------------------------------------------
section("STEP 2: pivot predictions to wide format")
# Each candidate has ONE gap_duration applied to all 5 reason variants.
# Build (person, JD) → duration map first (using any non-control variant).
non_ctrl = preds[preds["variant_type"] != "control"].copy()
duration_map = non_ctrl.groupby("person_id")["gap_duration_months"].first()
print(f"  built duration map for {len(duration_map):,} test people")

pivoted = preds.pivot_table(
    index=["person_id", "jd_id", "label"],
    columns="variant_type",
    values="predicted_proba",
    aggfunc="first",
).reset_index()
pivoted["gap_duration_months"] = pivoted["person_id"].map(duration_map).astype(int)
print(f"  pivoted to {len(pivoted):,} (person × JD) rows")
print(f"  duration distribution in pivoted table:")
print(pivoted.drop_duplicates("person_id")["gap_duration_months"].value_counts().sort_index().to_string())

# Audit-relevant cohort: label = 1
audit_cohort = pivoted[pivoted["label"] == 1].copy()
print(f"\n  audit-relevant (label=1) pairs: {len(audit_cohort):,}")

# -----------------------------------------------------------------------------
# Step 3: compute per-duration metrics
# -----------------------------------------------------------------------------
section("STEP 3: compute EOD and flip rate per (duration, reason)")

records = []

print(f"\n  EOD by duration × reason (positive = bias against variant):\n")
print(f"  {'duration':<10s} | " + " | ".join(f"{r:>10s}" for r in REASONS))
print(f"  {'-' * 10}-+-" + "-+-".join("-" * 10 for _ in REASONS))

for duration in DURATIONS:
    sub = audit_cohort[audit_cohort["gap_duration_months"] == duration]
    if len(sub) == 0:
        continue
    n_pairs = len(sub)

    # EOD per variant: TPR_control - TPR_variant (on label=1 cases only)
    ctrl_tpr = (sub["control"] >= THRESHOLD).mean()
    eods = {}
    flip_rates = {}
    n_ctrl_match = (sub["control"] >= THRESHOLD).sum()
    for r in REASONS:
        var_tpr = (sub[r] >= THRESHOLD).mean()
        eod = ctrl_tpr - var_tpr
        eods[r] = eod
        # flip rate: of cases where control matched, fraction where variant didn't
        flippable = sub["control"] >= THRESHOLD
        flipped = flippable & (sub[r] < THRESHOLD)
        flip_rate = flipped.sum() / flippable.sum() if flippable.sum() > 0 else np.nan
        flip_rates[r] = flip_rate

        records.append({
            "duration_months": duration,
            "reason": r,
            "n_pairs": n_pairs,
            "n_control_match": int(n_ctrl_match),
            "ctrl_tpr": round(float(ctrl_tpr), 6),
            "var_tpr": round(float(var_tpr), 6),
            "eod": round(float(eod), 6),
            "flip_rate": round(float(flip_rate), 6),
        })

    row_str = " | ".join(f"{eods[r]:>+10.4f}" for r in REASONS)
    print(f"  {duration:>3d} mo     | {row_str}")

# -----------------------------------------------------------------------------
# Step 4: flip rate by duration × reason
# -----------------------------------------------------------------------------
section("STEP 4: flip rate by duration × reason")
print(f"\n  Flip rate (%) by duration × reason:\n")
print(f"  {'duration':<10s} | " + " | ".join(f"{r:>10s}" for r in REASONS))
print(f"  {'-' * 10}-+-" + "-+-".join("-" * 10 for _ in REASONS))
for duration in DURATIONS:
    sub_records = [r for r in records if r["duration_months"] == duration]
    if not sub_records:
        continue
    row_str = " | ".join(
        f"{next((r['flip_rate'] for r in sub_records if r['reason'] == reason), float('nan')) * 100:>9.1f}%"
        for reason in REASONS
    )
    print(f"  {duration:>3d} mo     | {row_str}")

# -----------------------------------------------------------------------------
# Step 5: linear regression — does EOD scale with duration per reason?
# -----------------------------------------------------------------------------
section("STEP 5: does EOD scale with duration?")

print(f"\n  Per-reason linear fit: EOD = a + b * duration\n")
print(f"  {'reason':<12s}  {'slope (b)':>12s}  {'intercept (a)':>14s}  {'R^2':>8s}  {'interpretation':<40s}")

results_df = pd.DataFrame(records)
slopes = {}
for r in REASONS:
    sub = results_df[results_df["reason"] == r].sort_values("duration_months")
    durations = sub["duration_months"].to_numpy(dtype=float)
    eods = sub["eod"].to_numpy(dtype=float)
    if len(durations) < 3:
        continue
    # linear fit
    slope, intercept = np.polyfit(durations, eods, 1)
    # R²
    pred = slope * durations + intercept
    ss_res = ((eods - pred) ** 2).sum()
    ss_tot = ((eods - eods.mean()) ** 2).sum()
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    slopes[r] = slope

    # interpretation
    slope_per_year = slope * 12
    if abs(slope_per_year) < 0.01:
        interp = "essentially flat (reason-driven, not duration)"
    elif slope_per_year > 0.01:
        interp = f"EOD grows by {slope_per_year:+.4f} per year of gap"
    else:
        interp = f"EOD shrinks by {slope_per_year:+.4f} per year of gap"

    print(f"  {r:<12s}  {slope:>+12.5f}  {intercept:>+14.4f}  {r_sq:>8.3f}  {interp:<40s}")

# -----------------------------------------------------------------------------
# Step 6: save CSV
# -----------------------------------------------------------------------------
section("STEP 6: write CSV")
results_df.to_csv(RESULTS_CSV, index=False)
print(f"  wrote {RESULTS_CSV}")

# -----------------------------------------------------------------------------
# Step 7: figure
# -----------------------------------------------------------------------------
section("STEP 7: generate figure")

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

fig, ax = plt.subplots(figsize=(8.5, 5.5))

for r in REASONS:
    sub = results_df[results_df["reason"] == r].sort_values("duration_months")
    ax.plot(
        sub["duration_months"], sub["eod"],
        marker=MARKERS[r], color=COLOURS[r], linewidth=2, markersize=7,
        label=r, alpha=0.9,
    )

ax.axhline(0, color="grey", linewidth=0.7, alpha=0.7)
ax.set_xlabel("Gap duration (months)")
ax.set_ylabel("Equalised Odds Difference (EOD) vs. control")
ax.set_title(
    "EOD by gap duration: is the disclosure paradox duration-driven or reason-driven?\n"
    "(positive = qualified variants rejected more than control)",
    fontsize=11, pad=12,
)
ax.set_xticks(DURATIONS)
ax.legend(loc="upper left", framealpha=0.95, ncol=2)
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