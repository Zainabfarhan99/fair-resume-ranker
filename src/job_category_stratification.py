"""
job_category_stratification.py
==============================
Does the disclosure paradox vary by JOB CATEGORY?

Per-JD analysis in Section 4.1 already hinted that the caregiving penalty
varied across JDs (microservices +0.112 vs dba_sqlserver +0.046). This script
systematically stratifies the audit by job category to test whether:
  - Bias magnitude depends on technical complexity of the role
  - Some categories of JD are uniformly more / less biased than others
  - The reason-stratified ordering holds within every category

OUTPUTS:
    data/processed/audit/category_stratification.csv
    outputs/figures/fig8_category_stratification.png
    outputs/14_category_stratification_log.txt

Run from repo root:
    python src/job_category_stratification.py
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
RESULTS_CSV = AUDIT_DIR / "category_stratification.csv"
FIG_PATH = FIG_DIR / "fig8_category_stratification.png"
LOG_PATH = OUT_DIR / "14_category_stratification_log.txt"

THRESHOLD = 0.5
REASONS = ["no_reason", "caregiving", "health", "layoff", "education"]

# JD category mapping
JD_CATEGORIES = {
    # JVM / Java
    "java_lead": "JVM/Java",
    "java_backend_dev": "JVM/Java",
    "java_microservices_dev": "JVM/Java",

    # Frontend
    "frontend_dev": "Frontend",
    "angular_dev": "Frontend",
    "react_dev": "Frontend",
    "fullstack_dev": "Frontend",

    # Other dev
    "python_dev": "Other dev",
    "python_data_dev": "Other dev",
    "node_dev": "Other dev",
    "dotnet_dev": "Other dev",
    "php_dev": "Other dev",

    # Data / Cloud / DevOps
    "data_engineer": "Data/Cloud/DevOps",
    "devops_engineer": "Data/Cloud/DevOps",
    "cloud_engineer": "Data/Cloud/DevOps",

    # Database
    "dba_sqlserver": "Database",
    "dba_oracle": "Database",
    "senior_dba_ops": "Database",
    "database_developer": "Database",

    # Network / Security / Admin
    "network_admin": "Network/Sec/Admin",
    "network_engineer": "Network/Sec/Admin",
    "systems_admin": "Network/Sec/Admin",
    "security_analyst": "Network/Sec/Admin",

    # Other professional (non-coding)
    "qa_engineer": "Other professional",
    "business_analyst": "Other professional",
    "scrum_pm": "Other professional",
}

CATEGORY_ORDER = [
    "JVM/Java",
    "Frontend",
    "Other dev",
    "Data/Cloud/DevOps",
    "Database",
    "Network/Sec/Admin",
    "Other professional",
]

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

# verify category mapping covers all jd_ids
jd_ids_in_data = preds["jd_id"].unique()
missing = set(jd_ids_in_data) - set(JD_CATEGORIES.keys())
extra = set(JD_CATEGORIES.keys()) - set(jd_ids_in_data)
print(f"  jd_ids in data: {len(jd_ids_in_data)}")
print(f"  jd_ids in mapping: {len(JD_CATEGORIES)}")
if missing:
    print(f"  WARNING: jd_ids in data but not in mapping: {sorted(missing)}")
if extra:
    print(f"  WARNING: jd_ids in mapping but not in data: {sorted(extra)}")

preds["category"] = preds["jd_id"].map(JD_CATEGORIES)
if preds["category"].isna().any():
    print(f"  ERROR: some predictions have no category assigned")
    sys.exit(1)

pivoted = preds.pivot_table(
    index=["person_id", "jd_id", "category", "label"],
    columns="variant_type",
    values="predicted_proba",
    aggfunc="first",
).reset_index()
print(f"  pivoted to {len(pivoted):,} (person × JD) rows in {time.time()-t0:.1f}s")

audit_cohort = pivoted[pivoted["label"] == 1].copy()
print(f"  audit-relevant pairs (label=1): {len(audit_cohort):,}")

print(f"\n  audit-relevant pairs per category:")
for cat in CATEGORY_ORDER:
    n = (audit_cohort["category"] == cat).sum()
    print(f"    {cat:<22s}  {n:>5,}")

# -----------------------------------------------------------------------------
# Step 2: per-category metrics
# -----------------------------------------------------------------------------
section("STEP 2: compute EOD per category")

records = []
print(f"\n  EOD by category × reason:\n")
print(f"  {'category':<22s} | " + " | ".join(f"{r:>10s}" for r in REASONS))
print(f"  {'-' * 22}-+-" + "-+-".join("-" * 10 for _ in REASONS))

for cat in CATEGORY_ORDER:
    sub = audit_cohort[audit_cohort["category"] == cat]
    if len(sub) == 0:
        continue
    n_pairs = len(sub)
    ctrl_tpr = (sub["control"] >= THRESHOLD).mean()
    n_ctrl_match = int((sub["control"] >= THRESHOLD).sum())

    eods = {}
    flip_rates = {}
    for r in REASONS:
        var_tpr = (sub[r] >= THRESHOLD).mean()
        eod = ctrl_tpr - var_tpr
        eods[r] = eod

        flippable = sub["control"] >= THRESHOLD
        flipped = flippable & (sub[r] < THRESHOLD)
        flip_rate = flipped.sum() / flippable.sum() if flippable.sum() > 0 else np.nan
        flip_rates[r] = flip_rate

        records.append({
            "category": cat,
            "reason": r,
            "n_pairs": n_pairs,
            "n_control_match": n_ctrl_match,
            "ctrl_tpr": round(float(ctrl_tpr), 6),
            "var_tpr": round(float(var_tpr), 6),
            "eod": round(float(eod), 6),
            "flip_rate": round(float(flip_rate), 6),
        })

    row_str = " | ".join(f"{eods[r]:>+10.4f}" for r in REASONS)
    print(f"  {cat:<22s} | {row_str}")

# -----------------------------------------------------------------------------
# Step 3: flip rates
# -----------------------------------------------------------------------------
section("STEP 3: flip rate by category × reason")
print(f"\n  Flip rate (%) — qualified candidates rejected after disclosure:\n")
print(f"  {'category':<22s} | " + " | ".join(f"{r:>10s}" for r in REASONS))
print(f"  {'-' * 22}-+-" + "-+-".join("-" * 10 for _ in REASONS))
for cat in CATEGORY_ORDER:
    sub_records = [r for r in records if r["category"] == cat]
    if not sub_records:
        continue
    row_str = " | ".join(
        f"{next((r['flip_rate'] for r in sub_records if r['reason'] == reason), float('nan')) * 100:>9.1f}%"
        for reason in REASONS
    )
    print(f"  {cat:<22s} | {row_str}")

# -----------------------------------------------------------------------------
# Step 4: ordering preservation test
# -----------------------------------------------------------------------------
section("STEP 4: is reason-ordering preserved within each category?")

print(f"\n  Ranking by EOD (worst → best penalty) per category:\n")
print(f"  {'category':<22s}  ranking")
print(f"  {'-' * 22}  {'-' * 60}")
preserved = 0
results_df = pd.DataFrame(records)
for cat in CATEGORY_ORDER:
    sub = results_df[results_df["category"] == cat].sort_values("eod", ascending=False)
    if len(sub) == 0:
        continue
    ranking = sub["reason"].tolist()
    top3 = ranking[:3]
    top3_expected = {"education", "caregiving", "layoff"}
    is_preserved = (ranking[0] == "education") and (set(top3) == top3_expected)
    if is_preserved:
        preserved += 1
    marker = "✓" if is_preserved else "✗"
    print(f"  {cat:<22s}  {' > '.join(ranking)} [{marker}]")

print(f"\n  Ordering preserved (education first, top 3 = expected set) in {preserved}/{len(CATEGORY_ORDER)} categories.")

# -----------------------------------------------------------------------------
# Step 5: which categories are most/least biased?
# -----------------------------------------------------------------------------
section("STEP 5: rank categories by overall bias")

print(f"\n  Mean |EOD| across the 3 penalised reasons (caregiving, layoff, education):\n")
print(f"  {'category':<22s}  {'mean |EOD|':>12s}  {'education EOD':>14s}  {'caregiving EOD':>16s}  {'layoff EOD':>12s}")
print(f"  {'-' * 22}  {'-' * 12}  {'-' * 14}  {'-' * 16}  {'-' * 12}")

cat_summary = []
for cat in CATEGORY_ORDER:
    sub = results_df[(results_df["category"] == cat) & 
                     (results_df["reason"].isin(["caregiving", "layoff", "education"]))]
    if len(sub) == 0:
        continue
    mean_abs_eod = sub["eod"].abs().mean()
    edu_eod = results_df[(results_df["category"] == cat) & (results_df["reason"] == "education")]["eod"].iloc[0]
    care_eod = results_df[(results_df["category"] == cat) & (results_df["reason"] == "caregiving")]["eod"].iloc[0]
    layoff_eod = results_df[(results_df["category"] == cat) & (results_df["reason"] == "layoff")]["eod"].iloc[0]
    cat_summary.append({
        "category": cat,
        "mean_abs_eod": mean_abs_eod,
        "education_eod": edu_eod,
        "caregiving_eod": care_eod,
        "layoff_eod": layoff_eod,
    })

cat_summary_df = pd.DataFrame(cat_summary).sort_values("mean_abs_eod", ascending=False)
for _, row in cat_summary_df.iterrows():
    print(f"  {row['category']:<22s}  {row['mean_abs_eod']:>+12.4f}  "
          f"{row['education_eod']:>+14.4f}  {row['caregiving_eod']:>+16.4f}  {row['layoff_eod']:>+12.4f}")

most_biased = cat_summary_df.iloc[0]["category"]
least_biased = cat_summary_df.iloc[-1]["category"]
ratio = cat_summary_df.iloc[0]["mean_abs_eod"] / max(cat_summary_df.iloc[-1]["mean_abs_eod"], 1e-6)
print(f"\n  Most biased category:  {most_biased}")
print(f"  Least biased category: {least_biased}")
print(f"  Bias ratio (most / least): {ratio:.2f}x")

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
    "font.size": 10,
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

# Grouped bar chart: x-axis = category, bars = reasons
fig, ax = plt.subplots(figsize=(13, 6))

n_categories = len(CATEGORY_ORDER)
n_reasons = len(REASONS)
x = np.arange(n_categories)
width = 0.16

for i, r in enumerate(REASONS):
    eods = []
    for cat in CATEGORY_ORDER:
        rec = results_df[(results_df["category"] == cat) & (results_df["reason"] == r)]
        eods.append(rec["eod"].iloc[0] if len(rec) else 0)
    offset = (i - n_reasons/2 + 0.5) * width
    ax.bar(x + offset, eods, width, label=r, color=COLOURS[r], alpha=0.88,
           edgecolor="white", linewidth=0.5)

ax.axhline(0, color="grey", linewidth=0.7, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(CATEGORY_ORDER, rotation=20, ha="right")
ax.set_ylabel("Equalised Odds Difference (EOD) vs. control")
ax.set_xlabel("Job category")
ax.set_title(
    "Reason-stratified bias by job category\n"
    "(positive = qualified variants rejected more than control)",
    fontsize=11, pad=12,
)
ax.legend(loc="upper right", framealpha=0.95, ncol=5)
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