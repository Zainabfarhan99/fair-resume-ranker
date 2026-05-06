"""
make_baseline_figures.py
========================
Generate the four figures for the baseline-audit section of the paper.

Figures:
  fig1_eod_with_ci.png        -- EOD per variant + 95% CIs (the headline visual)
  fig2_selection_rate_di.png  -- selection rate + 4/5 rule line (legal frame)
  fig3_flip_rates.png         -- binary flip rates (intuitive impact metric)
  fig4_caregiving_per_jd.png  -- caregiving penalty across JDs (heterogeneity)

OUTPUT:
    outputs/figures/fig1_eod_with_ci.png        (300 dpi)
    outputs/figures/fig2_selection_rate_di.png
    outputs/figures/fig3_flip_rates.png
    outputs/figures/fig4_caregiving_per_jd.png

Run from repo root:
    python src/make_baseline_figures.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = REPO_ROOT / "data" / "processed"
AUDIT_DIR = PROC_DIR / "audit"
FIG_DIR = REPO_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

PRED_PATH = AUDIT_DIR / "baseline_predictions.csv"
METRICS_PATH = AUDIT_DIR / "baseline_fairness_metrics.csv"

# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------
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

# Colour scheme: distinct hue per gap reason; control in muted grey
COLOURS = {
    "control":    "#9b9b9b",
    "no_reason":  "#3b8ed1",
    "caregiving": "#e87b35",
    "health":     "#2ca44b",
    "layoff":     "#b8407a",
    "education":  "#8e44ad",
}

VARIANT_LABELS = {
    "no_reason":  "no reason",
    "caregiving": "caregiving",
    "health":     "health",
    "layoff":     "layoff",
    "education":  "education",
}

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
metrics = pd.read_csv(METRICS_PATH)
preds = pd.read_csv(PRED_PATH)

# pivot for figures 3 and 4
pivot = preds.pivot_table(
    index=["person_id", "jd_id", "label"],
    columns="variant_type",
    values="predicted_proba",
    aggfunc="first",
).reset_index()
positive_pairs = pivot[pivot["label"] == 1].copy()

# -----------------------------------------------------------------------------
# FIGURE 1 — EOD per variant + 95% CIs
# -----------------------------------------------------------------------------
print("creating fig1_eod_with_ci.png ...")
fig, ax = plt.subplots(figsize=(7, 4.5))

variants = ["no_reason", "health", "layoff", "caregiving", "education"]
eods = metrics.set_index("variant_type").loc[variants, "EOD"].values
lo = metrics.set_index("variant_type").loc[variants, "EOD_ci_low"].values
hi = metrics.set_index("variant_type").loc[variants, "EOD_ci_high"].values
errs_lower = eods - lo
errs_upper = hi - eods
colours = [COLOURS[v] for v in variants]
labels = [VARIANT_LABELS[v] for v in variants]

bars = ax.bar(labels, eods, color=colours, alpha=0.85, edgecolor="white", linewidth=1.5)
ax.errorbar(labels, eods, yerr=[errs_lower, errs_upper], fmt="none",
            ecolor="black", capsize=4, capthick=1.2, lw=1.2)

ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
ax.set_ylabel("Equalized Odds Difference (EOD)\nvs. control variant", fontsize=11)
ax.set_title(
    "Reason-stratified bias in baseline ranker\n"
    "(positive = qualified candidates rejected more than control)",
    fontsize=12, pad=12,
)
ax.set_ylim(-0.02, max(hi) + 0.02)

# annotate values above bars
for bar, val in zip(bars, eods):
    height = bar.get_height()
    y = height + 0.003 if height >= 0 else height - 0.008
    ax.text(bar.get_x() + bar.get_width() / 2, y, f"{val:+.3f}",
            ha="center", va="bottom" if height >= 0 else "top", fontsize=9)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_eod_with_ci.png")
plt.close()
print(f"  -> {FIG_DIR / 'fig1_eod_with_ci.png'}")

# -----------------------------------------------------------------------------
# FIGURE 2 — selection rate + 4/5 rule line
# -----------------------------------------------------------------------------
print("creating fig2_selection_rate_di.png ...")
fig, ax = plt.subplots(figsize=(7, 4.5))

# include control as the reference group
sr_by_variant = {"control": metrics["sr_control"].iloc[0]}
for _, r in metrics.iterrows():
    sr_by_variant[r["variant_type"]] = r["sr_variant"]

ordered_variants = ["control", "no_reason", "health", "layoff", "caregiving", "education"]
ordered_labels = ["control"] + [VARIANT_LABELS[v] for v in ordered_variants[1:]]
ordered_colours = [COLOURS[v] for v in ordered_variants]
ordered_srs = [sr_by_variant[v] for v in ordered_variants]

bars = ax.bar(ordered_labels, ordered_srs, color=ordered_colours, alpha=0.85,
              edgecolor="white", linewidth=1.5)

# 4/5 rule reference: 80% of control's SR
sr_ctrl = sr_by_variant["control"]
threshold_80 = 0.80 * sr_ctrl
ax.axhline(threshold_80, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
           label=f"EEOC 4/5 rule threshold: 0.80 × control SR = {threshold_80:.3f}")

# annotate selection rate values
for bar, val in zip(bars, ordered_srs):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003, f"{val:.3f}",
            ha="center", va="bottom", fontsize=9)

# annotate DI for non-control variants
for _, r in metrics.iterrows():
    vt = r["variant_type"]
    di = r["DI"]
    idx = ordered_variants.index(vt)
    bar = bars[idx]
    flag = " ✗" if di < 0.80 else ""
    ax.text(bar.get_x() + bar.get_width() / 2, -0.012,
            f"DI={di:.2f}{flag}", ha="center", va="top", fontsize=8.5,
            color="red" if di < 0.80 else "#444444",
            fontweight="bold" if di < 0.80 else "normal")

ax.set_ylabel("Selection rate at threshold 0.5", fontsize=11)
ax.set_title(
    "Selection rates by variant: education violates the 4/5 rule\n"
    "(DI = selection rate ratio relative to control)",
    fontsize=12, pad=12,
)
ax.legend(loc="upper right", framealpha=0.95)
ax.set_ylim(-0.02, max(ordered_srs) * 1.18)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_selection_rate_di.png")
plt.close()
print(f"  -> {FIG_DIR / 'fig2_selection_rate_di.png'}")

# -----------------------------------------------------------------------------
# FIGURE 3 — flip rates (control predicted yes, variant predicted no)
# -----------------------------------------------------------------------------
print("creating fig3_flip_rates.png ...")
fig, ax = plt.subplots(figsize=(7, 4.5))

ctrl_pos = positive_pairs["control"] > 0.5
flip_data = []
for vt in ["no_reason", "health", "layoff", "caregiving", "education"]:
    var_neg = positive_pairs[vt] <= 0.5
    flips = (ctrl_pos & var_neg).sum()
    flippable = ctrl_pos.sum()
    rate = flips / flippable * 100 if flippable else 0
    flip_data.append((vt, flips, flippable, rate))

variants = [d[0] for d in flip_data]
labels = [VARIANT_LABELS[v] for v in variants]
rates = [d[3] for d in flip_data]
flips = [d[1] for d in flip_data]
flippable = flip_data[0][2]
colours = [COLOURS[v] for v in variants]

bars = ax.bar(labels, rates, color=colours, alpha=0.85, edgecolor="white", linewidth=1.5)
for bar, val, n in zip(bars, rates, flips):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.4,
            f"{val:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Flip rate (% of control-shortlisted candidates\nrejected after gap injection)", fontsize=11)
ax.set_title(
    f"Binary flip rates: candidates the system would have shortlisted,\n"
    f"now rejected after adding the gap (n={flippable:,} eligible cases)",
    fontsize=12, pad=12,
)
ax.set_ylim(0, max(rates) * 1.30)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_flip_rates.png")
plt.close()
print(f"  -> {FIG_DIR / 'fig3_flip_rates.png'}")

# -----------------------------------------------------------------------------
# FIGURE 4 — caregiving penalty per JD (top 12 JDs)
# -----------------------------------------------------------------------------
print("creating fig4_caregiving_per_jd.png ...")
fig, ax = plt.subplots(figsize=(8, 6))

per_jd = []
for jd_id, sub in positive_pairs.groupby("jd_id"):
    if len(sub) < 20:
        continue
    ctrl_m = sub["control"].mean()
    care_m = sub["caregiving"].mean()
    delta = ctrl_m - care_m
    per_jd.append((jd_id, len(sub), delta))
per_jd = sorted(per_jd, key=lambda r: r[2], reverse=True)[:12]

jd_ids = [r[0] for r in per_jd]
ns = [r[1] for r in per_jd]
deltas = [r[2] for r in per_jd]

# horizontal bars sorted ascending so largest is at top
y_pos = np.arange(len(jd_ids))
ax.barh(y_pos, deltas[::-1], color=COLOURS["caregiving"], alpha=0.85,
        edgecolor="white", linewidth=1.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{j} (n={n})" for j, n in zip(jd_ids[::-1], ns[::-1])], fontsize=9.5)

for i, (delta, n) in enumerate(zip(deltas[::-1], ns[::-1])):
    ax.text(delta + 0.002, i, f"{delta:+.3f}", va="center", fontsize=9)

ax.set_xlabel("Caregiving prediction penalty\n(control mean − caregiving mean)", fontsize=11)
ax.set_title(
    "Caregiving-gap penalty varies by job category\n"
    "(top 12 JDs by penalty magnitude, label=1 cases only)",
    fontsize=12, pad=12,
)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlim(0, max(deltas) * 1.20)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig4_caregiving_per_jd.png")
plt.close()
print(f"  -> {FIG_DIR / 'fig4_caregiving_per_jd.png'}")

print(f"\nall figures saved to {FIG_DIR}")