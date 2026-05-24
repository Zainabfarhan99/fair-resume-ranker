"""
make_comparison_figure.py
=========================
Single figure that visualises Section 4.3's main story: baseline vs uniform
vs reason-aware EOD per variant.

Currently the mitigation comparison lives only in Table 2 — readers must scan
the table to assess "which method closes which gap." A grouped bar chart makes
the closure pattern immediately visible.

OUTPUTS:
    outputs/figures/fig9_mitigation_comparison.png

Run from repo root:
    python src/make_comparison_figure.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys

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

METRICS_CSV = AUDIT_DIR / "all_models_fairness_metrics.csv"
FIG_PATH = FIG_DIR / "fig9_mitigation_comparison.png"
LOG_PATH = OUT_DIR / "15_comparison_figure_log.txt"

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
# Step 1: load all-models metrics CSV
# -----------------------------------------------------------------------------
section("STEP 1: load fairness metrics")
df = pd.read_csv(METRICS_CSV)
print(f"  loaded {len(df)} rows")
print(f"  columns: {list(df.columns)}")
print(f"\n  preview:")
print(df.head(10).to_string(index=False))

# -----------------------------------------------------------------------------
# Step 2: extract per (model, variant) EOD
# -----------------------------------------------------------------------------
section("STEP 2: extract EOD per (model, variant)")

# Check that all expected models and variants are present.
expected_models = {"baseline", "uniform", "reason_aware"}
expected_variants = {"no_reason", "caregiving", "health", "layoff", "education"}
found_models = set(df["model"].unique()) if "model" in df.columns else set()
found_variants = set(df["variant_type"].unique()) if "variant_type" in df.columns else set()

print(f"\n  models found: {sorted(found_models)}")
print(f"  variants found: {sorted(found_variants)}")
missing_m = expected_models - found_models
missing_v = expected_variants - found_variants
if missing_m:
    print(f"  WARNING: missing models: {missing_m}")
if missing_v:
    print(f"  WARNING: missing variants: {missing_v}")

# pivot: variant on rows, model on cols, value = EOD
pivot = df.pivot_table(index="variant_type", columns="model", values="EOD", aggfunc="first")
variant_order = ["no_reason", "health", "layoff", "caregiving", "education"]
pivot = pivot.reindex(variant_order)
print(f"\n  EOD pivot table:")
print(pivot.to_string())

# -----------------------------------------------------------------------------
# Step 3: figure
# -----------------------------------------------------------------------------
section("STEP 3: render combined comparison figure")

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

# Colours for the three models
MODEL_COLOURS = {
    "baseline":     "#6c757d",    # grey — the un-mitigated reference
    "uniform":      "#1f78b4",    # blue — generic mitigation
    "reason_aware": "#33a02c",    # green — reason-aware mitigation
}

MODEL_LABELS = {
    "baseline":     "Baseline (no mitigation)",
    "uniform":      "Uniform sample weighting",
    "reason_aware": "Reason-aware sample weighting",
}

fig, ax = plt.subplots(figsize=(10, 5.5))

n_variants = len(variant_order)
models_in_order = ["baseline", "uniform", "reason_aware"]
n_models = len(models_in_order)
x = np.arange(n_variants)
width = 0.27

for i, model in enumerate(models_in_order):
    values = pivot[model].values
    offset = (i - n_models/2 + 0.5) * width
    bars = ax.bar(
        x + offset, values, width,
        label=MODEL_LABELS[model],
        color=MODEL_COLOURS[model],
        edgecolor="white",
        linewidth=0.7,
        alpha=0.92,
    )
    # add value annotations
    for bar, v in zip(bars, values):
        height = bar.get_height()
        y_offset = 0.003 if height >= 0 else -0.003
        va = "bottom" if height >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width()/2, height + y_offset,
            f"{v:+.3f}",
            ha="center", va=va,
            fontsize=8, color="black",
        )

ax.axhline(0, color="grey", linewidth=0.7, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(variant_order)
ax.set_xlabel("Gap reason variant")
ax.set_ylabel("Equalised Odds Difference (EOD) vs. control")
ax.set_title(
    "Mitigation comparison: how each method closes reason-stratified bias\n"
    "(lower magnitude = closer to fairness parity; both methods substantially reduce baseline bias)",
    fontsize=11, pad=12,
)
ax.legend(loc="upper left", framealpha=0.95)

# Padding so annotations don't get clipped
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin - 0.005, ymax + 0.012)

plt.tight_layout()
plt.savefig(FIG_PATH)
plt.close()
print(f"  saved {FIG_PATH}")

section("DONE")
print(f"  figure: {FIG_PATH}")
print(f"  log:    {LOG_PATH}")

sys.stdout = sys.__stdout__
log_file.close()