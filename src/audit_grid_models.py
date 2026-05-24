"""
audit_grid_models.py
====================
Audit all 4 grid cells (MiniLM/mpnet × LR/XGBoost) for the disclosure paradox.

For each model, score all 6 variants × 26 JDs of test candidates, then compute
EOD per variant. The output is a 4-model × 5-variant table showing whether the
disclosure paradox holds across all architecture combinations.

OUTPUTS:
    data/processed/audit/grid_audit_metrics.csv
    outputs/figures/fig10_grid_audit.png
    outputs/18_grid_audit_log.txt

Run from repo root:
    python src/audit_grid_models.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys
import time
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = REPO_ROOT / "data" / "processed"
EMB_DIR = PROC_DIR / "embeddings"
MODEL_DIR = PROC_DIR / "models"
AUDIT_DIR = PROC_DIR / "audit"
OUT_DIR = REPO_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

LABELS_CSV = PROC_DIR / "labels.csv"
TEST_CSV = PROC_DIR / "test.csv"

RESULTS_CSV = AUDIT_DIR / "grid_audit_metrics.csv"
FIG_PATH = FIG_DIR / "fig10_grid_audit.png"
LOG_PATH = OUT_DIR / "18_grid_audit_log.txt"

VARIANT_TYPES = ["control", "no_reason", "caregiving", "health", "layoff", "education"]
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
# Load shared
# -----------------------------------------------------------------------------
section("STEP 1: load shared test data + labels")
labels = pd.read_csv(LABELS_CSV)
label_lookup = {(r["person_id"], r["jd_id"]): r["label"] for _, r in labels.iterrows()}
test = pd.read_csv(TEST_CSV, usecols=["candidate_id", "person_id", "variant_type"])
print(f"  test variants: {len(test):,}  (people: {test['person_id'].nunique():,})")

# -----------------------------------------------------------------------------
# Define the 4 grid cells with their embedding sources and trained models
# -----------------------------------------------------------------------------
GRID_CELLS = [
    {
        "name": "baseline_minilm_lr",
        "embedding": "minilm", "classifier": "lr",
        "resume_npy":  EMB_DIR / "resume_vectors.npy",
        "resume_idx":  EMB_DIR / "resume_index.csv",
        "jd_npy":      EMB_DIR / "jd_vectors.npy",
        "jd_idx":      EMB_DIR / "jd_index.csv",
        "model_path":  MODEL_DIR / "baseline_ranker.joblib",
    },
    {
        "name": "mpnet_lr",
        "embedding": "mpnet", "classifier": "lr",
        "resume_npy":  EMB_DIR / "resume_vectors_mpnet.npy",
        "resume_idx":  EMB_DIR / "resume_index_mpnet.csv",
        "jd_npy":      EMB_DIR / "jd_vectors_mpnet.npy",
        "jd_idx":      EMB_DIR / "jd_index_mpnet.csv",
        "model_path":  MODEL_DIR / "mpnet_lr.joblib",
    },
    {
        "name": "minilm_xgb",
        "embedding": "minilm", "classifier": "xgb",
        "resume_npy":  EMB_DIR / "resume_vectors.npy",
        "resume_idx":  EMB_DIR / "resume_index.csv",
        "jd_npy":      EMB_DIR / "jd_vectors.npy",
        "jd_idx":      EMB_DIR / "jd_index.csv",
        "model_path":  MODEL_DIR / "minilm_xgb.joblib",
    },
    {
        "name": "mpnet_xgb",
        "embedding": "mpnet", "classifier": "xgb",
        "resume_npy":  EMB_DIR / "resume_vectors_mpnet.npy",
        "resume_idx":  EMB_DIR / "resume_index_mpnet.csv",
        "jd_npy":      EMB_DIR / "jd_vectors_mpnet.npy",
        "jd_idx":      EMB_DIR / "jd_index_mpnet.csv",
        "model_path":  MODEL_DIR / "mpnet_xgb.joblib",
    },
]

# -----------------------------------------------------------------------------
# Audit each cell
# -----------------------------------------------------------------------------
all_records = []
per_cell_pivots = {}

for cell in GRID_CELLS:
    section(f"AUDIT: {cell['name']}  ({cell['embedding']} + {cell['classifier']})")

    # Load
    clf = joblib.load(cell["model_path"])
    r_vecs = np.load(cell["resume_npy"])
    r_idx  = pd.read_csv(cell["resume_idx"])
    j_vecs = np.load(cell["jd_npy"])
    j_idx  = pd.read_csv(cell["jd_idx"])
    cand_to_row = dict(zip(r_idx["candidate_id"].tolist(), range(len(r_idx))))
    jd_ids = j_idx["jd_id"].tolist()
    n_jds = len(jd_ids)
    dim = r_vecs.shape[1]

    # Score in chunks
    test_sorted = test.sort_values("candidate_id").reset_index(drop=True)
    cand_ids = test_sorted["candidate_id"].to_numpy()
    pids     = test_sorted["person_id"].to_numpy()
    vtypes   = test_sorted["variant_type"].to_numpy()

    resume_rows = np.array([cand_to_row[c] for c in cand_ids])
    n_variants = len(test_sorted)

    print(f"  scoring {n_variants:,} variants × {n_jds} JDs = {n_variants*n_jds:,} pairs")
    t0 = time.time()

    CHUNK = 1000
    all_probas = []
    for chunk_start in range(0, n_variants, CHUNK):
        chunk_end = min(chunk_start + CHUNK, n_variants)
        chunk_size = chunk_end - chunk_start
        chunk_R = r_vecs[resume_rows[chunk_start:chunk_end]]

        X = np.empty((chunk_size * n_jds, 2*dim + 1), dtype=np.float32)
        X[:, :dim] = np.repeat(chunk_R, n_jds, axis=0)
        X[:, dim:2*dim] = np.tile(j_vecs, (chunk_size, 1))
        X[:, 2*dim] = (X[:, :dim] * X[:, dim:2*dim]).sum(axis=1)

        probas = clf.predict_proba(X)[:, 1]
        all_probas.append(probas)

    all_probas = np.concatenate(all_probas)
    print(f"  scored in {time.time()-t0:.1f}s")

    # Build pivot table: (person, jd) → variant predictions
    records = []
    for i in range(n_variants):
        pid = pids[i]; vt = vtypes[i]
        base = i * n_jds
        for j, jd_id in enumerate(jd_ids):
            records.append({
                "person_id": pid,
                "jd_id": jd_id,
                "variant_type": vt,
                "predicted_proba": float(all_probas[base + j]),
                "label": label_lookup.get((pid, jd_id), 0),
            })
    pred_df = pd.DataFrame(records)

    pivoted = pred_df.pivot_table(
        index=["person_id", "jd_id", "label"],
        columns="variant_type",
        values="predicted_proba",
        aggfunc="first",
    ).reset_index()

    audit_cohort = pivoted[pivoted["label"] == 1].copy()
    per_cell_pivots[cell["name"]] = audit_cohort

    # Compute EOD per variant
    ctrl_tpr = (audit_cohort["control"] >= THRESHOLD).mean()
    print(f"\n  EOD by variant (label=1 cases, n={len(audit_cohort):,}):")
    for r in REASONS:
        if r not in audit_cohort.columns:
            continue
        var_tpr = (audit_cohort[r] >= THRESHOLD).mean()
        eod = ctrl_tpr - var_tpr

        # Flip rate
        flippable = audit_cohort["control"] >= THRESHOLD
        flipped = flippable & (audit_cohort[r] < THRESHOLD)
        flip_rate = flipped.sum() / flippable.sum() if flippable.sum() > 0 else np.nan

        # DI on full test cohort
        all_pivot = pivoted.copy()
        ctrl_sr = (all_pivot["control"] >= THRESHOLD).mean()
        var_sr = (all_pivot[r] >= THRESHOLD).mean()
        di = var_sr / ctrl_sr if ctrl_sr > 0 else np.nan

        all_records.append({
            "cell": cell["name"],
            "embedding": cell["embedding"],
            "classifier": cell["classifier"],
            "reason": r,
            "eod": round(float(eod), 6),
            "flip_rate": round(float(flip_rate), 6),
            "di": round(float(di), 6),
            "violates_4_5": bool(di < 0.80),
        })
        print(f"    {r:<12s}  EOD={eod:+.4f}  flip={100*flip_rate:5.1f}%  DI={di:.3f}"
              + (" ✗" if di < 0.80 else ""))

# -----------------------------------------------------------------------------
# Cross-cell comparison
# -----------------------------------------------------------------------------
section("CROSS-CELL: EOD comparison across grid cells")
results_df = pd.DataFrame(all_records)

print(f"\n  EOD by cell × reason:\n")
print(f"  {'cell':<22s} | " + " | ".join(f"{r:>10s}" for r in REASONS))
print(f"  {'-'*22}-+-" + "-+-".join("-"*10 for _ in REASONS))
for cell_name in [c["name"] for c in GRID_CELLS]:
    sub = results_df[results_df["cell"] == cell_name].set_index("reason")
    row = " | ".join(
        f"{sub.loc[r,'eod']:>+10.4f}" if r in sub.index else f"{'n/a':>10s}"
        for r in REASONS
    )
    print(f"  {cell_name:<22s} | {row}")

print(f"\n  DI by cell × reason (✗ = violates 4/5 rule):\n")
print(f"  {'cell':<22s} | " + " | ".join(f"{r:>11s}" for r in REASONS))
print(f"  {'-'*22}-+-" + "-+-".join("-"*11 for _ in REASONS))
for cell_name in [c["name"] for c in GRID_CELLS]:
    sub = results_df[results_df["cell"] == cell_name].set_index("reason")
    parts = []
    for r in REASONS:
        if r in sub.index:
            di = sub.loc[r,'di']
            mark = " ✗" if di < 0.80 else "  "
            parts.append(f"{di:>9.3f}{mark}")
        else:
            parts.append(f"{'n/a':>11s}")
    print(f"  {cell_name:<22s} | " + " | ".join(parts))

# -----------------------------------------------------------------------------
# Ordering preservation
# -----------------------------------------------------------------------------
section("ORDERING: is the reason-stratified pattern preserved across cells?")
preserved = 0
for cell_name in [c["name"] for c in GRID_CELLS]:
    sub = results_df[results_df["cell"] == cell_name].sort_values("eod", ascending=False)
    ranking = sub["reason"].tolist()
    top3 = ranking[:3]
    is_preserved = (ranking[0] == "education") and (set(top3) == {"education", "caregiving", "layoff"})
    if is_preserved:
        preserved += 1
    marker = "✓" if is_preserved else "✗"
    print(f"  {cell_name:<22s}  {' > '.join(ranking)} [{marker}]")
print(f"\n  Ordering preserved in {preserved} / {len(GRID_CELLS)} cells.")

# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------
section("SAVE")
results_df.to_csv(RESULTS_CSV, index=False)
print(f"  wrote {RESULTS_CSV}")

# -----------------------------------------------------------------------------
# Figure: grouped bars
# -----------------------------------------------------------------------------
section("FIGURE")

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

CELL_LABELS = {
    "baseline_minilm_lr": "MiniLM + LR (baseline)",
    "mpnet_lr":           "mpnet + LR",
    "minilm_xgb":         "MiniLM + XGBoost",
    "mpnet_xgb":          "mpnet + XGBoost",
}
CELL_COLOURS = {
    "baseline_minilm_lr": "#6c757d",
    "mpnet_lr":           "#1f78b4",
    "minilm_xgb":         "#33a02c",
    "mpnet_xgb":          "#e31a1c",
}

fig, ax = plt.subplots(figsize=(11, 5.5))

cells_in_order = [c["name"] for c in GRID_CELLS]
n_cells = len(cells_in_order)
n_reasons = len(REASONS)
x = np.arange(n_reasons)
width = 0.21

for i, cn in enumerate(cells_in_order):
    sub = results_df[results_df["cell"] == cn].set_index("reason")
    vals = [sub.loc[r, "eod"] if r in sub.index else 0 for r in REASONS]
    offset = (i - n_cells/2 + 0.5) * width
    ax.bar(x + offset, vals, width, label=CELL_LABELS[cn], color=CELL_COLOURS[cn],
           alpha=0.9, edgecolor="white", linewidth=0.5)

ax.axhline(0, color="grey", linewidth=0.7, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(REASONS)
ax.set_xlabel("Gap reason variant")
ax.set_ylabel("Equalised Odds Difference (EOD) vs. control")
ax.set_title(
    "Architecture robustness: disclosure paradox across 2×2 grid of embeddings × classifiers\n"
    "(education dominates in all 4 configurations)",
    fontsize=11, pad=12,
)
ax.legend(loc="upper left", framealpha=0.95)
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