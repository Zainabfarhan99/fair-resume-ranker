"""
audit_mitigated.py
==================
Run the gap-fairness audit on all three rankers and produce side-by-side
comparison: baseline vs uniform-weighted vs reason-aware.

This is the script that answers the paper's central empirical question:
    "Does reason-aware sample weighting reduce gap-reason bias more
     effectively than uniform sample weighting?"

INPUT:
    - baseline_ranker.joblib
    - uniform_weighted_ranker.joblib
    - reason_aware_ranker.joblib
    - resume_vectors.npy (all 60K variants)
    - jd_vectors.npy (26 JDs)
    - test.csv (1,999 test people, 11,994 variants)
    - labels.csv

OUTPUT:
    data/processed/audit/all_models_predictions.csv
    data/processed/audit/all_models_summary.csv
    data/processed/audit/all_models_fairness_metrics.csv
    outputs/10_mitigation_audit_log.txt

Run from repo root:
    python src/audit_mitigated.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import sys
import time

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
EMB_DIR = PROC_DIR / "embeddings"
MODEL_DIR = PROC_DIR / "models"
AUDIT_DIR = PROC_DIR / "audit"
OUT_DIR = REPO_ROOT / "outputs"
AUDIT_DIR.mkdir(exist_ok=True, parents=True)

RESUME_VECTORS = EMB_DIR / "resume_vectors.npy"
RESUME_INDEX = EMB_DIR / "resume_index.csv"
JD_VECTORS = EMB_DIR / "jd_vectors.npy"
JD_INDEX = EMB_DIR / "jd_index.csv"
LABELS_CSV = PROC_DIR / "labels.csv"
TEST_CSV = PROC_DIR / "test.csv"

MODEL_PATHS = {
    "baseline":     MODEL_DIR / "baseline_ranker.joblib",
    "uniform":      MODEL_DIR / "uniform_weighted_ranker.joblib",
    "reason_aware": MODEL_DIR / "reason_aware_ranker.joblib",
}

PRED_PATH = AUDIT_DIR / "all_models_predictions.csv"
SUMMARY_PATH = AUDIT_DIR / "all_models_summary.csv"
METRICS_PATH = AUDIT_DIR / "all_models_fairness_metrics.csv"
LOG_PATH = OUT_DIR / "10_mitigation_audit_log.txt"

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

VARIANT_TYPES = ["control", "no_reason", "caregiving", "health", "layoff", "education"]
GAP_VARIANTS = [v for v in VARIANT_TYPES if v != "control"]

# -----------------------------------------------------------------------------
# Step 1: load
# -----------------------------------------------------------------------------
section("STEP 1: load models, embeddings, labels")
t0 = time.time()
models = {name: joblib.load(p) for name, p in MODEL_PATHS.items()}
resume_vecs = np.load(RESUME_VECTORS)
jd_vecs = np.load(JD_VECTORS)
resume_index = pd.read_csv(RESUME_INDEX)
jd_index = pd.read_csv(JD_INDEX)
labels = pd.read_csv(LABELS_CSV)
test = pd.read_csv(TEST_CSV, usecols=["candidate_id", "person_id", "variant_type",
                                       "gap_reason", "gap_duration_months"])
print(f"  loaded {len(models)} models")
print(f"  loaded {len(test):,} test variants from {test['person_id'].nunique():,} people")

candidate_id_to_row = dict(zip(resume_index["candidate_id"].tolist(), range(len(resume_index))))
jd_id_to_row = dict(zip(jd_index["jd_id"].tolist(), range(len(jd_index))))
jd_ids_ordered = jd_index["jd_id"].tolist()

label_lookup = {(row["person_id"], row["jd_id"]): row["label"] for _, row in labels.iterrows()}
print(f"  built label lookup ({len(label_lookup):,} entries)")
print(f"  loaded in {time.time()-t0:.1f}s")

# -----------------------------------------------------------------------------
# Step 2: build full test feature matrix once
# -----------------------------------------------------------------------------
section("STEP 2: score every (test variant × JD) pair with all three models")

n_variants = len(test)
n_jds = len(jd_ids_ordered)
n_pairs = n_variants * n_jds
print(f"  {n_variants:,} variants × {n_jds} JDs = {n_pairs:,} pairs per model")

test_sorted = test.sort_values("candidate_id").reset_index(drop=True)
test_cand_ids = test_sorted["candidate_id"].to_numpy()
test_person_ids = test_sorted["person_id"].to_numpy()
test_variant_types = test_sorted["variant_type"].to_numpy()
test_gap_reasons = test_sorted["gap_reason"].to_numpy()
test_durations = test_sorted["gap_duration_months"].to_numpy()
resume_rows = np.array([candidate_id_to_row[c] for c in test_cand_ids])

CHUNK_VARIANTS = 1000
predictions_by_model = {name: [] for name in models}

t0 = time.time()
for chunk_start in range(0, n_variants, CHUNK_VARIANTS):
    chunk_end = min(chunk_start + CHUNK_VARIANTS, n_variants)
    chunk_size = chunk_end - chunk_start
    chunk_resume_rows = resume_rows[chunk_start:chunk_end]
    chunk_resume_vecs = resume_vecs[chunk_resume_rows]

    X = np.empty((chunk_size * n_jds, 769), dtype=np.float32)
    X[:, :384] = np.repeat(chunk_resume_vecs, n_jds, axis=0)
    X[:, 384:768] = np.tile(jd_vecs, (chunk_size, 1))
    X[:, 768] = (X[:, :384] * X[:, 384:768]).sum(axis=1)

    for name, clf in models.items():
        probas = clf.predict_proba(X)[:, 1]
        predictions_by_model[name].append(probas)

    if (chunk_start // CHUNK_VARIANTS) % 4 == 0:
        print(f"    scored {chunk_end:,} / {n_variants:,} variants...")

for name in models:
    predictions_by_model[name] = np.concatenate(predictions_by_model[name])
print(f"  scored all pairs in {time.time()-t0:.1f}s")

# -----------------------------------------------------------------------------
# Step 3: assemble predictions dataframe
# -----------------------------------------------------------------------------
section("STEP 3: assemble predictions records")
records = []
for i in range(n_variants):
    pid = test_person_ids[i]
    cand_id = test_cand_ids[i]
    vtype = test_variant_types[i]
    reason = test_gap_reasons[i]
    duration = test_durations[i]
    base_idx = i * n_jds
    for j, jd_id in enumerate(jd_ids_ordered):
        rec = {
            "candidate_id": cand_id,
            "person_id": pid,
            "variant_type": vtype,
            "gap_reason": reason,
            "gap_duration_months": duration,
            "jd_id": jd_id,
            "label": label_lookup.get((pid, jd_id), 0),
        }
        for name in models:
            rec[f"prob_{name}"] = round(float(predictions_by_model[name][base_idx + j]), 6)
        records.append(rec)

pred_df = pd.DataFrame(records)
print(f"  built {len(pred_df):,} prediction records")
pred_df.to_csv(PRED_PATH, index=False)
print(f"  wrote {PRED_PATH}  ({PRED_PATH.stat().st_size / (1024*1024):.1f} MB)")

# -----------------------------------------------------------------------------
# Step 4: compute fairness metrics for each model
# -----------------------------------------------------------------------------
section("STEP 4: compute DPD, EOD, DI per (model × variant)")

# pivot into one row per (person, JD) with one column per model-variant prediction
def compute_model_metrics(model_name):
    """For a given model, return dict mapping variant -> {DPD, EOD, DI, CIs}."""
    prob_col = f"prob_{model_name}"

    # pivot: (person, JD, label) -> column per variant prediction
    pv = pred_df.pivot_table(
        index=["person_id", "jd_id", "label"],
        columns="variant_type",
        values=prob_col,
        aggfunc="first",
    ).reset_index()

    for vt in VARIANT_TYPES:
        pv[f"{vt}_pred"] = (pv[vt] >= DECISION_THRESHOLD).astype(int)

    ctrl_pred = pv["control_pred"].to_numpy()
    label = pv["label"].to_numpy()

    def selection_rate(arr):
        return float(np.mean(arr)) if len(arr) else 0.0

    def tpr(pred, lbl):
        pos = lbl == 1
        if pos.sum() == 0: return 0.0
        return float(np.mean(pred[pos]))

    sr_ctrl = selection_rate(ctrl_pred)
    tpr_ctrl = tpr(ctrl_pred, label)

    rng_local = np.random.default_rng(42)
    n_total = len(ctrl_pred)

    out = {}
    for vt in GAP_VARIANTS:
        var_pred = pv[f"{vt}_pred"].to_numpy()
        sr_var = selection_rate(var_pred)
        tpr_var = tpr(var_pred, label)
        dpd = sr_ctrl - sr_var
        eod = tpr_ctrl - tpr_var
        di = sr_var / sr_ctrl if sr_ctrl > 0 else float("nan")

        # bootstrap EOD
        eod_estimates = []
        for _ in range(BOOTSTRAP_N):
            idx = rng_local.integers(0, n_total, size=n_total)
            eod_b = tpr(ctrl_pred[idx], label[idx]) - tpr(var_pred[idx], label[idx])
            eod_estimates.append(eod_b)
        eod_lo = float(np.percentile(eod_estimates, 2.5))
        eod_hi = float(np.percentile(eod_estimates, 97.5))

        out[vt] = {
            "sr_control": sr_ctrl,
            "sr_variant": sr_var,
            "DPD": dpd,
            "tpr_control": tpr_ctrl,
            "tpr_variant": tpr_var,
            "EOD": eod,
            "EOD_ci_low": eod_lo,
            "EOD_ci_high": eod_hi,
            "DI": di,
        }
    return out

all_metrics = {}
for name in models:
    print(f"\n  computing metrics for {name}...")
    all_metrics[name] = compute_model_metrics(name)

# -----------------------------------------------------------------------------
# Step 5: HEADLINE side-by-side comparison
# -----------------------------------------------------------------------------
section("STEP 5: HEADLINE — side-by-side comparison")

print(f"\n  EOD by variant (lower is better; positive = bias against variant):\n")
print(f"  {'variant':<12s} {'baseline':>14s} {'uniform':>14s} {'reason-aware':>16s}   {'best':>12s}")
print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*16}   {'-'*12}")
for vt in GAP_VARIANTS:
    b = all_metrics["baseline"][vt]["EOD"]
    u = all_metrics["uniform"][vt]["EOD"]
    r = all_metrics["reason_aware"][vt]["EOD"]
    eods = {"baseline": b, "uniform": u, "reason_aware": r}
    # "best" = closest to zero
    best = min(eods, key=lambda k: abs(eods[k]))
    print(f"  {vt:<12s} {b:>+14.4f} {u:>+14.4f} {r:>+16.4f}   {best:>12s}")

print(f"\n  DI by variant (higher is better; <0.80 = legal violation):\n")
print(f"  {'variant':<12s} {'baseline':>14s} {'uniform':>14s} {'reason-aware':>16s}   {'best':>12s}")
print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*16}   {'-'*12}")
for vt in GAP_VARIANTS:
    b = all_metrics["baseline"][vt]["DI"]
    u = all_metrics["uniform"][vt]["DI"]
    r = all_metrics["reason_aware"][vt]["DI"]
    dis = {"baseline": b, "uniform": u, "reason_aware": r}
    best = max(dis, key=lambda k: dis[k])  # higher is better
    flag_b = "✗" if b < 0.80 else " "
    flag_u = "✗" if u < 0.80 else " "
    flag_r = "✗" if r < 0.80 else " "
    print(f"  {vt:<12s} {b:>13.4f}{flag_b} {u:>13.4f}{flag_u} {r:>15.4f}{flag_r}   {best:>12s}")

# -----------------------------------------------------------------------------
# Step 6: improvement analysis
# -----------------------------------------------------------------------------
section("STEP 6: improvement vs baseline (positive = mitigation reduced bias)")

print(f"\n  EOD reduction vs baseline (higher = better mitigation):\n")
print(f"  {'variant':<12s} {'baseline EOD':>14s} {'uniform Δ':>14s} {'reason Δ':>14s}   {'winner':>14s}")
print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*14}   {'-'*14}")
for vt in GAP_VARIANTS:
    b = all_metrics["baseline"][vt]["EOD"]
    u = all_metrics["uniform"][vt]["EOD"]
    r = all_metrics["reason_aware"][vt]["EOD"]
    delta_u = b - u   # positive = uniform improved over baseline
    delta_r = b - r   # positive = reason-aware improved over baseline
    if delta_u > delta_r:
        winner = "uniform"
    elif delta_r > delta_u:
        winner = "reason-aware"
    else:
        winner = "tie"
    print(f"  {vt:<12s} {b:>+14.4f} {delta_u:>+14.4f} {delta_r:>+14.4f}   {winner:>14s}")

# count wins
wins_uniform = 0
wins_reason = 0
for vt in GAP_VARIANTS:
    b = all_metrics["baseline"][vt]["EOD"]
    u = all_metrics["uniform"][vt]["EOD"]
    r = all_metrics["reason_aware"][vt]["EOD"]
    if (b - u) > (b - r):
        wins_uniform += 1
    elif (b - r) > (b - u):
        wins_reason += 1

print(f"\n  win count (most-reduced EOD across {len(GAP_VARIANTS)} variants):")
print(f"    uniform method beat reason-aware on {wins_uniform} variants")
print(f"    reason-aware beat uniform on {wins_reason} variants")
print(f"    ties: {len(GAP_VARIANTS) - wins_uniform - wins_reason}")

# -----------------------------------------------------------------------------
# Step 7: write metrics CSV
# -----------------------------------------------------------------------------
section("STEP 7: write structured outputs")

rows = []
for model_name, vt_dict in all_metrics.items():
    for vt, m in vt_dict.items():
        rows.append({
            "model": model_name,
            "variant_type": vt,
            **m,
        })
metrics_df = pd.DataFrame(rows)
for c in ["DPD", "EOD", "EOD_ci_low", "EOD_ci_high", "DI", "sr_control",
          "sr_variant", "tpr_control", "tpr_variant"]:
    metrics_df[c] = metrics_df[c].round(4)
metrics_df.to_csv(METRICS_PATH, index=False)
print(f"  wrote {METRICS_PATH}")

# summary across models
summary_rows = []
for model_name, vt_dict in all_metrics.items():
    for vt, m in vt_dict.items():
        summary_rows.append({
            "model": model_name,
            "variant": vt,
            "EOD": round(m["EOD"], 4),
            "DI": round(m["DI"], 4),
        })
pd.DataFrame(summary_rows).to_csv(SUMMARY_PATH, index=False)
print(f"  wrote {SUMMARY_PATH}")

section("DONE")
print(f"  predictions: {PRED_PATH}")
print(f"  metrics:     {METRICS_PATH}")
print(f"  log:         {LOG_PATH}")
print(f"\n  next: comparison figures + write up Methods + Results.")

sys.stdout = sys.__stdout__
log_file.close()