"""
audit_baseline.py
=================
THE AUDIT. Run the trained baseline ranker on all 6 variants of every test
person, and measure how predictions change as a function of gap_reason.

This is the script that answers the paper's central question:
    "Does the baseline ranker penalise gap-injected variants relative to control,
    and does the penalty differ by stated gap reason?"

INPUT:
    - baseline_ranker.joblib (trained on control variants only)
    - resume_vectors.npy + indices (all 60K variant embeddings)
    - jd_vectors.npy + indices (26 JD embeddings)
    - test.csv (1,999 test-set people, 11,994 variants)
    - labels.csv (label per (person, JD) pair, computed from skills)

OUTPUT:
    data/processed/audit/baseline_predictions.csv     -- raw predictions for every (variant, JD) pair
    data/processed/audit/baseline_audit_summary.csv   -- per-(reason, JD) aggregated metrics
    outputs/07_baseline_audit_log.txt                 -- run log + headline numbers

Run from repo root:
    python src/audit_baseline.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
import joblib
import sys
import time

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

MODEL_PATH = MODEL_DIR / "baseline_ranker.joblib"
PRED_PATH = AUDIT_DIR / "baseline_predictions.csv"
SUMMARY_PATH = AUDIT_DIR / "baseline_audit_summary.csv"
LOG_PATH = OUT_DIR / "07_baseline_audit_log.txt"

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

# -----------------------------------------------------------------------------
# Step 1: load everything
# -----------------------------------------------------------------------------
section("STEP 1: load model, embeddings, labels")
t0 = time.time()
clf = joblib.load(MODEL_PATH)
resume_vecs = np.load(RESUME_VECTORS)
jd_vecs = np.load(JD_VECTORS)
resume_index = pd.read_csv(RESUME_INDEX)
jd_index = pd.read_csv(JD_INDEX)
labels = pd.read_csv(LABELS_CSV)
test = pd.read_csv(TEST_CSV, usecols=["candidate_id", "person_id", "variant_type",
                                       "gap_reason", "gap_duration_months"])
print(f"  loaded {len(test):,} test variants from {test['person_id'].nunique():,} people")

# fast lookups
candidate_id_to_row = dict(zip(resume_index["candidate_id"].tolist(),
                               range(len(resume_index))))
jd_id_to_row = dict(zip(jd_index["jd_id"].tolist(), range(len(jd_index))))
jd_ids_ordered = jd_index["jd_id"].tolist()

# label lookup: (pid, jd_id) -> label
label_lookup = {}
for _, row in labels.iterrows():
    label_lookup[(row["person_id"], row["jd_id"])] = row["label"]
print(f"  built label lookup ({len(label_lookup):,} entries)")
print(f"  loaded in {time.time()-t0:.1f}s")

# -----------------------------------------------------------------------------
# Step 2: score every (test variant × JD) pair
# -----------------------------------------------------------------------------
section("STEP 2: score every (test variant × JD) pair")

n_variants = len(test)
n_jds = len(jd_ids_ordered)
n_pairs = n_variants * n_jds
print(f"  scoring {n_variants:,} variants × {n_jds} JDs = {n_pairs:,} pairs")

# We build the feature matrix in one shot then call predict_proba once.
# That's much faster than scoring one at a time.

# we need to pair up: each variant gets scored against EVERY JD
# create arrays for indexing into resume_vecs and jd_vecs
test_sorted = test.sort_values("candidate_id").reset_index(drop=True)
test_cand_ids = test_sorted["candidate_id"].to_numpy()
test_person_ids = test_sorted["person_id"].to_numpy()
test_variant_types = test_sorted["variant_type"].to_numpy()
test_gap_reasons = test_sorted["gap_reason"].to_numpy()
test_durations = test_sorted["gap_duration_months"].to_numpy()

# resume_row[i] = row in resume_vecs for the i-th test variant
resume_rows = np.array([candidate_id_to_row[c] for c in test_cand_ids])

t0 = time.time()
# build the big feature matrix: n_pairs rows, 769 cols
# we tile resumes (each repeated n_jds times) and tile JDs (full set repeated n_variants times)
# too memory-heavy at 11994 * 26 = 311K rows, 769 cols of float32 → ~960MB. doable, but let's chunk.

CHUNK_VARIANTS = 1000  # process 1000 variants at a time → 26K rows × 769 cols × 4 bytes ≈ 80MB chunks
all_predictions = []

for chunk_start in range(0, n_variants, CHUNK_VARIANTS):
    chunk_end = min(chunk_start + CHUNK_VARIANTS, n_variants)
    chunk_size = chunk_end - chunk_start
    chunk_resume_rows = resume_rows[chunk_start:chunk_end]
    chunk_resume_vecs = resume_vecs[chunk_resume_rows]   # (chunk_size, 384)

    # build (chunk_size * n_jds) × 769 feature matrix
    X = np.empty((chunk_size * n_jds, 769), dtype=np.float32)
    # repeat resumes n_jds times along axis 0
    X[:, :384] = np.repeat(chunk_resume_vecs, n_jds, axis=0)
    # tile JD vectors n_variants times
    X[:, 384:768] = np.tile(jd_vecs, (chunk_size, 1))
    # cosine similarity (vectors are unit-normalised so dot = cosine)
    X[:, 768] = (X[:, :384] * X[:, 384:768]).sum(axis=1)

    # predict probabilities
    probas = clf.predict_proba(X)[:, 1]
    all_predictions.append(probas)

    if (chunk_start // CHUNK_VARIANTS) % 5 == 0:
        print(f"    scored {chunk_end:,} / {n_variants:,} variants...")

all_predictions = np.concatenate(all_predictions)
print(f"  scored all pairs in {time.time()-t0:.1f}s")
print(f"  predictions shape: {all_predictions.shape}")

# -----------------------------------------------------------------------------
# Step 3: build the prediction-records dataframe
# -----------------------------------------------------------------------------
section("STEP 3: assemble prediction records")
# For each variant i, predictions[i*n_jds : (i+1)*n_jds] are the scores for that variant
# against each JD in jd_ids_ordered.

records = []
for i in range(n_variants):
    pid = test_person_ids[i]
    cand_id = test_cand_ids[i]
    vtype = test_variant_types[i]
    reason = test_gap_reasons[i]
    duration = test_durations[i]
    base_idx = i * n_jds
    for j, jd_id in enumerate(jd_ids_ordered):
        records.append({
            "candidate_id": cand_id,
            "person_id": pid,
            "variant_type": vtype,
            "gap_reason": reason,
            "gap_duration_months": duration,
            "jd_id": jd_id,
            "predicted_proba": round(float(all_predictions[base_idx + j]), 6),
            "label": label_lookup.get((pid, jd_id), 0),
        })

pred_df = pd.DataFrame(records)
print(f"  built {len(pred_df):,} prediction records")

pred_df.to_csv(PRED_PATH, index=False)
print(f"  wrote {PRED_PATH}  ({PRED_PATH.stat().st_size / (1024*1024):.1f} MB)")

# -----------------------------------------------------------------------------
# Step 4: HEADLINE ANALYSIS — compare predictions across variants of the same person
# -----------------------------------------------------------------------------
section("STEP 4: HEADLINE — per-variant prediction shifts")

# For each (person, JD), pivot so we have one row with all 6 variant predictions side-by-side
pivoted = pred_df.pivot_table(
    index=["person_id", "jd_id", "label"],
    columns="variant_type",
    values="predicted_proba",
    aggfunc="first",
).reset_index()

# Restrict the analysis to (person, JD) pairs where label==1 — these are the meaningful
# audit cases (the model is being asked to say "yes" and we measure whether it flips to "no").
positive_pairs = pivoted[pivoted["label"] == 1].copy()
print(f"  total (person × JD) pairs in test: {len(pivoted):,}")
print(f"  pairs with label=1 (audit-relevant): {len(positive_pairs):,}")

print(f"\n  --- mean predicted_proba per variant (label=1 cases) ---")
for vt in VARIANT_TYPES:
    if vt in positive_pairs.columns:
        m = positive_pairs[vt].mean()
        print(f"    {vt:12s}  {m:.4f}")

print(f"\n  --- delta from control (positive = gap variant scores LOWER than control) ---")
ctrl_mean = positive_pairs["control"].mean()
for vt in VARIANT_TYPES:
    if vt == "control" or vt not in positive_pairs.columns: continue
    m = positive_pairs[vt].mean()
    delta = ctrl_mean - m
    pct_drop = (delta / ctrl_mean) * 100
    print(f"    control - {vt:12s} = {delta:+.4f}  ({pct_drop:+.2f}% drop from control)")

# binary flip rate: fraction of (person, JD) where control predicts >0.5 but variant predicts <=0.5
print(f"\n  --- binary FLIP rate (control predicted match, variant predicted no-match) ---")
for vt in VARIANT_TYPES:
    if vt == "control" or vt not in positive_pairs.columns: continue
    if "control" not in positive_pairs.columns: continue
    ctrl_pos = positive_pairs["control"] > 0.5
    var_neg = positive_pairs[vt] <= 0.5
    flips = (ctrl_pos & var_neg).sum()
    flippable = ctrl_pos.sum()
    rate = flips / flippable * 100 if flippable else 0
    print(f"    {vt:12s}  flips: {flips:>5,} / {flippable:>5,}  ({rate:.2f}%)")

# -----------------------------------------------------------------------------
# Step 5: per-JD breakdown
# -----------------------------------------------------------------------------
section("STEP 5: per-JD breakdown of caregiving penalty (most-watched reason)")

if "caregiving" in positive_pairs.columns:
    print(f"  showing top 10 JDs ranked by caregiving penalty\n")
    rows = []
    for jd_id, sub in positive_pairs.groupby("jd_id"):
        if len(sub) < 5:  # skip JDs with too few label=1 examples
            continue
        ctrl_m = sub["control"].mean()
        care_m = sub["caregiving"].mean()
        delta = ctrl_m - care_m
        rows.append((jd_id, len(sub), ctrl_m, care_m, delta))
    rows = sorted(rows, key=lambda r: r[4], reverse=True)
    print(f"    {'jd_id':<25s}  {'n':>5s}  {'ctrl':>7s}  {'care':>7s}  {'penalty':>9s}")
    for jd_id, n, ctrl_m, care_m, delta in rows[:10]:
        print(f"    {jd_id:<25s}  {n:>5}  {ctrl_m:>7.4f}  {care_m:>7.4f}  {delta:>+9.4f}")

# -----------------------------------------------------------------------------
# Step 6: write summary CSV
# -----------------------------------------------------------------------------
section("STEP 6: write summary CSV")
summary_records = []
for vt in VARIANT_TYPES:
    if vt not in pivoted.columns: continue
    pos = pivoted[pivoted["label"] == 1][vt].dropna()
    neg = pivoted[pivoted["label"] == 0][vt].dropna()
    summary_records.append({
        "variant_type": vt,
        "n_label_1": len(pos),
        "mean_proba_label_1": round(pos.mean(), 6) if len(pos) else None,
        "n_label_0": len(neg),
        "mean_proba_label_0": round(neg.mean(), 6) if len(neg) else None,
    })
summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(SUMMARY_PATH, index=False)
print(f"  wrote {SUMMARY_PATH}")
print(summary_df.to_string(index=False))

section("DONE")
print(f"  predictions: {PRED_PATH}")
print(f"  summary:     {SUMMARY_PATH}")
print(f"  log:         {LOG_PATH}")
print(f"\n  next: standard fairness metrics (DPD, EOD, DI) per gap_reason.")

sys.stdout = sys.__stdout__
log_file.close()