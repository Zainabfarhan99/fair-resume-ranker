"""
train_with_mitigation.py
========================
Phase 3 (revised twice): train two rankers with sample-weighting mitigation.

CHANGE LOG:
    v1: Tried Kamiran-Calders reweighing. Failed because labels are constant
        across variants by construction → KC weights collapse to 1.0.
    v2: Switched to normative sample weighting (Calmon-style). Worked
        algorithmically but consumed 12+ GB RAM training on all 8001 train
        people × 6 variants × 26 JDs = 1.25M rows. Crashed Apple Silicon
        MacBook Air.
    v3 (this version): Subsampled training to 4000 train people. Reduces
        feature matrix to ~624K rows, ~2 GB peak memory. Test set untouched.

METHODS:
    A. UNIFORM SAMPLE WEIGHTING — all 5 gap variants get the same upweight.
    B. REASON-AWARE SAMPLE WEIGHTING (paper's novel contribution) —
       different upweights per reason on a normative spectrum.

TRAINING DATA:
    Subsample 4000 of 8001 train-set people (seed=42, deterministic).
    All 6 variants of each subsampled person are kept.
    Final pair count: 4000 people × 6 variants × 26 JDs = 624,000 pairs.

OUTPUTS:
    data/processed/models/uniform_weighted_ranker.joblib
    data/processed/models/reason_aware_ranker.joblib
    data/processed/models/mitigation_metrics.json
    outputs/09_mitigation_training_log.txt

Run from repo root:
    python src/train_with_mitigation.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
import joblib
import sys
import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
LR_MAX_ITER = 200
LR_C = 1.0

# subsample 4000 of the 8001 train-set people for memory safety on Apple Silicon
TRAIN_SUBSAMPLE = 4000

UNIFORM_GAP_WEIGHT = 2.0
REASON_AWARE_WEIGHTS = {
    "control":    1.0,
    "no_reason":  1.0,
    "caregiving": 2.0,
    "health":     2.0,
    "layoff":     2.0,
    "education":  1.5,
}

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = REPO_ROOT / "data" / "processed"
EMB_DIR = PROC_DIR / "embeddings"
MODEL_DIR = PROC_DIR / "models"
OUT_DIR = REPO_ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

RESUME_VECTORS = EMB_DIR / "resume_vectors.npy"
RESUME_INDEX = EMB_DIR / "resume_index.csv"
JD_VECTORS = EMB_DIR / "jd_vectors.npy"
JD_INDEX = EMB_DIR / "jd_index.csv"
LABELS_CSV = PROC_DIR / "labels.csv"
TRAIN_CSV = PROC_DIR / "train.csv"
TEST_CSV = PROC_DIR / "test.csv"

UNIFORM_MODEL_PATH = MODEL_DIR / "uniform_weighted_ranker.joblib"
REASON_MODEL_PATH = MODEL_DIR / "reason_aware_ranker.joblib"
METRICS_PATH = MODEL_DIR / "mitigation_metrics.json"
LOG_PATH = OUT_DIR / "09_mitigation_training_log.txt"

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

rng = np.random.default_rng(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -----------------------------------------------------------------------------
# Step 1: load
# -----------------------------------------------------------------------------
section("STEP 1: load embeddings, labels, splits")
t0 = time.time()
resume_vecs = np.load(RESUME_VECTORS)
jd_vecs = np.load(JD_VECTORS)
resume_index = pd.read_csv(RESUME_INDEX)
jd_index = pd.read_csv(JD_INDEX)
labels = pd.read_csv(LABELS_CSV)
train = pd.read_csv(TRAIN_CSV, usecols=["candidate_id", "person_id", "variant_type", "gap_reason"])
test = pd.read_csv(TEST_CSV, usecols=["candidate_id", "person_id", "variant_type", "gap_reason"])
print(f"  loaded in {time.time()-t0:.1f}s")
print(f"  full train pool: {train['person_id'].nunique():,} people")
print(f"  test set:        {test['person_id'].nunique():,} people")

candidate_id_to_row = dict(zip(resume_index["candidate_id"].tolist(), range(len(resume_index))))
jd_id_to_row = dict(zip(jd_index["jd_id"].tolist(), range(len(jd_index))))
jd_ids_ordered = jd_index["jd_id"].tolist()
n_jds = len(jd_ids_ordered)

# -----------------------------------------------------------------------------
# Step 2: SUBSAMPLE to 4000 train people
# -----------------------------------------------------------------------------
section(f"STEP 2: subsample {TRAIN_SUBSAMPLE:,} train people for memory safety")
all_train_pids = sorted(train["person_id"].unique())
sampled_pids = rng.choice(all_train_pids, size=TRAIN_SUBSAMPLE, replace=False)
sampled_pids_set = set(sampled_pids.tolist())
train_sub = train[train["person_id"].isin(sampled_pids_set)].copy()
print(f"  subsampled to {len(sampled_pids):,} people")
print(f"  variants in subsample: {len(train_sub):,}  (expected {TRAIN_SUBSAMPLE * 6:,})")
assert len(train_sub) == TRAIN_SUBSAMPLE * 6, "subsample missing variants — investigate"

# -----------------------------------------------------------------------------
# Step 3: build training feature matrix
# -----------------------------------------------------------------------------
section("STEP 3: build training features for all 6 variants of subsampled people")
train_labels = labels[labels["person_id"].isin(sampled_pids_set)].copy()
print(f"  subsampled (person, JD) pairs: {len(train_labels):,}")

n_total = len(train_sub) * n_jds
print(f"  building feature matrix: {len(train_sub):,} variants × {n_jds} JDs = {n_total:,} pairs")
print(f"  expected memory: {n_total * 769 * 4 / (1024**3):.2f} GB for X")

X = np.empty((n_total, 769), dtype=np.float32)
y = np.empty(n_total, dtype=np.int8)
variant_arr = np.empty(n_total, dtype="<U12")

t0 = time.time()
label_lookup = {(row["person_id"], row["jd_id"]): row["label"] for _, row in train_labels.iterrows()}
print(f"  built label lookup ({len(label_lookup):,} entries) in {time.time()-t0:.1f}s")

t0 = time.time()
train_sorted = train_sub.sort_values("candidate_id").reset_index(drop=True)
write_idx = 0
for i, (_, row) in enumerate(train_sorted.iterrows()):
    cand_id = row["candidate_id"]
    pid = row["person_id"]
    vtype = row["variant_type"]
    r_vec = resume_vecs[candidate_id_to_row[cand_id]]
    for j, jd_id in enumerate(jd_ids_ordered):
        j_vec = jd_vecs[jd_id_to_row[jd_id]]
        cos_sim = float(np.dot(r_vec, j_vec))
        X[write_idx, :384] = r_vec
        X[write_idx, 384:768] = j_vec
        X[write_idx, 768] = cos_sim
        y[write_idx] = label_lookup.get((pid, jd_id), 0)
        variant_arr[write_idx] = vtype
        write_idx += 1
    if (i + 1) % 5000 == 0:
        print(f"    processed {i+1:,} / {len(train_sorted):,} variants...")

assert write_idx == n_total
print(f"  built X shape={X.shape} in {time.time()-t0:.1f}s")
print(f"  positive rate: {y.mean()*100:.2f}%")

# -----------------------------------------------------------------------------
# Step 4: build sample weights for both methods
# -----------------------------------------------------------------------------
section("STEP 4: build sample weights")

sample_weight_uniform = np.ones(n_total, dtype=np.float32)
sample_weight_uniform[variant_arr != "control"] = UNIFORM_GAP_WEIGHT

sample_weight_reason = np.ones(n_total, dtype=np.float32)
for vt, w in REASON_AWARE_WEIGHTS.items():
    sample_weight_reason[variant_arr == vt] = w

print(f"\n  Method A (uniform) — gap variants weight = {UNIFORM_GAP_WEIGHT}")
print(f"    distribution: min={sample_weight_uniform.min():.2f}  "
      f"max={sample_weight_uniform.max():.2f}  mean={sample_weight_uniform.mean():.3f}")

print(f"\n  Method B (reason-aware) — per-variant weights:")
for vt, w in REASON_AWARE_WEIGHTS.items():
    n_vt = (variant_arr == vt).sum()
    print(f"    {vt:12s}: weight={w:.1f}  (applied to {n_vt:>7,} examples)")
print(f"    distribution: min={sample_weight_reason.min():.2f}  "
      f"max={sample_weight_reason.max():.2f}  mean={sample_weight_reason.mean():.3f}")

# -----------------------------------------------------------------------------
# Step 5: train both models (n_jobs=1 to avoid macOS multiprocessing deadlock)
# -----------------------------------------------------------------------------
section("STEP 5: train Method A (uniform) and Method B (reason-aware)")

def train_lr(X, y, sample_weight, name):
    print(f"\n  training {name}...")
    clf = LogisticRegression(
        max_iter=LR_MAX_ITER,
        C=LR_C,
        random_state=RANDOM_SEED,
        n_jobs=1,  # single-threaded; avoids macOS L-BFGS deadlock
    )
    t0 = time.time()
    clf.fit(X, y, sample_weight=sample_weight)
    print(f"    converged in {clf.n_iter_[0]} iters, {time.time()-t0:.1f}s")
    return clf

clf_uniform = train_lr(X, y, sample_weight_uniform, "Method A (uniform weighting)")
clf_reason = train_lr(X, y, sample_weight_reason, "Method B (reason-aware weighting)")

# free training memory before building test set
del X, y, variant_arr, sample_weight_uniform, sample_weight_reason

# -----------------------------------------------------------------------------
# Step 6: evaluate on test (control variants only)
# -----------------------------------------------------------------------------
section("STEP 6: evaluate on test set (control variants only)")

test_control = test[test["variant_type"] == "control"].reset_index(drop=True)
test_pids = set(test_control["person_id"].tolist())
test_cand_ids = test_control.set_index("person_id")["candidate_id"].to_dict()
test_labels = labels[labels["person_id"].isin(test_pids)].copy()

n_test = len(test_labels)
X_test = np.empty((n_test, 769), dtype=np.float32)
y_test = np.empty(n_test, dtype=np.int8)
for i, (_, row) in enumerate(test_labels.iterrows()):
    pid = row["person_id"]
    jd_id = row["jd_id"]
    cand_id = test_cand_ids[pid]
    r_vec = resume_vecs[candidate_id_to_row[cand_id]]
    j_vec = jd_vecs[jd_id_to_row[jd_id]]
    X_test[i, :384] = r_vec
    X_test[i, 384:768] = j_vec
    X_test[i, 768] = float(np.dot(r_vec, j_vec))
    y_test[i] = row["label"]

def eval_clf(clf, name):
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

m_uniform = eval_clf(clf_uniform, "uniform_weighted")
m_reason = eval_clf(clf_reason, "reason_aware")

baseline_metrics_path = MODEL_DIR / "baseline_metrics.json"
if baseline_metrics_path.exists():
    with open(baseline_metrics_path) as f:
        m_baseline = json.load(f)
else:
    m_baseline = {"accuracy": None, "precision": None, "recall": None, "f1": None, "roc_auc": None}

print(f"\n  {'metric':<10s} {'baseline':>10s} {'uniform':>10s} {'reason':>10s}")
for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    bv = m_baseline.get(k)
    bv_str = f"{bv:.4f}" if isinstance(bv, (int, float)) else "  n/a"
    print(f"  {k:<10s} {bv_str:>10s} {m_uniform[k]:>10.4f} {m_reason[k]:>10.4f}")

# -----------------------------------------------------------------------------
# Step 7: save
# -----------------------------------------------------------------------------
section("STEP 7: save")
joblib.dump(clf_uniform, UNIFORM_MODEL_PATH)
joblib.dump(clf_reason, REASON_MODEL_PATH)
print(f"  saved {UNIFORM_MODEL_PATH}")
print(f"  saved {REASON_MODEL_PATH}")

with open(METRICS_PATH, "w") as f:
    json.dump({
        "train_subsample_size": TRAIN_SUBSAMPLE,
        "train_pairs": int(n_total),
        "method_uniform": {
            "config": {"uniform_gap_weight": UNIFORM_GAP_WEIGHT},
            "test_metrics": m_uniform,
        },
        "method_reason_aware": {
            "config": {"reason_weights": REASON_AWARE_WEIGHTS},
            "test_metrics": m_reason,
        },
        "baseline_for_reference": m_baseline,
    }, f, indent=2)
print(f"  saved {METRICS_PATH}")

# -----------------------------------------------------------------------------
# Step 8: sanity check on coefficients
# -----------------------------------------------------------------------------
section("STEP 8: sanity check on coefficient vectors")
def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

baseline_path = MODEL_DIR / "baseline_ranker.joblib"
if baseline_path.exists():
    baseline_clf = joblib.load(baseline_path)
    sim_b_u = cosine(baseline_clf.coef_[0], clf_uniform.coef_[0])
    sim_b_r = cosine(baseline_clf.coef_[0], clf_reason.coef_[0])
    sim_u_r = cosine(clf_uniform.coef_[0], clf_reason.coef_[0])
    print(f"  cosine similarity of coefficient vectors:")
    print(f"    baseline vs uniform:    {sim_b_u:.4f}")
    print(f"    baseline vs reason:     {sim_b_r:.4f}")
    print(f"    uniform vs reason:      {sim_u_r:.4f}")
    if sim_b_u > 0.9999 or sim_b_r > 0.9999:
        print(f"  WARNING: coefficient vectors are essentially identical to baseline.")
        print(f"  weighting may not have had the intended effect.")

section("DONE")
print(f"  next: audit_mitigated.py to compare baseline vs uniform vs reason-aware on the gap audit.")

sys.stdout = sys.__stdout__
log_file.close()