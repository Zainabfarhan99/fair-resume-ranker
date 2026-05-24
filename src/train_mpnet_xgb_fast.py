"""
train_mpnet_xgb_fast.py
=======================
Train the missing 4th cell (mpnet + XGBoost) using faster XGBoost settings:
  - tree_method='hist' (histogram-based, much faster on dense features)
  - n_jobs=-1 (multi-threaded; libomp now installed)
  - n_estimators=150 (vs 200 in the standard script)

The other 3 cells in the 2x2 grid are already trained from the previous run.
This script only fills the missing cell.

OUTPUTS:
    data/processed/models/mpnet_xgb.joblib
    outputs/17b_mpnet_xgb_log.txt

Run from repo root:
    python src/train_mpnet_xgb_fast.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys
import time
import joblib
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from xgboost import XGBClassifier

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = REPO_ROOT / "data" / "processed"
EMB_DIR = PROC_DIR / "embeddings"
MODEL_DIR = PROC_DIR / "models"
OUT_DIR = REPO_ROOT / "outputs"

LABELS_CSV = PROC_DIR / "labels.csv"
TRAIN_CSV = PROC_DIR / "train.csv"
TEST_CSV = PROC_DIR / "test.csv"

EMB_RESUME = EMB_DIR / "resume_vectors_mpnet.npy"
EMB_RESUME_IDX = EMB_DIR / "resume_index_mpnet.csv"
EMB_JD = EMB_DIR / "jd_vectors_mpnet.npy"
EMB_JD_IDX = EMB_DIR / "jd_index_mpnet.csv"

OUT_MODEL = MODEL_DIR / "mpnet_xgb.joblib"
LOG_PATH = OUT_DIR / "17b_mpnet_xgb_log.txt"

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
# Step 1: load
# -----------------------------------------------------------------------------
section("STEP 1: load")
t0 = time.time()

labels = pd.read_csv(LABELS_CSV)
label_lookup = {(r["person_id"], r["jd_id"]): r["label"] for _, r in labels.iterrows()}

train = pd.read_csv(TRAIN_CSV, usecols=["candidate_id", "person_id", "variant_type"])
test = pd.read_csv(TEST_CSV, usecols=["candidate_id", "person_id", "variant_type"])
train_control = train[train["variant_type"] == "control"].copy()
test_control = test[test["variant_type"] == "control"].copy()

r_vecs = np.load(EMB_RESUME)
r_idx = pd.read_csv(EMB_RESUME_IDX)
j_vecs = np.load(EMB_JD)
j_idx = pd.read_csv(EMB_JD_IDX)

cand_to_row = dict(zip(r_idx["candidate_id"].tolist(), range(len(r_idx))))
jd_ids = j_idx["jd_id"].tolist()
n_jds = len(jd_ids)
dim = r_vecs.shape[1]
print(f"  loaded in {time.time()-t0:.1f}s")
print(f"  embedding dim: {dim}, JDs: {n_jds}")

# -----------------------------------------------------------------------------
# Step 2: build features (control only)
# -----------------------------------------------------------------------------
section("STEP 2: build training features")
t0 = time.time()

def build_features(persons_df):
    cand_ids = persons_df["candidate_id"].tolist()
    pids = persons_df["person_id"].tolist()
    resume_rows = np.array([cand_to_row[c] for c in cand_ids])
    R = r_vecs[resume_rows]

    n_pairs = len(persons_df) * n_jds
    feat_dim = 2 * dim + 1
    X = np.empty((n_pairs, feat_dim), dtype=np.float32)
    y = np.empty(n_pairs, dtype=np.int8)

    idx = 0
    for i, pid in enumerate(pids):
        r_vec = R[i]
        for j, jd_id in enumerate(jd_ids):
            X[idx, :dim] = r_vec
            X[idx, dim:2*dim] = j_vecs[j]
            X[idx, 2*dim] = float(np.dot(r_vec, j_vecs[j]))
            y[idx] = label_lookup.get((pid, jd_id), 0)
            idx += 1
    return X, y

X_train, y_train = build_features(train_control)
print(f"  X_train: {X_train.shape}  positives: {y_train.sum():,} ({100*y_train.mean():.2f}%)")
print(f"  built in {time.time()-t0:.1f}s")

# -----------------------------------------------------------------------------
# Step 3: train with FAST settings
# -----------------------------------------------------------------------------
section("STEP 3: train mpnet_xgb (fast settings)")
print(f"  tree_method='hist', n_jobs=-1, n_estimators=150")
t0 = time.time()

clf = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",   # histogram-based: much faster on dense features
    n_jobs=-1,            # parallel (libomp now installed)
    random_state=42,
    eval_metric="logloss",
    verbosity=1,
)
clf.fit(X_train, y_train)
train_time = time.time() - t0
print(f"  trained in {train_time:.1f}s")

# -----------------------------------------------------------------------------
# Step 4: evaluate
# -----------------------------------------------------------------------------
section("STEP 4: evaluate on test control")
X_test, y_test = build_features(test_control)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

metrics = {
    "cell": "mpnet_xgb",
    "embedding": "mpnet",
    "classifier": "xgb",
    "train_time_s": round(train_time, 2),
    "test_accuracy":  round(accuracy_score(y_test, y_pred), 4),
    "test_precision": round(precision_score(y_test, y_pred), 4),
    "test_recall":    round(recall_score(y_test, y_pred), 4),
    "test_f1":        round(f1_score(y_test, y_pred), 4),
    "test_roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
}
print(f"  acc={metrics['test_accuracy']:.4f}  prec={metrics['test_precision']:.4f}  "
      f"rec={metrics['test_recall']:.4f}  f1={metrics['test_f1']:.4f}  "
      f"auc={metrics['test_roc_auc']:.4f}")

# -----------------------------------------------------------------------------
# Step 5: save
# -----------------------------------------------------------------------------
section("STEP 5: save")
joblib.dump(clf, OUT_MODEL)
print(f"  saved {OUT_MODEL}")

# Append to existing metrics file if present
existing = []
metrics_path = MODEL_DIR / "grid_training_metrics.json"
if metrics_path.exists():
    with open(metrics_path) as f:
        existing = json.load(f)
# remove any prior entry for this cell
existing = [m for m in existing if m.get("cell") != "mpnet_xgb"]
existing.append(metrics)
with open(metrics_path, "w") as f:
    json.dump(existing, f, indent=2)
print(f"  updated {metrics_path}")

section("DONE")
print(f"  next: audit_grid_models.py")

sys.stdout = sys.__stdout__
log_file.close()