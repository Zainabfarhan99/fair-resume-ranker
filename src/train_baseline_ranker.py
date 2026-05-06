"""
train_baseline_ranker.py
========================
Phase 2: train the baseline ranker.

Architecture (locked):
    feature = [resume_vec (384) ; jd_vec (384) ; cosine(resume, jd) (1)] = 769 dims
    classifier = Logistic Regression (sklearn, deterministic)
    label = match label from compute_labels.py (1 if skill-overlap >= 0.5 else 0)

Training set (locked T2 decision):
    Use ONLY control variants of train-set people.
    Pairs: 8,001 train people × 26 JDs = 208,026 pairs
    The model never sees gap-injected variants during training.
    This is what makes the ranker "naive" — it represents a typical ATS,
    which mitigation methods will later try to de-bias.

Test evaluation:
    Use control variants of test-set people for headline metrics.
    The full audit (all 6 variants per test person) is run separately
    in a later script.

OUTPUTS:
    data/processed/models/baseline_ranker.joblib  -- the trained logistic regression
    data/processed/models/baseline_metrics.json   -- accuracy, precision, recall, F1, ROC-AUC
    outputs/06_baseline_training_log.txt          -- run log

Run from repo root:
    python src/train_baseline_ranker.py
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
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
LR_MAX_ITER = 200
LR_C = 1.0  # inverse regularisation strength (sklearn default)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = REPO_ROOT / "data" / "processed"
EMB_DIR = PROC_DIR / "embeddings"
MODEL_DIR = PROC_DIR / "models"
OUT_DIR = REPO_ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True, parents=True)
OUT_DIR.mkdir(exist_ok=True)

RESUME_VECTORS = EMB_DIR / "resume_vectors.npy"
RESUME_INDEX = EMB_DIR / "resume_index.csv"
JD_VECTORS = EMB_DIR / "jd_vectors.npy"
JD_INDEX = EMB_DIR / "jd_index.csv"
LABELS_CSV = PROC_DIR / "labels.csv"
TRAIN_CSV = PROC_DIR / "train.csv"
TEST_CSV = PROC_DIR / "test.csv"

MODEL_PATH = MODEL_DIR / "baseline_ranker.joblib"
METRICS_PATH = MODEL_DIR / "baseline_metrics.json"
LOG_PATH = OUT_DIR / "06_baseline_training_log.txt"

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

np.random.seed(RANDOM_SEED)

# -----------------------------------------------------------------------------
# Step 1: load embeddings + indices + labels
# -----------------------------------------------------------------------------
section("STEP 1: load cached embeddings and labels")
t0 = time.time()
resume_vecs = np.load(RESUME_VECTORS)
jd_vecs = np.load(JD_VECTORS)
resume_index = pd.read_csv(RESUME_INDEX)
jd_index = pd.read_csv(JD_INDEX)
labels = pd.read_csv(LABELS_CSV)
train = pd.read_csv(TRAIN_CSV, usecols=["candidate_id", "person_id", "variant_type"])
test = pd.read_csv(TEST_CSV, usecols=["candidate_id", "person_id", "variant_type"])
print(f"  resume_vecs: {resume_vecs.shape}, dtype={resume_vecs.dtype}")
print(f"  jd_vecs:     {jd_vecs.shape}, dtype={jd_vecs.dtype}")
print(f"  labels:      {len(labels):,} (person, JD) pairs")
print(f"  train: {len(train):,} variants  ({train['person_id'].nunique():,} people)")
print(f"  test:  {len(test):,} variants  ({test['person_id'].nunique():,} people)")
print(f"  loaded in {time.time()-t0:.1f}s")

# build fast lookups
# row index in resume_vecs is the same order as resume_index's candidate_id (sorted ascending)
candidate_id_to_row = dict(zip(resume_index["candidate_id"].tolist(),
                               range(len(resume_index))))
jd_id_to_row = dict(zip(jd_index["jd_id"].tolist(), range(len(jd_index))))

# also build candidate_id -> variant info dict for filtering
cand_info = resume_index.set_index("candidate_id")[["person_id", "variant_type"]].to_dict("index")

# -----------------------------------------------------------------------------
# Step 2: build training pairs
# -----------------------------------------------------------------------------
section("STEP 2: build training pairs (control variants of train people only)")

# Train set: control variants of train people
train_control = train[train["variant_type"] == "control"].copy()
train_pids = set(train_control["person_id"].tolist())
print(f"  train people: {len(train_pids):,}")
print(f"  train control variants: {len(train_control):,}")

# get the control candidate_id for each train person
train_cand_ids = train_control.set_index("person_id")["candidate_id"].to_dict()

# for each (train person × JD): build feature, attach label
train_labels = labels[labels["person_id"].isin(train_pids)].copy()
print(f"  train (person × JD) pairs: {len(train_labels):,}")

t0 = time.time()
n_train = len(train_labels)
X_train = np.empty((n_train, 769), dtype=np.float32)
y_train = np.empty(n_train, dtype=np.int8)
for i, (_, row) in enumerate(train_labels.iterrows()):
    pid = row["person_id"]
    jd_id = row["jd_id"]
    cand_id = train_cand_ids[pid]
    r_vec = resume_vecs[candidate_id_to_row[cand_id]]
    j_vec = jd_vecs[jd_id_to_row[jd_id]]
    cos_sim = float(np.dot(r_vec, j_vec))   # both unit-normalised → dot = cosine
    X_train[i, :384] = r_vec
    X_train[i, 384:768] = j_vec
    X_train[i, 768] = cos_sim
    y_train[i] = row["label"]
print(f"  built X_train shape={X_train.shape} in {time.time()-t0:.1f}s")
print(f"  train label balance: positives={y_train.sum():,}  ({y_train.mean()*100:.2f}%)")

# -----------------------------------------------------------------------------
# Step 3: build test pairs (control variants of test people)
# -----------------------------------------------------------------------------
section("STEP 3: build test pairs (control variants of test people)")
test_control = test[test["variant_type"] == "control"].copy()
test_pids = set(test_control["person_id"].tolist())
test_cand_ids = test_control.set_index("person_id")["candidate_id"].to_dict()
test_labels = labels[labels["person_id"].isin(test_pids)].copy()
print(f"  test people: {len(test_pids):,}")
print(f"  test (person × JD) pairs: {len(test_labels):,}")

t0 = time.time()
n_test = len(test_labels)
X_test = np.empty((n_test, 769), dtype=np.float32)
y_test = np.empty(n_test, dtype=np.int8)
for i, (_, row) in enumerate(test_labels.iterrows()):
    pid = row["person_id"]
    jd_id = row["jd_id"]
    cand_id = test_cand_ids[pid]
    r_vec = resume_vecs[candidate_id_to_row[cand_id]]
    j_vec = jd_vecs[jd_id_to_row[jd_id]]
    cos_sim = float(np.dot(r_vec, j_vec))
    X_test[i, :384] = r_vec
    X_test[i, 384:768] = j_vec
    X_test[i, 768] = cos_sim
    y_test[i] = row["label"]
print(f"  built X_test shape={X_test.shape} in {time.time()-t0:.1f}s")
print(f"  test label balance: positives={y_test.sum():,}  ({y_test.mean()*100:.2f}%)")

# -----------------------------------------------------------------------------
# Step 4: train logistic regression
# -----------------------------------------------------------------------------
section("STEP 4: train logistic regression")
clf = LogisticRegression(
    max_iter=LR_MAX_ITER,
    C=LR_C,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)
t0 = time.time()
clf.fit(X_train, y_train)
print(f"  trained in {time.time()-t0:.1f}s")
print(f"  converged: {clf.n_iter_[0] < LR_MAX_ITER}  (n_iter={clf.n_iter_[0]})")

# -----------------------------------------------------------------------------
# Step 5: evaluate on test set
# -----------------------------------------------------------------------------
section("STEP 5: evaluate on test set")
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print(f"\n  metrics:")
print(f"    accuracy:  {acc:.4f}")
print(f"    precision: {prec:.4f}")
print(f"    recall:    {rec:.4f}")
print(f"    F1:        {f1:.4f}")
print(f"    ROC-AUC:   {auc:.4f}")

print(f"\n  confusion matrix (rows=true, cols=predicted):")
print(f"            pred=0  pred=1")
print(f"    true=0  {cm[0,0]:>6}  {cm[0,1]:>6}")
print(f"    true=1  {cm[1,0]:>6}  {cm[1,1]:>6}")

print(f"\n  classification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# -----------------------------------------------------------------------------
# Step 6: save model and metrics
# -----------------------------------------------------------------------------
section("STEP 6: save artefacts")
joblib.dump(clf, MODEL_PATH)
print(f"  saved model: {MODEL_PATH}")

metrics = {
    "model": "LogisticRegression (baseline)",
    "train_pairs": int(n_train),
    "train_positives": int(y_train.sum()),
    "test_pairs": int(n_test),
    "test_positives": int(y_test.sum()),
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1": float(f1),
    "roc_auc": float(auc),
    "confusion_matrix": cm.tolist(),
    "n_iter": int(clf.n_iter_[0]),
    "C": LR_C,
    "random_seed": RANDOM_SEED,
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"  saved metrics: {METRICS_PATH}")

# -----------------------------------------------------------------------------
# Step 7: health check
# -----------------------------------------------------------------------------
section("STEP 7: health check")
if acc < 0.55:
    print(f"  WARNING: accuracy {acc:.3f} is barely above chance.")
    print(f"  the model isn't learning meaningful signal. investigate before audit.")
elif acc > 0.95:
    print(f"  WARNING: accuracy {acc:.3f} is suspiciously high.")
    print(f"  possible data leakage or label is too easy. investigate.")
else:
    print(f"  accuracy {acc:.3f} is in a healthy range (0.55-0.95). proceed to audit.")

if auc < 0.6:
    print(f"  WARNING: ROC-AUC {auc:.3f} is poor. model has weak ranking ability.")
elif auc >= 0.8:
    print(f"  good ROC-AUC {auc:.3f}. model ranks meaningfully.")

section("DONE")
print(f"  next: build the audit script that runs the trained ranker on all 6 variants per test person.")

sys.stdout = sys.__stdout__
log_file.close()