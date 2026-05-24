"""
train_all_grid_models.py
========================
Train the 3 new models in the 2x2 architecture-robustness grid for Section 4.7.

The existing baseline (MiniLM + LR) is already trained in baseline_ranker.joblib.
This script adds:
  - mpnet + LR
  - MiniLM + XGBoost
  - mpnet + XGBoost

All four models use the same feature construction: [resume_emb; jd_emb; cosine_sim].
All four models are trained ONLY on control variants (8001 candidates × 26 JDs).

OUTPUTS:
    data/processed/models/mpnet_lr.joblib
    data/processed/models/minilm_xgb.joblib
    data/processed/models/mpnet_xgb.joblib
    outputs/17_grid_training_log.txt

Run from repo root:
    python src/train_all_grid_models.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys
import time
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

try:
    from xgboost import XGBClassifier
except ImportError:
    print("ERROR: xgboost not installed. Run:  pip install xgboost --break-system-packages")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = REPO_ROOT / "data" / "processed"
EMB_DIR = PROC_DIR / "embeddings"
MODEL_DIR = PROC_DIR / "models"
OUT_DIR = REPO_ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

LABELS_CSV = PROC_DIR / "labels.csv"
TEST_CSV = PROC_DIR / "test.csv"
TRAIN_CSV = PROC_DIR / "train.csv"

# embedding caches
EMB_MINILM_RESUME = EMB_DIR / "resume_vectors.npy"
EMB_MINILM_RESUME_IDX = EMB_DIR / "resume_index.csv"
EMB_MINILM_JD = EMB_DIR / "jd_vectors.npy"
EMB_MINILM_JD_IDX = EMB_DIR / "jd_index.csv"

EMB_MPNET_RESUME = EMB_DIR / "resume_vectors_mpnet.npy"
EMB_MPNET_RESUME_IDX = EMB_DIR / "resume_index_mpnet.csv"
EMB_MPNET_JD = EMB_DIR / "jd_vectors_mpnet.npy"
EMB_MPNET_JD_IDX = EMB_DIR / "jd_index_mpnet.csv"

LOG_PATH = OUT_DIR / "17_grid_training_log.txt"

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
# Helper: build feature matrix for given embedding set
# -----------------------------------------------------------------------------
def build_features(persons_df, label_lookup, resume_vecs, resume_index,
                   jd_vecs, jd_index):
    """Build (X, y) for a given set of persons × all JDs."""
    cand_to_row = dict(zip(resume_index["candidate_id"].tolist(),
                           range(len(resume_index))))
    jd_ids = jd_index["jd_id"].tolist()
    n_jds = len(jd_ids)
    dim = resume_vecs.shape[1]
    feat_dim = 2 * dim + 1

    n_pairs = len(persons_df) * n_jds
    X = np.empty((n_pairs, feat_dim), dtype=np.float32)
    y = np.empty(n_pairs, dtype=np.int8)

    cand_ids = persons_df["candidate_id"].tolist()
    pids = persons_df["person_id"].tolist()
    resume_rows = np.array([cand_to_row[c] for c in cand_ids])
    R = resume_vecs[resume_rows]  # (n_persons, dim)

    idx = 0
    for i, pid in enumerate(pids):
        r_vec = R[i]
        for j, jd_id in enumerate(jd_ids):
            X[idx, :dim] = r_vec
            X[idx, dim:2*dim] = jd_vecs[j]
            X[idx, 2*dim] = float(np.dot(r_vec, jd_vecs[j]))
            y[idx] = label_lookup.get((pid, jd_id), 0)
            idx += 1
    return X, y

# -----------------------------------------------------------------------------
# Step 1: load shared resources
# -----------------------------------------------------------------------------
section("STEP 1: load shared data")
t0 = time.time()

labels = pd.read_csv(LABELS_CSV)
label_lookup = {(r["person_id"], r["jd_id"]): r["label"] for _, r in labels.iterrows()}
print(f"  loaded {len(label_lookup):,} (person, jd) labels")

train = pd.read_csv(TRAIN_CSV, usecols=["candidate_id", "person_id", "variant_type"])
test = pd.read_csv(TEST_CSV, usecols=["candidate_id", "person_id", "variant_type"])

train_control = train[train["variant_type"] == "control"].copy()
test_control = test[test["variant_type"] == "control"].copy()
print(f"  train control persons: {len(train_control):,}")
print(f"  test control persons:  {len(test_control):,}")

# Load both embedding sets
emb_minilm_resume = np.load(EMB_MINILM_RESUME)
emb_minilm_resume_idx = pd.read_csv(EMB_MINILM_RESUME_IDX)
emb_minilm_jd = np.load(EMB_MINILM_JD)
emb_minilm_jd_idx = pd.read_csv(EMB_MINILM_JD_IDX)

emb_mpnet_resume = np.load(EMB_MPNET_RESUME)
emb_mpnet_resume_idx = pd.read_csv(EMB_MPNET_RESUME_IDX)
emb_mpnet_jd = np.load(EMB_MPNET_JD)
emb_mpnet_jd_idx = pd.read_csv(EMB_MPNET_JD_IDX)

print(f"  MiniLM resume vecs: {emb_minilm_resume.shape}")
print(f"  mpnet  resume vecs: {emb_mpnet_resume.shape}")
print(f"  MiniLM JD vecs:     {emb_minilm_jd.shape}")
print(f"  mpnet  JD vecs:     {emb_mpnet_jd.shape}")
print(f"  loaded in {time.time()-t0:.1f}s")

# -----------------------------------------------------------------------------
# Step 2: train each cell of the 2x2 grid (skip existing baseline)
# -----------------------------------------------------------------------------
section("STEP 2: train 3 new grid cells")

CELLS = [
    ("mpnet_lr",     "mpnet",  "lr",  emb_mpnet_resume,  emb_mpnet_resume_idx,  emb_mpnet_jd,  emb_mpnet_jd_idx),
    ("minilm_xgb",   "minilm", "xgb", emb_minilm_resume, emb_minilm_resume_idx, emb_minilm_jd, emb_minilm_jd_idx),
    ("mpnet_xgb",    "mpnet",  "xgb", emb_mpnet_resume,  emb_mpnet_resume_idx,  emb_mpnet_jd,  emb_mpnet_jd_idx),
]

all_metrics = []

for cell_id, emb_name, clf_name, r_vecs, r_idx, j_vecs, j_idx in CELLS:
    print(f"\n  --- training cell: {cell_id} ---")
    t_cell = time.time()

    # Build training features (control variants only)
    print(f"    building training features...")
    X_train, y_train = build_features(train_control, label_lookup, r_vecs, r_idx, j_vecs, j_idx)
    print(f"    X_train shape: {X_train.shape}")
    print(f"    positives: {y_train.sum():,}  ({100*y_train.mean():.2f}%)")

    # Train
    print(f"    training {clf_name}...")
    t0 = time.time()
    if clf_name == "lr":
        clf = LogisticRegression(C=1.0, max_iter=200, random_state=42, n_jobs=1)
    elif clf_name == "xgb":
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            n_jobs=1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"    trained in {train_time:.1f}s")

    # Evaluate on test control variants
    X_test, y_test = build_features(test_control, label_lookup, r_vecs, r_idx, j_vecs, j_idx)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "cell": cell_id,
        "embedding": emb_name,
        "classifier": clf_name,
        "train_time_s": round(train_time, 2),
        "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
        "test_precision": round(precision_score(y_test, y_pred), 4),
        "test_recall": round(recall_score(y_test, y_pred), 4),
        "test_f1": round(f1_score(y_test, y_pred), 4),
        "test_roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    all_metrics.append(metrics)
    print(f"    test acc={metrics['test_accuracy']:.4f}  "
          f"prec={metrics['test_precision']:.4f}  "
          f"rec={metrics['test_recall']:.4f}  "
          f"f1={metrics['test_f1']:.4f}  "
          f"auc={metrics['test_roc_auc']:.4f}")

    # Save
    out_path = MODEL_DIR / f"{cell_id}.joblib"
    joblib.dump(clf, out_path)
    print(f"    saved {out_path}")
    print(f"    cell total time: {time.time()-t_cell:.1f}s")

    # Free memory for next cell
    del X_train, y_train, X_test, y_test, y_pred, y_proba

# -----------------------------------------------------------------------------
# Step 3: print summary table
# -----------------------------------------------------------------------------
section("STEP 3: grid summary (control-variant test performance)")
print(f"\n  {'cell':<14s}  {'emb':<7s}  {'clf':<5s}  {'acc':>6s}  {'prec':>6s}  {'rec':>6s}  {'f1':>6s}  {'auc':>6s}")
print(f"  {'-'*14}  {'-'*7}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

# Include baseline for comparison (from baseline_metrics.json)
import json
baseline_path = MODEL_DIR / "baseline_metrics.json"
if baseline_path.exists():
    with open(baseline_path) as f:
        baseline = json.load(f)
    print(f"  {'baseline':<14s}  {'minilm':<7s}  {'lr':<5s}  "
          f"{baseline.get('accuracy', 0):>6.4f}  "
          f"{baseline.get('precision', 0):>6.4f}  "
          f"{baseline.get('recall', 0):>6.4f}  "
          f"{baseline.get('f1', 0):>6.4f}  "
          f"{baseline.get('roc_auc', 0):>6.4f}")

for m in all_metrics:
    print(f"  {m['cell']:<14s}  {m['embedding']:<7s}  {m['classifier']:<5s}  "
          f"{m['test_accuracy']:>6.4f}  {m['test_precision']:>6.4f}  "
          f"{m['test_recall']:>6.4f}  {m['test_f1']:>6.4f}  {m['test_roc_auc']:>6.4f}")

# Save metrics
metrics_path = MODEL_DIR / "grid_training_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(all_metrics, f, indent=2)
print(f"\n  saved {metrics_path}")

section("DONE")
print(f"  next: audit_grid_models.py")

sys.stdout = sys.__stdout__
log_file.close()