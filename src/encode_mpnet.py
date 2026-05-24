"""
encode_mpnet.py
===============
Re-encode all 60K resume variants and 26 JDs with the larger mpnet-base-v2
embedding model (768-dim, ~110M parameters), in addition to the existing
all-MiniLM-L6-v2 cache (384-dim, ~22M parameters).

Used for the 2x2 architecture-robustness grid in Section 4.7:
  axis 1: MiniLM (existing) vs mpnet (new)
  axis 2: Logistic Regression vs XGBoost

OUTPUTS:
    data/processed/embeddings/resume_vectors_mpnet.npy
    data/processed/embeddings/jd_vectors_mpnet.npy
    outputs/16_mpnet_encoding_log.txt

Run from repo root:
    python src/encode_mpnet.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys
import time
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = REPO_ROOT / "data" / "processed"
EMB_DIR = PROC_DIR / "embeddings"
OUT_DIR = REPO_ROOT / "outputs"
EMB_DIR.mkdir(exist_ok=True, parents=True)

CANDIDATE_VARIANTS = PROC_DIR / "candidate_variants.csv"
JD_JSON = PROC_DIR / "jds.json"
RESUME_OUT = EMB_DIR / "resume_vectors_mpnet.npy"
RESUME_INDEX_OUT = EMB_DIR / "resume_index_mpnet.csv"
JD_OUT = EMB_DIR / "jd_vectors_mpnet.npy"
JD_INDEX_OUT = EMB_DIR / "jd_index_mpnet.csv"
LOG_PATH = OUT_DIR / "16_mpnet_encoding_log.txt"

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

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
# Step 1: cache check
# -----------------------------------------------------------------------------
section("STEP 1: cache check")
if RESUME_OUT.exists() and JD_OUT.exists():
    print(f"  mpnet caches already exist:")
    print(f"    {RESUME_OUT}  ({RESUME_OUT.stat().st_size / 1024**2:.1f} MB)")
    print(f"    {JD_OUT}      ({JD_OUT.stat().st_size / 1024**2:.1f} MB)")
    print(f"  delete them to force re-encoding, or skip this script.")
    print(f"  exiting.")
    sys.exit(0)
print(f"  cache incomplete or missing — proceeding with encoding.")

# -----------------------------------------------------------------------------
# Step 2: load model
# -----------------------------------------------------------------------------
section("STEP 2: load mpnet-base-v2")
t0 = time.time()
model = SentenceTransformer(MODEL_NAME)
print(f"  loaded {MODEL_NAME} in {time.time()-t0:.1f}s")
print(f"  embedding dimension: {model.get_sentence_embedding_dimension()}")

# -----------------------------------------------------------------------------
# Step 3: encode resumes
# -----------------------------------------------------------------------------
section("STEP 3: encode resume variants")
print(f"  reading {CANDIDATE_VARIANTS}...")
variants = pd.read_csv(CANDIDATE_VARIANTS)
print(f"  loaded {len(variants):,} variants")
texts = variants["resume_text"].astype(str).tolist()

t0 = time.time()
print(f"  encoding {len(texts):,} resumes (batch_size=32)...")
print(f"  expected ~10-15 min on CPU.")
resume_vecs = model.encode(
    texts, batch_size=32, show_progress_bar=True,
    convert_to_numpy=True, normalize_embeddings=True,
).astype(np.float32)
dt = time.time() - t0
print(f"  done in {dt/60:.1f} min  ({len(texts)/dt:.1f} resumes/sec)")
print(f"  shape: {resume_vecs.shape}")

np.save(RESUME_OUT, resume_vecs)
print(f"  saved {RESUME_OUT}  ({RESUME_OUT.stat().st_size / 1024**2:.1f} MB)")
variants[["candidate_id", "person_id", "variant_type"]].to_csv(RESUME_INDEX_OUT, index=False)
print(f"  saved {RESUME_INDEX_OUT}")

# -----------------------------------------------------------------------------
# Step 4: encode JDs
# -----------------------------------------------------------------------------
section("STEP 4: encode JDs")
import json
with open(JD_JSON) as f:
    jds = json.load(f)
print(f"  loaded {len(jds)} JDs")

jd_texts = [jd["description_text"] for jd in jds]
t0 = time.time()
jd_vecs = model.encode(
    jd_texts, batch_size=32, show_progress_bar=False,
    convert_to_numpy=True, normalize_embeddings=True,
).astype(np.float32)
print(f"  done in {time.time()-t0:.1f}s")
np.save(JD_OUT, jd_vecs)
print(f"  saved {JD_OUT}  ({JD_OUT.stat().st_size / 1024:.1f} KB)")
pd.DataFrame({"jd_id": [jd["jd_id"] for jd in jds]}).to_csv(JD_INDEX_OUT, index=False)
print(f"  saved {JD_INDEX_OUT}")

# -----------------------------------------------------------------------------
# Step 5: sanity checks
# -----------------------------------------------------------------------------
section("STEP 5: sanity check")
print(f"  resume norms: min={np.linalg.norm(resume_vecs, axis=1).min():.4f}  "
      f"mean={np.linalg.norm(resume_vecs, axis=1).mean():.4f}  "
      f"max={np.linalg.norm(resume_vecs, axis=1).max():.4f}")
print(f"  JD norms:     min={np.linalg.norm(jd_vecs, axis=1).min():.4f}  "
      f"mean={np.linalg.norm(jd_vecs, axis=1).mean():.4f}  "
      f"max={np.linalg.norm(jd_vecs, axis=1).max():.4f}")

# Quick semantic sanity check — Java backend vs Frontend JDs should be moderately similar
jd_ids = [jd["jd_id"] for jd in jds]
try:
    i_java = jd_ids.index("java_backend_dev")
    i_front = jd_ids.index("frontend_dev")
    sim = float(np.dot(jd_vecs[i_java], jd_vecs[i_front]))
    print(f"  cosine(java_backend_dev_JD, frontend_dev_JD) = {sim:.3f}")
    print(f"  (expected ~0.4-0.7: tech JDs, distinct roles)")
except ValueError:
    pass

section("DONE")
print(f"  next: train_all_grid_models.py")

sys.stdout = sys.__stdout__
log_file.close()