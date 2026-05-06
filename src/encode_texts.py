"""
encode_texts.py
===============
Phase 1 of the ranker pipeline: encode all resume_text and JD description_text
into Sentence-BERT vectors. Cache the result so subsequent training/audit
scripts can reload instantly without re-encoding.

We use all-MiniLM-L6-v2: 384-dim embeddings, fast on CPU, well-validated.

OUTPUTS:
    data/processed/embeddings/resume_vectors.npy   -- shape (60000, 384), float32
    data/processed/embeddings/resume_index.csv     -- candidate_id, person_id, variant_type (row order matches resume_vectors)
    data/processed/embeddings/jd_vectors.npy       -- shape (26, 384), float32
    data/processed/embeddings/jd_index.csv         -- jd_id (row order matches jd_vectors)
    outputs/05_encoding_log.txt                    -- run log

Run from repo root:
    python src/encode_texts.py

Expected runtime:
    First run: 30-60 min (encoding 60K resumes on CPU)
    Re-runs:   skipped if cached files exist (delete them to force re-encoding)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
import sys
import time
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64        # CPU-friendly; SBERT defaults to 32 but 64 works well on M-series
SHOW_PROGRESS = True

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = REPO_ROOT / "data" / "processed"
EMB_DIR = PROC_DIR / "embeddings"
OUT_DIR = REPO_ROOT / "outputs"
EMB_DIR.mkdir(exist_ok=True, parents=True)
OUT_DIR.mkdir(exist_ok=True)

VARIANTS_CSV = PROC_DIR / "candidate_variants.csv"
JDS_JSON = PROC_DIR / "jds.json"
RESUME_VECTORS = EMB_DIR / "resume_vectors.npy"
RESUME_INDEX = EMB_DIR / "resume_index.csv"
JD_VECTORS = EMB_DIR / "jd_vectors.npy"
JD_INDEX = EMB_DIR / "jd_index.csv"
LOG_PATH = OUT_DIR / "05_encoding_log.txt"

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
existing = [p for p in [RESUME_VECTORS, RESUME_INDEX, JD_VECTORS, JD_INDEX] if p.exists()]
if len(existing) == 4:
    print(f"  all four cache files already exist — skipping encoding.")
    print(f"  delete files in {EMB_DIR} to force re-encoding.")
    print(f"\n  cache contents:")
    for p in [RESUME_VECTORS, JD_VECTORS]:
        arr = np.load(p)
        print(f"    {p.name}: shape={arr.shape}, dtype={arr.dtype}")
    sys.stdout = sys.__stdout__
    log_file.close()
    sys.exit(0)
else:
    print(f"  cache incomplete or missing — proceeding with encoding.")

# -----------------------------------------------------------------------------
# Step 2: load model
# -----------------------------------------------------------------------------
section(f"STEP 2: load Sentence-BERT model ({MODEL_NAME})")
start = time.time()
model = SentenceTransformer(MODEL_NAME)
print(f"  loaded in {time.time()-start:.1f}s")
print(f"  embedding dim: {model.get_sentence_embedding_dimension()}")

# -----------------------------------------------------------------------------
# Step 3: encode resumes
# -----------------------------------------------------------------------------
section("STEP 3: encode resumes")
print(f"  reading {VARIANTS_CSV}...")
variants = pd.read_csv(VARIANTS_CSV)
print(f"  loaded {len(variants):,} variants")

# preserve a deterministic order: candidate_id ascending
variants = variants.sort_values("candidate_id").reset_index(drop=True)

resume_texts = variants["resume_text"].tolist()
print(f"  encoding {len(resume_texts):,} resumes (batch_size={BATCH_SIZE})...")
print(f"  this is the slow step. expected ~30-60 min on CPU.")
start = time.time()
resume_vecs = model.encode(
    resume_texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=SHOW_PROGRESS,
    convert_to_numpy=True,
).astype(np.float32)
elapsed = time.time() - start
print(f"  done in {elapsed/60:.1f} min  ({len(resume_texts)/elapsed:.1f} resumes/sec)")
print(f"  resulting shape: {resume_vecs.shape}, dtype: {resume_vecs.dtype}")

# save vectors + index
np.save(RESUME_VECTORS, resume_vecs)
variants[["candidate_id", "person_id", "variant_type", "gap_reason", "gap_duration_months"]].to_csv(
    RESUME_INDEX, index=False
)
print(f"  saved {RESUME_VECTORS} ({RESUME_VECTORS.stat().st_size / (1024*1024):.1f} MB)")
print(f"  saved {RESUME_INDEX}")

# -----------------------------------------------------------------------------
# Step 4: encode JDs
# -----------------------------------------------------------------------------
section("STEP 4: encode JDs")
with open(JDS_JSON) as f:
    jds = json.load(f)

# deterministic order: jd_id alphabetical
jds_sorted = sorted(jds, key=lambda j: j["jd_id"])
jd_texts = [j["description_text"] for j in jds_sorted]
print(f"  encoding {len(jd_texts)} JDs...")
start = time.time()
jd_vecs = model.encode(
    jd_texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=False,
    convert_to_numpy=True,
).astype(np.float32)
print(f"  done in {time.time()-start:.1f}s")

np.save(JD_VECTORS, jd_vecs)
pd.DataFrame({"jd_id": [j["jd_id"] for j in jds_sorted]}).to_csv(JD_INDEX, index=False)
print(f"  saved {JD_VECTORS} ({JD_VECTORS.stat().st_size / 1024:.1f} KB)")
print(f"  saved {JD_INDEX}")

# -----------------------------------------------------------------------------
# Step 5: sanity check
# -----------------------------------------------------------------------------
section("STEP 5: sanity check")
# verify a couple of obvious properties of the embeddings:
# 1. norms should all be reasonable (not zero, not exploding)
# 2. cosine similarity between an obvious pair should be high
resume_norms = np.linalg.norm(resume_vecs, axis=1)
print(f"  resume vector norms: min={resume_norms.min():.3f}  "
      f"mean={resume_norms.mean():.3f}  max={resume_norms.max():.3f}")
jd_norms = np.linalg.norm(jd_vecs, axis=1)
print(f"  JD vector norms:     min={jd_norms.min():.3f}  "
      f"mean={jd_norms.mean():.3f}  max={jd_norms.max():.3f}")

# dummy cosine test: java_backend_dev JD should be more similar to a Java resume
# than to a, say, frontend resume. just a smoke test.
def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

jd_idx_lookup = {j: i for i, j in enumerate([k["jd_id"] for k in jds_sorted])}
java_jd_vec = jd_vecs[jd_idx_lookup["java_backend_dev"]]
frontend_jd_vec = jd_vecs[jd_idx_lookup["frontend_dev"]]
sim = cosine(java_jd_vec, frontend_jd_vec)
print(f"  cosine(java_backend_dev_JD, frontend_dev_JD) = {sim:.3f}")
print(f"  (expected ~0.4-0.7: both are tech JDs but distinct roles)")

section("DONE")
print(f"  cached files in: {EMB_DIR}")
print(f"  next: train_baseline_ranker.py")
print(f"  total runtime: {(time.time() - start)/60:.1f} min")

sys.stdout = sys.__stdout__
log_file.close()