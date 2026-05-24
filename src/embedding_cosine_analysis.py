"""
embedding_cosine_analysis.py
============================
Mechanistic analysis: WHY does the disclosure paradox happen?

We test the hypothesis (Section 5.2 of the paper) that the embedding model
maps high-penalty reason phrases ("Pursuing MBA", "Caring for young children")
closer to junior/student archetypes than low-penalty phrases.

For each of the 5 gap-reason categories, we:
  1. Encode the full set of reason templates used in the audit
  2. Encode 4 archetype prototype sets (senior / mid / junior / student)
  3. Compute mean cosine similarity from each reason template to each archetype
  4. Also compute mean cosine similarity from each reason to the 26 JDs
  5. Test: is "more penalized = closer to junior/student"?

OUTPUTS:
    data/processed/audit/embedding_cosine_analysis.csv
    outputs/figures/fig5_embedding_proximity.png
    outputs/11_embedding_analysis_log.txt

Run from repo root:
    python src/embedding_cosine_analysis.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import json
import sys
import time
from sentence_transformers import SentenceTransformer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"

# Reason templates — taken from inject_gaps.py
# These are the exact phrases used to generate the audit dataset.
REASON_TEMPLATES = {
    "no_reason": [],   # no career-break entry; nothing to embed
    "caregiving": [
        "Career break — Family caregiving responsibilities",
        "Career break — Caring for young children",
        "Career break — Eldercare responsibilities",
        "Career break — Full-time parental responsibilities",
    ],
    "health": [
        "Career break — Health recovery",
        "Career break — Medical leave and recovery",
        "Career break — Personal health reasons",
    ],
    "layoff": [
        "Career break — Following position elimination, actively seeking new role",
        "Career break — Company restructuring, transitioning to new opportunities",
        "Career break — Layoff during organisational changes, in active job search",
    ],
    "education": [
        "Career break — Pursuing MBA",
        "Career break — Full-time graduate studies",
        "Career break — Professional certification programme",
        "Career break — Master's degree (full-time)",
    ],
}

# Archetype prototypes — 3 per archetype. Average embedding represents the archetype.
# Design principles:
#   1. Realistic resume-style language, not artificial
#   2. Distinct semantic content per archetype (senior != junior)
#   3. Comparable length across archetypes (~30-50 words each) to avoid length-bias
ARCHETYPES = {
    "senior": [
        "Senior Software Engineer with over 10 years of experience leading distributed teams, "
        "architecting microservices platforms, and mentoring junior engineers across multiple "
        "product launches.",

        "Principal Engineer with deep expertise in scalable system design, having shipped "
        "production systems serving millions of users and led cross-functional initiatives "
        "spanning backend, infrastructure, and security.",

        "Engineering Manager with extensive experience driving technical strategy, owning "
        "platform reliability, and growing high-performing teams through hiring, mentorship, "
        "and architectural reviews.",
    ],
    "mid_level": [
        "Software Engineer with five years of professional experience building backend "
        "services in Java and Python, participating in code reviews, and contributing to "
        "team agile ceremonies.",

        "Full-Stack Developer with mid-career experience delivering features across React "
        "frontends and Node.js backends, collaborating with designers and product managers "
        "on production releases.",

        "Backend Developer with four years of industry experience working on REST APIs, "
        "database design, and continuous integration pipelines in agile teams.",
    ],
    "junior": [
        "Recent computer science graduate with internship experience, eager to learn from "
        "senior engineers and contribute to entry-level developer projects in a supportive "
        "team environment.",

        "Entry-level developer seeking first full-time role after completing a coding "
        "bootcamp, with project experience in JavaScript and Python through coursework "
        "and personal projects.",

        "Junior Software Engineer with one year of professional experience, currently "
        "building skills in modern frameworks and learning best practices for code quality "
        "and version control.",
    ],
    "student": [
        "Full-time graduate student pursuing a Master's degree in Computer Science, "
        "completing advanced coursework in machine learning, distributed systems, and "
        "algorithms while working on research projects.",

        "MBA candidate at a full-time business school programme, balancing case studies, "
        "team projects, and coursework in finance, strategy, and operations management.",

        "PhD student in computer science focused on dissertation research, with teaching "
        "assistantship duties and academic publications in peer-reviewed conferences.",
    ],
}

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = REPO_ROOT / "data" / "processed"
AUDIT_DIR = PROC_DIR / "audit"
EMB_DIR = PROC_DIR / "embeddings"
OUT_DIR = REPO_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)
AUDIT_DIR.mkdir(exist_ok=True, parents=True)

JD_VECTORS = EMB_DIR / "jd_vectors.npy"
JD_INDEX = EMB_DIR / "jd_index.csv"
RESULTS_CSV = AUDIT_DIR / "embedding_cosine_analysis.csv"
FIG_PATH = FIG_DIR / "fig5_embedding_proximity.png"
LOG_PATH = OUT_DIR / "11_embedding_analysis_log.txt"

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
# Step 1: load model and JD embeddings (cached)
# -----------------------------------------------------------------------------
section("STEP 1: load model + cached JD embeddings")
t0 = time.time()
model = SentenceTransformer(MODEL_NAME)
print(f"  loaded {MODEL_NAME} in {time.time()-t0:.1f}s")

jd_vecs = np.load(JD_VECTORS)
jd_ids = pd.read_csv(JD_INDEX)["jd_id"].tolist()
print(f"  loaded {len(jd_ids)} JD vectors of shape {jd_vecs.shape}")

# -----------------------------------------------------------------------------
# Step 2: embed reason templates
# -----------------------------------------------------------------------------
section("STEP 2: embed reason templates")
reason_vecs = {}
for reason, templates in REASON_TEMPLATES.items():
    if not templates:
        print(f"  {reason}: skipping (no templates)")
        continue
    vecs = model.encode(templates, convert_to_numpy=True).astype(np.float32)
    # mean across templates = archetype-of-reason embedding
    mean_vec = vecs.mean(axis=0)
    mean_vec = mean_vec / np.linalg.norm(mean_vec)  # re-normalise after averaging
    reason_vecs[reason] = mean_vec
    print(f"  {reason}: {len(templates)} templates → mean vector (norm={np.linalg.norm(mean_vec):.4f})")

# -----------------------------------------------------------------------------
# Step 3: embed archetype prototypes
# -----------------------------------------------------------------------------
section("STEP 3: embed archetype prototypes")
archetype_vecs = {}
for archetype, prototypes in ARCHETYPES.items():
    vecs = model.encode(prototypes, convert_to_numpy=True).astype(np.float32)
    mean_vec = vecs.mean(axis=0)
    mean_vec = mean_vec / np.linalg.norm(mean_vec)
    archetype_vecs[archetype] = mean_vec
    print(f"  {archetype}: {len(prototypes)} prototypes → mean vector (norm={np.linalg.norm(mean_vec):.4f})")

# -----------------------------------------------------------------------------
# Step 4: pairwise cosine similarities
# -----------------------------------------------------------------------------
section("STEP 4: compute cosine similarities")

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Reason → archetype similarities
results = []
print(f"\n  Reason → archetype cosine similarity:")
print(f"  {'reason':<12s}  {'senior':>8s}  {'mid':>8s}  {'junior':>8s}  {'student':>8s}")
for reason in ["caregiving", "health", "layoff", "education"]:
    r_vec = reason_vecs[reason]
    sims = {a: cos(r_vec, archetype_vecs[a]) for a in ["senior", "mid_level", "junior", "student"]}
    print(f"  {reason:<12s}  {sims['senior']:>8.4f}  {sims['mid_level']:>8.4f}  "
          f"{sims['junior']:>8.4f}  {sims['student']:>8.4f}")
    for archetype, sim in sims.items():
        results.append({
            "reason": reason,
            "archetype": archetype,
            "cosine_similarity": round(sim, 6),
        })

# Reason → mean JD similarity
print(f"\n  Reason → mean JD cosine similarity:")
print(f"  {'reason':<12s}  {'mean cosine to JDs':>20s}")
for reason in ["caregiving", "health", "layoff", "education"]:
    r_vec = reason_vecs[reason]
    jd_sims = [cos(r_vec, jd_vecs[i]) for i in range(len(jd_ids))]
    mean_jd_sim = np.mean(jd_sims)
    print(f"  {reason:<12s}  {mean_jd_sim:>20.4f}")
    results.append({
        "reason": reason,
        "archetype": "mean_JD",
        "cosine_similarity": round(float(mean_jd_sim), 6),
    })

# Senior → JD similarities (reference baseline — what does a "good" embedding look like?)
print(f"\n  Reference baselines (archetype → mean JD):")
print(f"  {'archetype':<12s}  {'mean cosine to JDs':>20s}")
for archetype in ["senior", "mid_level", "junior", "student"]:
    a_vec = archetype_vecs[archetype]
    jd_sims = [cos(a_vec, jd_vecs[i]) for i in range(len(jd_ids))]
    mean_jd_sim = np.mean(jd_sims)
    print(f"  {archetype:<12s}  {mean_jd_sim:>20.4f}")

# -----------------------------------------------------------------------------
# Step 5: paper-side analysis
# -----------------------------------------------------------------------------
section("STEP 5: hypothesis testing")

baseline_eods = {
    "caregiving": 0.0583,
    "health":     -0.0038,
    "layoff":     0.0525,
    "education":  0.1027,
}

print(f"\n  Hypothesis: higher EOD (more bias) correlates with closer-to-student embedding.")
print(f"  Compute correlation between baseline EOD and (student - senior) cosine.\n")
print(f"  {'reason':<12s}  {'EOD':>8s}  {'cos(student)':>14s}  {'cos(senior)':>14s}  {'student-senior':>16s}")
deltas = []
eods = []
for reason in ["caregiving", "health", "layoff", "education"]:
    r_vec = reason_vecs[reason]
    s_sim = cos(r_vec, archetype_vecs["student"])
    sr_sim = cos(r_vec, archetype_vecs["senior"])
    delta = s_sim - sr_sim
    eod = baseline_eods[reason]
    eods.append(eod)
    deltas.append(delta)
    print(f"  {reason:<12s}  {eod:>+8.4f}  {s_sim:>14.4f}  {sr_sim:>14.4f}  {delta:>+16.4f}")

corr = np.corrcoef(eods, deltas)[0, 1]
print(f"\n  Pearson correlation (EOD vs student-senior delta): r = {corr:.4f}")
print(f"  Interpretation: positive correlation means high-EOD reasons are closer to student.")
if corr > 0.5:
    print(f"  >> Hypothesis supported: representational mechanism explains paradox.")
elif corr < -0.5:
    print(f"  >> Hypothesis contradicted: closer to senior associates with MORE bias (unexpected).")
else:
    print(f"  >> Hypothesis inconclusive: no clear correlation between archetype proximity and bias.")

# -----------------------------------------------------------------------------
# Step 6: write results CSV
# -----------------------------------------------------------------------------
section("STEP 6: write CSV")
pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
print(f"  wrote {RESULTS_CSV}")

# -----------------------------------------------------------------------------
# Step 7: figure
# -----------------------------------------------------------------------------
section("STEP 7: generate figure")

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

COLOURS = {
    "caregiving": "#e87b35",
    "health":     "#2ca44b",
    "layoff":     "#b8407a",
    "education":  "#8e44ad",
}

fig, ax = plt.subplots(figsize=(8, 5))

reasons = ["caregiving", "health", "layoff", "education"]
archetypes_ordered = ["senior", "mid_level", "junior", "student"]
x = np.arange(len(archetypes_ordered))
width = 0.20

for i, reason in enumerate(reasons):
    sims = [cos(reason_vecs[reason], archetype_vecs[a]) for a in archetypes_ordered]
    offset = (i - len(reasons)/2 + 0.5) * width
    ax.bar(x + offset, sims, width, label=reason, color=COLOURS[reason], alpha=0.85,
           edgecolor="white", linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(["senior", "mid-level", "junior", "student"])
ax.set_ylabel("Cosine similarity")
ax.set_xlabel("Archetype")
ax.set_title(
    "Reason templates' semantic proximity to candidate archetypes\n"
    "(higher cosine = closer in embedding space)",
    fontsize=11, pad=12,
)
ax.legend(loc="upper right", framealpha=0.95)
ax.set_ylim(0, max(0.5, ax.get_ylim()[1]))

plt.tight_layout()
plt.savefig(FIG_PATH)
plt.close()
print(f"  saved {FIG_PATH}")

section("DONE")
print(f"  results CSV: {RESULTS_CSV}")
print(f"  figure:      {FIG_PATH}")
print(f"  log:         {LOG_PATH}")

sys.stdout = sys.__stdout__
log_file.close()