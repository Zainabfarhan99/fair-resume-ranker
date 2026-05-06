"""
compute_labels.py
=================
Generate the (candidate, JD) -> label pairs that the ranker will train on.

METHODOLOGY (locked):
    1. For each candidate, clean their skill list:
       - lowercase + strip whitespace
       - keep only skills appearing >= 50 times in the full corpus
       - dedupe within person
    2. For each (candidate, JD) pair:
       - overlap = |JD_required_skills ∩ candidate_skills| / |JD_required_skills|
       - label = 1 if overlap >= 0.7, else 0
    3. Skills NEVER appear in resume_text — they are used only for label generation.
       This forces the ranker to predict the label from the work history alone,
       which is what the gap injection is designed to perturb.
    4. Labels are computed ONCE per (person, JD) pair from the raw person skill
       data — the label is identical across all 6 variants of a person, since
       the variants only modify work-history dates, not skills.

OUTPUTS:
    data/processed/labels.csv — columns: person_id, jd_id, overlap_score, label
    outputs/04_label_log.txt   — run log + diagnostics

Run from repo root:
    python src/compute_labels.py
"""

from pathlib import Path
import pandas as pd
import json
import sys

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SKILL_FREQ_CUTOFF = 50
MATCH_THRESHOLD = 0.50

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROC_DIR = REPO_ROOT / "data" / "processed"
OUT_DIR = REPO_ROOT / "outputs"
PROC_DIR.mkdir(exist_ok=True, parents=True)
OUT_DIR.mkdir(exist_ok=True)

LOG_PATH = OUT_DIR / "04_label_log.txt"
LABELS_CSV = PROC_DIR / "labels.csv"
JDS_JSON = PROC_DIR / "jds.json"
VARIANTS_CSV = PROC_DIR / "candidate_variants.csv"

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
# Step 1: load JDs and the candidate pool
# -----------------------------------------------------------------------------
section("STEP 1: load JDs and candidate pool")

with open(JDS_JSON) as f:
    jds = json.load(f)
print(f"  loaded {len(jds)} JDs from {JDS_JSON}")

# we only need labels for the 10K people in our experiment
variants = pd.read_csv(VARIANTS_CSV, usecols=["person_id"])
sampled_pids = set(variants["person_id"].unique())
print(f"  experiment pool: {len(sampled_pids):,} unique people")

# -----------------------------------------------------------------------------
# Step 2: load and clean person_skills
# -----------------------------------------------------------------------------
section("STEP 2: load + clean person_skills")

ps = pd.read_csv(RAW_DIR / "05_person_skills.csv")
print(f"  loaded {len(ps):,} raw person-skill rows")

# normalise
ps["skill"] = ps["skill"].astype(str).str.lower().str.strip()
print(f"  normalised (lowercase + strip)")

# compute corpus frequency over ALL people (not just our sample)
# this defines the canonical skill vocabulary
freq = ps["skill"].value_counts()
canonical_vocab = set(freq[freq >= SKILL_FREQ_CUTOFF].index)
print(f"  canonical vocabulary (skills with >= {SKILL_FREQ_CUTOFF} mentions): {len(canonical_vocab):,}")

# filter person_skills to canonical vocabulary and our sampled people
ps_clean = ps[
    ps["person_id"].isin(sampled_pids) & ps["skill"].isin(canonical_vocab)
].copy()

# dedupe within person — same person might list "java" twice
ps_clean = ps_clean.drop_duplicates(subset=["person_id", "skill"])
print(f"  after filter + dedupe: {len(ps_clean):,} (person, skill) rows")

# build a dict: person_id -> set of skills
person_skill_sets = (
    ps_clean.groupby("person_id")["skill"]
    .apply(set)
    .to_dict()
)
print(f"  built skill sets for {len(person_skill_sets):,} people")

# -----------------------------------------------------------------------------
# Step 3: compute labels for every (person, JD) pair
# -----------------------------------------------------------------------------
section(f"STEP 3: compute labels (threshold = {MATCH_THRESHOLD})")

records = []
people_with_no_skills = 0
for pid in sampled_pids:
    candidate_skills = person_skill_sets.get(pid, set())
    if not candidate_skills:
        people_with_no_skills += 1
    for jd in jds:
        jd_skills = set(jd["required_skills"])
        intersection = candidate_skills & jd_skills
        overlap = len(intersection) / len(jd_skills) if jd_skills else 0.0
        label = 1 if overlap >= MATCH_THRESHOLD else 0
        records.append({
            "person_id": pid,
            "jd_id": jd["jd_id"],
            "overlap_score": round(overlap, 4),
            "label": label,
        })

labels_df = pd.DataFrame(records)
print(f"  computed {len(labels_df):,} (person, JD) pairs")
print(f"  people with no canonical skills (label always 0): {people_with_no_skills:,}")

# -----------------------------------------------------------------------------
# Step 4: distribution diagnostics
# -----------------------------------------------------------------------------
section("STEP 4: label distribution")

total = len(labels_df)
n_pos = labels_df["label"].sum()
n_neg = total - n_pos
print(f"  total pairs: {total:,}")
print(f"  label=1:     {n_pos:>7,}  ({n_pos/total*100:.2f}%)")
print(f"  label=0:     {n_neg:>7,}  ({n_neg/total*100:.2f}%)")

print(f"\n  candidate-level: how many JDs each person matches?")
matches_per_person = labels_df.groupby("person_id")["label"].sum()
print(f"    mean JDs matched per person:   {matches_per_person.mean():.2f}")
print(f"    median:                        {matches_per_person.median():.0f}")
print(f"    max:                           {matches_per_person.max()}")
print(f"\n    distribution:")
for n in [0, 1, 2, 3, 4, 5]:
    cnt = (matches_per_person == n).sum()
    print(f"      matches {n} JDs: {cnt:>6,} people  ({cnt/len(matches_per_person)*100:.1f}%)")
cnt = (matches_per_person >= 6).sum()
print(f"      matches 6+ JDs: {cnt:>6,} people  ({cnt/len(matches_per_person)*100:.1f}%)")

print(f"\n  per-JD: how many candidates each JD attracts?")
matches_per_jd = labels_df.groupby("jd_id")["label"].sum().sort_values(ascending=False)
print(f"    {'jd_id':<25s}  {'positives':>10s}  {'rate':>6s}")
for jd_id, n_match in matches_per_jd.items():
    rate = n_match / len(sampled_pids) * 100
    print(f"    {jd_id:<25s}  {n_match:>10,}  {rate:>5.1f}%")

# -----------------------------------------------------------------------------
# Step 5: write outputs
# -----------------------------------------------------------------------------
section("STEP 5: write outputs")
labels_df.to_csv(LABELS_CSV, index=False)
size_kb = LABELS_CSV.stat().st_size / 1024
print(f"  wrote {LABELS_CSV}  ({size_kb:.1f} KB)")

# -----------------------------------------------------------------------------
# Step 6: health check on the distribution
# -----------------------------------------------------------------------------
section("STEP 6: health check")

pos_rate = n_pos / total
print(f"  overall positive rate: {pos_rate*100:.2f}%")
if pos_rate < 0.05:
    print(f"  WARNING: positive rate < 5%. The ranker may have trouble learning.")
    print(f"  Consider lowering the match threshold (currently {MATCH_THRESHOLD}).")
elif pos_rate > 0.50:
    print(f"  WARNING: positive rate > 50%. The label may be too easy to satisfy.")
    print(f"  Consider raising the match threshold.")
else:
    print(f"  positive rate is in a healthy range (5-50%) for binary classification.")

zero_match_pct = (matches_per_person == 0).sum() / len(matches_per_person) * 100
print(f"  fraction of people matching ZERO JDs: {zero_match_pct:.1f}%")
if zero_match_pct > 50:
    print(f"  WARNING: more than half of people don't match any JD.")
    print(f"  Consider broadening JDs or lowering the threshold.")

section("DONE")
print(f"  data: {LABELS_CSV}")
print(f"  log:  {LOG_PATH}")

sys.stdout = sys.__stdout__
log_file.close()