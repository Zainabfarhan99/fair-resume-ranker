"""
explore_data.py
================
Day 1-2 of Week 1: load the 54K structured resume dataset, verify its shape,
and check whether we have what we need for career-gap detection.

Run from repo root:
    python src/explore_data.py

Outputs go to:
    - console (for immediate reading)
    - outputs/01_data_exploration_log.txt (so you have a record)
"""

from pathlib import Path
import pandas as pd
import sys

# -----------------------------------------------------------------------------
# Setup: paths and dual-output (print to both console and file)
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
OUT_DIR = REPO_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
LOG_PATH = OUT_DIR / "01_data_exploration_log.txt"

# tee: print to console AND log file at the same time
class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, msg):
        for s in self.streams: s.write(msg)
    def flush(self):
        for s in self.streams: s.flush()

log_file = open(LOG_PATH, "w")
sys.stdout = Tee(sys.__stdout__, log_file)

def section(title):
    """Visual divider so output is scannable."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

# -----------------------------------------------------------------------------
# Step 1: load all 6 CSVs
# -----------------------------------------------------------------------------
section("STEP 1: loading CSV files")

files = {
    "people":       RAW_DIR / "01_people.csv",
    "abilities":    RAW_DIR / "02_abilities.csv",
    "education":    RAW_DIR / "03_education.csv",
    "experience":   RAW_DIR / "04_experience.csv",
    "person_skills":RAW_DIR / "05_person_skills.csv",
    "skills":       RAW_DIR / "06_skills.csv",
}

dfs = {}
for name, path in files.items():
    dfs[name] = pd.read_csv(path)
    print(f"  loaded {name:15s} -> {len(dfs[name]):>7,} rows  |  cols: {list(dfs[name].columns)}")

# -----------------------------------------------------------------------------
# Step 2: verify the experience table has what we need for gap detection
# -----------------------------------------------------------------------------
section("STEP 2: experience table — date completeness")

exp = dfs["experience"]
total = len(exp)
have_start = exp["start_date"].notna().sum()
have_end   = exp["end_date"].notna().sum()
have_both  = (exp["start_date"].notna() & exp["end_date"].notna()).sum()
end_present = (exp["end_date"] == "Present").sum()

print(f"  total experience rows:           {total:>7,}")
print(f"  with start_date:                 {have_start:>7,}  ({have_start/total*100:.1f}%)")
print(f"  with end_date:                   {have_end:>7,}  ({have_end/total*100:.1f}%)")
print(f"  with BOTH dates:                 {have_both:>7,}  ({have_both/total*100:.1f}%)")
print(f"  end_date == 'Present':           {end_present:>7,}  ({end_present/total*100:.1f}%)")

# -----------------------------------------------------------------------------
# Step 3: how many jobs per person? (we need at least 2 to detect a gap)
# -----------------------------------------------------------------------------
section("STEP 3: jobs per person distribution")

jobs_per_person = exp.groupby("person_id").size()
print(f"  total unique people in experience table: {jobs_per_person.shape[0]:,}")
print(f"  mean jobs per person:    {jobs_per_person.mean():.2f}")
print(f"  median jobs per person:  {jobs_per_person.median():.0f}")
print(f"  max jobs per person:     {jobs_per_person.max()}")
print()
print("  distribution of jobs-per-person:")
buckets = [(1, 1), (2, 3), (4, 6), (7, 10), (11, 999)]
for lo, hi in buckets:
    n = ((jobs_per_person >= lo) & (jobs_per_person <= hi)).sum()
    label = f"{lo}" if lo == hi else f"{lo}-{hi}" if hi < 999 else f"{lo}+"
    print(f"    {label:>6} jobs:  {n:>6,} people  ({n/len(jobs_per_person)*100:.1f}%)")

# we need >= 2 jobs to even talk about a gap between consecutive jobs
multi_job_people = (jobs_per_person >= 2).sum()
print(f"\n  >= 2 jobs (eligible for gap analysis): {multi_job_people:,} people "
      f"({multi_job_people/len(jobs_per_person)*100:.1f}%)")

# -----------------------------------------------------------------------------
# Step 4: most common job titles and firms (rough signal of dataset quality)
# -----------------------------------------------------------------------------
section("STEP 4: top job titles and firms (quality sanity check)")

print("  top 15 job titles:")
print(exp["title"].value_counts().head(15).to_string())
print("\n  top 15 firms:")
print(exp["firm"].value_counts().head(15).to_string())

# -----------------------------------------------------------------------------
# Step 5: people table — what do we actually have per person?
# -----------------------------------------------------------------------------
section("STEP 5: people table — completeness")

ppl = dfs["people"]
print(f"  total people: {len(ppl):,}")
print(f"  columns: {list(ppl.columns)}")
print(f"\n  null counts per column:")
print(ppl.isna().sum().to_string())

# -----------------------------------------------------------------------------
# Step 6: education — useful for institute-tier analysis later
# -----------------------------------------------------------------------------
section("STEP 6: education table")

edu = dfs["education"]
print(f"  total education rows: {len(edu):,}")
print(f"  unique people with education: {edu['person_id'].nunique():,}")
print(f"\n  top 15 institutions:")
print(edu["institution"].value_counts().head(15).to_string())

# -----------------------------------------------------------------------------
# Step 7: reconstruct 3 sample full "resumes" so you can eyeball
# -----------------------------------------------------------------------------
section("STEP 7: 3 sample reconstructed resumes (eyeball check)")

# pick people who have multiple jobs so the sample is representative
sample_ids = jobs_per_person[jobs_per_person.between(3, 6)].sample(3, random_state=42).index.tolist()

for pid in sample_ids:
    print(f"\n  ---------- person_id={pid} ----------")
    print("  PERSON:")
    print(ppl[ppl["person_id"] == pid].to_string(index=False))

    print("\n  EXPERIENCE (sorted by start_date):")
    person_exp = exp[exp["person_id"] == pid].copy()
    # sort chronologically: parse MM/YYYY, treating 'Present' as far future
    person_exp["_sort"] = pd.to_datetime(person_exp["start_date"], format="%m/%Y", errors="coerce")
    person_exp = person_exp.sort_values("_sort", ascending=False)
    print(person_exp[["title", "firm", "start_date", "end_date", "location"]].to_string(index=False))

    print("\n  EDUCATION:")
    person_edu = edu[edu["person_id"] == pid]
    if len(person_edu) > 0:
        print(person_edu[["institution", "program", "start_date", "location"]].to_string(index=False))
    else:
        print("    (no education records)")

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
section("DONE")
print(f"  full log saved to: {LOG_PATH}")
log_file.close()