"""
explore_skills.py
=================
Diagnostic before modifying inject_gaps to include skills.

Questions:
  1. How dirty is person_skills.csv really?
  2. Per person in our 10K subsample, what's the typical skill list size?
  3. What are the most common skills (clean signals to use in JDs)?
  4. What fraction of skills are usable vs garbage?

Run from repo root:
    python src/explore_skills.py
"""

from pathlib import Path
import pandas as pd
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROC_DIR = REPO_ROOT / "data" / "processed"
OUT_DIR = REPO_ROOT / "outputs"
LOG_PATH = OUT_DIR / "04_skills_diagnostic.txt"

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
section("STEP 1: load skill tables")
person_skills = pd.read_csv(RAW_DIR / "05_person_skills.csv")
abilities = pd.read_csv(RAW_DIR / "02_abilities.csv")
print(f"  person_skills: {len(person_skills):,} rows")
print(f"  abilities:     {len(abilities):,} rows")

# -----------------------------------------------------------------------------
section("STEP 2: skill string characteristics")

# normalise: lowercase + strip
person_skills["skill_norm"] = person_skills["skill"].astype(str).str.lower().str.strip()
unique_skills = person_skills["skill_norm"].unique()
print(f"  total unique skills (after lowercasing): {len(unique_skills):,}")

# skill string length distribution — long ones are usually resume fragments
lengths = person_skills["skill_norm"].str.len()
print(f"\n  skill string length:")
print(f"    mean:   {lengths.mean():.1f}")
print(f"    median: {lengths.median():.0f}")
print(f"    max:    {lengths.max()}")
print(f"    >50 chars: {(lengths > 50).sum():,}  ({(lengths>50).mean()*100:.1f}%)")
print(f"    >100 chars: {(lengths > 100).sum():,}  ({(lengths>100).mean()*100:.1f}%)")

# -----------------------------------------------------------------------------
section("STEP 3: most common skills (top 50 — these are the clean signals)")
top_skills = person_skills["skill_norm"].value_counts().head(50)
for skill, count in top_skills.items():
    print(f"  {count:>6}x  {skill[:80]}")

# -----------------------------------------------------------------------------
section("STEP 4: skills appearing only once or twice (the noise)")
skill_freq = person_skills["skill_norm"].value_counts()
once = (skill_freq == 1).sum()
twice = (skill_freq == 2).sum()
total = len(skill_freq)
print(f"  unique skills appearing exactly once:  {once:,}  ({once/total*100:.1f}%)")
print(f"  unique skills appearing exactly twice: {twice:,}  ({twice/total*100:.1f}%)")
print(f"  appearing 5+ times:                    {(skill_freq >= 5).sum():,}  ({(skill_freq >= 5).sum()/total*100:.1f}%)")
print(f"  appearing 50+ times (likely real tech skills): {(skill_freq >= 50).sum():,}")
print(f"  appearing 500+ times (definitely real):        {(skill_freq >= 500).sum():,}")

# -----------------------------------------------------------------------------
section("STEP 5: skills per person (in our 10K subsample)")
# load the subsample person IDs from the variants file
variants = pd.read_csv(PROC_DIR / "candidate_variants.csv", usecols=["person_id"])
sampled_pids = variants["person_id"].unique()
print(f"  sampled people in our experiment: {len(sampled_pids):,}")

ps_sample = person_skills[person_skills["person_id"].isin(sampled_pids)]
skills_per_person = ps_sample.groupby("person_id").size()
print(f"\n  skills per person (raw, no cleaning):")
print(f"    mean:    {skills_per_person.mean():.1f}")
print(f"    median:  {skills_per_person.median():.0f}")
print(f"    max:     {skills_per_person.max()}")
print(f"    people with 0 skills: {len(sampled_pids) - len(skills_per_person):,}")

# now after cleaning: only keep skills appearing in 5+ people total
common_skills = set(skill_freq[skill_freq >= 5].index)
ps_sample_clean = ps_sample[ps_sample["skill_norm"].isin(common_skills)]
skills_per_person_clean = ps_sample_clean.groupby("person_id").size()
print(f"\n  skills per person (after dropping rare skills, kept only skills with 5+ appearances):")
print(f"    mean:    {skills_per_person_clean.mean():.1f}")
print(f"    median:  {skills_per_person_clean.median():.0f}")
print(f"    max:     {skills_per_person_clean.max()}")

# -----------------------------------------------------------------------------
section("STEP 6: 5 sample people — their actual cleaned skill lists")
import random
random.seed(42)
sample_pids = random.sample(list(sampled_pids), 5)
for pid in sample_pids:
    skills = ps_sample_clean[ps_sample_clean["person_id"] == pid]["skill_norm"].tolist()
    print(f"\n  person_id={pid}  ({len(skills)} cleaned skills):")
    print("    " + ", ".join(skills[:30]))
    if len(skills) > 30:
        print(f"    ... and {len(skills)-30} more")

section("DONE")
sys.stdout = sys.__stdout__
log_file.close()