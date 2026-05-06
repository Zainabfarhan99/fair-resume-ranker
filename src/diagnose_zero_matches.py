"""
diagnose_zero_matches.py
========================
Why do 51% of our candidates match zero JDs at 50% threshold?

Look at the zero-matchers and answer:
  - How many skills do they have on average?
  - What are their most common skills (skills the JDs miss)?
  - Are there obvious role clusters we forgot to write JDs for?

Run from repo root:
    python src/diagnose_zero_matches.py
"""

from pathlib import Path
import pandas as pd
import json

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROC_DIR = REPO_ROOT / "data" / "processed"

SKILL_FREQ_CUTOFF = 50

# load data
labels = pd.read_csv(PROC_DIR / "labels.csv")
ps = pd.read_csv(RAW_DIR / "05_person_skills.csv")
ps["skill"] = ps["skill"].astype(str).str.lower().str.strip()

with open(PROC_DIR / "jds.json") as f:
    jds = json.load(f)

# canonical vocab
freq = ps["skill"].value_counts()
canonical_vocab = set(freq[freq >= SKILL_FREQ_CUTOFF].index)

# all skills that appear in any JD
jd_required = set()
for jd in jds:
    jd_required.update(jd["required_skills"])

# zero-matchers
matches_per_person = labels.groupby("person_id")["label"].sum()
zero_match_pids = set(matches_per_person[matches_per_person == 0].index)
print(f"zero-matchers: {len(zero_match_pids):,} people\n")

# build clean skill sets
ps_clean = ps[
    ps["person_id"].isin(zero_match_pids) & ps["skill"].isin(canonical_vocab)
].drop_duplicates(subset=["person_id", "skill"])

# distribution: how many skills do zero-matchers have?
skills_per_zm = ps_clean.groupby("person_id").size()
print(f"--- skill counts for zero-matchers ---")
print(f"  people with 0 canonical skills:  {len(zero_match_pids) - len(skills_per_zm):,}")
print(f"  among those WITH canonical skills:")
print(f"    mean:   {skills_per_zm.mean():.1f}")
print(f"    median: {skills_per_zm.median():.0f}")
print(f"    max:    {skills_per_zm.max()}")
print(f"\n  distribution:")
for lo, hi in [(1, 5), (6, 15), (16, 30), (31, 60), (61, 999)]:
    cnt = ((skills_per_zm >= lo) & (skills_per_zm <= hi)).sum()
    label = f"{lo}-{hi}" if hi < 999 else f"{lo}+"
    print(f"    {label} skills: {cnt:>5,} people")

print()

# what skills do zero-matchers have that JDs don't ask for?
zm_skills = ps_clean["skill"].value_counts()
not_in_jds = zm_skills[~zm_skills.index.isin(jd_required)].head(40)
print(f"--- top 40 skills among zero-matchers that NO JD requires ---")
for skill, count in not_in_jds.items():
    print(f"  {count:>5}x  {skill}")