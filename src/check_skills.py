"""Quick check: do all our JD-required skills actually exist with 50+ frequency?"""
import pandas as pd
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# load skill frequencies from the raw data
df = pd.read_csv(REPO_ROOT / "data/raw/05_person_skills.csv")
counts = df["skill"].astype(str).str.lower().str.strip().value_counts()

# load JDs
with open(REPO_ROOT / "data/processed/jds.json") as f:
    jds = json.load(f)

# collect every skill mentioned across all JDs
all_required = set()
for jd in jds:
    all_required.update(jd["required_skills"])

print(f"checking {len(all_required)} unique skills across {len(jds)} JDs\n")

problems = []
for skill in sorted(all_required):
    count = counts.get(skill, 0)
    flag = "OK" if count >= 50 else "LOW" if count >= 5 else "MISSING"
    if count < 50:
        problems.append((skill, count, flag))
    print(f"  {skill:25s}  count={count:>6}  {flag}")

if problems:
    print(f"\n{len(problems)} skill(s) below the 50+ threshold:")
    for s, c, f in problems:
        print(f"  - {s}: {c} ({f})")
else:
    print("\nAll required skills have >= 50 occurrences. Good to proceed.")