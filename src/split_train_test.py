"""
split_train_test.py
===================
Day 5-7 of Week 1: split candidate_variants.csv into train/test sets.

Critical detail:
    The split is at the PERSON level, not the variant level. All 6 variants
    of a person stay together in the same split. Otherwise the model would
    see near-identical resumes across train and test (data leakage).

Stratified by gap_duration_months (sampled once per person), so the test set
has the same proportional mix of gap durations as the train set.

Run from repo root:
    python src/split_train_test.py

Outputs:
    data/processed/train.csv          -- 80% of people, all 6 variants each
    data/processed/test.csv           -- 20% of people, all 6 variants each
    outputs/03_split_log.txt          -- run log + sanity checks
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_FRACTION = 0.20

# -----------------------------------------------------------------------------
# Setup paths and logger
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = REPO_ROOT / "data" / "processed"
OUT_DIR = REPO_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

INPUT_CSV = PROC_DIR / "candidate_variants.csv"
TRAIN_CSV = PROC_DIR / "train.csv"
TEST_CSV = PROC_DIR / "test.csv"
LOG_PATH = OUT_DIR / "03_split_log.txt"

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

rng = np.random.default_rng(RANDOM_SEED)

# -----------------------------------------------------------------------------
# Step 1: Load candidate_variants
# -----------------------------------------------------------------------------
section("STEP 1: load candidate_variants")
df = pd.read_csv(INPUT_CSV)
print(f"  loaded {len(df):,} variants from {df['person_id'].nunique():,} people")
print(f"  variant_type counts:")
print(df["variant_type"].value_counts().to_string())

# -----------------------------------------------------------------------------
# Step 2: Build person-level table for stratification
# -----------------------------------------------------------------------------
section("STEP 2: build person-level table for stratification")

# each person has one duration (the same value appears on their 5 reasoned variants;
# control has duration=0). Take the duration from any non-control variant.
non_control = df[df["variant_type"] != "control"]
person_durations = (
    non_control.groupby("person_id")["gap_duration_months"]
    .first()
    .reset_index()
)
print(f"  {len(person_durations):,} people, each with one chosen gap duration")
print(f"\n  duration distribution across all people:")
print(person_durations["gap_duration_months"].value_counts().sort_index().to_string())

# -----------------------------------------------------------------------------
# Step 3: Stratified split by gap_duration
# -----------------------------------------------------------------------------
section(f"STEP 3: stratified split (train {1-TEST_FRACTION:.0%} / test {TEST_FRACTION:.0%})")

train_pids = []
test_pids = []

for duration, group in person_durations.groupby("gap_duration_months"):
    pids = group["person_id"].to_numpy()
    rng.shuffle(pids)
    n_test = int(round(len(pids) * TEST_FRACTION))
    test_pids.extend(pids[:n_test].tolist())
    train_pids.extend(pids[n_test:].tolist())

train_pids = set(train_pids)
test_pids = set(test_pids)

print(f"  train people: {len(train_pids):,}")
print(f"  test people:  {len(test_pids):,}")
print(f"  overlap (must be 0): {len(train_pids & test_pids):,}")
assert len(train_pids & test_pids) == 0, "data leakage: person in both splits"

# -----------------------------------------------------------------------------
# Step 4: Apply split to the variants table
# -----------------------------------------------------------------------------
section("STEP 4: apply split to variants")

train_df = df[df["person_id"].isin(train_pids)].copy().reset_index(drop=True)
test_df = df[df["person_id"].isin(test_pids)].copy().reset_index(drop=True)

print(f"  train variants: {len(train_df):,}  ({len(train_df)/len(df)*100:.1f}% of total)")
print(f"  test variants:  {len(test_df):,}  ({len(test_df)/len(df)*100:.1f}% of total)")

# -----------------------------------------------------------------------------
# Step 5: Sanity checks
# -----------------------------------------------------------------------------
section("STEP 5: sanity checks")

print("  --- variant_type distribution ---")
print("              train    test")
counts_train = train_df["variant_type"].value_counts()
counts_test = test_df["variant_type"].value_counts()
for vt in ["control", "no_reason", "caregiving", "health", "layoff", "education"]:
    print(f"  {vt:12s} {counts_train.get(vt, 0):>6,}  {counts_test.get(vt, 0):>6,}")

print("\n  --- gap_duration distribution (across non-control variants) ---")
print("    duration   train     test     train%   test%")
for d in sorted(df["gap_duration_months"].unique()):
    if d == 0: continue
    t = (train_df[train_df["gap_duration_months"] == d]).shape[0]
    e = (test_df[test_df["gap_duration_months"] == d]).shape[0]
    total_train_with_dur = (train_df["gap_duration_months"] > 0).sum()
    total_test_with_dur = (test_df["gap_duration_months"] > 0).sum()
    print(f"    {int(d):>3} mo    {t:>6,}   {e:>6,}   "
          f"{t/total_train_with_dur*100:>5.2f}%  {e/total_test_with_dur*100:>5.2f}%")

print("\n  --- test sanity ---")
# Each person should have exactly 6 rows in their split.
train_per_person = train_df.groupby("person_id").size()
test_per_person = test_df.groupby("person_id").size()
print(f"  train: every person has 6 variants? {(train_per_person == 6).all()}  "
      f"(min={train_per_person.min()}, max={train_per_person.max()})")
print(f"  test:  every person has 6 variants? {(test_per_person == 6).all()}  "
      f"(min={test_per_person.min()}, max={test_per_person.max()})")

# -----------------------------------------------------------------------------
# Step 6: Write outputs
# -----------------------------------------------------------------------------
section("STEP 6: write outputs")
train_df.to_csv(TRAIN_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)
print(f"  wrote {TRAIN_CSV}  ({TRAIN_CSV.stat().st_size / (1024*1024):.1f} MB)")
print(f"  wrote {TEST_CSV}   ({TEST_CSV.stat().st_size / (1024*1024):.1f} MB)")

section("DONE")
print(f"  train: {TRAIN_CSV}")
print(f"  test:  {TEST_CSV}")
print(f"  log:   {LOG_PATH}")

sys.stdout = sys.__stdout__
log_file.close()