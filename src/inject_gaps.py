"""
inject_gaps.py
==============
Day 3-4 of Week 1: generate the candidate-variants dataset.

For each base person we generate 6 variants of their resume:
    1 control:               original resume, no gap
    5 mid-career variants:   gap inserted at the middle job-to-job transition
                             (no_reason, caregiving, health, layoff, education)

Mid-career gap mechanism:
    Shift later jobs forward by `gap_months`. Optionally insert a "Career break"
    entry covering the gap window (unless reason is no_reason, in which case the
    gap is implicit in the date shift only — no explanation appears on the resume).

NOTE on first-job gaps:
    An earlier version of this script also generated first-job-gap variants
    (gap between graduation and first job). We dropped that analysis after
    discovering the dataset's education table has only 28% parseable start_dates,
    making the (graduation -> first job) signal too sparse for reliable injection.
    First-job-gap discrimination remains an important question for future work
    on a dataset with cleaner education timeline data.

All variants share the same person_id but get a unique candidate_id.
The gap duration is sampled once per person and applied to all 5 reason variants
— so duration is held constant within a person across reasons, isolating the
effect of REASON from DURATION.

Run from repo root:
    python src/inject_gaps.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import random
import sys

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
N_BASE_PEOPLE = 10_000
GAP_THRESHOLD_MONTHS = 6                                # transitions > this count as existing gaps
GAP_DURATIONS_MONTHS = [6, 12, 18, 24, 36, 48, 60]      # uniform sampling

REASONS = ["no_reason", "caregiving", "health", "layoff", "education"]

# Templates per reason — sampled randomly per variant.
# Reviewer-defence: real candidates phrase reasons many ways. Hardcoding one phrase
# would mean we measure response to that phrase, not to the underlying reason.
REASON_TEMPLATES = {
    "no_reason": [None],   # no career-break entry; gap is implicit in date shift only
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

# -----------------------------------------------------------------------------
# Setup paths and dual-output logger
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROC_DIR = REPO_ROOT / "data" / "processed"
OUT_DIR = REPO_ROOT / "outputs"
PROC_DIR.mkdir(exist_ok=True, parents=True)
OUT_DIR.mkdir(exist_ok=True)

LOG_PATH = OUT_DIR / "02_gap_injection_log.txt"
SAMPLES_PATH = OUT_DIR / "02_sample_variants.txt"
OUTPUT_CSV = PROC_DIR / "candidate_variants.csv"

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

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -----------------------------------------------------------------------------
# Step 1: Load
# -----------------------------------------------------------------------------
section("STEP 1: load data")
people = pd.read_csv(RAW_DIR / "01_people.csv")
exp = pd.read_csv(RAW_DIR / "04_experience.csv")
print(f"  loaded {len(people):,} people, {len(exp):,} experience rows")

# -----------------------------------------------------------------------------
# Step 2: Parse dates
# -----------------------------------------------------------------------------
section("STEP 2: parse dates")

TODAY = pd.Timestamp("2026-05-01")  # fixed for reproducibility

def parse_date(s):
    if pd.isna(s): return pd.NaT
    s = str(s).strip()
    if s.lower() == "present": return TODAY
    try: return pd.to_datetime(s, format="%m/%Y")
    except Exception:
        try: return pd.to_datetime(s, format="%Y")
        except Exception: return pd.NaT

exp["start_dt"] = exp["start_date"].apply(parse_date)
exp["end_dt"] = exp["end_date"].apply(parse_date)

before = len(exp)
exp = exp.dropna(subset=["start_dt", "end_dt"]).copy()
print(f"  experience: dropped {before - len(exp):,} rows with unparseable dates")

# -----------------------------------------------------------------------------
# Step 3: Eligible people (no existing gaps + >= 2 jobs)
# -----------------------------------------------------------------------------
section("STEP 3: filter to people with no existing gaps")
exp = exp.sort_values(["person_id", "start_dt"]).reset_index(drop=True)
exp["next_start"] = exp.groupby("person_id")["start_dt"].shift(-1)
exp["gap_to_next_months"] = ((exp["next_start"] - exp["end_dt"]).dt.days / 30.44).round(1)

person_max_gap = exp.groupby("person_id")["gap_to_next_months"].max()
job_counts = exp.groupby("person_id").size()

eligible_ids = person_max_gap[
    (person_max_gap <= GAP_THRESHOLD_MONTHS)
    & (job_counts.reindex(person_max_gap.index) >= 2)
].index.tolist()

print(f"  people with >=2 jobs:                           {(job_counts >= 2).sum():,}")
print(f"  people with no existing gap (>{GAP_THRESHOLD_MONTHS}mo):            {len(eligible_ids):,}")

# -----------------------------------------------------------------------------
# Step 4: Subsample
# -----------------------------------------------------------------------------
section(f"STEP 4: subsample {N_BASE_PEOPLE:,} base people")
if len(eligible_ids) < N_BASE_PEOPLE:
    print(f"  WARNING: only {len(eligible_ids):,} eligible, using all of them")
    sampled_ids = eligible_ids
else:
    sampled_ids = random.sample(eligible_ids, N_BASE_PEOPLE)
print(f"  selected {len(sampled_ids):,} base people")

# -----------------------------------------------------------------------------
# Step 5: Helpers
# -----------------------------------------------------------------------------
section("STEP 5: define helpers")

def fmt_date(dt):
    if pd.isna(dt): return ""
    if dt >= TODAY - pd.Timedelta(days=30): return "Present"
    return dt.strftime("%m/%Y")

def build_resume_text(name, jobs, career_break_entries=None):
    """Reconstruct resume into a text string. Newest-first (standard convention)."""
    lines = []
    if isinstance(name, str) and name.strip():
        lines.append(f"Name: {name.strip()}")
    lines.append("\nWORK EXPERIENCE")
    lines.append("-" * 40)
    all_entries = list(jobs)
    if career_break_entries:
        all_entries = all_entries + list(career_break_entries)
    for entry in sorted(all_entries, key=lambda e: e["start_dt"], reverse=True):
        date_range = f"{fmt_date(entry['start_dt'])} – {fmt_date(entry['end_dt'])}"
        if entry.get("firm") is None:
            # career-break entry
            lines.append(f"{date_range}")
            lines.append(f"  {entry['title']}")
        else:
            loc = (
                f", {entry['location']}"
                if isinstance(entry.get("location"), str) and entry["location"].strip()
                else ""
            )
            lines.append(f"{date_range}")
            lines.append(f"  {entry['title']} — {entry['firm']}{loc}")
        lines.append("")
    return "\n".join(lines)

def inject_mid_career_gap(jobs, gap_months, reason_type):
    """
    Shift jobs from the middle transition onwards forward by gap_months.
    Insert a career-break entry covering the gap window (unless no_reason).
    Returns: (new_jobs_list, career_break_entry_or_None).
    """
    if len(jobs) < 2:
        return jobs, None
    mid = max(1, len(jobs) // 2)
    delta = pd.DateOffset(months=int(gap_months))

    new_jobs = [dict(j) for j in jobs[:mid]]
    for j in jobs[mid:]:
        nj = dict(j)
        nj["start_dt"] = j["start_dt"] + delta
        if j["end_dt"] < TODAY - pd.Timedelta(days=30):
            nj["end_dt"] = j["end_dt"] + delta
        else:
            nj["end_dt"] = TODAY
        new_jobs.append(nj)

    cb_entry = None
    if reason_type != "no_reason":
        text = random.choice(REASON_TEMPLATES[reason_type])
        gap_start = jobs[mid - 1]["end_dt"]
        gap_end = gap_start + delta
        cb_entry = {
            "title": text,
            "firm": None,
            "location": None,
            "start_dt": gap_start,
            "end_dt": gap_end,
        }
    return new_jobs, cb_entry

# -----------------------------------------------------------------------------
# Step 6: Generate variants
# -----------------------------------------------------------------------------
section("STEP 6: generate variants")

exp_sample = exp[exp["person_id"].isin(sampled_ids)].copy()
person_jobs = {}
for pid, group in exp_sample.groupby("person_id"):
    jobs = group[["title", "firm", "location", "start_dt", "end_dt"]].to_dict("records")
    jobs = sorted(jobs, key=lambda j: j["start_dt"])
    person_jobs[pid] = jobs

names_lookup = dict(zip(people["person_id"], people["name"]))

records = []
candidate_id = 0

for i, pid in enumerate(sampled_ids):
    if (i + 1) % 1000 == 0:
        print(f"  processed {i+1:,} / {len(sampled_ids):,} people...")

    base_jobs = person_jobs[pid]
    name = names_lookup.get(pid, "")
    chosen_duration = random.choice(GAP_DURATIONS_MONTHS)

    # control variant
    records.append({
        "candidate_id": candidate_id,
        "person_id": pid,
        "variant_type": "control",
        "gap_reason": "none",
        "gap_duration_months": 0,
        "resume_text": build_resume_text(name, base_jobs),
    })
    candidate_id += 1

    # 5 reason variants
    for reason in REASONS:
        new_jobs, cb_entry = inject_mid_career_gap(base_jobs, chosen_duration, reason)
        cb_list = [cb_entry] if cb_entry else None
        records.append({
            "candidate_id": candidate_id,
            "person_id": pid,
            "variant_type": reason,
            "gap_reason": reason,
            "gap_duration_months": chosen_duration,
            "resume_text": build_resume_text(name, new_jobs, cb_list),
        })
        candidate_id += 1

variants_df = pd.DataFrame(records)

print(f"\n  generated {len(variants_df):,} candidate-variants from {len(sampled_ids):,} people")
print(f"\n  variant_type distribution:")
print(variants_df["variant_type"].value_counts().to_string())

# -----------------------------------------------------------------------------
# Step 7: Write outputs
# -----------------------------------------------------------------------------
section("STEP 7: write outputs")
variants_df.to_csv(OUTPUT_CSV, index=False)
size_mb = OUTPUT_CSV.stat().st_size / (1024 * 1024)
print(f"  wrote {OUTPUT_CSV}  ({size_mb:.1f} MB)")

# -----------------------------------------------------------------------------
# Step 8: Hand-validation samples
# -----------------------------------------------------------------------------
section("STEP 8: write hand-validation samples")
sample_pids = random.sample(sampled_ids, 5)

with open(SAMPLES_PATH, "w") as f:
    for pid in sample_pids:
        f.write("\n" + "#" * 80 + "\n")
        f.write(f"# person_id = {pid}\n")
        f.write("#" * 80 + "\n")
        person_variants = variants_df[variants_df["person_id"] == pid]
        for _, row in person_variants.iterrows():
            f.write(
                f"\n--- variant: {row['variant_type']}  "
                f"(reason={row['gap_reason']}, "
                f"duration={row['gap_duration_months']}mo, "
                f"candidate_id={row['candidate_id']}) ---\n"
            )
            f.write(row["resume_text"])
            f.write("\n")

print(f"  wrote {SAMPLES_PATH}")

section("DONE")
print(f"  data:    {OUTPUT_CSV}")
print(f"  log:     {LOG_PATH}")
print(f"  samples: {SAMPLES_PATH}")

sys.stdout = sys.__stdout__
log_file.close()