# The Disclosure Paradox in Algorithmic Résumé Screening

> **Reason-Stratified Bias in Career-Gap Penalties**
> Reproducibility repository for the AIMCON 2026 submission.

---

## Overview

This repository contains the audit pipeline, trained models, and analysis scripts behind a controlled empirical study of **career-gap disclosure bias in embedding-based résumé rankers**.

**The headline finding — the disclosure paradox:** Across every architecture tested (a 2×2 grid of two transformer embeddings × two classifier families), candidates who explain their career gap face larger algorithmic penalties than candidates who leave the gap unexplained. Under the baseline ranker, education-disclosure violates the United States four-fifths rule (Disparate Impact = 0.76). The general silence-beats-explanation pattern is invariant across architectures, while the specific worst-penalty reason shifts with the underlying model.

## Citation

```bibtex
@inproceedings{farhan2026disclosure,
  author    = {Farhan, Zainab},
  title     = {The Disclosure Paradox in Algorithmic Resume Screening:
               Reason-Stratified Bias in Career-Gap Penalties},
  booktitle = {Proceedings of AIMCON 2026 (under review)},
  year      = {2026}
}
```

---

## Quick start

```bash
# 1. Clone and checkout the audit branch
git clone https://github.com/Zainabfarhan99/fair-resume-ranker.git
cd fair-resume-ranker
git checkout career-gap-fairness

# 2. Install dependencies (Python 3.12 recommended)
pip install -r requirements.txt

# 3. End-to-end pipeline
python src/explore_data.py             # sanity-check the corpus
python src/inject_gaps.py              # generate 60,000 variants
python src/build_jds.py                # construct 26 synthetic Job Descriptions
python src/compute_labels.py           # skill-overlap match labels
python src/split_train_test.py         # 80/20 candidate-level split
python src/encode_texts.py             # MiniLM embedding cache
python src/encode_mpnet.py             # mpnet embedding cache (~30 min, one-time)
python src/train_baseline_ranker.py    # baseline MiniLM + LR
python src/train_all_grid_models.py    # 4 grid-cell rankers
python src/train_with_mitigation.py    # uniform + reason-aware mitigation
python src/audit_baseline.py           # baseline audit (§4.1)
python src/audit_grid_models.py        # 2×2 grid audit (§4.5)
python src/audit_mitigated.py          # mitigation comparison (§4.3)
python src/per_duration_audit.py       # gap duration robustness (§4.4)
python src/threshold_sensitivity.py    # decision threshold robustness (§4.4)
python src/embedding_cosine_analysis.py  # archetype proximity analysis (§5.2)
python src/make_baseline_figures.py    # Figures 1, 2
python src/make_comparison_figure.py   # Figure 3 (mitigation)
```

All outputs are written to `data/processed/audit/` and `outputs/figures/`.

---

## Study design

### Variant generation

From the public Suriyaganesh 54K Résumé Dataset (54,933 résumés scraped from LiveCareer.com), 10,000 base candidates are sampled. For each, **six variants** are generated:

| Variant | Description |
|---------|-------------|
| `control` | Original résumé, no gap inserted |
| `no_reason` | Gap inserted at the middle job-to-job transition, no `Career break` entry |
| `caregiving` | Same gap with `Career break — Caring for young children` (template) |
| `health` | Same gap with `Career break — Health-related leave` (template) |
| `layoff` | Same gap with `Career break — Industry layoff` (template) |
| `education` | Same gap with `Career break — Pursuing MBA` (template) |

Gap duration is sampled once per candidate from `{6, 12, 18, 24, 36, 48, 60}` months and held constant across the five reasoned variants. Total dataset: **60,000 controlled variants**.

### Architecture grid (2×2)

The audit replicates across two embedding models and two classifier families:

|                | Logistic Regression | XGBoost |
|----------------|---------------------|---------|
| `all-MiniLM-L6-v2` (384-dim) | Baseline ranker | MiniLM + XGBoost |
| `all-mpnet-base-v2` (768-dim) | mpnet + LR | mpnet + XGBoost |

Each ranker is trained on **control variants only** — the disclosure paradox is therefore a property of the embedding-and-classifier composition at inference time, not a learned association.

### Fairness metrics

For each test candidate, all six variants are scored against 26 Job Descriptions at decision threshold 0.5, treating the control variant as the privileged group. Three group-fairness metrics are reported:

- **Demographic Parity Difference (DPD)** — selection-rate gap
- **Equalised Odds Difference (EOD)** — true-positive-rate gap on qualified candidates (most policy-relevant)
- **Disparate Impact (DI)** — selection-rate ratio; DI < 0.80 violates the United States four-fifths rule

95% confidence intervals from 1,000 paired bootstrap iterations.

### Mitigation

Two sample-weighting interventions are compared against the baseline:
- **Uniform weighting:** all gap variants `sample_weight = 2.0`, control 1.0
- **Reason-aware weighting:** weights vary by gap reason on a normative spectrum (involuntary reasons upweighted 2.0×, partly voluntary 1.5×, control 1.0)

---

## Results summary

### Baseline (MiniLM + Logistic Regression)

| Variant | DPD | EOD | DI |
|---------|------|------|-----|
| `no_reason` | −0.002 | −0.005 | 1.02 |
| `caregiving` | **+0.016** | **+0.058** | 0.87 |
| `health` | −0.002 | −0.004 | 1.01 |
| `layoff` | **+0.014** | **+0.053** | 0.88 |
| `education` | **+0.029** | **+0.103** | **0.76** ⚠️ |

**Education-disclosure violates the four-fifths rule.** Bold entries have 95% CIs that exclude zero.

### Architecture grid (EOD by reason × cell)

| Cell | no_reason | caregiving | health | layoff | education |
|------|-----------|------------|--------|--------|-----------|
| MiniLM + LR | −0.005 | +0.058 | −0.004 | +0.053 | **+0.103** |
| mpnet + LR | +0.005 | **+0.107** | +0.085 | +0.094 | +0.098 |
| MiniLM + XGBoost | +0.036 | +0.151 | +0.103 | **+0.208** | +0.167 |
| mpnet + XGBoost | +0.038 | +0.164 | +0.161 | **+0.250** | +0.126 |

**Bold = worst-penalty reason per cell.** In every cell, the no_reason variant has substantially smaller EOD magnitude than any disclosed-reason variant (ratio: 5.7× to 20×+). The worst-penalty reason itself is architecture-dependent.

### Mitigation

Both sample-weighting methods reduce reason-stratified bias by 80% or more at negligible accuracy cost. Education DI improves from 0.76 → 0.94–0.95 (resolves the four-fifths-rule violation); caregiving DI from 0.87 → 0.97–0.98. Test accuracy on control: 0.906 (mitigated) vs 0.907 (baseline).

---

## Repository structure

```
fair-resume-ranker/  (branch: career-gap-fairness)
│
├── README.md                                # This file
├── requirements.txt                         # Minimal dependencies
│
├── src/
│   │── inject_gaps.py                       # §3.2 Variant generation (6 per candidate)
│   │── build_jds.py                         # §3.3 Construct 26 synthetic Job Descriptions
│   │── compute_labels.py                    # §3.3 Skill-overlap match labels
│   │── split_train_test.py                  # §3.4 Candidate-level 80/20 split (seed 42)
│   │── encode_texts.py                      # MiniLM (all-MiniLM-L6-v2) embedding cache
│   │── encode_mpnet.py                      # mpnet (all-mpnet-base-v2) embedding cache
│   │── train_baseline_ranker.py             # §3.4 Baseline ranker (MiniLM + LR)
│   │── train_all_grid_models.py             # §4.5 Train all four grid cells
│   │── train_mpnet_xgb_fast.py              # Optimised mpnet + XGBoost trainer
│   │── train_with_mitigation.py             # §4.3 Uniform + reason-aware sample weighting
│   │── audit_baseline.py                    # §4.1 Baseline ranker audit
│   │── audit_grid_models.py                 # §4.5 2×2 architecture grid audit
│   │── audit_mitigated.py                   # §4.3 Mitigation comparison
│   │── per_duration_audit.py                # §4.4 Gap-duration stratification
│   │── threshold_sensitivity.py             # §4.4 Decision-threshold robustness
│   │── job_category_stratification.py       # Per-JD heterogeneity analysis
│   │── embedding_cosine_analysis.py         # §5.2 Reason × archetype proximity
│   │── fairness_metrics.py                  # DPD / EOD / DI with bootstrap CIs
│   │── make_baseline_figures.py             # Fig. 1 (EOD), Fig. 2 (flip rates)
│   │── make_comparison_figure.py            # Fig. 3 (mitigation comparison)
│   │── explore_data.py                      # Corpus EDA utility
│   │── explore_skills.py                    # Skills-vocabulary EDA
│   │── check_skills.py                      # Skills-extraction sanity check
│   └── diagnose_zero_matches.py             # Sanity check for empty (cand, JD) matches
│
├── data/
│   ├── raw/                                 # Suriyaganesh corpus (not committed)
│   └── processed/
│       ├── train.csv                        # Candidate-level training split
│       └── audit/                           # Audit-metric CSVs
│
└── outputs/                                 # Figures and log files
```

---

## Reproducibility

- **Random seed:** 42 throughout (sampling, train/test split, classifier initialisation, bootstrap)
- **Train/test split:** 80/20 candidate-level (8,001 / 1,999), stratified by gap duration; all six variants of a candidate stay in the same partition
- **Python:** 3.12 (tested)
- **Hardware:** Apple Silicon MacBook Air (16 GB RAM); MiniLM embedding ~10 min, mpnet ~30 min; full pipeline end-to-end ~2 hours

The mpnet embedding cache (~175 MB) is not committed; it is reconstructed deterministically by `encode_mpnet.py`. The raw corpus is excluded because of its size; it is freely available from the dataset link below.

---

## Data source

**Suriyaganesh 54K Résumé Dataset** (Kaggle, 2023):
https://www.kaggle.com/datasets/suriyaganesh/resume-dataset-structured

54,933 anonymised résumés scraped from LiveCareer.com. After filtering for candidates with ≥2 job entries and no existing employment gap exceeding six months (20,688 eligible), 10,000 base candidates are sampled with `random_state=42`.

The 26 Job Descriptions are researcher-constructed (see `build_jds.py`); the reason-template library is defined in `inject_gaps.py`.

---

## License

MIT. Free to use for research and education.

**Ethical use requirement:** This pipeline is for *auditing* algorithmic résumé-ranking systems, not for making actual hiring decisions. The reason templates and Job Descriptions are research artifacts, not deployable hiring criteria.

---

## Contact

**Zainab Farhan**
Software Engineer, Stepping Cloud Consulting Pvt. Ltd., New Delhi
zainabfarhan304@gmail.com
ORCID: [0009-0009-8915-2695](https://orcid.org/0009-0009-8915-2695)
GitHub: [@Zainabfarhan99](https://github.com/Zainabfarhan99)