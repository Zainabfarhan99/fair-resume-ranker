# Fair Resume Ranker

**A human-centered AI research tool that ranks resumes fairly and transparently — and then asks whether the explanations it provides actually help humans make better decisions.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![Fairlearn](https://img.shields.io/badge/Fairlearn-0.10-green)](https://fairlearn.org)
[![SHAP](https://img.shields.io/badge/SHAP-0.44-orange)](https://shap.readthedocs.io)

---

## The Research Question

Automated resume screening systems are known to encode bias. Most research addresses this by adding fairness constraints to the model. But there is a deeper, underexplored problem:

**Even when an AI ranking system is made more fair and its decisions are explained using XAI tools like SHAP and LIME — do users actually use those explanations to reflect critically on the ranking? Or do they simply accept the algorithmic verdict?**

This project was built to investigate that question. It provides:
1. A working NLP-based resume ranker (TF-IDF + cosine similarity + Logistic Regression)
2. A multi-dimensional fairness audit across four protected attribute proxies
3. SHAP and LIME explanations connected to the same underlying model
4. Contrastive "what if" explanations designed to prompt recruiter self-reflection
5. Preliminary observations from a 5-person pilot study comparing explanation formats

---

## Research Context

This project is motivated by two bodies of literature:

- **Algorithmic fairness in hiring**: Resume screening tools have been shown to reproduce biases related to gender, ethnicity, educational background, and career gaps (Dwork et al., 2012; Mehrabi et al., 2021). Most interventions focus on the model — few focus on the human who receives the output.

- **Human-centered XAI (HCXAI)**: Ehsan & Riedl (2020) argue that explanations should be designed to "foster reflection rather than acceptance." Current XAI tools are largely algorithm-centered — they describe what the model did, but do not invite the user to question whether that decision was appropriate.

This project sits at the intersection: it uses XAI not just as a transparency tool, but as a potential intervention for reducing over-reliance on algorithmic rankings.

---

## Fairness Audit: Four Dimensions of Bias

The audit examines bias across four protected attribute proxies:

| Attribute | What it detects |
|---|---|
| `gender_proxy` | Whether explicit gender signals correlate with score |
| `institution_tier` | Whether Tier-1 university background inflates scores |
| `career_gap` | Whether career breaks (disproportionately affecting women and carers) are penalised |
| `name_origin_proxy` | Whether name-based ethnic proxies correlate with outcomes |

**Important ethical note**: These attributes are used *only* to audit outcomes. They are never used as input features to the ranking model. Detecting bias is the prerequisite for correcting it.

---

## Pipeline

```
resume_parser.py        Parse resumes + extract protected attributes
      ↓
rank_resumes.py         TF-IDF vectorisation + cosine similarity + Logistic Regression
      ↓
fairness_audit.py       MetricFrame across 4 protected attributes → bias_audit_report.json
      ↓
shap_explainer.py       Per-candidate SHAP waterfall charts + contrastive explanations
      ↓
lime_explainer.py       Per-candidate LIME word-level attributions + annotated HTML
```

---

## Key Finding

Career gap penalisation was the highest-disparity attribute in this dataset — a candidate (Fatima Al-Hassan) with a 2-year career break for family caregiving was ranked #4 despite having 6 years of relevant ML and NLP experience. This disparity was **invisible in SHAP outputs alone** but became apparent when the fairness audit report was read alongside the explanations.

This illustrates the central research finding: **XAI and fairness auditing are complementary, not substitutes**. Neither alone is sufficient to surface all forms of bias to human decision-makers.

---

## Preliminary User Observations

5 participants compared two explanation formats (SHAP bar chart vs contrastive "what if" text) for the same ranking. Key observations:

- SHAP charts were described as "technical" but did not prompt participants to question rankings
- Contrastive explanations generated discussion about whether absence of a keyword was a genuine skill gap or a writing style artefact
- Career gap penalisation was not noticed from SHAP alone; visible only when combined with audit report

These observations motivate a formal user study design (see `RESEARCH_NOTES.md`).

---

## Installation

```bash
pip install -r requirements.txt
```

## Run the full pipeline

```bash
python resume_parser.py      # Parse resumes
python rank_resumes.py       # Rank with TF-IDF + model
python fairness_audit.py     # Multi-dimensional bias audit
python shap_explainer.py     # SHAP waterfall charts
python lime_explainer.py     # LIME word attributions
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| NLP & Ranking | Python, scikit-learn, TF-IDF, cosine similarity |
| ML Model | Logistic Regression (scikit-learn) |
| Fairness | Fairlearn MetricFrame |
| Explainability | SHAP (LinearExplainer), LIME (LimeTextExplainer) |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib |

---

## References

- Dwork, C., et al. (2012). Fairness through awareness. *ITCS*.
- Ehsan, U., & Riedl, M. O. (2020). Human-centered explainable AI. *HCII*.
- Mehrabi, N., et al. (2021). A survey on bias and fairness in machine learning. *ACM Computing Surveys*.
- Liao, Q. V., & Varshney, K. R. (2021). Human-centered explainable AI (XAI): From algorithms to user experiences. *arXiv:2110.10790*.

---

## See also

[Health-eSystems](https://github.com/Zainabfarhan99/Health-eSystems) — the clinical EHR project that first motivated my interest in AI over-reliance in high-stakes human domains.

---

*Created by Zainab Farhan — Human-Centered AI Research*
*Contact: zainabfarhan304@gmail.com | [LinkedIn](https://linkedin.com/in/zainab-farhan)*