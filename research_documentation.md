# Fair Resume Ranker — Research Documentation
## FEAS: Fairness-Explainability Alignment Score

**Author:** Zainab Farhan  
**Target:** PhD Applications (German Universities, 2025–2026)  
**Target supervisors:** AI Ethics, HCI, NLP, Business Informatics departments  

---

## 1. The Novel Contribution in One Paragraph

This project introduces **FEAS (Fairness-Explainability Alignment Score)**, a metric that formally quantifies the gap between what fairness audits detect (using AIF360 / Fairlearn) and what explainability tools reveal to users (via SHAP / LIME). Defined as `FEAS = |B ∩ E| / |B|`, where B is the set of bias-signal features identified by the fairness audit and E is the set of features highlighted by the explainability tool, FEAS measures how much of the detected bias is visible in user-facing explanations. A FEAS gap close to 1.0 indicates that explanations are completely hiding the bias the auditor has detected — a critical failure mode in human-in-the-loop hiring systems. This is the first formal metric bridging these two previously siloed research areas.

---

## 2. Research Questions (for 5 papers)

### RQ1 (Paper 1): FEAS Definition and Validation
> What fraction of algorithmically-detected bias in resume ranking is visible in SHAP/LIME explanations, and how does this vary across candidates and protected groups?

### RQ2 (Paper 2): AIF360 vs Fairlearn Comparative Study
> Which bias mitigation approach (AIF360 Reweighing vs Fairlearn ExponentiatedGradient) produces fairer rankings, and do they differ in their effect on FEAS?

### RQ3 (Paper 3): Intersectional Bias Invisibility
> Does single-axis fairness auditing underestimate the XAI-Fairness gap for intersectionally marginalised candidates (e.g. Female + Career gap + Non-Western name)?

### RQ4 (Paper 4): Over-Reliance and the XAI Gap (User Study)
> Do SHAP/LIME explanations cause recruiters to accept biased rankings more readily than they would without explanations?

### RQ5 (Paper 5): The Mitigation-Explainability Decoupling Problem
> Does applying bias mitigation (making the model fairer) cause the FEAS to improve — or do fairer models produce explanations that are equally opaque about their reduced bias?

---

## 3. FEAS — Technical Specification

### 3.1 Formal Definition

Let:
- `X` = TF-IDF feature matrix of all resumes
- `A` = protected attribute vector (e.g. gender_proxy)
- `B_k` = top-k features most correlated with A (bias-signal set)
  - Measured by mutual information: `MI(X_j; A)` for each feature j
  - Or combined: normalised `0.5 * MI + 0.5 * |ρ_pb|` for binary A
- `E_k^SHAP` = top-k features by |SHAP value| for candidate i
- `E_k^LIME` = top-k features by |LIME weight| for candidate i

Then:

```
FEAS_i^SHAP = |B_k ∩ E_k^SHAP| / |B_k|
FEAS_i^LIME = |B_k ∩ E_k^LIME| / |B_k|
FEAS_i = (FEAS_i^SHAP + FEAS_i^LIME) / 2

mean_FEAS = (1/n) * Σ FEAS_i
FEAS_gap  = 1 - mean_FEAS
```

### 3.2 Group-Level FEAS (second-order fairness)

```
FEAS_group(g) = mean(FEAS_i) for all i in group g
group_FEAS_diff = max_g(FEAS_group) - min_g(FEAS_group)
```

A high `group_FEAS_diff` means explanations are revealing more bias for some groups than others — a second-order fairness concern: the XAI tool itself is unequal in its transparency.

### 3.3 Post-Mitigation FEAS Delta

```
ΔFEAS = FEAS_after_mitigation - FEAS_before_mitigation
ΔFEAS_gap = FEAS_gap_after - FEAS_gap_before
```

Hypothesis: If bias mitigation reduces the model's reliance on bias-signal features, SHAP should assign lower weights to those features after mitigation, and FEAS should decrease (less bias to be revealed). If FEAS gap is unchanged after mitigation, this confirms the Mitigation-Explainability Decoupling hypothesis.

### 3.4 Intersectional FEAS

For attribute pair (A1, A2), define:
```
A_intersect = A1 × A2  (Cartesian product of group labels)
B_intersect = top-k features correlated with A_intersect
FEAS_intersectional_i = |B_intersect ∩ E_i| / |B_intersect|
```

This reveals compound disadvantage invisibility: the XAI gap for `Female + Career gap` candidates may be larger than either attribute alone.

---

## 4. Paper Outlines

---

### Paper 1: FEAS — Definition and Empirical Validation
**Title:** *"FEAS: A Metric for Quantifying the Fairness-Explainability Alignment Gap in Algorithmic Resume Ranking"*

**Target venues:**
- ACM FAccT (Fairness, Accountability, and Transparency) — top venue
- ECAI (European Conference on AI)
- IJCAI Workshop on Responsible AI

**Abstract (draft):**
Algorithmic fairness auditing and explainable AI (XAI) are widely studied but rarely connected. Fairness audits quantify disparate outcomes; XAI tools explain individual decisions. But how much of the bias detected by an audit is actually visible in user-facing explanations? We introduce FEAS (Fairness-Explainability Alignment Score), a metric that formally measures this gap. Applied to a resume ranking system using TF-IDF + Logistic Regression, audited with AIF360 and Fairlearn, and explained with SHAP and LIME, we find that [X]% of the bias detected by the fairness audit is invisible in SHAP explanations. We further show that this gap differs significantly across privileged and unprivileged groups — a second-order fairness concern. FEAS provides a principled bridge between two previously siloed research areas.

**Structure:**
1. Introduction: the silo problem (fairness vs XAI)
2. Related work: AIF360, Fairlearn, SHAP, LIME, over-reliance
3. FEAS definition (formal, as above)
4. Experimental setup: resume ranking, 5 candidates, 4 protected attrs
5. Results: per-candidate FEAS, group FEAS, FEAS gap
6. Discussion: when should FEAS be high vs low?
7. Limitations and future work
8. Conclusion

**Key figures:**
- Figure 1: FEAS framework diagram (B, E, intersection)
- Figure 2: Per-candidate FEAS bar chart (SHAP vs LIME)
- Figure 3: Bias-signal features vs SHAP top features (overlap visualisation)
- Figure 4: Group FEAS disparity chart

---

### Paper 2: AIF360 vs Fairlearn Empirical Comparison
**Title:** *"Pre-Processing vs In-Processing: An Empirical Comparison of AIF360 and Fairlearn for Resume Ranking Fairness"*

**Target venues:**
- IEEE Transactions on Neural Networks and Learning Systems
- ACM KDD Workshop on Data Science for Social Good
- ECML-PKDD

**Abstract (draft):**
We present the first systematic comparison of IBM's AIF360 (Reweighing, pre-processing) and Microsoft's Fairlearn (ExponentiatedGradient, in-processing) applied to algorithmic resume ranking. Using a TF-IDF + Logistic Regression baseline, we measure disparate impact, demographic parity difference, equalized odds difference, and accuracy trade-off before and after mitigation, across four protected attributes. We further introduce FEAS to measure whether either mitigation approach improves explanation transparency. Results show [specific finding about which wins on which metric]. Critically, neither method substantially changes the FEAS score, demonstrating that bias mitigation and explanation transparency are structurally decoupled.

**Key tables:**
- Table 1: Metric comparison (9 metrics × 3 models × 4 attributes)
- Table 2: FEAS before/after each mitigation
- Table 3: Accuracy trade-off analysis

---

### Paper 3: Intersectional Bias Invisibility
**Title:** *"Beyond Single-Axis Auditing: Intersectional Bias and Its Invisibility in Resume Ranking Explanations"*

**Target venues:**
- ACM FAccT
- CHI (Human Factors in Computing Systems)
- CSCW (Computer-Supported Cooperative Work)

**Abstract (draft):**
Algorithmic fairness audits typically examine one protected attribute at a time (gender, race, disability). But discrimination is intersectional: a candidate who is simultaneously female, South Asian, and has a career gap may face compounded disadvantages that single-axis audits completely miss. We introduce Intersectional FEAS, an extension of our FEAS metric to attribute combinations. Applied to resume ranking, we show that the XAI-Fairness gap is [X]% larger for intersectionally marginalised candidates than single-axis audits predict. This finding has direct implications for hiring law compliance and algorithmic accountability.

---

### Paper 4: Over-Reliance User Study
**Title:** *"Do SHAP Explanations Help Recruiters Detect Algorithmic Bias? A Mixed-Methods Study"*

**Target venues:**
- CHI (highest impact HCI venue)
- CSCW
- IUI (Intelligent User Interfaces)

**Study design:**
- Participants: 30 HR professionals / hiring managers
- Between-subjects: 3 conditions
  - Condition A: Ranking only (no explanation)
  - Condition B: Ranking + SHAP charts
  - Condition C: Ranking + SHAP + FEAS report
- Measures:
  - Overrule rate (did they change the ranking?)
  - Bias detection rate (did they identify the problematic ranking?)
  - Confidence rating (how certain were they in their decision?)
  - Think-aloud transcripts (qualitative)
- Hypothesis: FEAS report (Condition C) significantly increases bias detection
  over SHAP alone (Condition B), which is not significantly better than no explanation (Condition A)

**Why this is publishable:** It directly tests the practical utility of FEAS as an intervention.

---

### Paper 5: The Decoupling Problem (Survey + Framework)
**Title:** *"The Mitigation-Explainability Decoupling Problem: Toward Unified Fairness-XAI Frameworks for Hiring AI"*

**Target venues:**
- IEEE Transactions on Artificial Intelligence
- AI & Society (Springer)
- ACM Computing Surveys

**Contribution:** A theoretical framework and survey arguing that:
1. Bias mitigation reduces model unfairness but does not improve explanation transparency
2. XAI tools reveal model decisions but not whether those decisions are fair
3. FEAS is the bridge metric that can evaluate both simultaneously
4. Future hiring AI systems should be designed with FEAS as a deployment criterion

---

## 5. Target German Universities and Supervisors

### Priority universities (AI Ethics / Responsible AI):
1. **TU Berlin** — Research group on Machine Learning and AI Safety
2. **LMU Munich** — Human-Centered AI, Center for Information and Language Processing
3. **University of Hamburg** — Research group on Intelligent Systems, NLP
4. **KIT Karlsruhe** — Institute of Applied Informatics and Formal Description Methods
5. **TU Darmstadt** — UKP Lab (NLP), Centre for Cognitive Science
6. **Saarland University** — Max Planck Institute for Software Systems (MPI-SWS)
7. **Humboldt University Berlin** — Computer Science, AI Ethics

### Keywords for finding supervisors:
- "algorithmic fairness" site:uni-*.de
- "XAI explainability HCI" German university
- "responsible AI hiring" site:tu-*.de
- "bias mitigation NLP" German research group

### How FEAS fits each department:
- **Computer Science / NLP:** Novel metric, TF-IDF, fairness algorithms
- **HCI:** User study design, over-reliance, explanation interfaces
- **Business Informatics:** HR automation, algorithmic decision systems
- **AI Ethics:** Fairness, transparency, accountability, discrimination law

---

## 6. Research Proposal Core Argument

**Gap in literature:**
> "Existing work treats algorithmic fairness and explainability as independent concerns. Fairness auditing tools (AIF360, Fairlearn) report whether a model produces disparate outcomes. Explainability tools (SHAP, LIME) show which features drive individual decisions. But the critical question — whether the features driving bias are the same features highlighted to users in explanations — has never been formally studied."

**Your contribution:**
> "We introduce FEAS (Fairness-Explainability Alignment Score), which formally measures this gap. We show that [X]% of bias is invisible in SHAP explanations, that this gap persists even after bias mitigation, and that intersectional candidates experience disproportionately large invisibility. We further validate FEAS in a user study showing that presenting FEAS alongside SHAP significantly improves recruiter bias detection."

**PhD extension (2-3 years beyond this project):**
- Extend FEAS to other domains: loan approval, medical triage, university admissions
- Develop FEAS-guided XAI: explanation systems that specifically surface bias-signal features
- Study FEAS in dynamic settings: does bias shift over time as training data changes?
- FEAS for generative AI: how do LLM-generated hiring recommendations align with fairness?

---

## 7. File Structure

```
fair-resume-ranker/
├── pipeline.py              # Core ML pipeline (original)
├── pipeline_enhanced.py     # AIF360 + Fairlearn integration
├── feas_metric.py           # NOVEL: FEAS metric (all functions)
├── app.py                   # Original Streamlit UI
├── app_enhanced.py          # AIF360 + Fairlearn UI
├── app_research.py          # FEAS research edition (full integration)
├── requirements.txt
├── README.md
├── RESEARCH_DOCUMENTATION.md  (this file)
│
├── data/
│   ├── resumes/
│   └── cleaned_resumes.csv
│
├── models/
│   ├── vectorizer.pkl
│   └── model.pkl
│
└── outputs/
    ├── bias_audit.json
    ├── fairlearn_audit.json
    ├── aif360_reweighing_results.json
    ├── mitigation_comparison.png
    ├── feas_summary_table.csv        ← paper Table 1
    ├── feas_per_candidate.csv        ← paper Table 2
    ├── feas_delta_table.csv          ← paper Table 3
    ├── feas_intersectional.csv       ← paper Table 4
    ├── bias_signal_features.csv      ← paper Table 5
    └── feas_full_results.json        ← full JSON export
```

---

## 8. Quick-Start for Reviewers (Supervisors)

```bash
# Install
pip install -r requirements.txt

# Run the research edition (FEAS)
streamlit run app_research.py

# Or run the original
streamlit run app.py

# Or run the enhanced (AIF360 + Fairlearn only)
streamlit run app_enhanced.py
```

The research edition (`app_research.py`) includes:
- The novel FEAS metric with interactive computation
- Intersectional bias analysis across all attribute pairs
- Post-mitigation FEAS delta (the decoupling finding)
- Paper-ready table exports (5 CSV files)
- Full JSON results export

---

## 9. Ethical Considerations

All protected attribute values used in this project are **proxies inferred from text**, not ground truth. They are:
- Used only for auditing and research purposes
- Never used as model input features
- Clearly labeled as "proxy" throughout the codebase and UI
- Not stored or associated with real individuals

The resume ranking system is a **research artifact** demonstrating bias detection and mitigation methodology. It is not intended for use in actual hiring decisions.

FEAS itself is a measurement tool, not a prescription. Knowing that explanations hide bias does not automatically tell you how to fix that — that is the open research question this PhD will address.

---

*Last updated: April 2026*  
*Contact: GitHub @Zainabfarhan99*