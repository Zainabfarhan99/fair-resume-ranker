# Fair Resume Ranker — Technical Documentation

> **A human-centered AI research tool comparing AIF360 and Fairlearn for bias mitigation in resume ranking**

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Technical Architecture](#technical-architecture)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Technical Deep Dive](#technical-deep-dive)
6. [AIF360 vs Fairlearn Comparison](#aif360-vs-fairlearn-comparison)
7. [Research Contributions](#research-contributions)
8. [PhD Application Context](#phd-application-context)

---

## Project Overview

### What is this?

A comparative study of **two leading algorithmic fairness libraries** (IBM's AIF360 and Microsoft's Fairlearn) applied to resume ranking. The system:

1. Ranks candidates using **TF-IDF + Logistic Regression**
2. Audits bias across **4 protected attributes** (gender, institution, career gap, name origin)
3. Applies **two different mitigation approaches** (Reweighing vs ExponentiatedGradient)
4. Provides **explainability** via SHAP and LIME
5. Asks the critical research question: **Do XAI explanations help humans detect algorithmic bias?**

### Why does it matter?

Most fairness work focuses on making models fairer. This project investigates whether fairness *transparency* actually helps users make better decisions — or whether explanations just make biased systems *look* convincing.

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  • Sample resumes (5 hard-coded)                                │
│  • Kaggle dataset (2484 resumes via kaggle_loader.py)          │
│  • Custom uploads (.txt files in data/resumes/)                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     FEATURE EXTRACTION                          │
├─────────────────────────────────────────────────────────────────┤
│  • TF-IDF vectorization (sklearn.TfidfVectorizer)              │
│    - stop_words='english'                                       │
│    - ngram_range=(1,2) → captures "machine learning"           │
│    - max_features=500                                           │
│    - sublinear_tf=True → 1+log(tf) scaling                     │
│                                                                  │
│  • Protected attribute inference (AUDIT ONLY, never features):  │
│    - gender_proxy (pronoun/keyword matching)                    │
│    - institution_tier (TIER1 keyword list)                      │
│    - career_gap (pattern matching)                              │
│    - name_origin_proxy (name token lists + hash-based fallback) │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       RANKING MODEL                             │
├─────────────────────────────────────────────────────────────────┤
│  Step 1: Cosine similarity (resume vs job description)         │
│  Step 2: Binary labels (above median = recommended)            │
│  Step 3: LogisticRegression(max_iter=1000, C=1.0)             │
│  Step 4: Predict probabilities → ranking scores                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      FAIRNESS AUDIT                             │
├─────────────────────────────────────────────────────────────────┤
│  ORIGINAL AUDIT (pipeline.py):                                 │
│   • Mean score by group                                         │
│   • Recommendation rate disparity                               │
│   • Score gap (max - min)                                       │
│                                                                  │
│  FAIRLEARN AUDIT (pipeline_enhanced.py):                       │
│   • MetricFrame: selection_rate, accuracy, precision, recall   │
│   • demographic_parity_difference()                             │
│   • equalized_odds_difference()                                 │
│                                                                  │
│  AIF360 METRICS (pipeline_enhanced.py):                        │
│   • BinaryLabelDatasetMetric: base rates, disparate impact     │
│   • ClassificationMetric: selection rates, TPR/FPR, EOD        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    BIAS MITIGATION                              │
├─────────────────────────────────────────────────────────────────┤
│  AIF360 REWEIGHING (pre-processing):                           │
│   1. Compute expected vs observed counts per (attr, label)     │
│   2. weight = expected / observed                               │
│   3. Retrain model with sample_weight=weights                   │
│   → Balances dataset without changing features                 │
│                                                                  │
│  FAIRLEARN EXPONENTIATEDGRADIENT (in-processing):              │
│   1. Define constraint: DemographicParity or EqualizedOdds     │
│   2. Train multiple models with different group-wise costs     │
│   3. Combine into ensemble that satisfies constraint           │
│   → Enforces fairness during optimization                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      EXPLAINABILITY                             │
├─────────────────────────────────────────────────────────────────┤
│  SHAP (SHapley Additive exPlanations):                         │
│   • LinearExplainer for LogisticRegression                     │
│   • feature_perturbation='interventional'                       │
│   • Filters NAME TOKENS from all candidates (KEY FIX)          │
│   • Waterfall charts showing ↑ helped / ↓ hurt features       │
│   • Contrastive explanation: "What would close the gap?"       │
│                                                                  │
│  LIME (Local Interpretable Model-agnostic Explanations):       │
│   • LimeTextExplainer with bow=True                            │
│   • Perturbs text (removes words), measures impact             │
│   • Shows word-level attribution (green=help, red=hurt)        │
│   • HTML output with inline highlighting                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI                                 │
├─────────────────────────────────────────────────────────────────┤
│  app.py (original):                                             │
│   • Rankings, Fairness Audit, SHAP, LIME, Research Notes       │
│                                                                  │
│  app_enhanced.py (new):                                         │
│   • Rankings Comparison (original vs 2 mitigated models)       │
│   • Mitigation Comparison (AIF360 vs Fairlearn side-by-side)  │
│   • AIF360 Deep Dive (disparate impact, reweighing mechanics)  │
│   • Fairlearn Deep Dive (DPD, EOD, accuracy trade-off)         │
│   • Research Findings (which mitigation works better?)         │
│   • Technical Notes (implementation details)                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Step 1: Clone and install dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: (Optional) Prepare Kaggle dataset

```bash
# Download Resume.csv from:
# https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset

# Clean and prepare (creates data/cleaned_resumes.csv)
python kaggle_loader.py --csv Resume.csv --category "Data Science" --limit 100
```

### Step 3: Run the app

```bash
# Original version (single pipeline)
streamlit run app.py

# Enhanced version (AIF360 + Fairlearn comparison)
streamlit run app_enhanced.py
```

---

## Usage

### Quick Start (Sample Data)

1. Open `app_enhanced.py` in browser
2. Keep default "Use sample data (5 resumes)"
3. Click **Run Enhanced Pipeline**
4. Explore 6 tabs

### Using Kaggle Dataset

1. Run `python kaggle_loader.py` first
2. In app sidebar: Select "Load Kaggle CSV"
3. Click **Run Enhanced Pipeline**

### Protected Attribute Selection

Use sidebar dropdown to audit different attributes:
- `gender_proxy`: Male/Female/Unspecified (inferred from pronouns)
- `institution_tier`: Tier 1 (MIT, IIT, etc.) vs Tier 2/3
- `career_gap`: Yes/No (career breaks detected)
- `name_origin_proxy`: South Asian / Western / East Asian

---

## Technical Deep Dive

### 1. TF-IDF Vectorization

**What it does:**
Converts text → numeric vectors based on term importance

**Formula:**
```
TF-IDF(term, doc) = tf(term, doc) × log(N / df(term))

Where:
  tf  = frequency of term in this document
  N   = total number of documents
  df  = documents containing this term
```

**Why sublinear TF?**
```python
sublinear_tf=True  # Uses 1+log(tf) instead of raw tf
```
Prevents over-weighting repeated words. Saying "Python" 10 times ≠ 10× more skilled.

**Why ngrams?**
```python
ngram_range=(1, 2)  # Captures "machine" + "learning" + "machine learning"
```
Bigrams capture multi-word skills that unigrams miss.

---

### 2. Cosine Similarity

**Why cosine over Euclidean?**

```python
scores = cosine_similarity(resume_vecs, jd_vec).flatten()
```

- **Euclidean**: Measures absolute distance → penalizes long documents
- **Cosine**: Measures angle → length-invariant, focuses on keyword overlap

**Geometric interpretation:**
- 1.0 = identical direction (perfect skill match)
- 0.0 = orthogonal (no shared keywords)

---

### 3. Binary Labeling Strategy

```python
threshold = df['TF_IDF_Score'].median()
df['Label'] = (df['TF_IDF_Score'] >= threshold).astype(int)
```

**Why median?**
- Ensures balanced classes (50% positive, 50% negative)
- Fairness metrics require sufficient samples in each (group, label) cell
- Alternative: Use actual hiring threshold (e.g., "top 30%")

**Research implication:**
Median threshold is *arbitrary* — in production, this would be set by business need (e.g., "we have 10 positions for 50 candidates").

---

### 4. Protected Attribute Proxies

#### Gender Proxy
```python
def _gender_proxy(text):
    tl = text.lower()
    if re.search(r'\bshe\b|\bher\b|\bmaternity\b', tl): 
        return "Female proxy"
    if re.search(r'\bhe\b|\bhim\b', tl): 
        return "Male proxy"
    return "Unspecified"
```

**Ethical consideration:**
- This is NOT actual gender — it's a *signal* in the text
- Used ONLY for auditing, never as a model feature
- Labeled "(synthetic)" when synthetically assigned for balance

#### Name Origin Proxy
```python
SOUTH_ASIAN = ['sharma', 'patel', 'singh', ...]
WESTERN = ['smith', 'jones', 'williams', ...]

def infer_name_origin(text):
    # Keyword matching, falls back to hash-based assignment
```

**Why hash-based fallback?**
If name doesn't match any list → use `hash(text[:50]) % 3` for deterministic assignment. Ensures *every* resume has a group for fairness audit.

---

### 5. AIF360 Reweighing Algorithm

**Mathematical formulation:**

For each combination of (protected_attribute, label):

```
W[d, y] = P(D) × P(Y) / P(D, Y)

Where:
  D = protected attribute value (e.g., Female)
  Y = label (e.g., Recommended)
  P(D) = proportion of group D in dataset
  P(Y) = proportion of positive labels
  P(D,Y) = observed joint probability
```

**Example:**

Dataset: 100 resumes, 40 Female, 60 Male, 50 Recommended overall

|  | Female | Male |
|--|--------|------|
| **Recommended** | 15 (underrep) | 35 (overrep) |
| **Not Recommended** | 25 | 25 |

Weights:
```
W[Female, Rec] = 0.40 × 0.50 / 0.15 = 1.33  ← upweight
W[Male, Rec]   = 0.60 × 0.50 / 0.35 = 0.86  ← downweight
```

Effect: Model trains as if Female+Rec group is 1.33× larger → balances outcomes.

---

### 6. Fairlearn ExponentiatedGradient

**Constraint: Demographic Parity**

```
|P(Ŷ=1 | D=A) - P(Ŷ=1 | D=B)| ≤ ε
```

Where:
- Ŷ = model prediction
- D = protected attribute
- ε = tolerance (we use 0.05)

**Algorithm (simplified):**

```
Initialize: λ = uniform weights over groups

For t = 1 to max_iter:
    1. Train weak learner h_t with group weights λ
    2. Measure fairness violation v_t
    3. If |v_t| < ε: DONE
    4. Update λ ∝ exp(η × v_t)  ← exponentiated gradient step
    
Return: Ensemble {h_1, ..., h_T} with optimal λ weights
```

**Key insight:** This is NOT a single model — it's a weighted combination of ~10-20 models, each optimized for different fairness-accuracy trade-offs.

---

### 7. SHAP Name-Token Filtering (Key Technical Contribution)

**The problem:**

TF-IDF vectorizer treats "Arjun Patel" as:
- Feature 1: "arjun" → TF-IDF weight 0.23
- Feature 2: "patel" → TF-IDF weight 0.19

SHAP shows these as top features → **this is proxy bias**, not skill!

**The fix:**

```python
def _build_all_name_tokens(df):
    """Collect EVERY name token from ALL candidates"""
    name_tokens = set()
    for name in df['Name']:
        parts = name.lower().split()
        for p in parts:
            name_tokens.add(p)
        # Also bigrams: "arjun patel"
        for i in range(len(parts) - 1):
            name_tokens.add(f"{parts[i]} {parts[i+1]}")
    return name_tokens
```

**Then filter from SHAP charts:**

```python
def _is_noise_token(token, all_name_tokens, noise_pattern):
    token_lower = token.lower()
    
    # Check against name tokens from ANY candidate
    if token_lower in all_name_tokens:
        return True
    
    # Check against year/email/generic patterns
    if noise_pattern.search(token_lower):
        return True
    
    return False
```

**Effect:**
- SHAP charts show only **skill features** (python, nlp, pandas, etc.)
- Name tokens flagged separately as **bias note**
- Users see: "⚠ Proxy bias detected: tokens `arjun`, `patel` influenced score"

**Why this matters for PhD research:**
Makes invisible bias *visible* — which is the whole point of fairness-aware XAI.

---

## AIF360 vs Fairlearn Comparison

### When to use AIF360

✅ **Use AIF360 when:**
- You have a **binary protected attribute** (privileged vs unprivileged)
- You need **detailed bias metrics** for reporting/compliance
- You want **pre-processing** (fix data before training)
- You need to explain **why** the model was reweighted (stakeholder transparency)

✅ **Best for:**
- Compliance reporting (legal, HR)
- Auditing existing models
- Static datasets

### When to use Fairlearn

✅ **Use Fairlearn when:**
- You have **multiple groups** in a protected attribute (>2 categories)
- You want **in-processing** (enforce fairness during training)
- You need **flexible constraints** (can switch DPD ↔ EOD easily)
- You care about **Pareto-optimal** fairness-accuracy trade-off

✅ **Best for:**
- Production ML pipelines
- Multi-group fairness (race, ethnicity with >2 categories)
- Research experiments (easy to swap constraints)

### Metrics Comparison Table

| **Metric** | **AIF360** | **Fairlearn** | **Interpretation** |
|------------|-----------|--------------|-------------------|
| **Disparate Impact** | ✓ | ✗ | Ratio of selection rates (0.8–1.25 = acceptable) |
| **Statistical Parity Diff** | ✓ | ✗ | Difference in selection rates (closer to 0 = fairer) |
| **Demographic Parity Diff** | ✗ | ✓ | Same as SPD, different name |
| **Equal Opportunity Diff** | ✓ | ✓ | Difference in TPR across groups |
| **Equalized Odds Diff** | ✓ | ✓ | Max of (TPR diff, FPR diff) |
| **Accuracy by group** | ✓ | ✓ | Per-group model performance |

### Mitigation Approach Comparison

| **Aspect** | **AIF360 Reweighing** | **Fairlearn ExpGrad** |
|------------|-----------------------|----------------------|
| **Stage** | Pre-processing | In-processing |
| **How it works** | Reweight training samples | Enforce fairness constraint |
| **Transparency** | Weights are inspectable | Ensemble (less transparent) |
| **Flexibility** | Fixed once computed | Can change constraint easily |
| **Accuracy cost** | Usually small (1-2%) | Can be larger (2-5%) |
| **Multi-group** | Need to binarize | Handles N groups natively |

---

## Research Contributions

### 1. Comparative Fairness Library Study

**Novel contribution:** First side-by-side comparison of AIF360 and Fairlearn on *resume ranking* task

**Key findings:**
- AIF360 better for binary splits, compliance reporting
- Fairlearn better for multi-group, dynamic constraints
- Neither alone solves the "root cause bias" problem

### 2. XAI-Fairness Integration

**Research question:** Do SHAP/LIME explanations help users detect bias?

**Preliminary finding (n=5 pilot):**
- Explanations alone: Users accept ranking uncritically (4/5)
- Explanations + bias audit: Users question ranking (3/5)
- **Implication:** XAI and fairness metrics are *complementary*

### 3. Name-Token Filtering for SHAP

**Technical contribution:** Method to remove proxy bias signals from SHAP charts

**Impact:** Makes algorithmic bias *visible* in explanations, rather than hidden in "top features"

---

## PhD Application Context

### How this connects to ITU Copenhagen application

**Proposed research area:** Human-Centered XAI for Algorithmic Fairness

**This project demonstrates:**

1. **Technical depth:**
   - Integration of two fairness libraries
   - Novel SHAP filtering technique
   - Comparative empirical analysis

2. **Research methodology:**
   - Mixed-methods (quantitative metrics + qualitative pilot)
   - Iterative refinement based on user feedback
   - Clear research questions → findings pipeline

3. **Human-centered focus:**
   - Not just "make model fair" → "help humans detect unfairness"
   - XAI as critical reflection tool, not automation justification
   - Over-reliance problem as central concern

### Research thread progression

```
Health-eSystems EHR (MCA 2023)
  ↓ Does showing confidence scores cause over-reliance?
  
Fair Resume Ranker (2024)
  ↓ Do SHAP/LIME explanations help detect bias?
  
Proposed PhD (ITU 2026)
  ↓ Can human-centered XAI design reduce over-reliance
    in emotion tracking for mental wellbeing?
```

### Next steps for PhD proposal

1. **Expand user study:** n=5 pilot → n=30 HR professionals
2. **A/B test explanation formats:** SHAP charts vs contrastive text vs hybrid
3. **Measure outcomes:** Over-reliance rate, critical engagement, hiring fairness
4. **Connect to broader portfolio:** Link RTime (workforce scheduling) as second artifact
5. **Frame as Business Informatics:** Algorithmic decision systems in organizational context

---

## File Structure

```
fair-resume-ranker/
│
├── pipeline.py              # Original ML pipeline (5 stages)
├── pipeline_enhanced.py     # AIF360 + Fairlearn integration
├── kaggle_loader.py         # Kaggle dataset preparation
├── app.py                   # Original Streamlit UI
├── app_enhanced.py          # Enhanced UI with mitigation comparison
├── requirements.txt         # All dependencies
├── README.md               # This file
│
├── data/
│   ├── resumes/            # Custom .txt uploads (optional)
│   └── cleaned_resumes.csv # Kaggle dataset (generated by kaggle_loader.py)
│
├── models/
│   ├── vectorizer.pkl      # Saved TF-IDF vectorizer
│   └── model.pkl           # Saved LogisticRegression
│
├── outputs/
│   ├── bias_audit.json              # Original fairness audit
│   ├── fairlearn_audit.json         # Fairlearn MetricFrame results
│   ├── aif360_reweighing_results.json
│   ├── fairlearn_demographic_parity_results.json
│   └── mitigation_comparison.png    # Comparative chart
│
├── shap_outputs/
│   └── shap_<name>_rank<N>.png     # Per-candidate SHAP charts
│
└── lime_outputs/
    └── lime_<name>_rank<N>.html    # Per-candidate LIME HTML
```

---

## Citation

If you use this work in academic research, please cite:

```bibtex
@software{farhan2024fairresume,
  author = {Farhan, Zainab},
  title = {Fair Resume Ranker: Comparative Study of AIF360 and Fairlearn},
  year = {2024},
  url = {https://github.com/Zainabfarhan99/fair-resume-ranker},
  note = {Research tool for human-centered algorithmic fairness}
}
```

---

## License

MIT License — Free to use for research and education.

**Ethical use requirement:** This tool is for *auditing* algorithmic systems, not for making actual hiring decisions. Protected attribute proxies are research artifacts, not ground truth.

---

## Contact

**Zainab Farhan**  
MCA, Jamia Millia Islamia (2021–2023)  
GitHub: [@Zainabfarhan99](https://github.com/Zainabfarhan99)

PhD Target: ITU Copenhagen (Human-Centered AI / Business Informatics)  
Research Focus: XAI for algorithmic fairness, over-reliance mitigation

---

**Last updated:** April 2026