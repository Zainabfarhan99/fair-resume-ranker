# Quick Start Guide — Fair Resume Ranker Enhanced

## 5-Minute Setup

### Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install all packages
pip install -r requirements.txt
```

### Step 2: Run the Enhanced App

```bash
streamlit run app_enhanced.py
```

That's it! The app will open in your browser at `http://localhost:8501`

---

## What to Do in the App

### First Run (Sample Data)

1. **Leave default settings:**
   - Data: "Use sample data (5 resumes)"
   - Protected attribute: "gender_proxy"

2. **Click:** 🚀 Run Enhanced Pipeline

3. **Wait ~10 seconds** while it:
   - Ranks resumes
   - Runs Fairlearn audit
   - Computes AIF360 metrics
   - Applies Reweighing mitigation
   - Applies ExponentiatedGradient mitigation
   - Generates SHAP/LIME explanations

4. **Explore the 6 tabs:**

   **Tab 1: Rankings Comparison**
   - See how 3 models (original, AIF360, Fairlearn) rank the same candidates differently
   - Yellow highlights = disagreements between models

   **Tab 2: Mitigation Comparison** ⭐
   - Side-by-side charts comparing both approaches
   - Key question: Which reduced bias more?

   **Tab 3: AIF360 Deep Dive**
   - Disparate Impact explained
   - How Reweighing works (with weights)
   - Equal Opportunity analysis

   **Tab 4: Fairlearn Deep Dive**
   - Demographic Parity constraint explained
   - ExponentiatedGradient algorithm walkthrough
   - Accuracy trade-off analysis

   **Tab 5: Research Findings** ⭐
   - Which mitigation works better?
   - When to use each library
   - The XAI gap finding

   **Tab 6: Technical Notes**
   - Implementation details
   - Key design choices
   - Future enhancements

---

## Understanding the Output

### What You'll See

#### Original Model Results
```
Candidate: Sarah Johnson
Rank: #1
TF-IDF Score: 0.7234
Decision: ✓ Recommended
```

#### After Mitigation
```
Original:  ✓ Recommended
AIF360:    ✓ Recommended  
Fairlearn: ✗ Not Recommended  ← Changed decision!
```

### Bias Metrics Before/After

| Metric | Before | After (AIF360) | After (Fairlearn) |
|--------|--------|----------------|-------------------|
| Disparate Impact | 0.72 | 0.94 ✓ | — |
| Demographic Parity Diff | 0.18 | — | 0.06 ✓ |
| Accuracy | 82% | 80% ↓ | 78% ↓ |

**Key insight:** Both mitigation methods improved fairness, but at a small accuracy cost.

---

## Try Different Scenarios

### Scenario 1: Test Different Protected Attributes

**In sidebar:** Change "Protected Attribute" dropdown

- `gender_proxy` → See gender-based disparities
- `institution_tier` → See prestige bias (Tier 1 vs Tier 2/3)
- `career_gap` → See career break penalties
- `name_origin_proxy` → See name-based discrimination

**Re-run pipeline** after each change to see which attribute shows the most bias.

---

### Scenario 2: Use Kaggle Dataset (Optional)

#### Step 1: Download Dataset
```bash
# Get Resume.csv from Kaggle:
https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
```

#### Step 2: Prepare Data
```bash
python kaggle_loader.py --csv Resume.csv --category "Data Science" --limit 100
```

This creates `data/cleaned_resumes.csv` with:
- 100 Data Science resumes
- Balanced protected attributes
- Synthetic labels clearly marked

#### Step 3: Load in App
- Sidebar: Select "Load Kaggle CSV"
- Click: 🚀 Run Enhanced Pipeline
- Now you have 100 candidates instead of 5!

---

## Common Questions

### Q: Why do AIF360 and Fairlearn give different results?

**A:** They optimize for different metrics:
- **AIF360 Reweighing** → Targets Disparate Impact (ratio of selection rates)
- **Fairlearn ExpGrad** → Targets Demographic Parity (difference in selection rates)

Both are valid fairness definitions — which one matters depends on your context.

---

### Q: Why did accuracy go down after mitigation?

**A:** Fairness-accuracy trade-off is expected:
- Original model optimized for *accuracy only*
- Mitigated models optimize for *accuracy + fairness constraint*
- Small accuracy drop (2-5%) is typical and often acceptable

**Research question:** Is 2% accuracy loss acceptable to avoid discriminating against protected groups?

---

### Q: Which mitigation method should I use?

**Use AIF360 Reweighing when:**
- ✓ You have binary protected attribute (Male/Female, Tier1/Tier2)
- ✓ You need to explain weights to stakeholders
- ✓ Pre-processing fits your pipeline

**Use Fairlearn ExpGrad when:**
- ✓ You have multiple groups (3+ categories)
- ✓ You want flexibility to change fairness constraint
- ✓ In-processing fits your pipeline

**Use BOTH when:**
- ✓ You're doing research and want to compare approaches (like this project!)

---

## File Outputs

After running the pipeline, you'll find:

### JSON Reports
```
outputs/
  fairlearn_audit.json                     ← Fairlearn metrics for all attributes
  aif360_reweighing_results.json          ← AIF360 before/after comparison
  fairlearn_demographic_parity_results.json
```

### Visualizations
```
outputs/
  mitigation_comparison.png               ← Side-by-side charts

shap_outputs/
  shap_Sarah_Johnson_rank1.png           ← Per-candidate SHAP charts
  shap_Priya_Sharma_rank2.png
  ...

lime_outputs/
  lime_Sarah_Johnson_rank1.html          ← Interactive LIME explanations
  lime_Priya_Sharma_rank2.html
  ...
```

---

## Next Steps

### For PhD Application

1. **Run with Kaggle dataset** (larger n → more robust findings)
2. **Document key findings** from Tab 5 (Research Findings)
3. **Screenshot comparison charts** for proposal visuals
4. **Write up methodology** using README_TECHNICAL.md as reference

### For Further Development

1. **Add AIF360 post-processing:**
   ```python
   from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
   ```

2. **Try Fairlearn's ThresholdOptimizer:**
   ```python
   from fairlearn.postprocessing import ThresholdOptimizer
   ```

3. **Implement embeddings:**
   Replace TF-IDF with BERT/Sentence-Transformers for semantic matching

4. **User study:**
   A/B test different explanation formats on real HR professionals

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'aif360'`

**Fix:**
```bash
pip install aif360==0.5.0
```

AIF360 has some complex dependencies. If install fails, try:
```bash
pip install --upgrade pip
pip install aif360 --no-cache-dir
```

---

### Issue: SHAP charts show no features

**Cause:** All features filtered as noise (rare with 5 candidates)

**Fix:** Check `_build_all_name_tokens()` — might be over-filtering

---

### Issue: Streamlit app won't start

**Check:**
```bash
# Verify Streamlit is installed
streamlit --version

# If missing:
pip install streamlit==1.29.0
```

---

## Performance Notes

### Expected Runtime

**5 sample resumes:**
- Original pipeline: ~3 seconds
- Enhanced pipeline (with mitigation): ~8-10 seconds

**100 Kaggle resumes:**
- Original pipeline: ~10 seconds
- Enhanced pipeline: ~25-30 seconds

**Why the difference?**
- AIF360 Reweighing: Retrains model with weights
- Fairlearn ExpGrad: Trains 10-20 models internally
- SHAP/LIME: Computes explanations per candidate

---

## Key Takeaways

✅ **Technical accomplishment:**
- Integrated two leading fairness libraries
- Implemented novel SHAP filtering for proxy bias detection
- Built comparative analysis framework

✅ **Research contribution:**
- Empirical comparison of mitigation approaches
- XAI-fairness integration
- Human-centered evaluation framework

✅ **PhD relevance:**
- Demonstrates mixed-methods research skill
- Shows technical depth + human-centered focus
- Clear research questions → findings pipeline

---

**Happy experimenting! 🚀**

For questions or issues, check README_TECHNICAL.md or open a GitHub issue.