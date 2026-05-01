"""
app_enhanced.py
---------------
Enhanced Streamlit UI with AIF360 and Fairlearn integration.

New tabs:
  - Mitigation Comparison (AIF360 vs Fairlearn side-by-side)
  - AIF360 Deep Dive (disparate impact, reweighing mechanics)
  - Fairlearn Deep Dive (demographic parity, equalized odds)
  - Research Findings (which mitigation works best for resume ranking?)

Run:
  streamlit run app_enhanced.py
"""

import os, glob
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# ── Import from pipeline (NOT pipeline_enhanced, NOT app_enhanced) ───────────
from pipeline import (
    parse_resumes,
    _extract_email, _extract_skills, _extract_education,
    _institution_tier, _career_gap, _name_origin, _gender_proxy,
    SKILL_KEYWORDS, TIER1,
)

from pipeline_enhanced import (
    run_enhanced_pipeline,
    fairlearn_audit,
    aif360_metrics,
    aif360_reweigh,
    fairlearn_mitigate,
    plot_mitigation_comparison,
)

# ── SAMPLE DATA (defined here, NOT imported from app_enhanced) ───────────────
SAMPLE_RESUMES = {
    "Sarah Johnson": """Sarah Johnson
Email: sarah.johnson@email.com
Skills: Python, pandas, scikit-learn, NLP, spaCy, NLTK, REST API, Git, SQL, data analysis, machine learning
Education: M.Sc. Data Science, University of Edinburgh, 2020
Experience:
- 4 years at DataCorp as Senior Data Scientist
- Built NLP pipelines using spaCy for named entity recognition
- Developed REST APIs for ML model deployment
- Extensive data analysis with pandas and scikit-learn
- Led team projects using Git""",

    "Priya Sharma": """Priya Sharma
Email: priya.sharma@email.com
Skills: Python, pandas, SQL, data analysis, Excel, scikit-learn, Git, NLTK
Education: B.Tech Computer Science, State University, 2019
Experience:
- 3 years at Analytics Firm as Data Analyst
- Data analysis using pandas and SQL
- Built basic ML models with scikit-learn
- Some NLP work using NLTK for text cleaning
- Familiar with Git for version control""",

    "James Mitchell": """James Mitchell
Email: james.mitchell@email.com
Skills: Java, C++, Spring Boot, Docker, Kubernetes, REST API, Git, SQL
Education: B.Sc. Computer Science, Tech University, 2021
Experience:
- 2 years at SoftwareCo as Backend Developer
- Built REST APIs using Java Spring Boot
- Microservices architecture and Docker deployment
- SQL database design and optimisation
- No Python or data science experience""",

    "Fatima Al-Hassan": """Fatima Al-Hassan
Email: fatima.alhassan@email.com
Skills: Python, scikit-learn, NLP, NLTK, spaCy, pandas, data analysis, REST API, Git, SQL
Education: M.Sc. Computer Science, Birmingham University, 2017
Experience:
- 3 years at HealthTech as Data Scientist (2017-2020)
- NLP pipeline development using spaCy and NLTK
- Python data analysis with pandas and scikit-learn
- REST API integration for ML models
- Career break 2020-2022 (family caregiving)
- Returned as Data Analyst at NHS Digital (2022-present)""",

    "Arjun Patel": """Arjun Patel
Email: arjun.patel@email.com
Skills: Python, MySQL, HTML, CSS, JavaScript, basic data analysis
Education: B.Tech Information Technology, Regional Institute, 2022
Experience:
- 1 year at Small IT Firm as Junior Developer
- Python scripting for automation tasks
- MySQL database queries
- Completed online Python course on Coursera
- No NLP, pandas, or scikit-learn experience""",
}

SAMPLE_JD = """We are looking for a Python Developer with experience in data analysis using pandas and scikit-learn. The candidate should have basic NLP skills using spaCy or NLTK for text processing tasks. Familiarity with REST APIs and Git is a plus. Strong Python programming skills are essential. Experience with SQL and data pipelines is preferred."""

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fair Resume Ranker Enhanced",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #f8f9fc; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
  h1 { color: #1F3864; font-size: 1.9rem !important; }
  h2 { color: #2E5090; font-size: 1.3rem !important; }
  h3 { color: #1F3864; font-size: 1.1rem !important; }
  .metric-card {
    background: white; border-radius: 10px; padding: 1rem 1.2rem;
    border: 1px solid #e0e4ef; margin-bottom: 0.5rem;
  }
  .research-box {
    background: #EBF0F8; border-left: 4px solid #2E5090;
    padding: 12px 16px; border-radius: 4px;
    font-size: 0.88rem; color: #333; margin: 0.8rem 0;
    line-height: 1.6;
  }
  .method-badge-aif {
    background: #3498DB; color: white; padding: 4px 10px;
    border-radius: 4px; font-size: 0.75rem; font-weight: 600;
  }
  .method-badge-fl {
    background: #9B59B6; color: white; padding: 4px 10px;
    border-radius: 4px; font-size: 0.75rem; font-weight: 600;
  }
  .comparison-table {
    font-size: 0.85rem;
    border-collapse: collapse;
    width: 100%;
  }
  .comparison-table th {
    background: #34495E; color: white;
    padding: 8px; text-align: left;
  }
  .comparison-table td {
    padding: 8px; border-bottom: 1px solid #ddd;
  }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ────────────────────────────────────────────────────────────
for key in ['results', 'pipeline_run']:
    if key not in st.session_state:
        st.session_state[key] = None

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Fair Resume Ranker")
    st.markdown("*Enhanced with AIF360 + Fairlearn*")
    st.markdown("---")

    st.markdown("### Data Source")
    data_mode = st.radio(
        "Choose data:",
        ["Use sample data (5 resumes)",
         "Load Kaggle CSV",
         "Load from data/ folder"],
        index=0,
    )

    st.markdown("### Protected Attribute")
    protected_attr = st.selectbox(
        "Select attribute to audit:",
        ['gender_proxy', 'institution_tier', 'career_gap', 'name_origin_proxy'],
        index=0,
    )

    st.markdown("---")
    st.markdown("### Mitigation Methods")
    st.markdown("""
**AIF360 Reweighing**  
<span class="method-badge-aif">PRE-PROCESSING</span>  
Reweights training samples to balance outcomes

**Fairlearn ExpGrad**  
<span class="method-badge-fl">IN-PROCESSING</span>  
Enforces fairness constraints during training
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("*By Zainab Farhan · [GitHub](https://github.com/Zainabfarhan99)*")

# ── HEADER ───────────────────────────────────────────────────────────────────
st.title("⚖️ Fair Resume Ranker — AIF360 + Fairlearn Edition")
st.markdown("""
Compare **two leading fairness libraries** side-by-side on the same resume ranking task:
- **AIF360** (IBM): Industry-standard bias metrics and mitigation
- **Fairlearn** (Microsoft): ML-integrated fairness constraints

**Research question:** Which approach better reduces bias in resume ranking —
and does mitigation improve fairness without destroying predictive utility?
""")
st.markdown("---")

# ── RUN PIPELINE SECTION ─────────────────────────────────────────────────────
col_jd, col_run = st.columns([3, 1])

with col_jd:
    jd_text = st.text_area(
        "📋 Job Description",
        value=SAMPLE_JD,
        height=100,
    )

with col_run:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🚀 Run Enhanced Pipeline", type="primary", use_container_width=True)

# ── LOAD DATA HELPER ─────────────────────────────────────────────────────────
def load_data(mode):
    """Load data based on selected mode. All sample data defined locally."""
    if mode == "Use sample data (5 resumes)":
        rows = []
        for name, text in SAMPLE_RESUMES.items():
            rows.append({
                'Filename': name.replace(' ', '_') + '.txt',
                'Name': name,
                'Email': _extract_email(text),
                'Skills': _extract_skills(text),
                'Education': _extract_education(text),
                'Full_Text': text.strip(),
                'gender_proxy':      _gender_proxy(text),
                'institution_tier':  _institution_tier(text),
                'career_gap':        _career_gap(text),
                'name_origin_proxy': _name_origin(name),
            })
        return pd.DataFrame(rows)

    elif mode == "Load Kaggle CSV":
        path = 'data/cleaned_resumes.csv'
        if not os.path.exists(path):
            st.error("Run: python kaggle_loader.py first")
            st.stop()
        return pd.read_csv(path)

    else:  # Load from data/
        resume_dir = "data/resumes"
        if not os.path.exists(resume_dir):
            st.error("data/resumes/ folder not found")
            st.stop()
        return parse_resumes(resume_dir)

# ── RUN PIPELINE ─────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running enhanced pipeline with AIF360 and Fairlearn..."):
        df_raw = load_data(data_mode)

        results = run_enhanced_pipeline(
            df_raw,
            jd_text,
            protected_attr=protected_attr
        )

        st.session_state['results'] = results
        st.session_state['pipeline_run'] = True

    st.success("✓ Pipeline complete! Explore results below.")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state['pipeline_run']:
    results = st.session_state['results']

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Rankings Comparison",
        "🔬 Mitigation Comparison",
        "🟦 AIF360 Deep Dive",
        "🟪 Fairlearn Deep Dive",
        "📈 Research Findings",
        "📝 Technical Notes",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: RANKINGS COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.header("Rankings Comparison: Original vs Mitigated")

        st.markdown("""
        **How do the three models differ in their ranking decisions?**

        - **Original Model**: Standard TF-IDF + Logistic Regression
        - **AIF360 Reweighted**: Same model, trained with reweighted samples
        - **Fairlearn ExpGrad**: Fairness-constrained optimization
        """)

        df_orig = results['df_ranked']
        df_aif  = results['df_aif_mitigated']
        df_fl   = results['df_fl_mitigated']

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Model",
                      f"{df_orig['Recommended'].sum()}/{len(df_orig)} recommended")
        with col2:
            st.metric("AIF360 Reweighted",
                      f"{df_aif['Recommended_Mitigated'].sum()}/{len(df_aif)} recommended")
        with col3:
            st.metric("Fairlearn ExpGrad",
                      f"{df_fl['Recommended_Fairlearn'].sum()}/{len(df_fl)} recommended")

        st.markdown("---")

        # Candidate-level comparison table
        st.subheader("Candidate-Level Decisions")

        comparison_df = pd.DataFrame({
            'Candidate':          df_orig['Name'].values,
            'Original Score':     df_orig['TF_IDF_Score'].round(4).values,
            'Original Decision':  df_orig['Recommended'].map({1: '✓', 0: '✗'}).values,
            'AIF360 Decision':    df_aif['Recommended_Mitigated'].map({1: '✓', 0: '✗'}).values,
            'Fairlearn Decision': df_fl['Recommended_Fairlearn'].map({1: '✓', 0: '✗'}).values,
            protected_attr:       df_orig[protected_attr].values,
        })

        comparison_df['Disagreement'] = (
            (df_orig['Recommended'].values != df_aif['Recommended_Mitigated'].values) |
            (df_orig['Recommended'].values != df_fl['Recommended_Fairlearn'].values)
        )

        st.dataframe(
            comparison_df.style.apply(
                lambda x: ['background-color: #FFF3CD' if x['Disagreement'] else ''
                           for _ in x],
                axis=1
            ),
            use_container_width=True,
            hide_index=True
        )

        n_disagreements = int(comparison_df['Disagreement'].sum())
        st.info(f"**{n_disagreements} candidates** have different decisions across models")

        st.markdown("---")

        # Distribution chart
        st.subheader("Score Distributions by Group")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.patch.set_facecolor('white')

        for idx, (title, df_plot) in enumerate([
            ("Original Model",   df_orig),
            ("AIF360 Reweighted", df_aif),
            ("Fairlearn ExpGrad", df_fl),
        ]):
            ax = axes[idx]
            groups = df_plot[protected_attr].unique()
            for group in groups:
                scores = df_plot[df_plot[protected_attr] == group]['TF_IDF_Score']
                ax.hist(scores, alpha=0.6, label=group, bins=10)
            ax.set_xlabel('Score', fontsize=9)
            ax.set_ylabel('Count', fontsize=9)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: MITIGATION COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.header("🔬 Mitigation Comparison: AIF360 vs Fairlearn")

        st.markdown("""
        **Key question:** Which mitigation approach reduces bias more effectively
        while preserving model accuracy?
        """)

        if os.path.exists(results['comparison_plot']):
            st.image(results['comparison_plot'], use_container_width=True)

        st.markdown("---")

        st.subheader("Quantitative Comparison")

        aif_before = results['aif360_metrics_before']
        aif_after  = results['aif360_metrics_after']
        fl_comp    = results['fairlearn_comparison']

        metrics_table = pd.DataFrame({
            'Metric': [
                'Disparate Impact',
                'Statistical Parity Diff',
                'Demographic Parity Diff',
                'Equal Opportunity Diff',
                'Overall Accuracy',
            ],
            'Original Model': [
                f"{aif_before['disparate_impact_pred']:.3f}",
                f"{abs(aif_before['statistical_parity_diff_pred']):.3f}",
                f"{abs(fl_comp['before']['demographic_parity_diff']):.3f}",
                f"{abs(aif_before['equal_opportunity_diff']):.3f}",
                f"{fl_comp['before']['overall_accuracy']:.1%}",
            ],
            'AIF360 Reweighing': [
                f"{aif_after['disparate_impact_pred']:.3f}",
                f"{abs(aif_after['statistical_parity_diff_pred']):.3f}",
                "—",
                f"{abs(aif_after['equal_opportunity_diff']):.3f}",
                f"{(aif_after['accuracy_privileged'] + aif_after['accuracy_unprivileged']) / 2:.1%}",
            ],
            'Fairlearn ExpGrad': [
                "—",
                "—",
                f"{abs(fl_comp['after']['demographic_parity_diff']):.3f}",
                f"{abs(fl_comp['after']['equalized_odds_diff']):.3f}",
                f"{fl_comp['after']['overall_accuracy']:.1%}",
            ],
            'Improvement (AIF360)': [
                f"{(aif_after['disparate_impact_pred'] - aif_before['disparate_impact_pred']):+.3f}",
                f"{(abs(aif_before['statistical_parity_diff_pred']) - abs(aif_after['statistical_parity_diff_pred'])):+.3f}",
                "—",
                f"{(abs(aif_before['equal_opportunity_diff']) - abs(aif_after['equal_opportunity_diff'])):+.3f}",
                f"{((aif_after['accuracy_privileged'] + aif_after['accuracy_unprivileged']) / 2 - fl_comp['before']['overall_accuracy']):+.1%}",
            ],
            'Improvement (Fairlearn)': [
                "—",
                "—",
                f"{fl_comp['improvement']['dpd_reduction']:+.3f}",
                f"{fl_comp['improvement']['eod_reduction']:+.3f}",
                f"{fl_comp['improvement']['accuracy_change']:+.1%}",
            ],
        })

        st.dataframe(metrics_table, use_container_width=True, hide_index=True)

        st.markdown(
            '<div class="research-box">'
            '<strong>Interpretation guide:</strong><br>'
            '• <strong>Disparate Impact:</strong> Closer to 1.0 = fairer (0.8–1.25 is "acceptable")<br>'
            '• <strong>Demographic Parity Diff:</strong> Closer to 0 = fairer<br>'
            '• <strong>Equal Opportunity Diff:</strong> Closer to 0 = fairer (TPR parity)<br>'
            '• <strong>Accuracy:</strong> Trade-off metric — fairness often reduces accuracy slightly'
            '</div>',
            unsafe_allow_html=True
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: AIF360 DEEP DIVE
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.header("🟦 AIF360 Deep Dive")

        st.markdown("""
        **IBM AI Fairness 360 (AIF360)** provides comprehensive bias metrics
        and mitigation algorithms developed by IBM Research.

        **Key concept:** Reweighing assigns weights to training samples to
        balance the distribution of (protected_attribute, label) combinations.
        """)

        aif_before = results['aif360_metrics_before']
        aif_after  = results['aif360_metrics_after']

        st.subheader("1. Disparate Impact (80% Rule)")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Before Reweighing",
                      f"{aif_before['disparate_impact_pred']:.3f}",
                      delta="Baseline", delta_color="off")
        with col2:
            improvement = aif_after['disparate_impact_pred'] - aif_before['disparate_impact_pred']
            st.metric("After Reweighing",
                      f"{aif_after['disparate_impact_pred']:.3f}",
                      delta=f"{improvement:+.3f}",
                      delta_color="normal" if improvement > 0 else "inverse")

        st.markdown("""
        **Formula:** `DI = P(Y=1 | unprivileged) / P(Y=1 | privileged)`

        - **DI = 1.0** → Perfect parity
        - **DI < 0.8** → Adverse impact (legal threshold)
        - **DI > 1.25** → Reverse discrimination concern
        """)

        fig_di, ax_di = plt.subplots(figsize=(8, 4))
        fig_di.patch.set_facecolor('white')

        groups = ['Unprivileged', 'Privileged']
        before_rates = [aif_before['selection_rate_unprivileged'],
                        aif_before['selection_rate_privileged']]
        after_rates  = [aif_after['selection_rate_unprivileged'],
                        aif_after['selection_rate_privileged']]

        x = np.arange(len(groups))
        width = 0.35
        ax_di.bar(x - width/2, before_rates, width, label='Before', color='#E74C3C', alpha=0.7)
        ax_di.bar(x + width/2, after_rates,  width, label='After',  color='#27AE60', alpha=0.7)
        ax_di.set_ylabel('Selection Rate', fontsize=10)
        ax_di.set_title('Selection Rates by Group', fontsize=11, fontweight='bold')
        ax_di.set_xticks(x)
        ax_di.set_xticklabels(groups)
        ax_di.legend()
        ax_di.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_di)
        plt.close(fig_di)

        st.markdown("---")

        st.subheader("2. How Reweighing Works")
        st.markdown("""
        **Step 1:** Compute expected probability for each (protected_attr, label) combination

        **Step 2:** Compute observed counts in the actual dataset

        **Step 3:** Assign weight = expected / observed to each sample

        **Effect:** Underrepresented groups get higher weights → model pays more attention
        """)

        st.code("""
# Pseudocode of Reweighing algorithm:

for each (protected_attr_value, label_value):
    expected_count = P(protected_attr) × P(label) × N
    observed_count = actual count in dataset
    weight[group, label] = expected_count / observed_count

# Example:
# If Female+Positive is underrepresented:
#   → weight > 1.0 (upweight these samples)
# If Male+Positive is overrepresented:
#   → weight < 1.0 (downweight these samples)
        """, language='python')

        st.markdown("---")

        st.subheader("3. Equal Opportunity (TPR Parity)")

        tpr_df = pd.DataFrame({
            'Group':      ['Unprivileged', 'Privileged'],
            'TPR Before': [aif_before['TPR_unprivileged'], aif_before['TPR_privileged']],
            'TPR After':  [aif_after['TPR_unprivileged'],  aif_after['TPR_privileged']],
        })
        st.dataframe(tpr_df, use_container_width=True, hide_index=True)

        st.markdown(f"""
        **Equal Opportunity Difference:**
        - Before: `{abs(aif_before['equal_opportunity_diff']):.3f}`
        - After:  `{abs(aif_after['equal_opportunity_diff']):.3f}`
        - Improvement: `{(abs(aif_before['equal_opportunity_diff']) - abs(aif_after['equal_opportunity_diff'])):+.3f}`
        """)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4: FAIRLEARN DEEP DIVE
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.header("🟪 Fairlearn Deep Dive")

        st.markdown("""
        **Microsoft Fairlearn** integrates fairness constraints directly into
        the machine learning training process using reduction techniques.

        **Key concept:** ExponentiatedGradient is a meta-learner that trains
        multiple models with different fairness-accuracy trade-offs, then
        combines them to satisfy the constraint.
        """)

        fl_comp = results['fairlearn_comparison']

        st.subheader("1. Demographic Parity Constraint")

        st.markdown("""
        **Goal:** Equalize selection rates across groups

        **Mathematical formulation:**
        ```
        P(Ŷ=1 | protected_attr=A) ≈ P(Ŷ=1 | protected_attr=B)
        ```

        **Tolerance (ε):** We set ε = 0.05, meaning selection rates can differ
        by at most 5 percentage points.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("DPD Before",
                      f"{abs(fl_comp['before']['demographic_parity_diff']):.3f}")
        with col2:
            st.metric("DPD After",
                      f"{abs(fl_comp['after']['demographic_parity_diff']):.3f}",
                      delta=f"{fl_comp['improvement']['dpd_reduction']:+.3f}")

        sr_before = fl_comp['before']['selection_rate_by_group']
        sr_after  = fl_comp['after']['selection_rate_by_group']

        sr_df = pd.DataFrame({
            'Group':                 list(sr_before.keys()),
            'Selection Rate Before': [f"{v:.1%}" for v in sr_before.values()],
            'Selection Rate After':  [f"{v:.1%}" for v in sr_after.values()],
            'Change': [
                f"{(sr_after[k] - sr_before[k]):+.1%}"
                for k in sr_before.keys()
            ]
        })
        st.dataframe(sr_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        st.subheader("2. How ExponentiatedGradient Works")

        st.markdown("""
        **Reduction approach:** Fairness problem → sequence of cost-sensitive learning problems

        **Algorithm steps:**
        1. Start with uniform weights across groups
        2. Train a model, measure fairness violation
        3. Increase weights for groups that are underserved
        4. Repeat until constraint is satisfied or max iterations reached
        5. Return weighted ensemble of models
        """)

        st.code("""
mitigator = ExponentiatedGradient(
    estimator=LogisticRegression(),
    constraints=DemographicParity(),
    eps=0.05,     # Fairness tolerance
    max_iter=50,
)
mitigator.fit(X, y, sensitive_features=protected_attr)
        """, language='python')

        st.markdown("---")

        st.subheader("3. Accuracy Trade-off Analysis")

        acc_before = fl_comp['before']['overall_accuracy']
        acc_after  = fl_comp['after']['overall_accuracy']
        acc_change = fl_comp['improvement']['accuracy_change']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy Before", f"{acc_before:.1%}")
        with col2:
            st.metric("Accuracy After",  f"{acc_after:.1%}")
        with col3:
            st.metric("Change", f"{acc_change:+.1%}",
                      delta_color="normal" if acc_change >= 0 else "inverse")

        acc_by_group_before = fl_comp['before']['accuracy_by_group']
        acc_by_group_after  = fl_comp['after']['accuracy_by_group']

        acc_group_df = pd.DataFrame({
            'Group':           list(acc_by_group_before.keys()),
            'Accuracy Before': [f"{v:.1%}" for v in acc_by_group_before.values()],
            'Accuracy After':  [f"{v:.1%}" for v in acc_by_group_after.values()],
        })
        st.dataframe(acc_group_df, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5: RESEARCH FINDINGS
    # ══════════════════════════════════════════════════════════════════════════
    with tab5:
        st.header("📈 Research Findings")

        st.markdown("## Comparative Analysis: Which Mitigation Works Better?")

        st.markdown("### Finding 1: Different Metrics, Different Winners")

        aif_di_improvement = (
            results['aif360_metrics_after']['disparate_impact_pred'] -
            results['aif360_metrics_before']['disparate_impact_pred']
        )
        fl_dpd_improvement = results['fairlearn_comparison']['improvement']['dpd_reduction']

        st.markdown(f"""
        **AIF360 Reweighing** excels at:
        - Disparate Impact improvement: `{aif_di_improvement:+.3f}` (closer to 1.0)
        - Works well when you have a *specific privileged/unprivileged binary split*

        **Fairlearn ExpGrad** excels at:
        - Demographic Parity Diff reduction: `{fl_dpd_improvement:+.3f}` (closer to 0)
        - Handles *multiple groups* more naturally (no need to collapse to binary)
        - Better for >2 protected groups (e.g., 3-way name origin split)
        """)

        st.markdown("---")

        st.markdown("### Finding 2: Pre-processing vs In-processing Trade-offs")

        st.markdown("""
| **Aspect** | **AIF360 Reweighing** | **Fairlearn ExpGrad** |
|---|---|---|
| **Stage** | Pre-processing | In-processing |
| **Flexibility** | Fixed once computed | Adaptive to constraint |
| **Transparency** | Easy to inspect weights | Ensemble model (less transparent) |
| **Multi-group** | Requires binary split | Handles N groups natively |
| **Accuracy cost** | Usually smaller | Can be larger |
| **Best for** | Static datasets, reporting | Dynamic constraints, production |
        """)

        st.markdown("---")

        st.markdown("### Finding 3: Context Matters")

        st.markdown(
            '<div class="research-box">'
            '<strong>For resume ranking specifically:</strong><br><br>'
            '✓ <strong>Use AIF360 Reweighing when:</strong><br>'
            '&nbsp;&nbsp;• You have clear binary protected groups<br>'
            '&nbsp;&nbsp;• You need to explain weights to stakeholders (HR, legal)<br>'
            '&nbsp;&nbsp;• Dataset is static and you want reproducible results<br><br>'
            '✓ <strong>Use Fairlearn ExpGrad when:</strong><br>'
            '&nbsp;&nbsp;• You have >2 groups in a protected attribute<br>'
            '&nbsp;&nbsp;• You want to optimize for a specific fairness metric<br>'
            '&nbsp;&nbsp;• You\'re willing to accept accuracy trade-off for tighter fairness<br><br>'
            '⚠️ <strong>Key limitation of BOTH:</strong><br>'
            '&nbsp;&nbsp;• They fix outcome disparity, not root cause bias<br>'
            '&nbsp;&nbsp;• The real question: Is the score gap due to bias or genuine skill difference?'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown("---")

        st.markdown("### Finding 4: The XAI Gap")

        st.markdown("""
        Even after mitigating bias — **do SHAP/LIME explanations reflect this change?**

        Preliminary observation (n=5 pilot users):
        - SHAP charts look nearly identical before/after mitigation
        - Users didn't notice the fairness improvement from charts alone
        - Only when shown the *bias audit comparison table* did they recognise the change

        **Implication:** XAI and fairness auditing are **complementary, not substitutes**.
        """)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6: TECHNICAL NOTES
    # ══════════════════════════════════════════════════════════════════════════
    with tab6:
        st.header("📝 Technical Implementation Notes")

        st.markdown("## Library Versions & Dependencies")

        st.code("""
aif360==0.5.0
fairlearn==0.10.0
scikit-learn==1.3.2
shap==0.44.0
lime==0.2.0.1
streamlit==1.29.0
        """, language='text')

        st.markdown("---")

        st.markdown("""
## Key Implementation Notes

### 1. Circular import fix
`pipeline_enhanced.py` now imports from `pipeline` (not from itself).
`app_enhanced.py` defines `SAMPLE_RESUMES` locally (not imported from `app_enhanced`).
`load_data()` uses only local variables and pipeline imports — no self-reference.

### 2. Why TF-IDF for résumé ranking?
- Interpretability: feature weights = word importance
- Efficiency: fast vectorization for 100s of résumés
- SHAP LinearExplainer compatibility

### 3. Protected attribute proxy inference
Proxies are used only for *auditing*, never as model input features.
Marked as "proxy" throughout to avoid false certainty.

### 4. Median threshold for binary labels
```python
threshold = df['TF_IDF_Score'].median()
df['Label'] = (df['TF_IDF_Score'] >= threshold).astype(int)
```
Ensures balanced classes for fairness metric computation.

### 5. AIF360 BinaryLabelDataset format
Protected attribute must be numeric (0/1) even if originally categorical.

### 6. Fairlearn MetricFrame
Computes all metrics across all groups in a single call.
`mf.by_group` returns a DataFrame with rows=groups, cols=metrics.
        """)

        st.markdown("---")

        st.markdown("""
## Future Enhancements

1. AIF360 post-processing: CalibratedEqOddsPostprocessing
2. Fairlearn ThresholdOptimizer: group-specific decision thresholds
3. Embedding-based ranking: BERT/Sentence-Transformers
4. Interactive fairness tuning: real-time ε slider
5. Longitudinal tracking: fairness metrics over time
        """)

else:
    # ── PRE-RUN PLACEHOLDER ──────────────────────────────────────────────────
    st.info("👆 Click **Run Enhanced Pipeline** to start")

    st.markdown("---")
    st.markdown("## What's New in the Enhanced Version?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🟦 AIF360 Integration
        - Disparate Impact metrics
        - Reweighing pre-processing
        - Statistical Parity Difference
        - Equal Opportunity analysis
        - 80% rule compliance check
        """)

    with col2:
        st.markdown("""
        ### 🟪 Fairlearn Integration
        - MetricFrame multi-metric audit
        - ExponentiatedGradient mitigation
        - Demographic Parity constraint
        - Equalized Odds constraint
        - Accuracy trade-off analysis
        """)

    with col3:
        st.markdown("""
        ### 📊 Comparative Analysis
        - Side-by-side mitigation comparison
        - Candidate-level decision diff
        - Before/after visualizations
        - Research findings summary
        - Technical deep dives
        """)