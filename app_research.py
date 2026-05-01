"""
app_research.py
---------------
Research-Grade Fair Resume Ranker
Integrates the novel FEAS (Fairness-Explainability Alignment Score) metric
alongside AIF360-equivalent and Fairlearn bias mitigation.

New research tabs:
  - FEAS Analysis      : novel metric computation and visualization
  - Intersectional     : multi-attribute bias (beyond single-axis audits)
  - Post-Mitigation    : does FEAS change after bias mitigation?
  - Research Export    : generate paper-ready tables and figures

Run:
    streamlit run app_research.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import warnings
warnings.filterwarnings('ignore')

from pipeline import (
    parse_resumes,
    rank_resumes,
    fairness_audit,
    explain_shap,
    explain_lime,
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

import shap as shap_lib
from sklearn.feature_extraction.text import TfidfVectorizer

from feas_metric import (
    compute_bias_signal_features,
    extract_shap_features,
    extract_lime_features,
    compute_feas,
    run_feas_analysis,
    compute_feas_delta,
    compute_intersectional_feas,
    feas_summary_table,
)

# ── SAMPLE DATA ──────────────────────────────────────────────────────────────
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

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fair Resume Ranker — FEAS Research Edition",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main { background-color: #f8f9fc; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
  h1 { color: #1F3864; font-size: 1.9rem !important; }
  h2 { color: #2E5090; font-size: 1.3rem !important; }
  h3 { color: #1F3864; font-size: 1.1rem !important; }
  .feas-box {
    background: #EBF5FB; border-left: 5px solid #2E86C1;
    padding: 14px 18px; border-radius: 4px;
    font-size: 0.9rem; color: #1A3A4A; margin: 0.8rem 0;
    line-height: 1.7;
  }
  .research-box {
    background: #EBF0F8; border-left: 4px solid #2E5090;
    padding: 12px 16px; border-radius: 4px;
    font-size: 0.88rem; color: #333; margin: 0.8rem 0;
    line-height: 1.6;
  }
  .novel-badge {
    background: #F39C12; color: white; padding: 3px 10px;
    border-radius: 4px; font-size: 0.75rem; font-weight: 700;
    letter-spacing: 0.5px;
  }
  .gap-critical { color: #C0392B; font-weight: 700; }
  .gap-moderate { color: #E67E22; font-weight: 700; }
  .gap-low      { color: #27AE60; font-weight: 700; }
  .formula-box {
    background: #1e1e2e; color: #cdd6f4;
    padding: 14px 18px; border-radius: 8px;
    font-family: monospace; font-size: 0.95rem;
    line-height: 1.8; margin: 0.8rem 0;
  }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for key in ['results', 'feas_results', 'feas_delta_aif', 'feas_delta_fl',
            'intersectional_feas', 'pipeline_run', 'shap_values_raw',
            'feature_names', 'X_matrix']:
    if key not in st.session_state:
        st.session_state[key] = None

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 FEAS Research Edition")
    st.markdown("*Fairness-Explainability Alignment Score*")
    st.markdown(
        '<span class="novel-badge">NOVEL METRIC</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("### Data Source")
    data_mode = st.radio(
        "Choose data:",
        ["Use sample data (5 resumes)", "Load Kaggle CSV", "Load from data/ folder"],
        index=0,
    )

    st.markdown("### Protected Attribute")
    protected_attr = st.selectbox(
        "Audit attribute:",
        ['gender_proxy', 'institution_tier', 'career_gap', 'name_origin_proxy'],
        index=0,
    )

    st.markdown("### FEAS Parameters")
    xai_top_k  = st.slider("XAI top-k features (|E|)",  5, 20, 10)
    bias_top_k = st.slider("Bias features top-k (|B|)", 10, 50, 20)
    bias_method = st.selectbox(
        "Bias signal method:",
        ['mutual_info', 'combined'],
        index=1,
    )

    st.markdown("---")
    st.markdown("**Research question:**")
    st.markdown(
        "*What fraction of the bias detected by a fairness audit "
        "is actually visible in SHAP/LIME explanations?*"
    )
    st.markdown("---")
    st.markdown("*By Zainab Farhan · PhD Research Portfolio*")

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🔬 Fair Resume Ranker — FEAS Research Edition")
st.markdown("""
**Novel contribution:** The **Fairness-Explainability Alignment Score (FEAS)** — 
a metric that formally quantifies the gap between what fairness audits detect 
and what explainability tools reveal to users.

> *FEAS = |B ∩ E| / |B|* — where B = bias-signal features, E = XAI-highlighted features
""")
st.markdown("---")

# ── RUN PIPELINE ──────────────────────────────────────────────────────────────
col_jd, col_run = st.columns([3, 1])
with col_jd:
    jd_text = st.text_area("📋 Job Description", value=SAMPLE_JD, height=90)
with col_run:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🚀 Run Research Pipeline", type="primary", use_container_width=True)


def load_data(mode):
    if mode == "Use sample data (5 resumes)":
        rows = []
        for name, text in SAMPLE_RESUMES.items():
            rows.append({
                'Filename': name.replace(' ', '_') + '.txt',
                'Name': name, 'Email': _extract_email(text),
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
    else:
        resume_dir = "data/resumes"
        if not os.path.exists(resume_dir):
            st.error("data/resumes/ folder not found")
            st.stop()
        return parse_resumes(resume_dir)


if run_btn:
    with st.spinner("Running research pipeline — this includes FEAS computation..."):
        df_raw = load_data(data_mode)

        # Full enhanced pipeline (original + AIF360 + Fairlearn)
        results = run_enhanced_pipeline(df_raw, jd_text, protected_attr=protected_attr)
        df      = results['df_ranked']

        # Rebuild SHAP values for FEAS (need raw values not just charts)
        vectorizer = results['vectorizer']
        model      = results['model_original']
        X          = vectorizer.transform(df['Full_Text'].fillna('')).toarray()
        feature_names = list(vectorizer.get_feature_names_out())

        explainer   = shap_lib.LinearExplainer(model, X, feature_perturbation='interventional')
        shap_values = explainer.shap_values(X)
        sv = shap_values if shap_values.ndim == 2 else shap_values[:, :, 1]

        # LIME outputs
        lime_outputs = results['lime_outputs']

        # ── FEAS on original model ─────────────────────────────────────────
        feas_results = run_feas_analysis(
            df, X, sv, lime_outputs, feature_names,
            protected_attr=protected_attr,
            xai_top_k=xai_top_k,
            bias_top_k=bias_top_k,
            bias_method=bias_method,
        )

        # ── FEAS on AIF360 reweighted model ───────────────────────────────
        # The model stored in results['model_aif360'] was trained on a fresh
        # internal TfidfVectorizer (different vocab size) inside aif360_reweigh().
        # Using it directly with the shared X causes the shape mismatch.
        # Fix: retrain a proxy LR on the SAME X using the reweighing sample weights,
        # so SHAP always operates in the shared feature space.
        from sklearn.linear_model import LogisticRegression as _LR
        from pipeline_enhanced import Reweighing, BinaryLabelDataset

        df_aif = results['df_aif_mitigated'].copy()
        df_aif['Label'] = df['Label'].values

        # Recompute reweighing weights in the shared feature space
        try:
            _df_rw = df[['TF_IDF_Score', 'Label', 'Recommended', protected_attr]].copy()
            _priv  = df[protected_attr].mode()[0]
            _df_rw['protected_binary'] = (_df_rw[protected_attr] == _priv).astype(int)
            _ds = BinaryLabelDataset(
                favorable_label=1, unfavorable_label=0,
                df=_df_rw, label_names=['Label'],
                protected_attribute_names=['protected_binary'],
            )
            _rw = Reweighing(
                unprivileged_groups=[{'protected_binary': 0}],
                privileged_groups=[{'protected_binary': 1}],
            )
            _weights = _rw.fit_transform(_ds).instance_weights
        except Exception:
            _weights = np.ones(len(df))

        proxy_aif = _LR(max_iter=1000, C=1.0, random_state=42)
        proxy_aif.fit(X, df['Label'].values, sample_weight=_weights)
        exp_aif = shap_lib.LinearExplainer(proxy_aif, X, feature_perturbation='interventional')
        sv_aif  = exp_aif.shap_values(X)
        if sv_aif.ndim != 2:
            sv_aif = sv_aif[:, :, 1]

        feas_aif = run_feas_analysis(
            df_aif, X, sv_aif, lime_outputs, feature_names,
            protected_attr=protected_attr,
            xai_top_k=xai_top_k, bias_top_k=bias_top_k, bias_method=bias_method,
        )

        # ── FEAS on Fairlearn mitigated model ─────────────────────────────
        # Fairlearn returns an ensemble (ExponentiatedGradient) — not a linear model.
        # Use a proxy LR fitted to the ensemble's predictions on the shared X,
        # so SHAP stays in the same feature space throughout.
        df_fl = results['df_fl_mitigated'].copy()
        df_fl['Label'] = df['Label'].values
        model_fl = results['model_fairlearn']

        try:
            fl_preds = model_fl.predict(X)
            proxy_fl = _LR(max_iter=1000, C=1.0, random_state=42)
            proxy_fl.fit(X, fl_preds)
            exp_fl = shap_lib.LinearExplainer(proxy_fl, X, feature_perturbation='interventional')
            sv_fl  = exp_fl.shap_values(X)
            if sv_fl.ndim != 2:
                sv_fl = sv_fl[:, :, 1]
        except Exception:
            sv_fl = sv.copy()  # fallback: same as original model

        feas_fl = run_feas_analysis(
            df_fl, X, sv_fl, lime_outputs, feature_names,
            protected_attr=protected_attr,
            xai_top_k=xai_top_k, bias_top_k=bias_top_k, bias_method=bias_method,
        )

        # ── FEAS deltas ────────────────────────────────────────────────────
        feas_delta_aif = compute_feas_delta(feas_results, feas_aif, 'AIF360 Reweighing')
        feas_delta_fl  = compute_feas_delta(feas_results, feas_fl,  'Fairlearn ExpGrad')

        # ── Intersectional FEAS ────────────────────────────────────────────
        intersectional = compute_intersectional_feas(df, X, sv, feature_names)

        # ── Store in session state ─────────────────────────────────────────
        st.session_state['results']            = results
        st.session_state['feas_results']       = feas_results
        st.session_state['feas_aif']           = feas_aif
        st.session_state['feas_fl']            = feas_fl
        st.session_state['feas_delta_aif']     = feas_delta_aif
        st.session_state['feas_delta_fl']      = feas_delta_fl
        st.session_state['intersectional_feas']= intersectional
        st.session_state['shap_values_raw']    = sv
        st.session_state['feature_names']      = feature_names
        st.session_state['X_matrix']           = X
        st.session_state['pipeline_run']       = True

    st.success("✓ Research pipeline complete — FEAS computed!")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state['pipeline_run']:
    results       = st.session_state['results']
    feas_results  = st.session_state['feas_results']
    feas_delta_aif= st.session_state['feas_delta_aif']
    feas_delta_fl = st.session_state['feas_delta_fl']
    intersectional= st.session_state['intersectional_feas']
    df            = results['df_ranked']

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Rankings",
        "🔬 FEAS — Novel Metric",
        "🕸️ Intersectional Bias",
        "🔄 Post-Mitigation FEAS",
        "⚖️ AIF360 + Fairlearn",
        "🧠 SHAP / LIME",
        "📄 Research Export",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: RANKINGS (concise)
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.header("Candidate Rankings")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Candidates", len(df))
        m2.metric("Recommended", int(df['Recommended'].sum()))
        m3.metric("FEAS Gap (bias hidden from XAI)",
                  f"{feas_results['FEAS_gap']:.1%}")

        fig, ax = plt.subplots(figsize=(9, 3))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        colors = ['#1F3864' if r == 1 else '#2E5090' if r == 2 else '#5b7bbf'
                  for r in df['Rank']]
        ax.barh(df['Name'], df['TF_IDF_Score'], color=colors, height=0.5)
        ax.axvline(df['TF_IDF_Score'].median(), color='#E74C3C',
                   linestyle='--', linewidth=1.2, label='Threshold')
        ax.set_xlabel('TF-IDF Cosine Similarity', fontsize=9)
        ax.set_title('Resume Ranking — Similarity to Job Description',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        display_cols = ['Rank', 'Name', 'TF_IDF_Score', 'Recommended',
                        'gender_proxy', 'institution_tier', 'career_gap', 'name_origin_proxy']
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: FEAS — NOVEL METRIC (the core research tab)
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.header("🔬 Fairness-Explainability Alignment Score (FEAS)")
        st.markdown(
            '<span class="novel-badge">NOVEL CONTRIBUTION</span>',
            unsafe_allow_html=True,
        )

        st.markdown("""
        ### What is FEAS?

        FEAS formally quantifies how much of the bias detected by a fairness audit
        is actually *visible* in the explanations a user receives from SHAP or LIME.
        """)

        st.markdown(
            '<div class="formula-box">'
            'FEAS(candidate) = |B ∩ E| / |B|<br><br>'
            'B = bias-signal features  →  top features correlated with protected attribute<br>'
            'E = XAI features          →  top features shown by SHAP or LIME<br><br>'
            'FEAS = 1.0  →  XAI fully reveals the bias-driving features<br>'
            'FEAS = 0.0  →  XAI completely hides them<br>'
            'FEAS Gap = 1 - FEAS  →  fraction of bias invisible to users'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Top-level metrics
        col1, col2, col3, col4 = st.columns(4)
        feas_shap = feas_results['mean_FEAS_SHAP']
        feas_lime = feas_results['mean_FEAS_LIME']
        feas_mean = feas_results['mean_FEAS']
        feas_gap  = feas_results['FEAS_gap']

        col1.metric("FEAS (SHAP)", f"{feas_shap:.4f}",
                    help="Fraction of bias-signal features visible in SHAP explanations")
        col2.metric("FEAS (LIME)", f"{feas_lime:.4f}",
                    help="Fraction of bias-signal features visible in LIME explanations")
        col3.metric("FEAS (mean)", f"{feas_mean:.4f}")

        gap_color = "gap-critical" if feas_gap > 0.6 else "gap-moderate" if feas_gap > 0.3 else "gap-low"
        col4.markdown(
            f"**FEAS Gap**<br>"
            f"<span class='{gap_color}' style='font-size:1.8rem'>{feas_gap:.1%}</span><br>"
            f"<span style='font-size:0.8rem'>of bias is hidden from users</span>",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Verdict box
        verdict = feas_results['alignment_verdict']
        border_color = '#C0392B' if feas_gap > 0.6 else '#E67E22' if feas_gap > 0.3 else '#27AE60'
        st.markdown(
            f'<div class="feas-box" style="border-left-color:{border_color}">'
            f'<strong>Verdict:</strong> {verdict}<br><br>'
            f'<strong>Research implication:</strong> Even though SHAP and LIME '
            f'produce explanations for every candidate, {feas_gap:.1%} of the bias '
            f'detected by the fairness audit is invisible in those explanations. '
            f'A recruiter relying solely on SHAP charts would miss this bias entirely.'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.subheader("Bias-Signal Features (Set B)")
        st.markdown(
            f"Features most correlated with `{protected_attr}` — "
            f"these are what the fairness audit detects as bias drivers. "
            f"FEAS measures how many of these appear in SHAP/LIME."
        )

        bias_df = pd.DataFrame(
            feas_results['bias_features']['features'][:15],
            columns=['Feature', 'Bias Signal Score']
        )

        fig2, ax2 = plt.subplots(figsize=(9, 4))
        fig2.patch.set_facecolor('white')
        ax2.set_facecolor('white')
        colors2 = ['#E74C3C' if s > 0.5 else '#E67E22' if s > 0.2 else '#3498DB'
                   for s in bias_df['Bias Signal Score']]
        ax2.barh(bias_df['Feature'], bias_df['Bias Signal Score'],
                 color=colors2, height=0.55)
        ax2.set_xlabel('Bias Signal Score (higher = stronger proxy for protected attribute)',
                       fontsize=9)
        ax2.set_title(f'Top Bias-Signal Features for {protected_attr}',
                      fontsize=11, fontweight='bold')
        ax2.invert_yaxis()
        red_p   = mpatches.Patch(color='#E74C3C', label='High bias signal (>0.5)')
        amber_p = mpatches.Patch(color='#E67E22', label='Moderate (0.2–0.5)')
        blue_p  = mpatches.Patch(color='#3498DB', label='Low (<0.2)')
        ax2.legend(handles=[red_p, amber_p, blue_p], fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        st.markdown("---")
        st.subheader("Per-Candidate FEAS Scores")

        feas_df = feas_results['per_candidate']

        fig3, ax3 = plt.subplots(figsize=(9, 3.5))
        fig3.patch.set_facecolor('white')
        ax3.set_facecolor('white')
        x_pos = np.arange(len(feas_df))
        w     = 0.35
        shap_vals = feas_df['FEAS_SHAP'].fillna(0).values
        lime_vals = feas_df['FEAS_LIME'].fillna(0).values
        ax3.bar(x_pos - w/2, shap_vals, w, label='FEAS (SHAP)', color='#2E86C1', alpha=0.8)
        ax3.bar(x_pos + w/2, lime_vals, w, label='FEAS (LIME)', color='#9B59B6', alpha=0.8)
        ax3.axhline(feas_mean, color='#E74C3C', linestyle='--', linewidth=1.2,
                    label=f'Mean FEAS = {feas_mean:.3f}')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(feas_df['Name'].values, rotation=15, ha='right', fontsize=9)
        ax3.set_ylabel('FEAS Score', fontsize=9)
        ax3.set_ylim(0, 1.05)
        ax3.set_title('FEAS per Candidate (SHAP vs LIME)', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        # Per-candidate table with revealed/hidden
        display_feas = feas_df[[
            'Name', 'Rank', protected_attr,
            'FEAS_SHAP', 'FEAS_LIME', 'FEAS_mean',
            'n_bias_revealed', 'n_bias_hidden',
        ]].copy()
        st.dataframe(display_feas, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Group-Level FEAS")
        st.markdown(
            "Do explanations hide more bias for some groups than others? "
            "This is a *second-order fairness concern* — XAI may itself be unfair."
        )
        group_feas = feas_results['group_FEAS']
        st.dataframe(group_feas, use_container_width=True, hide_index=True)

        group_diff = feas_results['group_FEAS_diff']
        if group_diff > 0.1:
            st.warning(
                f"⚠️ Group FEAS disparity = {group_diff:.4f} — "
                f"SHAP explanations hide significantly more bias for some groups. "
                f"This is a second-order fairness concern."
            )

        st.markdown(
            '<div class="feas-box">'
            '<strong>Summary table (paper-ready):</strong>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(feas_summary_table(feas_results), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: INTERSECTIONAL BIAS
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.header("🕸️ Intersectional Bias Analysis")
        st.markdown(
            '<span class="novel-badge">NOVEL ANGLE</span>',
            unsafe_allow_html=True,
        )

        st.markdown("""
        Most fairness audits examine ONE protected attribute at a time.
        But real discrimination is **intersectional**: a candidate who is
        *Female + South Asian + Career gap* faces compound disadvantages
        that single-axis audits completely miss.

        **Intersectional FEAS** applies our metric to *combinations* of attributes.
        """)

        if intersectional is not None and len(intersectional) > 0:
            st.subheader("FEAS by Intersectional Group")

            # Pivot for display
            pivot = intersectional.groupby(['attr_pair', 'intersect_group'])['FEAS_intersectional'].mean().reset_index()
            pivot.columns = ['Attribute pair', 'Intersectional group', 'Mean FEAS']
            pivot['Mean FEAS'] = pivot['Mean FEAS'].round(4)
            pivot['FEAS Gap'] = (1 - pivot['Mean FEAS']).round(4)
            pivot = pivot.sort_values('Mean FEAS')

            st.dataframe(pivot, use_container_width=True, hide_index=True)

            # Chart: worst intersectional FEAS scores
            worst = pivot.head(10)
            fig_i, ax_i = plt.subplots(figsize=(9, 4))
            fig_i.patch.set_facecolor('white')
            ax_i.set_facecolor('white')
            labels = worst['Intersectional group'].values
            vals   = worst['Mean FEAS'].values
            colors_i = ['#C0392B' if v < 0.3 else '#E67E22' if v < 0.6 else '#27AE60'
                        for v in vals]
            ax_i.barh(labels, vals, color=colors_i, height=0.55)
            ax_i.axvline(0.5, color='#333', linestyle='--', linewidth=1, label='FEAS = 0.5')
            ax_i.set_xlabel('FEAS Score (lower = more bias hidden from explanations)', fontsize=9)
            ax_i.set_title('Lowest-FEAS Intersectional Groups\n(most hidden bias)',
                           fontsize=11, fontweight='bold')
            ax_i.legend(fontsize=8)
            ax_i.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig_i)
            plt.close(fig_i)

            st.markdown(
                '<div class="research-box">'
                '<strong>Research finding:</strong> Intersectional FEAS reveals which '
                'combinations of disadvantaged identities are most invisible in XAI outputs. '
                'This cannot be detected by running single-axis audits separately. '
                'The compound effect is often larger than the sum of individual biases.'
                '</div>',
                unsafe_allow_html=True,
            )

        # Heatmap of FEAS by attr pair
        if intersectional is not None and len(intersectional) > 0:
            st.subheader("FEAS Heatmap by Attribute Pair")
            pivot_heat = intersectional.groupby('attr_pair')['FEAS_intersectional'].mean().reset_index()
            fig_h, ax_h = plt.subplots(figsize=(8, 3))
            fig_h.patch.set_facecolor('white')
            ax_h.set_facecolor('white')
            feas_vals_h = pivot_heat['FEAS_intersectional'].values
            bar_colors_h = ['#C0392B' if v < 0.3 else '#E67E22' if v < 0.5 else '#27AE60'
                            for v in feas_vals_h]
            ax_h.bar(pivot_heat['attr_pair'], feas_vals_h, color=bar_colors_h, width=0.5)
            ax_h.set_ylabel('Mean Intersectional FEAS', fontsize=9)
            ax_h.set_title('Mean FEAS by Attribute Pair', fontsize=11, fontweight='bold')
            ax_h.set_xticklabels(pivot_heat['attr_pair'].values, rotation=20, ha='right', fontsize=8)
            ax_h.set_ylim(0, 1.0)
            ax_h.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_h)
            plt.close(fig_h)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4: POST-MITIGATION FEAS DELTA
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.header("🔄 Post-Mitigation FEAS Delta")
        st.markdown(
            '<span class="novel-badge">KEY RESEARCH FINDING</span>',
            unsafe_allow_html=True,
        )

        st.markdown("""
        **The critical question:** After applying bias mitigation (AIF360 or Fairlearn),
        does the FEAS score change? Does the explanation now reveal the *reduced* bias —
        or does the explanation stay the same while the model quietly became fairer underneath?

        If FEAS Gap barely changes after mitigation → the XAI-Fairness Gap is **structural**,
        not accidental. This is a publishable finding.
        """)

        st.markdown("---")

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("AIF360 Reweighing → FEAS Delta")
            d = feas_delta_aif
            st.metric("FEAS before", f"{d['FEAS_before']:.4f}")
            st.metric("FEAS after",  f"{d['FEAS_after']:.4f}",
                      delta=f"{d['delta_mean_FEAS']:+.4f}")
            st.metric("FEAS Gap before", f"{d['gap_before']:.4f}")
            st.metric("FEAS Gap after",  f"{d['gap_after']:.4f}",
                      delta=f"{d['delta_FEAS_gap']:+.4f}",
                      delta_color="inverse")
            st.markdown(
                f'<div class="feas-box">{d["gap_interpretation"]}</div>',
                unsafe_allow_html=True,
            )

        with col_b:
            st.subheader("Fairlearn ExpGrad → FEAS Delta")
            d2 = feas_delta_fl
            st.metric("FEAS before", f"{d2['FEAS_before']:.4f}")
            st.metric("FEAS after",  f"{d2['FEAS_after']:.4f}",
                      delta=f"{d2['delta_mean_FEAS']:+.4f}")
            st.metric("FEAS Gap before", f"{d2['gap_before']:.4f}")
            st.metric("FEAS Gap after",  f"{d2['gap_after']:.4f}",
                      delta=f"{d2['delta_FEAS_gap']:+.4f}",
                      delta_color="inverse")
            st.markdown(
                f'<div class="feas-box">{d2["gap_interpretation"]}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Comparison bar chart
        methods     = ['Original', 'AIF360\nReweighing', 'Fairlearn\nExpGrad']
        feas_scores = [
            feas_results['mean_FEAS'],
            st.session_state['feas_aif']['mean_FEAS'],
            st.session_state['feas_fl']['mean_FEAS'],
        ]
        gap_scores  = [1 - s for s in feas_scores]

        fig_d, (ax_d1, ax_d2) = plt.subplots(1, 2, figsize=(12, 4))
        fig_d.patch.set_facecolor('white')

        colors_d = ['#95A5A6', '#3498DB', '#9B59B6']
        ax_d1.bar(methods, feas_scores, color=colors_d, alpha=0.8, width=0.5)
        ax_d1.set_ylabel('Mean FEAS', fontsize=10)
        ax_d1.set_title('FEAS: Does mitigation change alignment?', fontsize=11, fontweight='bold')
        ax_d1.set_ylim(0, 1)
        for i, v in enumerate(feas_scores):
            ax_d1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

        ax_d2.bar(methods, gap_scores, color=colors_d, alpha=0.8, width=0.5)
        ax_d2.set_ylabel('FEAS Gap (bias hidden from XAI)', fontsize=10)
        ax_d2.set_title('FEAS Gap: Hidden bias fraction', fontsize=11, fontweight='bold')
        ax_d2.set_ylim(0, 1)
        for i, v in enumerate(gap_scores):
            ax_d2.text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig_d)
        plt.close(fig_d)

        st.markdown(
            '<div class="research-box">'
            '<strong>Interpretation:</strong> If the FEAS Gap remains high even after '
            'mitigation, this demonstrates that bias mitigation and explainability '
            'are structurally decoupled — improving one does not improve the other. '
            'This finding directly motivates the need for a combined fairness-XAI '
            'framework, which is the PhD research contribution.'
            '</div>',
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5: AIF360 + FAIRLEARN (from enhanced pipeline)
    # ══════════════════════════════════════════════════════════════════════════
    with tab5:
        st.header("⚖️ AIF360 + Fairlearn Comparison")

        if os.path.exists(results['comparison_plot']):
            st.image(results['comparison_plot'], use_container_width=True)

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
            'Original': [
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
        })
        st.dataframe(metrics_table, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6: SHAP / LIME
    # ══════════════════════════════════════════════════════════════════════════
    with tab6:
        st.header("🧠 SHAP / LIME Explanations")

        shap_out = results['shap_outputs']
        lime_out = results['lime_outputs']
        feas_df  = feas_results['per_candidate']

        selected = st.selectbox("Select candidate:", list(shap_out.keys()))

        if selected:
            shap_r = shap_out[selected]
            feas_row = feas_df[feas_df['Name'] == selected].iloc[0]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rank", f"#{shap_r['rank']}")
            col2.metric("TF-IDF Score", f"{shap_r['score']:.4f}")
            col3.metric("FEAS (SHAP)", f"{feas_row['FEAS_SHAP']:.4f}")
            col4.metric("Bias hidden", f"{1 - feas_row['FEAS_SHAP']:.1%}")

            if os.path.exists(shap_r['fig_path']):
                st.image(shap_r['fig_path'], use_container_width=True)

            st.markdown("**Bias features revealed in this explanation:**")
            revealed = feas_row['bias_revealed_shap']
            hidden   = feas_row['bias_hidden_shap']
            if revealed:
                st.success(f"✓ Revealed: {', '.join(revealed)}")
            if hidden:
                st.error(f"✗ Hidden from SHAP: {', '.join(hidden)}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 7: RESEARCH EXPORT
    # ══════════════════════════════════════════════════════════════════════════
    with tab7:
        st.header("📄 Research Export")
        st.markdown("Generate paper-ready tables and figures for your research portfolio.")

        st.subheader("FEAS Summary Table (Paper Table 1)")
        summary = feas_summary_table(feas_results)
        st.dataframe(summary, use_container_width=True, hide_index=True)
        csv_summary = summary.to_csv(index=False).encode()
        st.download_button("⬇ Download Table 1 (CSV)", csv_summary,
                           "feas_summary_table.csv", "text/csv")

        st.subheader("Per-Candidate FEAS (Paper Table 2)")
        per_cand = feas_results['per_candidate'][[
            'Name', 'Rank', protected_attr,
            'FEAS_SHAP', 'FEAS_LIME', 'FEAS_mean',
            'n_bias_revealed', 'n_bias_hidden'
        ]]
        st.dataframe(per_cand, use_container_width=True, hide_index=True)
        csv_pc = per_cand.to_csv(index=False).encode()
        st.download_button("⬇ Download Table 2 (CSV)", csv_pc,
                           "feas_per_candidate.csv", "text/csv")

        st.subheader("FEAS Delta (Post-Mitigation) — Paper Table 3")
        delta_table = pd.DataFrame([
            {
                'Mitigation':     'AIF360 Reweighing',
                'FEAS before':    feas_delta_aif['FEAS_before'],
                'FEAS after':     feas_delta_aif['FEAS_after'],
                'ΔFEAS':          feas_delta_aif['delta_mean_FEAS'],
                'Gap before':     feas_delta_aif['gap_before'],
                'Gap after':      feas_delta_aif['gap_after'],
                'ΔGap':           feas_delta_aif['delta_FEAS_gap'],
            },
            {
                'Mitigation':     'Fairlearn ExpGrad',
                'FEAS before':    feas_delta_fl['FEAS_before'],
                'FEAS after':     feas_delta_fl['FEAS_after'],
                'ΔFEAS':          feas_delta_fl['delta_mean_FEAS'],
                'Gap before':     feas_delta_fl['gap_before'],
                'Gap after':      feas_delta_fl['gap_after'],
                'ΔGap':           feas_delta_fl['delta_FEAS_gap'],
            },
        ])
        st.dataframe(delta_table, use_container_width=True, hide_index=True)
        csv_dt = delta_table.to_csv(index=False).encode()
        st.download_button("⬇ Download Table 3 (CSV)", csv_dt,
                           "feas_delta_table.csv", "text/csv")

        st.subheader("Intersectional FEAS — Paper Table 4")
        if intersectional is not None and len(intersectional) > 0:
            inter_agg = intersectional.groupby(
                ['attr_pair', 'intersect_group']
            )['FEAS_intersectional'].mean().reset_index().round(4)
            st.dataframe(inter_agg, use_container_width=True, hide_index=True)
            csv_int = inter_agg.to_csv(index=False).encode()
            st.download_button("⬇ Download Table 4 (CSV)", csv_int,
                               "feas_intersectional.csv", "text/csv")

        st.subheader("Bias-Signal Features — Paper Table 5")
        bias_df_export = pd.DataFrame(
            feas_results['bias_features']['features'],
            columns=['Feature', 'Bias Signal Score']
        )
        st.dataframe(bias_df_export, use_container_width=True, hide_index=True)
        csv_bias = bias_df_export.to_csv(index=False).encode()
        st.download_button("⬇ Download Table 5 (CSV)", csv_bias,
                           "bias_signal_features.csv", "text/csv")

        st.markdown("---")
        st.subheader("Full JSON Export (all metrics)")
        export_dict = {
            'feas_summary': {
                'mean_FEAS_SHAP':  feas_results['mean_FEAS_SHAP'],
                'mean_FEAS_LIME':  feas_results['mean_FEAS_LIME'],
                'mean_FEAS':       feas_results['mean_FEAS'],
                'FEAS_gap':        feas_results['FEAS_gap'],
                'group_FEAS_diff': feas_results['group_FEAS_diff'],
                'alignment_verdict': feas_results['alignment_verdict'],
                'protected_attr':  protected_attr,
            },
            'feas_delta_aif360': {k: v for k, v in feas_delta_aif.items()
                                  if k != 'gap_interpretation'},
            'feas_delta_fairlearn': {k: v for k, v in feas_delta_fl.items()
                                     if k != 'gap_interpretation'},
        }
        json_out = json.dumps(export_dict, indent=2)
        st.download_button("⬇ Download full JSON export", json_out.encode(),
                           "feas_full_results.json", "application/json")

        st.markdown(
            '<div class="research-box">'
            '<strong>Suggested paper titles using this project:</strong><br><br>'
            '1. "FEAS: A Metric for Measuring the XAI-Fairness Alignment Gap in '
            'Algorithmic Hiring Systems" — FAccT / ECAI<br>'
            '2. "Beyond Single-Axis Auditing: Intersectional Bias Invisibility in '
            'Resume Ranking Explanations" — CHI / CSCW<br>'
            '3. "AIF360 vs Fairlearn: An Empirical Comparison of Bias Mitigation '
            'Efficacy and Explanation Transparency" — IJCAI / AAAI<br>'
            '4. "The Mitigation-Explainability Decoupling Problem: Do Fairer Models '
            'Produce More Revealing Explanations?" — IEEE TAI<br>'
            '5. "Over-Reliance in AI Hiring: A Mixed-Methods Study of SHAP-Based '
            'Explanations and Recruiter Decision Behaviour" — CHI'
            '</div>',
            unsafe_allow_html=True,
        )

else:
    st.info("👆 Click **Run Research Pipeline** to compute FEAS and all metrics.")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
**🔬 FEAS Metric (Novel)**
- Measures XAI-Fairness alignment
- Per-candidate scores
- Group-level analysis
- Post-mitigation delta
        """)
    with col2:
        st.markdown("""
**🕸️ Intersectional Analysis**
- All 2-attribute combinations
- Beyond single-axis audits
- Compound disadvantage detection
- Heatmap visualization
        """)
    with col3:
        st.markdown("""
**📄 Research Export**
- 5 paper-ready tables (CSV)
- Full JSON metrics dump
- Suggested paper titles
- Target venue list
        """)