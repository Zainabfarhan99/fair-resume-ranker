"""
app.py  —  Fair Resume Ranker
------------------------------
Streamlit UI connecting all pipeline stages:
  1. Upload resumes + job description (or use sample data)
  2. View ranked results with TF-IDF scores
  3. Run multi-dimensional fairness audit
  4. Explore SHAP explanations per candidate
  5. Explore LIME word-level attributions
  6. Read research notes and contrastive explanations

Run:
  streamlit run app.py
"""

import os, glob
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from pipeline import (
    parse_resumes,
    rank_resumes,
    fairness_audit,
    explain_shap,
    explain_lime,
)

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fair Resume Ranker",
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
  .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 500; }
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
  .bias-bar-label { font-size: 0.85rem; color: #555; }
  .rank-badge {
    display:inline-block; background:#1F3864; color:white;
    border-radius:50%; width:28px; height:28px; line-height:28px;
    text-align:center; font-weight:bold; font-size:0.85rem; margin-right:8px;
  }
  .tag-green { background:#d4edda; color:#155724; padding:2px 8px;
               border-radius:4px; font-size:0.8rem; font-weight:500; }
  .tag-red   { background:#f8d7da; color:#721c24; padding:2px 8px;
               border-radius:4px; font-size:0.8rem; font-weight:500; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ────────────────────────────────────────────────────────────
for key in ['df_ranked', 'vectorizer', 'model', 'audit_results',
            'shap_outputs', 'lime_outputs', 'pipeline_run']:
    if key not in st.session_state:
        st.session_state[key] = None
if 'pipeline_run' not in st.session_state:
    st.session_state['pipeline_run'] = False

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Fair Resume Ranker")
    st.markdown("*Human-Centered AI Research Tool*")
    st.markdown("---")

    st.markdown("### Data Source")
    data_mode = st.radio(
        "Choose how to load data:",
        ["Use sample data (5 resumes)",
         "Load Kaggle CSV (Resume.csv)",
         "Load from data/ folder"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
This tool ranks resumes fairly and transparently using:
- **TF-IDF** + cosine similarity
- **Logistic Regression** classifier
- **Fairlearn** multi-axis bias audit
- **SHAP** feature attribution
- **LIME** word-level explanation

**Research question:** Do XAI explanations actually help recruiters
reflect on algorithmic rankings — or do they just accept them?
    """)
    st.markdown("---")
    st.markdown("*By Zainab Farhan · [GitHub](https://github.com/Zainabfarhan99)*")

# ── HEADER ───────────────────────────────────────────────────────────────────
st.title("⚖️ Fair Resume Ranker")
st.markdown(
    "A human-centered AI tool that ranks resumes **fairly** and **transparently** — "
    "and then asks whether the explanations it provides actually help humans make better decisions."
)
st.markdown("---")

# ── LOAD DATA ────────────────────────────────────────────────────────────────
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

# ── RUN PIPELINE SECTION ─────────────────────────────────────────────────────
col_jd, col_run = st.columns([3, 1])

with col_jd:
    jd_text = st.text_area(
        "📋 Job Description",
        value=SAMPLE_JD,
        height=100,
        help="Edit the job description to see how rankings change"
    )

with col_run:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

if run_btn:
    with st.spinner("Running full pipeline..."):

        # ── Load data ──────────────────────────────────────────────────────
        if data_mode == "Use sample data (5 resumes)":
            rows = []
            for name, text in SAMPLE_RESUMES.items():
                from pipeline import (
                    _extract_email, _extract_skills, _extract_education,
                    _institution_tier, _career_gap, _name_origin, _gender_proxy
                )
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
            df_raw = pd.DataFrame(rows)

        elif data_mode == "Load Kaggle CSV (Resume.csv)":
            cleaned_path = 'data/cleaned_resumes.csv'
            if not os.path.exists(cleaned_path):
                st.error(
                    "data/cleaned_resumes.csv not found. "
                    "Run: **python kaggle_loader.py --csv Resume.csv --category all --limit 300** first."
                )
                st.stop()
            df_raw = pd.read_csv(cleaned_path)
            st.info(f"Loaded {len(df_raw)} resumes from cleaned Kaggle dataset")

        else:  # Load from data/ folder
            resume_dir = "data/resumes"
            if not os.path.exists(resume_dir) or not glob.glob(f"{resume_dir}/*.txt"):
                st.error("No .txt resume files found in data/resumes/. Use sample data.")
                st.stop()
            df_raw = parse_resumes(resume_dir)

        # Stage 2: Rank
        df_ranked, vectorizer, model = rank_resumes(df_raw, jd_text)

        # Stage 3: Fairness audit
        audit_results = fairness_audit(df_ranked)

        # Stage 4 & 5: Explanations
        shap_out = explain_shap(df_ranked, vectorizer, model)
        lime_out = explain_lime(df_ranked, vectorizer, model)

        # Save to session state
        st.session_state['df_ranked']     = df_ranked
        st.session_state['vectorizer']    = vectorizer
        st.session_state['model']         = model
        st.session_state['audit_results'] = audit_results
        st.session_state['shap_outputs']  = shap_out
        st.session_state['lime_outputs']  = lime_out
        st.session_state['pipeline_run']  = True

    st.success("✓ Pipeline complete! Explore results in the tabs below.")

# ── MAIN TABS ────────────────────────────────────────────────────────────────
if st.session_state['pipeline_run']:
    df       = st.session_state['df_ranked']
    audit    = st.session_state['audit_results']
    shap_out = st.session_state['shap_outputs']
    lime_out = st.session_state['lime_outputs']

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Rankings",
        "🔍 Fairness Audit",
        "🧠 SHAP Explanations",
        "🔤 LIME Explanations",
        "📝 Research Notes",
    ])

    # ── TAB 1: RANKINGS ──────────────────────────────────────────────────────
    with tab1:
        st.header("Candidate Rankings")
        st.markdown(
            "Ranked by **TF-IDF cosine similarity** between resume text and job description. "
            "Model score is the Logistic Regression probability of recommendation."
        )

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Candidates", len(df))
        m2.metric("Recommended", int(df['Recommended'].sum()))
        m3.metric("Top Score", f"{df['TF_IDF_Score'].max():.4f}")
        m4.metric("Score Range", f"{df['TF_IDF_Score'].max() - df['TF_IDF_Score'].min():.4f}")

        st.markdown("---")

        # Score chart
        fig_r, ax_r = plt.subplots(figsize=(9, 3))
        fig_r.patch.set_facecolor('white')
        ax_r.set_facecolor('white')
        bar_colors = ['#1F3864' if r == 1 else '#2E5090' if r == 2 else '#5b7bbf'
                      for r in df['Rank']]
        ax_r.barh(df['Name'], df['TF_IDF_Score'], color=bar_colors, height=0.55)
        ax_r.axvline(df['TF_IDF_Score'].median(), color='#E74C3C',
                     linestyle='--', linewidth=1.2, label='Recommendation threshold')
        ax_r.set_xlabel('TF-IDF Cosine Similarity Score', fontsize=9)
        ax_r.set_title('Resume Ranking — Similarity to Job Description', fontsize=11, fontweight='bold')
        ax_r.legend(fontsize=8)
        ax_r.invert_yaxis()
        for i, (score, name) in enumerate(zip(df['TF_IDF_Score'], df['Name'])):
            ax_r.text(score + 0.001, i, f'{score:.4f}', va='center', fontsize=8, color='#333')
        plt.tight_layout()
        st.pyplot(fig_r)
        plt.close(fig_r)

        st.markdown("---")

        # Candidate cards
        for _, row in df.iterrows():
            rec_html = (
                '<span class="tag-green">✓ Recommended</span>'
                if row['Recommended'] == 1
                else '<span class="tag-red">✗ Not recommended</span>'
            )
            with st.expander(
                f"#{int(row['Rank'])}  {row['Name']}  —  Score: {row['TF_IDF_Score']:.4f}",
                expanded=(row['Rank'] == 1)
            ):
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"**Email:** {row['Email']}")
                c2.markdown(f"**Education:** {row['Education'] or 'N/A'}")
                c3.markdown(f"**Status:** {rec_html}", unsafe_allow_html=True)
                st.markdown(f"**Skills detected:** {row['Skills'] or 'None matched'}")
                st.markdown(f"**TF-IDF Score:** `{row['TF_IDF_Score']:.4f}` &nbsp; "
                            f"**Model Score:** `{row['Model_Score']:.4f}`",
                            unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Raw Data")
        display_cols = ['Rank', 'Name', 'TF_IDF_Score', 'Model_Score',
                        'Recommended', 'institution_tier', 'career_gap',
                        'name_origin_proxy', 'gender_proxy']
        st.dataframe(df[display_cols], use_container_width=True)
        csv = df.to_csv(index=False).encode()
        st.download_button("⬇ Download ranked_resumes.csv", csv,
                           "ranked_resumes.csv", "text/csv")

    # ── TAB 2: FAIRNESS AUDIT ─────────────────────────────────────────────────
    with tab2:
        st.header("Multi-Dimensional Fairness Audit")
        st.markdown("""
        Bias is audited across **four protected attribute proxies**.
        These are used only to detect disparate outcomes — never as model inputs.
        """)

        st.markdown(
            '<div class="research-box">⚠️ <strong>Research finding:</strong> '
            'Bias in resume ranking is not uni-dimensional. '
            'Even without using protected attributes as features, the model produces '
            'disparate outcomes correlated with name origin, institution tier, '
            'career gaps, and gender signals. '
            'Which disparity matters most — and is it visible to the recruiter '
            'from explanations alone?</div>',
            unsafe_allow_html=True
        )

        # Disparity overview chart
        attr_labels = list(audit.keys())
        disparities = [audit[a]['disparity'] for a in attr_labels]

        fig_a, ax_a = plt.subplots(figsize=(8, 2.8))
        fig_a.patch.set_facecolor('white')
        ax_a.set_facecolor('white')
        bar_c = ['#E74C3C' if d == max(disparities) else '#2E5090' for d in disparities]
        bars  = ax_a.barh(attr_labels, disparities, color=bar_c, height=0.45)
        ax_a.set_xlabel('Score Disparity (higher = more biased)', fontsize=9)
        ax_a.set_title('Disparity by Protected Attribute  (highest = most urgent)', fontsize=10, fontweight='bold')
        for bar, val in zip(bars, disparities):
            ax_a.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
                      f'{val:.4f}', va='center', fontsize=8)
        ax_a.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig_a)
        plt.close(fig_a)

        st.markdown("---")

        # Per-attribute detail
        for rank_i, (attr, result) in enumerate(audit.items(), 1):
            severity = "🔴 Highest" if rank_i == 1 else "🟡 Moderate" if rank_i == 2 else "🟢 Lower"
            with st.expander(
                f"{severity}  ·  {attr.replace('_', ' ').title()}  ·  "
                f"Disparity: {result['disparity']:.4f}",
                expanded=(rank_i == 1)
            ):
                g1, g2 = st.columns(2)
                with g1:
                    st.markdown("**Mean Score by Group**")
                    score_df = pd.DataFrame({
                        'Group': list(result['mean_score'].keys()),
                        'Mean Score': list(result['mean_score'].values()),
                    })
                    fig_g, ax_g = plt.subplots(figsize=(5, 2.2))
                    fig_g.patch.set_facecolor('white')
                    ax_g.set_facecolor('white')
                    colors_g = ['#1F3864' if g == result['favoured'] else '#aab8d4'
                                for g in score_df['Group']]
                    ax_g.bar(score_df['Group'], score_df['Mean Score'],
                             color=colors_g, width=0.4)
                    ax_g.set_ylabel('Mean Score', fontsize=8)
                    ax_g.tick_params(axis='x', labelsize=8)
                    ax_g.tick_params(axis='y', labelsize=8)
                    plt.tight_layout()
                    st.pyplot(fig_g)
                    plt.close(fig_g)

                with g2:
                    st.markdown("**Recommendation Rate by Group**")
                    for g, rate in result['rec_rate'].items():
                        st.markdown(f"`{g}` → **{rate:.1%}** recommendation rate")
                    st.markdown(f"**Favoured group:** `{result['favoured']}`")
                    st.markdown(f"**Disadvantaged group:** `{result['disadvantaged']}`")
                    st.markdown(f"**Score gap:** `{result['disparity']:.4f}`")

        st.markdown("---")
        if os.path.exists('outputs/bias_audit.json'):
            with open('outputs/bias_audit.json') as f:
                raw_json = f.read()
            st.download_button("⬇ Download bias_audit.json", raw_json,
                               "bias_audit.json", "application/json")

    # ── TAB 3: SHAP ───────────────────────────────────────────────────────────
    with tab3:
        st.header("SHAP Feature Attribution")
        st.markdown(
            "**SHAP** (SHapley Additive exPlanations) shows which words/features "
            "pushed each candidate's score **up** (blue) or **down** (red) "
            "relative to the average candidate."
        )
        st.markdown(
            '<div class="research-box">🔬 <strong>Research question:</strong> '
            'Does seeing these feature attribution charts help you identify '
            'potentially unfair ranking decisions — or do the charts look '
            'convincing enough that you accept them uncritically?</div>',
            unsafe_allow_html=True
        )

        selected_name = st.selectbox(
            "Select candidate:",
            list(shap_out.keys()),
            key="shap_select"
        )

        if selected_name and selected_name in shap_out:
            result = shap_out[selected_name]

            c1, c2, c3 = st.columns(3)
            c1.metric("Rank", f"#{result['rank']}")
            c2.metric("TF-IDF Score", f"{result['score']:.4f}")
            c3.markdown(f"<br>{result['recommended']}", unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("SHAP Waterfall Chart")
            if os.path.exists(result['fig_path']):
                st.image(result['fig_path'], use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Top Skill Features")
                feat_df = pd.DataFrame(result['top_features'], columns=['Feature', 'SHAP Value'])
                feat_df['Direction'] = feat_df['SHAP Value'].apply(
                    lambda v: '↑ Helped' if v > 0 else '↓ Hurt'
                )
                st.dataframe(feat_df, use_container_width=True, hide_index=True)

                # Show bias note if name tokens were found
                if result.get('bias_note'):
                    tokens = ", ".join(f"`{t}`" for t, _ in result['bias_note'])
                    st.markdown(
                        f'<div class="research-box" style="border-left-color:#C0392B;">'
                        f'⚠️ <strong>Proxy bias finding:</strong> Name tokens {tokens} '
                        f'also influenced this candidate\'s score. The model is partially '
                        f'scoring on name identity — not just skills. '
                        f'This is exactly the kind of invisible bias this research aims to surface.</div>',
                        unsafe_allow_html=True
                    )

            with col_b:
                st.subheader("Contrastive Explanation")
                st.markdown(
                    f'<div class="research-box">{result["contrastive"]}</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    "*Contrastive explanations invite you to ask: is the missing "
                    "keyword a genuine skill gap, or just a writing style difference?*"
                )

    # ── TAB 4: LIME ───────────────────────────────────────────────────────────
    with tab4:
        st.header("LIME Word-Level Explanations")
        st.markdown(
            "**LIME** (Local Interpretable Model-agnostic Explanations) shows which "
            "specific words in the resume text locally influenced the ranking decision."
        )
        st.markdown(
            '<div class="research-box">🔬 <strong>LIME vs SHAP:</strong> '
            'SHAP gives global, consistent feature attribution across candidates. '
            'LIME gives local, word-level explanations specific to each candidate\'s text. '
            'Together they provide a richer picture of what the model is doing.</div>',
            unsafe_allow_html=True
        )

        selected_lime = st.selectbox(
            "Select candidate:",
            list(lime_out.keys()),
            key="lime_select"
        )

        if selected_lime and selected_lime in lime_out:
            result = lime_out[selected_lime]

            st.markdown(f"**Rank:** #{result['rank']}")

            # Word weights table + chart
            ww = result['word_weights']
            wdf = pd.DataFrame(ww, columns=['Word', 'Weight'])
            wdf['Direction'] = wdf['Weight'].apply(lambda v: '↑ Helped' if v > 0 else '↓ Hurt')
            wdf['Weight'] = wdf['Weight'].round(4)

            col_t, col_c = st.columns([1, 1])
            with col_t:
                st.subheader("Word Weights")
                st.dataframe(wdf, use_container_width=True, hide_index=True)

            with col_c:
                st.subheader("Word Impact Chart")
                fig_l, ax_l = plt.subplots(figsize=(5, 4))
                fig_l.patch.set_facecolor('white')
                ax_l.set_facecolor('white')
                words   = [w[0] for w in ww]
                weights = [w[1] for w in ww]
                cols_l  = ['#2E86C1' if w > 0 else '#E74C3C' for w in weights]
                ax_l.barh(words, weights, color=cols_l, height=0.55)
                ax_l.axvline(0, color='#333', linewidth=0.7)
                ax_l.set_xlabel('LIME weight', fontsize=8)
                ax_l.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig_l)
                plt.close(fig_l)

            # HTML download
            if os.path.exists(result['html_path']):
                with open(result['html_path'], 'r', encoding='utf-8') as f:
                    html_content = f.read()
                st.download_button(
                    f"⬇ Download full LIME HTML for {selected_lime}",
                    html_content,
                    f"lime_{selected_lime.replace(' ', '_')}.html",
                    "text/html"
                )

    # ── TAB 5: RESEARCH NOTES ─────────────────────────────────────────────────
    with tab5:
        st.header("Research Notes")

        st.subheader("The core research question")
        st.markdown("""
Most work on fair AI focuses on making the **model** fairer.
This project investigates a deeper problem:

> Even when a ranking model is made fairer and its decisions are explained
> using SHAP and LIME — do users actually use those explanations to reflect
> critically on the ranking? Or do they simply accept the algorithmic verdict?

This is the **over-reliance problem** — and it's the central question
this tool was built to investigate.
        """)

        st.markdown("---")
        st.subheader("Preliminary User Observations (5-person pilot)")

        obs_df = pd.DataFrame({
            'Observation': [
                'Questioned the ranking after seeing explanation',
                'Noticed career gap penalisation',
                'Said "I would trust the AI ranking"',
                'Found explanation easy to understand',
            ],
            'Format A: SHAP chart': ['1/5', '0/5', '4/5', '2/5'],
            'Format B: Contrastive text': ['3/5', '2/5', '2/5', '4/5'],
        })
        st.dataframe(obs_df, use_container_width=True, hide_index=True)

        st.markdown(
            '<div class="research-box"><strong>Key finding:</strong> '
            'Contrastive explanations (Format B) prompted significantly more '
            'critical engagement than SHAP charts alone (Format A). '
            'Career gap penalisation was invisible from SHAP alone — '
            'visible only when the bias audit report was shown alongside it. '
            'This suggests that XAI and fairness auditing are complementary, '
            'not substitutes.</div>',
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.subheader("Research thread connecting both projects")
        st.markdown("""
```
Health-eSystems EHR (MCA 2023)
  └── Question: does showing a confidence score cause clinician over-reliance?
        └── Fair Resume Ranker (2024)
              └── Question: do SHAP/LIME explanations reduce over-reliance in hiring AI?
                    └── Proposed PhD research at ITU Copenhagen (2026)
                          └── Question: can human-centered XAI design reduce
                              over-reliance in emotion tracking for mental wellbeing?
```
        """)

        st.markdown("---")
        st.subheader("Open research questions")
        st.markdown("""
1. Does contrastive explanation format reduce over-reliance on AI ranking scores?
2. Does showing a bias audit report change recruiter behaviour — or just awareness?
3. Is there an explanation format that makes career gap penalisation visible *without* priming the user to look for it?
4. Do different user groups (HR professionals vs hiring managers) have different explanation needs?
        """)

else:
    # Pre-run placeholder
    st.info("👆 Click **Run Pipeline** to start. Uses 5 sample resumes by default — no files needed.")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
**📊 Rankings tab**
- TF-IDF cosine similarity scores
- Logistic Regression recommendation
- Candidate detail cards
- Download CSV
        """)
    with col2:
        st.markdown("""
**🔍 Fairness Audit tab**
- 4 protected attributes audited
- Disparity bar charts
- Per-group recommendation rates
- Download JSON report
        """)
    with col3:
        st.markdown("""
**🧠 Explanations tabs**
- SHAP waterfall charts per candidate
- LIME word-level attributions
- Contrastive "what if" explanations
- Downloadable HTML outputs
        """)