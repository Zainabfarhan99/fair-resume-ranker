"""
pipeline.py
-----------
Core ML pipeline for Fair Resume Ranker.
Imported by app.py (Streamlit UI).

Stages:
  1. parse_resumes()     — extract text + protected attributes
  2. rank_resumes()      — TF-IDF + cosine similarity + Logistic Regression
  3. fairness_audit()    — MetricFrame across 4 protected attributes
  4. explain_shap()      — per-candidate SHAP feature attribution
  5. explain_lime()      — per-candidate LIME word-level explanation
"""

import os, re, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from fairlearn.metrics import MetricFrame
from lime.lime_text import LimeTextExplainer
import shap

# ── CONSTANTS ────────────────────────────────────────────────────────────────

SKILL_KEYWORDS = [
    'python', 'java', 'sql', 'react', 'aws', 'tensorflow', 'pytorch',
    'pandas', 'kubernetes', 'docker', 'machine learning', 'deep learning',
    'nlp', 'natural language processing', 'scikit-learn', 'data science',
    'rest api', 'node.js', 'vue.js', 'mongodb', 'statistics',
    'data analysis', 'tableau', 'excel', 'git', 'keras', 'bert',
    'transformers', 'spacy', 'nltk', 'tf-idf', 'logistic regression',
    'random forest', 'neural network', 'computer vision', 'microservices',
]

TIER1 = ['mit', 'stanford', 'oxford', 'cambridge', 'harvard',
         'imperial', 'eth zurich', 'iit', 'edinburgh', 'birmingham']

SOUTH_ASIAN_NAMES = ['sharma', 'patel', 'singh', 'kumar', 'gupta', 'rao',
                     'ali', 'khan', 'ahmed', 'hassan', 'farhan', 'al-']
WESTERN_NAMES     = ['johnson', 'mitchell', 'smith', 'jones', 'williams',
                     'brown', 'taylor', 'wilson', 'anderson', 'thomas']

# ── STAGE 1: PARSE ───────────────────────────────────────────────────────────

def _extract_email(text):
    m = re.search(r'[\w.\-+]+@[\w.\-]+\.\w+', text)
    return m.group() if m else ""

def _extract_name(text):
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[0] if lines else "Unknown"

def _extract_skills(text):
    tl = text.lower()
    return ", ".join(s.title() for s in SKILL_KEYWORDS if s in tl)

def _extract_education(text):
    for line in text.split('\n'):
        for kw in ['B.Tech','M.Tech','MBA','B.Sc','M.Sc','Bachelor',
                   'Master','PhD','MCA','BSc','MSc','B.E']:
            if kw.lower() in line.lower():
                return line.strip()
    return ""

def _institution_tier(text):
    tl = text.lower()
    return "Tier 1" if any(t in tl for t in TIER1) else "Tier 2/3"

def _career_gap(text):
    tl = text.lower()
    patterns = [r'career.break', r'career.gap', r'caregiving',
                r'maternity', r'parental.leave', r'took.time.off']
    return "Yes" if any(re.search(p, tl) for p in patterns) else "No"

def _name_origin(name):
    nl = name.lower()
    if any(n in nl for n in SOUTH_ASIAN_NAMES): return "South Asian proxy"
    if any(n in nl for n in WESTERN_NAMES):     return "Western proxy"
    return "Other"

def _gender_proxy(text):
    tl = text.lower()
    if re.search(r'she/her|maternity|caregiving', tl): return "Female proxy"
    if re.search(r'he/him', tl):                       return "Male proxy"
    return "Unspecified"

def parse_resumes(resume_dir):
    """Parse all .txt resumes in resume_dir. Returns DataFrame."""
    rows = []
    for fname in sorted(os.listdir(resume_dir)):
        if not fname.endswith('.txt'):
            continue
        with open(os.path.join(resume_dir, fname), 'r', encoding='utf-8') as f:
            text = f.read()
        name = _extract_name(text)
        rows.append({
            'Filename':          fname,
            'Name':              name,
            'Email':             _extract_email(text),
            'Skills':            _extract_skills(text),
            'Education':         _extract_education(text),
            'Full_Text':         text.strip(),
            # protected attributes (audit only — never used as model features)
            'gender_proxy':      _gender_proxy(text),
            'institution_tier':  _institution_tier(text),
            'career_gap':        _career_gap(text),
            'name_origin_proxy': _name_origin(name),
        })
    return pd.DataFrame(rows)

# ── STAGE 2: RANK ────────────────────────────────────────────────────────────

def rank_resumes(df, jd_text):
    """
    TF-IDF vectorise resumes + JD, compute cosine similarity,
    train Logistic Regression on similarity features.
    Returns enriched DataFrame + saved model artifacts.
    """
    corpus = list(df['Full_Text'].fillna('')) + [jd_text]

    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=500,
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    resume_vecs  = tfidf_matrix[:-1]
    jd_vec       = tfidf_matrix[-1:]

    scores = cosine_similarity(resume_vecs, jd_vec).flatten()
    df = df.copy()
    df['TF_IDF_Score'] = np.round(scores, 4)

    # Binary label: above median = recommended
    threshold       = df['TF_IDF_Score'].median()
    df['Label']     = (df['TF_IDF_Score'] >= threshold).astype(int)

    X = resume_vecs.toarray()
    y = df['Label'].values

    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X, y)

    df['Model_Score']  = np.round(model.predict_proba(X)[:, 1], 4)
    df['Recommended']  = model.predict(X)

    df = df.sort_values('TF_IDF_Score', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1

    os.makedirs('models', exist_ok=True)
    with open('models/vectorizer.pkl', 'wb') as f: pickle.dump(vectorizer, f)
    with open('models/model.pkl',      'wb') as f: pickle.dump(model, f)

    return df, vectorizer, model

# ── STAGE 3: FAIRNESS AUDIT ──────────────────────────────────────────────────

def fairness_audit(df):
    """
    Run MetricFrame across 4 protected attribute proxies.
    Returns dict of results per attribute, sorted by disparity.
    """
    ATTRS = ['gender_proxy', 'institution_tier', 'career_gap', 'name_origin_proxy']
    results = {}

    for attr in ATTRS:
        sensitive = df[attr].astype(str)
        groups    = sensitive.unique()
        if len(groups) < 2:
            continue

        scores_by_group = {
            g: float(np.round(df.loc[sensitive == g, 'TF_IDF_Score'].mean(), 4))
            for g in groups
        }

        rec_by_group = {
            g: float(np.round(df.loc[sensitive == g, 'Recommended'].mean(), 3))
            for g in groups
        }

        disparity = round(
            max(scores_by_group.values()) - min(scores_by_group.values()), 4
        )

        results[attr] = {
            'mean_score':      scores_by_group,
            'rec_rate':        rec_by_group,
            'disparity':       disparity,
            'favoured':        max(scores_by_group, key=scores_by_group.get),
            'disadvantaged':   min(scores_by_group, key=scores_by_group.get),
        }

    # Sort by disparity descending
    results = dict(sorted(results.items(), key=lambda x: x[1]['disparity'], reverse=True))

    os.makedirs('outputs', exist_ok=True)
    with open('outputs/bias_audit.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results

# ── STAGE 4: SHAP ────────────────────────────────────────────────────────────

# ── FIX: Build a global set of ALL person-name tokens from ALL candidates ──
# This is called once inside explain_shap() after df is available.

def _build_all_name_tokens(df):
    """
    Collect every word (and common bigrams) from all candidate names in df.
    Returns a set of lowercase strings to filter from SHAP feature lists.
    """
    name_tokens = set()
    for name in df['Name']:
        parts = name.lower().split()
        for p in parts:
            name_tokens.add(p)
        # also add bigrams from the name (e.g. "james mitchell", "arjun patel")
        for i in range(len(parts) - 1):
            name_tokens.add(f"{parts[i]} {parts[i+1]}")
    return name_tokens


def _build_noise_pattern():
    """Compile the static noise regex (years, emails, generic section headers)."""
    return re.compile(
        r'^(19|20)\d{2}$'                    # years: 2020, 2019 …
        r'|^[\w.+-]+@[\w.-]+$'               # email addresses
        r'|@'                                # anything with @
        r'|\bemail\b|\bphone\b|\baddress\b'  # contact labels
        r'|\bexperience\b|\beducation\b|\bskills\b'  # section headers
        r'|\byears\b|\bwork\b|\bteam\b|\busing\b'
    )


def _is_noise_token(token, all_name_tokens, noise_pattern):
    """
    Return True if token is:
      - a name token from ANY candidate (not just the current one)
      - a year, email fragment, or generic section-header word
    """
    token_lower = token.lower()

    # Check against static noise pattern
    if noise_pattern.search(token_lower):
        return True

    # Check every word in the token against all candidate name tokens
    # This catches both unigrams ("james", "patel") and bigrams ("arjun patel")
    token_words = set(token_lower.split())
    if token_words & all_name_tokens:
        return True

    return False


def _get_clean_shap_features(sv_i, feature_names, all_name_tokens, noise_pattern, n=8):
    """
    Return top-n SHAP features filtered to meaningful skill/domain terms.
    Also returns separately the name-based features as a bias note.

    Parameters
    ----------
    sv_i            : 1-D array of SHAP values for this candidate
    feature_names   : list of feature name strings from vectorizer
    all_name_tokens : set of ALL candidate name words (built once per run)
    noise_pattern   : compiled regex for year/email/generic noise
    n               : how many clean features to return
    """
    clean_idx = []
    noise_idx = []

    for j, f in enumerate(feature_names):
        if _is_noise_token(f, all_name_tokens, noise_pattern):
            if abs(sv_i[j]) > 1e-6:
                noise_idx.append(j)
        else:
            clean_idx.append(j)

    # From clean features, pick top-n by absolute SHAP value
    clean_by_abs = sorted(clean_idx, key=lambda j: abs(sv_i[j]), reverse=True)
    top_clean    = clean_by_abs[:n]

    # Split into positive (helped) and negative (hurt)
    pos = sorted([j for j in top_clean if sv_i[j] > 0],  key=lambda j: sv_i[j], reverse=True)
    neg = sorted([j for j in top_clean if sv_i[j] <= 0], key=lambda j: sv_i[j])

    # Interleave: negatives at bottom, positives at top — classic waterfall order
    ordered = neg + pos

    # Top name-based bias tokens (for annotation, flagged as fairness concern)
    noise_top = sorted(noise_idx, key=lambda j: abs(sv_i[j]), reverse=True)[:3]
    bias_note = [(feature_names[j], round(float(sv_i[j]), 5)) for j in noise_top]

    return ordered, bias_note


def explain_shap(df, vectorizer, model):
    """
    Generate clean SHAP waterfall charts per candidate.

    KEY FIX: Filters name tokens from ALL candidates (not just the current one),
    so charts show only meaningful skill/domain signals.
    Name-based tokens are flagged separately as a proxy bias finding.

    Returns dict: name → result dict
    """
    X             = vectorizer.transform(df['Full_Text'].fillna('')).toarray()
    feature_names = vectorizer.get_feature_names_out()

    explainer   = shap.LinearExplainer(model, X, feature_perturbation='interventional')
    shap_values = explainer.shap_values(X)
    sv = shap_values if shap_values.ndim == 2 else shap_values[:, :, 1]

    # ── Build global name token set (THE FIX) ──────────────────────────────
    all_name_tokens = _build_all_name_tokens(df)
    noise_pattern   = _build_noise_pattern()

    # Colour palette
    C_POS  = '#1F6FBF'   # strong blue  — skill helped
    C_NEG  = '#C0392B'   # strong red   — skill hurt
    C_ZERO = '#AAAAAA'   # grey         — negligible

    outputs = {}
    for i, row in df.iterrows():
        name  = row['Name']
        rank  = int(row['Rank'])
        score = row['TF_IDF_Score']
        rec   = '✓ Recommended' if row['Recommended'] == 1 else '✗ Not recommended'
        rec_color = '#1a7a3a' if row['Recommended'] == 1 else '#a02020'

        sv_i = sv[i]

        # Get clean features and bias note — now uses ALL name tokens
        ordered_idx, bias_note = _get_clean_shap_features(
            sv_i, feature_names, all_name_tokens, noise_pattern, n=10
        )

        if not ordered_idx:
            # Fallback: just use top-10 by abs value without filtering
            ordered_idx = list(np.argsort(np.abs(sv_i))[-10:])

        feats  = [feature_names[j] for j in ordered_idx]
        vals   = [sv_i[j] for j in ordered_idx]
        colors = [C_POS if v > 0 else C_NEG if v < -1e-6 else C_ZERO for v in vals]

        # ── Build clean chart ─────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#FAFAFA')

        y_pos = range(len(feats))
        bars  = ax.barh(y_pos, vals, color=colors, height=0.6,
                        edgecolor='white', linewidth=0.5)

        # Value labels on bars
        for bar, val in zip(bars, vals):
            if abs(val) > 1e-6:
                x_pos = val + (0.00008 if val > 0 else -0.00008)
                ha    = 'left' if val > 0 else 'right'
                ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                        f'{val:+.4f}', va='center', ha=ha,
                        fontsize=7.5, color='#333')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(feats, fontsize=10)
        ax.axvline(0, color='#444', linewidth=0.9, zorder=3)

        # Grid lines (horizontal only, subtle)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color='#E0E0E0', linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)

        ax.set_xlabel('SHAP value  (contribution to recommendation score)',
                      fontsize=9, color='#555')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCC')
        ax.spines['bottom'].set_color('#CCC')

        # Title block
        fig.text(0.13, 0.97,
                 f'Rank #{rank}  ·  {name}',
                 fontsize=13, fontweight='bold', color='#1F3864', va='top')
        fig.text(0.13, 0.92,
                 f'TF-IDF Score: {score:.4f}   |   ',
                 fontsize=10, color='#555', va='top')
        fig.text(0.40, 0.92,
                 rec,
                 fontsize=10, color=rec_color, fontweight='bold', va='top')

        # Legend
        blue_p = mpatches.Patch(color=C_POS, label='Skill / term helped score ↑')
        red_p  = mpatches.Patch(color=C_NEG, label='Missing / irrelevant term ↓')
        ax.legend(handles=[blue_p, red_p], fontsize=8.5,
                  loc='lower right', framealpha=0.9,
                  edgecolor='#CCC', facecolor='white')

        # Bias note at bottom if name tokens were found
        if bias_note:
            note_str = (
                f"⚠ Proxy bias detected: name tokens also influenced this score — "
                f"{', '.join(t for t, _ in bias_note[:2])}. "
                f"This is a fairness concern, not a skill signal."
            )
            fig.text(0.01, 0.01, note_str, fontsize=7.5,
                     color='#8B0000', style='italic', va='bottom',
                     wrap=True)

        plt.tight_layout(rect=[0, 0.05, 1, 0.90])

        os.makedirs('shap_outputs', exist_ok=True)
        safe     = name.replace(' ', '_')
        fig_path = f'shap_outputs/shap_{safe}_rank{rank}.png'
        fig.savefig(fig_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

        # Top skill features for table display (clean, no noise)
        top_abs_clean = sorted(ordered_idx, key=lambda j: abs(sv_i[j]), reverse=True)[:6]
        top_feats     = [(feature_names[j], round(float(sv_i[j]), 4))
                        for j in top_abs_clean]

        # Contrastive explanation using only clean features
        if rank > 1:
            sv_top    = sv[df[df['Rank'] == 1].index[0]]
            gap       = sv_top - sv_i
            # Only from clean indices
            clean_all = [j for j, f in enumerate(feature_names)
                         if not _is_noise_token(f, all_name_tokens, noise_pattern)]
            gap_clean = [(j, gap[j]) for j in clean_all if gap[j] > 0]
            gap_clean.sort(key=lambda x: x[1], reverse=True)
            imp_feats = [feature_names[j] for j, _ in gap_clean[:3]]

            if imp_feats:
                contrastive = (
                    f"To improve {name}'s ranking, stronger presence of "
                    f"**{imp_feats[0]}** would help most"
                    + (f", followed by **{imp_feats[1]}**" if len(imp_feats) > 1 else "")
                    + (f" and **{imp_feats[2]}**" if len(imp_feats) > 2 else "")
                    + f".\n\n"
                    f"*Research question: Is the absence of "
                    f"'{imp_feats[0]}' a genuine skill gap — or an artefact "
                    f"of how {name} described their experience?*"
                )
            else:
                contrastive = f"{name} scored lower due to a different technical focus area."
        else:
            best_pos = [f for f, v in top_feats if v > 0]
            contrastive = (
                f"{name} is the top-ranked candidate. "
                f"Key skill signals: "
                + (", ".join(f"**{f}**" for f in best_pos[:3]) if best_pos
                   else "strong overall keyword alignment with the job description")
                + "."
            )

        outputs[name] = {
            'fig_path':     fig_path,
            'top_features': top_feats,
            'bias_note':    bias_note,
            'contrastive':  contrastive,
            'rank':         rank,
            'score':        score,
            'recommended':  rec,
        }
        plt.close(fig)

    return outputs

# ── STAGE 5: LIME ────────────────────────────────────────────────────────────

def explain_lime(df, vectorizer, model):
    """
    Generate LIME word-level explanations for each candidate.
    Returns dict: name → list of (word, weight) tuples
    """
    def predict_fn(texts):
        return model.predict_proba(vectorizer.transform(texts))

    lime_exp = LimeTextExplainer(
        class_names=['Not Recommended', 'Recommended'],
        split_expression=r'\W+',
        bow=True
    )

    outputs = {}
    os.makedirs('lime_outputs', exist_ok=True)

    for i, row in df.iterrows():
        name = row['Name']
        text = str(row['Full_Text'])
        rank = row['Rank']

        exp        = lime_exp.explain_instance(
            text, predict_fn,
            num_features=10, num_samples=100, labels=[1]
        )
        word_weights = exp.as_list(label=1)

        # Save annotated HTML
        html      = exp.as_html()
        note      = f"""
        <div style="font-family:Arial,sans-serif;padding:14px 18px;
                    background:#EBF0F8;border-left:4px solid #2E5090;
                    margin:14px 0;font-size:13px;color:#333;line-height:1.6">
          <strong>Candidate:</strong> {name} &nbsp;·&nbsp;
          <strong>Rank #{rank}</strong> &nbsp;·&nbsp;
          <strong>Score: {row['TF_IDF_Score']:.4f}</strong><br><br>
          <em>Words highlighted in green pushed the score UP.
          Words in red pushed it DOWN.</em><br>
          <em>Research question: Does seeing word-level explanations help you
          identify whether this ranking is fair — or does it just look convincing?</em>
        </div>"""
        html = html.replace('</body>', note + '</body>')

        safe     = name.replace(' ', '_')
        out_path = f'lime_outputs/lime_{safe}_rank{rank}.html'
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(html)

        outputs[name] = {
            'word_weights': word_weights,
            'html_path':    out_path,
            'rank':         rank,
        }

    return outputs