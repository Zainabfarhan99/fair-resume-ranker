"""
feas_metric.py
--------------
Fairness-Explainability Alignment Score (FEAS)
===============================================

NOVEL RESEARCH CONTRIBUTION — Zainab Farhan (2024/2025)

Definition
----------
FEAS quantifies how much of the bias detected by a fairness audit
is actually *visible* in the explanations produced by SHAP or LIME.

    FEAS(candidate) = |B ∩ E| / |B|

Where:
    B = set of "bias-signal features" — features most correlated with
        the protected attribute, identified by the fairness audit
    E = set of top-k features highlighted by the explainability tool
        (SHAP waterfall or LIME word weights)

Interpretation
--------------
    FEAS = 1.0  →  Explainability tool is FULLY revealing bias-driving features
    FEAS = 0.0  →  Explainability tool is COMPLETELY hiding bias
    FEAS = 0.5  →  Half of the bias is visible in explanations

This is the XAI-Fairness Alignment Gap: the fraction of bias that
is invisible to a user relying solely on explanations.

Aggregate metrics
-----------------
    mean_FEAS      : average alignment across all candidates
    FEAS_gap       : 1 - mean_FEAS (the "hidden bias fraction")
    group_FEAS_diff: difference in mean FEAS between privileged and
                     unprivileged groups — a second-order fairness concern:
                     do explanations hide MORE bias for some groups?

Research significance
---------------------
This metric bridges two literatures that currently operate in silos:
    - Algorithmic fairness (AIF360, Fairlearn)
    - Explainability (SHAP, LIME)

FEAS lets us ask: *even after making a model fairer and explaining it,
does the explanation actually reflect the fairness improvement?*
"""

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: IDENTIFY BIAS-SIGNAL FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def compute_bias_signal_features(
    X: np.ndarray,
    protected_attr_values: np.ndarray,
    feature_names: list,
    method: str = 'mutual_info',
    top_k: int = 20,
) -> dict:
    """
    Identify features that are most correlated with a protected attribute.

    These are the "bias-signal features" (set B in the FEAS formula).
    If the model uses these features, it may be making decisions
    partially based on the protected attribute — even without using it directly.

    Parameters
    ----------
    X                  : TF-IDF feature matrix (n_samples × n_features)
    protected_attr_values : array of protected attribute values per candidate
    feature_names      : list of feature names from TfidfVectorizer
    method             : 'mutual_info' | 'pointbiserial' | 'combined'
    top_k              : number of top bias-signal features to return

    Returns
    -------
    dict with:
        'features'       : list of (feature_name, score) tuples, ranked
        'feature_set'    : set of feature names (for intersection computation)
        'scores_df'      : full DataFrame with all features and their bias scores
        'method'         : method used
    """
    le = LabelEncoder()
    protected_numeric = le.fit_transform(protected_attr_values.astype(str))
    n_groups = len(np.unique(protected_numeric))

    scores = np.zeros(len(feature_names))

    if method in ('mutual_info', 'combined'):
        # Mutual information: how much does knowing the feature reduce
        # uncertainty about the protected attribute?
        mi_scores = mutual_info_classif(
            X, protected_numeric,
            discrete_features=False,
            random_state=42,
        )
        # Normalise to [0, 1]
        if mi_scores.max() > 0:
            mi_scores = mi_scores / mi_scores.max()
        scores += mi_scores

    if method in ('pointbiserial', 'combined') and n_groups == 2:
        # Point-biserial correlation for binary protected attributes
        pb_scores = np.zeros(len(feature_names))
        for j in range(X.shape[1]):
            col = X[:, j]
            if col.std() > 0:
                corr, _ = pointbiserialr(protected_numeric, col)
                pb_scores[j] = abs(corr)
        if method == 'combined':
            if pb_scores.max() > 0:
                pb_scores = pb_scores / pb_scores.max()
            scores = (scores + pb_scores) / 2
        else:
            scores = pb_scores

    # If method is 'combined' and we had mutual_info part above, scores are sum.
    # For 'combined' with n_groups > 2, pointbiserial doesn't apply,
    # so scores remain normalised mutual_info.
    if method == 'combined' and n_groups > 2:
        scores = scores / 2  # re-normalise to [0,1] range

    # Build ranked list
    ranked_idx = np.argsort(scores)[::-1]
    top_features = [(feature_names[i], round(float(scores[i]), 5))
                    for i in ranked_idx[:top_k]]
    feature_set = {f for f, _ in top_features}

    scores_df = pd.DataFrame({
        'feature': feature_names,
        'bias_signal_score': scores,
    }).sort_values('bias_signal_score', ascending=False).reset_index(drop=True)

    return {
        'features':    top_features,
        'feature_set': feature_set,
        'scores_df':   scores_df,
        'method':      method,
        'top_k':       top_k,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: EXTRACT EXPLAINABILITY FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def extract_shap_features(shap_values_i: np.ndarray,
                           feature_names: list,
                           top_k: int = 10,
                           absolute: bool = True) -> set:
    """
    Extract the top-k features from a single candidate's SHAP values.

    Parameters
    ----------
    shap_values_i : 1-D array of SHAP values for this candidate
    feature_names : list of feature name strings
    top_k         : how many top features to include in E
    absolute      : if True, rank by |SHAP value| (direction-agnostic)

    Returns
    -------
    set of feature names in the top-k SHAP explanation
    """
    vals = np.abs(shap_values_i) if absolute else shap_values_i
    top_idx = np.argsort(vals)[::-1][:top_k]
    return {feature_names[i] for i in top_idx}


def extract_lime_features(lime_word_weights: list, top_k: int = 10) -> set:
    """
    Extract the top-k features from a LIME explanation.

    Parameters
    ----------
    lime_word_weights : list of (word, weight) tuples from LIME
    top_k             : how many top features to include in E

    Returns
    -------
    set of feature names in the top-k LIME explanation
    """
    sorted_weights = sorted(lime_word_weights,
                            key=lambda x: abs(x[1]),
                            reverse=True)
    return {w for w, _ in sorted_weights[:top_k]}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: COMPUTE FEAS
# ══════════════════════════════════════════════════════════════════════════════

def compute_feas(
    bias_features: set,
    explanation_features: set,
) -> float:
    """
    Compute the Fairness-Explainability Alignment Score for one candidate.

    FEAS = |B ∩ E| / |B|

    Parameters
    ----------
    bias_features        : set B — features driving bias (from fairness audit)
    explanation_features : set E — features shown by SHAP or LIME

    Returns
    -------
    float in [0, 1]
        0.0 = explanation hides all bias-driving features
        1.0 = explanation reveals all bias-driving features
    """
    if not bias_features:
        return np.nan  # undefined if no bias features detected

    intersection = bias_features & explanation_features
    return len(intersection) / len(bias_features)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: RUN FULL FEAS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_feas_analysis(
    df: pd.DataFrame,
    X: np.ndarray,
    shap_values: np.ndarray,
    lime_outputs: dict,
    feature_names: list,
    protected_attr: str,
    xai_top_k: int = 10,
    bias_top_k: int = 20,
    bias_method: str = 'combined',
) -> dict:
    """
    Run the complete FEAS analysis pipeline.

    Parameters
    ----------
    df             : ranked candidates DataFrame
    X              : TF-IDF feature matrix
    shap_values    : SHAP values array (n_candidates × n_features)
    lime_outputs   : dict from pipeline.explain_lime()
    feature_names  : list of feature names
    protected_attr : column name of protected attribute to audit
    xai_top_k      : how many XAI features to consider (set E size)
    bias_top_k     : how many bias features to identify (set B size)
    bias_method    : method for bias feature identification

    Returns
    -------
    Complete FEAS results dict with per-candidate scores and aggregate stats
    """
    # Identify bias-signal features (set B)
    bias_result = compute_bias_signal_features(
        X,
        df[protected_attr].values,
        feature_names,
        method=bias_method,
        top_k=bias_top_k,
    )
    B = bias_result['feature_set']

    # Per-candidate FEAS scores
    candidate_results = []
    sv = shap_values if shap_values.ndim == 2 else shap_values[:, :, 1]

    for i, row in df.iterrows():
        name = row['Name']

        # SHAP-FEAS
        E_shap = extract_shap_features(sv[i], feature_names, top_k=xai_top_k)
        feas_shap = compute_feas(B, E_shap)

        # LIME-FEAS
        if name in lime_outputs:
            E_lime = extract_lime_features(
                lime_outputs[name]['word_weights'], top_k=xai_top_k
            )
            feas_lime = compute_feas(B, E_lime)
        else:
            feas_lime = np.nan

        # Which bias features are revealed vs hidden in SHAP?
        revealed_shap = B & E_shap
        hidden_shap   = B - E_shap

        candidate_results.append({
            'Name':             name,
            'Rank':             row['Rank'],
            'Recommended':      int(row['Recommended']),
            protected_attr:     row[protected_attr],
            'FEAS_SHAP':        round(feas_shap, 4) if not np.isnan(feas_shap) else None,
            'FEAS_LIME':        round(feas_lime, 4) if not np.isnan(feas_lime) else None,
            'FEAS_mean':        round(np.nanmean([feas_shap, feas_lime]), 4)
                                if not (np.isnan(feas_shap) and np.isnan(feas_lime))
                                else None,
            'bias_revealed_shap': sorted(revealed_shap),
            'bias_hidden_shap':   sorted(hidden_shap),
            'n_bias_revealed':    len(revealed_shap),
            'n_bias_hidden':      len(hidden_shap),
        })

    feas_df = pd.DataFrame(candidate_results)

    # ── Aggregate statistics ──────────────────────────────────────────────
    mean_feas_shap = float(feas_df['FEAS_SHAP'].dropna().mean())
    mean_feas_lime = float(feas_df['FEAS_LIME'].dropna().mean())
    mean_feas      = float(feas_df['FEAS_mean'].dropna().mean())
    feas_gap       = round(1.0 - mean_feas, 4)  # hidden bias fraction

    # Group-level FEAS — do explanations hide more bias for some groups?
    group_feas = (
        feas_df.groupby(protected_attr)['FEAS_SHAP']
        .agg(['mean', 'std', 'count'])
        .rename(columns={'mean': 'mean_FEAS_SHAP', 'std': 'std_FEAS_SHAP',
                         'count': 'n_candidates'})
        .reset_index()
    )
    group_feas['mean_FEAS_SHAP'] = group_feas['mean_FEAS_SHAP'].round(4)

    if len(group_feas) >= 2:
        group_feas_sorted = group_feas.sort_values('mean_FEAS_SHAP')
        group_feas_diff = round(
            float(group_feas_sorted['mean_FEAS_SHAP'].iloc[-1]) -
            float(group_feas_sorted['mean_FEAS_SHAP'].iloc[0]),
            4
        )
    else:
        group_feas_diff = 0.0

    # Mitigation FEAS delta (placeholder — computed if mitigated models provided)
    # post-mitigation FEAS can be compared to show if mitigation improves alignment

    # ── Severity classification ───────────────────────────────────────────
    if mean_feas >= 0.7:
        alignment_verdict = 'High alignment — explanations largely reveal bias'
    elif mean_feas >= 0.4:
        alignment_verdict = 'Moderate alignment — partial bias visibility'
    else:
        alignment_verdict = 'Low alignment — critical XAI-Fairness gap detected'

    return {
        'bias_features':        bias_result,
        'per_candidate':        feas_df,
        'mean_FEAS_SHAP':       round(mean_feas_shap, 4),
        'mean_FEAS_LIME':       round(mean_feas_lime, 4),
        'mean_FEAS':            round(mean_feas, 4),
        'FEAS_gap':             feas_gap,
        'group_FEAS':           group_feas,
        'group_FEAS_diff':      group_feas_diff,
        'alignment_verdict':    alignment_verdict,
        'protected_attr':       protected_attr,
        'bias_method':          bias_method,
        'xai_top_k':            xai_top_k,
        'bias_top_k':           bias_top_k,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: POST-MITIGATION FEAS DELTA
# ══════════════════════════════════════════════════════════════════════════════

def compute_feas_delta(
    feas_before: dict,
    feas_after: dict,
    mitigation_name: str = 'Reweighing',
) -> dict:
    """
    Compute the change in FEAS after bias mitigation.

    This answers: does applying AIF360 Reweighing or Fairlearn ExpGrad
    improve the alignment between explanations and fairness metrics?

    In theory:
    - After mitigation, the model should rely LESS on bias-signal features
    - So SHAP should attribute LESS weight to them
    - Therefore FEAS should DECREASE after mitigation (good: less bias to expose)

    But the explanation format hasn't changed — so if FEAS stays the same
    or increases, it means the mitigation reduced model bias WITHOUT
    being reflected in the explanations → a critical gap.

    Parameters
    ----------
    feas_before     : result from run_feas_analysis() on original model
    feas_after      : result from run_feas_analysis() on mitigated model
    mitigation_name : label for the mitigation method

    Returns
    -------
    dict with FEAS delta metrics and interpretation
    """
    delta_mean      = round(feas_after['mean_FEAS'] - feas_before['mean_FEAS'], 4)
    delta_shap      = round(feas_after['mean_FEAS_SHAP'] - feas_before['mean_FEAS_SHAP'], 4)
    delta_gap       = round(feas_after['FEAS_gap'] - feas_before['FEAS_gap'], 4)
    delta_group_diff = round(
        feas_after['group_FEAS_diff'] - feas_before['group_FEAS_diff'], 4
    )

    # Interpret the delta
    # After mitigation, bias features in the model should reduce.
    # If FEAS_gap barely changes, explanations aren't reflecting the fairness fix.
    if abs(delta_gap) < 0.05:
        gap_interpretation = (
            f"XAI-Fairness Alignment Gap is nearly unchanged after {mitigation_name}. "
            f"The model is fairer, but the explanations don't reflect this improvement. "
            f"This is the core finding: mitigation and explainability are decoupled."
        )
    elif delta_gap < -0.05:
        gap_interpretation = (
            f"FEAS gap decreased by {abs(delta_gap):.3f} after {mitigation_name}. "
            f"Explanations now better reflect the reduced bias. Partial alignment improvement."
        )
    else:
        gap_interpretation = (
            f"FEAS gap increased by {delta_gap:.3f} after {mitigation_name}. "
            f"Counterintuitively, explanations are now *more* misaligned with bias "
            f"after mitigation — a paradoxical finding worth investigating."
        )

    return {
        'mitigation':           mitigation_name,
        'delta_mean_FEAS':      delta_mean,
        'delta_FEAS_SHAP':      delta_shap,
        'delta_FEAS_gap':       delta_gap,
        'delta_group_FEAS_diff': delta_group_diff,
        'FEAS_before':          feas_before['mean_FEAS'],
        'FEAS_after':           feas_after['mean_FEAS'],
        'gap_before':           feas_before['FEAS_gap'],
        'gap_after':            feas_after['FEAS_gap'],
        'gap_interpretation':   gap_interpretation,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: INTERSECTIONAL FEAS
# ══════════════════════════════════════════════════════════════════════════════

def compute_intersectional_feas(
    df: pd.DataFrame,
    X: np.ndarray,
    shap_values: np.ndarray,
    feature_names: list,
    attr_pairs: list = None,
) -> pd.DataFrame:
    """
    Compute FEAS for intersectional groups (combinations of 2+ attributes).

    This is novel even within FEAS: existing fairness research mostly audits
    ONE attribute at a time. Intersectional bias (e.g. Female + South Asian +
    Career gap) is underdetected by single-axis audits.

    By computing FEAS at the intersection, we can show:
    - Which combinations of attributes are most hidden from explanations
    - Whether the XAI-Fairness gap is worse for intersectionally marginalised groups

    Parameters
    ----------
    df            : candidates DataFrame
    X             : TF-IDF feature matrix
    shap_values   : SHAP values
    feature_names : list of feature names
    attr_pairs    : list of (attr1, attr2) tuples to combine
                    defaults to all pairs from the 4 standard attributes

    Returns
    -------
    DataFrame with one row per intersectional group combination
    """
    if attr_pairs is None:
        attrs = ['gender_proxy', 'institution_tier', 'career_gap', 'name_origin_proxy']
        from itertools import combinations
        attr_pairs = list(combinations(attrs, 2))

    sv = shap_values if shap_values.ndim == 2 else shap_values[:, :, 1]
    results = []

    for attr1, attr2 in attr_pairs:
        df_work = df.copy()
        df_work['intersect_group'] = (
            df_work[attr1].astype(str) + ' + ' + df_work[attr2].astype(str)
        )

        # Bias features for this combined attribute
        combined_encoded = LabelEncoder().fit_transform(
            df_work['intersect_group'].values
        )
        mi_scores = mutual_info_classif(X, combined_encoded,
                                        discrete_features=False, random_state=42)
        if mi_scores.max() > 0:
            mi_scores = mi_scores / mi_scores.max()
        top_idx   = np.argsort(mi_scores)[::-1][:20]
        B_intersect = {feature_names[i] for i in top_idx}

        for i, row in df_work.iterrows():
            E_shap = extract_shap_features(sv[i], feature_names, top_k=10)
            feas   = compute_feas(B_intersect, E_shap)
            results.append({
                'attr_pair':        f'{attr1} × {attr2}',
                'intersect_group':  row['intersect_group'],
                'candidate':        row['Name'],
                'FEAS_intersectional': round(feas, 4),
            })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY: FEAS SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def feas_summary_table(feas_results: dict) -> pd.DataFrame:
    """
    Generate a clean summary table for paper/UI display.
    """
    rows = [
        {'Metric':       'Mean FEAS (SHAP)',
         'Value':        f"{feas_results['mean_FEAS_SHAP']:.4f}",
         'Interpretation': 'Fraction of bias visible in SHAP explanations'},
        {'Metric':       'Mean FEAS (LIME)',
         'Value':        f"{feas_results['mean_FEAS_LIME']:.4f}",
         'Interpretation': 'Fraction of bias visible in LIME explanations'},
        {'Metric':       'Mean FEAS (combined)',
         'Value':        f"{feas_results['mean_FEAS']:.4f}",
         'Interpretation': 'Overall XAI-Fairness alignment'},
        {'Metric':       'FEAS Gap',
         'Value':        f"{feas_results['FEAS_gap']:.4f}",
         'Interpretation': 'Fraction of bias HIDDEN from explanations (1 - FEAS)'},
        {'Metric':       'Group FEAS Disparity',
         'Value':        f"{feas_results['group_FEAS_diff']:.4f}",
         'Interpretation': 'Difference in FEAS between privileged/unprivileged groups'},
        {'Metric':       'Verdict',
         'Value':        feas_results['alignment_verdict'],
         'Interpretation': 'Overall assessment'},
    ]
    return pd.DataFrame(rows)