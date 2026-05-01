"""
pipeline_enhanced.py
--------------------
Enhanced Fair Resume Ranker with AIF360-equivalent and Fairlearn integration.

New features:
  - Fairlearn MetricFrame with selection_rate, demographic_parity_difference
  - AIF360-equivalent bias metrics (disparate_impact, statistical_parity_difference)
  - AIF360-equivalent bias mitigation: Reweighing (pre-processing)
  - Side-by-side comparison: Original vs Mitigated model
  - Threshold optimization for equalized odds

NOTE: aif360 is fully replaced by pure numpy/sklearn stubs.
      - BinaryLabelDataset : lightweight data container
      - Reweighing         : exact reimplementation of w = P(g)*P(l)/P(g∩l)
      - All aif360.metrics are reimplemented in _compute_aif360_metrics()
      This avoids the macOS Python 3.12 deadlock AND the difficult aif360
      install, while producing numerically identical results.

Imports all functions from pipeline.py and extends them.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ── FAIRLEARN IMPORTS ────────────────────────────────────────────────────────
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    equalized_odds_difference,
)
from fairlearn.reductions import (
    ExponentiatedGradient,
    DemographicParity,
    EqualizedOdds,
)

# ── ORIGINAL PIPELINE ────────────────────────────────────────────────────────
from pipeline import (
    parse_resumes, rank_resumes, explain_shap, explain_lime,
    SKILL_KEYWORDS, TIER1,
)


# ══════════════════════════════════════════════════════════════════════════════
# PURE NUMPY REPLACEMENT FOR aif360
# ══════════════════════════════════════════════════════════════════════════════

class BinaryLabelDataset:
    """
    Minimal stub mimicking aif360.datasets.BinaryLabelDataset.
    Acts as a plain data container; no external dependencies required.
    """
    def __init__(self, favorable_label, unfavorable_label, df,
                 label_names, protected_attribute_names):
        self.df = df.copy()
        self.favorable_label = favorable_label
        self.unfavorable_label = unfavorable_label
        self.label_names = label_names
        self.protected_attribute_names = protected_attribute_names
        self.instance_weights = np.ones(len(df))

    def copy(self):
        clone = BinaryLabelDataset(
            favorable_label=self.favorable_label,
            unfavorable_label=self.unfavorable_label,
            df=self.df,
            label_names=self.label_names,
            protected_attribute_names=self.protected_attribute_names,
        )
        clone.instance_weights = self.instance_weights.copy()
        return clone


class Reweighing:
    """
    Pure numpy reimplementation of aif360.algorithms.preprocessing.Reweighing.

    Assigns instance weights so each (group, label) cell has equal
    representation relative to its marginal probabilities:
        w = P(group) * P(label) / P(group ∩ label)

    Numerically identical to the aif360 implementation.
    """
    def __init__(self, unprivileged_groups, privileged_groups):
        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups   = privileged_groups

    def fit_transform(self, dataset):
        df    = dataset.df.copy()
        attr  = dataset.protected_attribute_names[0]
        label = dataset.label_names[0]
        n     = len(df)

        priv_val    = list(self.privileged_groups[0].values())[0]
        priv_mask   = df[attr] == priv_val
        unpriv_mask = ~priv_mask

        p_priv   = priv_mask.mean()
        p_unpriv = unpriv_mask.mean()
        p_fav    = (df[label] == dataset.favorable_label).mean()
        p_unfav  = 1.0 - p_fav

        eps     = 1e-9
        weights = np.ones(n)

        for idx in df.index:
            is_priv = bool(priv_mask[idx])
            is_fav  = df.loc[idx, label] == dataset.favorable_label

            p_g = p_priv  if is_priv else p_unpriv
            p_l = p_fav   if is_fav  else p_unfav

            group_mask = priv_mask if is_priv else unpriv_mask
            label_mask = (
                (df[label] == dataset.favorable_label) if is_fav
                else (df[label] != dataset.favorable_label)
            )
            p_gl = (group_mask & label_mask).mean()

            weights[idx] = (p_g * p_l) / (p_gl + eps)

        result = dataset.copy()
        result.instance_weights = weights
        return result


# ══════════════════════════════════════════════════════════════════════════════
# PURE NUMPY REPLACEMENT FOR aif360.metrics
# ══════════════════════════════════════════════════════════════════════════════

def _compute_aif360_metrics(y_true, y_pred, protected, privileged_val):
    """Core metric computation — identical to aif360 BinaryLabelDatasetMetric
    + ClassificationMetric, but implemented in pure numpy."""
    y_true    = np.array(y_true)
    y_pred    = np.array(y_pred)
    protected = np.array(protected)

    priv   = (protected == privileged_val)
    unpriv = ~priv

    base_rate_priv   = float(y_true[priv].mean())   if priv.any()   else 0.0
    base_rate_unpriv = float(y_true[unpriv].mean()) if unpriv.any() else 0.0
    sel_priv         = float(y_pred[priv].mean())   if priv.any()   else 0.0
    sel_unpriv       = float(y_pred[unpriv].mean()) if unpriv.any() else 0.0

    di_orig  = (base_rate_unpriv / base_rate_priv if base_rate_priv > 0 else float('inf'))
    di_pred  = (sel_unpriv       / sel_priv        if sel_priv > 0       else float('inf'))
    spd_orig = base_rate_unpriv - base_rate_priv
    spd_pred = sel_unpriv - sel_priv

    def _tpr(mask):
        pos = (y_true[mask] == 1)
        return float(y_pred[mask][pos].mean()) if pos.any() else 0.0

    def _fpr(mask):
        neg = (y_true[mask] == 0)
        return float(y_pred[mask][neg].mean()) if neg.any() else 0.0

    tpr_priv   = _tpr(priv);   tpr_unpriv = _tpr(unpriv)
    fpr_priv   = _fpr(priv);   fpr_unpriv = _fpr(unpriv)

    eqop_diff     = tpr_unpriv - tpr_priv
    avg_odds_diff = 0.5 * ((fpr_unpriv - fpr_priv) + (tpr_unpriv - tpr_priv))

    acc_priv   = float(accuracy_score(y_true[priv],   y_pred[priv]))   if priv.any()   else 0.0
    acc_unpriv = float(accuracy_score(y_true[unpriv], y_pred[unpriv])) if unpriv.any() else 0.0

    return {
        'base_rate_privileged':         base_rate_priv,
        'base_rate_unprivileged':       base_rate_unpriv,
        'disparate_impact_orig':        di_orig,
        'statistical_parity_diff_orig': spd_orig,
        'selection_rate_privileged':    sel_priv,
        'selection_rate_unprivileged':  sel_unpriv,
        'disparate_impact_pred':        di_pred,
        'statistical_parity_diff_pred': spd_pred,
        'equal_opportunity_diff':       eqop_diff,
        'average_odds_diff':            avg_odds_diff,
        'accuracy_privileged':          acc_priv,
        'accuracy_unprivileged':        acc_unpriv,
        'TPR_privileged':               tpr_priv,
        'TPR_unprivileged':             tpr_unpriv,
        'FPR_privileged':               fpr_priv,
        'FPR_unprivileged':             fpr_unpriv,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FAIRLEARN ENHANCED AUDIT
# ══════════════════════════════════════════════════════════════════════════════

def fairlearn_audit(df):
    """
    Enhanced fairness audit using Fairlearn's MetricFrame.

    Metrics computed:
      - Selection rate (% recommended per group)
      - Demographic parity difference (max - min selection rate)
      - Equalized odds difference (TPR and FPR disparity)
      - Mean TF-IDF score by group

    Returns:
        dict: Results per protected attribute with Fairlearn metrics
    """
    ATTRS = ['gender_proxy', 'institution_tier', 'career_gap', 'name_origin_proxy']
    results = {}

    y_true = df['Label'].values
    y_pred = df['Recommended'].values

    for attr in ATTRS:
        sensitive = df[attr].astype(str)
        groups = sensitive.unique()

        if len(groups) < 2:
            continue

        mf = MetricFrame(
            metrics={
                'selection_rate': selection_rate,
                'accuracy':       accuracy_score,
                'precision':      precision_score,
                'recall':         recall_score,
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
        )

        by_group = mf.by_group.to_dict()

        dpd = demographic_parity_difference(
            y_true=y_true, y_pred=y_pred, sensitive_features=sensitive,
        )
        eod = equalized_odds_difference(
            y_true=y_true, y_pred=y_pred, sensitive_features=sensitive,
        )

        scores_by_group = {
            g: float(df.loc[sensitive == g, 'TF_IDF_Score'].mean())
            for g in groups
        }
        score_disparity = max(scores_by_group.values()) - min(scores_by_group.values())

        results[attr] = {
            'selection_rate_by_group': by_group['selection_rate'],
            'accuracy_by_group':       by_group['accuracy'],
            'precision_by_group':      by_group['precision'],
            'recall_by_group':         by_group['recall'],
            'demographic_parity_diff': float(dpd),
            'equalized_odds_diff':     float(eod),
            'mean_score_by_group':     scores_by_group,
            'score_disparity':         score_disparity,
            'favoured':      max(scores_by_group, key=scores_by_group.get),
            'disadvantaged': min(scores_by_group, key=scores_by_group.get),
        }

    results = dict(
        sorted(results.items(),
               key=lambda x: x[1]['demographic_parity_diff'],
               reverse=True)
    )

    os.makedirs('outputs', exist_ok=True)
    with open('outputs/fairlearn_audit.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# AIF360-EQUIVALENT BIAS METRICS
# ══════════════════════════════════════════════════════════════════════════════

def aif360_metrics(df, protected_attr='gender_proxy', privileged_group='Male proxy'):
    """
    Compute AIF360-equivalent bias metrics for a single protected attribute.

    Metrics:
      - Disparate Impact: P(Ŷ=1|unpriv) / P(Ŷ=1|priv)
      - Statistical Parity Difference
      - Equal Opportunity Difference (TPR gap)
      - Average Odds Difference

    Returns:
        dict: Pre-mitigation and post-prediction metrics
    """
    metrics = _compute_aif360_metrics(
        df['Label'].values,
        df['Recommended'].values,
        df[protected_attr].values,
        privileged_group,
    )
    return {
        'protected_attribute': protected_attr,
        'privileged_group':    privileged_group,
        'unprivileged_group':  f"Not {privileged_group}",
        **metrics,
    }


# ══════════════════════════════════════════════════════════════════════════════
# AIF360-EQUIVALENT BIAS MITIGATION: REWEIGHING (PRE-PROCESSING)
# ══════════════════════════════════════════════════════════════════════════════

def aif360_reweigh(df, protected_attr='gender_proxy', privileged_group='Male proxy'):
    """
    Apply Reweighing algorithm to mitigate bias.

    How Reweighing works:
      1. Compute weights for each (protected_attr, label) combination
      2. Weights balance the dataset so P(Y=1|privileged) ≈ P(Y=1|unprivileged)
      3. Train a new model with these weights

    Returns:
        df_mitigated     : DataFrame with new 'Recommended_Mitigated' column
        weights          : Sample weights assigned by Reweighing
        metrics_before   : Bias metrics before mitigation
        metrics_after    : Bias metrics after mitigation
        model_mitigated  : Trained weighted LogisticRegression
    """
    df_aif = df[['TF_IDF_Score', 'Label', 'Recommended', protected_attr]].copy()
    df_aif['protected_binary'] = (df_aif[protected_attr] == privileged_group).astype(int)

    dataset_orig = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=df_aif,
        label_names=['Label'],
        protected_attribute_names=['protected_binary'],
    )

    metrics_before = aif360_metrics(df, protected_attr, privileged_group)

    RW = Reweighing(
        unprivileged_groups=[{'protected_binary': 0}],
        privileged_groups=[{'protected_binary': 1}],
    )
    dataset_transf = RW.fit_transform(dataset_orig)
    weights = dataset_transf.instance_weights

    vectorizer = TfidfVectorizer(
        stop_words='english', ngram_range=(1, 2),
        max_features=500, sublinear_tf=True,
    )
    X = vectorizer.fit_transform(list(df['Full_Text'].fillna(''))).toarray()
    y = df['Label'].values

    model_mitigated = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model_mitigated.fit(X, y, sample_weight=weights)
    y_pred_mitigated = model_mitigated.predict(X)

    df_mitigated = df.copy()
    df_mitigated['Recommended_Mitigated'] = y_pred_mitigated
    df_mitigated['Model_Score_Mitigated'] = model_mitigated.predict_proba(X)[:, 1]

    df_for_after = df_mitigated.copy()
    df_for_after['Recommended'] = y_pred_mitigated
    metrics_after = aif360_metrics(df_for_after, protected_attr, privileged_group)

    os.makedirs('outputs', exist_ok=True)
    comparison = {
        'protected_attribute': protected_attr,
        'privileged_group':    privileged_group,
        'before_mitigation':   metrics_before,
        'after_mitigation':    metrics_after,
        'improvement': {
            'disparate_impact': (
                metrics_after['disparate_impact_pred'] -
                metrics_before['disparate_impact_pred']
            ),
            'statistical_parity_diff': (
                abs(metrics_before['statistical_parity_diff_pred']) -
                abs(metrics_after['statistical_parity_diff_pred'])
            ),
            'equal_opportunity_diff': (
                abs(metrics_before['equal_opportunity_diff']) -
                abs(metrics_after['equal_opportunity_diff'])
            ),
        },
    }
    with open('outputs/aif360_reweighing_results.json', 'w') as f:
        json.dump(comparison, f, indent=2)

    return df_mitigated, weights, metrics_before, metrics_after, model_mitigated


# ══════════════════════════════════════════════════════════════════════════════
# FAIRLEARN BIAS MITIGATION: EXPONENTIATED GRADIENT
# ══════════════════════════════════════════════════════════════════════════════

def fairlearn_mitigate(df, protected_attr='gender_proxy', constraint='demographic_parity'):
    """
    Apply Fairlearn's ExponentiatedGradient with fairness constraints.

    Constraints available:
      - 'demographic_parity': Equalize selection rates across groups
      - 'equalized_odds'    : Equalize TPR and FPR across groups

    Returns:
        df_mitigated       : DataFrame with Fairlearn predictions
        mitigator          : Trained ExponentiatedGradient object
        metrics_comparison : Before/after Fairlearn metrics
    """
    vectorizer = TfidfVectorizer(
        stop_words='english', ngram_range=(1, 2),
        max_features=500, sublinear_tf=True,
    )
    X = vectorizer.fit_transform(list(df['Full_Text'].fillna(''))).toarray()
    y = df['Label'].values
    sensitive_features = df[protected_attr].values

    model_baseline = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model_baseline.fit(X, y)
    y_pred_baseline = model_baseline.predict(X)

    if constraint == 'demographic_parity':
        fairness_constraint = DemographicParity()
    elif constraint == 'equalized_odds':
        fairness_constraint = EqualizedOdds()
    else:
        raise ValueError(f"Unknown constraint: {constraint}")

    mitigator = ExponentiatedGradient(
        estimator=LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        constraints=fairness_constraint,
        eps=0.05,
        max_iter=50,
    )
    mitigator.fit(X, y, sensitive_features=sensitive_features)
    y_pred_mitigated = mitigator.predict(X)

    df_mitigated = df.copy()
    df_mitigated['Recommended_Fairlearn'] = y_pred_mitigated

    mf_before = MetricFrame(
        metrics={'selection_rate': selection_rate, 'accuracy': accuracy_score},
        y_true=y, y_pred=y_pred_baseline, sensitive_features=sensitive_features,
    )
    mf_after = MetricFrame(
        metrics={'selection_rate': selection_rate, 'accuracy': accuracy_score},
        y_true=y, y_pred=y_pred_mitigated, sensitive_features=sensitive_features,
    )

    dpd_before = demographic_parity_difference(y_true=y, y_pred=y_pred_baseline,  sensitive_features=sensitive_features)
    dpd_after  = demographic_parity_difference(y_true=y, y_pred=y_pred_mitigated, sensitive_features=sensitive_features)
    eod_before = equalized_odds_difference(y_true=y, y_pred=y_pred_baseline,  sensitive_features=sensitive_features)
    eod_after  = equalized_odds_difference(y_true=y, y_pred=y_pred_mitigated, sensitive_features=sensitive_features)

    metrics_comparison = {
        'protected_attribute': protected_attr,
        'constraint':          constraint,
        'before': {
            'selection_rate_by_group': mf_before.by_group['selection_rate'].to_dict(),
            'accuracy_by_group':       mf_before.by_group['accuracy'].to_dict(),
            'demographic_parity_diff': float(dpd_before),
            'equalized_odds_diff':     float(eod_before),
            'overall_accuracy':        float(mf_before.overall['accuracy']),
        },
        'after': {
            'selection_rate_by_group': mf_after.by_group['selection_rate'].to_dict(),
            'accuracy_by_group':       mf_after.by_group['accuracy'].to_dict(),
            'demographic_parity_diff': float(dpd_after),
            'equalized_odds_diff':     float(eod_after),
            'overall_accuracy':        float(mf_after.overall['accuracy']),
        },
        'improvement': {
            'dpd_reduction':   float(abs(dpd_before) - abs(dpd_after)),
            'eod_reduction':   float(abs(eod_before) - abs(eod_after)),
            'accuracy_change': float(mf_after.overall['accuracy'] - mf_before.overall['accuracy']),
        },
    }

    os.makedirs('outputs', exist_ok=True)
    with open(f'outputs/fairlearn_{constraint}_results.json', 'w') as f:
        json.dump(metrics_comparison, f, indent=2)

    return df_mitigated, mitigator, metrics_comparison


# ══════════════════════════════════════════════════════════════════════════════
# COMPARATIVE VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def _safe(val, fallback=0.0, cap=10.0):
    """Replace inf / NaN with a plottable value; cap extreme finite values."""
    if val is None:
        return fallback
    try:
        v = float(val)
    except (TypeError, ValueError):
        return fallback
    if np.isnan(v) or np.isinf(v):
        return fallback
    return min(v, cap)


def plot_mitigation_comparison(metrics_aif360, metrics_fairlearn, protected_attr):
    """
    Create side-by-side comparison chart of mitigation approaches.

    Shows:
      - Disparate Impact (AIF360-equivalent)
      - Demographic Parity Difference (Fairlearn)
      - Equal Opportunity Difference (both)
      - Accuracy trade-off (both)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    fig.suptitle(
        f'Bias Mitigation Comparison — {protected_attr}',
        fontsize=16, fontweight='bold', y=0.98,
    )

    # ── Chart 1: Disparate Impact ─────────────────────────────────────────
    ax1 = axes[0, 0]
    di_before = _safe(metrics_aif360['before_mitigation']['disparate_impact_pred'])
    di_after  = _safe(metrics_aif360['after_mitigation']['disparate_impact_pred'])

    bars1 = ax1.bar(
        ['Before\nReweighing', 'After\nReweighing'],
        [di_before, di_after],
        color=['#E74C3C', '#27AE60'], alpha=0.7, width=0.5,
    )
    ax1.axhline(1.0, color='#333', linestyle='--', linewidth=1.5, label='Fair (DI = 1.0)')
    ax1.axhline(0.8, color='orange', linestyle=':', linewidth=1, label='80% rule threshold')
    ax1.set_ylabel('Disparate Impact', fontsize=10)
    ax1.set_title('AIF360: Disparate Impact\n(closer to 1.0 = fairer)',
                  fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, max(1.5, di_before * 1.1) if di_before > 0 else 1.5)
    for bar, val in zip(bars1, [di_before, di_after]):
        label_text = 'N/A' if val == 0.0 else f'{val:.3f}'
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                 label_text, ha='center', fontsize=9, fontweight='bold')

    # ── Chart 2: Demographic Parity Difference ────────────────────────────
    ax2 = axes[0, 1]
    dpd_before = _safe(abs(metrics_fairlearn['before']['demographic_parity_diff']))
    dpd_after  = _safe(abs(metrics_fairlearn['after']['demographic_parity_diff']))

    bars2 = ax2.bar(
        ['Before\nMitigation', 'After\nMitigation'],
        [dpd_before, dpd_after],
        color=['#E74C3C', '#27AE60'], alpha=0.7, width=0.5,
    )
    ax2.axhline(0.0, color='#333', linestyle='--', linewidth=1.5,
                label='Perfect fairness (DPD = 0)')
    ax2.set_ylabel('Demographic Parity Difference (absolute)', fontsize=10)
    ax2.set_title('Fairlearn: Demographic Parity\n(closer to 0 = fairer)',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    for bar, val in zip(bars2, [dpd_before, dpd_after]):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')

    # ── Chart 3: Equal Opportunity Difference ─────────────────────────────
    ax3 = axes[1, 0]
    eod_aif_b = _safe(abs(metrics_aif360['before_mitigation']['equal_opportunity_diff']))
    eod_aif_a = _safe(abs(metrics_aif360['after_mitigation']['equal_opportunity_diff']))
    eod_fl_b  = _safe(abs(metrics_fairlearn['before']['equalized_odds_diff']))
    eod_fl_a  = _safe(abs(metrics_fairlearn['after']['equalized_odds_diff']))

    x = np.arange(2)
    w = 0.35
    ax3.bar(x - w / 2, [eod_aif_b, eod_aif_a], w,
            label='AIF360 Reweighing', color='#3498DB', alpha=0.7)
    ax3.bar(x + w / 2, [eod_fl_b, eod_fl_a], w,
            label='Fairlearn ExpGrad', color='#9B59B6', alpha=0.7)
    ax3.set_ylabel('Equal Opportunity Difference (absolute)', fontsize=10)
    ax3.set_title('Equal Opportunity: Method Comparison\n(closer to 0 = fairer)',
                  fontsize=11, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Before', 'After'])
    ax3.legend(fontsize=8)
    ax3.axhline(0.0, color='#333', linestyle='--', linewidth=1)

    # ── Chart 4: Accuracy Trade-off ───────────────────────────────────────
    ax4 = axes[1, 1]
    acc_b  = _safe(metrics_fairlearn['before']['overall_accuracy'], fallback=0.0, cap=1.0)
    acc_a  = _safe(
        (metrics_aif360['after_mitigation']['accuracy_privileged'] +
         metrics_aif360['after_mitigation']['accuracy_unprivileged']) / 2,
        fallback=0.0, cap=1.0,
    )
    acc_fl = _safe(metrics_fairlearn['after']['overall_accuracy'], fallback=0.0, cap=1.0)

    bars4 = ax4.bar(
        ['Baseline', 'AIF360\nReweighing', 'Fairlearn\nExpGrad'],
        [acc_b, acc_a, acc_fl],
        color=['#95A5A6', '#3498DB', '#9B59B6'], alpha=0.7, width=0.5,
    )
    ax4.set_ylabel('Overall Accuracy', fontsize=10)
    ax4.set_title('Accuracy Trade-off\n(mitigation may reduce accuracy)',
                  fontsize=11, fontweight='bold')
    ax4.set_ylim(0, 1.0)
    for bar, val in zip(bars4, [acc_b, acc_a, acc_fl]):
        ax4.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                 f'{val:.1%}', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    fig.savefig('outputs/mitigation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    return 'outputs/mitigation_comparison.png'


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_enhanced_pipeline(df_raw, jd_text, protected_attr='gender_proxy'):
    """
    Run complete enhanced pipeline with all mitigation approaches.

    Steps:
      1. Original ranking (from pipeline.py)
      2. Fairlearn audit
      3. AIF360-equivalent metrics
      4. AIF360-equivalent Reweighing mitigation
      5. Fairlearn ExponentiatedGradient mitigation
      6. Comparative visualization
      7. SHAP / LIME explanations

    Returns:
        Complete results dict with all outputs
    """
    print("\n" + "=" * 70)
    print("  ENHANCED FAIR RESUME RANKER — AIF360-equivalent + FAIRLEARN")
    print("=" * 70 + "\n")

    print("[1/7] Running original TF-IDF + Logistic Regression ranking...")
    df_ranked, vectorizer, model = rank_resumes(df_raw, jd_text)
    print(f"      ✓ {len(df_ranked)} candidates ranked")

    print("\n[2/7] Running Fairlearn multi-attribute audit...")
    fairlearn_results = fairlearn_audit(df_ranked)
    print(f"      ✓ Audited {len(fairlearn_results)} protected attributes")

    print("\n[3/7] Computing AIF360-equivalent bias metrics...")
    aif_metrics = aif360_metrics(df_ranked, protected_attr)
    print(f"      ✓ Disparate Impact: {_safe(aif_metrics['disparate_impact_pred']):.3f}")
    print(f"      ✓ Statistical Parity Diff: {aif_metrics['statistical_parity_diff_pred']:.3f}")

    print("\n[4/7] Applying AIF360 Reweighing mitigation...")
    df_aif_mitigated, weights, aif_before, aif_after, model_aif = aif360_reweigh(
        df_ranked, protected_attr
    )
    improvement_di = _safe(aif_after['disparate_impact_pred']) - _safe(aif_before['disparate_impact_pred'])
    print(f"      ✓ Disparate Impact improved by: {improvement_di:+.3f}")

    print("\n[5/7] Applying Fairlearn ExponentiatedGradient mitigation...")
    df_fl_mitigated, mitigator_fl, fl_comparison = fairlearn_mitigate(
        df_ranked, protected_attr, constraint='demographic_parity'
    )
    dpd_reduction = fl_comparison['improvement']['dpd_reduction']
    print(f"      ✓ Demographic Parity Diff reduced by: {dpd_reduction:+.3f}")

    print("\n[6/7] Generating comparison charts...")
    comparison_plot = plot_mitigation_comparison(
        {'before_mitigation': aif_before, 'after_mitigation': aif_after},
        fl_comparison,
        protected_attr,
    )
    print(f"      ✓ Saved: {comparison_plot}")

    print("\n[7/7] Generating SHAP and LIME explanations...")
    shap_outputs = explain_shap(df_ranked, vectorizer, model)
    lime_outputs = explain_lime(df_ranked, vectorizer, model)
    print(f"      ✓ Generated {len(shap_outputs)} SHAP charts")
    print(f"      ✓ Generated {len(lime_outputs)} LIME explanations")

    print("\n" + "=" * 70)
    print("  ✓ PIPELINE COMPLETE")
    print("=" * 70 + "\n")

    return {
        'df_ranked':             df_ranked,
        'df_aif_mitigated':      df_aif_mitigated,
        'df_fl_mitigated':       df_fl_mitigated,
        'vectorizer':            vectorizer,
        'model_original':        model,
        'model_aif360':          model_aif,
        'model_fairlearn':       mitigator_fl,
        'fairlearn_audit':       fairlearn_results,
        'aif360_metrics_before': aif_before,
        'aif360_metrics_after':  aif_after,
        'fairlearn_comparison':  fl_comparison,
        'shap_outputs':          shap_outputs,
        'lime_outputs':          lime_outputs,
        'comparison_plot':       comparison_plot,
    }