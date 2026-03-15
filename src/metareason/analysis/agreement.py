import warnings
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import pearsonr, spearmanr

from metareason.pipeline.runner import SampleResult


def compute_pairwise_correlations(
    scores_by_oracle: Dict[str, List[float]],
) -> dict:
    """Pairwise Pearson and Spearman correlations between judges.

    Returns dict with 'pearson' and 'spearman' sub-dicts mapping
    (oracle_a, oracle_b) tuple keys to correlation coefficients.
    """
    names = list(scores_by_oracle.keys())
    pearson_corrs = {}
    spearman_corrs = {}

    for a, b in combinations(names, 2):
        scores_a = scores_by_oracle[a]
        scores_b = scores_by_oracle[b]

        # Only use indices where both have values
        paired = [
            (sa, sb)
            for sa, sb in zip(scores_a, scores_b)
            if sa is not None and sb is not None
        ]

        if len(paired) < 3:
            pearson_corrs[(a, b)] = None
            spearman_corrs[(a, b)] = None
            continue

        arr_a = [p[0] for p in paired]
        arr_b = [p[1] for p in paired]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_pearson, _ = pearsonr(arr_a, arr_b)
            r_spearman, _ = spearmanr(arr_a, arr_b)

        pearson_corrs[(a, b)] = None if np.isnan(r_pearson) else float(r_pearson)
        spearman_corrs[(a, b)] = None if np.isnan(r_spearman) else float(r_spearman)

    return {"pearson": pearson_corrs, "spearman": spearman_corrs}


def compute_krippendorff_alpha(
    scores_by_oracle: Dict[str, List[Optional[float]]],
    level_of_measurement: str = "interval",
) -> float:
    """Krippendorff's alpha for inter-rater reliability.

    Implements the algorithm directly using numpy.
    Handles missing data (None values). Returns alpha in [-1, 1].
    """
    oracle_names = list(scores_by_oracle.keys())
    n_units = max(len(v) for v in scores_by_oracle.values())
    n_coders = len(oracle_names)

    # Build reliability data matrix: coders x units
    # NaN for missing data
    data = np.full((n_coders, n_units), np.nan)
    for i, name in enumerate(oracle_names):
        scores = scores_by_oracle[name]
        for j, s in enumerate(scores):
            if s is not None:
                data[i, j] = s

    # Difference function for interval data
    def _diff_sq(v1, v2):
        return (v1 - v2) ** 2

    # Observed disagreement (D_o)
    # For each unit, compute pairwise disagreements among coders who rated it
    d_o_num = 0.0
    d_o_den = 0.0

    for u in range(n_units):
        values = data[:, u]
        valid = values[~np.isnan(values)]
        m_u = len(valid)
        if m_u < 2:
            continue
        # All pairs within this unit
        for i in range(m_u):
            for j in range(i + 1, m_u):
                d_o_num += _diff_sq(valid[i], valid[j])
        d_o_den += m_u - 1

    if d_o_den == 0:
        return 1.0  # No valid pairs at all

    D_o = d_o_num / d_o_den

    # Expected disagreement (D_e)
    # Collect all values with their unit weights
    all_values = []
    weights = []
    for u in range(n_units):
        values = data[:, u]
        valid = values[~np.isnan(values)]
        m_u = len(valid)
        if m_u < 2:
            continue
        for v in valid:
            all_values.append(v)
            weights.append(1.0)  # each value contributes equally

    all_values = np.array(all_values)
    n_total = len(all_values)

    if n_total < 2:
        return 1.0

    d_e_num = 0.0
    for i in range(n_total):
        for j in range(i + 1, n_total):
            d_e_num += _diff_sq(all_values[i], all_values[j])

    D_e = d_e_num / (n_total - 1)

    if D_e == 0:
        return 1.0  # Perfect agreement (all values identical)

    alpha = 1.0 - D_o / D_e
    return float(alpha)


def compute_agreement_summary(
    scores_by_oracle: Dict[str, List[Optional[float]]],
) -> dict:
    """Full agreement summary: alpha + pairwise correlations + means."""
    # Filter to non-None scores for correlations
    clean_scores = {}
    for name, scores in scores_by_oracle.items():
        clean_scores[name] = [s for s in scores if s is not None]

    correlations = compute_pairwise_correlations(scores_by_oracle)
    alpha = compute_krippendorff_alpha(scores_by_oracle)

    # Per-oracle means and stds
    oracle_stats = {}
    for name, scores in clean_scores.items():
        if scores:
            oracle_stats[name] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "n": len(scores),
            }

    # Mean pairwise correlations
    pearson_vals = [v for v in correlations["pearson"].values() if v is not None]
    spearman_vals = [v for v in correlations["spearman"].values() if v is not None]

    return {
        "krippendorff_alpha": alpha,
        "pairwise_correlations": {
            "pearson": {
                f"{k[0]}_vs_{k[1]}": v for k, v in correlations["pearson"].items()
            },
            "spearman": {
                f"{k[0]}_vs_{k[1]}": v for k, v in correlations["spearman"].items()
            },
        },
        "mean_pearson": float(np.mean(pearson_vals)) if pearson_vals else None,
        "mean_spearman": float(np.mean(spearman_vals)) if spearman_vals else None,
        "oracle_stats": oracle_stats,
        "n_judges": len(scores_by_oracle),
    }


def extract_scores_by_oracle(
    results: List[SampleResult],
) -> Dict[str, List[Optional[float]]]:
    """Extract per-oracle score lists from SampleResult objects.

    Returns None for missing evaluations (handles ragged data).
    """
    # Collect all oracle names across all results
    all_oracle_names = set()
    for r in results:
        all_oracle_names.update(r.evaluations.keys())

    scores_by_oracle: Dict[str, List[Optional[float]]] = {
        name: [] for name in sorted(all_oracle_names)
    }

    for r in results:
        for name in scores_by_oracle:
            if name in r.evaluations:
                scores_by_oracle[name].append(r.evaluations[name].score)
            else:
                scores_by_oracle[name].append(None)

    return scores_by_oracle
