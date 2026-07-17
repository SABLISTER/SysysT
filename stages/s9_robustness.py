"""Stage 9: Robustness analyses.

Five statistical tests to confirm that findings are not artifacts:
1. Bootstrap confidence intervals for mean relevance score
2. Split-half reliability (random splits + correlation)
3. Sensitivity analysis at different relevance thresholds
4. Permutation test on the condition-criticality gradient
5. Condition–brain-region co-occurrence network analysis

No LLM calls — pure statistics with numpy.
scipy is NOT required — Spearman rank correlation is implemented manually.
"""

import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np

from core.io import atomic_write_json, atomic_write_text

logger = logging.getLogger(__name__)

# ── Expected E/I disruption ordering ──────────────────────────────────────
# Conditions ordered by predicted severity of excitatory/inhibitory imbalance.
# Stronger E/I disruption should correlate with higher criticality rates.

EI_RANKING = [
    "Fragile X",
    "Rett syndrome",
    "Epilepsy",
    "Intellectual disability",
    "Schizophrenia",
    "Anxiety",
    "Depression",
    "ADHD",
]

_CONDITION_PATTERNS = {
    "fragile x": "Fragile X",
    "rett": "Rett syndrome",
    "epilep": "Epilepsy",
    "seizure": "Epilepsy",
    "intellectual disab": "Intellectual disability",
    "schizophren": "Schizophrenia",
    "anxiety": "Anxiety",
    "depress": "Depression",
    "adhd": "ADHD",
    "attention deficit": "ADHD",
    "attention-deficit": "ADHD",
}


# ── Paper field helpers ───────────────────────────────────────────────────

def _get_relevance_score(paper: dict) -> int:
    """Get relevance score from a paper dict, defaulting to 0."""
    score = paper.get("relevance_score", 0)
    if score is None:
        return 0
    try:
        return int(score)
    except (ValueError, TypeError):
        return 0


def _supports_criticality(paper: dict) -> bool:
    """Check if a paper supports E/I criticality.

    Looks at the ``supports_criticality`` field first; falls back to
    scanning ``findings`` for criticality-related keywords.
    """
    val = paper.get("supports_criticality")
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "yes", "1")

    # Fallback: scan findings for criticality language
    findings = paper.get("findings", [])
    if isinstance(findings, list):
        text = " ".join(str(f) for f in findings).lower()
        keywords = ["e/i", "excitatory", "inhibitory", "criticality",
                     "excitation", "inhibition", "balance", "imbalance"]
        return any(kw in text for kw in keywords)
    return False


def _normalize_condition(raw: str) -> Optional[str]:
    """Map raw text to a canonical condition name from EI_RANKING."""
    raw_lower = raw.lower()
    for pattern, canonical in _CONDITION_PATTERNS.items():
        if pattern in raw_lower:
            return canonical
    return None


def _extract_conditions(paper: dict) -> list[str]:
    """Extract normalized conditions from a paper.

    Checks the ``conditions`` field (list of strings), ``title``,
    and ``abstract`` for condition pattern matches.
    """
    found: set[str] = set()

    # From explicit conditions field
    conditions_field = paper.get("conditions", [])
    if isinstance(conditions_field, list):
        for c in conditions_field:
            if isinstance(c, str):
                norm = _normalize_condition(c)
                if norm:
                    found.add(norm)

    # From title + abstract text
    text = " ".join(filter(None, [
        paper.get("title", ""),
        paper.get("abstract", ""),
    ]))
    for pattern, canonical in _CONDITION_PATTERNS.items():
        if pattern in text.lower():
            found.add(canonical)

    return list(found)


# ── Spearman rank correlation (no scipy) ──────────────────────────────────

def _rank_array(values: list[float]) -> list[float]:
    """Assign fractional ranks to values (handles ties)."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i
        # Find all tied values
        while j < n - 1 and values[indexed[j]] == values[indexed[j + 1]]:
            j += 1
        # Assign average rank to ties (ranks are 1-based)
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1

    return ranks


def _spearman_rank_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation without scipy.

    Ranks both arrays, then computes Pearson correlation on the ranks.
    Returns 0.0 if the data is degenerate (constant values, too few points).
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    rx = _rank_array(x)
    ry = _rank_array(y)

    n = len(rx)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    var_x = sum((rx[i] - mean_rx) ** 2 for i in range(n))
    var_y = sum((ry[i] - mean_ry) ** 2 for i in range(n))

    denom = math.sqrt(var_x * var_y)
    if denom < 1e-12:
        return 0.0

    return cov / denom


# ── Standardised effect size ─────────────────────────────────────────────

def cluster_effect_size(
    group_a: np.ndarray,
    group_b: np.ndarray,
    hedges_correct: bool = True,
) -> dict:
    """Cohen's d (and optionally Hedges' g) between two groups.

    Returns dict with d, hedges_g, ci_lower, ci_upper, magnitude, n_a, n_b.
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    n_a, n_b = len(a), len(b)
    df = n_a + n_b - 2

    if df < 1 or n_a < 1 or n_b < 1:
        return {"d": 0.0, "hedges_g": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                "magnitude": "negligible", "n_a": n_a, "n_b": n_b}

    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a = float(np.var(a, ddof=1)) if n_a > 1 else 0.0
    var_b = float(np.var(b, ddof=1)) if n_b > 1 else 0.0

    pooled_sd = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / df)
    if pooled_sd < 1e-12:
        return {"d": 0.0, "hedges_g": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                "magnitude": "negligible", "n_a": n_a, "n_b": n_b}

    d = (mean_a - mean_b) / pooled_sd

    # Hedges' correction factor J
    j = 1.0 - 3.0 / (4.0 * df - 1.0) if hedges_correct else 1.0
    g = d * j

    # SE of d (Borenstein et al., 2009)
    se = math.sqrt((n_a + n_b) / (n_a * n_b) + d * d / (2.0 * (n_a + n_b)))
    ci_lower = d - 1.96 * se
    ci_upper = d + 1.96 * se

    abs_d = abs(d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    return {
        "d": round(d, 6),
        "hedges_g": round(g, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "magnitude": magnitude,
        "n_a": n_a,
        "n_b": n_b,
    }


# ── Kendall's τ and ICC ─────────────────────────────────────────────────

def kendall_tau(x: list[float], y: list[float]) -> dict:
    """Kendall's tau-b with p-value via scipy."""
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return {"tau": 0.0, "p_value": 1.0}
    from scipy.stats import kendalltau
    tau, p = kendalltau(x, y)
    if math.isnan(tau):
        tau = 0.0
    if math.isnan(p):
        p = 1.0
    return {"tau": float(tau), "p_value": float(p)}


def icc_oneway(ratings: np.ndarray) -> dict:
    """ICC(1,1) — one-way random, single measures.

    Parameters
    ----------
    ratings : (n_subjects, k_raters) array

    Returns
    -------
    dict with icc, ci_lower, ci_upper
    """
    ratings = np.asarray(ratings, dtype=np.float64)
    n, k = ratings.shape
    if n < 3:
        return {"icc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    # One-way ANOVA decomposition
    grand_mean = np.mean(ratings)
    row_means = np.mean(ratings, axis=1)

    ss_between = k * np.sum((row_means - grand_mean) ** 2)
    ss_within = np.sum((ratings - row_means[:, np.newaxis]) ** 2)

    df_between = n - 1
    df_within = n * (k - 1)

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within if df_within > 0 else 0.0

    # ICC(1,1) = (MS_between - MS_within) / (MS_between + (k-1)*MS_within)
    denom = ms_between + (k - 1) * ms_within
    if denom < 1e-12:
        return {"icc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    icc = (ms_between - ms_within) / denom

    # Confidence interval via F-distribution
    from scipy.stats import f as f_dist
    f_val = ms_between / ms_within if ms_within > 1e-12 else 1e6
    alpha = 0.05

    fl = f_val / f_dist.ppf(1 - alpha / 2, df_between, df_within)
    fu = f_val / f_dist.ppf(alpha / 2, df_between, df_within)

    ci_lower = (fl - 1) / (fl + k - 1)
    ci_upper = (fu - 1) / (fu + k - 1)

    return {
        "icc": round(float(icc), 6),
        "ci_lower": round(float(ci_lower), 6),
        "ci_upper": round(float(ci_upper), 6),
    }


# ── Analysis 1: Bootstrap CI ─────────────────────────────────────────────

def bootstrap_ci(papers: list[dict], n_iter: int = 10000,
                 seed: int = 42) -> dict:
    """Bootstrap confidence intervals for the mean relevance score.

    Resamples papers with replacement ``n_iter`` times and computes the
    mean relevance score for each resample.  Returns the overall mean,
    2.5th/97.5th percentile CI, standard deviation, and metadata.
    """
    logger.info("bootstrap_ci: START — %d papers, %d iterations", len(papers), n_iter)

    n = len(papers)
    if n < 3:
        logger.warning("bootstrap_ci: < 3 papers, returning defaults")
        return {
            "mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
            "std": 0.0, "n_papers": n, "n_iterations": 0,
        }

    scores = np.array([_get_relevance_score(p) for p in papers], dtype=np.float64)
    rng = np.random.default_rng(seed)

    # Vectorised bootstrap: generate all resample indices at once
    indices = rng.integers(0, n, size=(n_iter, n))
    means = scores[indices].mean(axis=1)

    result = {
        "mean": float(np.mean(means)),
        "ci_lower": float(np.percentile(means, 2.5)),
        "ci_upper": float(np.percentile(means, 97.5)),
        "std": float(np.std(means)),
        "n_papers": n,
        "n_iterations": n_iter,
    }
    logger.info(
        "bootstrap_ci: DONE — mean=%.3f, 95%% CI=[%.3f, %.3f]",
        result["mean"], result["ci_lower"], result["ci_upper"],
    )
    return result


# ── Analysis 2: Split-half reliability ────────────────────────────────────

def split_half(papers: list[dict], n_splits: int = 1000,
               seed: int = 42) -> dict:
    """Split-half reliability test.

    Randomly splits papers into two halves ``n_splits`` times.  For each
    split, computes the mean relevance score of each half and measures
    the Pearson correlation between the two series of half-means.
    """
    logger.info("split_half: START — %d papers, %d splits", len(papers), n_splits)

    n = len(papers)
    if n < 4:
        logger.warning("split_half: < 4 papers, returning defaults")
        return {
            "mean_correlation": 0.0, "std_correlation": 0.0,
            "min_correlation": 0.0, "max_correlation": 0.0,
            "n_splits": 0, "interpretation": "poor",
        }

    scores = np.array([_get_relevance_score(p) for p in papers], dtype=np.float64)
    rng = np.random.default_rng(seed)
    half = n // 2
    correlations = []

    for _ in range(n_splits):
        perm = rng.permutation(n)
        half_a = scores[perm[:half]]
        half_b = scores[perm[half:half + half]]

        mean_a = float(np.mean(half_a))
        mean_b = float(np.mean(half_b))
        correlations.append(mean_a * mean_b)

    # Compute correlation of half-means across splits
    # We collect (mean_a, mean_b) for each split and correlate them
    means_a = []
    means_b = []
    rng2 = np.random.default_rng(seed)
    for _ in range(n_splits):
        perm = rng2.permutation(n)
        means_a.append(float(np.mean(scores[perm[:half]])))
        means_b.append(float(np.mean(scores[perm[half:half + half]])))

    corr_val = float(np.corrcoef(means_a, means_b)[0, 1])
    if np.isnan(corr_val):
        corr_val = 0.0

    # Also compute per-split correlations for spread info
    split_corrs = []
    rng3 = np.random.default_rng(seed)
    for _ in range(n_splits):
        perm = rng3.permutation(n)
        a = scores[perm[:half]]
        b = scores[perm[half:half + half]]
        min_len = min(len(a), len(b))
        if min_len < 2:
            continue
        r = float(np.corrcoef(a[:min_len], b[:min_len])[0, 1])
        if not np.isnan(r):
            split_corrs.append(r)

    if split_corrs:
        mean_r = float(np.mean(split_corrs))
        std_r = float(np.std(split_corrs))
        min_r = float(np.min(split_corrs))
        max_r = float(np.max(split_corrs))
    else:
        mean_r = corr_val
        std_r = 0.0
        min_r = corr_val
        max_r = corr_val

    # Kendall's τ on the same half-means
    tau_result = kendall_tau(means_a, means_b)

    # ICC: treat each split as a "rater", build (n_splits, 2) matrix
    ratings_matrix = np.column_stack([means_a, means_b])
    icc_result = icc_oneway(ratings_matrix)

    if mean_r > 0.8:
        interpretation = "excellent"
    elif mean_r > 0.6:
        interpretation = "good"
    elif mean_r > 0.4:
        interpretation = "fair"
    else:
        interpretation = "poor"

    result = {
        "mean_correlation": mean_r,
        "std_correlation": std_r,
        "min_correlation": min_r,
        "max_correlation": max_r,
        "kendall_tau": tau_result["tau"],
        "kendall_tau_p": tau_result["p_value"],
        "icc": icc_result["icc"],
        "icc_ci_lower": icc_result["ci_lower"],
        "icc_ci_upper": icc_result["ci_upper"],
        "n_splits": n_splits,
        "interpretation": interpretation,
    }
    logger.info(
        "split_half: DONE — mean_r=%.3f, tau=%.3f, icc=%.3f (%s), std=%.3f",
        mean_r, tau_result["tau"], icc_result["icc"], interpretation, std_r,
    )
    return result


# ── Analysis 3: Sensitivity analysis ──────────────────────────────────────

def sensitivity_analysis(papers: list[dict]) -> dict:
    """Test how results change at different relevance thresholds (1–5).

    For each threshold, counts papers passing, computes the percentage
    supporting criticality, and the mean relevance score of passing papers.
    """
    logger.info("sensitivity_analysis: START — %d papers", len(papers))

    thresholds: dict[int, dict] = {}
    prev_direction: Optional[bool] = None
    direction_flipped = False

    for thresh in range(1, 6):
        passing = [p for p in papers if _get_relevance_score(p) >= thresh]
        n_pass = len(passing)

        if n_pass > 0:
            n_crit = sum(1 for p in passing if _supports_criticality(p))
            pct_crit = (n_crit / n_pass) * 100.0
            mean_score = sum(_get_relevance_score(p) for p in passing) / n_pass
        else:
            pct_crit = 0.0
            mean_score = 0.0

        row = {
            "n_pass": n_pass,
            "pct_criticality": round(pct_crit, 2),
            "mean_score": round(mean_score, 3),
        }

        # Effect size: passing vs failing papers on relevance scores
        failing = [p for p in papers if _get_relevance_score(p) < thresh]
        if n_pass >= 2 and len(failing) >= 2:
            pass_scores = np.array([_get_relevance_score(p) for p in passing],
                                   dtype=np.float64)
            fail_scores = np.array([_get_relevance_score(p) for p in failing],
                                   dtype=np.float64)
            row["effect_size"] = cluster_effect_size(pass_scores, fail_scores)

        thresholds[thresh] = row

        # Track whether criticality direction flips
        current_above_50 = pct_crit > 50.0
        if prev_direction is not None and current_above_50 != prev_direction:
            direction_flipped = True
        if n_pass > 0:
            prev_direction = current_above_50

    stable = not direction_flipped

    result = {
        "thresholds": thresholds,
        "stable": stable,
    }
    logger.info(
        "sensitivity_analysis: DONE — stable=%s, thresholds tested=5",
        stable,
    )
    return result


# ── Analysis 4: Permutation test ──────────────────────────────────────────

def _condition_gradient(papers: list[dict], min_n: int = 3) -> dict:
    """Compute per-condition criticality rates.

    Returns a dict mapping condition name to
    ``{"n_papers": int, "criticality_rate": float}``.
    Conditions with fewer than ``min_n`` papers are excluded.
    """
    counts: dict[str, dict] = {}
    for cond in EI_RANKING:
        counts[cond] = {"total": 0, "critical": 0}

    for paper in papers:
        conds = _extract_conditions(paper)
        is_crit = _supports_criticality(paper)
        for c in conds:
            if c in counts:
                counts[c]["total"] += 1
                if is_crit:
                    counts[c]["critical"] += 1

    gradient: dict[str, dict] = {}
    for cond in EI_RANKING:
        total = counts[cond]["total"]
        if total >= min_n:
            rate = counts[cond]["critical"] / total
            gradient[cond] = {
                "n_papers": total,
                "criticality_rate": round(rate, 4),
            }
        elif total > 0:
            logger.info(
                "permutation_test: skipping %s (n=%d < min_n=%d)",
                cond, total, min_n,
            )

    return gradient


def permutation_test(papers: list[dict], n_perm: int = 10000,
                     seed: int = 42) -> dict:
    """Permutation test on the condition-criticality gradient.

    Tests whether the observed Spearman correlation between EI rank
    (predicted E/I disruption severity) and criticality rate is
    significant by permuting condition labels.
    """
    logger.info("permutation_test: START — %d papers, %d permutations", len(papers), n_perm)

    gradient = _condition_gradient(papers)

    if len(gradient) < 3:
        logger.warning(
            "permutation_test: only %d conditions with enough papers, "
            "returning defaults", len(gradient),
        )
        return {
            "observed_correlation": 0.0,
            "p_value": 1.0,
            "n_permutations": 0,
            "significant": False,
            "condition_gradient": gradient,
        }

    # Build ordered vectors: EI rank (1-based) vs criticality rate
    # Only include conditions present in the gradient
    ranks = []
    rates = []
    for i, cond in enumerate(EI_RANKING):
        if cond in gradient:
            ranks.append(float(i + 1))
            rates.append(gradient[cond]["criticality_rate"])

    observed_r = _spearman_rank_correlation(ranks, rates)

    # Permutation: shuffle rates, recompute correlation
    rng = random.Random(seed)
    n_extreme = 0
    for _ in range(n_perm):
        shuffled = list(rates)
        rng.shuffle(shuffled)
        perm_r = _spearman_rank_correlation(ranks, shuffled)
        if abs(perm_r) >= abs(observed_r):
            n_extreme += 1

    p_value = n_extreme / n_perm

    result = {
        "observed_correlation": round(observed_r, 4),
        "p_value": round(p_value, 4),
        "n_permutations": n_perm,
        "significant": p_value < 0.05,
        "condition_gradient": gradient,
    }
    logger.info(
        "permutation_test: DONE — r=%.4f, p=%.4f, significant=%s",
        observed_r, p_value, result["significant"],
    )
    return result


# ── Analysis 6: ADHD comorbidity sensitivity ─────────────────────────────

def adhd_comorbidity_analysis(papers: list[dict]) -> dict:
    """Compare criticality support and effect sizes for ADHD-included vs excluded subsets.

    Splits the corpus by whether each paper explicitly studies ADHD as a
    co-occurring or comorbid condition. Reports criticality rates and
    Hedges' g (relevance score distribution) for both groups — testing
    whether ADHD co-occurrence mediates observed E/I support rates.
    """
    adhd_kws = {"adhd", "attention deficit", "attention-deficit"}

    def _has_adhd(paper: dict) -> bool:
        text = " ".join(filter(None, [
            paper.get("title", ""),
            paper.get("abstract", ""),
            " ".join(str(c) for c in paper.get("conditions", [])),
        ])).lower()
        return any(kw in text for kw in adhd_kws)

    adhd_papers = [p for p in papers if _has_adhd(p)]
    clean_papers = [p for p in papers if not _has_adhd(p)]

    def _rate(subset: list[dict]) -> float:
        if not subset:
            return 0.0
        return sum(1 for p in subset if _supports_criticality(p)) / len(subset)

    n_adhd = len(adhd_papers)
    n_clean = len(clean_papers)
    rate_adhd = _rate(adhd_papers)
    rate_clean = _rate(clean_papers)

    # Effect size: relevance score distribution (ADHD-included vs excluded)
    scores_adhd = np.array([_get_relevance_score(p) for p in adhd_papers], dtype=np.float64)
    scores_clean = np.array([_get_relevance_score(p) for p in clean_papers], dtype=np.float64)
    es = cluster_effect_size(scores_adhd, scores_clean)

    result = {
        "n_with_adhd": n_adhd,
        "n_without_adhd": n_clean,
        "criticality_rate_with_adhd": round(rate_adhd, 4),
        "criticality_rate_without_adhd": round(rate_clean, 4),
        "rate_difference": round(rate_adhd - rate_clean, 4),
        "effect_size_with_adhd": es,
        "hedges_g": round(es.get("hedges_g", 0.0), 4),
        "interpretation": (
            "ADHD co-occurrence substantially inflates observed E/I support rates"
            if abs(rate_adhd - rate_clean) >= 0.05 else
            "ADHD co-occurrence does not substantially alter E/I support rates"
        ),
    }
    logger.info(
        "adhd_comorbidity: n_adhd=%d, n_clean=%d, rate_adhd=%.3f, "
        "rate_clean=%.3f, g=%.3f",
        n_adhd, n_clean, rate_adhd, rate_clean, result["hedges_g"],
    )
    return result


# ── Report writer ─────────────────────────────────────────────────────────

def _write_report(results: dict, path: Path, n_papers: int) -> None:
    """Write a markdown robustness report."""
    lines = [
        "# Robustness Analysis Report",
        "",
        f"**Papers analysed:** {n_papers}",
        "",
    ]

    # Bootstrap CI
    bs = results.get("bootstrap", {})
    lines += [
        "## 1. Bootstrap Confidence Intervals",
        "",
        f"- **Mean relevance score:** {bs.get('mean', 0):.3f}",
        f"- **95% CI:** [{bs.get('ci_lower', 0):.3f}, {bs.get('ci_upper', 0):.3f}]",
        f"- **Std:** {bs.get('std', 0):.3f}",
        f"- **Iterations:** {bs.get('n_iterations', 0):,}",
        "",
    ]

    # Split-half
    sh = results.get("split_half", {})
    lines += [
        "## 2. Split-Half Reliability",
        "",
        f"- **Spearman ρ (mean):** {sh.get('mean_correlation', 0):.3f}",
        f"- **Kendall τ:** {sh.get('kendall_tau', 0):.3f} (p = {sh.get('kendall_tau_p', 1):.4f})",
        f"- **ICC(1,1):** {sh.get('icc', 0):.3f} "
        f"[{sh.get('icc_ci_lower', 0):.3f}, {sh.get('icc_ci_upper', 0):.3f}]",
        f"- **Std:** {sh.get('std_correlation', 0):.3f}",
        f"- **Range:** [{sh.get('min_correlation', 0):.3f}, {sh.get('max_correlation', 0):.3f}]",
        f"- **Interpretation:** {sh.get('interpretation', 'N/A')}",
        f"- **Splits:** {sh.get('n_splits', 0):,}",
        "",
    ]

    # Sensitivity
    sa = results.get("sensitivity", {})
    lines += [
        "## 3. Sensitivity Analysis",
        "",
        f"- **Stable across thresholds:** {sa.get('stable', 'N/A')}",
        "",
        "| Threshold | Papers | Criticality % | Mean Score | Cohen's d | Magnitude |",
        "|-----------|--------|---------------|------------|-----------|-----------|",
    ]
    for t in range(1, 6):
        info = sa.get("thresholds", {}).get(str(t), sa.get("thresholds", {}).get(t, {}))
        es = info.get("effect_size", {})
        d_str = f"{es.get('d', 0):.2f}" if es else "—"
        mag_str = es.get("magnitude", "—") if es else "—"
        lines.append(
            f"| {t} | {info.get('n_pass', 0)} | "
            f"{info.get('pct_criticality', 0):.1f}% | "
            f"{info.get('mean_score', 0):.3f} | "
            f"{d_str} | {mag_str} |"
        )
    lines.append("")

    # Permutation test
    pt = results.get("permutation", {})
    lines += [
        "## 4. Permutation Test (Condition–Criticality Gradient)",
        "",
        f"- **Observed Spearman r:** {pt.get('observed_correlation', 0):.4f}",
        f"- **p-value:** {pt.get('p_value', 1):.4f}",
        f"- **Significant (p < 0.05):** {pt.get('significant', False)}",
        f"- **Permutations:** {pt.get('n_permutations', 0):,}",
        "",
        "### Condition Gradient",
        "",
        "| Condition | Papers | Criticality Rate |",
        "|-----------|--------|-----------------|",
    ]
    gradient = pt.get("condition_gradient", {})
    for cond in EI_RANKING:
        if cond in gradient:
            info = gradient[cond]
            lines.append(
                f"| {cond} | {info['n_papers']} | "
                f"{info['criticality_rate']:.2%} |"
            )
    lines.append("")

    # Network analysis
    net = results.get("network_analysis", {})
    if net:
        comm = net.get("communities", {})
        lines += [
            "## 5. Condition–Brain-Region Co-occurrence Networks",
            "",
            f"- **Edges:** {net.get('n_edges', 0)} "
            f"({net.get('n_significant_edges', '?')} significant)",
            f"- **Edge threshold:** {net.get('edge_threshold', '—')} "
            f"(permutation null, α=0.05)",
            f"- **Conditions found:** {net.get('n_conditions', 0)}",
            f"- **Regions found:** {net.get('n_regions', 0)}",
            f"- **Communities:** {comm.get('n_communities', 0)} "
            f"(modularity = {comm.get('modularity', 0):.3f})",
            "",
        ]

    # ADHD comorbidity analysis
    adhd = results.get("adhd_comorbidity", {})
    if adhd:
        g = adhd.get("hedges_g", 0.0)
        lines += [
            "## 6. ADHD Comorbidity Sensitivity",
            "",
            f"- **Papers with ADHD co-occurrence:** {adhd.get('n_with_adhd', 0)}",
            f"- **Papers without ADHD:** {adhd.get('n_without_adhd', 0)}",
            f"- **Criticality rate (ADHD-included):** {adhd.get('criticality_rate_with_adhd', 0):.1%}",
            f"- **Criticality rate (ADHD-excluded):** {adhd.get('criticality_rate_without_adhd', 0):.1%}",
            f"- **Rate difference:** {adhd.get('rate_difference', 0):+.1%}",
            f"- **Hedges' g:** {g:.2f} ({adhd.get('effect_size_with_adhd', {}).get('magnitude', '—')})",
            f"- **Interpretation:** {adhd.get('interpretation', '')}",
            "",
        ]

    lines += [
        "---",
        "*Generated by SystS Stage 9: Robustness Analyses*",
    ]

    atomic_write_text(path, "\n".join(lines))
    logger.info("Robustness report written: %s", path)


def _write_network_report(edges: list[dict], path: Path,
                          n_papers: int) -> None:
    """Write a markdown report for the condition–brain-region networks."""
    lines = [
        "# Autism Condition–Brain-Region Co-occurrence Networks",
        "",
        f"**Papers analysed:** {n_papers}",
        f"**Edges detected:** {len(edges)}",
        "",
        "## Top Co-occurrences",
        "",
        "| Condition | Region | Weight | Region Type |",
        "|-----------|--------|--------|-------------|",
    ]

    for edge in edges[:30]:
        lines.append(
            f"| {edge['source']} | {edge['target']} | "
            f"{edge['weight']} | {edge.get('target_type', 'region')} |"
        )

    lines += [
        "",
        "---",
        "*Generated by SystS Stage 9: Network Analysis*",
    ]

    atomic_write_text(path, "\n".join(lines))
    logger.info("Network report written: %s", path)


# ── Main entry point ──────────────────────────────────────────────────────

def run(
    claims_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    config=None,
    seed: int = 42,
    bootstrap_iterations: int = 10000,
    split_iterations: int = 1000,
    permutation_iterations: int = 10000,
    cancel_event=None,
    progress_callback=None,
    **kw,
) -> dict:
    """Run all 5 robustness analyses.

    Parameters
    ----------
    claims_path : str, optional
        Explicit path to claims JSON.  If not provided, standard locations
        under ``config.data_dir`` are searched.
    output_dir : str, optional
        Override for output directory.
    config : Config, optional
        Pipeline configuration object.
    seed : int
        Random seed for reproducibility.
    bootstrap_iterations, split_iterations, permutation_iterations : int
        Iteration counts for each analysis.
    cancel_event : threading.Event or similar, optional
        Set to cancel the run.
    progress_callback : callable, optional
        Called with progress message strings.
    """

    def _cancelled() -> bool:
        if cancel_event is None:
            return False
        if hasattr(cancel_event, "is_set"):
            return cancel_event.is_set()
        return bool(cancel_event)

    def _progress(msg: str) -> None:
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    _progress("═" * 60)
    _progress("STAGE 9: ROBUSTNESS ANALYSES")
    _progress("═" * 60)

    # ── Resolve paths ─────────────────────────────────────────────────
    project_root = config.project_root if config else Path.cwd()
    data_dir = project_root / "data"
    validation_dir = data_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    # ── Load claims corpus ────────────────────────────────────────────
    corpus_path = None
    if claims_path:
        corpus_path = Path(claims_path)
    else:
        candidates = [
            data_dir / "claims" / "claims_filtered.json",
            data_dir / "claims" / "claims.json",
            data_dir / "corpus" / "relevant.json",
        ]
        if config:
            candidates = [
                config.claims_dir / "claims_filtered.json",
                config.claims_dir / "claims.json",
                config.corpus_dir / "relevant.json",
            ] + candidates
        for cand in candidates:
            if cand.exists():
                corpus_path = cand
                break

    if corpus_path is None or not corpus_path.exists():
        _progress("ERROR: No claims corpus found. Run earlier stages first.")
        return {"error": "No claims corpus found"}

    with open(corpus_path, encoding="utf-8") as f:
        papers = json.load(f)
    if not isinstance(papers, list):
        papers = list(papers.values()) if isinstance(papers, dict) else []

    n_papers = len(papers)
    _progress(f"Loaded {n_papers} papers from {corpus_path}")

    if n_papers == 0:
        _progress("WARNING: Empty corpus — returning default results")
        empty_result = {
            "n_papers": 0,
            "bootstrap": {},
            "split_half": {},
            "sensitivity": {},
            "permutation": {},
            "network_analysis": {},
        }
        atomic_write_json(validation_dir / "robustness_results.json", empty_result)
        return empty_result

    results: dict[str, Any] = {"n_papers": n_papers}

    # ── Analysis 1: Bootstrap CI ──────────────────────────────────────
    if _cancelled():
        _progress("Cancelled.")
        return {"cancelled": True}

    _progress("Analysis 1/5: Bootstrap confidence intervals...")
    results["bootstrap"] = bootstrap_ci(
        papers, n_iter=bootstrap_iterations, seed=seed,
    )
    _progress(
        f"  Bootstrap: mean={results['bootstrap']['mean']:.3f}, "
        f"95% CI=[{results['bootstrap']['ci_lower']:.3f}, "
        f"{results['bootstrap']['ci_upper']:.3f}]"
    )

    # ── Analysis 2: Split-half reliability ────────────────────────────
    if _cancelled():
        _progress("Cancelled.")
        return {"cancelled": True}

    _progress("Analysis 2/5: Split-half reliability...")
    results["split_half"] = split_half(
        papers, n_splits=split_iterations, seed=seed,
    )
    _sh = results["split_half"]
    _progress(
        f"  Split-half: ρ={_sh['mean_correlation']:.3f}, "
        f"τ={_sh.get('kendall_tau', 0):.3f}, "
        f"ICC={_sh.get('icc', 0):.3f} "
        f"({_sh['interpretation']})"
    )

    # ── Analysis 3: Sensitivity analysis ──────────────────────────────
    if _cancelled():
        _progress("Cancelled.")
        return {"cancelled": True}

    _progress("Analysis 3/5: Sensitivity analysis (thresholds 1-5)...")
    results["sensitivity"] = sensitivity_analysis(papers)
    _progress(
        f"  Sensitivity: stable={results['sensitivity']['stable']}"
    )

    # ── Analysis 4: Permutation test ──────────────────────────────────
    if _cancelled():
        _progress("Cancelled.")
        return {"cancelled": True}

    _progress("Analysis 4/5: Permutation test (condition gradient)...")
    results["permutation"] = permutation_test(
        papers, n_perm=permutation_iterations, seed=seed,
    )
    _progress(
        f"  Permutation: r={results['permutation']['observed_correlation']:.4f}, "
        f"p={results['permutation']['p_value']:.4f}, "
        f"significant={results['permutation']['significant']}"
    )

    # ── Analysis 5: Network analysis ──────────────────────────────────
    if _cancelled():
        _progress("Cancelled.")
        return {"cancelled": True}

    _progress("Analysis 5/5: Condition–brain-region co-occurrence networks...")
    try:
        from process.autism_graph_analysis import (
            load_patterns,
            build_cooccurrence_edges,
            compute_edge_threshold,
            detect_communities,
        )

        patterns = load_patterns(config)
        edges = build_cooccurrence_edges(papers, patterns)

        # Data-driven thresholding
        threshold = compute_edge_threshold(
            papers, patterns, n_permutations=500, seed=seed,
        )
        for e in edges:
            e["significant"] = e["weight"] >= threshold
        n_significant = sum(1 for e in edges if e["significant"])
        _progress(f"  Edge threshold: {threshold} ({n_significant}/{len(edges)} significant)")

        # Community detection
        community_result = detect_communities(
            [e for e in edges if e["significant"]]
        )
        _progress(
            f"  Communities: {community_result['n_communities']} "
            f"(modularity={community_result['modularity']:.3f})"
        )

        # Summarise network
        conditions_found = set()
        regions_found = set()
        for e in edges:
            conditions_found.add(e["source"])
            regions_found.add(e["target"])

        network_result = {
            "n_edges": len(edges),
            "n_significant_edges": n_significant,
            "edge_threshold": threshold,
            "n_conditions": len(conditions_found),
            "n_regions": len(regions_found),
            "top_edges": edges[:20],
            "edges": edges,
            "communities": community_result,
        }
        results["network_analysis"] = network_result

        # Save network JSON (with threshold + communities)
        network_path = validation_dir / "autism_condition_networks.json"
        atomic_write_json(network_path, {
            "n_papers": n_papers,
            "n_edges": len(edges),
            "n_significant_edges": n_significant,
            "edge_threshold": threshold,
            "conditions": sorted(conditions_found),
            "regions": sorted(regions_found),
            "edges": edges,
            "communities": community_result,
        })
        _progress(f"  Networks: {len(edges)} edges, {len(conditions_found)} conditions, "
                  f"{len(regions_found)} regions")
        _progress(f"  Saved: {network_path}")

        # Write network report
        net_report_path = validation_dir / "autism_condition_networks_report.md"
        _write_network_report(edges, net_report_path, n_papers)
        results["network_report_path"] = str(net_report_path)

    except Exception as exc:
        logger.warning("Network analysis failed: %s", exc, exc_info=True)
        _progress(f"  Network analysis error: {exc}")
        results["network_analysis"] = {"error": str(exc)}

    # ── Analysis 6: ADHD comorbidity sensitivity ──────────────────────
    if _cancelled():
        _progress("Cancelled.")
        return {"cancelled": True}

    _progress("Analysis 6/6: ADHD comorbidity sensitivity...")
    results["adhd_comorbidity"] = adhd_comorbidity_analysis(papers)
    _ac = results["adhd_comorbidity"]
    _progress(
        f"  ADHD split: {_ac['n_with_adhd']} ADHD-included, "
        f"{_ac['n_without_adhd']} ADHD-excluded, "
        f"g={_ac['hedges_g']:.3f}"
    )

    # ── Save combined results ─────────────────────────────────────────
    results_path = validation_dir / "robustness_results.json"
    # Make a serialisable copy (strip full edge list from top-level)
    save_results = dict(results)
    if "network_analysis" in save_results:
        net = dict(save_results["network_analysis"])
        net.pop("edges", None)  # Too large for the summary JSON
        save_results["network_analysis"] = net
    atomic_write_json(results_path, save_results)
    _progress(f"Results saved: {results_path}")
    results["results_path"] = str(results_path)

    # ── Write robustness report ───────────────────────────────────────
    report_path = validation_dir / "robustness_report.md"
    _write_report(results, report_path, n_papers)
    results["report_path"] = str(report_path)

    _progress("═" * 60)
    _progress("STAGE 9 COMPLETE")
    _progress("═" * 60)

    return results


# ── Output checker ────────────────────────────────────────────────────────

def check_output(config=None) -> dict:
    """Check whether Stage 9 output already exists."""
    if config is None:
        return {"exists": False}
    path = config.data_dir / "validation" / "robustness_results.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return {
                "exists": True,
                "n_papers": data.get("n_papers", 0),
                "path": str(path),
            }
        except (json.JSONDecodeError, OSError):
            pass
    return {"exists": False}
