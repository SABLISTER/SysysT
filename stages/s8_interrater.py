"""Stage 8: Inter-rater reliability — cross-model validation.

Re-scores a stratified sample of papers using a SECOND independent LLM,
then computes agreement metrics (Cohen's kappa, weighted kappa, Jaccard,
Pearson correlation) against the original scores.  This validates that the
pipeline's findings aren't artifacts of the specific model used.
"""
import json
import logging
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

from core.checkpoint import CheckpointManager
from core.io import atomic_write_json, atomic_write_text

logger = logging.getLogger(__name__)

SUPPORTED_SECOND_RATERS = {
    "ollama", "lmstudio", "openai", "anthropic",
    "brt_bert", "brt-bert",
    "brt_irr", "brt-irr",
}

# Providers that score with a local cross-encoder instead of an LLM.  These
# emit a relevance logit only — no free-text generation — so claim extraction
# is skipped and IRR runs in score-only mode.  Both spellings are accepted
# because the GUI uses "brt-bert"/"brt-irr" while configs may use underscores.
# brt-bert  = brt-bert-relevance (original scorer, used as independent rater)
# brt-irr   = brt-bert-irr-buddy-mlm (model fine-tuned specifically for IRR)
EMBEDDING_SECOND_RATERS = {"brt_bert", "brt-bert", "brt_irr", "brt-irr"}



# ── Stratified sampling ─────────────────────────────────────────────────

def stratified_sample(papers: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Sample papers stratified by relevance_score.

    Ensure representation across score bins (0-2, 3-4, 5).
    If n >= len(papers), return all.
    """
    if n >= len(papers):
        return list(papers)

    bins: dict[str, list[dict]] = {"low": [], "mid": [], "high": []}
    for p in papers:
        score = p.get("relevance_score", 0)
        if score <= 2:
            bins["low"].append(p)
        elif score <= 4:
            bins["mid"].append(p)
        else:
            bins["high"].append(p)

    rng = random.Random(seed)
    sampled: list[dict] = []
    remaining = n

    # Proportional allocation
    non_empty_bins = {k: v for k, v in bins.items() if v}
    if not non_empty_bins:
        return []

    for bin_name, bin_papers in non_empty_bins.items():
        proportion = len(bin_papers) / len(papers)
        alloc = max(1, round(proportion * n))
        alloc = min(alloc, len(bin_papers), remaining)
        chosen = rng.sample(bin_papers, alloc)
        sampled.extend(chosen)
        remaining -= len(chosen)
        if remaining <= 0:
            break

    # Fill any remaining slots from the full pool (excluding already sampled)
    if remaining > 0:
        sampled_ids = {id(p) for p in sampled}
        pool = [p for p in papers if id(p) not in sampled_ids]
        if pool:
            extra = rng.sample(pool, min(remaining, len(pool)))
            sampled.extend(extra)

    return sampled


# ── Second-rater LLM calls ──────────────────────────────────────────────

def provider_score_relevance(model_name: str, paper: dict,
                             hypothesis: str = "") -> dict:
    """Score a paper using the second rater's LLM."""
    from process.llm import chat as llm_chat
    from process.relevance_filter import (
        RELEVANCE_PROMPT,
        _paper_text_for_scoring,
        _parse_relevance_response,
    )

    title = paper.get("title", "Unknown")
    paper_text, text_source = _paper_text_for_scoring(paper)

    if not paper_text.strip():
        logger.warning("Paper '%s' has no text to score", title[:60])
        return {"relevance_score": 0, "relevance_quote": "", "error": "no text"}

    # Fall back to a neutral hypothesis if none supplied.  (The
    # relevance_filter prompt itself substitutes "the research topic" when
    # the hypothesis is blank, so an empty string here is also safe.)
    if not hypothesis:
        hypothesis = "the research topic"

    prompt = RELEVANCE_PROMPT.format(
        hypothesis=hypothesis,
        title=title,
        text_source=text_source,
        paper_text=paper_text,
    )
    messages = [{"role": "user", "content": prompt}]

    t0 = time.time()
    response = llm_chat(model_name, messages, temperature=0, max_tokens=4096)
    elapsed = time.time() - t0

    result = _parse_relevance_response(response)
    logger.info(
        "[IRR-Score] %s → score=%d (%.1fs)",
        title[:60], result.get("relevance_score", 0), elapsed,
    )
    return result


def provider_extract_claims(model_name: str, paper: dict) -> dict:
    """Extract claims using the second rater's LLM."""
    from process.llm import chat as llm_chat
    from process.claim_extractor import (
        build_extraction_prompt,
        _parse_extraction_response,
    )

    title = paper.get("title", "Unknown")
    abstract = paper.get("abstract", "")
    if not abstract.strip():
        logger.warning("[IRR-Extract] No abstract for '%s'", title[:60])
        return {"error": "no_abstract"}

    prompt = build_extraction_prompt(title, abstract)
    messages = [{"role": "user", "content": prompt}]

    t0 = time.time()
    response = llm_chat(model_name, messages, temperature=0, max_tokens=4096)
    elapsed = time.time() - t0

    result = _parse_extraction_response(response)
    n_findings = len(result.get("findings", []))
    logger.info(
        "[IRR-Extract] %s → %d findings (%.1fs)",
        title[:60], n_findings, elapsed,
    )
    return result


# ── Agreement metrics ────────────────────────────────────────────────────

def normalize_label(text: str) -> str:
    """Normalize text for comparison (lowercase, strip, collapse whitespace)."""
    return re.sub(r"\s+", " ", str(text).lower().strip())


def cohens_kappa(labels_a: list, labels_b: list) -> float:
    """Compute Cohen's kappa for nominal agreement.

    κ = (p_o - p_e) / (1 - p_e)
    where p_o = observed agreement, p_e = expected agreement by chance.
    """
    if not labels_a or not labels_b or len(labels_a) != len(labels_b):
        return 0.0

    n = len(labels_a)
    categories = sorted(set(labels_a) | set(labels_b))

    if len(categories) <= 1:
        # All labels are the same — perfect agreement by definition
        return 1.0

    # Observed agreement
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p_o = agree / n

    # Expected agreement by chance
    p_e = 0.0
    for cat in categories:
        count_a = sum(1 for x in labels_a if x == cat)
        count_b = sum(1 for x in labels_b if x == cat)
        p_e += (count_a / n) * (count_b / n)

    if abs(1.0 - p_e) < 1e-10:
        return 1.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    return kappa


def weighted_kappa(scores_a: list[int], scores_b: list[int],
                   max_score: int = 5) -> float:
    """Compute linearly weighted kappa for ordinal scores (0-5).

    Uses linear weights: w_ij = 1 - |i - j| / max_score.
    """
    if not scores_a or not scores_b or len(scores_a) != len(scores_b):
        return 0.0

    n = len(scores_a)
    categories = list(range(max_score + 1))
    n_cat = len(categories)

    # Build observed confusion matrix
    observed = [[0] * n_cat for _ in range(n_cat)]
    for a, b in zip(scores_a, scores_b):
        a_clamped = max(0, min(max_score, a))
        b_clamped = max(0, min(max_score, b))
        observed[a_clamped][b_clamped] += 1

    # Marginals
    row_sums = [sum(observed[i]) for i in range(n_cat)]
    col_sums = [sum(observed[i][j] for i in range(n_cat)) for j in range(n_cat)]

    # Expected confusion matrix under independence
    expected = [[0.0] * n_cat for _ in range(n_cat)]
    for i in range(n_cat):
        for j in range(n_cat):
            expected[i][j] = (row_sums[i] * col_sums[j]) / n if n > 0 else 0

    # Weight matrix — linear weights
    weights = [[0.0] * n_cat for _ in range(n_cat)]
    for i in range(n_cat):
        for j in range(n_cat):
            weights[i][j] = abs(i - j) / max_score

    # Compute weighted observed and expected
    w_observed = sum(
        weights[i][j] * observed[i][j]
        for i in range(n_cat) for j in range(n_cat)
    ) / n if n > 0 else 0

    w_expected = sum(
        weights[i][j] * expected[i][j]
        for i in range(n_cat) for j in range(n_cat)
    ) / n if n > 0 else 0

    if abs(w_expected) < 1e-10:
        return 1.0

    kappa = 1.0 - (w_observed / w_expected)
    return kappa


def fleiss_kappa(ratings_matrix: list[list[int]]) -> dict:
    """Compute Fleiss' kappa for multi-rater agreement.

    Parameters
    ----------
    ratings_matrix : list of lists, shape (n_subjects, n_categories)
        Each row is one subject; each column is a category.
        Entry [i][j] = number of raters who assigned subject i to category j.

    Returns
    -------
    dict with kappa, p_observed, p_expected, interpretation, n_subjects, n_raters
    """
    if not ratings_matrix or not ratings_matrix[0]:
        return {"kappa": 0.0, "p_observed": 0.0, "p_expected": 0.0,
                "interpretation": "insufficient data",
                "n_subjects": 0, "n_raters": 0}

    n = len(ratings_matrix)          # number of subjects
    k = len(ratings_matrix[0])       # number of categories
    n_raters = sum(ratings_matrix[0])  # total raters per subject (assumed constant)

    if n < 2 or n_raters < 2:
        return {"kappa": 0.0, "p_observed": 0.0, "p_expected": 0.0,
                "interpretation": "insufficient data",
                "n_subjects": n, "n_raters": n_raters}

    # P_i for each subject: proportion of agreeing rater pairs
    p_i_list = []
    for row in ratings_matrix:
        sum_sq = sum(r * r for r in row)
        p_i = (sum_sq - n_raters) / (n_raters * (n_raters - 1)) if n_raters > 1 else 0
        p_i_list.append(p_i)

    p_bar = sum(p_i_list) / n  # mean observed agreement

    # p_j for each category: proportion of all assignments in that category
    p_e = 0.0
    total_assignments = n * n_raters
    for j in range(k):
        col_sum = sum(ratings_matrix[i][j] for i in range(n))
        p_j = col_sum / total_assignments
        p_e += p_j * p_j

    if abs(1.0 - p_e) < 1e-10:
        kappa = 1.0
    else:
        kappa = (p_bar - p_e) / (1.0 - p_e)

    # Interpretation (Landis & Koch 1977)
    if kappa > 0.80:
        interpretation = "almost perfect"
    elif kappa > 0.60:
        interpretation = "substantial"
    elif kappa > 0.40:
        interpretation = "moderate"
    elif kappa > 0.20:
        interpretation = "fair"
    elif kappa > 0.0:
        interpretation = "slight"
    else:
        interpretation = "poor"

    return {
        "kappa": round(kappa, 6),
        "p_observed": round(p_bar, 6),
        "p_expected": round(p_e, 6),
        "interpretation": interpretation,
        "n_subjects": n,
        "n_raters": n_raters,
    }


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard index for set overlap."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def pearson_correlation(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)

    denom = math.sqrt(var_x * var_y)
    if denom < 1e-10:
        return 0.0
    return cov / denom


def _interpret_kappa(k: float) -> str:
    """Interpret kappa value into agreement band."""
    if k < 0.2:
        return "poor"
    elif k < 0.4:
        return "fair"
    elif k < 0.6:
        return "moderate"
    elif k < 0.8:
        return "substantial"
    else:
        return "almost perfect"


def compute_agreement(results: list[dict], score_only: bool = False) -> dict:
    """Compute all agreement metrics from per-paper comparison results.

    Returns dict with kappa scores, Jaccard similarities, Pearson r, etc.

    When ``score_only`` is True (e.g. a cross-encoder second rater that
    cannot extract claims), the claim-Jaccard fields are reported as None
    and ``overall_agreement`` is the weighted kappa alone — blending in
    forced-zero Jaccards would otherwise understate true agreement.
    """
    if not results:
        return {
            "n_papers": 0,
            "score_kappa": 0.0,
            "score_weighted_kappa": 0.0,
            "score_mean_diff": 0.0,
            "score_correlation": 0.0,
            "claim_jaccard_findings": 0.0,
            "claim_jaccard_mechanisms": 0.0,
            "claim_jaccard_conditions": 0.0,
            "claim_jaccard_regions": 0.0,
            "overall_agreement": 0.0,
            "interpretation": "poor",
        }

    # Score agreement
    orig_scores = [r["original_score"] for r in results]
    second_scores = [r["second_score"] for r in results]

    # Bin scores for nominal kappa: low (0-2), mid (3-4), high (5)
    def _bin(s: int) -> str:
        if s <= 2:
            return "low"
        elif s <= 4:
            return "mid"
        return "high"

    orig_bins = [_bin(s) for s in orig_scores]
    second_bins = [_bin(s) for s in second_scores]

    score_kappa = cohens_kappa(orig_bins, second_bins)
    score_wk = weighted_kappa(orig_scores, second_scores)
    score_corr = pearson_correlation(
        [float(s) for s in orig_scores],
        [float(s) for s in second_scores],
    )
    score_mean_diff = sum(
        abs(a - b) for a, b in zip(orig_scores, second_scores)
    ) / len(results)

    # Claim Jaccard similarities
    jaccard_fields = {
        "claim_jaccard_findings": "findings",
        "claim_jaccard_mechanisms": "mechanisms",
        "claim_jaccard_conditions": "conditions",
        "claim_jaccard_regions": "brain_regions",
    }

    if score_only:
        # No claim extraction from this rater — report kappa-only agreement.
        overall = score_wk
        return {
            "n_papers": len(results),
            "score_only": True,
            "score_kappa": round(score_kappa, 4),
            "score_weighted_kappa": round(score_wk, 4),
            "score_mean_diff": round(score_mean_diff, 4),
            "score_correlation": round(score_corr, 4),
            **{k: None for k in jaccard_fields},
            "overall_agreement": round(overall, 4),
            "interpretation": _interpret_kappa(overall),
        }

    jaccards = {}
    for metric_key, field_name in jaccard_fields.items():
        sims = []
        for r in results:
            orig_set = {normalize_label(x) for x in r.get(f"original_{field_name}", [])}
            second_set = {normalize_label(x) for x in r.get(f"second_{field_name}", [])}
            sims.append(jaccard_similarity(orig_set, second_set))
        jaccards[metric_key] = sum(sims) / len(sims) if sims else 0.0

    # Overall agreement: weighted average of weighted kappa and mean Jaccard
    mean_jaccard = sum(jaccards.values()) / len(jaccards) if jaccards else 0.0
    overall = 0.5 * score_wk + 0.5 * mean_jaccard

    return {
        "n_papers": len(results),
        "score_only": False,
        "score_kappa": round(score_kappa, 4),
        "score_weighted_kappa": round(score_wk, 4),
        "score_mean_diff": round(score_mean_diff, 4),
        "score_correlation": round(score_corr, 4),
        **{k: round(v, 4) for k, v in jaccards.items()},
        "overall_agreement": round(overall, 4),
        "interpretation": _interpret_kappa(overall),
    }


# ── Markdown report ──────────────────────────────────────────────────────

def generate_report(summary: dict, results: list[dict]) -> str:
    """Generate a markdown report with tables and interpretation."""
    lines = [
        "# Inter-Rater Reliability Report",
        "",
        "## Summary",
        "",
        f"- **Papers evaluated**: {summary.get('n_papers', 0)}",
        f"- **Overall agreement**: {summary.get('overall_agreement', 0):.4f}"
        f" ({summary.get('interpretation', 'N/A')})",
        "",
        "## Score Agreement",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Cohen's κ (binned) | {summary.get('score_kappa', 0):.4f} |",
        f"| Weighted κ (linear, 0-5) | {summary.get('score_weighted_kappa', 0):.4f} |",
        f"| Mean absolute difference | {summary.get('score_mean_diff', 0):.4f} |",
        f"| Pearson correlation | {summary.get('score_correlation', 0):.4f} |",
        "",
    ]

    if summary.get("score_only"):
        lines.extend([
            "## Claim Extraction Agreement",
            "",
            "_Not applicable — this rater scores relevance only "
            "(cross-encoder) and does not extract claims._",
            "",
        ])
    else:
        lines.extend([
            "## Claim Extraction Agreement (Jaccard Similarity)",
            "",
            "| Field | Jaccard |",
            "|-------|---------|",
            f"| Findings | {summary.get('claim_jaccard_findings', 0):.4f} |",
            f"| Mechanisms | {summary.get('claim_jaccard_mechanisms', 0):.4f} |",
            f"| Conditions | {summary.get('claim_jaccard_conditions', 0):.4f} |",
            f"| Brain regions | {summary.get('claim_jaccard_regions', 0):.4f} |",
            "",
        ])

    lines.extend([
        "## Interpretation",
        "",
        f"Overall agreement is **{summary.get('interpretation', 'N/A')}**"
        f" (κ = {summary.get('overall_agreement', 0):.4f}).",
        "",
    ])

    # Kappa interpretation bands
    lines.extend([
        "| Range | Interpretation |",
        "|-------|---------------|",
        "| < 0.20 | Poor |",
        "| 0.20 – 0.40 | Fair |",
        "| 0.40 – 0.60 | Moderate |",
        "| 0.60 – 0.80 | Substantial |",
        "| > 0.80 | Almost perfect |",
        "",
    ])

    # Per-paper detail table
    if results:
        lines.extend([
            "## Per-Paper Detail",
            "",
            "| # | Title | Original | Second | Δ |",
            "|---|-------|----------|--------|---|",
        ])
        for i, r in enumerate(results, 1):
            title = r.get("title", "?")[:50]
            orig = r.get("original_score", "?")
            sec = r.get("second_score", "?")
            diff = abs(int(orig) - int(sec)) if isinstance(orig, int) and isinstance(sec, int) else "?"
            lines.append(f"| {i} | {title} | {orig} | {sec} | {diff} |")
        lines.append("")

    return "\n".join(lines)


# ── Main entry point ─────────────────────────────────────────────────────

def run(
    claims_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    config: Any = None,
    second_provider: str = "ollama",
    second_model: Optional[str] = None,
    sample_size: int = 100,
    seed: int = 42,
    cancel_event: Any = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kw,
) -> dict:
    """Run inter-rater reliability analysis.

    1. Load claims
    2. Stratified sample of papers
    3. Re-score each with second LLM provider
    4. Re-extract claims with second LLM
    5. Compute agreement metrics
    6. Generate report
    7. Save results + summary + report
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

    # ── Resolve paths ────────────────────────────────────────────────
    if config is not None:
        project_root = config.project_root
    else:
        project_root = Path.cwd()

    data_dir = project_root / "data"
    validation_dir = data_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path(output_dir) if output_dir else project_root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = CheckpointManager(validation_dir, "interrater_checkpoint.json")
    results_path = validation_dir / "interrater_results.json"
    summary_path = validation_dir / "interrater_summary.json"
    report_path = out_dir / "interrater_report.md"

    # ── Load claims corpus ───────────────────────────────────────────
    if claims_path:
        corpus_file = Path(claims_path)
    else:
        # Prefer filtered, then base claims, then scored relevant
        candidates = [
            data_dir / "claims" / "claims_filtered.json",
            data_dir / "claims" / "claims.json",
            data_dir / "corpus" / "relevant.json",
        ]
        corpus_file = None
        for c in candidates:
            if c.exists():
                corpus_file = c
                break
        if corpus_file is None:
            _progress("ERROR: No claims corpus found. Run prior stages first.")
            return {"error": "no_claims_corpus"}

    _progress(f"Loading corpus from {corpus_file}")
    with open(corpus_file, encoding="utf-8") as f:
        papers = json.load(f)

    if not papers:
        _progress("ERROR: Corpus is empty.")
        return {"error": "empty_corpus"}

    _progress(f"Corpus loaded: {len(papers)} papers")

    # ── Determine model name ─────────────────────────────────────────
    model_name = second_model
    if not model_name and config is not None:
        model_name = config.llm_model
    if not model_name:
        model_name = "default"

    _progress(f"Second rater: provider={second_provider}, model={model_name}")

    # ── Cross-encoder (brt_bert / brt_irr) setup ─────────────────────
    use_bert = second_provider in EMBEDDING_SECOND_RATERS
    bert_hypothesis = ""
    bert_model_path = ""
    if use_bert:
        from process import relevance_embedder as embedder

        # second_model, if given, overrides the preset checkpoint path.
        bert_model_path = (second_model or "").strip()
        backend = "embedding" if bert_model_path else second_provider
        if not embedder.model_available(backend, bert_model_path):
            _progress(
                f"ERROR: {second_provider} relevance model not found. "
                "brt-bert expects brt-bert-relevance/final; "
                "brt-irr expects brt-bert-irr-buddy-mlm/relevance/final. "
                "Pass a custom model path/HF id in the Model field to override."
            )
            return {"error": "embedding_model_unavailable"}

        # Use the SAME question as the primary (Stage 2) scorer so IRR
        # measures agreement on the same target.  config.hypothesis_text is
        # only a fallback because get_natural_language_question() reads the
        # active research_config that the GUI populates.
        if config is not None:
            getter = getattr(config, "get_natural_language_question", None)
            if callable(getter):
                bert_hypothesis = (getter() or "").strip()
            if not bert_hypothesis:
                bert_hypothesis = (getattr(config, "hypothesis_text", "") or "").strip()
        if not bert_hypothesis:
            _progress(
                "WARNING: no research question found in config "
                "(natural_language_question / hypothesis_text both empty). "
                "Cross-encoder will return 0 for every paper. "
                "Set the research question in the Research config tab."
            )

        rewritten = embedder.question_to_claim(bert_hypothesis) if bert_hypothesis else ""
        _progress(
            f"{second_provider} backend={backend} | score-only IRR | "
            f"hypothesis chars={len(bert_hypothesis)} -> "
            f"query={rewritten[:80]!r}"
        )

    # ── Stratified sample ────────────────────────────────────────────
    sample = stratified_sample(papers, sample_size, seed=seed)
    _progress(f"Sampled {len(sample)} papers (requested {sample_size})")

    # ── Load checkpoint for resume ───────────────────────────────────
    completed_results, start_idx = ckpt.load()
    completed_titles = {r.get("title") for r in completed_results}

    if start_idx > 0:
        _progress(f"Resuming from checkpoint: {start_idx} already completed")

    # ── Re-score and re-extract each paper ───────────────────────────
    results = list(completed_results)
    total = len(sample)
    checkpoint_every = 10

    for i, paper in enumerate(sample):
        if _cancelled():
            _progress("Cancelled by user.")
            if not ckpt.force_save(results, total):
                _progress("WARNING: Could not save interrater checkpoint.")
            return {"cancelled": True, "partial_results": len(results)}

        title = paper.get("title", "Unknown")

        # Skip already-completed papers
        if title in completed_titles:
            continue

        original_score = paper.get("relevance_score", 0)

        _progress(f"[{i + 1}/{total}] Re-scoring: {title[:60]}")

        # Re-score with second rater
        try:
            if use_bert:
                backend = "embedding" if bert_model_path else second_provider
                score_result = embedder.score_relevance_embedding(
                    paper,
                    hypothesis=bert_hypothesis,
                    backend=backend,
                    model_path=bert_model_path,
                    device="auto",
                )
            else:
                score_result = provider_score_relevance(model_name, paper)
            second_score = score_result.get("relevance_score", 0)
        except Exception as exc:
            logger.error("Score failed for '%s': %s", title[:60], exc)
            second_score = 0

        if use_bert:
            # Show paper-text length so an all-zero run is diagnosable:
            # 0 chars means the corpus record is missing title/abstract/
            # full_text, not that the model is broken.
            doc_chars = len(embedder._paper_text(paper))
            _progress(
                f"[{i + 1}/{total}] Re-scoring: {title[:60]} "
                f"→ original={original_score}, second={second_score} "
                f"(doc={doc_chars} chars)"
            )
        else:
            _progress(
                f"[{i + 1}/{total}] Re-scoring: {title[:60]} "
                f"→ original={original_score}, second={second_score}"
            )

        # Re-extract claims with second rater.  Cross-encoders cannot
        # generate structured claims, so brt_bert runs score-only.
        if use_bert:
            extract_result = {}
        else:
            try:
                extract_result = provider_extract_claims(model_name, paper)
            except Exception as exc:
                logger.error("Extract failed for '%s': %s", title[:60], exc)
                extract_result = {}

        # Gather original claim fields
        orig_findings = paper.get("findings", [])
        orig_mechanisms = paper.get("mechanisms", [])
        orig_conditions = paper.get("conditions", [])
        orig_regions = paper.get("brain_regions", [])

        record = {
            "title": title,
            "original_score": original_score,
            "second_score": second_score,
            "original_findings": orig_findings,
            "second_findings": extract_result.get("findings", []),
            "original_mechanisms": orig_mechanisms,
            "second_mechanisms": extract_result.get("mechanisms", []),
            "original_conditions": orig_conditions,
            "second_conditions": extract_result.get("conditions", []),
            "original_brain_regions": orig_regions,
            "second_brain_regions": extract_result.get("brain_regions", []),
        }
        results.append(record)
        completed_titles.add(title)

        # Checkpoint every N papers
        if (len(results) - start_idx) % checkpoint_every == 0:
            if ckpt.force_save(results, total):
                _progress(f"Checkpoint saved: {len(results)} results")
            else:
                _progress("WARNING: Could not save interrater checkpoint.")

    if _cancelled():
        _progress("Cancelled by user.")
        if not ckpt.force_save(results, total):
            _progress("WARNING: Could not save interrater checkpoint.")
        return {"cancelled": True, "partial_results": len(results)}

    # ── Compute agreement metrics ────────────────────────────────────
    _progress(f"Computing agreement metrics for {len(results)} papers…")
    summary = compute_agreement(results, score_only=use_bert)
    _progress(
        f"Agreement: κ={summary['score_kappa']:.3f}, "
        f"wκ={summary['score_weighted_kappa']:.3f}, "
        f"r={summary['score_correlation']:.3f}, "
        f"overall={summary['overall_agreement']:.3f} "
        f"({summary['interpretation']})"
    )

    # ── Generate report ──────────────────────────────────────────────
    report_text = generate_report(summary, results)

    # ── Save outputs ─────────────────────────────────────────────────
    atomic_write_json(results_path, results)
    _progress(f"Results saved: {results_path}")

    atomic_write_json(summary_path, summary)
    _progress(f"Summary saved: {summary_path}")

    atomic_write_text(report_path, report_text)
    _progress(f"Report saved: {report_path}")

    ckpt.cleanup()

    _progress("Stage 8 complete.")

    return {
        "n_papers": summary["n_papers"],
        "score_kappa": summary["score_kappa"],
        "score_weighted_kappa": summary["score_weighted_kappa"],
        "score_correlation": summary["score_correlation"],
        "overall_agreement": summary["overall_agreement"],
        "interpretation": summary["interpretation"],
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
    }
