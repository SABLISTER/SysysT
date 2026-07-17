"""Assemble preview text for GUI query/prompt inspection."""

from __future__ import annotations

import json
from pathlib import Path

from process.claim_extractor import build_extraction_prompt, _claim_think_mode
from process.method_review import extraction_prompt as method_review_prompt
from process.relevance_filter import (
    BATCH_RELEVANCE_PROMPT,
    RELEVANCE_PROMPT,
    _batch_max_tokens,
    _build_relevance_batches,
    _paper_text_for_scoring,
    _score_batch_payload,
    _stage2_think_mode,
)
from process.synthesizer import (
    ABSTRACT_PROMPT,
    NARRATIVE_PROMPT,
    _synthesis_think_mode,
    cluster_claims,
)
from process.queries import (
    get_natural_language_query,
    get_openalex_filters,
    get_pubmed_queries,
    get_scopus_queries,
    get_s2_queries,
    get_wos_queries,
)
from stages.s8_interrater import stratified_sample


def build_search_preview_sections(
    research: dict | None,
    natural_language_question: str,
    semantic_terms: list[str],
    boolean_terms: list[str],
) -> list[tuple[str, str]]:
    research = research or {}
    sections: list[tuple[str, str]] = []

    gui_lines = [
        "Current GUI search inputs",
        "=========================",
        "",
        f"Elicit / natural-language query: {natural_language_question or '(blank)'}",
        "",
        "Semantic Scholar queries:",
    ]
    if semantic_terms:
        gui_lines.extend(f"{idx}. {query}" for idx, query in enumerate(semantic_terms, start=1))
    else:
        gui_lines.append("(none)")
    gui_lines += [
        "",
        "OpenAlex searches:",
    ]
    if semantic_terms:
        gui_lines.extend(f"{idx}. {query}" for idx, query in enumerate(semantic_terms, start=1))
    else:
        gui_lines.append("(none)")
    gui_lines += [
        "",
        "PubMed queries:",
    ]
    if boolean_terms:
        gui_lines.extend(f"{idx}. {query}" for idx, query in enumerate(boolean_terms, start=1))
    else:
        gui_lines.append("(none)")
    gui_lines += [
        "",
        "Web of Science queries:",
    ]
    if boolean_terms:
        gui_lines.extend(f"{idx}. {query}" for idx, query in enumerate(boolean_terms, start=1))
    else:
        gui_lines.append("(none)")
    gui_lines += [
        "",
        "Scopus queries:",
    ]
    if boolean_terms:
        gui_lines.extend(f"{idx}. {query}" for idx, query in enumerate(boolean_terms, start=1))
    else:
        gui_lines.append("(none)")
    sections.append(("GUI Search", "\n".join(gui_lines)))

    search_cfg = research.get("search") or {}
    config_query = get_natural_language_query(research)
    config_lines = [
        "Structured config-driven Stage 1 search",
        "=====================================",
        "",
        f"Elicit / natural-language query: {config_query or '(none configured)'}",
        "",
        "Semantic Scholar queries:",
    ]
    s2_queries = get_s2_queries(search_cfg)
    if s2_queries:
        config_lines.extend(f"{idx}. {query}" for idx, query in enumerate(s2_queries, start=1))
    else:
        config_lines.append("(none)")
    config_lines += [
        "",
        "OpenAlex searches:",
    ]
    oa_filters = get_openalex_filters(search_cfg)
    if oa_filters:
        config_lines.extend(
            f"{idx}. {row.get('search', '')}" for idx, row in enumerate(oa_filters, start=1)
        )
    else:
        config_lines.append("(none)")
    config_lines += [
        "",
        "PubMed queries:",
    ]
    pm_queries = get_pubmed_queries(search_cfg)
    if pm_queries:
        config_lines.extend(f"{idx}. {query}" for idx, query in enumerate(pm_queries, start=1))
    else:
        config_lines.append("(none)")
    config_lines += [
        "",
        "Web of Science queries:",
    ]
    wos_queries = get_wos_queries(search_cfg)
    if wos_queries:
        config_lines.extend(f"{idx}. {query}" for idx, query in enumerate(wos_queries, start=1))
    else:
        config_lines.append("(none)")
    config_lines += [
        "",
        "Scopus queries:",
    ]
    scopus_queries = get_scopus_queries(search_cfg)
    if scopus_queries:
        config_lines.extend(f"{idx}. {query}" for idx, query in enumerate(scopus_queries, start=1))
    else:
        config_lines.append("(none)")
    sections.append(("Config Search", "\n".join(config_lines)))

    return sections


def build_score_preview_sections(
    config,
    corpus_path: Path,
    workload: str,
    batch_size: int,
) -> list[tuple[str, str]]:
    papers = _load_json_list(corpus_path)
    provider = config.llm_provider_for(workload)
    model = config.llm_model_for(workload)

    if not papers:
        return [("Stage 2 Prompt", f"No papers found in {corpus_path}")]

    effective_batch_size = max(1, int(batch_size or 1))
    if effective_batch_size > 1:
        batches = _build_relevance_batches(
            papers,
            0,
            effective_batch_size,
            abstract_chars=config.score_abstract_chars,
            fulltext_chars=config.score_fulltext_chars,
        )
        indices = batches[0] if batches else [0]
        batch_papers = [papers[idx] for idx in indices]
        payload = _score_batch_payload(
            batch_papers,
            abstract_chars=config.score_abstract_chars,
            fulltext_chars=config.score_fulltext_chars,
        )
        paper_blocks = []
        for item in payload:
            paper_blocks.append(
                "\n".join(
                    [
                        f"ID: {item['id']}",
                        f"TITLE: {item['title']}",
                        f"TEXT SOURCE: {item['text_source']}",
                        f"PAPER TEXT: {item['paper_text']}",
                        "END PAPER",
                    ]
                )
            )
        prompt = BATCH_RELEVANCE_PROMPT.format(paper_blocks="\n\n".join(paper_blocks))
        meta = "\n".join(
            [
                "Stage 2 batched relevance prompt preview",
                "=======================================",
                f"Provider: {provider}",
                f"Model: {model}",
                f"Workload: {workload}",
                f"Think mode: {_stage2_think_mode(model, enabled=bool(config.score_use_thinking))!r}",
                f"Input file: {corpus_path}",
                f"Batch size requested: {effective_batch_size}",
                f"Batch size used in preview: {len(batch_papers)}",
                f"Max output tokens: {_batch_max_tokens(len(batch_papers), max_tokens=config.score_batch_max_tokens)}",
                f"Abstract chars: {config.score_abstract_chars}",
                f"Full-text chars: {config.score_fulltext_chars}",
                f"Sample titles: {', '.join((paper.get('title') or '?')[:80] for paper in batch_papers)}",
                "",
                prompt,
            ]
        )
        return [("Stage 2 Prompt", meta)]

    paper = papers[0]
    paper_text, text_source = _paper_text_for_scoring(
        paper,
        abstract_chars=config.score_abstract_chars,
        fulltext_chars=config.score_fulltext_chars,
    )
    prompt = RELEVANCE_PROMPT.format(
        title=paper.get("title", ""),
        text_source=text_source,
        paper_text=paper_text,
    )
    meta = "\n".join(
        [
            "Stage 2 single-paper relevance prompt preview",
            "============================================",
            f"Provider: {provider}",
            f"Model: {model}",
            f"Workload: {workload}",
            f"Think mode: {_stage2_think_mode(model, enabled=bool(config.score_use_thinking))!r}",
            f"Max output tokens: {config.score_max_tokens}",
            f"Abstract chars: {config.score_abstract_chars}",
            f"Full-text chars: {config.score_fulltext_chars}",
            f"Input file: {corpus_path}",
            f"Sample paper: {paper.get('title', '(untitled)')}",
            f"Text source: {text_source}",
            "",
            prompt,
        ]
    )
    return [("Stage 2 Prompt", meta)]


def build_extract_preview_sections(config, relevant_path: Path) -> list[tuple[str, str]]:
    papers = _load_json_list(relevant_path)
    if not papers:
        return [("Stage 3 Prompt", f"No papers found in {relevant_path}")]

    paper = papers[0]
    model = config.llm_model_for("abstract")
    prompt = build_extraction_prompt(
        paper.get("title", ""),
        paper.get("abstract", ""),
        getattr(config, "research_config", None),
    )
    body = "\n".join(
        [
            "Stage 3 claim extraction prompt preview",
            "======================================",
            f"Provider: {config.llm_provider_for('abstract')}",
            f"Model: {model}",
            f"Think mode: {_claim_think_mode(model) if config.claim_use_thinking else None!r}",
            f"Max output tokens: {config.claim_max_tokens}",
            f"Input file: {relevant_path}",
            f"Sample paper: {paper.get('title', '(untitled)')}",
            "",
            prompt,
        ]
    )
    return [("Stage 3 Prompt", body)]


def build_validate_preview_sections(
    config,
    relevant_path: Path,
    model: str | None,
) -> list[tuple[str, str]]:
    papers = _load_json_list(relevant_path)
    if not papers:
        return [("Stage 4 Prompt", f"No papers found in {relevant_path}")]

    paper = papers[0]
    chosen_provider = getattr(config, "llm_provider", "") or "anthropic"
    chosen_model = (model or "").strip() or config.llm_model
    prompt = method_review_prompt(
        paper.get("title", ""),
        paper.get("abstract", ""),
    )
    body = "\n".join(
        [
            "Stage 4 method-review prompt preview",
            "===================================",
            f"Provider: {chosen_provider}",
            f"Model: {chosen_model}",
            f"Input file: {relevant_path}",
            f"Sample paper: {paper.get('title', '(untitled)')}",
            "",
            prompt,
        ]
    )
    return [("Stage 4 Prompt", body)]


def build_synthesize_preview_sections(
    config,
    claims_path: Path,
    n_clusters: int,
    embedding_model: str,
) -> list[tuple[str, str]]:
    papers = _load_json_list(claims_path)
    if not papers:
        return [("Stage 6 Prompt", f"No papers found in {claims_path}")]

    sections: list[tuple[str, str]] = []
    model = config.llm_model_for("abstract")
    try:
        clusters = cluster_claims(
            papers,
            n_clusters=n_clusters,
            embedding_model=embedding_model,
        )
    except Exception as e:
        sections.append(
            (
                "Stage 6 Narrative Prompt",
                "\n".join(
                    [
                        "Could not build an exact narrative prompt preview.",
                        f"Claims file: {claims_path}",
                        f"Embedding model: {embedding_model}",
                        f"Reason: {e}",
                    ]
                ),
            )
        )
        clusters = []

    if clusters:
        cluster = clusters[0]
        summaries = []
        for paper in cluster["papers"][:10]:
            claims = paper.get("claims", {})
            findings = "; ".join((claims.get("findings") or [])[:2])
            summaries.append(f"- {paper.get('title', '?')} ({paper.get('year', '?')}): {findings}")
        support_count = cluster.get("primary_support_count", cluster.get("criticality_count", 0))
        support_pct = round(100 * support_count / cluster["paper_count"]) if cluster["paper_count"] else 0
        prompt = NARRATIVE_PROMPT.format(
            research_question=get_natural_language_query(getattr(config, "research_config", None)),
            paper_count=cluster["paper_count"],
            mechanisms=", ".join((cluster.get("top_mechanisms") or cluster.get("mechanisms") or [])[:5]) or "various",
            conditions=", ".join((cluster.get("top_conditions") or cluster.get("conditions") or [])[:5]) or "various",
            regions=", ".join((cluster.get("top_regions") or cluster.get("brain_regions") or [])[:5]) or "various",
            primary_support_pct=support_pct,
            paper_summaries="\n".join(summaries),
        )
        sections.append(
            (
                "Narrative Prompt",
                "\n".join(
                    [
                        "Stage 6 cluster narrative prompt preview",
                        "=======================================",
                        f"Provider: {config.llm_provider_for('abstract')}",
                        f"Model: {model}",
                        f"Think mode: {_synthesis_think_mode(model, enabled=bool(config.synthesis_use_thinking))!r}",
                        f"Max output tokens: {config.synthesis_cluster_max_tokens}",
                        f"Input file: {claims_path}",
                        f"Embedding model: {embedding_model}",
                        f"Preview cluster size: {cluster['paper_count']}",
                        "",
                        prompt,
                    ]
                ),
            )
        )

    narratives = _load_json_list(config.clusters_dir / "narratives.json")
    if narratives:
        cluster_summaries = []
        for item in narratives:
            cluster_summaries.append(
                f"[{item.get('theme', 'Theme')} -- {item.get('paper_count', 0)} papers]: {item.get('narrative', '')}"
            )
        prompt = ABSTRACT_PROMPT.format(
            research_question=get_natural_language_query(getattr(config, "research_config", None)),
            total_papers=len(papers),
            cluster_summaries="\n\n".join(cluster_summaries),
        )
        sections.append(
            (
                "Abstract Prompt",
                "\n".join(
                    [
                        "Stage 6 abstract prompt preview",
                        "==============================",
                        f"Provider: {config.llm_provider_for('abstract')}",
                        f"Model: {model}",
                        f"Think mode: {_synthesis_think_mode(model, enabled=bool(config.synthesis_use_thinking))!r}",
                        f"Max output tokens: {config.synthesis_abstract_max_tokens}",
                        f"Input narratives: {config.clusters_dir / 'narratives.json'}",
                        "",
                        prompt,
                    ]
                ),
            )
        )
    else:
        sections.append(
            (
                "Abstract Prompt",
                "\n".join(
                    [
                        "Exact abstract prompt preview is not available yet.",
                        "The abstract prompt is assembled from generated cluster narratives,",
                        "so it can only be previewed after narratives.json exists from a prior run.",
                    ]
                ),
            )
        )

    return sections


def build_interrater_preview_sections(
    config,
    claims_path: Path,
    sample_size: int,
    provider: str,
    model_name: str,
) -> list[tuple[str, str]]:
    papers = _load_json_list(claims_path)
    if not papers:
        return [("Stage 8 Prompt", f"No papers found in {claims_path}")]

    sample = stratified_sample(papers, max(1, int(sample_size or 1)))
    paper = sample[0] if sample else papers[0]
    paper_text, text_source = _paper_text_for_scoring(paper)

    score_prompt = RELEVANCE_PROMPT.format(
        hypothesis=get_natural_language_query(getattr(config, "research_config", None)),
        title=paper.get("title", ""),
        text_source=text_source,
        paper_text=paper_text,
    )
    extract_prompt = build_extraction_prompt(
        paper.get("title", ""),
        paper.get("abstract", ""),
        getattr(config, "research_config", None),
    )

    metadata = [
        f"Second-rater provider: {provider}",
        f"Second-rater model: {model_name}",
        f"Input file: {claims_path}",
        f"Requested sample size: {sample_size}",
        f"Preview paper: {paper.get('title', '(untitled)')}",
    ]
    return [
        (
            "Relevance Prompt",
            "\n".join(
                [
                    "Stage 8 second-rater relevance prompt preview",
                    "============================================",
                    *metadata,
                    "",
                    score_prompt,
                ]
            ),
        ),
        (
            "Extraction Prompt",
            "\n".join(
                [
                    "Stage 8 second-rater claim extraction prompt preview",
                    "===================================================",
                    *metadata,
                    "",
                    extract_prompt,
                ]
            ),
        ),
    ]


def _load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    return data if isinstance(data, list) else []
