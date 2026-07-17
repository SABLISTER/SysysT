"""Per-study reporting and provenance snapshots for pipeline runs."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from acquire.fulltext import download_record_status
from core.io import atomic_write_json, atomic_write_text
from process.queries import get_natural_language_query

STAGE_LABELS = {
    "search": "Search",
    "config search": "Search",
    "score": "Score",
    "pull full texts": "Pull Full Texts (2b)",
    "pull full texts (2b)": "Pull Full Texts (2b)",
    "extract": "Extract",
    "validate": "Validate",
    "audit": "Audit",
    "synthesize": "Synthesize",
    "fulltext": "Fulltext",
    "interrater": "Interrater",
    "robustness": "Robustness",
    "verify": "Verify",
    "compare runs": "Compare Runs",
    "compare": "Compare Runs",
}

STAGE_ORDER = [
    "Search",
    "Score",
    "Pull Full Texts (2b)",
    "Extract",
    "Validate",
    "Audit",
    "Synthesize",
    "Fulltext",
    "Interrater",
    "Robustness",
    "Verify",
    "Compare Runs",
]


def record_study_stage_event(
    config,
    research: dict | None,
    config_path: Path | None,
    stage: str,
    status: str,
    *,
    source: str,
    result=None,
    error: str | None = None,
    started_at: float | None = None,
    finished_at: float | None = None,
) -> dict:
    paths = study_report_paths(config, research, config_path)
    paths["report_dir"].mkdir(parents=True, exist_ok=True)

    snapshot = collect_study_snapshot(config, research, config_path, paths)
    atomic_write_json(paths["config_snapshot"], snapshot["study"]["config_snapshot"])

    event = {
        "timestamp_utc": _iso_now() if finished_at is None else _iso_from_epoch(finished_at),
        "status": status,
        "stage": canonical_stage_name(stage),
        "raw_stage": stage,
        "source": source,
    }
    if started_at is not None:
        event["started_at_utc"] = _iso_from_epoch(started_at)
    if finished_at is not None:
        event["finished_at_utc"] = _iso_from_epoch(finished_at)
    if started_at is not None and finished_at is not None:
        event["duration_sec"] = round(max(0.0, finished_at - started_at), 2)
    summary = summarize_stage_result(stage, result)
    if summary:
        event["summary"] = summary
    if error:
        event["error"] = error

    _append_jsonl(paths["run_log"], event)
    events = load_stage_events(paths["run_log"])

    summary_doc = {
        **snapshot,
        "latest_event": event,
        "event_count": len(events),
    }
    atomic_write_json(paths["summary"], summary_doc)
    atomic_write_text(paths["report"], render_study_report(summary_doc, events))
    return {
        "report_dir": str(paths["report_dir"]),
        "report_path": str(paths["report"]),
        "summary_path": str(paths["summary"]),
        "run_log_path": str(paths["run_log"]),
    }


def study_report_paths(config, research: dict | None, config_path: Path | None) -> dict[str, Path]:
    identity = study_identity(research, config_path)
    report_dir = config.output_dir / "study_reports" / identity["study_id"]
    return {
        "report_dir": report_dir,
        "report": report_dir / "study_report.md",
        "summary": report_dir / "latest_summary.json",
        "run_log": report_dir / "run_log.jsonl",
        "config_snapshot": report_dir / "config_snapshot.json",
    }


def study_identity(research: dict | None, config_path: Path | None) -> dict[str, str]:
    research = research or {}
    config_json = json.dumps(research, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha1(config_json.encode("utf-8")).hexdigest()[:10]

    question = get_natural_language_query(research)
    study_name = (
        str(research.get("name") or "").strip()
        or question
        or (config_path.stem if config_path else "study")
    )
    slug = _slugify(study_name)[:64] or "study"
    return {
        "study_id": f"{slug}-{digest}",
        "study_name": study_name,
        "question": question,
        "config_source": str(config_path) if config_path else "",
    }


def collect_study_snapshot(config, research: dict | None, config_path: Path | None, paths: dict[str, Path] | None = None) -> dict:
    research = research or {}
    paths = paths or study_report_paths(config, research, config_path)
    ident = study_identity(research, config_path)

    corpus = _load_json_list(config.corpus_dir / "corpus.json")
    relevant = _load_json_list(config.claims_dir / "relevant.json")
    claims = _load_json_list(config.claims_dir / "claims.json")
    filtered = _load_json_list(config.claims_dir / "claims_filtered.json")
    removal_log = _load_json_list(config.claims_dir / "removal_log.json")
    method_summary = _load_json_dict(config.data_dir / "method_review" / "method_type_summary.json")

    fulltext_corpus = _load_json_list(config.corpus_dir / "corpus_fulltext.json")
    fulltext_corpus_log = _load_json_list(config.data_dir / "fulltext_corpus" / "download_log.json")
    fulltext_stage_log = _load_json_list(config.data_dir / "fulltext" / "download_log.json")
    oa_stage = _load_json_list(config.data_dir / "fulltext" / "oa_status.json")
    metrics = _load_json_list(config.data_dir / "fulltext" / "auto_extracted_metrics.json")

    clusters = _load_json_list(config.clusters_dir / "clusters.json")
    narratives = _load_json_list(config.clusters_dir / "narratives.json")
    abstract_text = _read_text(config.output_dir / "abstract_draft.md")

    interrater = _load_json_dict(config.data_dir / "validation" / "interrater_summary.json")
    robustness = _load_json_dict(config.data_dir / "validation" / "robustness_results.json")
    verification = _load_json_dict(config.data_dir / "validation" / "verification_results.json")

    criticality_count = sum(
        1 for paper in claims
        if (paper.get("claims") or {}).get("supports_criticality") is True
    )
    included_records = len(filtered) if filtered else len(claims)
    search_count = len(corpus)
    screened_count = search_count
    relevant_count = len(relevant)
    excluded_count = max(screened_count - relevant_count, 0)

    method_type_counts = method_summary.get("method_type_counts") or {}
    top_method_types = sorted(
        method_type_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[:3]

    fulltext_corpus_status = _count_download_statuses(fulltext_corpus_log)
    fulltext_stage_status = _count_download_statuses(fulltext_stage_log)
    enriched_fulltext_count = sum(1 for paper in fulltext_corpus if paper.get("full_text"))
    oa_available = sum(1 for row in oa_stage if row.get("is_oa") and row.get("oa_url"))

    top_themes = []
    for item in narratives[:3]:
        if not isinstance(item, dict):
            continue
        top_themes.append({
            "theme": str(item.get("theme") or "Theme").strip() or "Theme",
            "paper_count": int(item.get("paper_count") or 0),
            "narrative": str(item.get("narrative") or "").strip(),
        })

    stage_rows = [
        {
            "stage": "Search",
            "status": "Complete" if search_count else "Not run",
            "stats": f"{search_count} papers in corpus" if search_count else "No corpus yet",
            "artifact": _display_path(config, config.corpus_dir / "corpus.json"),
        },
        {
            "stage": "Score",
            "status": "Complete" if relevant_count else "Not run",
            "stats": (
                f"{relevant_count}/{screened_count} retained"
                f" ({_pct(relevant_count, screened_count):.1f}%) at threshold >= {config.relevance_threshold}"
                if relevant_count else "No relevant set yet"
            ),
            "artifact": _display_path(config, config.claims_dir / "relevant.json"),
        },
        {
            "stage": "Pull Full Texts (2b)",
            "status": "Complete" if fulltext_corpus else "Partial" if fulltext_corpus_log else "Not run",
            "stats": (
                f"{enriched_fulltext_count} enriched; "
                f"{fulltext_corpus_status['downloaded']} downloaded, "
                f"{fulltext_corpus_status['no_usable_text']} no usable text, "
                f"{fulltext_corpus_status['error']} errors"
            ),
            "artifact": _display_path(config, config.corpus_dir / "corpus_fulltext.json"),
        },
        {
            "stage": "Extract",
            "status": "Complete" if claims else "Not run",
            "stats": (
                f"{len(claims)} extracted; {criticality_count} support criticality"
                f" ({_pct(criticality_count, len(claims)):.1f}%)"
                if claims else "No structured claims yet"
            ),
            "artifact": _display_path(config, config.claims_dir / "claims.json"),
        },
        {
            "stage": "Validate",
            "status": "Complete" if method_summary else "Not run",
            "stats": (
                f"{method_summary.get('total_records', 0)} reviewed; "
                f"{method_summary.get('manual_review_pct', 0)}% manual review"
                if method_summary else "No method review summary yet"
            ),
            "artifact": _display_path(config, config.data_dir / "method_review" / "method_type_summary.json"),
        },
        {
            "stage": "Audit",
            "status": "Complete" if filtered else "Not run",
            "stats": (
                f"{included_records} included, {len(removal_log)} removed"
                if filtered else "No audited claims set yet"
            ),
            "artifact": _display_path(config, config.claims_dir / "claims_filtered.json"),
        },
        {
            "stage": "Synthesize",
            "status": "Complete" if abstract_text else "Partial" if clusters or narratives else "Not run",
            "stats": (
                f"{len(clusters)} clusters, {len(narratives)} narratives, abstract draft ready"
                if abstract_text else
                f"{len(clusters)} clusters, {len(narratives)} narratives"
                if clusters or narratives else "No synthesis outputs yet"
            ),
            "artifact": _display_path(config, config.output_dir / "abstract_draft.md"),
        },
        {
            "stage": "Fulltext",
            "status": "Complete" if metrics or fulltext_stage_log else "Not run",
            "stats": (
                f"{oa_available} OA links, {fulltext_stage_status['downloaded']} downloaded, "
                f"{len(metrics)} metric files"
            ),
            "artifact": _display_path(config, config.data_dir / "fulltext" / "auto_extracted_metrics.json"),
        },
        {
            "stage": "Interrater",
            "status": "Complete" if interrater else "Not run",
            "stats": _interrater_stats(interrater),
            "artifact": _display_path(config, config.data_dir / "validation" / "interrater_summary.json"),
        },
        {
            "stage": "Robustness",
            "status": "Complete" if robustness else "Not run",
            "stats": _robustness_stats(robustness),
            "artifact": _display_path(config, config.data_dir / "validation" / "robustness_results.json"),
        },
        {
            "stage": "Verify",
            "status": "Complete" if verification else "Not run",
            "stats": _verification_stats(verification),
            "artifact": _display_path(config, config.data_dir / "validation" / "verification_results.json"),
        },
    ]

    return {
        "generated_at_utc": _iso_now(),
        "study": {
            **ident,
            "report_dir": str(paths["report_dir"]),
            "report_path": str(paths["report"]),
            "summary_path": str(paths["summary"]),
            "run_log_path": str(paths["run_log"]),
            "config_snapshot": {
                "config_source": str(config_path) if config_path else "",
                "research": research,
            },
        },
        "question": ident["question"],
        "description": str(research.get("description") or "").strip(),
        "settings": {
            "project_root": str(config.project_root),
            "relevance_threshold": config.relevance_threshold,
            "embedding_model": config.embedding_model,
            "abstract_profile": {
                "provider": config.llm_provider_for("abstract"),
                "model": config.llm_model_for("abstract"),
            },
            "fulltext_profile": {
                "provider": config.llm_provider_for("fulltext"),
                "model": config.llm_model_for("fulltext"),
            },
        },
        "counts": {
            "corpus": search_count,
            "screened": screened_count,
            "excluded_after_relevance": excluded_count,
            "relevant": relevant_count,
            "claims": len(claims),
            "criticality_support": criticality_count,
            "included_after_audit": included_records,
            "removed_in_audit": len(removal_log),
            "enriched_fulltext": enriched_fulltext_count,
            "stage7_downloaded": fulltext_stage_status["downloaded"],
            "stage7_metrics": len(metrics),
        },
        "fulltext": {
            "corpus_enrichment": {
                "downloaded": fulltext_corpus_status["downloaded"],
                "no_usable_text": fulltext_corpus_status["no_usable_text"],
                "error": fulltext_corpus_status["error"],
                "enriched_records": enriched_fulltext_count,
            },
            "deep_extraction": {
                "open_access_links": oa_available,
                "downloaded": fulltext_stage_status["downloaded"],
                "no_usable_text": fulltext_stage_status["no_usable_text"],
                "error": fulltext_stage_status["error"],
                "metrics": len(metrics),
            },
        },
        "method_review": {
            "summary": method_summary,
            "top_method_types": top_method_types,
        },
        "synthesis": {
            "clusters": len(clusters),
            "narratives": len(narratives),
            "top_themes": top_themes,
            "abstract_excerpt": _extract_abstract_excerpt(abstract_text),
        },
        "robustness": robustness,
        "verification": verification.get("summary") if verification else {},
        "stage_rows": stage_rows,
        "prisma": {
            "identified": search_count,
            "screened": screened_count,
            "excluded_after_relevance": excluded_count,
            "retained_after_relevance": relevant_count,
            "claims_extracted": len(claims),
            "removed_in_audit": len(removal_log),
            "included_in_synthesis": included_records,
            "full_text_pulled": enriched_fulltext_count or fulltext_stage_status["downloaded"],
        },
    }


def render_study_report(summary: dict, events: list[dict]) -> str:
    study = summary["study"]
    question = summary.get("question") or "No natural-language research question configured yet."

    lines = [
        f"# Study Report: {study['study_name']}",
        "",
        f"**Generated:** {summary['generated_at_utc']}",
        f"**Study ID:** `{study['study_id']}`",
        f"**Config source:** `{study.get('config_source') or 'GUI in-memory config'}`",
        f"**Research question:** {question}",
    ]
    if summary.get("description"):
        lines.append(f"**Description:** {summary['description']}")
    lines += [
        "",
        "## Pipeline Snapshot",
        "",
        "| Stage | Status | Statistics | Artifact |",
        "|-------|--------|------------|----------|",
    ]
    for row in sorted(summary["stage_rows"], key=lambda item: STAGE_ORDER.index(item["stage"])):
        lines.append(
            f"| {row['stage']} | {row['status']} | {row['stats']} | `{row['artifact']}` |"
        )

    lines += [
        "",
        "## PRISMA-Like Flow",
        "",
        "```mermaid",
        "flowchart TD",
        f'    A["Records in deduplicated corpus\\nn={_count_or_pending(summary["prisma"]["identified"])}"]',
        f'    B["Records screened for relevance\\nn={_count_or_pending(summary["prisma"]["screened"])}"]',
        f'    C["Records excluded after relevance screening\\nn={_count_or_pending(summary["prisma"]["excluded_after_relevance"])}"]',
        f'    D["Records retained after relevance screening\\nn={_count_or_pending(summary["prisma"]["retained_after_relevance"])}"]',
        f'    E["Records with structured claims\\nn={_count_or_pending(summary["prisma"]["claims_extracted"])}"]',
        f'    F["Records removed in audit\\nn={_count_or_pending(summary["prisma"]["removed_in_audit"])}"]',
        f'    G["Records included in synthesis\\nn={_count_or_pending(summary["prisma"]["included_in_synthesis"])}"]',
        f'    H["Records with pulled full text\\nn={_count_or_pending(summary["prisma"]["full_text_pulled"])}"]',
        "    A --> B",
        "    B --> C",
        "    B --> D",
        "    D --> E",
        "    E --> F",
        "    E --> G",
        "    G --> H",
        "```",
        "",
        "## Stage Log",
        "",
        "| Time (UTC) | Stage | Status | Duration (s) | Detail |",
        "|------------|-------|--------|--------------|--------|",
    ]
    for event in events[-30:]:
        lines.append(
            f"| {event.get('timestamp_utc', '')} | {event.get('stage', '')} | "
            f"{event.get('status', '')} | {event.get('duration_sec', '')} | "
            f"{_event_detail(event)} |"
        )

    lines += [
        "",
        "## Mini Review",
        "",
        _mini_review_overview(summary),
        "",
        _mini_review_methods(summary),
        "",
        _mini_review_fulltext(summary),
        "",
        _mini_review_synthesis(summary),
    ]

    abstract_excerpt = summary.get("synthesis", {}).get("abstract_excerpt") or ""
    if abstract_excerpt:
        lines += [
            "",
            "### Current Draft Abstract",
            "",
            abstract_excerpt,
        ]
    else:
        top_themes = summary.get("synthesis", {}).get("top_themes") or []
        if top_themes:
            lines += [
                "",
                "### Current Narrative Themes",
                "",
            ]
            for item in top_themes:
                lines += [
                    f"#### {item['theme']} ({item['paper_count']} papers)",
                    "",
                    item["narrative"] or "Narrative not generated yet.",
                    "",
                ]

    return "\n".join(lines).rstrip() + "\n"


def canonical_stage_name(stage: str) -> str:
    return STAGE_LABELS.get(str(stage or "").strip().lower(), str(stage or "").strip() or "Stage")


def summarize_stage_result(stage: str, result) -> dict:
    stage_name = canonical_stage_name(stage)
    if result is None:
        return {}
    if isinstance(result, list):
        label = "items"
        if stage_name == "Search":
            label = "papers"
        elif stage_name == "Score":
            label = "relevant papers"
        elif stage_name == "Extract":
            label = "claims records"
        elif stage_name == "Validate":
            label = "method review records"
        elif stage_name == "Audit":
            label = "included papers"
        return {"item_count": len(result), "detail": f"{len(result)} {label}"}
    if isinstance(result, dict):
        scalars = {}
        for key, value in result.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                scalars[key] = value
        return scalars
    if isinstance(result, str):
        preview = " ".join(result.split())
        return {"text_preview": preview[:240]}
    return {"repr": repr(result)[:240]}


def load_stage_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                events.append(row)
    return events


def _mini_review_overview(summary: dict) -> str:
    counts = summary["counts"]
    corpus = counts["corpus"]
    relevant = counts["relevant"]
    claims = counts["claims"]
    included = counts["included_after_audit"]
    criticality = counts["criticality_support"]
    return (
        f"The current synthesis contains {corpus} deduplicated papers in the working corpus. "
        f"Relevance screening retained {relevant} papers ({_pct(relevant, corpus):.1f}% of the corpus), "
        f"structured claim extraction has been completed for {claims} papers, and the current included set "
        f"for downstream synthesis contains {included} papers. Among extracted claims, {criticality} papers "
        f"({_pct(criticality, claims):.1f}%) explicitly support a criticality framing."
    )


def _mini_review_methods(summary: dict) -> str:
    method = summary.get("method_review", {})
    meta = method.get("summary") or {}
    top = method.get("top_method_types") or []
    if not meta:
        return "Method-review outputs are not available yet, so study-type distribution and manual-review burden have not been summarized."
    top_text = ", ".join(f"{name} ({count})" for name, count in top) or "no dominant study types yet"
    return (
        f"Method review has processed {meta.get('total_records', 0)} records. "
        f"{meta.get('manual_review_pct', 0)}% of reviewed papers are currently flagged for manual review. "
        f"The most common study types so far are {top_text}."
    )


def _mini_review_fulltext(summary: dict) -> str:
    ft = summary.get("fulltext", {})
    enrich = ft.get("corpus_enrichment") or {}
    deep = ft.get("deep_extraction") or {}
    if not enrich and not deep:
        return "Full-text retrieval has not produced any tracked outputs yet."
    return (
        f"Full-text corpus enrichment has yielded {enrich.get('enriched_records', 0)} papers with embedded full text, "
        f"with {enrich.get('downloaded', 0)} successful pulls, {enrich.get('no_usable_text', 0)} sources that had no usable text, "
        f"and {enrich.get('error', 0)} retrieval errors logged. In the deeper full-text extraction stage, "
        f"{deep.get('open_access_links', 0)} open-access links have been identified, {deep.get('downloaded', 0)} papers have been downloaded, "
        f"and {deep.get('metrics', 0)} metric extraction files have been produced."
    )


def _mini_review_synthesis(summary: dict) -> str:
    synthesis = summary.get("synthesis", {})
    verification = summary.get("verification") or {}
    robustness = summary.get("robustness") or {}
    clusters = synthesis.get("clusters", 0)
    narratives = synthesis.get("narratives", 0)
    top_themes = synthesis.get("top_themes") or []

    pieces = [
        f"Synthesis outputs currently include {clusters} thematic clusters and {narratives} narrative summaries."
    ]
    if top_themes:
        pieces.append(
            "Leading themes include "
            + ", ".join(f"{item['theme']} ({item['paper_count']} papers)" for item in top_themes)
            + "."
        )
    if robustness:
        overall = ((robustness.get("bootstrap") or {}).get("criticality_overall") or {})
        perm = robustness.get("permutation_test") or {}
        graph = robustness.get("graph_networks") or {}
        if overall:
            pieces.append(
                f"Robustness analyses estimate an overall criticality rate of {overall.get('rate', 'N/A')}% "
                f"with a 95% CI of {overall.get('ci_low', 'N/A')}–{overall.get('ci_high', 'N/A')}%."
            )
        if perm:
            pieces.append(
                f"The permutation test currently reports rho={perm.get('observed_rho', 'N/A')} "
                f"with p={perm.get('permutation_p', 'N/A')}."
            )
        if graph:
            struct = ", ".join(
                item.get("location", "")
                for item in graph.get("top_structural_locations", [])[:2]
                if item.get("location")
            )
            func = ", ".join(
                item.get("location", "")
                for item in graph.get("top_functional_locations", [])[:2]
                if item.get("location")
            )
            if struct or func:
                pieces.append(
                    "Autism co-occurrence network hubs currently concentrate around "
                    f"structural-focus locations {struct or 'N/A'} and "
                    f"functional-focus locations {func or 'N/A'}."
                )
    if verification:
        pieces.append(
            f"Abstract verification has {verification.get('passed', 0)}/{verification.get('total_checks', 0)} checks passing, "
            f"with {verification.get('failed', 0)} failures."
        )
    return " ".join(pieces)


def _interrater_stats(summary: dict) -> str:
    if not summary:
        return "No interrater summary yet"
    rel = summary.get("relevance_scores") or {}
    threshold = summary.get("threshold_pass_fail") or {}
    return (
        f"{summary.get('n_sampled', 0)} sampled; exact agreement "
        f"{rel.get('exact_agreement_pct', 'N/A')}%; kappa {threshold.get('kappa', 'N/A')}"
    )


def _robustness_stats(results: dict) -> str:
    if not results:
        return "No robustness results yet"
    overall = ((results.get("bootstrap") or {}).get("criticality_overall") or {})
    perm = results.get("permutation_test") or {}
    graph = results.get("graph_networks") or {}
    parts = [
        f"criticality {overall.get('rate', 'N/A')}% "
        f"[{overall.get('ci_low', 'N/A')}-{overall.get('ci_high', 'N/A')}], "
        f"rho={perm.get('observed_rho', 'N/A')}, p={perm.get('permutation_p', 'N/A')}"
    ]
    struct = ", ".join(
        item.get("location", "")
        for item in graph.get("top_structural_locations", [])[:2]
        if item.get("location")
    )
    func = ", ".join(
        item.get("location", "")
        for item in graph.get("top_functional_locations", [])[:2]
        if item.get("location")
    )
    if struct or func:
        parts.append(
            f"; network hubs struct={struct or 'N/A'} func={func or 'N/A'}"
        )
    return "".join(parts)


def _verification_stats(results: dict) -> str:
    if not results:
        return "No verification results yet"
    summary = results.get("summary") or {}
    return (
        f"{summary.get('passed', 0)}/{summary.get('total_checks', 0)} passed; "
        f"{summary.get('failed', 0)} failed"
    )


def _count_download_statuses(rows: list[dict]) -> dict[str, int]:
    counts = Counter({"downloaded": 0, "no_usable_text": 0, "error": 0})
    for row in rows:
        if not isinstance(row, dict):
            continue
        status = download_record_status(row)
        if status in counts:
            counts[status] += 1
    return dict(counts)


def _extract_abstract_excerpt(text: str) -> str:
    if not text.strip():
        return ""
    body = text.split("\n\n---", 1)[0]
    lines = [line for line in body.splitlines() if line.strip()]
    if lines and lines[0].startswith("#"):
        lines = lines[1:]
    if lines and lines[0].startswith("*Generated from "):
        lines = lines[1:]
    return "\n\n".join(lines).strip()


def _event_detail(event: dict) -> str:
    summary = event.get("summary")
    if isinstance(summary, dict):
        if summary.get("detail"):
            return str(summary["detail"])
        parts = []
        for key, value in summary.items():
            if key == "detail":
                continue
            parts.append(f"{key}={value}")
        if parts:
            return _markdown_cell(", ".join(parts))
    if event.get("error"):
        return _markdown_cell(str(event["error"]))
    return ""


def _load_json_list(path: Path) -> list:
    data = _load_json(path)
    return data if isinstance(data, list) else []


def _load_json_dict(path: Path) -> dict:
    data = _load_json(path)
    return data if isinstance(data, dict) else {}


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")




def _display_path(config, path: Path) -> str:
    try:
        return str(path.relative_to(config.project_root))
    except Exception:
        return str(path)


def _pct(numerator: int, denominator: int) -> float:
    if not denominator:
        return 0.0
    return numerator / denominator * 100.0


def _count_or_pending(value) -> str:
    if value is None:
        return "pending"
    return str(value)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _markdown_cell(value: str) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|").strip()


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _iso_from_epoch(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).replace(microsecond=0).isoformat()
