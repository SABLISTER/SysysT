"""Claim clustering and narrative synthesis."""
import logging
import json
from typing import Optional
from collections import Counter, defaultdict

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding model cache
# ---------------------------------------------------------------------------
_model_cache: dict = {}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
NARRATIVE_PROMPT = """You are synthesizing research findings from a cluster of {paper_count} academic papers for a systematic evidence review.

Research question:
{research_question}

Key mechanisms discussed: {mechanisms}
Populations, constructs, or contextual factors mentioned: {conditions}
Brain regions, networks, or cognitive systems involved: {regions}
Papers directly informing the research question: {primary_support_pct}%

Paper summaries:
{paper_summaries}

Write a cohesive 2-3 paragraph narrative synthesizing these findings. Focus on:
1. What the papers collectively show about the configured research question and described mechanisms
2. Areas of agreement and disagreement
3. Strength of the evidence

Write in academic style suitable for a review paper."""

ABSTRACT_PROMPT = """You are drafting a structured abstract for a systematic review.

Research question:
{research_question}

Total papers analyzed: {total_papers}

Cluster summaries:
{cluster_summaries}

Write a structured abstract with these sections:
- Background (1-2 sentences)
- Objective (1 sentence)
- Methods (2-3 sentences summarizing the pipeline)
- Results (3-5 sentences, the key findings across all clusters)
- Conclusions (2-3 sentences)

Keep it under 350 words."""


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def _research_question_from_config(research_config: dict | None) -> str:
    if not isinstance(research_config, dict):
        return "What does the scholarly literature report for this research question?"
    relevance = research_config.get("relevance")
    if not isinstance(relevance, dict):
        relevance = {}
    for value in (
        research_config.get("natural_language_question"),
        relevance.get("research_question"),
        research_config.get("hypothesis_text"),
        research_config.get("hypothesis"),
        research_config.get("question"),
        research_config.get("description"),
    ):
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return "What does the scholarly literature report for this research question?"


def _get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load sentence-transformers model (cached).  Returns None if not installed."""
    if model_name in _model_cache:
        return _model_cache[model_name]
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        _model_cache[model_name] = model
        logger.info("Loaded sentence-transformers model: %s", model_name)
        return model
    except ImportError:
        logger.warning(
            "sentence_transformers not installed — using TF-IDF fallback for clustering")
        return None


def _paper_to_text(paper: dict) -> str:
    """Build text representation for embedding."""
    claims = paper.get("claims") if isinstance(paper.get("claims"), dict) else {}
    parts = [paper.get("title", "")]
    parts.append(paper.get("abstract", ""))
    parts.extend(paper.get("findings", []) or claims.get("findings", []))
    parts.extend(paper.get("mechanisms", []) or claims.get("mechanisms", []))
    return " ".join(p for p in parts if p)


def _claim_value(paper: dict, key: str, default=None):
    claims = paper.get("claims") if isinstance(paper.get("claims"), dict) else {}
    if key in paper:
        return paper.get(key, default)
    return claims.get(key, default)


def _as_list(value) -> list:
    """Normalize common scalar/list claim values to a clean list."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, tuple) or isinstance(value, set):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def _paper_id(paper: dict, fallback: str = "") -> str:
    for key in ("id", "article_id", "doi", "pmid", "pmcid", "paperId", "title"):
        value = paper.get(key)
        if value:
            return str(value)
    return fallback


def _paper_title(paper: dict) -> str:
    return str(paper.get("title") or paper.get("article_title") or "Untitled")


def _paper_score(paper: dict) -> float:
    for key in (
        "relevance_score",
        "score",
        "llm_relevance",
        "support_score",
        "combined_score",
    ):
        value = paper.get(key)
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    if _claim_value(paper, "supports_primary_question", False):
        return 3.0
    return 0.0


def _paper_direction(paper: dict) -> str:
    for key in ("direction", "llm_direction", "evidence_direction"):
        value = paper.get(key)
        if value:
            text = str(value).lower()
            if "contradict" in text:
                return "contradicts"
            if "support" in text:
                return "supports"
            if "neutral" in text:
                return "neutral"
    if _claim_value(paper, "supports_primary_question", False):
        return "supports"
    return "neutral"


def _paper_quote(paper: dict) -> str:
    for key in ("key_quote", "quoted_span", "quote", "excerpt"):
        value = _claim_value(paper, key, "")
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    spans = _as_list(paper.get("evidence_spans") or _claim_value(paper, "evidence_spans", []))
    return spans[0] if spans else ""


def _method_lookup(method_profiles: list[dict] | dict | None) -> dict[str, dict]:
    if not method_profiles:
        return {}
    rows = method_profiles.values() if isinstance(method_profiles, dict) else method_profiles
    lookup: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("article_id") or row.get("id") or "")
        if pid:
            lookup[pid] = row
    return lookup


def _grade_summary(grade_results: list[dict] | dict | None) -> dict:
    if not grade_results:
        return {}
    rows = grade_results.values() if isinstance(grade_results, dict) else grade_results
    counts: Counter[str] = Counter()
    examples: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        cert = (
            row.get("overall_certainty")
            or row.get("certainty")
            or row.get("overall")
            or ""
        )
        if isinstance(cert, dict):
            cert = cert.get("value") or cert.get("label") or ""
        cert_text = str(cert).strip()
        if cert_text:
            counts[cert_text] += 1
        label = row.get("component_label") or row.get("component") or row.get("label")
        if label and cert_text and len(examples) < 8:
            examples.append({"component": str(label), "certainty": cert_text})
    return {"certainty_counts": dict(sorted(counts.items())), "examples": examples}


def _top_terms(papers: list[dict], field: str, limit: int = 12) -> list[dict]:
    counter: Counter[str] = Counter()
    for paper in papers:
        for value in _as_list(_claim_value(paper, field, [])):
            key = " ".join(value.split())
            if key:
                counter[key] += 1
    return [
        {"term": term, "count": count}
        for term, count in counter.most_common(limit)
    ]


def _rank_papers(papers: list[dict], limit: int = 8) -> list[dict]:
    ranked = sorted(
        papers,
        key=lambda p: (
            _paper_direction(p) == "supports",
            _paper_score(p),
            int(p.get("citations") or p.get("citation_count") or 0),
            str(p.get("year") or ""),
            _paper_title(p).lower(),
        ),
        reverse=True,
    )
    rows = []
    for paper in ranked[:limit]:
        rows.append({
            "id": _paper_id(paper),
            "title": _paper_title(paper),
            "year": paper.get("year"),
            "score": _paper_score(paper),
            "direction": _paper_direction(paper),
            "citations": paper.get("citations") or paper.get("citation_count") or 0,
            "quote": _paper_quote(paper),
            "findings": _as_list(_claim_value(paper, "findings", []))[:3],
            "mechanisms": _as_list(_claim_value(paper, "mechanisms", []))[:5],
        })
    return rows


def build_hypothesis_explanation(
    clusters: list[dict],
    narratives: list[dict],
    abstract: str,
    *,
    research_config: dict | None = None,
    method_profiles: list[dict] | dict | None = None,
    grade_results: list[dict] | dict | None = None,
) -> dict:
    """Build a deterministic provenance explanation for the synthesis."""
    all_papers: list[dict] = []
    for cluster in clusters:
        all_papers.extend(cluster.get("papers", []) or [])

    method_by_id = _method_lookup(method_profiles)
    method_counts: Counter[str] = Counter()
    for paper in all_papers:
        profile = method_by_id.get(_paper_id(paper)) or method_by_id.get(str(paper.get("article_id") or ""))
        if profile:
            method_counts[str(profile.get("study_method_type") or "unknown")] += 1

    narrative_by_cluster = {
        n.get("cluster_id"): n
        for n in narratives
        if isinstance(n, dict)
    }

    cluster_rows = []
    for cluster in clusters:
        papers = cluster.get("papers", []) or []
        support_count = sum(1 for p in papers if _paper_direction(p) == "supports")
        contradict_count = sum(1 for p in papers if _paper_direction(p) == "contradicts")
        neutral_count = len(papers) - support_count - contradict_count
        avg_score = (
            sum(_paper_score(p) for p in papers) / len(papers)
            if papers else 0.0
        )
        cluster_rows.append({
            "cluster_id": cluster.get("cluster_id"),
            "paper_count": len(papers),
            "avg_relevance_score": round(avg_score, 3),
            "supporting_count": support_count,
            "contradicting_count": contradict_count,
            "neutral_count": neutral_count,
            "mechanisms": cluster.get("mechanisms", [])[:12],
            "conditions": cluster.get("conditions", [])[:12],
            "brain_regions": cluster.get("brain_regions", [])[:12],
            "top_papers": _rank_papers(papers, limit=6),
            "narrative_excerpt": (
                narrative_by_cluster.get(cluster.get("cluster_id"), {}).get("narrative", "")[:600]
            ),
        })

    cluster_rows.sort(
        key=lambda row: (
            row["supporting_count"],
            row["avg_relevance_score"],
            row["paper_count"],
        ),
        reverse=True,
    )

    support_total = sum(row["supporting_count"] for row in cluster_rows)
    contradict_total = sum(row["contradicting_count"] for row in cluster_rows)
    neutral_total = sum(row["neutral_count"] for row in cluster_rows)

    caveats: list[str] = []
    if not all_papers:
        caveats.append("No papers were available for deterministic provenance.")
    if contradict_total:
        caveats.append(
            f"{contradict_total} paper(s) were marked as contradicting or directionally opposed."
        )
    if support_total and neutral_total > support_total:
        caveats.append("Neutral or weakly directional evidence outnumbers direct support.")
    if not method_counts:
        caveats.append("Method validation output was not available for study-design weighting.")
    if not grade_results:
        caveats.append("GRADE output was not available when this explanation was generated.")

    explanation = {
        "schema_version": 1,
        "research_question": _research_question_from_config(research_config),
        "total_papers": len(all_papers),
        "cluster_count": len(clusters),
        "supporting_count": support_total,
        "contradicting_count": contradict_total,
        "neutral_count": neutral_total,
        "top_mechanisms": _top_terms(all_papers, "mechanisms"),
        "top_conditions": _top_terms(all_papers, "conditions"),
        "top_brain_regions": _top_terms(all_papers, "brain_regions"),
        "top_findings": _top_terms(all_papers, "findings"),
        "method_mix": dict(sorted(method_counts.items())),
        "grade_summary": _grade_summary(grade_results),
        "clusters": cluster_rows,
        "top_papers_overall": _rank_papers(all_papers, limit=10),
        "caveats": caveats,
        "abstract_excerpt": (abstract or "")[:1200],
    }
    return explanation


def render_hypothesis_explanation_markdown(explanation: dict) -> str:
    """Render the deterministic explanation artifact as readable Markdown."""
    lines = [
        "# Why This Hypothesis?",
        "",
        "## Research Question",
        "",
        str(explanation.get("research_question") or "Not specified."),
        "",
        "## Evidence Basis",
        "",
        f"- Papers in synthesis: {explanation.get('total_papers', 0)}",
        f"- Clusters: {explanation.get('cluster_count', 0)}",
        f"- Supporting signals: {explanation.get('supporting_count', 0)}",
        f"- Contradicting signals: {explanation.get('contradicting_count', 0)}",
        f"- Neutral or unclear signals: {explanation.get('neutral_count', 0)}",
        "",
    ]

    for title, key in (
        ("Top Mechanisms", "top_mechanisms"),
        ("Top Conditions or Contexts", "top_conditions"),
        ("Top Brain Regions or Systems", "top_brain_regions"),
    ):
        rows = explanation.get(key) or []
        if rows:
            lines.extend([f"## {title}", ""])
            for row in rows[:10]:
                lines.append(f"- {row['term']} (n={row['count']})")
            lines.append("")

    method_mix = explanation.get("method_mix") or {}
    if method_mix:
        lines.extend(["## Method Mix", ""])
        for method, count in method_mix.items():
            lines.append(f"- {method}: {count}")
        lines.append("")

    grade_summary = explanation.get("grade_summary") or {}
    if grade_summary.get("certainty_counts"):
        lines.extend(["## GRADE Snapshot", ""])
        for certainty, count in grade_summary["certainty_counts"].items():
            lines.append(f"- {certainty}: {count}")
        lines.append("")

    lines.extend(["## Top Contributing Clusters", ""])
    for cluster in explanation.get("clusters", [])[:6]:
        cid = cluster.get("cluster_id")
        lines.append(
            f"### Cluster {cid} - {cluster.get('paper_count', 0)} papers"
        )
        lines.append(
            f"Average score: {cluster.get('avg_relevance_score', 0)}; "
            f"supporting: {cluster.get('supporting_count', 0)}; "
            f"contradicting: {cluster.get('contradicting_count', 0)}"
        )
        terms = []
        for key in ("mechanisms", "conditions", "brain_regions"):
            terms.extend(cluster.get(key) or [])
        if terms:
            lines.append(f"Key terms: {', '.join(terms[:10])}")
        for paper in cluster.get("top_papers", [])[:3]:
            title = paper.get("title") or "Untitled"
            year = paper.get("year") or "n.d."
            score = paper.get("score", 0)
            direction = paper.get("direction", "neutral")
            lines.append(f"- {title} ({year}) - {direction}, score {score:g}")
            quote = paper.get("quote")
            if quote:
                lines.append(f"  Quote: {quote[:280]}")
        lines.append("")

    caveats = explanation.get("caveats") or []
    if caveats:
        lines.extend(["## Deterministic Caveats", ""])
        for caveat in caveats:
            lines.append(f"- {caveat}")
        lines.append("")

    if explanation.get("abstract_excerpt"):
        lines.extend(["## Synthesis Excerpt", "", str(explanation["abstract_excerpt"]).strip(), ""])

    return "\n".join(lines).rstrip() + "\n"


def build_evidence_network(
    clusters: list[dict],
    *,
    top_papers_per_cluster: int = 8,
    max_terms_per_type: int = 80,
) -> dict:
    """Build a generalized evidence network from clustered papers."""
    nodes: dict[str, dict] = {}
    edge_weights: dict[tuple[str, str, str], float] = defaultdict(float)
    edge_counts: Counter[tuple[str, str, str]] = Counter()
    term_counts: dict[str, Counter[str]] = {
        "mechanism": Counter(),
        "condition": Counter(),
        "brain_region": Counter(),
        "finding": Counter(),
        "method": Counter(),
    }

    def norm_id(kind: str, label: str) -> str:
        clean = " ".join(str(label).split()).lower()
        clean = "".join(ch if ch.isalnum() else "_" for ch in clean).strip("_")
        return f"{kind}:{clean or 'unknown'}"

    def add_node(node_id: str, label: str, node_type: str, **metadata) -> None:
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "label": label,
                "type": node_type,
                "count": 0,
                "metadata": {},
            }
        nodes[node_id]["count"] += int(metadata.pop("count", 1) or 1)
        nodes[node_id]["metadata"].update({k: v for k, v in metadata.items() if v not in (None, "")})

    def add_edge(source: str, target: str, edge_type: str, weight: float = 1.0) -> None:
        if source == target:
            return
        key = (source, target, edge_type)
        edge_weights[key] += float(weight or 1.0)
        edge_counts[key] += 1

    for cluster in clusters:
        cid = str(cluster.get("cluster_id"))
        cluster_id = f"cluster:{cid}"
        add_node(cluster_id, f"Cluster {cid}", "cluster", count=cluster.get("paper_count", 1))
        papers = cluster.get("papers", []) or []
        for paper in papers:
            for kind, field in (
                ("mechanism", "mechanisms"),
                ("condition", "conditions"),
                ("brain_region", "brain_regions"),
                ("finding", "findings"),
            ):
                for value in _as_list(_claim_value(paper, field, [])):
                    term_counts[kind][value] += 1
            methodology = _claim_value(paper, "methodology", "")
            if methodology:
                for token in str(methodology).replace("/", " ").split(",")[:3]:
                    token = token.strip()
                    if token:
                        term_counts["method"][token] += 1

        ranked_originals = sorted(
            papers,
            key=lambda p: (
                _paper_direction(p) == "supports",
                _paper_score(p),
                int(p.get("citations") or p.get("citation_count") or 0),
                str(p.get("year") or ""),
                _paper_title(p).lower(),
            ),
            reverse=True,
        )[:top_papers_per_cluster]
        for paper in ranked_originals:
            paper_id = norm_id("paper", _paper_id(paper))
            add_node(
                paper_id,
                _paper_title(paper),
                "paper",
                count=1,
                year=paper.get("year"),
                score=_paper_score(paper),
                direction=_paper_direction(paper),
            )
            add_edge(paper_id, cluster_id, "belongs_to", 1.0)

    allowed_terms = {
        kind: {term for term, _ in counter.most_common(max_terms_per_type)}
        for kind, counter in term_counts.items()
    }

    for cluster in clusters:
        cid = str(cluster.get("cluster_id"))
        cluster_id = f"cluster:{cid}"
        papers = cluster.get("papers", []) or []
        ranked_originals = sorted(
            papers,
            key=lambda p: (
                _paper_direction(p) == "supports",
                _paper_score(p),
                int(p.get("citations") or p.get("citation_count") or 0),
                str(p.get("year") or ""),
                _paper_title(p).lower(),
            ),
            reverse=True,
        )[:top_papers_per_cluster]
        for paper in ranked_originals:
            pid = norm_id("paper", _paper_id(paper))
            direction = _paper_direction(paper)
            if direction in ("supports", "contradicts"):
                add_edge(pid, cluster_id, direction, max(_paper_score(paper), 1.0))
            for kind, field, edge_type in (
                ("mechanism", "mechanisms", "mentions"),
                ("condition", "conditions", "mentions"),
                ("brain_region", "brain_regions", "mentions"),
                ("finding", "findings", "reports"),
            ):
                for value in _as_list(_claim_value(paper, field, [])):
                    if value not in allowed_terms[kind]:
                        continue
                    term_id = norm_id(kind, value)
                    if term_id not in nodes:
                        add_node(term_id, value, kind, count=term_counts[kind][value])
                    add_edge(pid, term_id, edge_type, 1.0)
                    add_edge(cluster_id, term_id, "cluster_theme", 1.0)
            methodology = _claim_value(paper, "methodology", "")
            for value in [v.strip() for v in str(methodology).replace("/", " ").split(",")[:3] if v.strip()]:
                if value not in allowed_terms["method"]:
                    continue
                method_id = norm_id("method", value)
                if method_id not in nodes:
                    add_node(method_id, value, "method", count=term_counts["method"][value])
                add_edge(pid, method_id, "uses_method", 1.0)

    edges = [
        {
            "source": source,
            "target": target,
            "type": edge_type,
            "weight": round(weight, 3),
            "evidence_count": edge_counts[(source, target, edge_type)],
        }
        for (source, target, edge_type), weight in edge_weights.items()
    ]
    edges.sort(key=lambda e: (e["source"], e["target"], e["type"]))
    node_list = sorted(nodes.values(), key=lambda n: (n["type"], n["label"].lower()))
    return {
        "schema_version": 1,
        "kind": "general_evidence_network",
        "nodes": node_list,
        "edges": edges,
        "summary": {
            "node_count": len(node_list),
            "edge_count": len(edges),
            "included_cluster_count": len(clusters),
            "top_papers_per_cluster": top_papers_per_cluster,
        },
    }


def _get_embeddings(
    papers: list[dict], model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """Embed papers.  Uses sentence-transformers if available, else TF-IDF."""
    model = _get_embedding_model(model_name)
    texts = [_paper_to_text(p) for p in papers]

    if model is not None:
        logger.info("Encoding %d papers with sentence-transformers", len(texts))
        return model.encode(texts, show_progress_bar=False)

    # TF-IDF fallback
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        logger.info("Encoding %d papers with TF-IDF fallback", len(texts))
        vec = TfidfVectorizer(max_features=5000, stop_words="english")
        return vec.fit_transform(texts).toarray()
    except ImportError:
        logger.warning("sklearn not available — returning random embeddings")
        return np.random.randn(len(texts), 64)


# ---------------------------------------------------------------------------
# Simple k-means fallback (no sklearn)
# ---------------------------------------------------------------------------
def _simple_kmeans(X: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
    """Minimal k-means via random partition.

    Returns an array of cluster labels (length = number of rows in X).
    """
    n = X.shape[0]
    # Random initial assignment
    labels = np.array([i % k for i in range(n)])
    np.random.shuffle(labels)

    for _iteration in range(max_iter):
        # Compute centroids
        centroids = np.zeros((k, X.shape[1]))
        for ci in range(k):
            members = X[labels == ci]
            if len(members) > 0:
                centroids[ci] = members.mean(axis=0)
            else:
                centroids[ci] = X[np.random.randint(n)]

        # Reassign
        new_labels = np.argmin(
            np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2),
            axis=1,
        )
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

    return labels


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def cluster_claims(
    papers: list[dict],
    n_clusters: int = 8,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> list[dict]:
    """Cluster papers thematically using k-means on embeddings.

    Returns list of cluster dicts::

        [{"cluster_id": 0, "papers": [...], "mechanisms": [...],
          "conditions": [...], "brain_regions": [...],
          "primary_support_count": int, "criticality_count": int}, ...]
    """
    if not papers:
        logger.warning("No papers to cluster — returning empty list")
        return []

    # Reduce n_clusters if fewer papers
    actual_k = min(n_clusters, len(papers))
    if actual_k < n_clusters:
        logger.info(
            "Reducing clusters from %d to %d (only %d papers)",
            n_clusters, actual_k, len(papers),
        )
    if actual_k < 1:
        actual_k = 1

    # Embed
    embeddings = _get_embeddings(papers, model_name=embedding_model)

    # K-means
    try:
        from sklearn.cluster import KMeans
        logger.info("Running sklearn KMeans with k=%d on %d papers", actual_k, len(papers))
        km = KMeans(n_clusters=actual_k, n_init=10, random_state=42)
        labels = km.fit_predict(embeddings)
    except ImportError:
        logger.info(
            "sklearn not available — running simple k-means fallback with k=%d", actual_k)
        labels = _simple_kmeans(embeddings, actual_k)

    # Build cluster summaries
    clusters: list[dict] = []
    for ci in range(actual_k):
        indices = [i for i, lb in enumerate(labels) if lb == ci]
        cluster_papers = [papers[i] for i in indices]
        mechanisms: set[str] = set()
        conditions: set[str] = set()
        brain_regions: set[str] = set()
        primary_support_count = 0
        criticality_count = 0

        for p in cluster_papers:
            mechanisms.update(_claim_value(p, "mechanisms", []) or [])
            conditions.update(_claim_value(p, "conditions", []) or [])
            brain_regions.update(_claim_value(p, "brain_regions", []) or [])
            if _claim_value(p, "supports_primary_question", False):
                primary_support_count += 1
            if _claim_value(p, "supports_criticality", False):
                criticality_count += 1

        clusters.append({
            "cluster_id": ci,
            "papers": cluster_papers,
            "paper_count": len(cluster_papers),
            "mechanisms": sorted(mechanisms),
            "conditions": sorted(conditions),
            "brain_regions": sorted(brain_regions),
            "primary_support_count": primary_support_count,
            "criticality_count": criticality_count,
        })

    # Log cluster sizes
    sizes = [c["paper_count"] for c in clusters]
    logger.info(
        "Clustered %d papers into %d clusters — sizes: %s",
        len(papers), actual_k, sizes,
    )
    return clusters


# ---------------------------------------------------------------------------
# Narrative generation
# ---------------------------------------------------------------------------
def generate_cluster_narrative(
    model: str,
    cluster: dict,
    *,
    research_config: dict | None = None,
    use_thinking: bool = False,
    max_tokens: int = 4096,
) -> dict:
    """Generate a narrative for one cluster via LLM.

    Returns ``{"cluster_id": ..., "narrative": ..., "paper_count": ...}``.
    """
    from process.llm import chat as llm_chat

    paper_count = cluster.get("paper_count", len(cluster.get("papers", [])))
    mechanisms = ", ".join(cluster.get("mechanisms", [])) or "not specified"
    conditions = ", ".join(cluster.get("conditions", [])) or "not specified"
    regions = ", ".join(cluster.get("brain_regions", [])) or "not specified"
    primary_count = cluster.get("primary_support_count", cluster.get("criticality_count", 0))
    primary_support_pct = round(100 * primary_count / paper_count) if paper_count else 0

    # Build per-paper summaries
    summaries: list[str] = []
    for p in cluster.get("papers", []):
        title = p.get("title", "Untitled")
        abstract_snippet = (p.get("abstract", "") or "")[:300]
        findings = "; ".join(p.get("findings", [])[:3])
        summaries.append(f"- {title}: {abstract_snippet}... Findings: {findings}")
    paper_summaries = "\n".join(summaries) if summaries else "No summaries available."

    prompt_text = NARRATIVE_PROMPT.format(
        research_question=_research_question_from_config(research_config),
        paper_count=paper_count,
        mechanisms=mechanisms,
        conditions=conditions,
        regions=regions,
        primary_support_pct=primary_support_pct,
        paper_summaries=paper_summaries,
    )

    logger.info(
        "Generating narrative for cluster %d (%d papers, model=%s)",
        cluster["cluster_id"], paper_count, model,
    )

    messages = [{"role": "user", "content": prompt_text}]
    narrative = llm_chat(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        think=use_thinking,
    )

    logger.info(
        "Cluster %d narrative: %d chars",
        cluster["cluster_id"], len(narrative),
    )

    return {
        "cluster_id": cluster["cluster_id"],
        "narrative": narrative,
        "paper_count": paper_count,
        "mechanisms": cluster.get("mechanisms", []),
        "conditions": cluster.get("conditions", []),
        "brain_regions": cluster.get("brain_regions", []),
        "primary_support_count": primary_count,
    }


# ---------------------------------------------------------------------------
# Abstract draft
# ---------------------------------------------------------------------------
def generate_abstract_draft(
    model: str,
    cluster_narratives: list[dict],
    total_papers: int,
    *,
    research_config: dict | None = None,
    use_thinking: bool = False,
    max_tokens: int = 8192,
) -> str:
    """Generate a structured abstract from all cluster narratives."""
    from process.llm import chat as llm_chat

    # Build cluster summaries block
    parts: list[str] = []
    for cn in cluster_narratives:
        cid = cn.get("cluster_id", "?")
        n = cn.get("paper_count", 0)
        mechs = ", ".join(cn.get("mechanisms", []))
        snippet = (cn.get("narrative", "") or "")[:500]
        parts.append(
            f"Cluster {cid} ({n} papers, mechanisms: {mechs}):\n{snippet}"
        )
    cluster_summaries = "\n\n".join(parts) if parts else "No cluster summaries."

    prompt_text = ABSTRACT_PROMPT.format(
        research_question=_research_question_from_config(research_config),
        total_papers=total_papers,
        cluster_summaries=cluster_summaries,
    )

    logger.info(
        "Generating abstract draft from %d cluster narratives (%d total papers, model=%s)",
        len(cluster_narratives), total_papers, model,
    )

    messages = [{"role": "user", "content": prompt_text}]
    abstract = llm_chat(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        think=use_thinking,
    )

    logger.info("Abstract draft generated: %d chars", len(abstract))
    return abstract
