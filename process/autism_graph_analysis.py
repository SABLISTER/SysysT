"""Configurable graph analysis for condition–brain-region co-occurrence.

Provides entity-recognition patterns for conditions, structural brain regions,
and functional networks.  Patterns can be customised via the ``graph_analysis``
    section of ``research_config.yaml``; when absent the built-in legacy
    neuroscience defaults are used.

Usage::

    from process.autism_graph_analysis import load_patterns

    patterns = load_patterns(config)
    conditions = patterns["conditions"]
    structural = patterns["structural_regions"]
    functional = patterns["functional_networks"]
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ── Built-in default patterns ─────────────────────────────────────────────
# These are the original legacy neuroscience patterns that ship with Systes.
# They are used when no ``graph_analysis`` section is present in the config.

DEFAULT_CONDITIONS: dict[str, list[str]] = {
    "Fragile X": [r"\bFragile X\b", r"\bFMR1\b", r"\bfragile X syndrome\b"],
    "Rett syndrome": [r"\bRett syndrome\b", r"\bRett\b", r"\bMECP2\b"],
    "Anxiety": [r"\banxiety\b", r"\banxious\b", r"\bGAD\b", r"\bgeneralized anxiety\b"],
    "ADHD": [r"\bADHD\b", r"\battention.deficit\b", r"\bhyperactivity\b"],
    "Epilepsy": [r"\bepilepsy\b", r"\bseizure\b", r"\bepileptic\b"],
    "Sensory Processing": [
        r"\bsensory processing\b", r"\bSPD\b",
        r"\bsensory.over.?responsivity\b", r"\bsensory.under.?responsivity\b",
    ],
    "Sleep Disorders": [
        r"\binsomnia\b", r"\bsleep disorder\b", r"\bsleep disturbance\b",
        r"\bcircadian\b",
    ],
    "GI Disorders": [
        r"\bgastrointestinal\b", r"\bGI disorder\b", r"\bgut.brain\b",
        r"\bmicrobiome\b",
    ],
    "Depression": [
        r"\bdepression\b", r"\bdepressive\b", r"\bMDD\b",
        r"\bmajor depressive\b",
    ],
    "OCD": [r"\bOCD\b", r"\bobsessive.compulsive\b", r"\brepetitive behavio", r"\bRRBI\b"],
    "Intellectual Disability": [
        r"\bintellectual disability\b", r"\bID\b",
        r"\bcognitive impairment\b",
    ],
}

DEFAULT_STRUCTURAL_REGIONS: dict[str, list[str]] = {
    "Prefrontal Cortex": [r"\bPFC\b", r"\bprefrontal\b", r"\bdlPFC\b", r"\bvmPFC\b"],
    "Amygdala": [r"\bamygdala\b", r"\bamygdalar\b"],
    "Hippocampus": [r"\bhippocampus\b", r"\bhippocampal\b"],
    "Insula": [r"\binsula\b", r"\binsular cortex\b"],
    "Anterior Cingulate": [r"\bACC\b", r"\banterior cingulate\b", r"\bdACC\b"],
    "Thalamus": [r"\bthalamus\b", r"\bthalamic\b"],
    "Cerebellum": [r"\bcerebellum\b", r"\bcerebellar\b"],
    "Basal Ganglia": [r"\bbasal ganglia\b", r"\bstriatum\b", r"\bcaudate\b", r"\bputamen\b"],
    "Superior Temporal Sulcus": [r"\bSTS\b", r"\bsuperior temporal sulcus\b"],
    "Fusiform Gyrus": [r"\bfusiform\b", r"\bFFA\b"],
    "Motor Cortex": [r"\bmotor cortex\b", r"\bprimary motor\b", r"\bM1\b"],
    "Sensory Cortex": [r"\bsomatosensory\b", r"\bS1\b", r"\bsensory cortex\b"],
    "Temporal Lobe": [r"\btemporal lobe\b", r"\btemporal cortex\b"],
    "Parietal Cortex": [r"\bparietal\b", r"\bIPL\b", r"\bSPL\b"],
    "Occipital Cortex": [r"\boccipital\b", r"\bvisual cortex\b", r"\bV1\b"],
}

DEFAULT_FUNCTIONAL_NETWORKS: dict[str, list[str]] = {
    "Default Mode": [r"\bDMN\b", r"\bdefault mode\b"],
    "Salience": [r"\bsalience network\b", r"\bSN\b"],
    "Central Executive": [r"\bCEN\b", r"\bcentral executive\b", r"\bfrontoparietal\b"],
    "Sensorimotor": [r"\bsensorimotor\b", r"\bSMN\b"],
    "Visual": [r"\bvisual network\b", r"\bVN\b"],
    "Dorsal Attention": [r"\bdorsal attention\b", r"\bDAN\b"],
    "Ventral Attention": [r"\bventral attention\b", r"\bVAN\b"],
    "Limbic": [r"\blimbic\b", r"\blimbic network\b"],
}


# ── Pattern loading ───────────────────────────────────────────────────────


def load_patterns(config) -> dict[str, dict[str, list[str]]]:
    """Load entity-recognition patterns from config, falling back to defaults.

    Checks ``config.research_config.get("graph_analysis", {})`` for
    custom pattern dictionaries keyed as ``conditions``,
    ``structural_regions``, and ``functional_networks``.  If the section
    is absent or a sub-key is missing, the corresponding built-in
    default is used.

    Returns a dict with three keys: ``conditions``, ``structural_regions``,
    ``functional_networks``.  Each maps entity names to lists of regex
    pattern strings.
    """
    custom: dict = {}
    try:
        custom = config.research_config.get("graph_analysis", {})
    except AttributeError:
        pass  # config may not have research_config

    if not isinstance(custom, dict):
        custom = {}

    conditions = custom.get("conditions", DEFAULT_CONDITIONS)
    structural = custom.get("structural_regions", DEFAULT_STRUCTURAL_REGIONS)
    functional = custom.get("functional_networks", DEFAULT_FUNCTIONAL_NETWORKS)

    # Validate patterns compile
    for label, patterns_dict in [
        ("conditions", conditions),
        ("structural_regions", structural),
        ("functional_networks", functional),
    ]:
        if not isinstance(patterns_dict, dict):
            logger.warning(
                "graph_analysis.%s is not a dict, using defaults", label
            )
            if label == "conditions":
                conditions = DEFAULT_CONDITIONS
            elif label == "structural_regions":
                structural = DEFAULT_STRUCTURAL_REGIONS
            else:
                functional = DEFAULT_FUNCTIONAL_NETWORKS
            continue

        for entity, regexes in patterns_dict.items():
            if not isinstance(regexes, list):
                logger.warning(
                    "graph_analysis.%s.%s patterns is not a list, skipping",
                    label, entity,
                )
                continue
            for pat in regexes:
                try:
                    re.compile(pat)
                except re.error as exc:
                    logger.warning(
                        "Invalid regex in graph_analysis.%s.%s: %s (%s)",
                        label, entity, pat, exc,
                    )

    return {
        "conditions": conditions,
        "structural_regions": structural,
        "functional_networks": functional,
    }


# ── Entity extraction ─────────────────────────────────────────────────────


def extract_entities(
    text: str,
    patterns: dict[str, dict[str, list[str]]],
) -> dict[str, list[str]]:
    """Extract named entities from text using the pattern dictionaries.

    Returns a dict with keys ``conditions``, ``structural_regions``,
    ``functional_networks``, each mapping to a list of matched entity
    names found in the text.
    """
    results: dict[str, list[str]] = {
        "conditions": [],
        "structural_regions": [],
        "functional_networks": [],
    }
    for category, entity_dict in patterns.items():
        if not isinstance(entity_dict, dict):
            continue
        for entity_name, regexes in entity_dict.items():
            if not isinstance(regexes, list):
                continue
            for pat in regexes:
                try:
                    if re.search(pat, text, re.IGNORECASE):
                        results[category].append(entity_name)
                        break
                except re.error:
                    continue
    return results


def build_cooccurrence_edges(
    corpus: list[dict],
    patterns: dict[str, dict[str, list[str]]],
) -> list[dict]:
    """Build condition–region co-occurrence edges from a corpus.

    For each article, extract entities and create edges between every
    condition and every brain region (structural or functional) that
    co-occur in the same text.  Returns a list of edge dicts with
    ``source``, ``target``, ``weight`` (co-occurrence count), and
    ``source_type`` / ``target_type``.
    """
    edge_counts: dict[tuple[str, str], int] = {}
    edge_types: dict[tuple[str, str], tuple[str, str]] = {}

    for paper in corpus:
        text = " ".join(filter(None, [
            paper.get("title", ""),
            paper.get("abstract", ""),
        ]))
        if not text.strip():
            continue

        entities = extract_entities(text, patterns)
        conditions = entities["conditions"]
        regions = entities["structural_regions"] + entities["functional_networks"]

        for cond in conditions:
            for region in regions:
                key = (cond, region)
                edge_counts[key] = edge_counts.get(key, 0) + 1
                if key not in edge_types:
                    is_structural = region in (
                        patterns.get("structural_regions", {})
                    )
                    edge_types[key] = (
                        "condition",
                        "structural" if is_structural else "functional",
                    )

    edges = []
    for (source, target), weight in edge_counts.items():
        src_type, tgt_type = edge_types.get((source, target), ("condition", "region"))
        edges.append({
            "source": source,
            "target": target,
            "weight": weight,
            "source_type": src_type,
            "target_type": tgt_type,
        })

    return sorted(edges, key=lambda e: e["weight"], reverse=True)


# ── Data-driven edge thresholding ────────────────────────────────────────

def compute_edge_threshold(
    corpus: list[dict],
    patterns: dict[str, dict[str, list[str]]],
    n_permutations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> float:
    """Compute a significance threshold for co-occurrence edge weights.

    Builds a null distribution by permuting condition labels across papers,
    then returns the (1−α) percentile of the max-weight distribution.
    Edges with weight > threshold are statistically significant.

    Parameters
    ----------
    corpus : list of paper dicts (must have 'title' and/or 'abstract')
    patterns : entity pattern dictionary from load_patterns()
    n_permutations : number of shuffles for the null distribution
    alpha : significance level (default 0.05 → 95th percentile)
    seed : random seed for reproducibility

    Returns
    -------
    float : weight threshold (edges above this are significant at level α)
    """
    import random as _random

    if not corpus:
        return 0

    # Extract condition labels per paper
    paper_conditions: list[list[str]] = []
    paper_regions: list[list[str]] = []
    for paper in corpus:
        text = " ".join(filter(None, [
            paper.get("title", ""),
            paper.get("abstract", ""),
        ]))
        entities = extract_entities(text, patterns)
        paper_conditions.append(entities["conditions"])
        paper_regions.append(
            entities["structural_regions"] + entities["functional_networks"]
        )

    rng = _random.Random(seed)
    null_max_weights: list[int] = []

    for _ in range(n_permutations):
        # Shuffle condition labels across papers
        shuffled_conditions = list(paper_conditions)
        rng.shuffle(shuffled_conditions)

        # Rebuild co-occurrence counts with shuffled conditions
        edge_counts: dict[tuple[str, str], int] = {}
        for conds, regions in zip(shuffled_conditions, paper_regions):
            for c in conds:
                for r in regions:
                    key = (c, r)
                    edge_counts[key] = edge_counts.get(key, 0) + 1

        max_w = max(edge_counts.values()) if edge_counts else 0
        null_max_weights.append(max_w)

    if not null_max_weights:
        return 0

    # Threshold at (1 - alpha) percentile of null max-weight distribution
    null_max_weights.sort()
    idx = int(len(null_max_weights) * (1 - alpha))
    idx = min(idx, len(null_max_weights) - 1)
    return null_max_weights[idx]


# ── Community detection ──────────────────────────────────────────────────

def detect_communities(edges: list[dict]) -> dict:
    """Louvain community detection on the co-occurrence graph.

    Parameters
    ----------
    edges : list of dicts with 'source', 'target', 'weight'

    Returns
    -------
    dict with:
        communities : {node_name: community_id}
        modularity : float
        n_communities : int
    """
    if not edges:
        return {"communities": {}, "modularity": 0.0, "n_communities": 0}

    try:
        import networkx as nx
        import community as community_louvain
    except ImportError:
        logger.warning(
            "networkx or python-louvain not installed — "
            "community detection unavailable"
        )
        return {"communities": {}, "modularity": 0.0, "n_communities": 0}

    G = nx.Graph()
    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        w = edge.get("weight", 1)
        if src and tgt:
            if G.has_edge(src, tgt):
                G[src][tgt]["weight"] += w
            else:
                G.add_edge(src, tgt, weight=w)

    if G.number_of_nodes() == 0:
        return {"communities": {}, "modularity": 0.0, "n_communities": 0}

    partition = community_louvain.best_partition(G, weight="weight")
    modularity = community_louvain.modularity(partition, G, weight="weight")
    n_communities = len(set(partition.values()))

    return {
        "communities": partition,
        "modularity": round(float(modularity), 6),
        "n_communities": n_communities,
    }


# ── Temporal network analysis ────────────────────────────────────────────

def temporal_network_analysis(
    corpus: list[dict],
    patterns: dict[str, dict[str, list[str]]],
    window_years: int = 5,
) -> list[dict]:
    """Build co-occurrence networks for successive time windows.

    Parameters
    ----------
    corpus : list of paper dicts (must have 'year', 'title', 'abstract')
    patterns : entity pattern dictionary from load_patterns()
    window_years : width of each time bin in years

    Returns
    -------
    list of dicts, one per window, each with:
        window_start, window_end, n_papers, n_edges, top_edges,
        edge_set (frozenset for Jaccard), jaccard_vs_prev
    """
    if not corpus or window_years < 1:
        return []

    # Extract years and find range
    papers_with_year = []
    for p in corpus:
        y = p.get("year")
        if y is not None:
            try:
                y = int(y)
                if 1900 <= y <= 2100:
                    papers_with_year.append((y, p))
            except (ValueError, TypeError):
                pass

    if not papers_with_year:
        return []

    min_year = min(y for y, _ in papers_with_year)
    max_year = max(y for y, _ in papers_with_year)

    # Build windows
    windows = []
    start = min_year
    while start <= max_year:
        end = start + window_years - 1
        window_papers = [p for y, p in papers_with_year if start <= y <= end]

        if window_papers:
            edges = build_cooccurrence_edges(window_papers, patterns)
            edge_set = frozenset((e["source"], e["target"]) for e in edges)

            windows.append({
                "window_start": start,
                "window_end": end,
                "n_papers": len(window_papers),
                "n_edges": len(edges),
                "top_edges": edges[:10],
                "edge_set": edge_set,
            })

        start += window_years

    # Compute Jaccard similarity between consecutive windows
    for i, w in enumerate(windows):
        if i == 0:
            w["jaccard_vs_prev"] = None
        else:
            prev_set = windows[i - 1]["edge_set"]
            curr_set = w["edge_set"]
            if prev_set or curr_set:
                jaccard = len(prev_set & curr_set) / len(prev_set | curr_set)
            else:
                jaccard = 1.0
            w["jaccard_vs_prev"] = round(jaccard, 4)

    # Remove frozensets before returning (not JSON-serializable)
    for w in windows:
        w.pop("edge_set", None)

    return windows
