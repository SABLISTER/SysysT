from process.synthesizer import (
    build_evidence_network,
    build_hypothesis_explanation,
    render_hypothesis_explanation_markdown,
)


def _sample_clusters():
    papers = [
        {
            "id": "p1",
            "title": "Mechanism A improves outcome B",
            "year": 2024,
            "relevance_score": 5,
            "citations": 12,
            "claims": {
                "findings": ["Mechanism A is associated with outcome B"],
                "mechanisms": ["mechanism A"],
                "conditions": ["population C"],
                "brain_regions": ["region D"],
                "methodology": "randomized controlled trial",
                "supports_primary_question": True,
                "key_quote": "Mechanism A was associated with outcome B.",
            },
        },
        {
            "id": "p2",
            "title": "Mechanism A replication",
            "year": 2023,
            "relevance_score": 4,
            "direction": "supports",
            "claims": {
                "findings": ["Mechanism A replicated in a second cohort"],
                "mechanisms": ["mechanism A"],
                "conditions": ["population C"],
                "methodology": "cohort observational",
                "supports_primary_question": True,
            },
        },
    ]
    return [
        {
            "cluster_id": 0,
            "papers": papers,
            "paper_count": len(papers),
            "mechanisms": ["mechanism A"],
            "conditions": ["population C"],
            "brain_regions": ["region D"],
            "primary_support_count": 2,
        }
    ]


def test_build_hypothesis_explanation_is_deterministic_and_traceable():
    explanation = build_hypothesis_explanation(
        _sample_clusters(),
        [{"cluster_id": 0, "narrative": "Mechanism A converges across studies."}],
        "The review supports mechanism A as a consensus hypothesis.",
        research_config={"natural_language_question": "Does mechanism A explain outcome B?"},
    )

    assert explanation["research_question"] == "Does mechanism A explain outcome B?"
    assert explanation["total_papers"] == 2
    assert explanation["supporting_count"] == 2
    assert explanation["clusters"][0]["top_papers"][0]["title"] == "Mechanism A improves outcome B"
    assert explanation["top_mechanisms"][0] == {"term": "mechanism A", "count": 2}
    assert "GRADE output was not available" in " ".join(explanation["caveats"])


def test_render_hypothesis_explanation_markdown_includes_core_sections():
    explanation = build_hypothesis_explanation(
        _sample_clusters(),
        [],
        "The review supports mechanism A as a consensus hypothesis.",
        research_config={"natural_language_question": "Does mechanism A explain outcome B?"},
    )
    markdown = render_hypothesis_explanation_markdown(explanation)

    assert "# Why This Hypothesis?" in markdown
    assert "## Evidence Basis" in markdown
    assert "Mechanism A improves outcome B" in markdown
    assert "## Deterministic Caveats" in markdown


def test_build_evidence_network_uses_general_schema_and_stable_ids():
    network = build_evidence_network(_sample_clusters())

    assert network["kind"] == "general_evidence_network"
    assert "nodes" in network
    assert "edges" in network

    node_ids = {node["id"] for node in network["nodes"]}
    assert "cluster:0" in node_ids
    assert "mechanism:mechanism_a" in node_ids
    assert "condition:population_c" in node_ids

    edge_types = {edge["type"] for edge in network["edges"]}
    assert "belongs_to" in edge_types
    assert "mentions" in edge_types
    assert "supports" in edge_types
