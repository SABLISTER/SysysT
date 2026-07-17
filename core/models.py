"""
Data models for the Hypothesis Consensus Analyzer.

Contains enumerations and dataclasses used throughout the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


class EvidenceDirection(Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"
    TANGENTIAL = "tangential"


class SourceType(Enum):
    PUBMED = "PubMed"
    SEMANTIC_SCHOLAR = "Semantic Scholar"
    OPENALEX = "OpenAlex"
    WEB_OF_SCIENCE = "Web of Science"
    SCOPUS = "Scopus"
    ELICIT = "Elicit"
    LOCAL_PDF = "Local PDF"


class SpanStatus(Enum):
    VERIFIED = "verified"
    CLOSE_MATCH = "close_match"
    NOT_FOUND = "not_found"
    UNCHECKED = "unchecked"


class LLMProvider(Enum):
    OPENAI = "openai"
    LONGCAT = "longcat"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    ANTHROPIC = "anthropic"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class PipelineState(Enum):
    IDLE = auto()
    SEARCHING = auto()
    SCORING = auto()
    FULLTEXT_ACQUIRE = auto()
    EXTRACTING = auto()
    VALIDATING = auto()
    AUDITING = auto()
    SYNTHESIZING = auto()
    FULLTEXT_PROCESS = auto()
    INTERRATING = auto()
    ROBUSTNESS = auto()
    VERIFYING = auto()
    COMPARING = auto()
    ERROR = auto()
    CANCELLED = auto()

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").title()

    @property
    def is_running(self) -> bool:
        return self not in (PipelineState.IDLE, PipelineState.ERROR, PipelineState.CANCELLED)


@dataclass
class HypothesisComponent:
    id: str
    label: str
    description: str
    keywords: list[str] = field(default_factory=list)
    weight: float = 1.0


@dataclass
class Article:
    id: str
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    full_text: str = ""
    year: Optional[int] = None
    journal: str = ""
    doi: str = ""
    source: SourceType = SourceType.PUBMED
    citations: int = 0
    url: str = ""
    pmid: str = ""
    keywords: list[str] = field(default_factory=list)
    has_full_text: bool = False
    open_access: bool = False
    full_text_url: str = ""

    @property
    def display_authors(self) -> str:
        if len(self.authors) > 3:
            return ", ".join(self.authors[:3]) + " et al."
        return ", ".join(self.authors)

    @property
    def text_content(self) -> str:
        return self.full_text if self.full_text else self.abstract

    @property
    def text_type(self) -> str:
        if self.full_text:
            return "Full text"
        if self.abstract:
            return "Abstract only"
        return "No text"


@dataclass
class QuoteSpan:
    text: str
    component_id: str = ""
    direction: EvidenceDirection = EvidenceDirection.NEUTRAL
    relevance_note: str = ""
    status: SpanStatus = SpanStatus.UNCHECKED
    matched_text: str = ""
    match_similarity: float = 0.0
    char_offset_start: int = -1
    char_offset_end: int = -1


@dataclass
class LLMScoringResult:
    article_id: str
    component_id: str
    llm_relevance: float = 0.0
    llm_direction: EvidenceDirection = EvidenceDirection.NEUTRAL
    llm_reasoning: str = ""
    quote_spans: list[QuoteSpan] = field(default_factory=list)
    model_name: str = ""
    provider: LLMProvider = LLMProvider.OPENAI
    spans_verified: int = 0
    spans_close: int = 0
    spans_not_found: int = 0
    spans_total: int = 0
    verification_score: float = 0.0

    @property
    def is_reliable(self) -> bool:
        if self.spans_total == 0:
            return False
        return (self.spans_verified + self.spans_close) / self.spans_total >= 0.6


@dataclass
class EvidenceLink:
    article_id: str
    component_id: str
    direction: EvidenceDirection = EvidenceDirection.NEUTRAL
    relevance_score: float = 0.0
    confidence: float = 0.0
    key_excerpts: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    llm_result: Optional[LLMScoringResult] = None

    @property
    def combined_score(self) -> float:
        if self.llm_result and self.llm_result.is_reliable:
            return 0.4 * self.relevance_score + 0.6 * self.llm_result.llm_relevance
        return self.relevance_score

    @property
    def combined_direction(self) -> EvidenceDirection:
        if self.llm_result and self.llm_result.is_reliable:
            return self.llm_result.llm_direction
        return self.direction


@dataclass
class AlignmentResult:
    component_id: str
    component_label: str
    support_score: float = 0.0
    evidence_count: int = 0
    supporting_count: int = 0
    contradicting_count: int = 0
    neutral_count: int = 0
    confidence: float = 0.0
    top_supporting: list[str] = field(default_factory=list)
    top_contradicting: list[str] = field(default_factory=list)


@dataclass
class TrendPoint:
    year: int
    article_count: int = 0
    avg_support_score: float = 0.0
    cumulative_citations: int = 0
    component_id: str = ""


@dataclass
class LLMConfig:
    provider: LLMProvider = LLMProvider.OPENAI
    model_name: str = "gpt-4o-mini"
    api_key: str = ""
    api_base_url: str = ""
    temperature: float = 0.1
    max_tokens: int = 2000
    batch_size: int = 5
    use_embeddings: bool = False

    @property
    def display_name(self) -> str:
        return f"{self.provider.value} / {self.model_name}"


@dataclass
class ClaimData:
    article_id: str
    findings: list[str] = field(default_factory=list)
    mechanisms: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    brain_regions: list[str] = field(default_factory=list)
    methodology: str = ""
    supports_criticality: bool = False
    network_effects_described: bool = False
    comorbidity_link: str = ""
    key_quote: str = ""
    raw_llm_response: str = ""
    extraction_model: str = ""
    extraction_provider: str = ""

    @property
    def is_empty(self) -> bool:
        return not (self.findings or self.mechanisms or self.conditions
                    or self.brain_regions or self.methodology)


@dataclass
class MethodProfile:
    article_id: str
    study_method_type: str = "unknown"
    confidence: float = 0.0
    evidence_spans: list[str] = field(default_factory=list)
    design_notes: str = ""
    flagged_for_review: bool = False
    sample_size: int = -1
    has_control_group: bool = False
    is_longitudinal: bool = False
    raw_llm_response: str = ""
    classification_model: str = ""

    @property
    def needs_review(self) -> bool:
        return self.flagged_for_review or self.study_method_type == "unknown"

    @property
    def evidence_tier(self) -> str:
        t = self.study_method_type.lower()
        if t in ("rct", "meta-analysis", "systematic review"):
            return "high"
        if t in ("cohort", "case-control"):
            return "moderate"
        if t in ("case report", "expert opinion"):
            return "low"
        return "unknown"


@dataclass
class InterraterResult:
    article_id: str
    scores_by_provider: dict[str, float] = field(default_factory=dict)
    directions_by_provider: dict[str, str] = field(default_factory=dict)
    agreement_score: float = 0.0
    mean_score: float = 0.0
    score_std: float = 0.0
    majority_direction: EvidenceDirection = EvidenceDirection.NEUTRAL
    flagged: bool = False
    notes: str = ""

    @property
    def providers(self) -> list[str]:
        return list(self.scores_by_provider.keys())

    @property
    def high_agreement(self) -> bool:
        return self.agreement_score >= 0.7


@dataclass
class RobustnessResult:
    run_id: str
    n_articles_included: int = 0
    n_articles_excluded: int = 0
    threshold_sensitivity: dict[int, float] = field(default_factory=dict)
    bootstrap_mean: float = 0.0
    bootstrap_ci_lower: float = 0.0
    bootstrap_ci_upper: float = 0.0
    effect_size: float = 0.0
    stability_score: float = 0.0
    provider_sensitivity: dict[str, float] = field(default_factory=dict)
    notes: str = ""
    passed: bool = False

    @property
    def ci_width(self) -> float:
        return self.bootstrap_ci_upper - self.bootstrap_ci_lower

    @property
    def inclusion_rate(self) -> float:
        total = self.n_articles_included + self.n_articles_excluded
        return self.n_articles_included / total if total else 0.0


@dataclass
class AnalysisSession:
    hypothesis_components: list[HypothesisComponent] = field(default_factory=list)
    articles: dict[str, Article] = field(default_factory=dict)
    evidence_links: list[EvidenceLink] = field(default_factory=list)
    alignment_results: list[AlignmentResult] = field(default_factory=list)
    trend_data: list[TrendPoint] = field(default_factory=list)
    total_articles_fetched: int = 0
    total_articles_analyzed: int = 0
    llm_results: dict[str, LLMScoringResult] = field(default_factory=dict)
    llm_config: Optional[LLMConfig] = None
    flagged_spans: list[QuoteSpan] = field(default_factory=list)
    corpus_json_path: str = ""
    fulltext_retrieval_results: list = field(default_factory=list)
    wordfish_model: object = None
    stylometry_profiles: dict = field(default_factory=dict)
    corpus_statistics: dict = field(default_factory=dict)
    pipeline_state: PipelineState = PipelineState.IDLE
    claim_data: dict[str, ClaimData] = field(default_factory=dict)
    method_profiles: dict[str, MethodProfile] = field(default_factory=dict)
    interrater_results: dict[str, InterraterResult] = field(default_factory=dict)
    robustness_result: Optional[RobustnessResult] = None
    run_id: str = ""
    stage_timings: dict[str, float] = field(default_factory=dict)
    raw_search_counts: dict[str, int] = field(default_factory=dict)
    synthesis_clusters: list[dict[str, Any]] = field(default_factory=list)
    synthesis_abstract: str = ""

    @property
    def article_count(self) -> int:
        return len(self.articles)

    @property
    def scored_article_count(self) -> int:
        scored = set()
        for key in self.llm_results:
            aid = key.split(":")[0] if ":" in key else key
            scored.add(aid)
        return len(scored)

    @property
    def extracted_article_count(self) -> int:
        return len(self.claim_data)

    @property
    def flagged_for_review_count(self) -> int:
        return sum(1 for m in self.method_profiles.values() if m.needs_review)

    @property
    def is_pipeline_running(self) -> bool:
        return self.pipeline_state.is_running
