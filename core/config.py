"""
Pipeline configuration dataclass.

Full runtime configuration for the hypothesis consensus analyzer pipeline.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

DEFAULT_HAIKU_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_INTERRATER_PROVIDER = "ollama"
DEFAULT_INTERRATER_OLLAMA_MODEL = "qwen3:14b"
DEFAULT_LMSTUDIO_BASE_URL = "http://localhost:1234"
DEFAULT_LMSTUDIO_MODEL = "openai/gpt-oss-20b"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_LONGCAT_BASE_URL = "https://api.longcat.chat/openai/v1"
DEFAULT_LONGCAT_MODEL = "LongCat-2.0-Preview"
DEFAULT_ANTHROPIC_MODEL = DEFAULT_HAIKU_MODEL
DEFAULT_INTERRATER_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_CONCURRENT_WORKERS = 3
DEFAULT_STAGGER_DELAY = 2.0
DEFAULT_CHECKPOINT_EVERY = 1
DEFAULT_WORKER_TIMEOUT_SECONDS = 120.0

# Maps provider name → Config attribute that holds its model string.
# Used in llm_model and anywhere a provider→model field lookup is needed.
PROVIDER_MODEL_FIELD: dict[str, str] = {
    "ollama": "ollama_model",
    "lmstudio": "lmstudio_model",
    "anthropic": "anthropic_model",
    "longcat": "longcat_model",
    "openai": "openai_model",
}
DEFAULT_OA_LOOKUP_WORKERS = 4
DEFAULT_FULLTEXT_DOWNLOAD_WORKERS = 3
DEFAULT_CITATION_SNOWBALL_SEED_LIMIT = 12
DEFAULT_CITATION_SNOWBALL_RELATION_LIMIT = 20
DEFAULT_SCORE_USE_THINKING = True
DEFAULT_SCORE_MAX_TOKENS = 4096
DEFAULT_SCORE_BATCH_MAX_TOKENS = 12288
DEFAULT_SCORE_ABSTRACT_CHARS = 8000
DEFAULT_SCORE_FULLTEXT_CHARS = 18000
DEFAULT_CLAIM_USE_THINKING = False
DEFAULT_CLAIM_MAX_TOKENS = 4096
DEFAULT_SYNTHESIS_USE_THINKING = False
DEFAULT_SYNTHESIS_CLUSTER_MAX_TOKENS = 4096
DEFAULT_SYNTHESIS_ABSTRACT_MAX_TOKENS = 8192

LEGACY_PROVIDER_ALIASES: dict[str, str] = {}


def normalize_provider_name(provider: str) -> str:
    lower = provider.lower().strip()
    return LEGACY_PROVIDER_ALIASES.get(lower, lower)


def _safe_run_name(name: str) -> str:
    """Sanitize a run name for use as a filesystem directory name."""
    import re
    clean = str(name or "").strip()
    clean = re.sub(r"[^\w\-. ]+", "_", clean)
    clean = re.sub(r"_+", "_", clean).strip("_. ")
    return clean


def _env_flag(name: str, default: bool) -> bool:
    val = os.environ.get(name, "")
    if val == "":
        return default
    return val.lower() not in ("0", "false", "no", "off", "")


@dataclass
class Config:
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    run_name: str = ""
    data_dir: Path = field(init=False)
    raw_dir: Path = field(init=False)
    corpus_dir: Path = field(init=False)
    claims_dir: Path = field(init=False)
    clusters_dir: Path = field(init=False)
    output_dir: Path = field(init=False)

    # API Keys
    s2_api_key: str = ""
    openalex_api_key: str = ""
    pubmed_api_key: str = ""
    pubmed_email: str = ""
    pubmed_tool: str = "hypothesis-consensus-analyzer"
    wos_api_key: str = ""
    wos_researcher_api_key: str = ""
    scopus_api_key: str = ""
    scopus_insttoken: str = ""
    elicit_api_key: str = ""
    openai_api_key: str = ""
    longcat_api_key: str = ""
    anthropic_api_key: str = ""

    # LLM Provider Settings
    llm_provider: str = "openai"
    ollama_model: str = "qwen2.5:32b-instruct-q4_K_M"
    ollama_fallback_model: str = "llama3:8b-instruct-q8_0"
    ollama_base_url: str = "http://localhost:11434"
    lmstudio_api_key: str = ""
    lmstudio_base_url: str = DEFAULT_LMSTUDIO_BASE_URL
    lmstudio_model: str = DEFAULT_LMSTUDIO_MODEL
    openai_base_url: str = DEFAULT_OPENAI_BASE_URL
    openai_model: str = DEFAULT_OPENAI_MODEL
    longcat_base_url: str = DEFAULT_LONGCAT_BASE_URL
    longcat_model: str = DEFAULT_LONGCAT_MODEL
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL

    # Per-workload overrides
    abstract_llm_provider: str = ""
    abstract_ollama_model: str = ""
    abstract_lmstudio_model: str = ""
    abstract_openai_model: str = ""
    abstract_longcat_model: str = ""
    abstract_anthropic_model: str = ""
    fulltext_llm_provider: str = ""
    fulltext_ollama_model: str = ""
    fulltext_lmstudio_model: str = ""
    fulltext_openai_model: str = ""
    fulltext_longcat_model: str = ""
    fulltext_anthropic_model: str = ""

    # Processing
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    relevance_model: str = "brt-bert"
    relevance_threshold: int = 3
    min_cluster_size: int = 5
    score_batch_size: int = 1

    # Concurrency
    concurrent_workers: int = DEFAULT_CONCURRENT_WORKERS
    stagger_delay: float = DEFAULT_STAGGER_DELAY
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    worker_timeout_seconds: float = DEFAULT_WORKER_TIMEOUT_SECONDS
    oa_lookup_workers: int = DEFAULT_OA_LOOKUP_WORKERS
    fulltext_download_workers: int = DEFAULT_FULLTEXT_DOWNLOAD_WORKERS

    # Citation Snowballing
    citation_snowball_enabled: bool = True
    citation_snowball_seed_limit: int = DEFAULT_CITATION_SNOWBALL_SEED_LIMIT
    citation_snowball_relation_limit: int = DEFAULT_CITATION_SNOWBALL_RELATION_LIMIT

    # LLM Budget / Quality
    score_use_thinking: bool = DEFAULT_SCORE_USE_THINKING
    score_max_tokens: int = DEFAULT_SCORE_MAX_TOKENS
    score_batch_max_tokens: int = DEFAULT_SCORE_BATCH_MAX_TOKENS
    score_abstract_chars: int = DEFAULT_SCORE_ABSTRACT_CHARS
    score_fulltext_chars: int = DEFAULT_SCORE_FULLTEXT_CHARS
    claim_use_thinking: bool = DEFAULT_CLAIM_USE_THINKING
    claim_max_tokens: int = DEFAULT_CLAIM_MAX_TOKENS
    synthesis_use_thinking: bool = DEFAULT_SYNTHESIS_USE_THINKING
    synthesis_cluster_max_tokens: int = DEFAULT_SYNTHESIS_CLUSTER_MAX_TOKENS
    synthesis_abstract_max_tokens: int = DEFAULT_SYNTHESIS_ABSTRACT_MAX_TOKENS

    # Research / Hypothesis
    base_term_s2: str = ""
    base_term_pubmed: str = ""
    hypothesis_text: str = (
        "What does the current scholarly evidence say about the user's "
        "research question?"
    )
    co_occurring_conditions: list = field(default_factory=list)
    alignment_dimension: str = "Evidence alignment to the research question"
    _research_config: dict = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):
        self.llm_provider = normalize_provider_name(self.llm_provider)
        self._resolve_paths()

        # Resolve API keys from env
        if not self.s2_api_key:
            self.s2_api_key = os.environ.get("S2_API_KEY", "")
        if not self.pubmed_api_key:
            self.pubmed_api_key = os.environ.get("PUBMED_API_KEY", "")
        if not self.pubmed_email:
            self.pubmed_email = os.environ.get("PUBMED_EMAIL", "")
        if not self.openai_api_key:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.longcat_api_key:
            self.longcat_api_key = os.environ.get("LONGCAT_API_KEY", "")
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.elicit_api_key:
            self.elicit_api_key = os.environ.get("ELICIT_API_KEY", "")
        if not self.scopus_api_key:
            self.scopus_api_key = os.environ.get("SCOPUS_API_KEY", "")
        if not self.wos_researcher_api_key:
            self.wos_researcher_api_key = os.environ.get("WOS_RESEARCHER_API_KEY", "")
        if not self.wos_api_key:
            self.wos_api_key = os.environ.get("WOS_API_KEY", "")

    def _resolve_paths(self) -> None:
        """(Re-)compute all directory paths based on project_root and run_name."""
        safe = _safe_run_name(self.run_name)
        if safe:
            self.data_dir = self.project_root / "data" / "runs" / safe
            self.output_dir = self.project_root / "output" / "runs" / safe
        else:
            self.data_dir = self.project_root / "data"
            self.output_dir = self.project_root / "output"
        self.raw_dir = self.data_dir / "raw"
        self.corpus_dir = self.data_dir / "corpus"
        self.claims_dir = self.data_dir / "claims"
        self.clusters_dir = self.data_dir / "clusters"

    def set_run_name(self, name: str) -> None:
        """Change the run name and re-route all data/output directories."""
        self.run_name = name
        self._resolve_paths()

    @property
    def llm_model(self) -> str:
        field = PROVIDER_MODEL_FIELD.get(self.llm_provider, "openai_model")
        return getattr(self, field)

    def llm_provider_for(self, workload: str = "abstract") -> str:
        override = getattr(self, f"{workload}_llm_provider", "")
        return normalize_provider_name(override) if override else self.llm_provider

    def llm_model_for(self, workload: str = "abstract") -> str:
        prov = self.llm_provider_for(workload)
        override = getattr(self, f"{workload}_{prov}_model", "")
        if override:
            return override
        return self.llm_model

    def runtime_config(self, workload: str = "abstract") -> "Config":
        cfg = copy.deepcopy(self)
        cfg.llm_provider = self.llm_provider_for(workload)
        model = self.llm_model_for(workload)
        setattr(cfg, f"{cfg.llm_provider}_model", model)
        return cfg

    def ensure_dirs(self) -> None:
        for d in [self.raw_dir, self.corpus_dir, self.claims_dir,
                  self.clusters_dir, self.output_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def research_config(self) -> dict:
        return self._research_config

    def get_search_config(self) -> dict:
        return self._research_config.get("search", {})

    def get_relevance_config(self) -> dict:
        return self._research_config.get("relevance", {})

    def get_extraction_fields(self) -> list:
        return self._research_config.get("extraction", {}).get("fields", [])

    def get_synthesis_config(self) -> dict:
        return self._research_config.get("synthesis", {})

    def get_method_review_config(self) -> dict:
        return self._research_config.get("method_review", {})

    def get_audit_config(self) -> dict:
        return self._research_config.get("audit", {})

    def get_natural_language_question(self) -> str:
        nlq = self._research_config.get("natural_language_question", "")
        if not nlq:
            nlq = self.get_relevance_config().get("research_question", "")
        return (nlq or "").strip()

    def get_full_text_config(self) -> dict:
        return self._research_config.get("full_text", {})

    def get_cheap_triage_config(self) -> dict:
        return self._research_config.get("cheap_triage", {})

    def get_llm_scoring_config(self) -> dict:
        return self._research_config.get("llm_scoring", {})

    def get_defaults_config(self) -> dict:
        return self._research_config.get("defaults", {})

    def get_axes_config(self) -> list:
        return self._research_config.get("axes", [])

    def get_dive_config(self) -> dict:
        return self._research_config.get("dive", {})

    def get_stopping_config(self) -> dict:
        return self._research_config.get("stopping", {})

    @classmethod
    def load_config(
        cls,
        config_path: Optional[Path | str] = None,
        *,
        project_root: Optional[Path | str] = None,
    ) -> "Config":
        root = (
            Path(project_root).expanduser().resolve()
            if project_root else Path(__file__).parent.parent
        )
        if config_path is None:
            for name in ("research_config.yaml", "research_config.yml",
                         "research_config.json"):
                p = root / name
                if p.exists():
                    config_path = p
                    break
        if config_path is None:
            return cls(project_root=root)

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        suffix = config_path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
                with open(config_path) as f:
                    data = yaml.safe_load(f) or {}
            except ImportError:
                raise ImportError("PyYAML is required for YAML config files")
        elif suffix == ".json":
            with open(config_path) as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")

        if project_root is None and data.get("project_root"):
            configured_root = Path(str(data["project_root"])).expanduser()
            if configured_root.is_absolute():
                root = configured_root.resolve()
            else:
                root = (config_path.parent / configured_root).resolve()

        # Sections stored in _research_config (research-domain data,
        # not direct Config dataclass fields).
        _RESEARCH_SECTIONS = {
            "search", "relevance", "extraction", "synthesis",
            "method_review", "audit",
            "natural_language_question", "name", "description",
            "query_families", "defaults", "cheap_triage", "full_text",
            "axes", "overlap_rules", "tier_rules", "evidence_types",
            "stopping", "seeds", "dive", "graph", "graph_analysis",
            "llm_scoring", "committee_brief", "abct_tech_ai_infosheet",
        }

        cfg_fields = {}
        research = {}
        init_field_names = {
            field_name
            for field_name, field_meta in cls.__dataclass_fields__.items()
            if field_meta.init
        }
        for key, val in data.items():
            if key == "project_root":
                continue
            if key in _RESEARCH_SECTIONS:
                research[key] = val
            elif key in init_field_names:
                cfg_fields[key] = val

        # Map llm_scoring sub-keys to Config dataclass fields
        llm_sc = research.get("llm_scoring") or {}
        if llm_sc.get("abstract_max_chars"):
            cfg_fields.setdefault("score_abstract_chars", llm_sc["abstract_max_chars"])
        if llm_sc.get("fulltext_max_chars"):
            cfg_fields.setdefault("score_fulltext_chars", llm_sc["fulltext_max_chars"])
        if llm_sc.get("max_output_tokens"):
            cfg_fields.setdefault("score_max_tokens", llm_sc["max_output_tokens"])
        if "use_reasoning" in llm_sc:
            cfg_fields.setdefault("score_use_thinking", bool(llm_sc["use_reasoning"]))

        # Map relevance threshold
        rel = research.get("relevance") or {}
        if rel.get("threshold") is not None:
            cfg_fields.setdefault("relevance_threshold", rel["threshold"])

        # Map synthesis embedding model
        syn = research.get("synthesis") or {}
        if syn.get("embedding_model"):
            cfg_fields.setdefault("embedding_model", syn["embedding_model"])

        # Map defaults.per_source_limit to citation snowball seed limit if present
        defaults = research.get("defaults") or {}
        if defaults.get("per_source_limit"):
            cfg_fields.setdefault("citation_snowball_seed_limit",
                                  min(defaults["per_source_limit"], 500))

        # Derive run_name from the config's 'name' field
        if "run_name" not in cfg_fields and research.get("name"):
            cfg_fields["run_name"] = str(research["name"]).strip()

        cfg = cls(project_root=root, **cfg_fields)
        cfg._research_config = research
        return cfg

    def __repr__(self) -> str:
        run = f", run={self.run_name!r}" if self.run_name else ""
        return f"Config(provider={self.llm_provider!r}, model={self.llm_model!r}, project_root={self.project_root!r}{run})"
