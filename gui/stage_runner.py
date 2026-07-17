"""Stage runner — async wrappers for each pipeline stage.

All 11 stages are fully wired to their stage modules in stages/.
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> list[dict]:
    """Load a JSON file and return its contents."""
    logger.info("Loading JSON: %s", path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    logger.info("  Loaded %d items from %s (%d bytes)", len(data), path, path.stat().st_size)
    return data


def _stage_banner(stage_num: int, stage_name: str) -> None:
    """Log a prominent stage boundary banner."""
    logger.info("═" * 60)
    logger.info("STAGE %d: %s", stage_num, stage_name.upper())
    logger.info("═" * 60)


# ---------------------------------------------------------------------------
# Stage 1: Search (delegates to search_runner)
# ---------------------------------------------------------------------------
async def run_search(
    config,
    cancel_event: Optional[asyncio.Event] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kw,
) -> list[dict]:
    """Stage 1: Literature search — delegates to search_runner."""
    from gui.search_runner import run_gui_search, ProviderChoice

    _stage_banner(1, "LITERATURE SEARCH")

    nl_question = kw.get("natural_language_question", "")
    semantic_terms = kw.get("semantic_terms", [nl_question] if nl_question else [])
    boolean_terms = kw.get("boolean_terms", [])
    provider = kw.get("provider", ProviderChoice.ALL)
    max_results = kw.get("max_results", 500)

    return await run_gui_search(
        natural_language_question=nl_question,
        semantic_terms=semantic_terms,
        boolean_terms=boolean_terms,
        provider=provider,
        config=config,
        max_results=max_results,
        progress_callback=progress_callback,
    )


# ---------------------------------------------------------------------------
# Stage 2: Score
# ---------------------------------------------------------------------------
async def run_score(
    config,
    threshold: int = 3,
    corpus_path: Optional[Path] = None,
    workload: str = "abstract",
    batch_size: Optional[int] = None,
    cancel_event: Optional[asyncio.Event] = None,
    output_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kw,
) -> list[dict]:
    """Stage 2: LLM relevance scoring."""
    import stages.s2_score as s2

    _stage_banner(2, "LLM RELEVANCE SCORING")
    logger.info("  Provider: %s", config.llm_provider)
    logger.info("  Model: %s", config.llm_model)
    logger.info("  Threshold: %d", threshold)
    logger.info("  Workload: %s", workload)
    logger.info("  Workers: %d", getattr(config, "concurrent_workers", 1))

    corpus = None
    if corpus_path:
        corpus = _load_json(corpus_path)

    config.relevance_threshold = threshold
    result = s2.run(
        config,
        corpus=corpus,
        workload=workload,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 2c: Statistical Evidence Extraction
# ---------------------------------------------------------------------------
async def run_statistical_extract(
    config,
    cancel_event: Optional[asyncio.Event] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kw,
) -> dict:
    """Stage 2c: deterministic statistical evidence extraction."""
    import stages.s2c_statistical_extract as s2c

    _stage_banner(2, "STATISTICAL EVIDENCE EXTRACTION")
    logger.info("  Claims dir: %s", config.claims_dir)
    logger.info("  Extra kwargs: %s", list(kw.keys()))

    result = s2c.run(
        config=config,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        **kw,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 3: Extract
# ---------------------------------------------------------------------------
async def run_extract(
    config,
    cancel_event: Optional[asyncio.Event] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kw,
) -> dict:
    """Stage 3: Claim extraction."""
    import stages.s3_extract as s3
    from process.llm import set_config

    _stage_banner(3, "CLAIM EXTRACTION")
    logger.info("  Provider: %s", config.llm_provider)
    logger.info("  Model: %s", config.llm_model)
    logger.info("  Max tokens: %s", kw.get("max_tokens", 4096))
    logger.info("  Use thinking: %s", kw.get("use_thinking", False))

    set_config(config)

    result = s3.run(
        config=config,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        **kw,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 4: Validate
# ---------------------------------------------------------------------------
async def run_validate(
    config,
    cancel_event: Optional[asyncio.Event] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kw,
) -> dict:
    """Stage 4: Method review & span validation."""
    import stages.s4_validate as s4
    from process.llm import set_config

    _stage_banner(4, "METHOD REVIEW & SPAN VALIDATION")
    logger.info("  Provider: %s", config.llm_provider)
    logger.info("  Model: %s", config.llm_model)

    set_config(config)

    result = s4.run(
        config=config,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        **kw,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 5: Audit
# ---------------------------------------------------------------------------
async def run_audit(
    config,
    cancel_event: Optional[asyncio.Event] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kw,
) -> dict:
    """Stage 5: Span audit & quality filtering."""
    import stages.s5_audit as s5

    _stage_banner(5, "SPAN AUDIT & QUALITY FILTERING")
    logger.info("  Claims dir: %s", config.claims_dir)
    logger.info("  Data dir:   %s", config.data_dir)

    result = s5.run(
        config=config,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        **kw,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 6: Synthesize
# ---------------------------------------------------------------------------
async def run_synthesize(
    config,
    cancel_event: Optional[asyncio.Event] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kw,
) -> dict:
    """Stage 6: Cluster-level synthesis — cluster, narrate, draft abstract."""
    import stages.s6_synthesize as s6
    from process.llm import set_config

    _stage_banner(6, "SYNTHESIS (CLUSTER · NARRATE · ABSTRACT)")
    logger.info("  Provider: %s", config.llm_provider)
    logger.info("  Model: %s", config.llm_model)
    logger.info("  Extra kwargs: %s", list(kw.keys()))

    set_config(config)

    result = s6.run(
        config=config,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        **kw,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 7: Fulltext retrieval
# ---------------------------------------------------------------------------
async def run_fulltext(
    config,
    cancel_event: Optional[asyncio.Event] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kw,
) -> dict:
    """Stage 7: Full-text retrieval & metric extraction."""
    import stages.s7_fulltext as s7

    _stage_banner(7, "FULL-TEXT RETRIEVAL & METRIC EXTRACTION")
    logger.info("  Data dir: %s", config.data_dir)
    logger.info("  Extra kwargs: %s", list(kw.keys()))

    result = s7.run(
        config=config,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        **kw,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 2b / 8: Corpus full-text re-score
# ---------------------------------------------------------------------------
async def run_corpus_fulltext(
    config,
    cancel_event: Optional[asyncio.Event] = None,
    **kw,
) -> list[dict]:
    """Stage 2b: Corpus full-text enrichment before scoring."""
    logger.info("═" * 60)
    logger.info("STAGE 2b: CORPUS FULL-TEXT ENRICHMENT")
    logger.info("═" * 60)
    try:
        import stages.s2b_fulltext as s2b
        result = await asyncio.to_thread(
            lambda: s2b.run(config=config, cancel_event=cancel_event, **kw)
        )
        return result
    except Exception as exc:
        logger.error("Corpus fulltext failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Stage 8: Interrater
# ---------------------------------------------------------------------------
async def run_interrater(
    config,
    cancel_event: Optional[asyncio.Event] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    sample_size: int = 100,
    second_provider: str = "ollama",
    second_model: Optional[str] = None,
    **kw,
) -> dict:
    """Stage 8: Inter-rater reliability — cross-model validation."""
    import copy
    import stages.s8_interrater as s8
    from process.llm import set_config

    _stage_banner(8, "INTER-RATER RELIABILITY")
    logger.info("  Primary provider: %s / %s", config.llm_provider, config.llm_model)
    logger.info("  Second provider: %s / %s", second_provider, second_model or "default")
    logger.info("  Sample size: %d", sample_size)

    # Configure for the second provider
    second_cfg = copy.deepcopy(config)
    _embedding_raters = {"brt-bert", "brt_bert", "brt-irr", "brt_irr"}
    if second_provider not in _embedding_raters:
        second_cfg.llm_provider = second_provider
        if second_model:
            if second_provider == "ollama":
                second_cfg.ollama_model = second_model
            elif second_provider == "openai":
                second_cfg.openai_model = second_model
            elif second_provider == "longcat":
                second_cfg.longcat_model = second_model
            elif second_provider == "anthropic":
                second_cfg.anthropic_model = second_model
            elif second_provider == "lmstudio":
                second_cfg.lmstudio_model = second_model

    set_config(second_cfg)

    result = s8.run(
        config=config,
        sample_size=sample_size,
        second_provider=second_provider,
        second_model=second_model or second_cfg.llm_model,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        **kw,
    )

    # Restore original config
    set_config(config)
    return result


# ---------------------------------------------------------------------------
# Stage 9: Robustness
# ---------------------------------------------------------------------------
async def run_robustness(
    config,
    cancel_event: Optional[asyncio.Event] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kw,
) -> dict:
    """Stage 9: Robustness analyses — bootstrap, split-half, sensitivity,
    permutation test, and condition–brain-region network analysis."""
    import stages.s9_robustness as s9

    _stage_banner(9, "ROBUSTNESS ANALYSES")
    logger.info("  Data dir: %s", config.data_dir)
    logger.info("  Extra kwargs: %s", list(kw.keys()))

    result = s9.run(
        config=config,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        **kw,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 10: Verify
# ---------------------------------------------------------------------------
async def run_verify(
    config,
    cancel_event: Optional[asyncio.Event] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kw,
) -> dict:
    """Stage 10: Abstract number verification — fact-check synthesised abstracts."""
    import stages.s10_verify as s10

    _stage_banner(10, "ABSTRACT NUMBER VERIFICATION")
    logger.info("  Data dir: %s", config.data_dir)
    logger.info("  Extra kwargs: %s", list(kw.keys()))

    result = s10.run(
        config=config,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        **kw,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 11: Compare Runs
# ---------------------------------------------------------------------------
async def run_compare_runs(
    config,
    cancel_event: Optional[asyncio.Event] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    claims_paths: Optional[list] = None,
    comparison_name: str = "run_compare",
    **kw,
) -> dict:
    """Stage 11: Compare pipeline runs."""
    import stages.s11_compare_runs as s11

    _stage_banner(11, "COMPARE RUNS")
    logger.info("  Claims paths: %s", claims_paths)
    logger.info("  Comparison name: %s", comparison_name)

    result = s11.run(
        config=config,
        claims_paths=claims_paths,
        comparison_name=comparison_name,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        **kw,
    )
    return result


# ---------------------------------------------------------------------------
# Stage Registry
# ---------------------------------------------------------------------------
# Each entry: stage_name -> (async_func, is_async_flag)
_STAGE_REGISTRY: dict[str, tuple[Callable, bool]] = {}


def _register_stages() -> None:
    """Populate the global _STAGE_REGISTRY dict."""
    global _STAGE_REGISTRY
    _STAGE_REGISTRY = {
        "search": (
            lambda config, cancel_event=None, **kw: run_search(config, cancel_event=cancel_event, **kw),
            True,
        ),
        "score": (
            lambda config, cancel_event=None, **kw: run_score(config, cancel_event=cancel_event, **kw),
            True,
        ),
        "statistical_extract": (
            lambda config, cancel_event=None, **kw: run_statistical_extract(config, cancel_event=cancel_event, **kw),
            True,
        ),
        "extract": (
            lambda config, cancel_event=None, **kw: run_extract(config, cancel_event=cancel_event, **kw),
            True,
        ),
        "validate": (
            lambda config, cancel_event=None, **kw: run_validate(config, cancel_event=cancel_event, **kw),
            True,
        ),
        "audit": (
            lambda config, cancel_event=None, **kw: run_audit(config, cancel_event=cancel_event, **kw),
            True,
        ),
        "synthesize": (
            lambda config, cancel_event=None, **kw: run_synthesize(config, cancel_event=cancel_event, **kw),
            True,
        ),
        "fulltext": (
            lambda config, cancel_event=None, **kw: run_fulltext(config, cancel_event=cancel_event, **kw),
            True,
        ),
        "corpus_fulltext": (
            lambda config, cancel_event=None, **kw: run_corpus_fulltext(config, cancel_event=cancel_event, **kw),
            True,
        ),
        "interrater": (
            lambda config, cancel_event=None, **kw: run_interrater(config, cancel_event=cancel_event, **kw),
            True,
        ),
        "robustness": (
            lambda config, cancel_event=None, **kw: run_robustness(config, cancel_event=cancel_event, **kw),
            True,
        ),
        "verify": (
            lambda config, cancel_event=None, **kw: run_verify(config, cancel_event=cancel_event, **kw),
            True,
        ),
        "compare_runs": (
            lambda config, cancel_event=None, **kw: run_compare_runs(config, cancel_event=cancel_event, **kw),
            True,
        ),
    }


# Auto-register on import
_register_stages()
