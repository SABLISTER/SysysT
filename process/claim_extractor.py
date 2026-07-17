"""LLM-based claim extraction for the Systes pipeline."""
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

from core.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_EVERY = 1

EXTRACTION_PROMPT = """You are extracting structured research claims from an academic paper for a systematic evidence review.

Research question:
{research_question}

Paper title: {title}
Abstract: {abstract}

Extract the following structured information as a JSON object:
{{
    "findings": ["list of key findings/results stated in the paper"],
    "mechanisms": ["list of mechanisms, constructs, or explanatory factors discussed"],
    "conditions": ["list of populations, conditions, constructs, or contextual factors mentioned"],
    "brain_regions": ["list of brain regions, networks, or cognitive systems mentioned when relevant"],
    "methodology": "brief description of study design",
    "supports_primary_question": true/false,
    "supports_criticality": true/false,
    "network_effects_described": true/false,
    "comorbidity_link": "description of relationships among the primary topic, populations, constructs, or contextual factors",
    "key_quote": "the single most relevant sentence from the abstract"
}}

Configured extraction guidance:
{extraction_guidance}

Use supports_primary_question for the configured research question above.
Keep supports_criticality false unless the paper explicitly discusses criticality.
Respond with ONLY the JSON object, no other text.
"""

_EMPTY_CLAIMS = {
    "findings": [],
    "mechanisms": [],
    "conditions": [],
    "brain_regions": [],
    "methodology": "",
    "supports_primary_question": False,
    "supports_criticality": False,
    "network_effects_described": False,
    "comorbidity_link": "",
    "key_quote": "",
    "error": None,
}

_CHECKPOINT_FILENAME = "extract_checkpoint.json"


def _claim_think_mode(model: str, enabled: bool = True) -> bool | str | None:
    """Prefer lightweight exposed reasoning for models that support it."""
    if not enabled:
        return None
    base = model.split(":", 1)[0].strip().lower()
    if base.startswith("gpt-oss"):
        return "low"
    return True


# ---------------------------------------------------------------------------
# Single-paper extraction
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


def _format_extraction_guidance(research_config: dict | None) -> str:
    if not isinstance(research_config, dict):
        return "- Use the generic schema descriptions above."
    extraction = research_config.get("extraction")
    if not isinstance(extraction, dict):
        return "- Use the generic schema descriptions above."
    fields = extraction.get("fields")
    if not isinstance(fields, list):
        return "- Use the generic schema descriptions above."

    lines: list[str] = []
    for field in fields:
        if not isinstance(field, dict):
            continue
        name = str(field.get("name") or "").strip()
        description = str(field.get("description") or "").strip()
        if name and description:
            lines.append(f"- {name}: {description}")
    return "\n".join(lines) if lines else "- Use the generic schema descriptions above."


def build_extraction_prompt(
    title: str,
    abstract: str,
    research_config: dict | None = None,
) -> str:
    """Build the configured claim-extraction prompt for one paper."""
    return EXTRACTION_PROMPT.format(
        research_question=_research_question_from_config(research_config),
        title=title,
        abstract=abstract[:8000],
        extraction_guidance=_format_extraction_guidance(research_config),
    )


def extract_claims(model: str, paper: dict, think=None,
                   max_tokens: int = 4096,
                   research_config: dict | None = None) -> dict:
    """Extract structured claims from a single paper using the LLM."""
    from process.llm import chat as llm_chat

    title = paper.get("title", "Unknown")
    abstract = paper.get("abstract", "")
    if not abstract.strip():
        logger.warning("[Extract] No abstract for '%s' — using heuristic fallback",
                       title[:60])
        return {**_EMPTY_CLAIMS, "error": "no_abstract"}

    prompt = build_extraction_prompt(title, abstract, research_config)
    messages = [{"role": "user", "content": prompt}]

    logger.info("[Extract] Extracting claims: %s", title[:80])
    response = llm_chat(model, messages, temperature=0, max_tokens=max_tokens,
                        think=think)
    result = _parse_extraction_response(response)

    n_findings = len(result.get("findings", []))
    n_mechanisms = len(result.get("mechanisms", []))
    n_regions = len(result.get("brain_regions", []))
    logger.info("[Extract] Result: %d findings, %d mechanisms, %d regions",
                n_findings, n_mechanisms, n_regions)
    return result


def _parse_extraction_response(text: str) -> dict:
    """Parse LLM response into claims dict."""
    # Strip markdown code fences if present
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    # Try direct JSON parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return _normalize_claims(data)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object in the text
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                return _normalize_claims(data)
        except json.JSONDecodeError:
            pass

    # Last resort: try to extract individual fields with regex
    logger.warning("[Extract] Could not parse JSON from LLM response, "
                   "attempting field-level regex extraction")
    result = dict(_EMPTY_CLAIMS)
    result["error"] = "parse_failed"

    # Extract list fields
    for field in ("findings", "mechanisms", "conditions", "brain_regions"):
        pattern = rf'"{field}"\s*:\s*\[(.*?)\]'
        m = re.search(pattern, text, re.DOTALL)
        if m:
            items = re.findall(r'"([^"]+)"', m.group(1))
            result[field] = items

    # Extract string fields
    for field in ("methodology", "comorbidity_link", "key_quote"):
        pattern = rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"'
        m = re.search(pattern, text, re.DOTALL)
        if m:
            result[field] = m.group(1)

    # Extract boolean fields
    for field in ("supports_primary_question", "supports_criticality",
                  "network_effects_described"):
        pattern = rf'"{field}"\s*:\s*(true|false)'
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            result[field] = m.group(1).lower() == "true"

    return result


def _normalize_claims(data: dict) -> dict:
    """Ensure all expected fields exist with correct types."""
    result = dict(_EMPTY_CLAIMS)
    for key in ("findings", "mechanisms", "conditions", "brain_regions"):
        val = data.get(key)
        if isinstance(val, list):
            result[key] = [str(v) for v in val if v]
    for key in ("methodology", "comorbidity_link", "key_quote"):
        val = data.get(key)
        if isinstance(val, str):
            result[key] = val
    for key in ("supports_primary_question", "supports_criticality",
                "network_effects_described"):
        val = data.get(key)
        if isinstance(val, bool):
            result[key] = val
    result["error"] = None
    return result


# ---------------------------------------------------------------------------
# Batch extraction with checkpoint/resume
# ---------------------------------------------------------------------------

def extract_all_claims(
    papers: list[dict],
    model: str,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    cancel_event: Any = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    use_thinking: bool = False,
    max_tokens: int = 4096,
    research_config: dict | None = None,
    **kw,
) -> tuple[list[dict], bool]:
    """Extract claims from all papers with checkpoint/resume.

    Returns (papers_with_claims, was_cancelled).
    """
    total = len(papers)
    logger.info("extract_all_claims: %d papers, model=%s", total, model)

    think = True if use_thinking else None

    # Resume from checkpoint
    ckpt = CheckpointManager(checkpoint_dir, _CHECKPOINT_FILENAME)
    extracted, start_idx = ckpt.load()
    if start_idx > 0 and start_idx <= total:
        logger.info("Resuming extraction from paper %d/%d", start_idx + 1, total)
        if progress_callback:
            progress_callback(
                f"Resuming extraction from paper {start_idx + 1}/{total}")

    # Rate limiting — default 2s between LLM calls
    stagger_delay = kw.get("stagger_delay", 2.0)
    logger.info("Stagger delay between LLM calls: %.1fs", stagger_delay)

    was_cancelled = False
    for i in range(start_idx, total):
        # Check cancellation
        if cancel_event is not None:
            is_set = (cancel_event.is_set()
                      if hasattr(cancel_event, "is_set") else bool(cancel_event))
            if is_set:
                logger.info("Extraction cancelled at paper %d/%d", i + 1, total)
                was_cancelled = True
                break

        paper = papers[i]
        title = paper.get("title", "?")[:80]
        progress_msg = f"[{i + 1}/{total}] Extracting: {title}"
        logger.info(progress_msg)
        if progress_callback:
            progress_callback(progress_msg)

        # Notify waiting for LLM
        if progress_callback:
            progress_callback(f"[{i + 1}/{total}] Waiting for LLM response...")

        try:
            t0 = time.time()
            claims = extract_claims(
                model,
                paper,
                think=think,
                max_tokens=max_tokens,
                research_config=research_config,
            )
            elapsed = time.time() - t0
        except Exception as exc:
            logger.error("[Extract] Error extracting paper %d '%s': %s",
                         i + 1, title, exc)
            claims = {**_EMPTY_CLAIMS, "error": str(exc)}
            elapsed = 0

        # Merge claims into paper dict
        extracted_paper = {**paper, "claims": claims}
        extracted.append(extracted_paper)

        n_f = len(claims.get("findings", []))
        n_m = len(claims.get("mechanisms", []))
        n_r = len(claims.get("brain_regions", []))
        result_msg = (f"[{i + 1}/{total}] {title} → "
                      f"{n_f} findings, {n_m} mechanisms, {n_r} regions ({elapsed:.1f}s)")
        logger.info(result_msg)
        if progress_callback:
            progress_callback(result_msg)

        # Checkpoint
        if (i + 1) % checkpoint_every == 0:
            ckpt.force_save(extracted, total)

        # Rate limit delay between calls (skip after last paper)
        if stagger_delay > 0 and i < total - 1:
            time.sleep(stagger_delay)

    # Final checkpoint if cancelled
    if was_cancelled:
        if not ckpt.force_save(extracted, total):
            logger.warning("Extraction cancelled but checkpoint save failed.")

    # Clear checkpoint on successful completion
    if not was_cancelled:
        ckpt.cleanup()

    logger.info("Extraction complete: %d/%d papers processed%s",
                len(extracted), total,
                " (cancelled)" if was_cancelled else "")
    return extracted, was_cancelled
