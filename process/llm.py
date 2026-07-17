"""Unified LLM client for Ollama, LM Studio, OpenAI-compatible, LongCat, and Anthropic backends."""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.config import Config

logger = logging.getLogger(__name__)

_config: "Config | None" = None
_openai_client: Any = None
_longcat_client: Any = None
_ollama_client: Any = None
_lmstudio_client: Any = None
_anthropic_client: Any = None
_lmstudio_models_without_reasoning: set[str] = set()

# Heartbeat: print a dot every HEARTBEAT_INTERVAL seconds while waiting,
# and a warning message after HEARTBEAT_WARN seconds.
HEARTBEAT_INTERVAL = 5      # seconds between "still thinking" dots
HEARTBEAT_WARN = 30         # seconds before printing a warning
HEARTBEAT_CRITICAL = 90     # seconds before printing a "likely stuck" warning


def set_config(config: "Config") -> None:
    """Set the active pipeline config so chat() uses the right provider and settings."""
    global _config, _openai_client, _longcat_client, _ollama_client, _lmstudio_client, _anthropic_client
    for client in (_openai_client, _longcat_client, _ollama_client, _lmstudio_client, _anthropic_client):
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    _config = config
    _openai_client = None
    _longcat_client = None
    _ollama_client = None
    _lmstudio_client = None
    _anthropic_client = None
    _lmstudio_models_without_reasoning.clear()


def _append_llm_debug(record: dict) -> None:
    """Append one LLM request/response to data/llm_responses.jsonl for inspection."""
    if _config is None:
        return
    path = _config.data_dir / "llm_responses.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug("Could not write llm_responses.jsonl: %s", e)


_client_lock = threading.Lock()


def _get_openai_client():
    """Lazy-init OpenAI/OpenAI-compatible client (thread-safe)."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    with _client_lock:
        # Double-check under lock
        if _openai_client is not None:
            return _openai_client
        if _config is None or _config.llm_provider != "openai":
            return None
        try:
            from openai import OpenAI
            _openai_client = OpenAI(
                api_key=_config.openai_api_key,
                base_url=_config.openai_base_url.rstrip("/"),
            )
            return _openai_client
        except Exception as e:
            logger.error("OpenAI client init failed: %s", e)
            return None


def _get_longcat_client():
    """Lazy-init LongCat's OpenAI-compatible client (thread-safe)."""
    global _longcat_client
    if _longcat_client is not None:
        return _longcat_client
    with _client_lock:
        if _longcat_client is not None:
            return _longcat_client
        if _config is None or _config.llm_provider != "longcat":
            return None
        try:
            from openai import OpenAI

            _longcat_client = OpenAI(
                api_key=_config.longcat_api_key,
                base_url=_config.longcat_base_url.rstrip("/"),
            )
            return _longcat_client
        except Exception as e:
            logger.error("LongCat client init failed: %s", e)
            return None


def _get_ollama_client():
    """Lazy-init Ollama client bound to the configured host."""
    global _ollama_client
    if _ollama_client is not None:
        return _ollama_client
    with _client_lock:
        if _ollama_client is not None:
            return _ollama_client
        if _config is None or _config.llm_provider != "ollama":
            return None
        try:
            import ollama

            host = _config.ollama_base_url.rstrip("/")
            _ollama_client = ollama.Client(host=host)
            return _ollama_client
        except Exception as e:
            logger.error("Ollama client init failed: %s", e)
            return None


def _normalize_lmstudio_base_url(url: str) -> str:
    """Normalize LM Studio server roots so callers can paste host or API paths."""
    clean = (url or "").strip().rstrip("/")
    for suffix in ("/api/v1", "/v1", "/api"):
        if clean.endswith(suffix):
            clean = clean[: -len(suffix)]
            break
    return clean.rstrip("/")


def _get_lmstudio_client():
    """Lazy-init LM Studio REST client bound to the configured host."""
    global _lmstudio_client
    if _lmstudio_client is not None:
        return _lmstudio_client
    with _client_lock:
        if _lmstudio_client is not None:
            return _lmstudio_client
        if _config is None or _config.llm_provider != "lmstudio":
            return None
        try:
            import httpx

            base_url = _normalize_lmstudio_base_url(_config.lmstudio_base_url)
            headers = {"Content-Type": "application/json"}
            if _config.lmstudio_api_key:
                headers["Authorization"] = f"Bearer {_config.lmstudio_api_key}"
            _lmstudio_client = httpx.Client(
                base_url=base_url,
                headers=headers,
                timeout=httpx.Timeout(connect=30, read=600, write=60, pool=600),
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                    keepalive_expiry=120,
                ),
            )
            return _lmstudio_client
        except Exception as e:
            logger.error("LM Studio client init failed: %s", e)
            return None


def _get_anthropic_client():
    """Lazy-init Anthropic client (thread-safe)."""
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client
    with _client_lock:
        if _anthropic_client is not None:
            return _anthropic_client
        if _config is None or _config.llm_provider != "anthropic":
            return None
        try:
            import anthropic

            _anthropic_client = anthropic.Anthropic(api_key=_config.anthropic_api_key)
            return _anthropic_client
        except Exception as e:
            logger.error("Anthropic client init failed: %s", e)
            return None


def chat(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0,
    max_tokens: int = 2500,
    think: bool | str | None = None,
) -> str:
    """
    Send a chat completion request to the configured LLM provider.
    Returns the assistant message content.
    """
    if _config is None:
        raise RuntimeError("llm.set_config(config) must be called before chat()")
    if _config.llm_provider == "openai":
        return _chat_openai(model, messages, temperature, max_tokens, think)
    if _config.llm_provider == "longcat":
        return _chat_openai(model, messages, temperature, max_tokens, think)
    if _config.llm_provider == "lmstudio":
        return _chat_lmstudio(model, messages, temperature, max_tokens, think)
    if _config.llm_provider == "anthropic":
        return _chat_anthropic(model, messages, temperature, max_tokens, think)
    return _chat_ollama(model, messages, temperature, max_tokens, think)


def _extract_reasoning(message: Any) -> str:
    """Extract the reasoning/thinking chain from a thinking model's response.

    Different providers expose this differently:
    - OpenAI o-series: message.reasoning_content
    - Some providers: message.reasoning or message.thought
    - Content blocks: list with type="thinking" blocks
    Returns empty string if no reasoning found.
    """
    # Direct attributes (most common for thinking models)
    for attr in ("reasoning_content", "reasoning", "thought", "thinking"):
        val = getattr(message, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()

    # Content as list of blocks — look for "thinking" type blocks
    content = getattr(message, "content", None)
    if isinstance(content, list):
        thinking_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                thinking_parts.append((block.get("thinking") or block.get("text") or "").strip())
            elif hasattr(block, "type") and getattr(block, "type", None) == "thinking":
                thinking_parts.append((getattr(block, "thinking", "") or getattr(block, "text", "") or "").strip())
        if thinking_parts:
            return "\n".join(p for p in thinking_parts if p)

    # model_dump() fallback
    if hasattr(message, "model_dump"):
        d = message.model_dump()
        for key in ("reasoning_content", "reasoning", "thought", "thinking"):
            if key in d and isinstance(d[key], str) and d[key].strip():
                return d[key].strip()

    return ""


def _append_thinking_log(record: dict) -> None:
    """Append reasoning chain to data/llm_thinking.jsonl for hallucination auditing."""
    if _config is None:
        return
    path = _config.data_dir / "llm_thinking.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug("Could not write llm_thinking.jsonl: %s", e)


def _extract_message_text(message: Any) -> str:
    """Get plain text from OpenAI-style message; handle content as str or list of parts."""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append((block.get("text") or "").strip())
            elif hasattr(block, "text"):
                parts.append((getattr(block, "text", "") or "").strip())
        return " ".join(p for p in parts if p).strip()
    # Raw dict or other shape (e.g. from model_dump())
    if hasattr(message, "model_dump"):
        d = message.model_dump()
        c = d.get("content")
        if isinstance(c, str):
            return c.strip()
        if isinstance(c, list):
            return " ".join(
                (x.get("text", "") if isinstance(x, dict) else str(x)).strip()
                for x in c
            ).strip()
        for key in ("reasoning", "reasoning_content", "thought", "output"):
            if key in d and isinstance(d[key], str):
                return d[key].strip()
    return ""


def _chat_ollama(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    think: bool | str | None,
) -> str:
    client = _get_ollama_client()
    if client is None:
        raise RuntimeError("Ollama is configured but client failed to initialize.")

    t0 = time.monotonic()
    try:
        response = client.chat(
            model=model,
            messages=messages,
            think=think,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
    except Exception as e:
        err_text = str(e)
        if (
            "Unexpected endpoint or method" in err_text
            and "/api/chat" in err_text
        ):
            raise RuntimeError(
                "The configured host is not speaking the Ollama API at /api/chat. "
                "If this is LM Studio, switch the provider to 'LM Studio (Native API)' "
                "and set the Stage 2 abstract profile to LM Studio as well."
            ) from e
        if think is not None and "think" in str(e).lower():
            logger.warning("Ollama server rejected think=%r; retrying without exposed reasoning", think)
            response = client.chat(
                model=model,
                messages=messages,
                options={"temperature": temperature, "num_predict": max_tokens},
            )
            think = None
        else:
            raise
    elapsed = time.monotonic() - t0

    message = getattr(response, "message", None)
    raw_content = getattr(message, "content", None)
    text = raw_content.strip() if isinstance(raw_content, str) else _extract_message_text(message)
    finish_reason = getattr(response, "done_reason", None)
    eval_count = getattr(response, "eval_count", None)
    prompt_eval_count = getattr(response, "prompt_eval_count", None)
    reasoning = _extract_reasoning(message) if message is not None else ""
    last_msg = (messages[-1] if messages else {}).get("content", "") or ""

    _append_llm_debug({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "provider": "ollama",
        "elapsed_s": round(elapsed, 2),
        "request_preview": last_msg[-500:],
        "raw_content": raw_content if isinstance(raw_content, str) else repr(raw_content),
        "extracted_text": (text[:3000] if text else ""),
        "finish_reason": finish_reason,
        "completion_tokens": eval_count,
        "prompt_tokens": prompt_eval_count,
        "think": think,
    })

    if reasoning:
        _append_thinking_log({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": model,
            "provider": "ollama",
            "elapsed_s": round(elapsed, 2),
            "request_preview": last_msg[:1000],
            "answer": raw_content if isinstance(raw_content, str) else repr(raw_content),
            "reasoning": reasoning,
            "reasoning_tokens": len(reasoning.split()),
            "finish_reason": finish_reason,
            "completion_tokens": eval_count,
            "prompt_tokens": prompt_eval_count,
            "think": think,
        })
    elif think:
        _append_thinking_log({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": model,
            "provider": "ollama",
            "elapsed_s": round(elapsed, 2),
            "request_preview": last_msg[:1000],
            "answer": raw_content if isinstance(raw_content, str) else repr(raw_content),
            "reasoning": "(requested but not exposed by Ollama response)",
            "reasoning_tokens": 0,
            "finish_reason": finish_reason,
            "completion_tokens": eval_count,
            "prompt_tokens": prompt_eval_count,
            "think": think,
        })

    return text


def _lmstudio_reasoning_payload(think: bool | str | None) -> str | None:
    """Translate pipeline think hints into LM Studio reasoning settings."""
    if think is None:
        return None
    if isinstance(think, str):
        effort = think.strip().lower()
        if effort in {"low", "medium", "high", "on"}:
            return effort
        if effort in {"true"}:
            return "on"
        if effort in {"false", "off"}:
            return None
        return "medium"
    if think:
        return "medium"
    return None


def _lmstudio_reasoning_unsupported(exc: Exception) -> bool:
    """Return True when LM Studio rejects the request's reasoning field."""
    response = getattr(exc, "response", None)
    error_payload = {}
    if response is not None:
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                error_payload = parsed.get("error") or {}
        except Exception:
            error_payload = {}

    if not isinstance(error_payload, dict):
        error_payload = {}

    haystack = " ".join(
        str(part or "")
        for part in (
            error_payload.get("message"),
            error_payload.get("param"),
            error_payload.get("code"),
            exc,
        )
    ).lower()

    return "reasoning" in haystack and (
        error_payload.get("param") == "reasoning"
        or "does not support reasoning" in haystack
        or "reasoning configuration" in haystack
    )


def _split_system_prompt(messages: list[dict[str, str]]) -> tuple[str, str]:
    """Flatten simple message history into LM Studio's system_prompt + input fields."""
    system_parts = []
    input_parts = []
    for message in messages:
        role = (message.get("role") or "user").strip().lower()
        content = (message.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
            continue
        if role == "user" and not input_parts:
            input_parts.append(content)
            continue
        input_parts.append(f"{role.upper()}: {content}")
    return "\n\n".join(system_parts).strip(), "\n\n".join(input_parts).strip()


def _split_system_messages(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """Separate system prompts from a simple role/content message list."""
    system_parts = []
    converted: list[dict[str, str]] = []
    for message in messages:
        role = (message.get("role") or "user").strip().lower()
        content = (message.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
            continue
        if role not in {"user", "assistant"}:
            role = "user"
            content = f"{(message.get('role') or 'user').upper()}: {content}"
        converted.append({"role": role, "content": content})
    return "\n\n".join(system_parts).strip(), converted


def _chat_lmstudio(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    think: bool | str | None,
) -> str:
    client = _get_lmstudio_client()
    if client is None:
        raise RuntimeError("LM Studio is configured but client failed to initialize.")

    system_prompt, input_text = _split_system_prompt(messages)
    if not input_text:
        raise RuntimeError("LM Studio request had no user input after message flattening.")

    payload = {
        "model": model,
        "input": input_text,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if system_prompt:
        payload["system_prompt"] = system_prompt
    reasoning = None
    if model not in _lmstudio_models_without_reasoning:
        reasoning = _lmstudio_reasoning_payload(think)
    if reasoning:
        payload["reasoning"] = reasoning

    t0 = time.monotonic()
    try:
        future = _executor.submit(client.post, "/api/v1/chat", json=payload)
        response = _wait_with_heartbeat(future, t0)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        if reasoning and _lmstudio_reasoning_unsupported(e):
            _lmstudio_models_without_reasoning.add(model)
            logger.warning(
                "LM Studio model %r rejected reasoning=%r; retrying without it",
                model,
                reasoning,
            )
            retry_payload = dict(payload)
            retry_payload.pop("reasoning", None)
            future = _executor.submit(client.post, "/api/v1/chat", json=retry_payload)
            response = _wait_with_heartbeat(future, t0)
            response.raise_for_status()
            data = response.json()
            think = None
        else:
            # Extract response body for better error messages
            err_detail = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    err_detail = f"HTTP {e.response.status_code}: {e.response.text[:500]}"
                except Exception:
                    err_detail = f"HTTP {e.response.status_code}"
            if not err_detail.strip():
                err_detail = f"{type(e).__name__} (no message)"
            logger.error("LM Studio request to %s failed: %s", model, err_detail)
            raise RuntimeError(f"LM Studio chat request failed: {err_detail}") from e

    elapsed = time.monotonic() - t0
    output = data.get("output") or []
    if not isinstance(output, list):
        output = []

    answer_parts = []
    reasoning_parts = []
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = (item.get("type") or "").strip().lower()
        content = (item.get("content") or "").strip()
        if not content:
            continue
        if item_type == "reasoning":
            reasoning_parts.append(content)
        elif item_type == "message":
            answer_parts.append(content)

    text = "\n\n".join(answer_parts).strip()
    reasoning_text = "\n\n".join(reasoning_parts).strip()
    stats = data.get("stats") or {}
    input_tokens = stats.get("input_tokens")
    total_output_tokens = stats.get("total_output_tokens")
    reasoning_output_tokens = stats.get("reasoning_output_tokens")
    ttft = stats.get("time_to_first_token_seconds")
    tokens_per_second = stats.get("tokens_per_second")
    response_id = data.get("response_id")

    icon = _status_icon(None, text)
    answer_preview = text[:60].replace("\n", " ")
    with _print_lock:
        print(
            f"  {icon} {elapsed:5.1f}s"
            f"  {total_output_tokens or '?':>5}tok"
            f"  tps={tokens_per_second or '?':<6}"
            f"  → {answer_preview or '(empty)'}"
        )

    _append_llm_debug({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "provider": "lmstudio",
        "elapsed_s": round(elapsed, 2),
        "request_preview": input_text[-500:],
        "system_prompt_preview": system_prompt[:500],
        "raw_output": output,
        "extracted_text": text[:3000] if text else "",
        "response_id": response_id,
        "input_tokens": input_tokens,
        "completion_tokens": total_output_tokens,
        "reasoning_output_tokens": reasoning_output_tokens,
        "tokens_per_second": tokens_per_second,
        "time_to_first_token_seconds": ttft,
        "think": think,
    })

    if reasoning_text:
        _append_thinking_log({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": model,
            "provider": "lmstudio",
            "elapsed_s": round(elapsed, 2),
            "request_preview": input_text[:1000],
            "answer": text,
            "reasoning": reasoning_text,
            "reasoning_tokens": len(reasoning_text.split()),
            "response_id": response_id,
            "input_tokens": input_tokens,
            "completion_tokens": total_output_tokens,
            "reasoning_output_tokens": reasoning_output_tokens,
            "tokens_per_second": tokens_per_second,
            "time_to_first_token_seconds": ttft,
            "think": think,
        })
    elif think:
        _append_thinking_log({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": model,
            "provider": "lmstudio",
            "elapsed_s": round(elapsed, 2),
            "request_preview": input_text[:1000],
            "answer": text,
            "reasoning": "(requested but not exposed by LM Studio response)",
            "reasoning_tokens": 0,
            "response_id": response_id,
            "input_tokens": input_tokens,
            "completion_tokens": total_output_tokens,
            "reasoning_output_tokens": reasoning_output_tokens,
            "tokens_per_second": tokens_per_second,
            "time_to_first_token_seconds": ttft,
            "think": think,
        })

    return text


def _status_icon(finish_reason: str | None, raw_content) -> str:
    """Single-char icon for quick visual scanning of responses."""
    if finish_reason == "length" and raw_content is None:
        return "⏱"  # thinking ran out of tokens
    if finish_reason == "length":
        return "✂"  # truncated but has content
    if raw_content is None or raw_content == "":
        return "∅"  # empty answer
    return "✓"  # good response


def _wait_with_heartbeat(future: Future, t0: float) -> Any:
    """Wait for a Future to complete, printing heartbeat messages while it runs.

    Prints a dot every HEARTBEAT_INTERVAL seconds, a warning after HEARTBEAT_WARN,
    and a critical warning after HEARTBEAT_CRITICAL.
    """
    warned = False
    critical_warned = False
    dots_printed = 0

    while True:
        try:
            result = future.result(timeout=HEARTBEAT_INTERVAL)
            # Got result — finish the dots line if we printed any
            with _print_lock:
                if dots_printed > 0:
                    print()  # newline after dots
            return result
        except TimeoutError:
            elapsed = time.monotonic() - t0
            with _print_lock:
                sys.stdout.write("·")
                sys.stdout.flush()
            dots_printed += 1

            if elapsed >= HEARTBEAT_CRITICAL and not critical_warned:
                with _print_lock:
                    print(f"\n  ⚠ {elapsed:.0f}s — response is taking unusually long, may be stuck")
                critical_warned = True
                warned = True
            elif elapsed >= HEARTBEAT_WARN and not warned:
                with _print_lock:
                    print(f"\n  💭 {elapsed:.0f}s — still thinking (this model reasons before answering)...")
                warned = True
        except Exception:
            with _print_lock:
                if dots_printed > 0:
                    print()
            raise


# Thread pool for running blocking API calls (supports concurrent LLM requests)
_executor = ThreadPoolExecutor(max_workers=3)
# Lock to prevent concurrent status/heartbeat lines from garbling each other
_print_lock = threading.Lock()


def _chat_openai(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    think: bool | str | None,
) -> str:
    provider = _config.llm_provider if _config is not None else "openai"
    client = _get_longcat_client() if provider == "longcat" else _get_openai_client()
    if client is None:
        key_name = "longcat_api_key" if provider == "longcat" else "openai_api_key"
        raise RuntimeError(
            f"{provider.title()} is configured but client failed to initialize. Set {key_name}."
        )
    last_err = None
    for attempt in range(4):
        t0 = time.monotonic()
        try:
            # Submit blocking call to thread so we can print heartbeats while waiting
            future = _executor.submit(
                client.chat.completions.create,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            resp = _wait_with_heartbeat(future, t0)

            elapsed = time.monotonic() - t0
            choice = (resp.choices or [None])[0]
            if not choice or choice.message is None:
                logger.warning(
                    "%s returned no choices or message (%.1fs): choices=%s",
                    provider.title(),
                    elapsed,
                    [getattr(c, "message", c) for c in (resp.choices or [])],
                )
                return ""
            text = _extract_message_text(choice.message)
            raw_content = getattr(choice.message, "content", None)
            finish_reason = getattr(choice, "finish_reason", None)
            usage = getattr(resp, "usage", None)
            comp_tok = getattr(usage, "completion_tokens", None)
            prompt_tok = getattr(usage, "prompt_tokens", None)

            icon = _status_icon(finish_reason, raw_content)
            answer_preview = (raw_content or "")[:60].replace("\n", " ") if isinstance(raw_content, str) else ""

            # Compact one-liner: ✓ 12.3s 847tok → "4"  |  ⏱ 8.1s 1024tok → (thinking, no answer)
            with _print_lock:
                print(
                    f"  {icon} {elapsed:5.1f}s"
                    f"  {comp_tok or '?':>5}tok"
                    f"  fin={finish_reason or '?':<6}"
                    f"  → {answer_preview or '(empty)'}"
                )

            # Stream to data/llm_responses.jsonl so you can double-check returns
            last_msg = (messages[-1] if messages else {}).get("content", "") or ""
            _append_llm_debug({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model": model,
                "provider": provider,
                "elapsed_s": round(elapsed, 2),
                "request_preview": last_msg[-500:],  # tail captures paper title/abstract
                "raw_content": raw_content if isinstance(raw_content, str) else repr(raw_content),
                "extracted_text": (text[:3000] if text else ""),
                "finish_reason": finish_reason,
                "completion_tokens": comp_tok,
                "prompt_tokens": prompt_tok,
            })

            # Save reasoning chain for hallucination auditing
            reasoning = _extract_reasoning(choice.message)
            if reasoning:
                _append_thinking_log({
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "model": model,
                    "provider": provider,
                    "elapsed_s": round(elapsed, 2),
                    "request_preview": last_msg[:1000],
                    "answer": raw_content if isinstance(raw_content, str) else repr(raw_content),
                    "reasoning": reasoning,
                    "reasoning_tokens": len(reasoning.split()),
                    "finish_reason": finish_reason,
                })
            elif comp_tok and comp_tok > 100 and (not raw_content or len(str(raw_content)) < 20):
                # Model used lots of tokens but answer is tiny — reasoning was likely
                # consumed internally. Log a note so we know it happened.
                _append_thinking_log({
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "model": model,
                    "provider": provider,
                    "elapsed_s": round(elapsed, 2),
                    "request_preview": last_msg[:1000],
                    "answer": raw_content if isinstance(raw_content, str) else repr(raw_content),
                    "reasoning": "(not exposed by API — consumed internally)",
                    "reasoning_tokens": comp_tok,
                    "finish_reason": finish_reason,
                    "note": "Model used many tokens but reasoning not in response object. "
                            "API may not expose thinking chain for this model.",
                })

            if finish_reason == "length":
                if raw_content is None:
                    # Thinking model ran out of tokens during reasoning — never
                    # produced a final answer.  Return empty so callers don't
                    # accidentally parse reasoning tokens as the answer.
                    logger.warning(
                        "%s hit max_tokens during thinking (content=None, finish_reason=length, %.1fs). "
                        "Increase max_tokens. usage=%s",
                        provider.title(), elapsed, usage,
                    )
                    return ""
                else:
                    logger.warning(
                        "%s hit max_tokens (finish_reason=length, %.1fs); response may be truncated. usage=%s",
                        provider.title(), elapsed, usage,
                    )
            return text
        except Exception as e:
            elapsed = time.monotonic() - t0
            last_err = e
            status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            if status == 429:
                wait = 2 ** attempt
                with _print_lock:
                    print(f"  ⚠ {elapsed:5.1f}s  429 rate-limited, retrying in {wait}s...")
                time.sleep(wait)
                continue
            # Connection/timeout errors
            err_name = type(e).__name__
            with _print_lock:
                print(f"  ✗ {elapsed:5.1f}s  {err_name}: {str(e)[:80]}")
            raise
    raise last_err or RuntimeError(f"{provider.title()} request failed after retries")


def _chat_anthropic(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    think: bool | str | None,
) -> str:
    client = _get_anthropic_client()
    if client is None:
        raise RuntimeError("Anthropic is configured but client failed to initialize. Set anthropic_api_key.")

    system_prompt, anthropic_messages = _split_system_messages(messages)
    if not anthropic_messages:
        raise RuntimeError("Anthropic request had no user/assistant messages to send.")

    payload: dict[str, Any] = {
        "model": model,
        "messages": anthropic_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if system_prompt:
        payload["system"] = system_prompt

    t0 = time.monotonic()
    future = _executor.submit(client.messages.create, **payload)
    resp = _wait_with_heartbeat(future, t0)

    elapsed = time.monotonic() - t0
    text = _extract_message_text(resp)
    raw_content = getattr(resp, "content", None)
    stop_reason = getattr(resp, "stop_reason", None)
    usage = getattr(resp, "usage", None)
    comp_tok = getattr(usage, "output_tokens", None)
    prompt_tok = getattr(usage, "input_tokens", None)
    reasoning = _extract_reasoning(resp)
    last_msg = (messages[-1] if messages else {}).get("content", "") or ""

    icon = _status_icon(stop_reason, raw_content)
    answer_preview = text[:60].replace("\n", " ") if text else ""
    with _print_lock:
        print(
            f"  {icon} {elapsed:5.1f}s"
            f"  {comp_tok or '?':>5}tok"
            f"  fin={stop_reason or '?':<6}"
            f"  → {answer_preview or '(empty)'}"
        )

    _append_llm_debug({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "provider": "anthropic",
        "elapsed_s": round(elapsed, 2),
        "request_preview": last_msg[-500:],
        "system_prompt_preview": system_prompt[:500],
        "raw_content": repr(raw_content),
        "extracted_text": text[:3000] if text else "",
        "finish_reason": stop_reason,
        "completion_tokens": comp_tok,
        "prompt_tokens": prompt_tok,
        "think": think,
    })

    if reasoning:
        _append_thinking_log({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": model,
            "provider": "anthropic",
            "elapsed_s": round(elapsed, 2),
            "request_preview": last_msg[:1000],
            "answer": text,
            "reasoning": reasoning,
            "reasoning_tokens": len(reasoning.split()),
            "finish_reason": stop_reason,
            "completion_tokens": comp_tok,
            "prompt_tokens": prompt_tok,
            "think": think,
        })
    elif think:
        _append_thinking_log({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": model,
            "provider": "anthropic",
            "elapsed_s": round(elapsed, 2),
            "request_preview": last_msg[:1000],
            "answer": text,
            "reasoning": "(requested but not exposed by Anthropic response)",
            "reasoning_tokens": 0,
            "finish_reason": stop_reason,
            "completion_tokens": comp_tok,
            "prompt_tokens": prompt_tok,
            "think": think,
        })

    return text
