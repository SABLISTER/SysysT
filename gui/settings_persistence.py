"""Helpers for applying persisted GUI settings to runtime config objects."""

from __future__ import annotations

from typing import Any


API_KEY_QSETTINGS_KEYS: dict[str, str] = {
    "openai_api_key": "keys/openai_api_key",
    "longcat_api_key": "keys/longcat_api_key",
    "anthropic_api_key": "keys/anthropic_api_key",
    "s2_api_key": "keys/s2_api_key",
    "pubmed_api_key": "keys/pubmed_api_key",
    "pubmed_email": "keys/pubmed_email",
    "elicit_api_key": "keys/elicit_api_key",
    "scopus_api_key": "keys/scopus_api_key",
    "scopus_insttoken": "keys/scopus_insttoken",
    "wos_api_key": "keys/wos_api_key",
    "wos_researcher_api_key": "keys/wos_researcher_api_key",
}

LLM_QSETTINGS_KEYS: dict[str, str] = {
    "llm_provider": "llm/provider",
    "ollama_model": "llm/ollama_model",
    "ollama_base_url": "llm/ollama_url",
    "lmstudio_model": "llm/lmstudio_model",
    "lmstudio_base_url": "llm/lmstudio_url",
    "openai_model": "llm/openai_model",
    "openai_base_url": "llm/openai_url",
    "longcat_model": "llm/longcat_model",
    "longcat_base_url": "llm/longcat_url",
    "anthropic_model": "llm/anthropic_model",
}


def _saved_text(settings: Any, key: str) -> str:
    value = settings.value(key, "", type=str)
    return str(value or "").strip()


def restore_saved_settings(config: Any, settings: Any) -> None:
    """Layer saved QSettings values onto a Config-like object.

    API credentials keep the conservative env/config > QSettings priority, but
    LLM provider/model/base-url fields are user preferences and should override
    the non-empty dataclass defaults such as ``openai``.
    """
    for cfg_field, qs_key in API_KEY_QSETTINGS_KEYS.items():
        if not hasattr(config, cfg_field):
            continue
        current = getattr(config, cfg_field, "")
        if current and str(current).strip():
            continue
        saved = _saved_text(settings, qs_key)
        if saved:
            setattr(config, cfg_field, saved)

    for cfg_field, qs_key in LLM_QSETTINGS_KEYS.items():
        if not hasattr(config, cfg_field):
            continue
        saved = _saved_text(settings, qs_key)
        if saved:
            setattr(config, cfg_field, saved)
