"""Tests for persisted GUI settings applied to runtime config."""
from __future__ import annotations

from core.config import Config
from gui.settings_persistence import restore_saved_settings


class FakeSettings:
    def __init__(self, values: dict[str, str]):
        self._values = values

    def value(self, key: str, default="", type=str):  # noqa: A002 - mirrors QSettings
        value = self._values.get(key, default)
        return type(value) if type is not None else value


def test_saved_llm_preferences_override_non_empty_config_defaults(tmp_path):
    cfg = Config(project_root=tmp_path)
    cfg.llm_provider = "openai"
    cfg.openai_model = "gpt-5"

    restore_saved_settings(
        cfg,
        FakeSettings({
            "llm/provider": "lmstudio",
            "llm/lmstudio_model": "local-stats-model",
            "llm/lmstudio_url": "http://127.0.0.1:1234",
        }),
    )

    assert cfg.llm_provider == "lmstudio"
    assert cfg.llm_model == "local-stats-model"
    assert cfg.lmstudio_base_url == "http://127.0.0.1:1234"


def test_saved_api_key_does_not_override_existing_config_value(tmp_path):
    cfg = Config(project_root=tmp_path)
    cfg.openai_api_key = "env-or-project-key"

    restore_saved_settings(
        cfg,
        FakeSettings({"keys/openai_api_key": "saved-qsettings-key"}),
    )

    assert cfg.openai_api_key == "env-or-project-key"
