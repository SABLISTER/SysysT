"""Filesystem layout for hard-mode runs."""

from pathlib import Path

from core.config import Config


def hard_mode_base(config: Config) -> Path:
    override = getattr(config, "hard_mode_base_dir", None)
    if override:
        return Path(override)
    return config.data_dir / "hard_mode"


def hard_mode_run_dir(config: Config, run_id: str) -> Path:
    return hard_mode_base(config) / run_id


def ensure_run_dir(config: Config, run_id: str) -> Path:
    d = hard_mode_run_dir(config, run_id)
    d.mkdir(parents=True, exist_ok=True)
    return d
