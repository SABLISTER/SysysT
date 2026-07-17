from __future__ import annotations

import json
import os
from pathlib import Path

from acquire.fulltext import write_json_atomic
from core.io import atomic_write_json, atomic_write_text


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_atomic_write_json_preserves_existing_mode(tmp_path: Path):
    path = tmp_path / "data.json"
    path.write_text("{}", encoding="utf-8")
    os.chmod(path, 0o640)

    atomic_write_json(path, {"ok": True})

    assert _mode(path) == 0o640
    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}


def test_atomic_write_text_preserves_existing_mode(tmp_path: Path):
    path = tmp_path / "report.md"
    path.write_text("old", encoding="utf-8")
    os.chmod(path, 0o644)

    atomic_write_text(path, "new")

    assert _mode(path) == 0o644
    assert path.read_text(encoding="utf-8") == "new"


def test_fulltext_write_json_atomic_keeps_ascii_escaping(tmp_path: Path):
    path = tmp_path / "fulltext.json"

    write_json_atomic(path, {"title": "café"})

    assert "caf\\u00e9" in path.read_text(encoding="utf-8")
