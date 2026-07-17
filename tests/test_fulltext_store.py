"""Tests for full-text storage and archival helpers."""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from acquire.fulltext import reusable_download_record
from process.fulltext_store import archive_original_sources


def test_archive_original_sources_compresses_source_and_keeps_text(tmp_path):
    text_path = tmp_path / "paper.txt"
    source_path = tmp_path / "paper.pdf"
    text_path.write_text("extracted text", encoding="utf-8")
    source_path.write_bytes(b"%PDF original bytes")

    rows = archive_original_sources(
        [{"path": str(text_path), "source_path": str(source_path), "source_format": "pdf"}],
        tmp_path / "source_archive",
        workers=3,
    )

    row = rows[0]
    archive_path = Path(row["source_archive_path"])
    assert text_path.exists()
    assert not source_path.exists()
    assert archive_path.exists()
    assert row["source_path"] == ""
    with gzip.open(archive_path, "rb") as f:
        assert f.read() == b"%PDF original bytes"


def test_archive_original_sources_ignores_blank_source_path(tmp_path):
    text_path = tmp_path / "paper.txt"
    text_path.write_text("extracted text", encoding="utf-8")

    rows = archive_original_sources(
        [{"path": str(text_path), "source_path": "", "source_format": "text"}],
        tmp_path / "source_archive",
        workers=3,
    )

    assert rows == [{"path": str(text_path), "source_path": "", "source_format": "text"}]
    assert text_path.exists()
    assert not (tmp_path / "source_archive").exists()


def test_reusable_download_record_accepts_archived_source(tmp_path):
    text_path = tmp_path / "paper.txt"
    archive_path = tmp_path / "paper.pdf.gz"
    text_path.write_text("extracted text", encoding="utf-8")
    archive_path.write_bytes(b"compressed")

    assert reusable_download_record({
        "path": str(text_path),
        "source_path": str(tmp_path / "paper.pdf"),
        "source_archive_path": str(archive_path),
    })
