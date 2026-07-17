"""Storage helpers for downloaded full-text artifacts."""
from __future__ import annotations

import asyncio
import gzip
import logging
import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from acquire.fulltext import write_json_atomic

logger = logging.getLogger(__name__)


def _row_key(row: dict) -> str:
    return str(
        row.get("doi")
        or row.get("pmc")
        or row.get("key")
        or row.get("title")
        or row.get("path")
        or ""
    ).strip()


def _archive_one_source(row: dict, archive_dir: Path) -> dict:
    updated = dict(row)
    source_value = str(row.get("source_path") or "").strip()
    if not source_value:
        return updated
    source_path = Path(source_value)
    if not source_path.is_file() or source_path.suffix.lower() == ".txt":
        return updated

    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{source_path.name}.gz"
    if not archive_path.exists():
        with open(source_path, "rb") as src, gzip.open(archive_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

    source_path.unlink(missing_ok=True)
    updated["source_archive_path"] = str(archive_path)
    updated["source_archive_format"] = "gzip"
    updated["source_path"] = ""
    return updated


def archive_original_sources(
    rows: list[dict],
    archive_dir: Path,
    *,
    workers: int = 3,
) -> list[dict]:
    """Compress loose PDF/HTML originals and keep extracted text in place."""
    if not rows:
        return []

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        return list(pool.map(lambda row: _archive_one_source(row, archive_dir), rows))


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"indexed": []}
    try:
        import json

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data.setdefault("indexed", [])
            return data
    except Exception:
        pass
    return {"indexed": []}


async def _add_rows_to_paperqa(rows: list[dict], db_dir: Path, manifest: dict) -> dict:
    from paperqa import Docs, Settings

    docs_path = db_dir / "paperqa_docs.pkl"
    if docs_path.exists():
        with open(docs_path, "rb") as f:
            docs = pickle.load(f)
    else:
        docs = Docs()
        manifest["indexed"] = []

    indexed = set(str(item) for item in manifest.get("indexed", []))
    settings = Settings(embedding="st-multi-qa-MiniLM-L6-cos-v1")
    added = 0
    skipped = 0
    failed: list[dict[str, str]] = []

    for row in rows:
        key = _row_key(row)
        text_value = str(row.get("path") or "").strip()
        if not text_value:
            skipped += 1
            continue
        text_path = Path(text_value)
        if not key or key in indexed or not text_path.is_file():
            skipped += 1
            continue
        try:
            await docs.aadd(
                str(text_path),
                citation=str(row.get("title") or key),
                docname=key,
                title=str(row.get("title") or ""),
                doi=str(row.get("doi") or ""),
                settings=settings,
            )
        except Exception as exc:
            failed.append({"key": key, "error": str(exc)})
            continue
        indexed.add(key)
        added += 1

    with open(docs_path, "wb") as f:
        pickle.dump(docs, f)
    manifest["indexed"] = sorted(indexed)
    return {
        "available": True,
        "docs_path": str(docs_path),
        "added": added,
        "skipped": skipped,
        "failed": failed,
        "indexed_total": len(indexed),
    }


def update_fulltext_store(
    rows: list[dict],
    store_dir: Path,
    *,
    workers: int = 3,
) -> tuple[list[dict], dict]:
    """Archive originals and optionally index extracted text with PaperQA."""
    store_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = store_dir / "source_archive"
    paperqa_dir = store_dir / "paperqa"
    manifest_path = paperqa_dir / "manifest.json"

    archived_rows = archive_original_sources(rows, archive_dir, workers=workers)
    archived_count = sum(1 for row in archived_rows if row.get("source_archive_path"))

    paperqa_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path)
    try:
        paperqa_result = asyncio.run(_add_rows_to_paperqa(archived_rows, paperqa_dir, manifest))
    except Exception as exc:
        paperqa_result = {
            "available": False,
            "error": str(exc),
            "docs_path": "",
            "added": 0,
            "skipped": 0,
            "failed": [],
            "indexed_total": 0,
        }

    write_json_atomic(manifest_path, manifest)
    metadata = {
        "archive_dir": str(archive_dir),
        "archived_originals": archived_count,
        "paperqa": paperqa_result,
    }
    return archived_rows, metadata
