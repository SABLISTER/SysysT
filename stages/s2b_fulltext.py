"""Stage 2b: Optional full-text corpus enrichment before relevance scoring.

Downloads legally accessible open-access full text for papers in the corpus,
adds cleaned plaintext to the paper records, and writes an enriched corpus file
that Stage 2 can score against.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from acquire.fulltext import (
    AsyncHostThrottle,
    copy_oa_hint_fields,
    download_cache_key,
    download_record_status,
    download_preferred_fulltext_async,
    load_download_cache,
    recover_download_artifact,
    reusable_download_record,
    write_json_atomic,
)

logger = logging.getLogger(__name__)


def _emit_progress(progress_callback, message: str, current: int | None = None, total: int | None = None) -> None:
    """Best-effort progress reporting hook for GUI callers."""
    if not progress_callback:
        return
    payload = {"message": message}
    if current is not None:
        payload["current"] = current
    if total is not None:
        payload["total"] = total
    try:
        progress_callback(payload)
    except Exception:
        pass


def _is_cancelled(cancel_event) -> bool:
    return bool(cancel_event and cancel_event.is_set())


def _paper_key(paper: dict) -> str:
    return (
        paper.get("doi")
        or paper.get("pmc")
        or paper.get("pmid")
        or paper.get("s2_id")
        or paper.get("oa_id")
        or paper.get("title", "")
    )


def _safe_name(identifier: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", identifier).strip("_") or "paper"


def _build_targets(corpus: list[dict]) -> list[dict]:
    targets: list[dict] = []
    seen: set[str] = set()

    for paper in corpus:
        key = _paper_key(paper)
        if not key or key in seen:
            continue
        seen.add(key)
        targets.append({
            "key": key,
            "doi": paper.get("doi"),
            "pmc": paper.get("pmc"),
            "title": paper.get("title", ""),
            "year": paper.get("year"),
            **copy_oa_hint_fields(paper),
        })

    return targets


def _merge_oa_locations(targets: list[dict], oa_rows: list[dict]) -> list[dict]:
    by_doi = {row.get("doi"): row for row in oa_rows if row.get("doi")}
    merged: list[dict] = []

    for target in targets:
        row = dict(target)
        oa = by_doi.get(target.get("doi"))
        if oa:
            row["is_oa"] = oa.get("is_oa", False)
            row["oa_url"] = oa.get("oa_url", "")
            row["host_type"] = oa.get("host_type", "")
            if oa.get("resolver"):
                row["resolver"] = oa.get("resolver", "")
        elif target.get("pmc"):
            row["is_oa"] = True
            row["oa_url"] = f"https://pmc.ncbi.nlm.nih.gov/articles/{target['pmc']}/"
            row["host_type"] = "repository"
        else:
            row["is_oa"] = False
            row["oa_url"] = ""
            row["host_type"] = ""
        merged.append(row)

    return merged


def _download_fulltext(
    targets: list[dict],
    output_dir: Path,
    progress_callback=None,
    cancel_event=None,
    log_path: Path | None = None,
    workers: int = 8,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_path or output_dir / "download_log.json"
    persisted_rows = load_download_cache(log_path)
    eligible_keys = {
        download_cache_key(target)
        for target in targets
        if target.get("is_oa") and target.get("oa_url")
    }
    downloaded: list[dict] = [
        row for key, row in persisted_rows.items()
        if key in eligible_keys and reusable_download_record(row)
    ]
    eligible = [target for target in targets if target.get("is_oa") and target.get("oa_url")]
    total = len(eligible)
    processed = 0
    logger.info("Stage 2b: %d papers are eligible for full-text download", total)
    _emit_progress(progress_callback, "Preparing corpus full-text downloads...", 0, max(1, total))

    if total == 0:
        logger.warning("Stage 2b: no open-access full-text targets were found in the selected corpus")
        return downloaded

    if downloaded:
        logger.info(
            "Stage 2b: reusing %d completed corpus full-text downloads from %s",
            len(downloaded),
            log_path,
        )
    pending: list[dict] = []

    for target in targets:
        if _is_cancelled(cancel_event):
            logger.info("Stage 2b: stop requested after %d/%d full-text items", processed, total)
            break
        url = target.get("oa_url", "")
        if not target.get("is_oa") or not url:
            continue

        processed += 1
        ident = target.get("doi") or target.get("pmc") or target["key"]
        safe_name = _safe_name(ident)
        cache_key = download_cache_key(target)
        cached = persisted_rows.get(cache_key)
        if cached and reusable_download_record(cached):
            if download_record_status(cached) != "downloaded":
                cached["status"] = "downloaded"
                persisted_rows[cache_key] = cached
                write_json_atomic(log_path, list(persisted_rows.values()))
            logger.info("  Reused cached corpus full text for %s", ident)
            _emit_progress(
                progress_callback,
                f"Reused cached corpus full text {processed}/{total}: {ident}",
                processed,
                max(1, total),
            )
            continue
        recovered = recover_download_artifact(output_dir, safe_name)
        if recovered:
            row = {
                **target,
                **recovered,
                "status": "downloaded",
            }
            downloaded.append(row)
            persisted_rows[cache_key] = row
            write_json_atomic(log_path, list(persisted_rows.values()))
            logger.info("  Recovered existing corpus full text from disk for %s", ident)
            _emit_progress(
                progress_callback,
                f"Recovered existing corpus full text {processed}/{total}: {ident}",
                processed,
                max(1, total),
            )
            continue
        cached_status = download_record_status(cached or {})
        if cached_status in {"no_usable_text", "error"}:
            logger.info("  Skipping previously checked corpus full text for %s (%s)", ident, cached_status)
            _emit_progress(
                progress_callback,
                f"Skipping previously checked corpus full text {processed}/{total}: {ident} ({cached_status})",
                processed,
                max(1, total),
            )
            continue
        pending.append({
            "target": target,
            "safe_name": safe_name,
            "cache_key": cache_key,
            "ident": ident,
        })

    async def _download_pending():
        import httpx

        throttle = AsyncHostThrottle(default_concurrency=max(1, int(workers)))
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            async def _download_one(item: dict) -> dict:
                target = item["target"]
                url = target.get("oa_url", "")
                try:
                    result = await download_preferred_fulltext_async(
                        client,
                        throttle,
                        url,
                        output_dir,
                        item["safe_name"],
                        logger,
                    )
                    if result:
                        return {
                            "cache_key": item["cache_key"],
                            "row": {
                                **target,
                                **result,
                                "status": "downloaded",
                            },
                        }
                    return {
                        "cache_key": item["cache_key"],
                        "row": {
                            **target,
                            "status": "no_usable_text",
                            "source_url": url,
                        },
                    }
                except Exception as e:
                    return {
                        "cache_key": item["cache_key"],
                        "row": {
                            **target,
                            "status": "error",
                            "source_url": url,
                            "error": str(e),
                        },
                    }

            tasks = [asyncio.create_task(_download_one(item)) for item in pending]
            completed = 0
            try:
                for task in asyncio.as_completed(tasks):
                    if _is_cancelled(cancel_event):
                        logger.info(
                            "Stage 2b: stop requested after %d/%d async full-text items",
                            completed,
                            len(pending),
                        )
                        for pending_task in tasks:
                            if not pending_task.done():
                                pending_task.cancel()
                        break

                    payload = await task
                    completed += 1
                    row = payload["row"]
                    cache_key = payload["cache_key"]
                    persisted_rows[cache_key] = row
                    if row.get("status") == "downloaded":
                        downloaded.append(row)
                        logger.info(
                            "  Full text %d/%d: %s (%s)",
                            processed - len(pending) + completed,
                            total,
                            row.get("doi") or row.get("pmc") or row.get("key"),
                            row.get("source_format", "unknown"),
                        )
                    elif row.get("status") == "no_usable_text":
                        logger.warning(
                            "  Full text skipped for %s: no usable text extracted",
                            row.get("doi") or row.get("pmc") or row.get("key"),
                        )
                    else:
                        logger.warning(
                            "  Full text failed for %s: %s",
                            row.get("doi") or row.get("pmc") or row.get("key"),
                            row.get("error", ""),
                        )
                    write_json_atomic(log_path, list(persisted_rows.values()))
                    _emit_progress(
                        progress_callback,
                        f"Processed corpus full text {processed - len(pending) + completed}/{total}",
                        processed - len(pending) + completed,
                        max(1, total),
                    )
            finally:
                await asyncio.gather(*tasks, return_exceptions=True)

    if pending:
        asyncio.run(_download_pending())

    write_json_atomic(log_path, list(persisted_rows.values()))
    return downloaded


def run(
    config,
    corpus_path: Path | None = None,
    output_path: Path | None = None,
    email: str | None = None,
    progress_callback=None,
    cancel_event=None,
) -> dict:
    """Build a corpus enriched with full-text plaintext where available."""
    from stages.s7_fulltext import check_oa_status

    corpus_path = corpus_path or config.corpus_dir / "corpus.json"
    output_path = output_path or config.corpus_dir / "corpus_fulltext.json"
    email = email or config.pubmed_email or "research@pipeline.dev"

    logger.info("Stage 2b: loading corpus from %s", corpus_path)
    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)

    targets = _build_targets(corpus)
    _emit_progress(progress_callback, f"Identified {len(targets)} corpus targets", 0, max(1, len(targets)))
    doi_targets = [t for t in targets if t.get("doi")]
    logger.info(
        "Stage 2b: identified %d unique targets, %d with DOI for OA lookup",
        len(targets),
        len(doi_targets),
    )
    _emit_progress(progress_callback, "Checking corpus open-access availability...", 0, max(1, len(targets)))
    oa_cache_path = config.data_dir / "fulltext_corpus" / "oa_status.json"
    oa_rows = check_oa_status(
        doi_targets,
        email=email,
        progress_callback=progress_callback,
        progress_label="Checking corpus open-access availability",
        cache_path=oa_cache_path,
        cancel_event=cancel_event,
        workers=max(1, int(config.oa_lookup_workers)),
    ) if doi_targets else []
    if _is_cancelled(cancel_event):
        _emit_progress(progress_callback, "Stop requested during corpus open-access lookup", 0, max(1, len(targets)))
        return {
            "paper_count": len(corpus),
            "fulltext_count": 0,
            "corpus_path": str(output_path),
            "download_dir": str(config.data_dir / "fulltext_corpus"),
            "cancelled": True,
        }
    targets = _merge_oa_locations(targets, oa_rows)
    oa_count = sum(1 for target in targets if target.get("is_oa") and target.get("oa_url"))
    logger.info("Stage 2b: %d targets have an open-access full-text URL", oa_count)

    download_dir = config.data_dir / "fulltext_corpus"
    download_log_path = download_dir / "download_log.json"
    downloaded = _download_fulltext(
        targets,
        download_dir,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
        log_path=download_log_path,
        workers=max(1, int(config.fulltext_download_workers)),
    )
    store_metadata = {
        "archive_dir": str(download_dir / "source_archive"),
        "archived_originals": 0,
        "paperqa": {"available": False},
    }
    if downloaded:
        from process.fulltext_store import update_fulltext_store

        _emit_progress(
            progress_callback,
            "Indexing extracted full text and archiving originals...",
            0,
            max(1, len(downloaded)),
        )
        downloaded, store_metadata = update_fulltext_store(
            downloaded,
            download_dir,
            workers=max(1, int(config.fulltext_download_workers)),
        )
        paperqa_meta = store_metadata.get("paperqa", {})
        if paperqa_meta.get("available"):
            logger.info(
                "Stage 2b: PaperQA indexed %d new texts (%d total) -> %s",
                paperqa_meta.get("added", 0),
                paperqa_meta.get("indexed_total", 0),
                paperqa_meta.get("docs_path", ""),
            )
        else:
            logger.info(
                "Stage 2b: PaperQA indexing skipped (%s)",
                paperqa_meta.get("error", "paper-qa not installed"),
            )
        logger.info(
            "Stage 2b: archived %d original full-text sources -> %s",
            store_metadata.get("archived_originals", 0),
            store_metadata.get("archive_dir", ""),
        )
    if _is_cancelled(cancel_event):
        _emit_progress(progress_callback, "Stop requested during corpus full-text download", 0, max(1, len(downloaded)))
        return {
            "paper_count": len(corpus),
            "fulltext_count": len(downloaded),
            "corpus_path": str(output_path),
            "download_dir": str(download_dir),
            "archive_dir": store_metadata.get("archive_dir", ""),
            "paperqa": store_metadata.get("paperqa", {}),
            "cancelled": True,
        }

    downloaded_map = {}
    total_downloaded = len(downloaded)
    for idx, row in enumerate(downloaded, start=1):
        if _is_cancelled(cancel_event):
            enriched = []
            for paper in corpus:
                key = _paper_key(paper)
                record = dict(paper)
                if key in downloaded_map:
                    record.update(downloaded_map[key])
                    record["text_source"] = "full_text"
                else:
                    record["text_source"] = "abstract"
                enriched.append(record)
            write_json_atomic(output_path, enriched)
            _emit_progress(
                progress_callback,
                "Stop requested while embedding downloaded full text",
                idx - 1,
                max(1, total_downloaded),
            )
            return {
                "paper_count": len(enriched),
                "fulltext_count": len(downloaded_map),
                "corpus_path": str(output_path),
                "download_dir": str(download_dir),
                "cancelled": True,
            }
        _emit_progress(
            progress_callback,
            f"Embedding downloaded full text {idx}/{total_downloaded}",
            idx,
            max(1, total_downloaded),
        )
        key = row["key"]
        clean_text = Path(row["path"]).read_text(encoding="utf-8")
        downloaded_map[key] = {
            "full_text": clean_text,
            "full_text_path": row["path"],
            "full_text_source_path": row.get("source_path") or row["path"],
            "full_text_source_archive_path": row.get("source_archive_path", ""),
            "full_text_source_format": row.get("source_format", "html"),
            "full_text_source_url": row.get("source_url", ""),
            "full_text_chars": len(clean_text),
        }

    enriched = []
    for paper in corpus:
        key = _paper_key(paper)
        record = dict(paper)
        if key in downloaded_map:
            record.update(downloaded_map[key])
            record["text_source"] = "full_text"
        else:
            record["text_source"] = "abstract"
        enriched.append(record)

    write_json_atomic(download_log_path, downloaded)
    write_json_atomic(output_path, enriched)

    enriched_count = sum(1 for paper in enriched if paper.get("full_text"))
    logger.info(
        "Stage 2b complete: %d/%d papers enriched with full text -> %s",
        enriched_count,
        len(enriched),
        output_path,
    )
    _emit_progress(
        progress_callback,
        f"Corpus full-text enrichment complete: {enriched_count}/{len(enriched)} papers",
        enriched_count,
        max(1, len(enriched)),
    )
    return {
        "paper_count": len(enriched),
        "fulltext_count": enriched_count,
        "corpus_path": str(output_path),
        "download_dir": str(download_dir),
        "archive_dir": store_metadata.get("archive_dir", ""),
        "paperqa": store_metadata.get("paperqa", {}),
    }
