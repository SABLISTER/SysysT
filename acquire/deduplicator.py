"""Deduplicate a list of paper dicts by DOI and fuzzy title match."""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

TITLE_SIMILARITY_THRESHOLD = 90


def _normalize_doi(doi: str | None) -> str | None:
    """Lowercase, strip whitespace and common URL prefixes."""
    if not doi:
        return None
    doi = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi if doi else None


def _merge_papers(existing: dict, new: dict) -> dict:
    """Merge metadata from *new* into *existing*, preferring non-empty values."""
    merged = dict(existing)
    for key, val in new.items():
        if key == "source":
            old_src = merged.get("source", "")
            if val and val not in old_src:
                merged["source"] = f"{old_src}+{val}" if old_src else val
        elif not merged.get(key) and val:
            merged[key] = val
    return merged


def _title_similarity(a: str, b: str) -> float:
    """Jaccard word-set similarity × 100 (to compare against threshold)."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return (len(intersection) / len(union)) * 100


def deduplicate_papers(papers: list[dict]) -> list[dict]:
    """Deduplicate by DOI (exact) then by fuzzy title match."""
    before = len(papers)

    # Phase 1: DOI dedup
    seen_dois: dict[str, int] = {}
    unique: list[dict] = []

    for p in papers:
        doi = _normalize_doi(p.get("doi"))
        if doi and doi in seen_dois:
            idx = seen_dois[doi]
            unique[idx] = _merge_papers(unique[idx], p)
            continue
        if doi:
            seen_dois[doi] = len(unique)
        unique.append(p)

    after_doi = len(unique)

    # Phase 2: Fuzzy title dedup
    final: list[dict] = []
    seen_titles: list[str] = []

    for p in unique:
        title = (p.get("title") or "").strip().lower()
        if not title:
            final.append(p)
            continue

        is_dup = False
        for seen in seen_titles:
            if _title_similarity(title, seen) >= TITLE_SIMILARITY_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            seen_titles.append(title)
            final.append(p)

    removed = before - len(final)
    logger.info(
        "Deduplication: %d → %d papers (%d removed: %d by DOI, %d by title)",
        before, len(final), removed,
        before - after_doi, after_doi - len(final),
    )
    return final
