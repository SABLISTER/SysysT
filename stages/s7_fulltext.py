"""Stage 7: Full-text retrieval & deep metric extraction.

Identifies high-value papers (SRs, MAs, RCTs) from method profiles,
checks open-access status via Unpaywall, downloads full text (PDF/HTML),
and extracts quantitative metrics (effect sizes, CIs, sample sizes, p-values).

No LLM calls — only HTTP downloads + regex extraction.
"""
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

from core.io import atomic_write_json

logger = logging.getLogger(__name__)

TARGET_TYPES = ["systematic_review_meta_analysis", "randomized_controlled_trial"]

_USER_AGENT = (
    "SystS-Pipeline/1.0 (Systematic Review Tool; "
    "mailto:research@pipeline.dev) python-requests/2.31"
)


def _import_requests():
    """Import requests with a clear install hint for missing environments."""
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'requests' for Stage 7 full-text retrieval. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return requests


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def run(
    config=None,
    email: Optional[str] = None,
    claims_path: Optional[str] = None,
    workers: int = 8,
    cancel_event: Any = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kw,
) -> dict:
    """Identify targets -> check OA -> download -> extract metrics."""

    def _progress(msg: str) -> None:
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    def _cancelled() -> bool:
        if cancel_event is None:
            return False
        if hasattr(cancel_event, "is_set"):
            return cancel_event.is_set()
        return bool(cancel_event)

    email = (
        email
        or getattr(config, "pubmed_email", "")
        or "research@pipeline.dev"
    )
    workers = workers or getattr(config, "fulltext_download_workers", 8)

    _progress("Stage 7: Full-Text Retrieval & Metric Extraction")
    _progress(f"  Email for Unpaywall: {email}")
    _progress(f"  Workers: {workers}")

    # Ensure output directory
    ft_dir = config.data_dir / "fulltext"
    ft_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Identify targets ───────────────────────────────────────────
    _progress("Phase 1: Identifying high-value targets...")
    targets = identify_targets(config, claims_path=claims_path)
    _progress(f"  Identified {len(targets)} high-value targets")

    if not targets:
        _progress("No targets found — nothing to retrieve. Stage 7 complete.")
        result = {
            "targets": 0, "oa_papers": 0, "downloaded": 0,
            "metrics_extracted": 0, "output_path": str(ft_dir),
        }
        atomic_write_json(ft_dir / "auto_extracted_metrics.json", [])
        return result

    if _cancelled():
        return {"cancelled": True}

    # ── 2. Check OA status ────────────────────────────────────────────
    _progress("Phase 2: Checking open-access status via Unpaywall...")
    cache_path = ft_dir / "oa_status.json"
    oa_targets = check_oa_status(
        targets, email=email, progress_callback=progress_callback,
        cache_path=cache_path, cancel_event=cancel_event,
    )
    oa_papers = [t for t in oa_targets if t.get("is_oa")]
    _progress(f"  {len(oa_papers)}/{len(targets)} papers are open access")

    if _cancelled():
        return {"cancelled": True}

    # ── 3. Retrieve full text ─────────────────────────────────────────
    _progress("Phase 3: Downloading full text for OA papers...")
    downloaded = retrieve_fulltext(
        oa_papers, output_dir=ft_dir,
        progress_callback=progress_callback, cancel_event=cancel_event,
    )
    success_count = sum(1 for d in downloaded if d.get("text"))
    _progress(f"  Downloaded and extracted text from {success_count}/{len(oa_papers)} papers")

    if _cancelled():
        return {"cancelled": True}

    # ── 4. Extract metrics ────────────────────────────────────────────
    _progress("Phase 4: Extracting quantitative metrics...")
    all_metrics: list[dict] = []
    papers_with_metrics = 0
    for i, paper in enumerate(downloaded):
        if _cancelled():
            return {"cancelled": True}
        text = paper.get("text", "")
        if not text:
            continue
        paper_id = paper.get("doi", paper.get("pmid", f"paper_{i}"))
        metrics = extract_metrics_from_text(text, paper_id=paper_id)
        total_found = sum(len(v) for v in metrics.values() if isinstance(v, list))
        if total_found > 0:
            papers_with_metrics += 1
        all_metrics.append({
            "doi": paper.get("doi", ""),
            "pmid": paper.get("pmid", ""),
            "title": paper.get("title", ""),
            "method_type": paper.get("method_type", ""),
            "metrics": metrics,
        })

    _progress(f"  Extracted metrics from {papers_with_metrics}/{len(downloaded)} papers")

    # ── 5. Save outputs ──────────────────────────────────────────────
    metrics_path = ft_dir / "auto_extracted_metrics.json"
    atomic_write_json(metrics_path, all_metrics)
    _progress(f"  Saved: {metrics_path}")

    # Save download log
    dl_log = []
    for d in downloaded:
        dl_log.append({
            "doi": d.get("doi", ""),
            "title": d.get("title", ""),
            "status": "ok" if d.get("text") else "failed",
            "format": d.get("format", "unknown"),
            "file": d.get("saved_file", ""),
            "size_bytes": d.get("size_bytes", 0),
        })
    atomic_write_json(ft_dir / "download_log.json", dl_log)

    _progress("Stage 7 complete")

    return {
        "targets": len(targets),
        "oa_papers": len(oa_papers),
        "downloaded": success_count,
        "metrics_extracted": papers_with_metrics,
        "output_path": str(metrics_path),
    }


# ---------------------------------------------------------------------------
# Phase 1: Identify high-value targets
# ---------------------------------------------------------------------------
def identify_targets(
    config, target_types: Optional[list[str]] = None,
    claims_path: Optional[str] = None,
) -> list[dict]:
    """Find SRs, MAs, RCTs from method_profiles_validated.json + claims."""
    target_types = target_types or TARGET_TYPES
    targets: list[dict] = []
    seen_ids: set[str] = set()

    # Source 1: method_profiles_validated.json
    mp_path = config.data_dir / "method_review" / "method_profiles_validated.json"
    if mp_path.exists():
        try:
            with open(mp_path, encoding="utf-8") as f:
                profiles = json.load(f)
            for p in profiles:
                mp = p.get("method_profile", {})
                study_type = mp.get("study_method_type", "")
                if not any(tt in study_type.lower() for tt in _type_keywords(target_types)):
                    continue
                doi = p.get("doi", "")
                pmid = str(p.get("pmid", ""))
                uid = doi or pmid
                if not uid or uid in seen_ids:
                    continue
                seen_ids.add(uid)
                targets.append({
                    "doi": doi,
                    "pmid": pmid,
                    "title": p.get("title", ""),
                    "year": p.get("year", ""),
                    "method_type": study_type,
                })
            logger.info("  Method profiles: %d targets from %d papers",
                        len(targets), len(profiles))
        except Exception as exc:
            logger.warning("Could not load method profiles: %s", exc)

    # Source 2: claims
    claims_file = (
        Path(claims_path) if claims_path
        else config.claims_dir / "claims_filtered.json"
    )
    if not claims_file.exists():
        claims_file = config.claims_dir / "claims.json"

    claims_added = 0
    if claims_file.exists():
        try:
            with open(claims_file, encoding="utf-8") as f:
                claims = json.load(f)
            for p in claims:
                method_field = str(p.get("methodology", "")).lower()
                if not any(kw in method_field for kw in _type_keywords(target_types)):
                    continue
                doi = p.get("doi", "")
                pmid = str(p.get("pmid", ""))
                uid = doi or pmid
                if not uid or uid in seen_ids:
                    continue
                seen_ids.add(uid)
                targets.append({
                    "doi": doi,
                    "pmid": pmid,
                    "title": p.get("title", ""),
                    "year": p.get("year", ""),
                    "method_type": method_field[:60],
                })
                claims_added += 1
            logger.info("  Claims: %d additional targets from %d papers",
                        claims_added, len(claims))
        except Exception as exc:
            logger.warning("Could not load claims: %s", exc)

    # Breakdown by type
    type_counts: dict[str, int] = {}
    for t in targets:
        mt = t["method_type"]
        type_counts[mt] = type_counts.get(mt, 0) + 1
    if type_counts:
        breakdown = ", ".join(f"{k}: {v}" for k, v in type_counts.items())
        logger.info("  Target breakdown: %s", breakdown)

    # Filter out papers without DOIs (can't query Unpaywall)
    with_doi = [t for t in targets if t.get("doi")]
    without_doi = len(targets) - len(with_doi)
    if without_doi:
        logger.warning("  %d targets lack DOIs — skipping OA lookup for those", without_doi)

    return with_doi


def _type_keywords(target_types: list[str]) -> list[str]:
    """Convert target type identifiers to search keywords."""
    keywords = []
    for tt in target_types:
        keywords.append(tt.lower().replace("_", " "))
        # Also add component keywords
        if "systematic" in tt.lower():
            keywords.extend(["systematic review", "meta-analysis", "meta analysis"])
        if "randomized" in tt.lower():
            keywords.extend(["randomized controlled trial", "rct"])
    return list(set(keywords))


# ---------------------------------------------------------------------------
# Phase 2: Check OA status via Unpaywall
# ---------------------------------------------------------------------------
def check_oa_status(
    targets: list[dict],
    email: str = "research@pipeline.dev",
    progress_callback: Optional[Callable[[Any], None]] = None,
    progress_label: Optional[str] = None,
    cache_path: Optional[Path] = None,
    cancel_event: Any = None,
    workers: int = 4,
) -> list[dict]:
    """Query Unpaywall API for each target's OA status."""
    requests = _import_requests()

    def _cancelled() -> bool:
        if cancel_event is None:
            return False
        if hasattr(cancel_event, "is_set"):
            return cancel_event.is_set()
        return bool(cancel_event)

    # Load cache
    cache: dict[str, dict] = {}
    if cache_path and cache_path.exists():
        try:
            with open(cache_path, encoding="utf-8") as f:
                cache = json.load(f)
            logger.info("  OA cache: loaded %d entries", len(cache))
        except Exception:
            cache = {}

    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})

    total = len(targets)
    cached_hits = 0
    api_hits = 0
    api_errors = 0

    def _emit_oa_progress(current: int) -> None:
        msg = f"  OA check [{current}/{total}] — {api_hits} queried, {cached_hits} cached"
        logger.info(msg)
        if not progress_callback:
            return
        if progress_label:
            progress_callback({
                "message": f"{progress_label}: {current}/{total} checked",
                "current": current,
                "total": total,
            })
        else:
            progress_callback(msg)

    for i, target in enumerate(targets):
        if _cancelled():
            break

        doi = target.get("doi", "")
        if not doi:
            continue

        # Check cache
        if doi in cache:
            entry = cache[doi]
            target["is_oa"] = entry.get("is_oa", False)
            target["oa_url"] = entry.get("oa_url", "")
            target["host_type"] = entry.get("host_type", "")
            cached_hits += 1
            if (i + 1) % 10 == 0 or i + 1 == total:
                _emit_oa_progress(i + 1)
            continue

        # Query Unpaywall
        url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                is_oa = data.get("is_oa", False)
                oa_url = ""
                host_type = ""
                best_loc = data.get("best_oa_location") or {}
                if is_oa and best_loc:
                    oa_url = (
                        best_loc.get("url_for_pdf")
                        or best_loc.get("url_for_landing_page")
                        or best_loc.get("url", "")
                    )
                    host_type = best_loc.get("host_type", "")

                target["is_oa"] = is_oa
                target["oa_url"] = oa_url
                target["host_type"] = host_type

                cache[doi] = {
                    "is_oa": is_oa, "oa_url": oa_url,
                    "host_type": host_type, "queried": time.strftime("%Y-%m-%d"),
                }
                api_hits += 1
            elif resp.status_code == 404:
                target["is_oa"] = False
                target["oa_url"] = ""
                cache[doi] = {"is_oa": False, "oa_url": "", "host_type": "",
                              "queried": time.strftime("%Y-%m-%d")}
                api_hits += 1
            else:
                logger.warning("  Unpaywall HTTP %d for DOI %s", resp.status_code, doi)
                target["is_oa"] = False
                api_errors += 1
        except Exception as exc:
            logger.warning("  Unpaywall error for DOI %s: %s", doi, exc)
            target["is_oa"] = False
            api_errors += 1

        # Rate limit: ~10 req/s
        time.sleep(0.1)

        if (i + 1) % 10 == 0 or i + 1 == total:
            _emit_oa_progress(i + 1)

    # Save cache
    if cache_path:
        try:
            atomic_write_json(cache_path, cache)
        except Exception as exc:
            logger.warning("Could not save OA cache: %s", exc)

    logger.info("  OA status: %d cached, %d API hits, %d errors",
                cached_hits, api_hits, api_errors)

    return targets


# ---------------------------------------------------------------------------
# Phase 3: Retrieve full text
# ---------------------------------------------------------------------------
def retrieve_fulltext(
    oa_papers: list[dict],
    output_dir: Path,
    progress_callback: Optional[Callable[[str], None]] = None,
    cancel_event: Any = None,
    workers: int = 8,
) -> list[dict]:
    """Download full text and extract plain text from PDF/HTML."""
    requests = _import_requests()

    def _cancelled() -> bool:
        if cancel_event is None:
            return False
        if hasattr(cancel_event, "is_set"):
            return cancel_event.is_set()
        return bool(cancel_event)

    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})

    results: list[dict] = []
    total = len(oa_papers)

    for i, paper in enumerate(oa_papers):
        if _cancelled():
            break

        doi = paper.get("doi", "")
        title = paper.get("title", "")[:60]
        oa_url = paper.get("oa_url", "")

        if not oa_url:
            logger.warning("  [%d/%d] No OA URL for %s — skipping", i + 1, total, doi)
            results.append({**paper, "text": "", "format": "none", "saved_file": ""})
            continue

        safe_name = _safe_filename(doi)
        result_entry: dict[str, Any] = {
            **paper, "text": "", "format": "unknown",
            "saved_file": "", "size_bytes": 0, "success": False,
        }

        try:
            resp = session.get(oa_url, timeout=60, allow_redirects=True)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "").lower()
            content = resp.content
            result_entry["size_bytes"] = len(content)

            if "pdf" in content_type or oa_url.endswith(".pdf"):
                # Save PDF
                pdf_path = output_dir / f"{safe_name}.pdf"
                pdf_path.write_bytes(content)
                result_entry["saved_file"] = str(pdf_path)
                result_entry["format"] = "pdf"
                result_entry["text"] = _extract_text_from_pdf(pdf_path)
            elif "html" in content_type or "xml" in content_type:
                # Save HTML
                html_path = output_dir / f"{safe_name}.html"
                html_path.write_bytes(content)
                result_entry["saved_file"] = str(html_path)
                result_entry["format"] = "html"
                result_entry["text"] = _extract_text_from_html(content)
            else:
                # Try as PDF first, then HTML
                pdf_path = output_dir / f"{safe_name}.bin"
                pdf_path.write_bytes(content)
                result_entry["saved_file"] = str(pdf_path)
                text = _extract_text_from_pdf(pdf_path)
                if text and len(text) > 100:
                    result_entry["format"] = "pdf"
                    result_entry["text"] = text
                else:
                    text = _extract_text_from_html(content)
                    if text and len(text) > 100:
                        result_entry["format"] = "html"
                        result_entry["text"] = text

            status = "ok" if result_entry["text"] else "no-text"
            result_entry["success"] = status == "ok"
            size_kb = result_entry["size_bytes"] // 1024
            msg = (f"  [{i + 1}/{total}] Downloaded: {title} "
                   f"({result_entry['format']}, {size_kb}KB) [{status}]")
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        except Exception as exc:
            logger.warning("  [%d/%d] Download failed for %s: %s",
                           i + 1, total, doi, exc)
            result_entry["format"] = "error"
            if progress_callback:
                progress_callback(f"  [{i + 1}/{total}] Failed: {title} — {exc}")

        results.append(result_entry)

    return results


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------
def _extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF. Try fitz (PyMuPDF) -> pypdf -> skip."""
    # Try PyMuPDF (fitz)
    try:
        import fitz  # type: ignore[import-untyped]
        doc = fitz.open(str(pdf_path))
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        text = "\n".join(pages).strip()
        if text:
            logger.debug("  PDF extracted with fitz: %d chars", len(text))
            return text
    except ImportError:
        logger.debug("  fitz (PyMuPDF) not available")
    except Exception as exc:
        logger.debug("  fitz extraction failed: %s", exc)

    # Try pypdf
    try:
        from pypdf import PdfReader  # type: ignore[import-untyped]
        reader = PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
        text = "\n".join(pages).strip()
        if text:
            logger.debug("  PDF extracted with pypdf: %d chars", len(text))
            return text
    except ImportError:
        logger.debug("  pypdf not available")
    except Exception as exc:
        logger.debug("  pypdf extraction failed: %s", exc)

    logger.warning("  Could not extract text from PDF: %s", pdf_path.name)
    return ""


# ---------------------------------------------------------------------------
# HTML text extraction
# ---------------------------------------------------------------------------
def _extract_text_from_html(content: bytes) -> str:
    """Extract text from HTML. Try BeautifulSoup -> regex fallback."""
    html_str = ""
    for enc in ("utf-8", "latin-1"):
        try:
            html_str = content.decode(enc)
            break
        except (UnicodeDecodeError, AttributeError):
            continue
    if not html_str and isinstance(content, str):
        html_str = content

    if not html_str:
        return ""

    # Try BeautifulSoup
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_str, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "header", "footer"]):
            tag.decompose()
        text = soup.get_text(separator=" ").strip()
        if text and len(text) > 100:
            logger.debug("  HTML extracted with BeautifulSoup: %d chars", len(text))
            return text
    except ImportError:
        logger.debug("  BeautifulSoup not available")
    except Exception as exc:
        logger.debug("  BeautifulSoup extraction failed: %s", exc)

    # Fallback: regex strip tags
    text = re.sub(r"<script[^>]*>.*?</script>", "", html_str, flags=re.S | re.I)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text:
        logger.debug("  HTML extracted with regex fallback: %d chars", len(text))
    return text


# ---------------------------------------------------------------------------
# Phase 4: Metric extraction via regex
# ---------------------------------------------------------------------------
def extract_metrics_from_text(text: str, paper_id: str = "") -> dict:
    """Extract quantitative metrics using regex.

    Returns dict with effect_sizes, sample_sizes, confidence_intervals, p_values.
    """
    effect_sizes: list[dict] = []
    sample_sizes: list[dict] = []
    confidence_intervals: list[dict] = []
    p_values: list[dict] = []

    if not text:
        return {
            "effect_sizes": effect_sizes,
            "sample_sizes": sample_sizes,
            "confidence_intervals": confidence_intervals,
            "p_values": p_values,
        }

    # ── Effect sizes ──────────────────────────────────────────────────
    # Cohen's d, Hedges' g: "d = 0.45", "g = 0.32", "Cohen's d = 1.2"
    for m in re.finditer(
        r"(?:Cohen[''\u2019]?s\s+)?([dg])\s*=\s*(-?\d+\.\d{1,4})",
        text, re.IGNORECASE,
    ):
        val = float(m.group(2))
        # Filter: effect sizes are typically -5 to 5
        if -5.0 <= val <= 5.0:
            effect_sizes.append({
                "type": m.group(1).lower(),
                "value": val,
                "context": _get_context(text, m.start(), 40),
            })

    # Hedges' g explicitly: "Hedges' g = 0.32"
    for m in re.finditer(
        r"Hedges[''\u2019]?\s*g\s*=\s*(-?\d+\.\d{1,4})",
        text, re.IGNORECASE,
    ):
        val = float(m.group(1))
        if -5.0 <= val <= 5.0:
            # Avoid duplicates from the pattern above
            if not any(e["value"] == val and e["type"] == "g" for e in effect_sizes):
                effect_sizes.append({
                    "type": "g",
                    "value": val,
                    "context": _get_context(text, m.start(), 40),
                })

    # Eta-squared: "η² = 0.12", "eta-squared = 0.12", "partial η² = 0.08"
    for m in re.finditer(
        r"(?:partial\s+)?(?:\u03b7\u00b2|\u03b7-squared|eta[- ]?squared)"
        r"\s*=\s*(\d+\.\d{1,4})",
        text, re.IGNORECASE,
    ):
        val = float(m.group(1))
        if 0.0 <= val <= 1.0:
            effect_sizes.append({
                "type": "eta_squared",
                "value": val,
                "context": _get_context(text, m.start(), 40),
            })

    # Odds ratio / risk ratio: "OR = 2.3", "RR = 1.5"
    for m in re.finditer(
        r"\b(OR|RR)\s*=\s*(\d+\.\d{1,4})", text,
    ):
        val = float(m.group(2))
        if 0.01 <= val <= 100.0:
            effect_sizes.append({
                "type": m.group(1).lower(),
                "value": val,
                "context": _get_context(text, m.start(), 40),
            })

    # ── Sample sizes ──────────────────────────────────────────────────
    # "N = 150", "n = 42", "(N = 200)", "participants (N=500)"
    for m in re.finditer(
        r"(?:participants|sample|subjects|patients)?\s*"
        r"[\(\[]?\s*[Nn]\s*=\s*(\d{2,6})\s*[\)\]]?",
        text,
    ):
        val = int(m.group(1))
        # Filter out years (1900-2099) and page numbers (typically < 10)
        if 10 <= val <= 999999 and not (1900 <= val <= 2099):
            sample_sizes.append({
                "value": val,
                "context": _get_context(text, m.start(), 40),
            })

    # "total of 1,234 participants"
    for m in re.finditer(
        r"(?:total\s+(?:of\s+)?|included\s+|enrolled\s+)"
        r"(\d{1,3}(?:,\d{3})*)\s+"
        r"(?:participants|patients|subjects|individuals)",
        text, re.IGNORECASE,
    ):
        val_str = m.group(1).replace(",", "")
        val = int(val_str)
        if 10 <= val <= 999999:
            sample_sizes.append({
                "value": val,
                "context": _get_context(text, m.start(), 40),
            })

    # ── Confidence intervals ──────────────────────────────────────────
    # "95% CI [0.21, 0.68]", "95% CI: 0.15-0.55", "CI 0.3 to 0.8"
    for m in re.finditer(
        r"(\d{2,3})%?\s*CI\s*[:=]?\s*"
        r"[\[\(]?\s*(-?\d+\.?\d*)\s*[,;\s]+\s*(-?\d+\.?\d*)\s*[\]\)]?",
        text, re.IGNORECASE,
    ):
        ci_level = int(m.group(1))
        low = float(m.group(2))
        high = float(m.group(3))
        if ci_level in (90, 95, 99) and low < high:
            confidence_intervals.append({
                "level": ci_level,
                "lower": low,
                "upper": high,
                "context": _get_context(text, m.start(), 50),
            })

    # "CI: 0.15 to 0.55" variant
    for m in re.finditer(
        r"(\d{2,3})%?\s*CI\s*[:=]?\s*"
        r"(-?\d+\.?\d*)\s+to\s+(-?\d+\.?\d*)",
        text, re.IGNORECASE,
    ):
        ci_level = int(m.group(1))
        low = float(m.group(2))
        high = float(m.group(3))
        if ci_level in (90, 95, 99) and low < high:
            # Avoid duplicates
            if not any(
                c["level"] == ci_level and c["lower"] == low and c["upper"] == high
                for c in confidence_intervals
            ):
                confidence_intervals.append({
                    "level": ci_level,
                    "lower": low,
                    "upper": high,
                    "context": _get_context(text, m.start(), 50),
                })

    # ── p-values ──────────────────────────────────────────────────────
    # "p < 0.001", "p = 0.03", "p<.05", "P = .001"
    for m in re.finditer(
        r"\bp\s*([<>=≤≥])\s*\.?(\d+\.?\d*(?:[eE][+-]?\d+)?)",
        text, re.IGNORECASE,
    ):
        operator = m.group(1)
        val_str = m.group(2)
        # Ensure it starts with "0." if no leading zero
        if not val_str.startswith("0") and "." not in val_str:
            val_str = "0." + val_str
        try:
            val = float(val_str)
        except ValueError:
            continue
        # p-values should be 0 < p <= 1
        if 0.0 < val <= 1.0:
            p_values.append({
                "operator": operator,
                "value": val,
                "context": _get_context(text, m.start(), 40),
            })

    if effect_sizes or sample_sizes or confidence_intervals or p_values:
        logger.debug(
            "  Metrics for %s: %d effect sizes, %d sample sizes, "
            "%d CIs, %d p-values",
            paper_id, len(effect_sizes), len(sample_sizes),
            len(confidence_intervals), len(p_values),
        )

    return {
        "effect_sizes": effect_sizes,
        "sample_sizes": sample_sizes,
        "confidence_intervals": confidence_intervals,
        "p_values": p_values,
    }


# ---------------------------------------------------------------------------
# check_output
# ---------------------------------------------------------------------------
def check_output(config) -> dict:
    """Check whether Stage 7 outputs exist."""
    ft_dir = config.data_dir / "fulltext"
    metrics = ft_dir / "auto_extracted_metrics.json"
    oa_status = ft_dir / "oa_status.json"
    download_log = ft_dir / "download_log.json"
    return {
        "metrics_exists": metrics.exists(),
        "oa_status_exists": oa_status.exists(),
        "download_log_exists": download_log.exists(),
        "fulltext_dir": str(ft_dir),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_filename(doi: str) -> str:
    """Convert a DOI to a safe filename."""
    safe = re.sub(r"[^\w\-.]", "_", doi)
    # Truncate to avoid filesystem limits
    return safe[:120]


def _get_context(text: str, pos: int, window: int = 40) -> str:
    """Extract a context snippet around a match position."""
    start = max(0, pos - window)
    end = min(len(text), pos + window)
    snippet = text[start:end].replace("\n", " ").strip()
    return snippet
