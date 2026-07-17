"""Async OpenAlex API client."""
import asyncio
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

OA_BASE = "https://api.openalex.org"
OPENALEX_SELECT_FIELDS = (
    "id,ids,doi,title,publication_year,cited_by_count,"
    "abstract_inverted_index,authorships,type,"
    "open_access,best_oa_location,primary_location,has_content,content_url"
)
OPENALEX_FALLBACK_SELECT_FIELDS = (
    "id,ids,doi,title,publication_year,cited_by_count,"
    "abstract_inverted_index,authorships,type,"
    "open_access,best_oa_location,primary_location"
)


def invert_abstract(inverted_index: Optional[dict]) -> Optional[str]:
    """Reconstruct abstract text from OpenAlex inverted-index format."""
    if not inverted_index:
        return None
    try:
        word_positions: list[tuple[int, str]] = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        return " ".join(w for _, w in word_positions)
    except Exception:
        return None


class OpenAlexClient:
    """Async client for the OpenAlex API."""

    def __init__(self, api_key: str = "", rate_limit_delay: float = 0.1):
        headers: dict[str, str] = {}
        if api_key and "@" in api_key:
            # OpenAlex uses mailto for polite pool
            headers["User-Agent"] = f"mailto:{api_key}"
        self._client = httpx.AsyncClient(
            base_url=OA_BASE,
            headers=headers,
            timeout=30.0,
        )
        self._delay = rate_limit_delay

    async def _get_with_select_fallback(
        self, path: str, params: dict
    ) -> httpx.Response:
        """GET with automatic select-field fallback on 400/500."""
        params.setdefault("select", OPENALEX_SELECT_FIELDS)
        await asyncio.sleep(self._delay)
        resp = await self._client.get(path, params=params)
        if resp.status_code in (400, 500):
            logger.warning("OpenAlex: field set rejected (%d), retrying with fallback", resp.status_code)
            params["select"] = OPENALEX_FALLBACK_SELECT_FIELDS
            await asyncio.sleep(self._delay)
            resp = await self._client.get(path, params=params)
        return resp

    async def search(
        self,
        search_text: str,
        extra_filter: str = "has_abstract:true",
        max_results: int = 1000,
    ) -> list[dict]:
        """Search OpenAlex for works matching *search_text*."""
        if not search_text:
            return []

        papers: list[dict] = []
        per_page = 100
        page = 1

        while len(papers) < max_results:
            params: dict[str, str] = {
                "search": search_text,
                "per_page": str(per_page),
                "page": str(page),
            }
            if extra_filter:
                params["filter"] = extra_filter

            logger.info("OpenAlex search: page=%d query='%s'", page, search_text[:80])
            resp = await self._get_with_select_fallback("/works", params)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            total_count = data.get("meta", {}).get("count", 0)
            logger.info("OpenAlex: page %d → %d results (total=%d)", page, len(results), total_count)

            if not results:
                break

            papers.extend(results)
            page += 1
            if len(papers) >= max_results or page * per_page > total_count:
                break

        logger.info("OpenAlex search complete: %d papers", len(papers))
        return papers[:max_results]

    async def close(self):
        await self._client.aclose()
