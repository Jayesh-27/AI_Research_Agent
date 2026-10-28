"""Web search tool — Tavily (primary) with DuckDuckGo fallback.

Returns a list of SearchHit objects wrapped in a ToolResult.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from agent.config import Budget, Settings
from agent.schemas import SearchHit
from agent.tools.base import ToolResult

logger = logging.getLogger(__name__)


async def web_search(
    query: str,
    *,
    max_results: int = 8,
    settings: Settings,
    budget: Budget,
    chaos_mode: str = "off",
) -> ToolResult:
    """Search the web for *query* using Tavily, falling back to DuckDuckGo.

    Returns ToolResult with data = list of SearchHit dicts.
    """
    start = time.time()
    budget.use_search()

    # Chaos: simulate search being down
    if chaos_mode == "search_down":
        logger.info("CHAOS: search_down — skipping Tavily, using DDG fallback")
        return await _ddg_search(query, max_results, start)

    if chaos_mode == "empty":
        logger.info("CHAOS: empty — returning no results")
        latency = int((time.time() - start) * 1000)
        return ToolResult(ok=True, data=[], latency_ms=latency)

    # Try Tavily first
    if settings.TAVILY_API_KEY:
        try:
            result = await _tavily_search(query, max_results, settings.TAVILY_API_KEY, start)
            if result.ok and result.data:
                return result
            logger.warning("Tavily returned empty, falling back to DDG")
        except Exception as exc:
            logger.warning("Tavily search failed (%s), falling back to DDG", exc)

    # Fallback to DuckDuckGo
    return await _ddg_search(query, max_results, start)


async def _tavily_search(
    query: str,
    max_results: int,
    api_key: str,
    start: float,
) -> ToolResult:
    """Call Tavily REST API."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        data = response.json()

    hits: list[dict[str, Any]] = []
    raw_results = data.get("results", [])

    for item in raw_results:
        try:
            hit = SearchHit(
                url=item.get("url", ""),
                title=item.get("title", ""),
                snippet=item.get("content", item.get("snippet", "")),
                published=item.get("published_date"),
                provider="tavily",
            )
            hits.append(hit.model_dump())
        except Exception as exc:
            logger.warning("Dropped malformed Tavily result: %s", exc)

    latency = int((time.time() - start) * 1000)
    return ToolResult(ok=True, data=hits, latency_ms=latency)


async def _ddg_search(
    query: str,
    max_results: int,
    start: float,
) -> ToolResult:
    """Fallback search using DuckDuckGo."""
    try:
        from duckduckgo_search import DDGS

        # DDG is synchronous, run in executor
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: list(DDGS().text(query, max_results=max_results)),
        )

        hits: list[dict[str, Any]] = []
        for item in results:
            try:
                hit = SearchHit(
                    url=item.get("href", item.get("link", "")),
                    title=item.get("title", ""),
                    snippet=item.get("body", item.get("snippet", "")),
                    provider="ddg",
                )
                hits.append(hit.model_dump())
            except Exception as exc:
                logger.warning("Dropped malformed DDG result: %s", exc)

        latency = int((time.time() - start) * 1000)
        if not hits:
            return ToolResult(ok=True, data=[], latency_ms=latency)
        return ToolResult(ok=True, data=hits, latency_ms=latency)

    except Exception as exc:
        latency = int((time.time() - start) * 1000)
        logger.error("DDG search also failed: %s", exc)
        return ToolResult(ok=False, error=f"all_search_failed: {exc}", latency_ms=latency)
