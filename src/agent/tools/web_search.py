"""Web search tool — Tavily (primary) with DuckDuckGo fallback.

Returns a list of SearchHit objects wrapped in a ToolResult.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

import httpx

from agent.config import Budget, Settings
from agent.schemas import SearchHit
from agent.tools.base import ToolResult

logger = logging.getLogger(__name__)


def simplify_query_for_search(query: str, max_words: int = 6) -> str:
    """Simplify query by removing operators, quotes, punctuation, and truncating words."""
    # Take the first segment if pipe-separated
    q = query.split("|")[0]
    # Remove quotes, punctuation, brackets, and arithmetic operators
    q = re.sub(r'["\'?:!(),;_{}\[\]*+~^]', " ", q)
    # Remove search syntax like site:, filetype:, inurl:, intitle:
    q = re.sub(r"\b(site|filetype|inurl|intitle):[^\s]+", " ", q, flags=re.IGNORECASE)
    # Remove boolean operators
    q = re.sub(r"\b(AND|OR|NOT)\b", " ", q)
    words = [w.strip() for w in q.split() if w.strip()]
    if len(words) > max_words:
        words = words[:max_words]
    return " ".join(words)


async def web_search(
    query: str,
    *,
    max_results: int = 8,
    settings: Settings,
    budget: Budget,
    chaos_mode: str = "off",
) -> ToolResult:
    """Search the web for *query* using Tavily, falling back to DuckDuckGo.

    Returns ToolResult with data = list of SearchHit dicts. Never raises.
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

    # Try Tavily first if configured
    if settings.TAVILY_API_KEY:
        try:
            result = await _tavily_search(query, max_results, settings.TAVILY_API_KEY, start)
            if result.ok and result.data:
                return result
            logger.warning("Tavily returned empty, falling back to DDG")
        except Exception as exc:
            logger.warning("Tavily search failed (%s), falling back to DDG", exc)
    else:
        logger.debug("Tavily not configured, using DDG directly")

    # Fallback to DuckDuckGo
    ddg_res = await _ddg_search(query, max_results, start)
    if ddg_res.ok and ddg_res.data:
        return ddg_res

    # If both providers failed or returned empty results, record warning once
    short_q = query[:60] + ("..." if len(query) > 60 else "")
    warn_text = f"Web search returned no results from search providers for query: '{short_q}'"
    if hasattr(budget, "warnings") and warn_text not in budget.warnings:
        budget.warnings.append(warn_text)
    logger.warning(warn_text)

    return ddg_res


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


_ddg_locks: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}


def _get_ddg_lock() -> asyncio.Lock:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.Lock()
    if loop not in _ddg_locks:
        _ddg_locks[loop] = asyncio.Lock()
    return _ddg_locks[loop]


def _do_ddg_search_sync(query: str, max_results: int) -> list[dict[str, Any]]:
    """Synchronous DuckDuckGo search with query simplification and backend fallback."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.warning("Neither ddgs nor duckduckgo_search is installed")
            return []

    try:
        from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
    except ImportError:
        try:
            from duckduckgo_search.exceptions import (
                DuckDuckGoSearchException as DDGSException,
            )
            from duckduckgo_search.exceptions import (
                RatelimitException,
                TimeoutException,
            )
        except ImportError:
            DDGSException = Exception  # type: ignore[assignment, misc]
            RatelimitException = Exception  # type: ignore[assignment, misc]
            TimeoutException = Exception  # type: ignore[assignment, misc]

    # Initial query cleanup
    initial_q = query.split("|")[0].replace('"', "").replace("'", "").strip()
    words = initial_q.split()
    if len(words) > 8:
        initial_q = " ".join(words[:8])

    # 1. Primary attempt: auto backend
    try:
        results = list(DDGS().text(initial_q, max_results=max_results, backend="auto") or [])
        if results:
            return results
        logger.info(
            "DDGS auto backend returned empty for query '%s'; retrying with simplified query",
            initial_q,
        )
    except (RatelimitException, TimeoutException, DDGSException, Exception) as exc:
        logger.warning(
            "DDGS auto backend failed for '%s' (%s: %s); retrying with simplified query and alternative backend",
            initial_q,
            type(exc).__name__,
            exc,
        )

    # 2. Retry with simplified query (<= 6 words, stripped operators) and html backend
    simpler_q = simplify_query_for_search(query, max_words=6)
    if not simpler_q:
        return []

    try:
        results = list(DDGS().text(simpler_q, max_results=max_results, backend="html") or [])
        if results:
            return results
        logger.info(
            "DDGS html backend returned empty for query '%s'; trying lite backend",
            simpler_q,
        )
    except (RatelimitException, TimeoutException, DDGSException, Exception) as exc:
        logger.warning(
            "DDGS html backend failed for '%s' (%s: %s); trying lite backend",
            simpler_q,
            type(exc).__name__,
            exc,
        )

    # 3. Final fallback: lite backend
    try:
        results = list(DDGS().text(simpler_q, max_results=max_results, backend="lite") or [])
        if results:
            return results
        logger.warning("DDGS lite backend returned empty for query '%s'", simpler_q)
    except (RatelimitException, TimeoutException, DDGSException, Exception) as exc:
        logger.warning(
            "DDGS lite backend failed for '%s' (%s: %s)",
            simpler_q,
            type(exc).__name__,
            exc,
        )

    return []


async def _ddg_search(
    query: str,
    max_results: int,
    start: float,
) -> ToolResult:
    """Fallback search using DuckDuckGo via ddgs. Never raises."""
    try:
        async with _get_ddg_lock():
            loop = asyncio.get_event_loop()
            raw_results = await loop.run_in_executor(
                None, lambda: _do_ddg_search_sync(query, max_results)
            )
            await asyncio.sleep(0.2)

        hits: list[dict[str, Any]] = []
        for item in raw_results:
            try:
                url = item.get("href") or item.get("link") or item.get("url") or ""
                if not url:
                    continue
                hit = SearchHit(
                    url=url,
                    title=item.get("title", ""),
                    snippet=item.get("body", item.get("snippet", "")),
                    provider="ddg",
                )
                hits.append(hit.model_dump())
            except Exception as exc:
                logger.warning("Dropped malformed DDG result: %s", exc)

        latency = int((time.time() - start) * 1000)
        return ToolResult(ok=True, data=hits, latency_ms=latency)

    except Exception as exc:
        latency = int((time.time() - start) * 1000)
        logger.warning("DDG search returned empty due to error: %s", exc)
        return ToolResult(ok=True, data=[], error=f"ddg_error: {exc}", latency_ms=latency)

