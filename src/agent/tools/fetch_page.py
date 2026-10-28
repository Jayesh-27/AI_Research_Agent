"""Fetch page tool — httpx + trafilatura extraction with SSRF guard.

Returns extracted main text from a web page.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from agent.config import Budget, Settings
from agent.safety import is_safe_url
from agent.tools.base import ToolResult

logger = logging.getLogger(__name__)


async def fetch_page(
    url: str,
    *,
    settings: Settings,
    budget: Budget,
    chaos_mode: str = "off",
) -> ToolResult:
    """Fetch and extract main text from *url*.

    Pre-checks URL safety (SSRF), enforces byte limits,
    extracts text with trafilatura.
    """
    start = time.time()
    budget.use_fetch()

    # SSRF guard
    if not is_safe_url(url):
        latency = int((time.time() - start) * 1000)
        logger.warning("Blocked unsafe URL: %s", url)
        return ToolResult(ok=False, error="unsafe_url", latency_ms=latency)

    # Chaos: random timeouts
    if chaos_mode == "timeout" and random.random() < 0.5:
        logger.info("CHAOS: timeout — sleeping past timeout for %s", url)
        await asyncio.sleep(settings.TOOL_TIMEOUT_S + 5)
        latency = int((time.time() - start) * 1000)
        return ToolResult(ok=False, error="timeout", latency_ms=latency)

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(settings.TOOL_TIMEOUT_S),
            follow_redirects=True,
            max_redirects=3,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

            # Content-type check
            content_type = response.headers.get("content-type", "")
            if not any(ct in content_type.lower() for ct in ("text/html", "text/plain")):
                latency = int((time.time() - start) * 1000)
                return ToolResult(ok=False, error="unsupported_content_type", latency_ms=latency)

            # Byte limit check
            raw_bytes = response.content
            if len(raw_bytes) > settings.FETCH_MAX_BYTES:
                raw_bytes = raw_bytes[: settings.FETCH_MAX_BYTES]

            html = raw_bytes.decode("utf-8", errors="replace")

        # Extract main text with trafilatura
        import trafilatura

        extracted = trafilatura.extract(
            html,
            favor_recall=True,
            include_comments=False,
            include_tables=True,
        )

        if not extracted or not extracted.strip():
            latency = int((time.time() - start) * 1000)
            return ToolResult(ok=False, error="empty_content", latency_ms=latency)

        # Truncate for return (keep full text for evidence verification)
        truncated = len(extracted) > 6000
        text_for_extraction = extracted[:6000] if truncated else extracted

        # Keep longer text for verbatim verification (up to 30k)
        full_text = extracted[:30000]

        data: dict[str, Any] = {
            "url": str(url),
            "title": _extract_title(html),
            "text": text_for_extraction,
            "full_text": full_text,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "truncated": truncated,
            "content_length": len(extracted),
        }

        latency = int((time.time() - start) * 1000)
        return ToolResult(ok=True, data=data, latency_ms=latency)

    except httpx.TimeoutException:
        latency = int((time.time() - start) * 1000)
        return ToolResult(ok=False, error="timeout", latency_ms=latency)

    except httpx.HTTPStatusError as exc:
        latency = int((time.time() - start) * 1000)
        code = exc.response.status_code
        error_type = "http_429" if code == 429 else f"http_{code}"
        return ToolResult(ok=False, error=error_type, latency_ms=latency)

    except Exception as exc:
        latency = int((time.time() - start) * 1000)
        logger.error("fetch_page failed for %s: %s", url, exc)
        return ToolResult(ok=False, error=str(exc), latency_ms=latency)


def _extract_title(html: str) -> str:
    """Extract <title> from HTML. Simple regex, no dependency."""
    import re

    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    return match.group(1).strip() if match else ""
