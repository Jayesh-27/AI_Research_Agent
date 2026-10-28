"""GitHub repo stats tool — fetches repository metadata from GitHub API.

Returns stars, forks, open issues, license, last push date, etc.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import httpx

from agent.config import Budget, Settings
from agent.tools.base import ToolResult

logger = logging.getLogger(__name__)

_REPO_PATTERN = re.compile(r"^[\w.-]+/[\w.-]+$")


async def github_repo_stats(
    repo: str,
    *,
    settings: Settings,
    budget: Budget,
) -> ToolResult:
    """Fetch GitHub repository stats for *repo* (format: 'owner/name').

    Returns structured data: stars, forks, open_issues, license,
    pushed_at, archived, description.
    """
    start = time.time()

    # Validate repo format
    if not _REPO_PATTERN.match(repo):
        latency = int((time.time() - start) * 1000)
        return ToolResult(
            ok=False,
            error=f"invalid_repo_format: '{repo}' (expected 'owner/name')",
            latency_ms=latency,
        )

    headers: dict[str, str] = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "research-agent/0.1",
    }
    token = (settings.GITHUB_TOKEN or "").strip()
    if token and not token.startswith("#"):
        headers["Authorization"] = f"Bearer {token}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"https://api.github.com/repos/{repo}",
                headers=headers,
            )

            if response.status_code == 404:
                latency = int((time.time() - start) * 1000)
                return ToolResult(ok=False, error="repo_not_found", latency_ms=latency)

            if response.status_code == 403:
                latency = int((time.time() - start) * 1000)
                return ToolResult(ok=False, error="http_403_rate_limit", latency_ms=latency)

            response.raise_for_status()
            data = response.json()

        repo_data: dict[str, Any] = {
            "repo": repo,
            "stars": data.get("stargazers_count", 0),
            "forks": data.get("forks_count", 0),
            "open_issues": data.get("open_issues_count", 0),
            "license": (data.get("license") or {}).get("spdx_id", "Unknown"),
            "pushed_at": data.get("pushed_at", ""),
            "archived": data.get("archived", False),
            "description": data.get("description", ""),
            "url": f"https://github.com/{repo}",
        }

        latency = int((time.time() - start) * 1000)
        return ToolResult(ok=True, data=repo_data, latency_ms=latency)

    except httpx.TimeoutException:
        latency = int((time.time() - start) * 1000)
        return ToolResult(ok=False, error="timeout", latency_ms=latency)

    except Exception as exc:
        latency = int((time.time() - start) * 1000)
        logger.error("github_repo_stats failed for %s: %s", repo, exc)
        return ToolResult(ok=False, error=str(exc), latency_ms=latency)
