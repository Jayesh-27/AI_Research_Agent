"""Unit tests for agent tools, resilience, and error handling."""

import asyncio

import pytest

from agent.config import Budget, Settings
from agent.tools.base import ToolResult, tool_call
from agent.tools.fetch_page import fetch_page
from agent.tools.github_stats import github_repo_stats
from agent.tools.web_search import web_search


@pytest.mark.asyncio
async def test_tool_call_decorator_timeout():
    @tool_call("sleepy_tool", timeout_s=0.1, retries=0, use_cache=False)
    async def sleepy():
        await asyncio.sleep(0.5)
        return ToolResult(ok=True, data={"res": "done"})

    res = await sleepy()
    assert res.ok is False
    assert res.error == "timeout"


@pytest.mark.asyncio
async def test_fetch_page_blocks_ssrf():
    settings = Settings()
    budget = Budget(settings)

    res = await fetch_page("http://127.0.0.1:8000/private", settings=settings, budget=budget)
    assert res.ok is False
    assert res.error == "unsafe_url"


@pytest.mark.asyncio
async def test_github_repo_stats_invalid_format():
    settings = Settings()
    budget = Budget(settings)

    res = await github_repo_stats(
        "invalid-repo-name-without-slash", settings=settings, budget=budget
    )
    assert res.ok is False
    assert "invalid_repo_format" in str(res.error)


@pytest.mark.asyncio
async def test_web_search_chaos_empty():
    settings = Settings()
    budget = Budget(settings)

    res = await web_search("query", settings=settings, budget=budget, chaos_mode="empty")
    assert res.ok is True
    assert res.data == []
