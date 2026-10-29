"""Unit tests for agent tools, resilience, and error handling."""

import asyncio

import pytest

from agent.config import Budget, Settings
from agent.executor import filter_and_rank_github_candidates
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


def test_github_candidate_selection_filters_junk_and_ranks():
    candidates = [
        "qdrant/qdrant-stars-handbook",
        "awesome-vectors/awesome-rag",
        "developer/my-tutorial",
        "qdrant/qdrant",
        "milvus-io/milvus",
        "random/unrelated-project",
    ]
    query = "Compare Qdrant vs Milvus on GitHub adoption"
    chosen, rejected = filter_and_rank_github_candidates(candidates, query)

    # Chosen must prioritize the exact match and entity repos
    assert "qdrant/qdrant" in chosen
    assert "milvus-io/milvus" in chosen
    assert "qdrant/qdrant-stars-handbook" not in chosen
    assert "awesome-vectors/awesome-rag" not in chosen
    assert "developer/my-tutorial" not in chosen

    # Check rejection reasons
    reasons = {r["repo"]: r["reason"] for r in rejected}
    assert reasons["qdrant/qdrant-stars-handbook"] == "contains_junk_keyword"
    assert reasons["awesome-vectors/awesome-rag"] == "contains_junk_keyword"
    assert reasons["developer/my-tutorial"] == "contains_junk_keyword"
    assert reasons["random/unrelated-project"] == "no_entity_token_match"
