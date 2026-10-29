"""Unit tests for agent tools, resilience, and error handling."""

import asyncio

import pytest

from agent.config import Budget, Settings
from agent.executor import filter_and_rank_github_candidates
from agent.filtering import filter_and_rank_hits
from agent.schemas import SearchHit
from agent.tools.base import ToolResult, tool_call
from agent.tools.fetch_page import fetch_page
from agent.tools.github_stats import github_repo_stats
from agent.tools.web_search import simplify_query_for_search, web_search


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


def test_simplify_query_for_search():
    # Strips quotes, boolean operators, pipes, and bounds word count
    q1 = '"Milvus GitHub stars" | Qdrant repository statistics | Weaviate GitHub activity'
    assert simplify_query_for_search(q1, max_words=6) == "Milvus GitHub stars"

    q2 = "vector database benchmark AND latency OR recall NOT memory"
    assert simplify_query_for_search(q2, max_words=6) == "vector database benchmark latency recall memory"

    q3 = "Compare the top 3 open-source vector databases for RAG?"
    words = simplify_query_for_search(q3, max_words=4).split()
    assert len(words) == 4
    assert "?" not in words


@pytest.mark.asyncio
async def test_ddg_fallback_never_raises_on_ratelimit(monkeypatch):
    settings = Settings(TAVILY_API_KEY="")
    budget = Budget(settings)

    class FakeRatelimitException(Exception):
        pass

    def mock_text(*args, **kwargs):
        raise FakeRatelimitException("Rate limit reached")

    for mod_name in ["ddgs.ddgs", "ddgs", "duckduckgo_search.duckduckgo_search", "duckduckgo_search"]:
        try:
            mod = __import__(mod_name, fromlist=["DDGS"])
            if hasattr(mod, "DDGS"):
                monkeypatch.setattr(mod.DDGS, "text", mock_text)
        except ImportError:
            pass

    # Must never raise an exception and should return empty list gracefully
    res = await web_search("test query", settings=settings, budget=budget)
    assert res.ok is True
    assert res.data == []
    # Warning recorded in budget
    assert any("returned no results from search providers" in w for w in budget.warnings)


@pytest.mark.asyncio
async def test_ddg_fallback_retries_with_alt_backend(monkeypatch):
    settings = Settings(TAVILY_API_KEY="")
    budget = Budget(settings)

    calls = []

    def mock_text(self, keywords, backend="auto", max_results=None, **kwargs):
        calls.append((keywords, backend))
        if backend == "auto":
            return []  # fails/empty on auto
        elif backend in ("duckduckgo", "html", "brave"):
            return [
                {
                    "href": "https://example.com/test",
                    "title": "Test Title",
                    "body": "Test snippet from alternative backend",
                }
            ]
        return []

    for mod_name in ["ddgs.ddgs", "ddgs", "duckduckgo_search.duckduckgo_search", "duckduckgo_search"]:
        try:
            mod = __import__(mod_name, fromlist=["DDGS"])
            if hasattr(mod, "DDGS"):
                monkeypatch.setattr(mod.DDGS, "text", mock_text)
        except ImportError:
            pass

    res = await web_search("complex AND noisy query | extra", settings=settings, budget=budget)
    assert res.ok is True
    assert len(res.data) == 1
    assert res.data[0]["url"] == "https://example.com/test"
    # Verify retry occurred with alternative backend and simplified query
    assert any(backend in ("duckduckgo", "html", "brave") for _, backend in calls)


@pytest.mark.asyncio
async def test_relevance_scoring_degradation_records_warning(monkeypatch):
    settings = Settings()
    budget = Budget(settings)

    async def mock_call_json(*args, **kwargs):
        raise RuntimeError("Simulated 429 rate limit")

    monkeypatch.setattr("agent.filtering.call_json", mock_call_json)

    hits = [
        SearchHit(url="https://example.com/1", title="Hit 1", snippet="Snippet 1"),
        SearchHit(url="https://example.com/2", title="Hit 2", snippet="Snippet 2"),
    ]

    ranked = await filter_and_rank_hits(hits, "test question", settings=settings, budget=budget)
    # Defaulted to 2, so results are kept
    assert len(ranked) == 2
    # Warning is recorded in budget
    assert any("Relevance scoring degraded" in w for w in budget.warnings)


@pytest.mark.asyncio
async def test_groq_daily_token_limit_no_retries_and_fallback(monkeypatch):
    import httpx
    import openai

    from agent.config import ModelSpec
    from agent.llm import (
        _call_with_retry_and_fallback,
        _should_retry_llm_call,
        is_groq_daily_token_limit,
    )
    from agent.schemas import Plan, TaskType

    # 1. Test detection logic
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp_tpd = httpx.Response(
        429,
        request=req,
        json={
            "error": {
                "message": (
                    "Rate limit reached for model on tokens per day (TPD): "
                    "Limit 100000, Used 100000. Please try again in 14h."
                )
            }
        },
    )
    tpd_err = openai.RateLimitError(
        message="Rate limit reached for model on tokens per day (TPD)",
        response=resp_tpd,
        body={"error": {"message": "Rate limit reached for model on tokens per day (TPD)"}},
    )

    resp_rpm = httpx.Response(
        429,
        request=req,
        json={"error": {"message": "Rate limit reached on requests per minute (RPM). Try again in 2s."}},
    )
    rpm_err = openai.RateLimitError(
        message="Rate limit reached on requests per minute (RPM)",
        response=resp_rpm,
        body={"error": {"message": "Rate limit reached on requests per minute (RPM)"}},
    )

    assert is_groq_daily_token_limit(tpd_err) is True
    assert is_groq_daily_token_limit(rpm_err) is False

    # 2. Test retry predicate returns False for TPD (no retries) and True for normal 429
    assert _should_retry_llm_call(tpd_err) is False
    assert _should_retry_llm_call(rpm_err) is True

    # 3. Test execution: verify NO retries on primary, warning recorded, and fallback used
    primary_attempts = 0
    fallback_attempts = 0

    async def mock_call_once(*, base_url, api_key, **kwargs):
        nonlocal primary_attempts, fallback_attempts
        if "fallback" in base_url or api_key == "fallback-key":
            fallback_attempts += 1
            return Plan(
                task_type=TaskType.survey,
                interpretation="Fallback success",
                sub_questions=[],
            )
        else:
            primary_attempts += 1
            raise tpd_err

    monkeypatch.setattr("agent.llm._call_once", mock_call_once)

    settings = Settings(
        LLM_BASE_URL="https://api.groq.com/openai/v1",
        LLM_API_KEY="groq-key",
        FALLBACK_LLM_BASE_URL="https://fallback.llm.com/v1",
        FALLBACK_LLM_API_KEY="fallback-key",
        FALLBACK_LARGE_MODEL="fallback-llama",
    )
    budget = Budget(settings)
    spec = ModelSpec(
        model_id="llama-3.3-70b-versatile",
        tier="large",
        price_in=0.0,
        price_out=0.0,
    )

    result = await _call_with_retry_and_fallback(
        spec=spec,
        system="System prompt",
        user="User question",
        schema=Plan,
        settings=settings,
        budget=budget,
        temperature=0.0,
        max_tokens=500,
        span_name="test_call",
    )

    # Exactly 1 primary attempt (NO RETRIES!)
    assert primary_attempts == 1
    # Fallback succeeded
    assert fallback_attempts == 1
    assert result.interpretation == "Fallback success"
    # Clear warning recorded in budget
    assert any("tokens per day" in w.lower() for w in budget.warnings)



