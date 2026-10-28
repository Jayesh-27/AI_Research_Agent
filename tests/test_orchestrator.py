"""End-to-end integration tests for the research orchestrator."""

from unittest.mock import AsyncMock, patch

import pytest

from agent.config import Settings
from agent.orchestrator import run_research
from agent.schemas import (
    EvidenceExtraction,
    EvidenceItem,
    FinalAnswer,
    KeyFinding,
    Plan,
    RelevanceBatch,
    RelevanceScore,
    SubQuestion,
    SynthesisOutput,
    TaskType,
    ToolName,
)
from agent.tools.base import ToolResult


@pytest.mark.asyncio
async def test_orchestrator_rejects_too_short_question():
    res = await run_research("Hi")
    assert res.status == "failed"
    assert "too short" in res.short_answer.lower()


@pytest.mark.asyncio
async def test_orchestrator_mocked_end_to_end():
    mock_plan = Plan(
        task_type=TaskType.comparison,
        interpretation="Compare vector databases for RAG",
        assumptions=["Open-source focus"],
        sub_questions=[
            SubQuestion(
                id="sq1",
                question="What are top open-source vector DBs?",
                rationale="Core comparison",
                tools=[ToolName.web_search, ToolName.fetch_page],
                search_queries=["vector databases for RAG"],
            )
        ],
        success_criteria=["Overview of features"],
    )

    mock_relevance = RelevanceBatch(
        scores=[RelevanceScore(url="https://qdrant.tech/features", score=3)]
    )

    mock_extraction = EvidenceExtraction(
        items=[
            EvidenceItem(
                claim="Qdrant provides vector search for RAG",
                quote="Qdrant provides vector search with fast filtering.",
            )
        ]
    )

    mock_synthesis = SynthesisOutput(
        short_answer="Qdrant is a performant vector database suited for RAG.",
        key_findings=[
            KeyFinding(
                statement="Qdrant provides vector search with fast filtering.",
                evidence_ids=["e1"],
            )
        ],
        assumptions=["Assumes self-hosted deployment"],
        limitations=["Benchmarks vary by hardware"],
        next_steps=["Evaluate on test cluster"],
        self_reported_confidence="high",
    )

    fake_page_text = (
        "Qdrant provides vector search with fast filtering. Designed for production workloads."
    )

    with (
        patch("agent.orchestrator.create_plan", new_callable=AsyncMock) as p_plan,
        patch("agent.executor.web_search", new_callable=AsyncMock) as p_search,
        patch("agent.executor.fetch_page", new_callable=AsyncMock) as p_fetch,
        patch("agent.filtering.call_json", new_callable=AsyncMock) as p_rel,
        patch("agent.evidence.call_json", new_callable=AsyncMock) as p_extract,
        patch("agent.orchestrator.synthesize_answer", new_callable=AsyncMock) as p_synth,
    ):
        p_plan.return_value = mock_plan
        p_search.return_value = ToolResult(
            ok=True,
            data=[
                {
                    "url": "https://qdrant.tech/features",
                    "title": "Qdrant Features",
                    "snippet": "Qdrant provides vector search with fast filtering.",
                    "provider": "tavily",
                }
            ],
        )
        p_fetch.return_value = ToolResult(
            ok=True,
            data={
                "url": "https://qdrant.tech/features",
                "title": "Qdrant Features",
                "text": fake_page_text,
                "full_text": fake_page_text,
                "fetched_at": "2026-10-28T12:00:00Z",
                "truncated": False,
            },
        )
        p_rel.return_value = mock_relevance
        p_extract.return_value = mock_extraction
        p_synth.return_value = mock_synthesis

        settings = Settings(LLM_API_KEY="test_key", MAX_REPLANS=0)
        result: FinalAnswer = await run_research(
            "Compare vector databases for RAG",
            settings=settings,
        )

        assert result.status in ("ok", "partial")
        assert len(result.key_findings) == 1
        assert len(result.sources) == 1
        assert result.sources[0].url == "https://qdrant.tech/features"
        assert result.stats is not None
        assert result.confidence.score > 0.0
