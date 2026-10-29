"""End-to-end integration tests for the research orchestrator."""

import asyncio
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


@pytest.mark.asyncio
async def test_orchestrator_timeout_with_evidence_yields_partial():
    mock_plan = Plan(
        task_type=TaskType.survey,
        interpretation="Survey vector DBs",
        assumptions=[],
        sub_questions=[
            SubQuestion(
                id="sq1",
                question="What are top open-source vector DBs?",
                rationale="Core comparison",
                tools=[ToolName.web_search],
                search_queries=["vector databases"],
            )
        ],
        success_criteria=[],
    )

    mock_synthesis = SynthesisOutput(
        short_answer="Partial synthesis from available evidence.",
        key_findings=[
            KeyFinding(
                statement="Qdrant is open-source vector store.",
                evidence_ids=["e1"],
            )
        ],
        assumptions=[],
        limitations=[],
        next_steps=[],
        self_reported_confidence="medium",
    )

    async def slow_execute_plan(sub_questions, ledger, **kwargs):
        ledger.add_structured_evidence(
            sub_question_id="sq1",
            claim="Qdrant is open-source vector store.",
            structured_data_str="Qdrant is an open-source vector database.",
            url="https://qdrant.tech",
            title="Qdrant",
        )
        await asyncio.sleep(1.5)

    with (
        patch("agent.orchestrator.create_plan", new_callable=AsyncMock) as p_plan,
        patch("agent.orchestrator.execute_plan", side_effect=slow_execute_plan),
        patch("agent.orchestrator.synthesize_answer", new_callable=AsyncMock) as p_synth,
    ):
        p_plan.return_value = mock_plan
        p_synth.return_value = mock_synthesis

        settings = Settings(
            LLM_API_KEY="test_key",
            PIPELINE_SOFT_DEADLINE_S=1,
            SYNTHESIS_TIMEOUT_S=2,
            RUN_TIMEOUT_S=3,
            MAX_REPLANS=0,
        )
        result: FinalAnswer = await run_research(
            "What are top open-source vector DBs?",
            settings=settings,
        )

        assert result.status == "partial"
        assert any(
            "deadline" in lim.lower() or "partial" in lim.lower() or "limit" in lim.lower()
            for lim in result.limitations
        )
        assert result.short_answer != ""
        assert not result.short_answer.startswith("Internal pipeline failure: ")
        assert result.stats is not None
        assert result.stats.latency_s > 0
        assert not any(w == "" for w in result.stats.warnings)


@pytest.mark.asyncio
async def test_orchestrator_timeout_with_empty_ledger_yields_failed():
    async def slow_create_plan(*args, **kwargs):
        await asyncio.sleep(0.5)

    with patch("agent.orchestrator.create_plan", side_effect=slow_create_plan):
        settings = Settings(
            LLM_API_KEY="test_key",
            PIPELINE_SOFT_DEADLINE_S=1,
            SYNTHESIS_TIMEOUT_S=1,
            RUN_TIMEOUT_S=1,
        )
        result: FinalAnswer = await run_research(
            "What are top open-source vector DBs?",
            settings=settings,
        )

        assert result.status == "failed"
        assert "No verified evidence gathered before the time limit." in result.short_answer
        assert result.short_answer != ""
        assert result.stats is not None
        assert not any(w == "" for w in result.stats.warnings)
