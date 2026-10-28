"""Unit tests for Pydantic schema validation."""

import pytest
from pydantic import ValidationError

from agent.schemas import (
    Confidence,
    Evidence,
    FinalAnswer,
    KeyFinding,
    Plan,
    RunStats,
    Source,
    SubQuestion,
    TaskType,
    ToolName,
)


def test_sub_question_validation():
    sq = SubQuestion(
        id="sq1",
        question="What are the top vector DBs?",
        rationale="Identify contenders",
        tools=[ToolName.web_search, ToolName.github_repo_stats],
        search_queries=["top vector databases 2026", "open source vector db"],
    )
    assert sq.id == "sq1"
    assert len(sq.tools) == 2
    assert len(sq.search_queries) == 2


def test_sub_question_max_queries_exceeded():
    with pytest.raises(ValidationError):
        SubQuestion(
            id="sq1",
            question="Too many queries",
            rationale="testing bounds",
            tools=[ToolName.web_search],
            search_queries=["q1", "q2", "q3", "q4"],  # max_length is 3
        )


def test_plan_validation():
    plan = Plan(
        task_type=TaskType.comparison,
        interpretation="Comparing vector databases",
        assumptions=["Focused on open-source solutions"],
        sub_questions=[
            SubQuestion(
                id="sq1",
                question="Vector DB comparison",
                rationale="Contenders",
                tools=[ToolName.web_search],
                search_queries=["vector db"],
            )
        ],
        success_criteria=["Table comparing latency and features"],
        needs_clarification=False,
    )
    assert plan.task_type == TaskType.comparison
    assert len(plan.sub_questions) == 1


def test_evidence_validation():
    ev = Evidence(
        id="e1",
        sub_question_id="sq1",
        claim="Qdrant is written in Rust",
        quote="Qdrant is written in Rust and provides high-performance vector search.",
        url="https://qdrant.tech",
        title="Qdrant Tech",
        domain="qdrant.tech",
        fetched_at="2026-10-28T12:00:00Z",
        quote_verified=True,
        trust="normal",
    )
    assert ev.quote_verified is True
    assert ev.trust == "normal"


def test_final_answer_complete_structure():
    ans = FinalAnswer(
        status="ok",
        user_question="Compare vector DBs",
        short_answer="Qdrant and Milvus are leading choices.",
        key_findings=[
            KeyFinding(
                statement="Qdrant offers Rust-based speed",
                evidence_ids=["e1"],
            )
        ],
        sources=[
            Source(
                id="s1",
                title="Qdrant",
                url="https://qdrant.tech",
                domain="qdrant.tech",
                used_in=["Finding 1"],
            )
        ],
        confidence=Confidence(
            level="high",
            score=0.88,
            rationale=["High source authority", "Complete coverage"],
        ),
        assumptions=["Focus on self-hosted"],
        limitations=["Benchmarks depend on hardware"],
        next_steps=["Test Qdrant on staging dataset"],
        stats=RunStats(
            run_id="run-123",
            latency_s=5.2,
            llm_calls=3,
            tokens_in=1200,
            tokens_out=400,
            est_cost_usd=0.0012,
        ),
    )
    assert ans.status == "ok"
    assert ans.confidence.level == "high"
    assert len(ans.key_findings) == 1
    assert len(ans.sources) == 1
