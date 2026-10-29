"""Unit tests for Pydantic schema validation."""


from agent.schemas import (
    Confidence,
    Evidence,
    FinalAnswer,
    KeyFinding,
    Plan,
    RunStats,
    Source,
    SubQuestion,
    SynthesisOutput,
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


def test_sub_question_truncates_excess_queries():
    sq = SubQuestion(
        id="sq1",
        question="Too many queries",
        rationale="testing bounds",
        tools=[ToolName.web_search],
        search_queries=["q1", "q2", "q3", "q4"],  # 4 queries truncated to 3
    )
    assert len(sq.search_queries) == 3
    assert sq.search_queries == ["q1", "q2", "q3"]


def test_sub_question_dedupes_and_strips_queries():
    sq = SubQuestion(
        id="sq1",
        question="Queries with blanks and dups",
        rationale="testing bounds",
        tools=[ToolName.web_search],
        search_queries=["  q1  ", "", "q1", "q2", "   "],
    )
    assert sq.search_queries == ["q1", "q2"]


def test_plan_truncates_excess_sub_questions():
    sub_qs = [
        SubQuestion(
            id=f"sq{i}",
            question=f"Q{i}",
            rationale="testing",
            tools=[ToolName.web_search],
            search_queries=["q"],
        )
        for i in range(7)
    ]
    plan = Plan(
        task_type=TaskType.survey,
        interpretation="testing",
        sub_questions=sub_qs,
    )
    assert len(plan.sub_questions) == 5


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


def test_synthesis_output_drops_empty_evidence_ids_finding():
    data = {
        "short_answer": "Summary",
        "key_findings": [
            {"statement": "Valid finding", "evidence_ids": ["e1"]},
            {"statement": "Invalid finding with no IDs", "evidence_ids": []},
            {"statement": "Another invalid finding", "evidence_ids": [""]},
        ],
        "assumptions": ["  assumption 1  ", "assumption 1", ""],
        "limitations": [],
        "next_steps": [],
    }
    s = SynthesisOutput.model_validate(data)
    assert len(s.key_findings) == 1
    assert s.key_findings[0].statement == "Valid finding"
    assert s.assumptions == ["assumption 1"]
