"""Unit tests for citation verification, confidence calculation, and guardrails."""

from agent.evidence import EvidenceLedger
from agent.schemas import (
    KeyFinding,
    Plan,
    RunStats,
    SubQuestion,
    SynthesisOutput,
    TaskType,
    ToolName,
)
from agent.verify import verify_and_construct_final_answer


def test_verify_drops_uncited_findings_and_builds_sources():
    ledger = EvidenceLedger()
    doc = "Chroma is an open-source embedding database for AI applications."
    ledger.add_evidence(
        sub_question_id="sq1",
        claim="Chroma is an embedding database",
        quote="Chroma is an open-source embedding database",
        url="https://trychroma.com",
        title="Chroma DB",
        document_text=doc,
    )

    plan = Plan(
        task_type=TaskType.survey,
        interpretation="Surveying Chroma",
        assumptions=[],
        sub_questions=[
            SubQuestion(
                id="sq1",
                question="What is Chroma?",
                rationale="Identify nature",
                tools=[ToolName.web_search],
                search_queries=["chroma db"],
            )
        ],
        success_criteria=[],
    )

    synthesis = SynthesisOutput(
        short_answer="Chroma is an open-source embedding database.",
        key_findings=[
            # Valid finding citing e1
            KeyFinding(statement="Chroma stores embeddings.", evidence_ids=["e1"]),
            # Invalid finding citing hallucinated e999
            KeyFinding(statement="Chroma was created in 1990.", evidence_ids=["e999"]),
        ],
        assumptions=[],
        limitations=[],
        next_steps=["Check docs"],
        self_reported_confidence="high",
    )

    stats = RunStats(run_id="test-run")

    final_ans = verify_and_construct_final_answer(
        question="What is Chroma?",
        plan=plan,
        synthesis=synthesis,
        ledger=ledger,
        stats=stats,
    )

    # The invalid finding must be dropped
    assert len(final_ans.key_findings) == 1
    assert final_ans.key_findings[0].statement == "Chroma stores embeddings."

    # Sources must be built deterministically from e1
    assert len(final_ans.sources) == 1
    assert final_ans.sources[0].url == "https://trychroma.com"
    assert "Finding 1" in final_ans.sources[0].used_in


def test_low_confidence_guardrail_applied():
    ledger = EvidenceLedger()
    # Empty ledger = no verified evidence

    plan = Plan(
        task_type=TaskType.survey,
        interpretation="Looking for nonexistent project",
        assumptions=[],
        sub_questions=[
            SubQuestion(
                id="sq1",
                question="Nonexistent project",
                rationale="Search",
                tools=[ToolName.web_search],
                search_queries=["nonexistent"],
            )
        ],
        success_criteria=[],
    )

    synthesis = SynthesisOutput(
        short_answer="Project X might exist somewhere.",
        key_findings=[],
        assumptions=[],
        limitations=[],
        next_steps=["Deploy in production"],
        self_reported_confidence="low",
    )

    stats = RunStats(run_id="test-empty")

    final_ans = verify_and_construct_final_answer(
        question="What is Project X?",
        plan=plan,
        synthesis=synthesis,
        ledger=ledger,
        stats=stats,
    )

    assert final_ans.status == "failed"
    assert final_ans.confidence.level == "low"
    assert "insufficient verified evidence" in final_ans.short_answer.lower()
