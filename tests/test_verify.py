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


def test_coverage_rule_distinct_domains_and_structured_api():
    ledger = EvidenceLedger()
    # sq1 has 2 evidence from 2 distinct domains -> covered
    ledger.add_evidence(
        sub_question_id="sq1",
        claim="Claim A",
        quote="Quote A",
        url="https://site-a.com/doc",
        title="Site A",
        document_text="Quote A",
    )
    ledger.add_evidence(
        sub_question_id="sq1",
        claim="Claim B",
        quote="Quote B",
        url="https://site-b.com/doc",
        title="Site B",
        document_text="Quote B",
    )

    # sq2 has 2 evidence from 1 domain (not structured) -> NOT covered
    ledger.add_evidence(
        sub_question_id="sq2",
        claim="Claim C1",
        quote="Quote C1",
        url="https://site-c.com/1",
        title="Site C 1",
        document_text="Quote C1",
    )
    ledger.add_evidence(
        sub_question_id="sq2",
        claim="Claim C2",
        quote="Quote C2",
        url="https://site-c.com/2",
        title="Site C 2",
        document_text="Quote C2",
    )

    # sq3 has 2 evidence from GitHub API (structured) -> covered
    ledger.add_structured_evidence(
        sub_question_id="sq3",
        claim="GitHub stats 1",
        structured_data_str="stars=100",
        url="https://github.com/org/repo1",
        title="Repo 1",
    )
    ledger.add_structured_evidence(
        sub_question_id="sq3",
        claim="GitHub stats 2",
        structured_data_str="stars=200",
        url="https://github.com/org/repo2",
        title="Repo 2",
    )

    plan = Plan(
        task_type=TaskType.comparison,
        interpretation="Compare",
        sub_questions=[
            SubQuestion(
                id="sq1",
                question="Q1",
                rationale="r1",
                tools=[ToolName.web_search],
                search_queries=["q1"],
            ),
            SubQuestion(
                id="sq2",
                question="Q2",
                rationale="r2",
                tools=[ToolName.web_search],
                search_queries=["q2"],
            ),
            SubQuestion(
                id="sq3",
                question="Q3",
                rationale="r3",
                tools=[ToolName.github_repo_stats],
                search_queries=["q3"],
            ),
        ],
    )

    synthesis = SynthesisOutput(
        short_answer="Findings",
        key_findings=[
            KeyFinding(statement="Finding 1", evidence_ids=["e1"]),
            KeyFinding(statement="Finding 2", evidence_ids=["e3"]),
            KeyFinding(statement="Finding 3", evidence_ids=["e5"]),
        ],
        assumptions=[],
        limitations=[],
        next_steps=[],
        self_reported_confidence="high",
    )
    stats = RunStats(run_id="test-coverage")

    final_ans = verify_and_construct_final_answer(
        question="Test",
        plan=plan,
        synthesis=synthesis,
        ledger=ledger,
        stats=stats,
    )

    # sq1 (covered), sq2 (not covered), sq3 (covered) -> 2/3 = 66.7% coverage
    assert "2/3 covered" in final_ans.confidence.rationale[1]
    # Coverage >= 0.5 so status is ok
    assert final_ans.status == "ok"
    # Coverage < 0.67 so confidence capped at medium
    assert final_ans.confidence.level == "medium"


def test_coverage_below_half_forces_partial_status():
    ledger = EvidenceLedger()
    # sq1 covered
    ledger.add_evidence(
        sub_question_id="sq1",
        claim="Claim A",
        quote="Quote A",
        url="https://site-a.com/doc",
        title="Site A",
        document_text="Quote A",
    )
    ledger.add_evidence(
        sub_question_id="sq1",
        claim="Claim B",
        quote="Quote B",
        url="https://site-b.com/doc",
        title="Site B",
        document_text="Quote B",
    )

    # sq2, sq3 have no evidence -> coverage 1/3 = 33.3% (< 0.5)
    plan = Plan(
        task_type=TaskType.comparison,
        interpretation="Compare",
        sub_questions=[
            SubQuestion(
                id="sq1",
                question="Q1",
                rationale="r1",
                tools=[ToolName.web_search],
                search_queries=["q1"],
            ),
            SubQuestion(
                id="sq2",
                question="Q2",
                rationale="r2",
                tools=[ToolName.web_search],
                search_queries=["q2"],
            ),
            SubQuestion(
                id="sq3",
                question="Q3",
                rationale="r3",
                tools=[ToolName.github_repo_stats],
                search_queries=["q3"],
            ),
        ],
    )

    synthesis = SynthesisOutput(
        short_answer="Findings",
        key_findings=[
            KeyFinding(statement="Finding 1", evidence_ids=["e1"]),
        ],
        assumptions=[],
        limitations=[],
        next_steps=[],
        self_reported_confidence="high",
    )
    stats = RunStats(run_id="test-low-cov")

    final_ans = verify_and_construct_final_answer(
        question="Test",
        plan=plan,
        synthesis=synthesis,
        ledger=ledger,
        stats=stats,
    )

    assert final_ans.status == "partial"
    assert final_ans.confidence.level == "low"  # Capped at low because < 0.34
    assert any("Coverage incomplete" in lim and "Q2" in lim and "Q3" in lim for lim in final_ans.limitations)


def test_stale_year_in_assumptions_emits_warning():
    ledger = EvidenceLedger()
    ledger.add_evidence(
        sub_question_id="sq1",
        claim="Claim A",
        quote="Qdrant vector search is fast in 2026.",
        url="https://qdrant.tech",
        title="Qdrant",
        document_text="Qdrant vector search is fast in 2026.",
    )

    plan = Plan(
        task_type=TaskType.survey,
        interpretation="Survey",
        sub_questions=[
            SubQuestion(
                id="sq1",
                question="Q1",
                rationale="r1",
                tools=[ToolName.web_search],
                search_queries=["q1"],
            )
        ],
    )

    synthesis = SynthesisOutput(
        short_answer="Findings",
        key_findings=[
            KeyFinding(statement="Qdrant is fast in 2026.", evidence_ids=["e1"]),
        ],
        assumptions=["Assessment based on ecosystem state as of late 2023"],
        limitations=[],
        next_steps=[],
        self_reported_confidence="high",
    )
    stats = RunStats(run_id="test-stale-year")

    final_ans = verify_and_construct_final_answer(
        question="Test",
        plan=plan,
        synthesis=synthesis,
        ledger=ledger,
        stats=stats,
    )

    assert any(
        "Stale date detected" in w and "2023" in w for w in final_ans.stats.warnings
    )
    assert any(
        "Stale date detected" in lim and "2023" in lim for lim in final_ans.limitations
    )

