"""Unit tests for citation verification, confidence calculation, and guardrails."""

from agent.evidence import EvidenceLedger
from agent.render import deduplicate_list, render_markdown
from agent.schemas import (
    Confidence,
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
    # Coverage is exactly 2/3 with tolerance -> NOT capped at medium (score >= 0.75 yields high)
    assert final_ans.confidence.level == "high"


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
    assert final_ans.confidence.level == "medium"  # 1/3 coverage is not < 1/3 with tolerance -> capped at medium
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


def test_deduplicate_list_normalizes_and_preserves_order():
    raw = [
        "Limitation A.",
        "limitation a",
        "  Limitation B  ",
        "Limitation B.",
        "Limitation C",
    ]
    res = deduplicate_list(raw)
    assert res == ["Limitation A.", "Limitation B", "Limitation C"]


def test_render_markdown_numbering_and_clean_lists():
    ans = FinalAnswer(
        status="ok",
        user_question="Vector DB comparison",
        short_answer="Answer summary",
        key_findings=[KeyFinding(statement="Finding 1", evidence_ids=["e1"])],
        sources=[Source(id="", title="Source Title", url="https://example.com", domain="example.com")],
        confidence=Confidence(level="high", score=0.85, rationale=[]),
        assumptions=["Assumption 1", "assumption 1.", "Assumption 2"],
        limitations=[f"Limitation {i}" for i in range(1, 9)],  # 8 limitations
        next_steps=["Step 1", "Step 2", "Step 3"],
    )

    md = render_markdown(ans)

    # Numbered next steps must be sequential (1. Step 1 \n 2. Step 2 \n 3. Step 3)
    assert "1. Step 1" in md
    assert "2. Step 2" in md
    assert "3. Step 3" in md
    assert "1. Step 2" not in md

    # Source ID must not be empty (not - ****)
    assert "**[s1]**" in md
    assert "- ****" not in md

    # Limitations capped at 6 with (+2 more)
    assert "(+2 more)" in md
    assert "- Limitation 6" in md
    assert "- Limitation 7" not in md


def test_llm_step_degradation_adds_warning_and_penalty():
    ledger = EvidenceLedger()
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
    plan = Plan(
        task_type=TaskType.survey,
        interpretation="Investigate",
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
        short_answer="Valid answer",
        key_findings=[KeyFinding(statement="Finding 1", evidence_ids=["e1"])],
        assumptions=[],
        limitations=[],
        next_steps=[],
        self_reported_confidence="high",
    )
    degrade_warn = "Relevance scoring degraded: LLM call failed (RateLimitError), defaulted scores to 2."
    stats_degraded = RunStats(
        run_id="test-degraded",
        warnings=[degrade_warn],
    )
    stats_clean = RunStats(run_id="test-clean")

    ans_degraded = verify_and_construct_final_answer(
        question="Test query",
        plan=plan,
        synthesis=synthesis,
        ledger=ledger,
        stats=stats_degraded,
    )
    ans_clean = verify_and_construct_final_answer(
        question="Test query",
        plan=plan,
        synthesis=synthesis,
        ledger=ledger,
        stats=stats_clean,
    )

    # 1. Degradation warning is in limitations
    assert any("Relevance scoring degraded" in lim for lim in ans_degraded.limitations)
    # 2. Confidence penalty is applied (-0.10)
    assert ans_degraded.confidence.score < ans_clean.confidence.score
    assert any("Penalty (-0.10): LLM step degraded" in r for r in ans_degraded.confidence.rationale)


def test_confidence_cap_two_thirds_float_tolerance():
    ledger = EvidenceLedger()
    # 2 subquestions covered with 2 distinct domains each
    for sq in ["sq1", "sq2"]:
        ledger.add_evidence(
            sub_question_id=sq,
            claim=f"Claim {sq} 1",
            quote=f"Quote {sq} 1",
            url=f"https://domain-{sq}-a.gov/page",
            title="Title",
            document_text=f"Quote {sq} 1",
        )
        ledger.add_evidence(
            sub_question_id=sq,
            claim=f"Claim {sq} 2",
            quote=f"Quote {sq} 2",
            url=f"https://domain-{sq}-b.gov/page",
            title="Title",
            document_text=f"Quote {sq} 2",
        )

    # 3 subquestions total -> coverage is exactly 2/3 (0.666666...)
    plan = Plan(
        task_type=TaskType.comparison,
        interpretation="Compare",
        sub_questions=[
            SubQuestion(id="sq1", question="Q1", rationale="r1", tools=[ToolName.web_search], search_queries=["q1"]),
            SubQuestion(id="sq2", question="Q2", rationale="r2", tools=[ToolName.web_search], search_queries=["q2"]),
            SubQuestion(id="sq3", question="Q3", rationale="r3", tools=[ToolName.web_search], search_queries=["q3"]),
        ],
    )
    synthesis = SynthesisOutput(
        short_answer="Answer",
        key_findings=[
            KeyFinding(statement="F1", evidence_ids=["e1"]),
            KeyFinding(statement="F2", evidence_ids=["e3"]),
        ],
        assumptions=[],
        limitations=[],
        next_steps=[],
        self_reported_confidence="high",
    )
    stats = RunStats(run_id="test-tolerance")

    final_ans = verify_and_construct_final_answer(
        question="Test",
        plan=plan,
        synthesis=synthesis,
        ledger=ledger,
        stats=stats,
    )

    # High domain scores (.gov = 0.85) + 2/3 coverage + full verification yields score >= 0.75
    assert final_ans.confidence.score >= 0.75
    # Must NOT be capped at medium by a strict 0.67 float comparison
    assert final_ans.confidence.level == "high"
    assert not any("capped at medium" in r.lower() for r in final_ans.confidence.rationale)


def test_github_source_scoring_by_path_and_authority_factor():
    from agent.filtering import calculate_domain_score, score_github_url

    # 1. Test official org repos score high (1.0)
    assert score_github_url("https://github.com/milvus-io/milvus") == 1.0
    assert score_github_url("https://github.com/qdrant/qdrant") == 1.0
    assert score_github_url("https://github.com/weaviate/weaviate") == 1.0
    assert score_github_url("https://github.com/facebook/react") == 1.0
    assert calculate_domain_score("https://github.com/milvus-io/milvus") == 1.0

    # 2. Test awesome-* lists, curated lists, and tutorials score low (0.30)
    assert score_github_url("https://github.com/awesome-vectors/awesome-rag") == 0.30
    assert score_github_url("https://github.com/sindresorhus/awesome-python") == 0.30
    assert score_github_url("https://github.com/curated-ai/collection") == 0.30
    assert score_github_url("https://github.com/developer/my-rag-tutorial") == 0.30
    assert score_github_url("https://github.com/user/demo-project") == 0.30
    assert calculate_domain_score("https://github.com/awesome-vectors/awesome-rag") == 0.30

    # 3. Test standard repos without junk keywords score 0.70
    assert score_github_url("https://github.com/someuser/unrelated-custom-tool") == 0.70

    # 4. Demonstrate effect on Authority Factor in FinalAnswer confidence
    def make_final_answer_with_source(url: str, title: str) -> FinalAnswer:
        ledger = EvidenceLedger()
        ledger.add_evidence(
            sub_question_id="sq1",
            claim="Claim",
            quote="Verified quote text",
            url=url,
            title=title,
            document_text="Verified quote text in doc",
        )
        plan = Plan(
            task_type=TaskType.survey,
            interpretation="Test",
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
            short_answer="Answer",
            key_findings=[KeyFinding(statement="F1", evidence_ids=["e1"])],
            assumptions=[],
            limitations=[],
            next_steps=[],
            self_reported_confidence="high",
        )
        return verify_and_construct_final_answer(
            question="Test",
            plan=plan,
            synthesis=synthesis,
            ledger=ledger,
            stats=RunStats(run_id="test-auth"),
        )

    ans_official = make_final_answer_with_source(
        "https://github.com/milvus-io/milvus", "Milvus Official"
    )
    ans_awesome = make_final_answer_with_source(
        "https://github.com/awesome-vectors/awesome-rag", "Awesome RAG"
    )

    # Check authority factor rationale
    auth_official = [r for r in ans_official.confidence.rationale if "Source authority:" in r][0]
    auth_awesome = [r for r in ans_awesome.confidence.rationale if "Source authority:" in r][0]

    assert "Source authority: 1.00 (weight: 30%)" in auth_official
    assert "Source authority: 0.30 (weight: 30%)" in auth_awesome

    # Mathematical delta in confidence score is exactly 30% * (1.00 - 0.30) = 0.21
    delta = ans_official.confidence.score - ans_awesome.confidence.score
    assert abs(delta - 0.21) < 1e-4




