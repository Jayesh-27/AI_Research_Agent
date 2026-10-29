"""Unit tests for faithfulness verification and hallucination backstop (B5)."""

from unittest.mock import AsyncMock, patch

import pytest

from agent.config import Settings
from agent.evidence import EvidenceLedger
from agent.faithfulness import (
    check_faithfulness,
    filter_unsupported_findings_deterministic,
)
from agent.schemas import (
    FaithfulnessCheckResult,
    FindingVerdict,
    KeyFinding,
    SynthesisOutput,
)


def test_deterministic_backstop_drops_speculative_hedge_words():
    ledger = EvidenceLedger()
    ledger.add_evidence(
        sub_question_id="sq1",
        claim="Milvus scaling",
        quote="Milvus scales horizontally for large datasets.",
        url="https://milvus.io",
        title="Milvus",
        document_text="Milvus scales horizontally for large datasets.",
    )

    findings = [
        KeyFinding(
            statement="Milvus likely supports all RAG setups presumably by its design.",
            evidence_ids=["e1"],
        ),
        KeyFinding(
            statement="Milvus scales horizontally for large datasets.",
            evidence_ids=["e1"],
        ),
    ]

    valid, dropped = filter_unsupported_findings_deterministic(findings, ledger)
    assert len(valid) == 1
    assert valid[0].statement == "Milvus scales horizontally for large datasets."
    assert len(dropped) == 1
    assert "speculative hedging" in dropped[0]


def test_deterministic_backstop_drops_unverified_license_claims():
    ledger = EvidenceLedger()
    # Quote mentions Go, but no license
    ledger.add_evidence(
        sub_question_id="sq1",
        claim="Weaviate database",
        quote="Weaviate is an open-source vector search engine written in Go.",
        url="https://weaviate.io",
        title="Weaviate",
        document_text="Weaviate is an open-source vector search engine written in Go.",
    )
    # Quote explicitly mentions Apache-2.0
    ledger.add_evidence(
        sub_question_id="sq2",
        claim="Milvus license",
        quote="Milvus is distributed under the Apache-2.0 license.",
        url="https://milvus.io",
        title="Milvus",
        document_text="Milvus is distributed under the Apache-2.0 license.",
    )

    findings = [
        # Unsupported license assertion
        KeyFinding(
            statement="Weaviate is released under Apache-2.0.",
            evidence_ids=["e1"],
        ),
        # Supported license assertion
        KeyFinding(
            statement="Milvus is released under Apache-2.0.",
            evidence_ids=["e2"],
        ),
    ]

    valid, dropped = filter_unsupported_findings_deterministic(findings, ledger)
    assert len(valid) == 1
    assert valid[0].statement == "Milvus is released under Apache-2.0."
    assert len(dropped) == 1
    assert "unverified license claim" in dropped[0]


@pytest.mark.asyncio
async def test_llm_faithfulness_check_drops_unsupported_finding():
    ledger = EvidenceLedger()
    ledger.add_evidence(
        sub_question_id="sq1",
        claim="Vector DB",
        quote="Qdrant is written in Rust.",
        url="https://qdrant.tech",
        title="Qdrant",
        document_text="Qdrant is written in Rust.",
    )

    synthesis = SynthesisOutput(
        short_answer="Overview",
        key_findings=[
            KeyFinding(statement="Qdrant is written in Rust.", evidence_ids=["e1"]),
            KeyFinding(statement="Qdrant supports quantum indexing.", evidence_ids=["e1"]),
        ],
        assumptions=[],
        limitations=[],
        next_steps=[],
    )

    mock_check_res = FaithfulnessCheckResult(
        verdicts=[
            FindingVerdict(
                statement="Qdrant is written in Rust.",
                verdict="supported",
                reason="Verbatim quote",
            ),
            FindingVerdict(
                statement="Qdrant supports quantum indexing.",
                verdict="unsupported",
                reason="No quantum indexing in quote",
            ),
        ]
    )

    with patch("agent.faithfulness.call_json", new_callable=AsyncMock) as p_llm:
        p_llm.return_value = mock_check_res
        settings = Settings()
        updated, dropped_count, notes = await check_faithfulness(
            synthesis,
            ledger,
            settings=settings,
        )

        assert dropped_count == 1
        assert len(updated.key_findings) == 1
        assert updated.key_findings[0].statement == "Qdrant is written in Rust."
        assert any("Dropped unsupported finding" in n for n in notes)


@pytest.mark.asyncio
async def test_llm_faithfulness_check_appends_partially_supported():
    ledger = EvidenceLedger()
    ledger.add_evidence(
        sub_question_id="sq1",
        claim="Vector DB",
        quote="Qdrant is fast.",
        url="https://qdrant.tech",
        title="Qdrant",
        document_text="Qdrant is fast.",
    )

    synthesis = SynthesisOutput(
        short_answer="Overview",
        key_findings=[
            KeyFinding(statement="Qdrant is fast in most cloud benchmarks.", evidence_ids=["e1"]),
        ],
        assumptions=[],
        limitations=[],
        next_steps=[],
    )

    mock_check_res = FaithfulnessCheckResult(
        verdicts=[
            FindingVerdict(
                statement="Qdrant is fast in most cloud benchmarks.",
                verdict="partially_supported",
                reason="Quote mentions fast but not cloud benchmarks",
            )
        ]
    )

    with patch("agent.faithfulness.call_json", new_callable=AsyncMock) as p_llm:
        p_llm.return_value = mock_check_res
        settings = Settings()
        updated, dropped_count, notes = await check_faithfulness(
            synthesis,
            ledger,
            settings=settings,
        )

        assert dropped_count == 0
        assert len(updated.key_findings) == 1
        assert "(partially supported by sources)" in updated.key_findings[0].statement
