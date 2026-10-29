"""Unit tests for evidence extraction and verbatim quote verification."""

import asyncio
from unittest.mock import patch

import pytest

from agent.config import Settings
from agent.evidence import EvidenceLedger, extract_evidence_from_page, verify_quote_verbatim


def test_quote_verification_exact_match():
    doc = (
        "Qdrant is an open-source vector database written in Rust. "
        "It provides fast vector search with payload filtering capabilities."
    )
    quote = "written in Rust. It provides fast vector search"
    verified, res = verify_quote_verbatim(quote, doc)
    assert verified is True
    assert res == quote


def test_quote_verification_whitespace_and_quote_variants():
    doc = 'Qdrant provides "fast vector search" with   multiple   spaces and line breaks.'
    quote = "Qdrant provides \u201cfast vector search\u201d with multiple spaces"
    verified, _ = verify_quote_verbatim(quote, doc)
    assert verified is True


def test_quote_verification_fabricated_quote_rejected():
    doc = "Chroma is an AI-native open-source embedding database."
    quote = "Chroma is written in C++ and outperforms Redis by 1000x."
    verified, _ = verify_quote_verbatim(quote, doc)
    assert verified is False


def test_evidence_ledger_stores_and_rejects():
    ledger = EvidenceLedger()
    doc = "Milvus is a distributed vector database designed for massive data."

    # 1. Valid quote
    ev1 = ledger.add_evidence(
        sub_question_id="sq1",
        claim="Milvus is distributed",
        quote="Milvus is a distributed vector database",
        url="https://milvus.io",
        title="Milvus",
        document_text=doc,
    )
    assert ev1 is not None
    assert ev1.id == "e1"
    assert ev1.quote_verified is True
    assert ledger.unverified_dropped_count == 0

    # 2. Fabricated quote
    ev2 = ledger.add_evidence(
        sub_question_id="sq1",
        claim="Milvus is written in Haskell",
        quote="Milvus was completely rewritten in Haskell in 2025",
        url="https://milvus.io",
        title="Milvus",
        document_text=doc,
    )
    assert ev2 is None
    assert ledger.unverified_dropped_count == 1

    # 3. Structured evidence
    ev3 = ledger.add_structured_evidence(
        sub_question_id="sq2",
        claim="Milvus GitHub Stats",
        structured_data_str="stars=28000, forks=4500",
        url="https://github.com/milvus-io/milvus",
        title="GitHub Stats",
    )
    assert ev3.id == "e2"
    assert ev3.quote_verified is True

    # Check synthesis render
    rendered = ledger.render_for_synthesis()
    assert "[e1]" in rendered
    assert "[e2]" in rendered
    assert "Milvus is a distributed vector database" in rendered


@pytest.mark.asyncio
async def test_extract_evidence_timeout_skips_page():
    ledger = EvidenceLedger()
    page_data = {
        "url": "https://example.com/slow",
        "title": "Slow Page",
        "text": "Some text",
        "full_text": "Some text",
    }

    async def slow_call_json(*args, **kwargs):
        await asyncio.sleep(1.5)

    with patch("agent.evidence.call_json", side_effect=slow_call_json):
        settings = Settings(PAGE_EXTRACTION_TIMEOUT_S=1)
        res = await extract_evidence_from_page(
            page_data=page_data,
            sub_question_id="sq1",
            sub_question_text="Slow question",
            ledger=ledger,
            settings=settings,
        )
        assert res == []
        assert ledger.all() == []
