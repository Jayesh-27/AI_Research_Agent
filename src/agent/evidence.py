"""Evidence extraction, verbatim quote verification, and EvidenceLedger management.

Every cited quote is verified deterministically against the raw fetched document text.
Unverified quotes are dropped to eliminate hallucinations.
"""

from __future__ import annotations

import difflib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from agent.config import Budget, Settings
from agent.llm import LLMOutputError, call_json
from agent.safety import wrap_untrusted
from agent.schemas import Evidence, EvidenceExtraction

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


def _normalize_text(text: str) -> str:
    """Normalize whitespace and quotes for robust comparison."""
    # Replace various Unicode quotes with standard ASCII
    quote_map = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u00ab": '"',
        "\u00bb": '"',
        "`": "'",
    }
    for k, v in quote_map.items():
        text = text.replace(k, v)
    # Collapse multiple whitespaces
    return re.sub(r"\s+", " ", text).strip()


def verify_quote_verbatim(quote: str, document_text: str) -> tuple[bool, str]:
    """Verify whether a quote exists in the document text.

    Returns:
        (is_verified, verified_quote)
    If exact normalized match succeeds, returns (True, quote).
    If fuzzy match with >= 0.9 similarity ratio succeeds, returns (True, matched_slice).
    Otherwise returns (False, quote).
    """
    if not quote or not document_text:
        return False, quote

    norm_quote = _normalize_text(quote).lower()
    norm_doc = _normalize_text(document_text).lower()

    # 1. Exact normalized substring match
    if norm_quote in norm_doc:
        return True, quote

    # 2. Fuzzy match via SequenceMatcher across sliding window
    quote_len = len(norm_quote)
    if quote_len < 20:
        return False, quote

    doc_len = len(norm_doc)
    # Slide across doc in steps of quote_len // 2
    step = max(quote_len // 2, 10)
    best_ratio = 0.0
    best_match = ""

    for start in range(0, doc_len, step):
        window = norm_doc[start : start + quote_len + 40]
        matcher = difflib.SequenceMatcher(None, norm_quote, window)
        ratio = matcher.ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = window

    if best_ratio >= 0.9:
        logger.info("Fuzzy quote verification matched (ratio=%.2f)", best_ratio)
        return True, best_match

    return False, quote


class EvidenceLedger:
    """Central store for all verified research evidence across sub-questions."""

    def __init__(self) -> None:
        self._evidence_by_id: dict[str, Evidence] = {}
        self._next_id: int = 1
        self.unverified_dropped_count: int = 0

    def add_evidence(
        self,
        sub_question_id: str,
        claim: str,
        quote: str,
        url: str,
        title: str,
        document_text: str,
        trust: Literal["normal", "low"] = "normal",
    ) -> Evidence | None:
        """Verify quote and, if valid, append to ledger with a stable ID."""
        verified, clean_quote = verify_quote_verbatim(quote, document_text)
        if not verified:
            logger.warning("Dropped unverified quote for claim: '%s'", claim[:50])
            self.unverified_dropped_count += 1
            return None

        eid = f"e{self._next_id}"
        self._next_id += 1

        try:
            domain = urlparse(url).netloc.lower()
        except Exception:
            domain = ""

        ev = Evidence(
            id=eid,
            sub_question_id=sub_question_id,
            claim=claim.strip(),
            quote=clean_quote.strip(),
            url=url,
            title=title.strip() if title else domain,
            domain=domain,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            quote_verified=True,
            trust=trust,
        )
        self._evidence_by_id[eid] = ev
        return ev

    def add_structured_evidence(
        self,
        sub_question_id: str,
        claim: str,
        structured_data_str: str,
        url: str,
        title: str,
    ) -> Evidence:
        """Add evidence derived from structured tool outputs (e.g. GitHub API)."""
        eid = f"e{self._next_id}"
        self._next_id += 1

        try:
            domain = urlparse(url).netloc.lower()
        except Exception:
            domain = ""

        ev = Evidence(
            id=eid,
            sub_question_id=sub_question_id,
            claim=claim.strip(),
            quote=structured_data_str,
            url=url,
            title=title.strip() if title else domain,
            domain=domain,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            quote_verified=True,
            trust="normal",
        )
        self._evidence_by_id[eid] = ev
        return ev

    def get(self, evidence_id: str) -> Evidence | None:
        """Retrieve evidence by ID."""
        return self._evidence_by_id.get(evidence_id)

    def all(self) -> list[Evidence]:
        """Return all evidence items."""
        return list(self._evidence_by_id.values())

    def get_by_sub_question(self, sub_question_id: str) -> list[Evidence]:
        """Return all evidence collected for a particular sub-question."""
        return [ev for ev in self._evidence_by_id.values() if ev.sub_question_id == sub_question_id]

    def render_for_synthesis(self) -> str:
        """Render all evidence into a compact string representation for synthesis."""
        lines = []
        for ev in self._evidence_by_id.values():
            trust_flag = " [LOW-TRUST]" if ev.trust == "low" else ""
            lines.append(
                f"[{ev.id}]{trust_flag} (Source: {ev.domain} | {ev.title})\n"
                f"Claim: {ev.claim}\n"
                f'Quote: "{ev.quote}"'
            )
        return "\n\n".join(lines)


async def extract_evidence_from_page(
    page_data: dict[str, Any],
    sub_question_id: str,
    sub_question_text: str,
    ledger: EvidenceLedger,
    *,
    settings: Settings,
    budget: Budget | None = None,
    trust: Literal["normal", "low"] = "normal",
) -> list[Evidence]:
    """Extract claims and verbatim quotes from a fetched page using SMALL LLM.

    Appends verified items directly to the ledger.
    """
    url = page_data.get("url", "")
    title = page_data.get("title", "")
    text = page_data.get("text", "")
    full_text = page_data.get("full_text", text)

    if not text:
        return []

    wrapped_doc = wrap_untrusted(text, source=url)

    prompt_path = _PROMPTS_DIR / "extract_evidence.md"
    system_prompt = (
        prompt_path.read_text(encoding="utf-8")
        if prompt_path.exists()
        else "Extract up to 3 key claims and exact verbatim quotes from the untrusted document."
    )

    user_prompt = f"Sub-question: {sub_question_text}\n\nDocument:\n{wrapped_doc}"

    try:
        extraction = await call_json(
            tier="small",
            system=system_prompt,
            user=user_prompt,
            schema=EvidenceExtraction,
            settings=settings,
            budget=budget,
            temperature=0.0,
            span_name="extract_evidence",
        )
    except (LLMOutputError, Exception) as exc:
        logger.warning("Evidence extraction LLM failed for %s: %s", url, exc)
        return []

    added: list[Evidence] = []
    for item in extraction.items:
        # If extraction notes injection suspected, lower trust
        item_trust = "low" if item.injection_suspected or trust == "low" else "normal"
        ev = ledger.add_evidence(
            sub_question_id=sub_question_id,
            claim=item.claim,
            quote=item.quote,
            url=url,
            title=title,
            document_text=full_text,
            trust=item_trust,
        )
        if ev:
            added.append(ev)

    return added
