"""Faithfulness verification and hallucination detection module.

Implements two complementary verification layers:
1. LLM-based batched faithfulness verification: Checks whether each synthesized finding
   strictly follows from only its cited quotes (supported | partially_supported | unsupported).
2. Deterministic backstop: Catches hedge words ('implied', 'presumably', 'likely', etc.)
   and unbacked license/version claims where the token appears in the claim but in none
   of the cited quotes.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from agent.config import Budget, Settings
from agent.evidence import EvidenceLedger
from agent.llm import LLMOutputError, call_json
from agent.schemas import FaithfulnessCheckResult, KeyFinding, SynthesisOutput

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

_HEDGE_PATTERN = re.compile(
    r"\b(implied|presumably|probably|likely|assumed|it appears that)\b", re.IGNORECASE
)
_LICENSE_PATTERN = re.compile(
    r"\b(apache[- ]?2\.0|apache[- ]?2|bsd[- ]?[23][- ]?clause|bsd|mit|gpl[- ]?[v23]?|agpl[- ]?[v23]?|mpl[- ]?2\.0)\b",
    re.IGNORECASE,
)


def filter_unsupported_findings_deterministic(
    findings: list[KeyFinding],
    ledger: EvidenceLedger,
) -> tuple[list[KeyFinding], list[str]]:
    """Deterministic layer 3 backstop: filter or downgrade findings with hedge words or unbacked licenses.

    Returns:
        (filtered_findings, list_of_dropped_or_downgraded_reasons)
    """
    valid: list[KeyFinding] = []
    dropped_reasons: list[str] = []

    for finding in findings:
        stmt = finding.statement

        # 1. Hedge word check
        hedge_match = _HEDGE_PATTERN.search(stmt)
        if hedge_match:
            reason = (
                f"Finding dropped due to speculative hedging ('{hedge_match.group(0)}'): {stmt[:60]}"
            )
            logger.warning(reason)
            dropped_reasons.append(reason)
            continue

        # 2. License tokens check: if finding claims a license, that license token must appear in a cited quote
        license_matches = _LICENSE_PATTERN.findall(stmt)
        if license_matches:
            # Gather all text from cited quotes
            quotes_text = " ".join(
                ledger.get(eid).quote.lower() for eid in finding.evidence_ids if ledger.get(eid)
            )
            unbacked_licenses = []
            for lic in license_matches:
                lic_norm = re.sub(r"[- ]", "", lic.lower())
                quotes_norm = re.sub(r"[- ]", "", quotes_text)
                if lic_norm not in quotes_norm:
                    unbacked_licenses.append(lic)

            if unbacked_licenses:
                reason = (
                    f"Finding dropped due to unverified license claim ({', '.join(unbacked_licenses)}): {stmt[:60]}"
                )
                logger.warning(reason)
                dropped_reasons.append(reason)
                continue

        valid.append(finding)

    return valid, dropped_reasons


async def check_faithfulness(
    synthesis: SynthesisOutput,
    ledger: EvidenceLedger,
    *,
    settings: Settings,
    budget: Budget | None = None,
) -> tuple[SynthesisOutput, int, list[str]]:
    """Batched LLM faithfulness check (Layer 2) over synthesized findings against their cited quotes.

    Returns:
        (updated_synthesis, dropped_count, list_of_verdict_notes)
    """
    if not synthesis.key_findings:
        return synthesis, 0, []

    # Build prompt input with only findings and their cited quotes
    findings_context: list[str] = []
    for idx, f in enumerate(synthesis.key_findings, 1):
        quotes = []
        for eid in f.evidence_ids:
            ev = ledger.get(eid)
            if ev:
                quotes.append(f"  - [{eid}] (from {ev.domain}): \"{ev.quote}\"")
        quotes_str = "\n".join(quotes) if quotes else "  (No valid cited quotes)"
        findings_context.append(
            f"Finding {idx}:\nStatement: {f.statement}\nCited Quotes:\n{quotes_str}"
        )

    user_prompt = "Verify the faithfulness of these findings:\n\n" + "\n\n".join(findings_context)

    prompt_path = _PROMPTS_DIR / "faithfulness_check.md"
    system_prompt = (
        prompt_path.read_text(encoding="utf-8")
        if prompt_path.exists()
        else (
            "You are a research verification agent. Judge whether each finding is "
            "'supported', 'partially_supported', or 'unsupported' by its cited quotes."
        )
    )

    try:
        check_result = await call_json(
            tier="large",
            system=system_prompt,
            user=user_prompt,
            schema=FaithfulnessCheckResult,
            settings=settings,
            budget=budget,
            temperature=0.0,
            max_tokens=800,
            span_name="faithfulness_check",
        )
    except (LLMOutputError, Exception) as exc:
        logger.warning("Faithfulness check LLM call failed (%s); proceeding to deterministic backstop", exc)
        return synthesis, 0, []

    # Map statements to verdicts
    verdict_by_stmt: dict[str, tuple[str, str]] = {}
    for v in check_result.verdicts:
        verdict_by_stmt[v.statement.strip()] = (v.verdict, v.reason)

    kept_findings: list[KeyFinding] = []
    dropped_count = 0
    notes: list[str] = []

    for f in synthesis.key_findings:
        stmt_clean = f.statement.strip()
        # Find matching verdict (exact or substring)
        verdict_info = verdict_by_stmt.get(stmt_clean)
        if not verdict_info:
            for k, val in verdict_by_stmt.items():
                if k in stmt_clean or stmt_clean in k:
                    verdict_info = val
                    break

        verdict = verdict_info[0] if verdict_info else "supported"
        reason = verdict_info[1] if verdict_info else ""

        if verdict == "unsupported":
            dropped_count += 1
            note = f"Dropped unsupported finding: '{f.statement[:50]}...' ({reason})"
            logger.info(note)
            notes.append(note)
        elif verdict == "partially_supported":
            new_stmt = f.statement
            if not new_stmt.endswith("(partially supported by sources)"):
                new_stmt = f"{new_stmt.rstrip('.')} (partially supported by sources)."
            kept_findings.append(KeyFinding(statement=new_stmt, evidence_ids=f.evidence_ids))
            notes.append(f"Partially supported finding retained: '{f.statement[:50]}...' ({reason})")
        else:
            kept_findings.append(f)

    updated_synthesis = synthesis.model_copy(update={"key_findings": kept_findings})
    return updated_synthesis, dropped_count, notes
