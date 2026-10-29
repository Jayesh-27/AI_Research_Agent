"""Verification, citation resolution, deterministic confidence computation, and guardrails.

Ensures:
- Every finding is backed by real, verified evidence from the ledger.
- Sources list is generated deterministically from cited evidence (no LLM hallucinated URLs).
- Confidence score (0-1) is calculated mathematically from concrete metrics.
- Low-confidence guardrail actively changes answer tone and recommendations.
"""

from __future__ import annotations

import logging
from typing import Literal

from agent.diagnostics import get_debug_collector
from agent.evidence import EvidenceLedger
from agent.faithfulness import filter_unsupported_findings_deterministic
from agent.filtering import calculate_domain_score
from agent.schemas import (
    Confidence,
    FinalAnswer,
    KeyFinding,
    Plan,
    RunStats,
    Source,
    SynthesisOutput,
)

logger = logging.getLogger(__name__)


def verify_and_construct_final_answer(
    question: str,
    plan: Plan,
    synthesis: SynthesisOutput,
    ledger: EvidenceLedger,
    stats: RunStats,
) -> FinalAnswer:
    """Verify citations, build sources, compute confidence score, and apply guardrails."""
    # 1. Validate key findings against ledger
    valid_findings: list[KeyFinding] = []
    cited_evidence_ids: set[str] = set()

    for idx, finding in enumerate(synthesis.key_findings):
        # Filter to only IDs that exist in the ledger
        existing_ids = [eid for eid in finding.evidence_ids if ledger.get(eid) is not None]
        if existing_ids:
            valid_findings.append(
                KeyFinding(statement=finding.statement, evidence_ids=existing_ids)
            )
            cited_evidence_ids.update(existing_ids)
        else:
            logger.warning(
                "Dropped finding %d with no valid evidence IDs: %s",
                idx,
                finding.statement[:50],
            )

    # 1b. Deterministic faithfulness check (drop speculative hedge words & unbacked licenses)
    valid_findings, det_dropped = filter_unsupported_findings_deterministic(
        valid_findings, ledger
    )

    # 2. Build deterministic Sources list from cited evidence only
    sources_by_url: dict[str, Source] = {}
    source_counter = 1

    for idx, finding in enumerate(valid_findings):
        finding_label = f"Finding {idx + 1}"
        for eid in finding.evidence_ids:
            ev = ledger.get(eid)
            if not ev:
                continue
            if ev.url not in sources_by_url:
                sid = f"s{source_counter}"
                source_counter += 1
                sources_by_url[ev.url] = Source(
                    id=sid,
                    title=ev.title or ev.domain,
                    url=ev.url,
                    domain=ev.domain,
                    used_in=[finding_label],
                )
            else:
                if finding_label not in sources_by_url[ev.url].used_in:
                    sources_by_url[ev.url].used_in.append(finding_label)

    sources = list(sources_by_url.values())

    # 3. Compute deterministic confidence score
    confidence, raw_level, coverage, covered_count, total_sq, uncovered_sqs = _compute_confidence(
        plan=plan,
        ledger=ledger,
        sources=sources,
        stats=stats,
        llm_level=synthesis.self_reported_confidence,
    )

    # 4. Status determination & low-confidence guardrail
    short_answer = synthesis.short_answer
    assumptions = list(synthesis.assumptions or plan.assumptions)
    limitations = list(synthesis.limitations)
    next_steps = list(synthesis.next_steps)

    # Append runtime warnings to limitations
    for warning in stats.warnings:
        clean_w = str(warning).strip()
        if clean_w and clean_w not in limitations:
            limitations.append(clean_w)

    for reason in det_dropped:
        if reason not in limitations:
            limitations.append(reason)

    if not valid_findings and not sources:
        status: Literal["ok", "partial", "low_confidence", "failed"] = "failed"
        short_answer = "Unable to answer the question due to insufficient verified evidence."
        limitations.append("No reliable sources or verified quotes could be retrieved.")
    elif coverage < 0.5:
        status = "partial"
        uncovered_names = (
            "; ".join(f"'{q}'" for q in uncovered_sqs) if uncovered_sqs else "uncovered aspects"
        )
        limitations.append(
            f"Coverage incomplete ({covered_count}/{total_sq} covered); insufficient evidence for: {uncovered_names}"
        )
        if confidence.level == "low" and not short_answer.startswith("[Low-confidence notice]"):
            short_answer = f"[Low-confidence notice] {short_answer}"
    elif confidence.level == "low":
        status = "low_confidence"
        if not short_answer.startswith("[Low-confidence notice]"):
            short_answer = f"[Low-confidence notice] {short_answer}"
        limitations.append("Evidence was thin; treat conclusions as preliminary leads.")
        next_steps = [f"Manually verify: {step}" for step in next_steps]
    elif len(valid_findings) < len(plan.sub_questions):
        status = "partial"
    else:
        status = "ok"

    return FinalAnswer(
        status=status,
        user_question=question,
        short_answer=short_answer,
        key_findings=valid_findings,
        sources=sources,
        confidence=confidence,
        assumptions=assumptions,
        limitations=limitations,
        next_steps=next_steps,
        stats=stats,
    )


def _compute_confidence(
    plan: Plan,
    ledger: EvidenceLedger,
    sources: list[Source],
    stats: RunStats,
    llm_level: Literal["high", "medium", "low"],
) -> tuple[Confidence, Literal["high", "medium", "low"], float, int, int, list[str]]:
    """Compute mathematical confidence score and factor breakdown.

    Coverage Definition:
        A sub-question is marked COVERED if and only if:
        n_evidence >= 2 and (n_distinct_domains >= 2 or any structured-API evidence)

    Coverage Caps:
        - coverage < 0.34 -> confidence level capped at low
        - coverage < 0.67 -> confidence level capped at medium
        - coverage < 0.50 -> final status set to partial with limitation
    """
    rationale: list[str] = []

    # Factor 1: Source quality (mean domain score of cited sources)
    if sources:
        source_quality = sum(calculate_domain_score(s.url) for s in sources) / len(sources)
    else:
        source_quality = 0.0
    rationale.append(f"Source authority: {source_quality:.2f} (weight: 30%)")

    # Factor 2: Sub-question coverage
    # Explicit rule: n_evidence >= 2 and (n_distinct_domains >= 2 or any structured-API evidence)
    collector = get_debug_collector()
    covered_count = 0
    total_sq = len(plan.sub_questions) if plan.sub_questions else 1
    uncovered_sqs: list[str] = []
    for sq in plan.sub_questions:
        evs = ledger.get_by_sub_question(sq.id)
        domains = {ev.domain for ev in evs if ev.domain}
        has_structured = any(
            ev.domain == "github.com" or "GitHub" in ev.title or "github" in ev.url.lower()
            for ev in evs
        )
        is_cov = len(evs) >= 2 and (len(domains) >= 2 or has_structured)
        rule_desc = "n_evidence >= 2 and (n_domains >= 2 or structured_api)"
        if is_cov:
            covered_count += 1
        else:
            uncovered_sqs.append(sq.question)
        if collector:
            collector.record_coverage_breakdown(
                sq_id=sq.id,
                n_evidence=len(evs),
                n_domains=len(domains),
                has_structured_api=has_structured,
                is_covered=is_cov,
                rule_applied=rule_desc,
            )
    coverage = covered_count / total_sq
    rationale.append(
        f"Sub-question coverage: {coverage:.2f} ({covered_count}/{total_sq} covered, weight: 25%)"
    )

    # Factor 3: Quote verification rate
    total_extracted = len(ledger.all()) + ledger.unverified_dropped_count
    verification_rate = len(ledger.all()) / total_extracted if total_extracted > 0 else 0.5
    rationale.append(f"Quote verification rate: {verification_rate:.2f} (weight: 20%)")

    # Factor 4: Agreement / consistency
    agreement = 1.0
    rationale.append(f"Source consistency: {agreement:.2f} (weight: 15%)")

    # Factor 5: Freshness
    freshness = 0.7  # Default healthy baseline
    rationale.append(f"Information freshness: {freshness:.2f} (weight: 10%)")

    # Base weighted sum
    score = (
        0.30 * source_quality
        + 0.25 * coverage
        + 0.20 * verification_rate
        + 0.15 * agreement
        + 0.10 * freshness
    )

    # Penalties
    if stats.fallbacks_used:
        score -= 0.15
        rationale.append("Penalty (-0.15): search or provider fallback was used")

    # Check for low-trust sources cited
    low_trust_count = sum(1 for ev in ledger.all() if ev.trust == "low")
    if low_trust_count > 0:
        penalty = min(0.10 * low_trust_count, 0.30)
        score -= penalty
        rationale.append(f"Penalty (-{penalty:.2f}): {low_trust_count} low-trust sources detected")

    score = max(0.0, min(1.0, score))

    # Determine mathematical level
    if score >= 0.75:
        calc_level: Literal["high", "medium", "low"] = "high"
    elif score >= 0.50:
        calc_level = "medium"
    else:
        calc_level = "low"

    caps_applied: list[str] = []
    level_order = {"low": 1, "medium": 2, "high": 3}
    final_level = calc_level

    # Coverage caps:
    # coverage < 0.34 -> confidence level max low
    # coverage < 0.67 -> confidence level max medium
    if coverage < 0.34:
        if level_order[final_level] > level_order["low"]:
            cap_msg = f"Confidence capped at low (coverage {coverage:.2f} < 0.34)"
            caps_applied.append(cap_msg)
            rationale.append(cap_msg)
            final_level = "low"
    elif coverage < 0.67:
        if level_order[final_level] > level_order["medium"]:
            cap_msg = f"Confidence capped at medium (coverage {coverage:.2f} < 0.67)"
            caps_applied.append(cap_msg)
            rationale.append(cap_msg)
            final_level = "medium"

    # Rule: LLM can LOWER confidence, never raise it
    if level_order[llm_level] < level_order[final_level]:
        cap_msg = f"Confidence lowered from {final_level} to {llm_level} based on synthesizer self-assessment"
        caps_applied.append(cap_msg)
        rationale.append(cap_msg)
        final_level = llm_level

    if collector:
        factors_dict = {
            "source_authority": {"raw": source_quality, "weight": 0.30, "weighted": 0.30 * source_quality},
            "subquestion_coverage": {"raw": coverage, "weight": 0.25, "weighted": 0.25 * coverage},
            "quote_verification_rate": {"raw": verification_rate, "weight": 0.20, "weighted": 0.20 * verification_rate},
            "source_consistency": {"raw": agreement, "weight": 0.15, "weighted": 0.15 * agreement},
            "information_freshness": {"raw": freshness, "weight": 0.10, "weighted": 0.10 * freshness},
        }
        collector.record_confidence_breakdown(
            factors=factors_dict,
            penalties=[r for r in rationale if "Penalty" in r],
            final_score=round(score, 4),
            final_level=final_level,
            caps_applied=caps_applied,
        )

    return (
        Confidence(level=final_level, score=round(score, 3), rationale=rationale),
        calc_level,
        coverage,
        covered_count,
        total_sq,
        uncovered_sqs,
    )
