"""Non-agentic single-prompt LLM baseline for comparative evaluation.

Issues a single zero-shot/few-shot call to the LARGE model without access to web search,
fetching, or external tools. Produces the same FinalAnswer schema to enable direct
side-by-side benchmarking against the agentic workflow.
"""

from __future__ import annotations

import time

from agent.config import Settings
from agent.llm import call_json
from agent.schemas import Confidence, FinalAnswer, RunStats, SynthesisOutput


async def run_baseline(question: str, *, settings: Settings | None = None) -> FinalAnswer:
    """Run a single-prompt LLM call with no tools."""
    if settings is None:
        settings = Settings()

    system_prompt = (
        "You are an AI research assistant. Answer the user research question thoroughly. "
        "Provide a concise short_answer, list key_findings with statement (set evidence_ids to []), "
        "state assumptions, limitations, and next_steps, and assess your self_reported_confidence ('high', 'medium', 'low')."
    )

    start = time.time()
    try:
        output = await call_json(
            tier="large",
            system=system_prompt,
            user=question,
            schema=SynthesisOutput,
            settings=settings,
            span_name="baseline_llm_call",
        )
        latency = time.time() - start

        # Set fake finding IDs to satisfy minimum constraint if needed
        findings = output.key_findings
        for f in findings:
            if not f.evidence_ids:
                f.evidence_ids = ["unverified_internal_knowledge"]

        return FinalAnswer(
            status="ok",
            user_question=question,
            short_answer=output.short_answer,
            key_findings=findings,
            sources=[],  # Non-agentic baseline has zero grounded sources
            confidence=Confidence(
                level=output.self_reported_confidence,
                score=0.5,
                rationale=["Single-prompt LLM baseline without external grounding or verification"],
            ),
            assumptions=output.assumptions or ["Relying strictly on model parametric knowledge."],
            limitations=["No live retrieval or tool verification; vulnerable to hallucination or stale training cutoffs."],
            next_steps=output.next_steps or ["Verify statements using primary documentation."],
            stats=RunStats(
                run_id="baseline",
                latency_s=round(latency, 2),
                llm_calls=1,
                warnings=["Non-agentic baseline run with no tools"],
            ),
        )
    except Exception as exc:
        latency = time.time() - start
        return FinalAnswer(
            status="failed",
            user_question=question,
            short_answer=f"Baseline call failed: {exc}",
            key_findings=[],
            sources=[],
            confidence=Confidence(level="low", score=0.0, rationale=[str(exc)]),
            assumptions=[],
            limitations=[str(exc)],
            next_steps=[],
            stats=RunStats(run_id="baseline_error", latency_s=round(latency, 2)),
        )

