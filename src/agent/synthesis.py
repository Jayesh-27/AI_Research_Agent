"""Synthesis module — synthesizes structured final research answers.

Key principles:
1. The synthesizer sees ONLY the verified evidence ledger and plan context, NEVER raw web pages.
2. Every key finding MUST cite one or more evidence IDs.
3. Sources and URLs are generated deterministically by code from cited IDs, never invented by the LLM.
"""

from __future__ import annotations

import logging
from pathlib import Path

from agent.config import Budget, Settings
from agent.evidence import EvidenceLedger
from agent.llm import LLMOutputError, call_json
from agent.schemas import Plan, SynthesisOutput

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


async def synthesize_answer(
    question: str,
    plan: Plan,
    ledger: EvidenceLedger,
    *,
    settings: Settings,
    budget: Budget | None = None,
) -> SynthesisOutput:
    """Produce structured findings and answer using the LARGE model over the evidence ledger."""
    ledger_text = ledger.render_for_synthesis()
    if not ledger_text:
        ledger_text = "NO VERIFIED EVIDENCE FOUND."

    prompt_path = _PROMPTS_DIR / "synthesize.md"
    system_prompt = (
        prompt_path.read_text(encoding="utf-8")
        if prompt_path.exists()
        else (
            "You are a research synthesis agent. Synthesize a structured answer based ONLY "
            "on the provided verified evidence ledger. Every key finding must cite evidence IDs."
        )
    )

    user_prompt = (
        f"User question: {question}\n\n"
        f"Plan Context:\n"
        f"- Interpretation: {plan.interpretation}\n"
        f"- Assumptions: {plan.assumptions}\n"
        f"- Success Criteria: {plan.success_criteria}\n\n"
        f"Evidence Ledger:\n{ledger_text}"
    )

    try:
        output = await call_json(
            tier="large",
            system=system_prompt,
            user=user_prompt,
            schema=SynthesisOutput,
            settings=settings,
            budget=budget,
            temperature=0.0,
            span_name="synthesis",
        )
        return output
    except (LLMOutputError, Exception) as exc:
        logger.error("Synthesis LLM failed: %s", exc)
        return SynthesisOutput(
            short_answer="Synthesis could not be completed due to an error.",
            key_findings=[],
            assumptions=plan.assumptions,
            limitations=[f"Synthesis failure: {exc}"],
            next_steps=["Retry the query with revised parameters."],
            self_reported_confidence="low",
        )

