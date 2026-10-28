"""Orchestrator module — the main agentic research workflow.

Coordinates the end-to-end pipeline:
1. Input validation & budget initialization
2. Planning (decomposition, tool selection, success criteria)
3. Sub-question execution (search, filter, fetch, scan, extract, verify)
4. Coverage reflection & bounded replanning
5. Synthesis over verified evidence ledger only
6. Citation verification, deterministic confidence calculation, and guardrails
7. Telemetry recording & trace persistence
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from agent.config import Budget, BudgetExceeded, Settings
from agent.evidence import EvidenceLedger
from agent.executor import execute_plan, execute_sub_question
from agent.planner import create_plan
from agent.schemas import (
    Confidence,
    FinalAnswer,
    Plan,
    RunStats,
    SubQuestion,
    TaskType,
)
from agent.synthesis import synthesize_answer
from agent.tracing import Trace, set_current_trace
from agent.verify import verify_and_construct_final_answer

logger = logging.getLogger(__name__)


async def run_research(
    question: str,
    *,
    settings: Settings | None = None,
    chaos_mode: str = "off",
    on_plan_callback: Callable[[Plan], None] | None = None,
) -> FinalAnswer:
    """Execute the complete research pipeline for a user question."""
    if settings is None:
        settings = Settings()

    # Active chaos mode priority: explicit argument overrides settings
    active_chaos = chaos_mode if chaos_mode != "off" else settings.CHAOS_MODE

    trace = Trace()
    set_current_trace(trace)
    start_time = time.time()

    budget = Budget(settings)
    warnings: list[str] = []

    logger.info("Starting research run [%s] for query: '%s'", trace.run_id, question)

    # Input validation
    clean_q = question.strip()
    if len(clean_q) < 5:
        return _make_failed_answer(
            clean_q,
            "Question is too short to research.",
            trace.run_id,
            time.time() - start_time,
        )

    try:
        # Wrap overall execution in RUN_TIMEOUT_S
        return await asyncio.wait_for(
            _execute_pipeline(
                clean_q,
                settings=settings,
                budget=budget,
                trace=trace,
                active_chaos=active_chaos,
                on_plan_callback=on_plan_callback,
                start_time=start_time,
            ),
            timeout=float(settings.RUN_TIMEOUT_S),
        )
    except TimeoutError:
        logger.warning(
            "Research run [%s] exceeded global timeout (%ds)", trace.run_id, settings.RUN_TIMEOUT_S
        )
        warnings.append(f"Global run timeout ({settings.RUN_TIMEOUT_S}s) reached.")
        return _make_failed_answer(
            clean_q,
            "Research timed out before completion.",
            trace.run_id,
            time.time() - start_time,
            status="partial",
            warnings=warnings,
        )
    except BudgetExceeded as exc:
        logger.warning("Research run [%s] hit budget limit: %s", trace.run_id, exc)
        warnings.append(f"Resource budget reached: {exc}")
        return _make_failed_answer(
            clean_q,
            f"Research stopped early because resource limit was reached ({exc}).",
            trace.run_id,
            time.time() - start_time,
            status="partial",
            warnings=warnings,
        )
    except Exception as exc:
        logger.exception("Unexpected error in research pipeline [%s]: %s", trace.run_id, exc)
        return _make_failed_answer(
            clean_q,
            f"Internal pipeline failure: {exc}",
            trace.run_id,
            time.time() - start_time,
            status="failed",
            warnings=[str(exc)],
        )
    finally:
        trace.save_to_disk()


async def _execute_pipeline(
    question: str,
    *,
    settings: Settings,
    budget: Budget,
    trace: Trace,
    active_chaos: str,
    on_plan_callback: Callable[[Plan], None] | None,
    start_time: float,
) -> FinalAnswer:
    """Core sequential pipeline logic."""
    warnings: list[str] = []
    fallbacks_used: list[str] = []

    # 1. Planning Stage
    span = trace.start_span("orchestrator:plan")
    plan = await create_plan(question, settings=settings, budget=budget)
    trace.end_span(span)

    if on_plan_callback:
        on_plan_callback(plan)

    # Check for safety / out-of-scope refusal
    if plan.task_type == TaskType.other and not plan.sub_questions:
        return _make_failed_answer(
            question,
            f"Query flagged as out of scope or invalid: {plan.interpretation}",
            trace.run_id,
            time.time() - start_time,
        )

    # 2. Sub-question Execution Stage
    ledger = EvidenceLedger()
    span = trace.start_span("orchestrator:execute_plan")
    await execute_plan(
        plan.sub_questions,
        ledger,
        settings=settings,
        budget=budget,
        chaos_mode=active_chaos,
    )
    trace.end_span(span)

    # 3. Coverage Check & Replan Stage (bounded to max_replans)
    if budget.replans < settings.MAX_REPLANS:
        gappy_questions: list[SubQuestion] = []
        for sq in plan.sub_questions:
            evs = ledger.get_by_sub_question(sq.id)
            domains = {ev.domain for ev in evs if ev.domain}
            if len(evs) < 2 or len(domains) < 2:
                gappy_questions.append(sq)

        if gappy_questions and budget.replans < budget.max_replans:
            budget.use_replan()
            logger.info(
                "Coverage check found %d gappy sub-questions. Triggering replan pass.",
                len(gappy_questions),
            )
            span_replan = trace.start_span("orchestrator:replan_pass")
            sem = asyncio.Semaphore(2)
            # Re-run with refined search queries
            replan_tasks = [
                execute_sub_question(
                    sq,
                    ledger,
                    settings=settings,
                    budget=budget,
                    semaphore=sem,
                    chaos_mode=active_chaos,
                )
                for sq in gappy_questions
            ]
            await asyncio.gather(*replan_tasks, return_exceptions=True)
            trace.end_span(span_replan)

    # 4. Synthesis Stage
    span = trace.start_span("orchestrator:synthesize")
    synthesis_output = await synthesize_answer(
        question,
        plan,
        ledger,
        settings=settings,
        budget=budget,
    )
    trace.end_span(span)

    # 5. Operational Telemetry Calculation
    latency = time.time() - start_time
    total_tokens_in = sum(s.metadata.get("tokens_in", 0) for s in trace.spans)
    total_tokens_out = sum(s.metadata.get("tokens_out", 0) for s in trace.spans)
    total_cost = sum(s.metadata.get("est_cost_usd", 0.0) for s in trace.spans)
    llm_calls_count = sum(1 for s in trace.spans if "tokens_in" in s.metadata)

    tool_counts: dict[str, int] = {
        "search_calls": budget.search_calls,
        "fetch_calls": budget.fetches,
    }

    if active_chaos != "off":
        warnings.append(f"Chaos mode active: {active_chaos}")
    if active_chaos == "search_down":
        fallbacks_used.append("search_fallback_ddg")

    stats = RunStats(
        run_id=trace.run_id,
        latency_s=round(latency, 2),
        llm_calls=llm_calls_count,
        tokens_in=total_tokens_in,
        tokens_out=total_tokens_out,
        est_cost_usd=round(total_cost, 4),
        tool_calls=tool_counts,
        retries=0,
        fallbacks_used=fallbacks_used,
        warnings=warnings,
    )

    # 6. Verification, Citation Matching & Guardrails
    final_answer = verify_and_construct_final_answer(
        question=question,
        plan=plan,
        synthesis=synthesis_output,
        ledger=ledger,
        stats=stats,
    )

    return final_answer


def _make_failed_answer(
    question: str,
    reason: str,
    run_id: str,
    latency_s: float,
    status: str = "failed",
    warnings: list[str] | None = None,
) -> FinalAnswer:
    """Construct a clean, schema-compliant failed/partial FinalAnswer."""
    stats = RunStats(
        run_id=run_id,
        latency_s=round(latency_s, 2),
        warnings=warnings or [reason],
    )
    return FinalAnswer(
        status=status,  # type: ignore[arg-type]
        user_question=question,
        short_answer=reason,
        key_findings=[],
        sources=[],
        confidence=Confidence(
            level="low",
            score=0.0,
            rationale=[f"Execution terminated prematurely: {reason}"],
        ),
        assumptions=[],
        limitations=[reason],
        next_steps=["Re-run with a different or more specific query."],
        stats=stats,
    )
