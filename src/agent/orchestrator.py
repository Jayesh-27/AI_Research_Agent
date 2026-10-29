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
from dataclasses import dataclass, field
from typing import Any, Literal

from agent.config import Budget, BudgetExceeded, Settings
from agent.diagnostics import get_debug_collector
from agent.evidence import EvidenceLedger
from agent.executor import execute_plan, execute_sub_question
from agent.faithfulness import check_faithfulness
from agent.planner import create_plan
from agent.schemas import (
    Confidence,
    FinalAnswer,
    KeyFinding,
    Plan,
    RunStats,
    Source,
    SubQuestion,
    TaskType,
)
from agent.synthesis import synthesize_answer
from agent.tracing import Trace, set_current_trace
from agent.verify import verify_and_construct_final_answer

logger = logging.getLogger(__name__)


@dataclass
class RunState:
    """Encapsulates mutable research state across pipeline phases and deadlines."""

    question: str
    settings: Settings
    budget: Budget
    trace: Trace
    active_chaos: str = "off"
    start_time: float = field(default_factory=time.time)
    on_plan_callback: Callable[[Plan], None] | None = None
    plan: Plan | None = None
    ledger: EvidenceLedger = field(default_factory=EvidenceLedger)
    warnings: list[str] = field(default_factory=list)
    fallbacks_used: list[str] = field(default_factory=list)
    pending_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    is_partial: bool = False
    partial_reason: str | None = None
    refusal_reason: str | None = None


def _format_exc(exc: BaseException) -> str:
    """Format an exception to a non-empty descriptive string."""
    msg = str(exc).strip()
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


def _build_run_stats(state: RunState, latency_s: float | None = None) -> RunStats:
    """Construct RunStats from active trace and state without zeroing metrics."""
    if latency_s is None:
        latency_s = time.time() - state.start_time

    total_tokens_in = sum(s.metadata.get("tokens_in", 0) for s in state.trace.spans)
    total_tokens_out = sum(s.metadata.get("tokens_out", 0) for s in state.trace.spans)
    total_cost = sum(s.metadata.get("est_cost_usd", 0.0) for s in state.trace.spans)
    llm_calls_count = sum(
        1 for s in state.trace.spans if "tokens_in" in s.metadata or s.name.startswith("llm:")
    )
    total_retries = sum(s.metadata.get("retries", 0) for s in state.trace.spans)

    tool_counts: dict[str, int] = {
        "search_calls": state.budget.search_calls,
        "fetch_calls": state.budget.fetches,
    }

    clean_warnings = [str(w).strip() for w in state.warnings if str(w).strip()]
    if hasattr(state, "budget") and hasattr(state.budget, "warnings"):
        for bw in state.budget.warnings:
            bw_clean = str(bw).strip()
            if bw_clean and bw_clean not in clean_warnings:
                clean_warnings.append(bw_clean)
    fallbacks = list(state.fallbacks_used)
    for s in state.trace.spans:
        if s.name.endswith("_fallback") and s.name not in fallbacks:
            fallbacks.append(s.name)

    unverified_dropped = state.ledger.unverified_dropped_count if hasattr(state, "ledger") else 0

    return RunStats(
        run_id=state.trace.run_id,
        latency_s=round(latency_s, 2),
        llm_calls=llm_calls_count,
        tokens_in=total_tokens_in,
        tokens_out=total_tokens_out,
        est_cost_usd=round(total_cost, 4),
        tool_calls=tool_counts,
        retries=total_retries,
        fallbacks_used=fallbacks,
        warnings=clean_warnings,
        unverified_quotes_dropped=unverified_dropped,
    )


async def _cancel_pending(state: RunState) -> None:
    """Cancel outstanding background or sub-question tasks."""
    pending = [t for t in state.pending_tasks if not t.done()]
    for t in pending:
        t.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    state.pending_tasks.clear()


def _make_failed_answer(
    state_or_question: RunState | str,
    reason: str,
    run_id: str | None = None,
    latency_s: float | None = None,
    status: Literal["failed", "partial"] = "failed",
    warnings: list[str] | None = None,
) -> FinalAnswer:
    """Construct a clean, schema-compliant failed/partial FinalAnswer."""
    clean_reason = (
        reason.strip() if reason and reason.strip() else "Execution terminated prematurely."
    )
    if isinstance(state_or_question, RunState):
        state = state_or_question
        if clean_reason not in state.warnings:
            state.warnings.append(clean_reason)
        stats = _build_run_stats(state)
        question = state.question
    else:
        question = state_or_question
        clean_warnings = [
            str(w).strip() for w in (warnings or [clean_reason]) if str(w).strip()
        ]
        stats = RunStats(
            run_id=run_id or "unknown",
            latency_s=round(latency_s or 0.0, 2),
            warnings=clean_warnings,
        )

    return FinalAnswer(
        status=status,
        user_question=question,
        short_answer=clean_reason,
        key_findings=[],
        sources=[],
        confidence=Confidence(
            level="low",
            score=0.0,
            rationale=[f"Execution terminated prematurely: {clean_reason}"],
        ),
        assumptions=[],
        limitations=[clean_reason],
        next_steps=["Re-run with a different or more specific query."],
        stats=stats,
    )


def _make_partial_from_evidence(state: RunState, reason: str) -> FinalAnswer:
    """Construct a partial FinalAnswer directly from verified evidence in the ledger."""
    stats = _build_run_stats(state)
    all_ev = state.ledger.all()

    sources: list[Source] = []
    seen_urls: set[str] = set()
    for ev in all_ev:
        if ev.url not in seen_urls:
            seen_urls.add(ev.url)
            sources.append(
                Source(
                    id=f"s{len(sources) + 1}",
                    title=ev.title or ev.domain or "Source",
                    url=ev.url,
                    domain=ev.domain,
                    used_in=["Partial evidence summary"],
                )
            )

    findings: list[KeyFinding] = []
    for ev in all_ev[:5]:
        findings.append(
            KeyFinding(
                statement=ev.claim or f"Evidence gathered from {ev.domain}",
                evidence_ids=[ev.id],
            )
        )

    clean_reason = reason.strip() if reason and reason.strip() else "Research stopped early."
    short_answer = (
        f"Research partially completed ({len(all_ev)} evidence items retrieved from {len(sources)} sources). {clean_reason}"
    )

    clean_warnings = [str(w).strip() for w in state.warnings if str(w).strip()]
    if clean_reason not in clean_warnings:
        clean_warnings.append(clean_reason)
    stats.warnings = clean_warnings

    return FinalAnswer(
        status="partial",
        user_question=state.question,
        short_answer=short_answer,
        key_findings=findings,
        sources=sources,
        confidence=Confidence(
            level="low",
            score=0.3,
            rationale=[f"Partial run: {clean_reason}"],
        ),
        assumptions=list(state.plan.assumptions) if state.plan else [],
        limitations=[clean_reason],
        next_steps=["Re-run with higher timeout or more focused sub-questions."],
        stats=stats,
    )


async def _research_phase(state: RunState) -> None:
    """Core research execution phase (plan + execution + replan)."""
    collector = get_debug_collector()

    # 1. Planning Stage
    stage_t0 = time.time()
    span = state.trace.start_span("orchestrator:plan")
    plan = await create_plan(state.question, settings=state.settings, budget=state.budget)
    if len(plan.sub_questions) > 3:
        plan.sub_questions = plan.sub_questions[:3]
    state.plan = plan
    state.trace.end_span(span)
    if collector:
        collector.record_stage_timing(
            "orchestrator:plan",
            time.time() - stage_t0,
            max(0.0, float(state.settings.RUN_TIMEOUT_S) - (time.time() - state.start_time)),
        )

    if state.on_plan_callback:
        state.on_plan_callback(plan)

    # Check for safety / out-of-scope refusal
    if plan.task_type == TaskType.other and not plan.sub_questions:
        state.refusal_reason = f"Query flagged as out of scope or invalid: {plan.interpretation}"
        return

    # 2. Sub-question Execution Stage
    stage_t0 = time.time()
    span = state.trace.start_span("orchestrator:execute_plan")
    await execute_plan(
        plan.sub_questions,
        state.ledger,
        settings=state.settings,
        budget=state.budget,
        chaos_mode=state.active_chaos,
    )
    state.trace.end_span(span)
    if collector:
        collector.record_stage_timing(
            "orchestrator:execute_plan",
            time.time() - stage_t0,
            max(0.0, float(state.settings.RUN_TIMEOUT_S) - (time.time() - state.start_time)),
        )

    # 3. Coverage Check & Replan Stage (bounded to max_replans)
    stage_t0 = time.time()
    elapsed = time.time() - state.start_time
    remaining_total = float(state.settings.RUN_TIMEOUT_S) - elapsed
    remaining_soft = float(state.settings.PIPELINE_SOFT_DEADLINE_S) - elapsed

    if remaining_total < 25.0 or remaining_soft < 15.0:
        logger.info(
            "Skipping replan pass: remaining time (%.1fs total, %.1fs soft) is under threshold (25s/15s)",
            remaining_total,
            remaining_soft,
        )
        if collector:
            collector.record_stage_timing(
                "orchestrator:replan_pass_skipped",
                time.time() - stage_t0,
                max(0.0, remaining_total),
            )
    elif state.budget.replans < state.settings.MAX_REPLANS:
        gappy_questions: list[SubQuestion] = []
        for sq in plan.sub_questions:
            evs = state.ledger.get_by_sub_question(sq.id)
            if not evs:
                gappy_questions.append(sq)

        if gappy_questions and state.budget.replans < state.budget.max_replans:
            state.budget.use_replan()
            logger.info(
                "Coverage check found %d gappy sub-questions. Triggering replan pass.",
                len(gappy_questions),
            )
            span_replan = state.trace.start_span("orchestrator:replan_pass")
            sem = asyncio.Semaphore(2)
            replan_tasks = [
                execute_sub_question(
                    sq,
                    state.ledger,
                    settings=state.settings,
                    budget=state.budget,
                    semaphore=sem,
                    chaos_mode=state.active_chaos,
                )
                for sq in gappy_questions
            ]
            await asyncio.gather(*replan_tasks, return_exceptions=True)
            state.trace.end_span(span_replan)
            if collector:
                collector.record_stage_timing(
                    "orchestrator:replan_pass",
                    time.time() - stage_t0,
                    max(0.0, float(state.settings.RUN_TIMEOUT_S) - (time.time() - state.start_time)),
                )


async def _synthesize_and_verify(state: RunState) -> FinalAnswer:
    """Synthesize findings from ledger and verify citations."""
    collector = get_debug_collector()

    stage_t0 = time.time()
    span = state.trace.start_span("orchestrator:synthesize")

    plan = state.plan
    if plan is None:
        plan = Plan(
            task_type=TaskType.survey,
            interpretation=state.question,
            sub_questions=[],
            assumptions=[],
        )
        state.plan = plan

    synthesis_output = await synthesize_answer(
        state.question,
        plan,
        state.ledger,
        settings=state.settings,
        budget=state.budget,
    )
    if state.settings.ENABLE_FAITHFULNESS_CHECK and synthesis_output.key_findings:
        synthesis_output, dropped_count, notes = await check_faithfulness(
            synthesis_output,
            state.ledger,
            settings=state.settings,
            budget=state.budget,
        )
        if dropped_count > 0:
            state.warnings.append(
                f"{dropped_count} unsupported finding(s) dropped during faithfulness check."
            )
    state.trace.end_span(span)
    if collector:
        collector.record_stage_timing(
            "orchestrator:synthesize",
            time.time() - stage_t0,
            max(0.0, float(state.settings.RUN_TIMEOUT_S) - (time.time() - state.start_time)),
        )

    if state.active_chaos != "off":
        state.warnings.append(f"Chaos mode active: {state.active_chaos}")
    if state.active_chaos == "search_down":
        state.fallbacks_used.append("search_fallback_ddg")

    stats = _build_run_stats(state)

    final_answer = verify_and_construct_final_answer(
        question=state.question,
        plan=plan,
        synthesis=synthesis_output,
        ledger=state.ledger,
        stats=stats,
    )

    if state.is_partial and final_answer.status != "failed":
        final_answer.status = "partial"
        if state.partial_reason and state.partial_reason not in final_answer.limitations:
            final_answer.limitations.append(state.partial_reason)

    return final_answer


async def _run_pipeline_with_deadlines(state: RunState) -> FinalAnswer:
    """Execute research and synthesis stages respecting separate soft and synthesis deadlines."""
    # 1. Research phase (bounded by PIPELINE_SOFT_DEADLINE_S)
    research_task = asyncio.create_task(_research_phase(state))
    state.pending_tasks.add(research_task)
    try:
        await asyncio.wait_for(
            research_task,
            timeout=float(state.settings.PIPELINE_SOFT_DEADLINE_S),
        )
    except (asyncio.TimeoutError, TimeoutError):
        logger.warning(
            "Research run [%s] reached research soft deadline (%ds)",
            state.trace.run_id,
            state.settings.PIPELINE_SOFT_DEADLINE_S,
        )
        state.is_partial = True
        state.partial_reason = "Research deadline reached; answering from partial evidence."
        state.warnings.append("Research deadline reached; answering from partial evidence.")
    except BudgetExceeded as be:
        logger.warning(
            "Research run [%s] hit budget limit: %s",
            state.trace.run_id,
            be,
        )
        state.is_partial = True
        state.partial_reason = f"Resource budget reached ({be}); answering from partial evidence."
        state.warnings.append(f"Resource budget reached ({be}); answering from partial evidence.")
    except Exception as exc:
        logger.exception("Error in research phase [%s]: %s", state.trace.run_id, exc)
        err_msg = _format_exc(exc)
        state.warnings.append(err_msg)
    finally:
        await _cancel_pending(state)

    # Check for refusal / out-of-scope
    if state.refusal_reason:
        return _make_failed_answer(
            state,
            reason=state.refusal_reason,
            status="failed",
        )

    # If no evidence was gathered at all
    if not state.ledger.all():
        return _make_failed_answer(
            state,
            reason="No verified evidence gathered before the time limit.",
            status="failed",
        )

    # 2. Synthesis phase (bounded by SYNTHESIS_TIMEOUT_S)
    try:
        return await asyncio.wait_for(
            _synthesize_and_verify(state),
            timeout=float(state.settings.SYNTHESIS_TIMEOUT_S),
        )
    except (asyncio.TimeoutError, TimeoutError):
        logger.warning(
            "Synthesis timed out (%ds) for run [%s]",
            state.settings.SYNTHESIS_TIMEOUT_S,
            state.trace.run_id,
        )
        state.warnings.append(
            f"Synthesis timeout ({state.settings.SYNTHESIS_TIMEOUT_S}s) reached."
        )
        return _make_partial_from_evidence(
            state,
            reason=f"Synthesis timeout ({state.settings.SYNTHESIS_TIMEOUT_S}s) reached; returning verified evidence summary.",
        )
    except Exception as exc:
        logger.exception("Synthesis failed for run [%s]: %s", state.trace.run_id, exc)
        err_msg = _format_exc(exc)
        state.warnings.append(f"Synthesis failed: {err_msg}")
        return _make_partial_from_evidence(
            state,
            reason=f"Synthesis stage failed ({err_msg}); returning verified evidence summary.",
        )


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

    clean_q = question.strip()
    logger.info("Starting research run [%s] for query: '%s'", trace.run_id, clean_q)

    state = RunState(
        question=clean_q,
        settings=settings,
        budget=budget,
        trace=trace,
        active_chaos=active_chaos,
        start_time=start_time,
        on_plan_callback=on_plan_callback,
    )

    # Input validation
    if len(clean_q) < 5:
        return _make_failed_answer(
            state,
            "Question is too short to research.",
            status="failed",
        )

    try:
        # Wrap overall execution in RUN_TIMEOUT_S as absolute cap
        return await asyncio.wait_for(
            _run_pipeline_with_deadlines(state),
            timeout=float(settings.RUN_TIMEOUT_S),
        )
    except (asyncio.TimeoutError, TimeoutError):
        logger.warning(
            "Research run [%s] exceeded global timeout (%ds)",
            state.trace.run_id,
            settings.RUN_TIMEOUT_S,
        )
        state.warnings.append(f"Global run timeout ({settings.RUN_TIMEOUT_S}s) reached.")
        await _cancel_pending(state)
        if state.ledger.all():
            return _make_partial_from_evidence(
                state,
                reason=f"Global run timeout ({settings.RUN_TIMEOUT_S}s) reached.",
            )
        return _make_failed_answer(
            state,
            reason="No verified evidence gathered before the time limit.",
            status="failed",
        )
    except Exception as exc:
        logger.exception("Unexpected error in research pipeline [%s]: %s", trace.run_id, exc)
        await _cancel_pending(state)
        err_msg = _format_exc(exc)
        state.warnings.append(f"Internal pipeline failure: {err_msg}")
        if state.ledger.all():
            return _make_partial_from_evidence(
                state,
                reason=f"Pipeline error: {err_msg}",
            )
        return _make_failed_answer(
            state,
            reason=f"Internal pipeline failure: {err_msg}",
            status="failed",
        )
    finally:
        trace.save_to_disk()
