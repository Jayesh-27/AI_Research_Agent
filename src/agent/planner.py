"""Planner module for decomposing research questions into structured execution plans.

Uses a LARGE model call with strict Pydantic validation.
Falls back to a safe default plan if the LLM output fails repeatedly.
"""

from __future__ import annotations

import logging
from pathlib import Path

from agent.config import Budget, Settings
from agent.llm import LLMOutputError, call_json
from agent.schemas import Plan, SubQuestion, TaskType, ToolName

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    path = _PROMPTS_DIR / f"{name}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


async def create_plan(
    question: str,
    *,
    settings: Settings,
    budget: Budget | None = None,
) -> Plan:
    """Generate a structured Plan for the user's research question.

    Calls the LARGE model. If parsing/validation fails, falls back to a deterministic
    default plan with web_search + fetch_page.
    """
    system_prompt = _load_prompt("planner")
    if not system_prompt:
        system_prompt = (
            "You are a research planning agent. Analyze the user's research question "
            "and produce a structured JSON Plan with task_type, interpretation, "
            "assumptions, sub_questions with tool choices, and success_criteria."
        )

    user_prompt = f"User Question: {question}"

    try:
        plan = await call_json(
            tier="large",
            system=system_prompt,
            user=user_prompt,
            schema=Plan,
            settings=settings,
            budget=budget,
            temperature=0.0,
            span_name="plan_generation",
        )
        return plan
    except (LLMOutputError, Exception) as exc:
        logger.warning("Planner LLM failed (%s), using fallback default plan", exc)
        return _create_default_plan(question)


def _create_default_plan(question: str) -> Plan:
    """Create a minimal deterministic fallback plan."""
    return Plan(
        task_type=TaskType.survey,
        interpretation=f"Investigate the question: {question}",
        assumptions=["Fallback default plan used due to planner failure."],
        sub_questions=[
            SubQuestion(
                id="sq1",
                question=question,
                rationale="Direct investigation of the main user query",
                tools=[ToolName.web_search, ToolName.fetch_page],
                search_queries=[question[:100]],
            )
        ],
        success_criteria=["Synthesize information found from web search."],
        needs_clarification=False,
    )
