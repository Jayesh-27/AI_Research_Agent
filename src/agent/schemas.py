"""Pydantic v2 data models — single source of truth for all agent data contracts.

Every LLM output and tool output is parsed into one of these models;
parse failure is a handled error, never a crash.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskType(str, Enum):
    """Classification of the user's research question."""

    comparison = "comparison"
    list_and_summarize = "list_and_summarize"
    pros_cons = "pros_cons"
    survey = "survey"
    other = "other"


class ToolName(str, Enum):
    """Canonical names for all agent tools."""

    web_search = "web_search"
    fetch_page = "fetch_page"
    github_repo_stats = "github_repo_stats"


# ---------------------------------------------------------------------------
# Planner models
# ---------------------------------------------------------------------------


class SubQuestion(BaseModel):
    """A decomposed sub-question produced by the planner."""

    id: str = Field(description="Stable id, e.g. 'sq1'")
    question: str
    rationale: str = Field(description="Why this sub-question matters")
    tools: list[ToolName] = Field(description="Which tools to use")
    search_queries: list[str] = Field(
        default_factory=list, max_length=3, description="Up to 3 search queries"
    )

    @field_validator("search_queries", mode="before")
    @classmethod
    def _clean_search_queries(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            return []
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in v:
            if not isinstance(item, str):
                continue
            s = item.strip()
            if not s:
                continue
            s_lower = s.lower()
            if s_lower not in seen:
                seen.add(s_lower)
                cleaned.append(s)
            if len(cleaned) == 3:
                break
        return cleaned


class Plan(BaseModel):
    """Structured research plan produced by the planner LLM."""

    task_type: TaskType
    interpretation: str = Field(description="How the agent understood the question")
    assumptions: list[str] = Field(default_factory=list)
    sub_questions: list[SubQuestion] = Field(max_length=5)
    success_criteria: list[str] = Field(default_factory=list)
    needs_clarification: bool = False

    @field_validator("sub_questions", mode="before")
    @classmethod
    def _clean_sub_questions(cls, v: Any) -> list[Any]:
        if not isinstance(v, list):
            return []
        return v[:5]

    @field_validator("assumptions", "success_criteria", mode="before")
    @classmethod
    def _clean_string_lists(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            return []
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in v:
            if not isinstance(item, str):
                continue
            s = item.strip()
            if not s:
                continue
            s_lower = s.lower()
            if s_lower not in seen:
                seen.add(s_lower)
                cleaned.append(s)
            if len(cleaned) == 10:
                break
        return cleaned


# ---------------------------------------------------------------------------
# Search & evidence models
# ---------------------------------------------------------------------------


class SearchHit(BaseModel):
    """A single search result from web_search."""

    url: str
    title: str = ""
    snippet: str = ""
    published: str | None = None
    provider: Literal["tavily", "ddg"] = "tavily"
    relevance: int | None = Field(default=None, ge=0, le=3, description="0-3 score from filter")
    domain_score: float | None = Field(default=None, ge=0, le=1, description="Deterministic 0-1")


class Evidence(BaseModel):
    """A single piece of verified evidence extracted from a fetched page."""

    id: str = Field(description="Stable id, e.g. 'e1'")
    sub_question_id: str
    claim: str = Field(description="Short paraphrase of what the quote establishes")
    quote: str = Field(description="VERBATIM text from the fetched page")
    url: str
    title: str = ""
    domain: str = ""
    fetched_at: str = ""
    quote_verified: bool = False
    trust: Literal["normal", "low"] = "normal"


# ---------------------------------------------------------------------------
# Synthesis models
# ---------------------------------------------------------------------------


class KeyFinding(BaseModel):
    """A key finding in the final answer, backed by evidence."""

    statement: str
    evidence_ids: list[str] = Field(min_length=1)


class Source(BaseModel):
    """A source used in the final answer."""

    id: str
    title: str
    url: str
    domain: str = ""
    used_in: list[str] = Field(default_factory=list, description="Finding indices that cite this")


class Confidence(BaseModel):
    """Deterministically computed confidence for the final answer."""

    level: Literal["high", "medium", "low"]
    score: float = Field(ge=0, le=1)
    rationale: list[str] = Field(default_factory=list)


class RunStats(BaseModel):
    """Operational statistics for a single research run."""

    run_id: str
    latency_s: float = 0.0
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    est_cost_usd: float = 0.0
    tool_calls: dict[str, int] = Field(default_factory=dict)
    retries: int = 0
    fallbacks_used: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FinalAnswer(BaseModel):
    """The complete structured answer returned to the user.

    Contains every field the assignment brief requires: user question,
    short answer, key findings, sources, confidence, limitations/assumptions,
    and suggested next steps.
    """

    status: Literal["ok", "partial", "low_confidence", "failed"]
    user_question: str
    short_answer: str = ""
    key_findings: list[KeyFinding] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    confidence: Confidence = Field(
        default_factory=lambda: Confidence(level="low", score=0.0, rationale=[])
    )
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    stats: RunStats | None = None


# ---------------------------------------------------------------------------
# Internal / utility models (used between pipeline stages)
# ---------------------------------------------------------------------------


class RelevanceScore(BaseModel):
    """Relevance score for a single search hit."""

    url: str
    score: int = Field(ge=0, le=3)


class RelevanceBatch(BaseModel):
    """Batch of relevance scores returned by the LLM."""

    scores: list[RelevanceScore] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    """A single claim-quote pair extracted by the LLM."""

    claim: str
    quote: str
    injection_suspected: bool = False


class EvidenceExtraction(BaseModel):
    """LLM output from evidence extraction."""

    items: list[EvidenceItem] = Field(default_factory=list)


class CoverageGap(BaseModel):
    """A sub-question with insufficient evidence."""

    sub_question_id: str
    suggested_queries: list[str] = Field(default_factory=list)

    @field_validator("suggested_queries", mode="before")
    @classmethod
    def _clean_gap_queries(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            return []
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in v:
            if not isinstance(item, str):
                continue
            s = item.strip()
            if not s:
                continue
            s_lower = s.lower()
            if s_lower not in seen:
                seen.add(s_lower)
                cleaned.append(s)
            if len(cleaned) == 3:
                break
        return cleaned


class CoverageResult(BaseModel):
    """Result of the coverage check."""

    gaps: list[CoverageGap] = Field(default_factory=list)
    all_covered: bool = True


class SynthesisOutput(BaseModel):
    """LLM output from the synthesis step."""

    short_answer: str
    key_findings: list[KeyFinding] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    self_reported_confidence: Literal["high", "medium", "low"] = "medium"

    @field_validator("key_findings", mode="before")
    @classmethod
    def _clean_key_findings(cls, v: Any) -> list[Any]:
        if not isinstance(v, list):
            return []
        cleaned = []
        for item in v:
            if isinstance(item, dict):
                eids = item.get("evidence_ids")
                if isinstance(eids, list):
                    item["evidence_ids"] = [str(e).strip() for e in eids if str(e).strip()]
                # Drop findings without evidence IDs rather than crashing validation
                if not item.get("evidence_ids"):
                    continue
            cleaned.append(item)
        return cleaned

    @field_validator("assumptions", "limitations", "next_steps", mode="before")
    @classmethod
    def _clean_synthesis_lists(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            return []
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in v:
            if not isinstance(item, str):
                continue
            s = item.strip()
            if not s:
                continue
            s_lower = s.lower()
            if s_lower not in seen:
                seen.add(s_lower)
                cleaned.append(s)
        return cleaned


class QueryRewrite(BaseModel):
    """Rewritten search queries from the LLM."""

    queries: list[str] = Field(default_factory=list)

    @field_validator("queries", mode="before")
    @classmethod
    def _clean_queries(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            return []
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in v:
            if not isinstance(item, str):
                continue
            s = item.strip()
            if not s:
                continue
            s_lower = s.lower()
            if s_lower not in seen:
                seen.add(s_lower)
                cleaned.append(s)
            if len(cleaned) == 5:
                break
        return cleaned


class ScanResult(BaseModel):
    """Result of the prompt-injection scanner."""

    score: float = Field(ge=0, le=1, description="Injection likelihood 0-1")
    matches: list[str] = Field(default_factory=list)
    flagged: bool = False


class InjectionCheckResult(BaseModel):
    """LLM classifier result for prompt injection."""

    is_injection: bool = False
    confidence: float = 0.0
    reasons: list[str] = Field(default_factory=list)


class JudgeResult(BaseModel):
    """LLM-as-judge evaluation result."""

    faithfulness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    rationale: str = ""
