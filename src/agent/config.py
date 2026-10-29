"""Configuration, budget management, and model specifications for the research agent.

Loads settings from `.env` via pydantic-settings, provides a mutable per-run
Budget tracker with hard limits, and defines ModelSpec with pricing constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class BudgetExceeded(Exception):
    """Raised when a per-run budget counter exceeds its configured limit."""

    def __init__(self, resource: str, limit: int) -> None:
        self.resource = resource
        self.limit = limit
        super().__init__(f"Budget exceeded for '{resource}': limit is {limit}")


# ---------------------------------------------------------------------------
# Settings (immutable, loaded once from .env)
# ---------------------------------------------------------------------------


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE_CANDIDATES = [
    Path.cwd() / ".env",
    _PROJECT_ROOT / ".env",
]

class Settings(BaseSettings):
    """Application-wide settings loaded from environment / `.env` file.

    All fields have sensible defaults that match `.env.example`.
    """

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE_CANDIDATES,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider (primary) ---
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"
    LLM_API_KEY: str = ""
    SMALL_MODEL: str = "llama-3.1-8b-instant"
    LARGE_MODEL: str = "llama-3.3-70b-versatile"
    ROUTING: Literal["hybrid", "all-large"] = "hybrid"

    # --- LLM provider (fallback, all optional) ---
    FALLBACK_LLM_BASE_URL: str | None = None
    FALLBACK_LLM_API_KEY: str | None = None
    FALLBACK_LARGE_MODEL: str | None = None

    # --- Tool API keys ---
    TAVILY_API_KEY: str | None = None
    GITHUB_TOKEN: str | None = None

    # --- Budget defaults ---
    MAX_SUBQUESTIONS: int = Field(default=5, ge=1)
    MAX_SEARCH_CALLS: int = Field(default=10, ge=1)
    MAX_FETCHES: int = Field(default=12, ge=1)
    MAX_LLM_CALLS: int = Field(default=30, ge=1)
    MAX_REPLANS: int = Field(default=1, ge=0)
    RUN_TIMEOUT_S: int = Field(default=120, ge=1, description="Overall run timeout in seconds")
    PIPELINE_SOFT_DEADLINE_S: int = Field(
        default=75, ge=1, description="Soft deadline for research phase"
    )
    SYNTHESIS_TIMEOUT_S: int = Field(
        default=30, ge=1, description="Timeout for final synthesis and verification"
    )
    TOOL_TIMEOUT_S: int = Field(default=10, ge=1, description="Per-tool timeout in seconds")
    FETCH_MAX_BYTES: int = Field(default=1_500_000, ge=1024, description="Max bytes per fetch")

    # --- Performance tuning & token limits ---
    LLM_CONCURRENCY: int = Field(default=3, ge=1, description="Max concurrent LLM calls")
    SMALL_REASONING_EFFORT: str = Field(
        default="low", description="Reasoning effort for small reasoning model"
    )
    LARGE_REASONING_EFFORT: str = Field(
        default="medium", description="Reasoning effort for large reasoning model"
    )
    PAGE_EXTRACTION_TIMEOUT_S: int = Field(
        default=15, ge=1, description="Per-page extraction timeout in seconds"
    )
    MAX_TOKENS_RELEVANCE: int = Field(default=1000, ge=100)
    MAX_TOKENS_EXTRACTION: int = Field(default=1000, ge=100)
    MAX_TOKENS_COVERAGE: int = Field(default=500, ge=100)
    MAX_TOKENS_PLAN: int = Field(default=1500, ge=200)
    MAX_TOKENS_SYNTHESIS: int = Field(default=2000, ge=500)

    # --- Feature flags ---
    ENABLE_JUDGE: bool = False
    ENABLE_FAITHFULNESS_CHECK: bool = True
    CHAOS_MODE: str = "off"  # off | search_down | timeout | malformed | injection | empty
    AGENT_DEBUG: bool = False


# ---------------------------------------------------------------------------
# Budget (mutable, per-run)
# ---------------------------------------------------------------------------


class Budget:
    """Mutable per-run budget tracker.

    Instantiated from a :class:`Settings` instance at the start of each run.
    Each ``use_*()`` method increments its counter and raises
    :class:`BudgetExceeded` if the configured limit is exceeded.
    """

    def __init__(self, settings: Settings) -> None:
        # Limits (copied from settings, immutable for this run)
        self.max_search_calls: int = settings.MAX_SEARCH_CALLS
        self.max_fetches: int = settings.MAX_FETCHES
        self.max_llm_calls: int = settings.MAX_LLM_CALLS
        self.max_replans: int = settings.MAX_REPLANS

        # Counters (mutable)
        self.search_calls: int = 0
        self.fetches: int = 0
        self.llm_calls: int = 0
        self.replans: int = 0
        self.warnings: list[str] = []

    # --- Consumption helpers ---

    def use_search(self) -> None:
        """Record one search-tool invocation; raise if limit exceeded."""
        self.search_calls += 1
        if self.search_calls > self.max_search_calls:
            raise BudgetExceeded("search_calls", self.max_search_calls)

    def use_fetch(self) -> None:
        """Record one fetch-page invocation; raise if limit exceeded."""
        self.fetches += 1
        if self.fetches > self.max_fetches:
            raise BudgetExceeded("fetches", self.max_fetches)

    def use_llm(self) -> None:
        """Record one LLM call; raise if limit exceeded."""
        self.llm_calls += 1
        if self.llm_calls > self.max_llm_calls:
            raise BudgetExceeded("llm_calls", self.max_llm_calls)

    def use_replan(self) -> None:
        """Record one re-plan cycle; raise if limit exceeded."""
        self.replans += 1
        if self.replans > self.max_replans:
            raise BudgetExceeded("replans", self.max_replans)

    # --- Introspection ---

    def remaining(self) -> dict[str, int]:
        """Return a dict of remaining budget for each tracked resource."""
        return {
            "search_calls": self.max_search_calls - self.search_calls,
            "fetches": self.max_fetches - self.fetches,
            "llm_calls": self.max_llm_calls - self.llm_calls,
            "replans": self.max_replans - self.replans,
        }

    def __repr__(self) -> str:
        return (
            f"Budget(search={self.search_calls}/{self.max_search_calls}, "
            f"fetch={self.fetches}/{self.max_fetches}, "
            f"llm={self.llm_calls}/{self.max_llm_calls}, "
            f"replan={self.replans}/{self.max_replans})"
        )


# ---------------------------------------------------------------------------
# Model specification & pricing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Immutable specification for an LLM model, including pricing."""

    model_id: str
    tier: Literal["small", "large"]
    price_in: float  # USD per 1 M input tokens
    price_out: float  # USD per 1 M output tokens


# verify pricing — values sourced from provider pricing pages; update as needed
MODEL_PRICING: dict[str, ModelSpec] = {
    "llama-3.1-8b-instant": ModelSpec(
        model_id="llama-3.1-8b-instant",
        tier="small",
        price_in=0.05,
        price_out=0.08,
    ),
    "llama-3.3-70b-versatile": ModelSpec(
        model_id="llama-3.3-70b-versatile",
        tier="large",
        price_in=0.59,
        price_out=0.79,
    ),
    "gpt-4o-mini": ModelSpec(
        model_id="gpt-4o-mini",
        tier="small",
        price_in=0.15,
        price_out=0.60,
    ),
    "gpt-4o": ModelSpec(
        model_id="gpt-4o",
        tier="large",
        price_in=2.50,
        price_out=10.00,
    ),
    "openai/gpt-oss-20b": ModelSpec(
        model_id="openai/gpt-oss-20b",
        tier="small",
        price_in=0.10,
        price_out=0.15,
    ),
    "openai/gpt-oss-120b": ModelSpec(
        model_id="openai/gpt-oss-120b",
        tier="large",
        price_in=0.40,
        price_out=0.60,
    ),
    "qwen/qwen3.8-27b": ModelSpec(
        model_id="qwen/qwen3.8-27b",
        tier="small",
        price_in=0.15,
        price_out=0.20,
    ),
}
