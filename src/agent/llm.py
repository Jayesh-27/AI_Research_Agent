"""LLM client — ModelRouter + call_json with retries, repair, fallback, and cost tracking.

Uses the OpenAI SDK with a configurable base_url (Groq-compatible).
Temperature 0 everywhere. Every call is Pydantic-validated.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Literal, TypeVar

import openai
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from agent.config import MODEL_PRICING, Budget, ModelSpec, Settings
from agent.tracing import get_current_trace

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LLMOutputError(Exception):
    """Raised when the LLM output cannot be parsed into the expected schema after repair."""

    def __init__(self, message: str, raw_output: str = "") -> None:
        self.raw_output = raw_output
        super().__init__(message)


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------


class ModelRouter:
    """Resolves model tier ('small' / 'large') to a concrete ModelSpec."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def get(self, tier: Literal["small", "large"]) -> ModelSpec:
        """Return the ModelSpec for the requested tier."""
        model_id = self._settings.SMALL_MODEL if tier == "small" else self._settings.LARGE_MODEL
        if model_id in MODEL_PRICING:
            return MODEL_PRICING[model_id]
        # Fallback: construct a spec with zero pricing (unknown model)
        return ModelSpec(model_id=model_id, tier=tier, price_in=0.0, price_out=0.0)


# ---------------------------------------------------------------------------
# Client cache (avoid re-creating per call)
# ---------------------------------------------------------------------------

_clients: dict[str, openai.AsyncOpenAI] = {}
_llm_call_locks: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}


def _get_call_lock() -> asyncio.Lock:
    """Return an asyncio.Lock bound to the current running event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.Lock()
    if loop not in _llm_call_locks:
        _llm_call_locks[loop] = asyncio.Lock()
    return _llm_call_locks[loop]


def _get_client(base_url: str, api_key: str) -> openai.AsyncOpenAI:
    """Return a cached AsyncOpenAI client for the given base URL."""
    cache_key = f"{base_url}:{api_key[:8]}"
    if cache_key not in _clients:
        _clients[cache_key] = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=1,
        )
    return _clients[cache_key]


# ---------------------------------------------------------------------------
# Core call_json function
# ---------------------------------------------------------------------------


async def call_json(
    tier: Literal["small", "large"],
    system: str,
    user: str,
    schema: type[T],
    *,
    settings: Settings,
    budget: Budget | None = None,
    temperature: float = 0,
    max_tokens: int = 2048,
    span_name: str = "llm_call",
) -> T:
    """Call the LLM and return a Pydantic-validated instance of *schema*.

    Flow:
    1. Inject JSON schema into system prompt.
    2. Call with response_format={"type": "json_object"}.
    3. Parse + validate. On failure → one repair attempt.
    4. Retry on 429/5xx with exponential backoff (max 3).
    5. On persistent failure → try fallback provider if configured.
    6. Record tokens, latency, cost on trace span.
    """
    if budget:
        budget.use_llm()

    router = ModelRouter(settings)
    spec = router.get(tier)

    # Inject schema into system prompt
    json_schema = json.dumps(schema.model_json_schema(), indent=2)
    full_system = (
        f"{system}\n\n## Output format\n"
        f"Return ONLY valid JSON matching this schema:\n```json\n{json_schema}\n```"
    )

    result = await _call_with_retry_and_fallback(
        spec=spec,
        system=full_system,
        user=user,
        schema=schema,
        settings=settings,
        budget=budget,
        temperature=temperature,
        max_tokens=max_tokens,
        span_name=span_name,
    )
    return result


async def _call_with_retry_and_fallback(
    *,
    spec: ModelSpec,
    system: str,
    user: str,
    schema: type[T],
    settings: Settings,
    budget: Budget | None,
    temperature: float,
    max_tokens: int,
    span_name: str,
) -> T:
    """Try primary provider with retries, then fallback if available."""
    try:
        return await _call_primary(
            spec=spec,
            system=system,
            user=user,
            schema=schema,
            settings=settings,
            temperature=temperature,
            max_tokens=max_tokens,
            span_name=span_name,
        )
    except (openai.RateLimitError, openai.APIStatusError, LLMOutputError) as exc:
        # Try fallback provider if configured
        if settings.FALLBACK_LLM_BASE_URL and settings.FALLBACK_LLM_API_KEY:
            logger.warning("Primary LLM failed (%s), trying fallback provider", exc)
            trace = get_current_trace()
            if trace:
                span = trace.start_span(f"{span_name}_fallback")
                span.metadata["fallback_reason"] = str(exc)
                trace.end_span(span)

            fallback_model = settings.FALLBACK_LARGE_MODEL or spec.model_id
            fallback_spec = ModelSpec(
                model_id=fallback_model, tier=spec.tier, price_in=0.0, price_out=0.0
            )
            return await _call_once(
                base_url=settings.FALLBACK_LLM_BASE_URL,
                api_key=settings.FALLBACK_LLM_API_KEY,
                spec=fallback_spec,
                system=system,
                user=user,
                schema=schema,
                temperature=temperature,
                max_tokens=max_tokens,
                span_name=f"{span_name}_fallback",
            )
        raise


@retry(
    retry=retry_if_exception_type(
        (openai.RateLimitError, openai.APIStatusError, openai.APITimeoutError)
    ),
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=30, jitter=2),
    reraise=True,
)
async def _call_primary(
    *,
    spec: ModelSpec,
    system: str,
    user: str,
    schema: type[T],
    settings: Settings,
    temperature: float,
    max_tokens: int,
    span_name: str,
) -> T:
    """Single-provider call with tenacity retries."""
    return await _call_once(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        spec=spec,
        system=system,
        user=user,
        schema=schema,
        temperature=temperature,
        max_tokens=max_tokens,
        span_name=span_name,
    )


async def _call_once(
    *,
    base_url: str,
    api_key: str,
    spec: ModelSpec,
    system: str,
    user: str,
    schema: type[T],
    temperature: float,
    max_tokens: int,
    span_name: str,
) -> T:
    """Make a single LLM call, parse, optionally repair once."""
    client = _get_client(base_url, api_key)
    trace = get_current_trace()
    span = trace.start_span(span_name, model=spec.model_id, tier=spec.tier) if trace else None

    start = time.time()
    try:
        async with _get_call_lock():
            response = await client.chat.completions.create(
                model=spec.model_id,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                timeout=30.0,
            )
            # Throttle between calls to protect free-tier rate limits
            await asyncio.sleep(0.5)

        raw = response.choices[0].message.content or "{}"
        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0

        if span:
            span.metadata["tokens_in"] = tokens_in
            span.metadata["tokens_out"] = tokens_out
            cost = (tokens_in * spec.price_in + tokens_out * spec.price_out) / 1_000_000
            span.metadata["est_cost_usd"] = cost

        # Parse and validate
        try:
            data = json.loads(raw)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as parse_err:
            logger.warning("LLM output parse failed for %s: %s", span_name, parse_err)
            # One repair attempt
            return await _repair_once(
                client=client,
                spec=spec,
                schema=schema,
                raw_output=raw,
                error=str(parse_err),
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
    except Exception as exc:
        if span:
            span.error = str(exc)
        raise
    finally:
        latency_ms = (time.time() - start) * 1000
        if span:
            span.metadata["latency_ms"] = latency_ms
            if trace:
                trace.end_span(span)


async def _repair_once(
    *,
    client: openai.AsyncOpenAI,
    spec: ModelSpec,
    schema: type[T],
    raw_output: str,
    error: str,
    system: str,
    temperature: float,
    max_tokens: int,
) -> T:
    """Send the error and previous output back, asking for corrected JSON."""
    repair_prompt = (
        f"Your previous JSON output had an error:\n"
        f"```\n{error}\n```\n\n"
        f"Previous output:\n```json\n{raw_output[:2000]}\n```\n\n"
        f"Return ONLY the corrected, valid JSON. No explanation."
    )

    try:
        response = await client.chat.completions.create(
            model=spec.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": repair_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            timeout=30.0,
        )

        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        return schema.model_validate(data)
    except (json.JSONDecodeError, ValidationError, Exception) as exc:
        raise LLMOutputError(
            f"Repair failed for {schema.__name__}: {exc}", raw_output=raw_output
        ) from exc


# ---------------------------------------------------------------------------
# FakeLLM for testing
# ---------------------------------------------------------------------------


class FakeLLM:
    """Test double that returns canned JSON responses by matching prompt keywords.

    Usage::

        fake = FakeLLM({
            "planner": {"task_type": "comparison", ...},
            "synthesize": {"short_answer": "...", ...},
        })

    Supports an ``invalid_then_valid`` mode: first call returns invalid JSON,
    second call returns valid, to test the repair path.
    """

    def __init__(
        self,
        responses: dict[str, dict[str, Any]],
        *,
        invalid_first: bool = False,
    ) -> None:
        self.responses = responses
        self.invalid_first = invalid_first
        self._call_count = 0

    def match(self, prompt: str) -> dict[str, Any]:
        """Find the first response whose key appears in the prompt."""
        for keyword, response in self.responses.items():
            if keyword.lower() in prompt.lower():
                return response
        return {}

    async def call(self, system: str, user: str) -> str:
        """Return a JSON string matching the prompt content."""
        self._call_count += 1
        combined = f"{system} {user}"
        response = self.match(combined)

        if self.invalid_first and self._call_count == 1:
            return "{ invalid json here !!!"

        return json.dumps(response)
