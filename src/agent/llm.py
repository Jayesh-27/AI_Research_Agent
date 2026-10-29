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
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
)

from agent.config import MODEL_PRICING, Budget, ModelSpec, Settings
from agent.diagnostics import get_debug_collector
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
_llm_semaphores: dict[tuple[asyncio.AbstractEventLoop, int], asyncio.Semaphore] = {}


def _get_call_semaphore(concurrency: int = 3) -> asyncio.Semaphore:
    """Return an asyncio.Semaphore bound to the current running event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.Semaphore(concurrency)
    key = (loop, concurrency)
    if key not in _llm_semaphores:
        _llm_semaphores[key] = asyncio.Semaphore(concurrency)
    return _llm_semaphores[key]


def _get_call_lock() -> asyncio.Semaphore:
    """Backward compatibility alias for _get_call_semaphore."""
    return _get_call_semaphore(3)


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


def _before_sleep_callback(retry_state: RetryCallState) -> None:
    """Record retry attempts, status codes, and backoff sleep times for diagnostics."""
    telemetry = retry_state.kwargs.get("call_telemetry")
    if telemetry is not None:
        telemetry["retries"] += 1
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        status = getattr(exc, "status_code", None)
        if status is None and hasattr(exc, "response") and exc.response:
            status = getattr(exc.response, "status_code", None)
        sleep_s = retry_state.next_action.sleep if retry_state.next_action else 0.0
        telemetry["failed_attempts"].append({
            "attempt": retry_state.attempt_number,
            "status_code": status,
            "error": str(exc),
        })
        telemetry["backoff_sleeps"].append(sleep_s)


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
    call_telemetry: dict[str, Any] = {
        "retries": 0,
        "failed_attempts": [],
        "backoff_sleeps": [],
    }
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
            call_telemetry=call_telemetry,
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
                settings=settings,
                temperature=temperature,
                max_tokens=max_tokens,
                span_name=f"{span_name}_fallback",
                call_telemetry=call_telemetry,
            )
        raise


def _compute_retry_wait(retry_state: RetryCallState) -> float:
    """Respect Retry-After header if present, with exponential backoff capped at 8s."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc and hasattr(exc, "response") and exc.response is not None:
        headers = getattr(exc.response, "headers", {})
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                return min(float(retry_after), 8.0)
            except ValueError:
                pass
    # Exponential backoff capped at 8 seconds
    base = 1.0 * (2 ** (retry_state.attempt_number - 1))
    return min(base, 8.0)


@retry(
    retry=retry_if_exception_type(
        (openai.RateLimitError, openai.APIStatusError, openai.APITimeoutError)
    ),
    stop=(stop_after_attempt(3) | stop_after_delay(15.0)),
    wait=_compute_retry_wait,
    before_sleep=_before_sleep_callback,
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
    call_telemetry: dict[str, Any] | None = None,
) -> T:
    """Single-provider call with tenacity retries."""
    return await _call_once(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        spec=spec,
        system=system,
        user=user,
        schema=schema,
        settings=settings,
        temperature=temperature,
        max_tokens=max_tokens,
        span_name=span_name,
        call_telemetry=call_telemetry,
    )


async def _call_once(
    *,
    base_url: str,
    api_key: str,
    spec: ModelSpec,
    system: str,
    user: str,
    schema: type[T],
    settings: Settings | None = None,
    temperature: float,
    max_tokens: int,
    span_name: str,
    call_telemetry: dict[str, Any] | None = None,
) -> T:
    """Make a single LLM call, parse, optionally repair once."""
    client = _get_client(base_url, api_key)
    trace = get_current_trace()
    span = trace.start_span(span_name, model=spec.model_id, tier=spec.tier) if trace else None

    concurrency = settings.LLM_CONCURRENCY if settings else 3
    sem = _get_call_semaphore(concurrency)

    extra_kwargs: dict[str, Any] = {}
    if any(pat in spec.model_id.lower() for pat in ("gpt-oss", "o1", "o3")):
        if settings:
            effort = (
                settings.SMALL_REASONING_EFFORT
                if spec.tier == "small"
                else settings.LARGE_REASONING_EFFORT
            )
            if effort:
                extra_kwargs["reasoning_effort"] = effort
        else:
            extra_kwargs["reasoning_effort"] = "low" if spec.tier == "small" else "medium"

    start = time.time()
    try:
        async with sem:
            try:
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
                    **extra_kwargs,
                )
            except openai.BadRequestError as err:
                err_str = str(err).lower()
                if "reasoning_effort" in extra_kwargs and (
                    "reasoning_effort" in err_str or "unsupported" in err_str
                ):
                    logger.warning(
                        "Model %s rejected reasoning_effort (%s); retrying without it",
                        spec.model_id,
                        err,
                    )
                    extra_kwargs.pop("reasoning_effort", None)
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
                        **extra_kwargs,
                    )
                elif "max completion tokens reached" in err_str or "json_validate_failed" in err_str:
                    bumped_tokens = max(max_tokens * 2, 1500)
                    logger.warning(
                        "JSON validation truncated at %d tokens for %s; retrying with %d tokens",
                        max_tokens,
                        span_name,
                        bumped_tokens,
                    )
                    response = await client.chat.completions.create(
                        model=spec.model_id,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=temperature,
                        max_tokens=bumped_tokens,
                        response_format={"type": "json_object"},
                        timeout=30.0,
                        **extra_kwargs,
                    )
                else:
                    raise

        raw = response.choices[0].message.content or "{}"
        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0

        completion_details = getattr(response.usage, "completion_tokens_details", None)
        reasoning_tokens = None
        if completion_details is not None:
            if hasattr(completion_details, "reasoning_tokens"):
                reasoning_tokens = completion_details.reasoning_tokens
            elif isinstance(completion_details, dict):
                reasoning_tokens = completion_details.get("reasoning_tokens")
        finish_reason = response.choices[0].finish_reason if response.choices else None

        if span:
            span.metadata["tokens_in"] = tokens_in
            span.metadata["tokens_out"] = tokens_out
            cost = (tokens_in * spec.price_in + tokens_out * spec.price_out) / 1_000_000
            span.metadata["est_cost_usd"] = cost

        collector = get_debug_collector()
        if collector:
            collector.record_llm_call(
                step_name=span_name,
                model=spec.model_id,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                reasoning_tokens=reasoning_tokens,
                finish_reason=finish_reason,
                latency_ms=(time.time() - start) * 1000,
                retries=call_telemetry.get("retries", 0) if call_telemetry else 0,
                failed_attempts=call_telemetry.get("failed_attempts", []) if call_telemetry else [],
                backoff_sleeps=call_telemetry.get("backoff_sleeps", []) if call_telemetry else [],
            )

        # Parse and validate
        try:
            data = json.loads(raw)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as parse_err:
            logger.warning("LLM output parse failed for %s: %s", span_name, parse_err)
            if collector:
                collector.record_parse_failure(span_name, raw, str(parse_err))
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
                finish_reason=finish_reason,
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
    finish_reason: str | None = None,
) -> T:
    """Send the error and previous output back, asking for corrected JSON."""
    repair_max_tokens = max(max_tokens, 1000)
    trunc_note = ""
    if finish_reason == "length":
        trunc_note = "Note: Previous output was truncated by length limit. Return a more concise, valid, complete JSON object."

    repair_prompt = (
        f"Your previous JSON output had an error:\n"
        f"```\n{error}\n```\n\n"
        f"{trunc_note}\n\n"
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
            max_tokens=repair_max_tokens,
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
