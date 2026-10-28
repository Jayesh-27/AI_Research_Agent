"""Base tool infrastructure — ToolResult model and @tool_call decorator.

Every tool returns a ToolResult (never raises to the orchestrator).
The decorator provides: timeout, retry, cache, tracing, chaos hooks.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from agent.cache import DiskCacheWrapper
from agent.config import BudgetExceeded
from agent.tracing import get_current_trace

logger = logging.getLogger(__name__)

# Shared cache instance
_cache = DiskCacheWrapper()


class ToolResult(BaseModel):
    """Uniform return type for all agent tools."""

    ok: bool
    data: dict[str, Any] | list[Any] | None = None
    error: str | None = None  # "timeout" | "http_429" | "malformed" | "empty" | ...
    latency_ms: int = 0
    cached: bool = False


def tool_call(
    name: str,
    timeout_s: float = 10.0,
    retries: int = 2,
    use_cache: bool = True,
) -> Callable:
    """Decorator that wraps an async tool function with standard infrastructure.

    The decorated function receives its normal args/kwargs and must return
    a ToolResult. This decorator adds:
    - asyncio timeout
    - tenacity retry on transient errors
    - diskcache lookup/store
    - trace span recording
    - chaos mode hook (checked via 'chaos_mode' kwarg)
    - guaranteed ToolResult return (never raises)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> ToolResult:
            start = time.time()
            trace = get_current_trace()
            span = trace.start_span(f"tool:{name}") if trace else None

            # Check cache
            if use_cache:
                cache_key_data = json.dumps(
                    {
                        "args": [str(a) for a in args],
                        "kwargs": {
                            k: str(v)
                            for k, v in kwargs.items()
                            if k not in ("settings", "budget", "chaos_mode")
                        },
                    },
                    sort_keys=True,
                )
                cached = _cache.get(name, cache_key_data)
                if cached is not None:
                    latency = int((time.time() - start) * 1000)
                    if span and trace:
                        span.metadata["cached"] = True
                        trace.end_span(span)
                    return ToolResult(ok=True, data=cached, latency_ms=latency, cached=True)

            try:
                result = await asyncio.wait_for(
                    _retry_wrapper(func, retries, *args, **kwargs),
                    timeout=timeout_s,
                )

                # Store in cache on success
                if use_cache and result.ok and result.data is not None:
                    _cache.set(name, cache_key_data, result.data)

                return result

            except (asyncio.TimeoutError, TimeoutError):
                latency = int((time.time() - start) * 1000)
                logger.warning("Tool %s timed out after %.1fs", name, timeout_s)
                if span:
                    span.error = "timeout"
                    if trace:
                        trace.end_span(span)
                return ToolResult(ok=False, error="timeout", latency_ms=latency)

            except BudgetExceeded as exc:
                latency = int((time.time() - start) * 1000)
                logger.warning("Tool %s hit budget limit: %s", name, exc)
                if span:
                    span.error = "budget_exceeded"
                    if trace:
                        trace.end_span(span)
                return ToolResult(ok=False, error="budget_exceeded", latency_ms=latency)

            except Exception as exc:
                latency = int((time.time() - start) * 1000)
                logger.error("Tool %s failed: %s", name, exc)
                if span:
                    span.error = str(exc)
                    if trace:
                        trace.end_span(span)
                return ToolResult(ok=False, error=str(exc), latency_ms=latency)

            finally:
                latency = int((time.time() - start) * 1000)
                if span:
                    span.metadata["latency_ms"] = latency
                    if trace and span.end_ms is None:
                        trace.end_span(span)

        return wrapper

    return decorator


async def _retry_wrapper(func: Callable, max_retries: int, *args: Any, **kwargs: Any) -> ToolResult:
    """Execute function with retries on transient errors."""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            result = await func(*args, **kwargs)
            return result
        except (ConnectionError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < max_retries:
                wait = min(2**attempt, 10)
                logger.warning(
                    "Tool retry %d/%d after %s (waiting %.1fs)",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)
            else:
                return ToolResult(
                    ok=False,
                    error=f"retries_exhausted: {exc}",
                    latency_ms=0,
                )
    return ToolResult(ok=False, error=f"retries_exhausted: {last_error}", latency_ms=0)
