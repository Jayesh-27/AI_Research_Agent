"""Lightweight tracing and telemetry infrastructure for the research agent.

Provides in-memory trace recording of spans (tools, LLMs, orchestration steps),
token and cost aggregation, and JSON persistence to `runs/{run_id}.json`.
"""

from __future__ import annotations

import contextvars
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """A single timed operation within a trace."""

    name: str
    start_ms: float
    end_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        if self.end_ms is None:
            return 0.0
        return self.end_ms - self.start_ms


@dataclass
class Trace:
    """A collection of spans representing a single research run."""

    run_id: str = field(
        default_factory=lambda: (
            datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        )
    )
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    spans: list[Span] = field(default_factory=list)

    def start_span(self, name: str, **metadata: Any) -> Span:
        """Create and register a new span."""
        span = Span(
            name=name,
            start_ms=time.time() * 1000,
            metadata=metadata,
        )
        self.spans.append(span)
        return span

    def end_span(self, span: Span, error: str | None = None) -> None:
        """Close a span by recording end time and optional error."""
        span.end_ms = time.time() * 1000
        span.error = error

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to a JSON-serializable dictionary."""
        return {
            "run_id": self.run_id,
            "start_time": self.start_time,
            "spans": [
                {
                    "name": s.name,
                    "start_ms": s.start_ms,
                    "end_ms": s.end_ms,
                    "duration_ms": round(s.duration_ms, 2),
                    "metadata": s.metadata,
                    "error": s.error,
                }
                for s in self.spans
            ],
        }

    def save_to_disk(self, directory: Path | str = "runs") -> Path:
        """Persist trace JSON to runs/{run_id}.json."""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / f"{self.run_id}.json"
        file_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("Trace persisted to %s", file_path)
        return file_path


# ---------------------------------------------------------------------------
# ContextVar for active trace
# ---------------------------------------------------------------------------

_current_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    "_current_trace", default=None
)


def get_current_trace() -> Trace | None:
    """Return the active trace in the current async context."""
    return _current_trace.get()


def set_current_trace(trace: Trace) -> None:
    """Set the active trace in the current async context."""
    _current_trace.set(trace)
