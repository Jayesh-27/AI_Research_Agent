"""FastAPI application exposing the research agent as an HTTP API service.

Endpoints:
- `POST /research`: Submit a research question, returns FinalAnswer JSON
- `GET /runs/{run_id}`: Retrieve telemetry trace JSON for a specific run
- `GET /health`: System health and readiness check
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from agent.config import Settings
from agent.orchestrator import run_research
from agent.schemas import FinalAnswer

app = FastAPI(
    title="Predixion AI Research Agent Service",
    description="Production-oriented agentic research engine with citations, failure handling, and observability.",
    version="0.1.0",
)

_RUNS_DIR = Path("runs")


class ResearchRequest(BaseModel):
    """Payload for submitting a research query."""

    question: str = Field(..., min_length=5, max_length=1000, description="The research question")
    chaos_mode: Literal["off", "search_down", "timeout", "malformed", "empty", "injection"] = "off"


@app.get("/health", tags=["Monitoring"])
async def health_check() -> dict[str, str]:
    """Basic health check and configuration verification."""
    settings = Settings()
    return {
        "status": "healthy",
        "llm_base_url": settings.LLM_BASE_URL,
        "small_model": settings.SMALL_MODEL,
        "large_model": settings.LARGE_MODEL,
        "tavily_configured": "true" if bool(settings.TAVILY_API_KEY) else "false (using ddg)",
    }


@app.post("/research", response_model=FinalAnswer, tags=["Research"])
async def create_research(request: ResearchRequest) -> FinalAnswer:
    """Execute end-to-end research for the submitted question."""
    try:
        final_answer = await run_research(
            question=request.question,
            chaos_mode=request.chaos_mode,
        )
        return final_answer
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Research processing error: {exc}",
        ) from exc


@app.get("/runs/{run_id}", tags=["Observability"])
async def get_run_trace(run_id: str) -> dict:
    """Retrieve full execution telemetry trace by run ID."""
    trace_file = _RUNS_DIR / f"{run_id}.json"
    if not trace_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run ID '{run_id}' not found.",
        )
    try:
        data = json.loads(trace_file.read_text(encoding="utf-8"))
        return data
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read trace file: {exc}",
        ) from exc
