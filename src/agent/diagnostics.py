"""Diagnostics and debug tracing for the research agent.

Collects comprehensive telemetry across configuration, model availability,
LLM calls, sub-question execution, coverage, GitHub tool decisions,
confidence calculation, and stage timings.

Prints readable Rich tables in terminal and saves JSON to `runs/<run_id>.debug.json`.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent.config import Settings

logger = logging.getLogger(__name__)


def check_model_availability(settings: Settings) -> dict[str, Any]:
    """Call GET {LLM_BASE_URL}/models and verify availability of SMALL and LARGE models."""
    base_url = settings.LLM_BASE_URL.rstrip("/")
    url = f"{base_url}/models"
    headers: dict[str, str] = {}
    if settings.LLM_API_KEY:
        headers["Authorization"] = f"Bearer {settings.LLM_API_KEY}"

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code != 200:
                return {
                    "ok": False,
                    "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                    "models": [],
                    "small_model_available": False,
                    "large_model_available": False,
                }
            data = resp.json()
            model_items = data.get("data", [])
            model_ids = [m.get("id") for m in model_items if isinstance(m, dict) and "id" in m]
            return {
                "ok": True,
                "error": None,
                "models": model_ids,
                "small_model_available": settings.SMALL_MODEL in model_ids,
                "large_model_available": settings.LARGE_MODEL in model_ids,
            }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "models": [],
            "small_model_available": False,
            "large_model_available": False,
        }


class DebugCollector:
    """In-memory collector for fine-grained debug telemetry."""

    def __init__(self, run_id: str, settings: Settings) -> None:
        self.run_id = run_id
        self.settings = settings
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        self.model_availability: dict[str, Any] = {}
        self.llm_calls: list[dict[str, Any]] = []
        self.parse_failures: list[dict[str, Any]] = []
        self.sub_questions: dict[str, dict[str, Any]] = {}
        self.coverage_breakdown: list[dict[str, Any]] = []
        self.github_diagnostics: list[dict[str, Any]] = []
        self.confidence_breakdown: dict[str, Any] = {}
        self.stage_timings: list[dict[str, Any]] = []

    def get_config_snapshot(self) -> dict[str, Any]:
        """Return config snapshot with secrets redacted."""
        return {
            "today_date": self.today_date,
            "python_version": sys.version.split()[0],
            "llm_base_url": self.settings.LLM_BASE_URL,
            "llm_api_key_set": bool(self.settings.LLM_API_KEY),
            "small_model": self.settings.SMALL_MODEL,
            "large_model": self.settings.LARGE_MODEL,
            "fallback_base_url": self.settings.FALLBACK_LLM_BASE_URL,
            "fallback_large_model": self.settings.FALLBACK_LARGE_MODEL,
            "tavily_api_key_set": bool(self.settings.TAVILY_API_KEY),
            "github_token_set": bool(self.settings.GITHUB_TOKEN),
            "max_subquestions": self.settings.MAX_SUBQUESTIONS,
            "max_search_calls": self.settings.MAX_SEARCH_CALLS,
            "max_fetches": self.settings.MAX_FETCHES,
            "max_llm_calls": self.settings.MAX_LLM_CALLS,
            "max_replans": self.settings.MAX_REPLANS,
            "run_timeout_s": self.settings.RUN_TIMEOUT_S,
            "tool_timeout_s": self.settings.TOOL_TIMEOUT_S,
            "fetch_max_bytes": self.settings.FETCH_MAX_BYTES,
            "enable_judge": self.settings.ENABLE_JUDGE,
            "chaos_mode": self.settings.CHAOS_MODE,
        }

    def record_stage_timing(
        self,
        stage_name: str,
        duration_s: float,
        remaining_budget_s: float,
    ) -> None:
        self.stage_timings.append({
            "stage": stage_name,
            "duration_s": round(duration_s, 3),
            "remaining_budget_s": round(remaining_budget_s, 3),
        })

    def record_llm_call(
        self,
        step_name: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        reasoning_tokens: int | None,
        finish_reason: str | None,
        latency_ms: float,
        retries: int,
        failed_attempts: list[dict[str, Any]],
        backoff_sleeps: list[float],
    ) -> None:
        self.llm_calls.append({
            "step_name": step_name,
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "reasoning_tokens": reasoning_tokens,
            "finish_reason": finish_reason or "stop",
            "latency_ms": round(latency_ms, 2),
            "retries": retries,
            "failed_attempts": failed_attempts,
            "backoff_sleeps": [round(s, 2) for s in backoff_sleeps],
        })

    def record_parse_failure(self, step_name: str, raw_output: str, error: str) -> None:
        self.parse_failures.append({
            "step_name": step_name,
            "raw_output": raw_output[:1500],
            "error": error,
        })

    def record_sub_question_search(
        self,
        sq_id: str,
        question: str,
        queries_run: list[str],
        hits_returned: int,
        hits_kept: list[dict[str, Any]],
    ) -> None:
        if sq_id not in self.sub_questions:
            self.sub_questions[sq_id] = {
                "sq_id": sq_id,
                "question": question,
                "queries_run": [],
                "hits_returned": 0,
                "hits_kept": [],
                "pages_fetched": [],
                "evidence_count": 0,
                "distinct_domains": [],
                "evidence_items": [],
            }
        sq = self.sub_questions[sq_id]
        sq["queries_run"].extend(queries_run)
        sq["hits_returned"] += hits_returned
        sq["hits_kept"].extend(hits_kept)

    def record_page_fetch(
        self,
        sq_id: str,
        url: str,
        ok: bool,
        error: str | None = None,
    ) -> None:
        if sq_id in self.sub_questions:
            self.sub_questions[sq_id]["pages_fetched"].append({
                "url": url,
                "ok": ok,
                "error": error,
            })

    def finalize_sub_question_evidence(
        self,
        sq_id: str,
        evidence_items: list[dict[str, Any]],
    ) -> None:
        if sq_id in self.sub_questions:
            sq = self.sub_questions[sq_id]
            sq["evidence_items"] = evidence_items
            sq["evidence_count"] = len(evidence_items)
            domains = {item.get("domain", "") for item in evidence_items if item.get("domain")}
            sq["distinct_domains"] = sorted(domains)

    def record_coverage_breakdown(
        self,
        sq_id: str,
        n_evidence: int,
        n_domains: int,
        has_structured_api: bool,
        is_covered: bool,
        rule_applied: str,
    ) -> None:
        self.coverage_breakdown.append({
            "sub_question_id": sq_id,
            "n_evidence": n_evidence,
            "n_domains": n_domains,
            "has_structured_api": has_structured_api,
            "is_covered": is_covered,
            "rule_applied": rule_applied,
        })

    def record_github_decision(
        self,
        candidates: list[str],
        chosen: str | None,
        rejected: list[dict[str, str]],
    ) -> None:
        self.github_diagnostics.append({
            "candidates": candidates,
            "chosen": chosen,
            "rejected": rejected,
        })

    def record_confidence_breakdown(
        self,
        factors: dict[str, Any],
        penalties: list[str],
        final_score: float,
        final_level: str,
        caps_applied: list[str],
    ) -> None:
        self.confidence_breakdown = {
            "factors": factors,
            "penalties": penalties,
            "final_score": round(final_score, 4),
            "final_level": final_level,
            "caps_applied": caps_applied,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete debug telemetry."""
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "config": self.get_config_snapshot(),
            "model_availability": self.model_availability,
            "stage_timings": self.stage_timings,
            "llm_calls": self.llm_calls,
            "parse_failures": self.parse_failures,
            "sub_questions": list(self.sub_questions.values()),
            "coverage_breakdown": self.coverage_breakdown,
            "github_diagnostics": self.github_diagnostics,
            "confidence_breakdown": self.confidence_breakdown,
        }

    def save_to_disk(self, directory: Path | str = "runs") -> Path:
        """Persist to runs/<run_id>.debug.json."""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{self.run_id}.debug.json"
        out_file.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("Debug diagnostics saved to %s", out_file)
        return out_file

    def print_rich_tables(self, console: Console | None = None) -> None:
        """Render all debug diagnostics in clean rich tables."""
        c = console or Console(legacy_windows=False)

        # 1. Config & Models
        c.print("\n")
        cfg_table = Table(title=f"Debug: Config Snapshot [{self.run_id}]", border_style="cyan")
        cfg_table.add_column("Setting", style="bold")
        cfg_table.add_column("Value", style="white")
        for k, v in self.get_config_snapshot().items():
            cfg_table.add_row(k, str(v))
        c.print(cfg_table)

        # 2. Model Availability
        if self.model_availability:
            avail_table = Table(title="Debug: Model Availability", border_style="green")
            avail_table.add_column("Model Tier", style="bold")
            avail_table.add_column("Configured ID", style="white")
            avail_table.add_column("Available on Provider", style="bold")
            avail_table.add_row(
                "SMALL",
                self.settings.SMALL_MODEL,
                "[green]YES[/green]" if self.model_availability.get("small_model_available") else "[red]NO[/red]",
            )
            avail_table.add_row(
                "LARGE",
                self.settings.LARGE_MODEL,
                "[green]YES[/green]" if self.model_availability.get("large_model_available") else "[red]NO[/red]",
            )
            c.print(avail_table)

        # 3. Stage Timings & Budget
        if self.stage_timings:
            time_table = Table(title="Debug: Stage Timings & Remaining Budget", border_style="yellow")
            time_table.add_column("Stage", style="bold cyan")
            time_table.add_column("Duration (s)", style="green", justify="right")
            time_table.add_column("Remaining Budget (s)", style="magenta", justify="right")
            for st in self.stage_timings:
                time_table.add_row(st["stage"], f"{st['duration_s']:.3f}s", f"{st['remaining_budget_s']:.3f}s")
            c.print(time_table)

        # 4. LLM Calls
        if self.llm_calls:
            llm_table = Table(title="Debug: Per-LLM-Call Telemetry", border_style="blue")
            llm_table.add_column("Step", style="bold yellow")
            llm_table.add_column("Model", style="dim")
            llm_table.add_column("In/Out Tokens", justify="right")
            llm_table.add_column("Reasoning", justify="right", style="cyan")
            llm_table.add_column("Finish", style="dim")
            llm_table.add_column("Latency (ms)", justify="right")
            llm_table.add_column("Retries", justify="right")
            llm_table.add_column("Backoff Sleeps", style="magenta")
            for call in self.llm_calls:
                reasoning = str(call["reasoning_tokens"]) if call["reasoning_tokens"] is not None else "-"
                sleeps_str = ", ".join(f"{s:.1f}s" for s in call["backoff_sleeps"]) or "none"
                llm_table.add_row(
                    call["step_name"],
                    call["model"],
                    f"{call['tokens_in']} / {call['tokens_out']}",
                    reasoning,
                    call["finish_reason"],
                    f"{call['latency_ms']:.1f}",
                    str(call["retries"]),
                    sleeps_str,
                )
            c.print(llm_table)

        # 5. Parse Failures
        if self.parse_failures:
            c.print(Panel(
                "\n\n".join(
                    f"[bold red]Step: {f['step_name']}[/bold red]\n"
                    f"Error: {f['error']}\n"
                    f"Raw (truncated): {f['raw_output'][:300]}..."
                    for f in self.parse_failures
                ),
                title="Debug: Raw LLM Output on Parse Failure",
                border_style="red",
            ))

        # 6. Sub-Questions
        if self.sub_questions:
            sq_table = Table(title="Debug: Sub-Question Execution", border_style="magenta")
            sq_table.add_column("ID", style="bold yellow")
            sq_table.add_column("Queries Run", style="dim")
            sq_table.add_column("Hits (Tot/Kept)", justify="right")
            sq_table.add_column("Pages Fetched (Ok/Err)", justify="right")
            sq_table.add_column("Evidence", justify="right")
            sq_table.add_column("Domains", style="green")
            for sq in self.sub_questions.values():
                ok_pages = sum(1 for p in sq["pages_fetched"] if p["ok"])
                err_pages = sum(1 for p in sq["pages_fetched"] if not p["ok"])
                sq_table.add_row(
                    sq["sq_id"],
                    " | ".join(sq["queries_run"][:2]),
                    f"{sq['hits_returned']} / {len(sq['hits_kept'])}",
                    f"{ok_pages} / {err_pages}",
                    str(sq["evidence_count"]),
                    ", ".join(sq["distinct_domains"]) or "none",
                )
            c.print(sq_table)

        # 7. Coverage Breakdown
        if self.coverage_breakdown:
            cov_table = Table(title="Debug: Coverage Calculation Breakdown", border_style="green")
            cov_table.add_column("Sub-Question", style="bold yellow")
            cov_table.add_column("Evidence Count", justify="right")
            cov_table.add_column("Distinct Domains", justify="right")
            cov_table.add_column("Structured API?", style="dim")
            cov_table.add_column("Decision", style="bold")
            cov_table.add_column("Rule Applied", style="dim")
            for item in self.coverage_breakdown:
                decision_str = "[green]COVERED[/green]" if item["is_covered"] else "[red]NOT COVERED[/red]"
                cov_table.add_row(
                    item["sub_question_id"],
                    str(item["n_evidence"]),
                    str(item["n_domains"]),
                    "YES" if item["has_structured_api"] else "NO",
                    decision_str,
                    item["rule_applied"],
                )
            c.print(cov_table)

        # 8. GitHub Tool Diagnostics
        if self.github_diagnostics:
            gh_table = Table(title="Debug: GitHub Tool Candidate Selection", border_style="cyan")
            gh_table.add_column("Candidates", style="dim")
            gh_table.add_column("Chosen", style="bold green")
            gh_table.add_column("Rejected & Why", style="red")
            for gh in self.github_diagnostics:
                rej_str = "; ".join(f"{r['repo']} ({r['reason']})" for r in gh.get("rejected", [])) or "none"
                gh_table.add_row(
                    ", ".join(gh.get("candidates", [])) or "none",
                    str(gh.get("chosen")),
                    rej_str,
                )
            c.print(gh_table)

        # 9. Confidence Breakdown
        if self.confidence_breakdown:
            conf_table = Table(title="Debug: Confidence Breakdown", border_style="magenta")
            conf_table.add_column("Factor", style="bold")
            conf_table.add_column("Raw Value", justify="right")
            conf_table.add_column("Weight", justify="right")
            conf_table.add_column("Weighted", justify="right", style="cyan")
            factors = self.confidence_breakdown.get("factors", {})
            for factor_name, f_data in factors.items():
                if isinstance(f_data, dict):
                    conf_table.add_row(
                        factor_name,
                        f"{f_data.get('raw', 0):.2f}",
                        f"{f_data.get('weight', 0):.2f}",
                        f"{f_data.get('weighted', 0):.4f}",
                    )
            c.print(conf_table)
            caps = ", ".join(self.confidence_breakdown.get("caps_applied", [])) or "none"
            c.print(
                f"[bold]Final Score:[/bold] {self.confidence_breakdown.get('final_score')} "
                f"([bold cyan]{self.confidence_breakdown.get('final_level')}[/bold cyan]) | "
                f"[bold]Caps Applied:[/bold] {caps}"
            )


# ---------------------------------------------------------------------------
# Context variable for active collector
# ---------------------------------------------------------------------------

_debug_collector: contextvars.ContextVar[DebugCollector | None] = contextvars.ContextVar(
    "_debug_collector", default=None
)
_last_completed_collector: DebugCollector | None = None


def get_debug_collector() -> DebugCollector | None:
    """Return active DebugCollector in current async context or last completed."""
    return _debug_collector.get() or _last_completed_collector


def set_debug_collector(collector: DebugCollector | None) -> None:
    """Set active DebugCollector in current async context."""
    global _last_completed_collector
    _last_completed_collector = collector
    _debug_collector.set(collector)
