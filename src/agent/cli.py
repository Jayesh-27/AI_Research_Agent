"""Typer CLI interface for the research agent.

Commands:
- `ask`: Run the research agent with full options (--format, --save, --show-plan, --show-trace, --chaos)
- `baseline`: Run non-agentic single-call baseline for comparison
- `doctor`: Check environment, credentials, provider connectivity, and budget settings
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from agent.config import Settings
from agent.diagnostics import (
    DebugCollector,
    check_model_availability,
    set_debug_collector,
)
from agent.orchestrator import run_research
from agent.render import render_markdown
from agent.schemas import FinalAnswer, Plan
from agent.tracing import get_current_trace

# Configure default logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

app = typer.Typer(
    name="research-agent",
    help="Production-Oriented AI Research Agent CLI (Predixion AI Intern Assignment)",
    add_completion=False,
)
console = Console(legacy_windows=False)


@app.command()
def ask(
    question: str = typer.Argument(..., help="The research question to investigate"),
    format: str = typer.Option("md", "--format", "-f", help="Output format: 'md' or 'json'"),
    save: str | None = typer.Option(
        None, "--save", "-s", help="Directory to save .md and .json results"
    ),
    force_save: bool = typer.Option(
        False, "--force-save", help="Save failed runs to target directory instead of runs/failed"
    ),
    routing: str = typer.Option(
        "hybrid", "--routing", help="Model routing strategy: 'hybrid' or 'all-large'"
    ),
    chaos: str = typer.Option(
        "off",
        "--chaos",
        help="Chaos failure injection mode: search_down|timeout|malformed|empty|injection",
    ),
    show_plan: bool = typer.Option(
        False, "--show-plan", help="Display the execution plan before research"
    ),
    show_trace: bool = typer.Option(
        False, "--show-trace", help="Display execution span trace timeline"
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Enable fine-grained diagnostics tables and save debug JSON"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose info logging"),
) -> None:
    """Investigate a research question using planning, tool execution, and synthesis."""
    if verbose:
        logging.getLogger("agent").setLevel(logging.INFO)

    settings = Settings()
    if routing == "all-large" or settings.ROUTING == "all-large":
        settings.SMALL_MODEL = settings.LARGE_MODEL
    is_debug = debug or settings.AGENT_DEBUG or os.environ.get("AGENT_DEBUG") == "1"
    debug_collector = None
    if is_debug:
        debug_collector = DebugCollector(run_id="init", settings=settings)
        debug_collector.model_availability = check_model_availability(settings)
        set_debug_collector(debug_collector)

    def plan_callback(plan: Plan) -> None:
        if show_plan:
            table = Table(title=f"Plan: {plan.task_type.value.upper()}", border_style="cyan")
            table.add_column("ID", style="bold yellow", width=6)
            table.add_column("Sub-question", style="white")
            table.add_column("Tools", style="green")
            table.add_column("Search Queries", style="dim")

            for sq in plan.sub_questions:
                tools_str = ", ".join(t.value for t in sq.tools)
                queries_str = " | ".join(sq.search_queries)
                table.add_row(sq.id, sq.question, tools_str, queries_str)

            console.print(table)
            console.print(f"[dim]Interpretation:[/dim] {plan.interpretation}\n")

    with console.status("[bold green]Agent researching query...", spinner="dots"):
        final_answer = asyncio.run(
            run_research(
                question,
                settings=settings,
                chaos_mode=chaos,
                on_plan_callback=plan_callback,
            )
        )

    # Render output
    md_output = render_markdown(final_answer)

    if format.lower() == "json":
        console.print_json(final_answer.model_dump_json(indent=2))
    else:
        console.print(Panel(Markdown(md_output), title="Research Agent Report", border_style="blue"))

    # Show trace if requested
    if show_trace:
        trace = get_current_trace()
        if trace and trace.spans:
            table = Table(title=f"Execution Trace [{trace.run_id}]", border_style="magenta")
            table.add_column("Span Name", style="cyan")
            table.add_column("Duration (ms)", style="green", justify="right")
            table.add_column("Details", style="dim")
            table.add_column("Status", style="bold")

            for span in trace.spans:
                status_str = f"[red]ERR: {span.error}[/red]" if span.error else "[green]OK[/green]"
                details = []
                if "tokens_in" in span.metadata:
                    details.append(
                        f"{span.metadata['tokens_in']} in / {span.metadata['tokens_out']} out"
                    )
                if "cached" in span.metadata:
                    details.append("CACHED")
                if "model" in span.metadata:
                    details.append(str(span.metadata["model"]))
                table.add_row(span.name, f"{span.duration_ms:.1f}", ", ".join(details), status_str)

            console.print(table)

    # Show and save debug diagnostics if enabled
    if debug_collector:
        run_id = final_answer.stats.run_id if final_answer.stats else "debug"
        debug_collector.run_id = run_id
        saved_debug_path = debug_collector.save_to_disk()
        debug_collector.print_rich_tables(console)
        console.print(f"[bold cyan]Debug telemetry saved to:[/bold cyan] {saved_debug_path}")

    # Save to disk if requested
    if save:
        if final_answer.status == "failed" and not force_save:
            out_dir = Path("runs/failed")
            console.print(
                f"[bold yellow]Run status is 'failed'. Writing outputs to {out_dir} instead of '{save}' (pass --force-save to override).[/bold yellow]"
            )
        else:
            out_dir = Path(save)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(c if c.isalnum() else "_" for c in question[:30]).strip("_")
        run_id = final_answer.stats.run_id if final_answer.stats else "out"

        md_file = out_dir / f"{safe_name}_{run_id}.md"
        json_file = out_dir / f"{safe_name}_{run_id}.json"

        md_file.write_text(md_output, encoding="utf-8")
        json_file.write_text(final_answer.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[bold green]Saved output to:[/bold green] {md_file} and {json_file}")


@app.command()
def baseline(
    question: str = typer.Argument(..., help="The research question to ask directly to LLM"),
) -> None:
    """Run a non-agentic single LLM call with no tools for comparative evaluation."""
    from agent.llm import call_json
    from agent.schemas import RunStats

    settings = Settings()

    system_prompt = (
        "You are an AI research assistant. Answer the user question comprehensively. "
        "Provide a short answer, list key findings, state assumptions, limitations, and next steps."
    )

    async def _run() -> None:
        import time

        from agent.schemas import SynthesisOutput

        start = time.time()
        output = await call_json(
            tier="large",
            system=system_prompt,
            user=question,
            schema=SynthesisOutput,
            settings=settings,
            span_name="baseline_call",
        )
        latency = time.time() - start

        final_ans = FinalAnswer(
            status="ok",
            user_question=question,
            short_answer=output.short_answer,
            key_findings=output.key_findings,
            sources=[],  # Baseline has 0 grounded sources
            confidence={
                "level": "medium",
                "score": 0.5,
                "rationale": ["Non-agentic single LLM baseline (unverified)"],
            },  # type: ignore[arg-type]
            assumptions=output.assumptions,
            limitations=[
                "Baseline generated without tool retrieval; potential hallucinations or stale data."
            ],
            next_steps=output.next_steps,
            stats=RunStats(
                run_id="baseline",
                latency_s=round(latency, 2),
                warnings=["Non-agentic baseline run"],
            ),
        )
        console.print(
            Panel(
                render_markdown(final_ans),
                title="Baseline (No Tools) Report",
                border_style="yellow",
            )
        )

    with console.status("[bold yellow]Running baseline single-prompt LLM...", spinner="dots"):
        asyncio.run(_run())


@app.command()
def diagnose() -> None:
    """Run deep diagnostics on configuration, connectivity, and model availability."""
    settings = Settings()
    collector = DebugCollector(run_id="diagnose", settings=settings)
    avail = check_model_availability(settings)
    collector.model_availability = avail
    out_file = collector.save_to_disk()
    collector.print_rich_tables(console)
    console.print(f"\n[bold cyan]Diagnostic output saved to:[/bold cyan] {out_file}\n")
    if not avail.get("ok"):
        console.print(f"[bold red]Provider error:[/bold red] {avail.get('error')}")
    elif not avail.get("small_model_available") or not avail.get("large_model_available"):
        console.print("[bold yellow]Warning:[/bold yellow] One or more configured models are not listed by provider.")
    else:
        console.print("[bold green]All systems operational.[/bold green]")


@app.command()
def doctor() -> None:
    """Check configuration, API credentials, and provider availability."""

    settings = Settings()
    table = Table(title="System & Configuration Health Check", border_style="cyan")
    table.add_column("Component", style="bold")
    table.add_column("Configured Value", style="white")
    table.add_column("Status", style="bold")

    # LLM check
    table.add_row("LLM Base URL", settings.LLM_BASE_URL, "[green]OK[/green]")
    api_key_status = "[green]SET[/green]" if settings.LLM_API_KEY else "[yellow]EMPTY[/yellow]"
    table.add_row("LLM API Key", "******" if settings.LLM_API_KEY else "Not set", api_key_status)
    table.add_row("Small Model", settings.SMALL_MODEL, "[cyan]Configured[/cyan]")
    table.add_row("Large Model", settings.LARGE_MODEL, "[cyan]Configured[/cyan]")

    # Tools check
    tavily_status = (
        "[green]SET[/green]"
        if settings.TAVILY_API_KEY
        else "[dim]Not set (uses DDG fallback)[/dim]"
    )
    table.add_row("Tavily API Key", "******" if settings.TAVILY_API_KEY else "None", tavily_status)
    gh_status = (
        "[green]SET[/green]"
        if settings.GITHUB_TOKEN
        else "[dim]Optional (public rate limits apply)[/dim]"
    )
    table.add_row("GitHub Token", "******" if settings.GITHUB_TOKEN else "None", gh_status)

    # Budgets
    table.add_row("Max LLM Calls", str(settings.MAX_LLM_CALLS), "[green]Active[/green]")
    table.add_row("Max Search Calls", str(settings.MAX_SEARCH_CALLS), "[green]Active[/green]")
    table.add_row("Run Timeout", f"{settings.RUN_TIMEOUT_S}s", "[green]Active[/green]")

    console.print(table)


if __name__ == "__main__":
    app()
