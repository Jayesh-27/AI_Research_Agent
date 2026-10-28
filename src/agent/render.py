"""Markdown rendering for FinalAnswer instances.

Produces clean, structured GitHub-flavored Markdown matching the brief's required fields.
"""

from __future__ import annotations

from agent.schemas import FinalAnswer


def render_markdown(answer: FinalAnswer) -> str:
    """Render a FinalAnswer into formatted GitHub-flavored Markdown."""
    lines: list[str] = []

    # Title & Question
    lines.append(f"# Research Report: {answer.user_question}\n")
    lines.append(
        f"**Status:** `{answer.status.upper()}` | **Confidence:** `{answer.confidence.level.upper()}` ({answer.confidence.score * 100:.1f}%)\n"
    )

    # Short Answer
    lines.append("## Short Answer")
    lines.append(f"{answer.short_answer}\n")

    # Key Findings
    lines.append("## Key Findings")
    if answer.key_findings:
        for idx, finding in enumerate(answer.key_findings, 1):
            citations = ", ".join(f"`[{eid}]`" for eid in finding.evidence_ids)
            lines.append(f"{idx}. {finding.statement} (Sources: {citations})")
    else:
        lines.append("_No verified findings._")
    lines.append("")

    # Sources
    lines.append("## Sources")
    if answer.sources:
        for src in answer.sources:
            used_str = f" _(Used in: {', '.join(src.used_in)})_" if src.used_in else ""
            lines.append(f"- **[{src.id}]** [{src.title}]({src.url}) - `{src.domain}`{used_str}")
    else:
        lines.append("_No external sources cited._")
    lines.append("")

    # Confidence Analysis
    lines.append("## Confidence & Rationale")
    lines.append(
        f"- **Overall Level:** {answer.confidence.level.capitalize()} ({answer.confidence.score:.3f})"
    )
    if answer.confidence.rationale:
        for r in answer.confidence.rationale:
            lines.append(f"  - {r}")
    lines.append("")

    # Assumptions
    lines.append("## Assumptions")
    if answer.assumptions:
        for a in answer.assumptions:
            lines.append(f"- {a}")
    else:
        lines.append("- None explicitly assumed.")
    lines.append("")

    # Limitations
    lines.append("## Limitations")
    if answer.limitations:
        for lim in answer.limitations:
            lines.append(f"- {lim}")
    else:
        lines.append("- No critical limitations flagged.")
    lines.append("")

    # Next Steps
    lines.append("## Suggested Next Steps")
    if answer.next_steps:
        for step in answer.next_steps:
            lines.append(f"1. {step}")
    else:
        lines.append("1. Validate findings against primary vendor documentation.")
    lines.append("")

    # Execution Statistics
    if answer.stats:
        s = answer.stats
        lines.append("## Operational Telemetry")
        lines.append(f"- **Run ID:** `{s.run_id}`")
        lines.append(f"- **Latency:** {s.latency_s:.2f}s")
        lines.append(f"- **LLM Calls:** {s.llm_calls}")
        lines.append(f"- **Tokens:** {s.tokens_in:,} in / {s.tokens_out:,} out")
        lines.append(f"- **Estimated Cost:** ${s.est_cost_usd:.4f} USD")
        if s.tool_calls:
            calls_str = ", ".join(f"{k}: {v}" for k, v in s.tool_calls.items())
            lines.append(f"- **Tool Calls:** {calls_str}")
        if s.fallbacks_used:
            lines.append(f"- **Fallbacks Triggered:** {', '.join(s.fallbacks_used)}")
        if s.warnings:
            lines.append(f"- **Warnings:** {', '.join(s.warnings)}")
        lines.append("")

    return "\n".join(lines)
