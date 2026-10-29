"""Markdown rendering for FinalAnswer instances.

Produces clean, structured GitHub-flavored Markdown matching the brief's required fields.
"""

from __future__ import annotations

import re

from agent.schemas import FinalAnswer


def deduplicate_list(items: list[str]) -> list[str]:
    """Deduplicate strings normalizing whitespace, case, and punctuation, preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if not item or not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not cleaned:
            continue
        norm_key = re.sub(r"[^\w\s]", "", cleaned.lower())
        norm_key = re.sub(r"\s+", " ", norm_key).strip()
        if norm_key and norm_key not in seen:
            seen.add(norm_key)
            result.append(cleaned)
    return result


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
        for idx, src in enumerate(answer.sources, 1):
            sid = src.id if src.id and src.id.strip() else f"s{idx}"
            title = src.title if src.title and src.title.strip() else (src.domain or f"Source {idx}")
            used_str = f" _(Used in: {', '.join(src.used_in)})_" if src.used_in else ""
            lines.append(f"- **[{sid}]** [{title}]({src.url}) - `{src.domain}`{used_str}")
    else:
        lines.append("_No external sources cited._")
    lines.append("")

    # Confidence Analysis
    lines.append("## Confidence & Rationale")
    lines.append(
        f"- **Overall Level:** {answer.confidence.level.capitalize()} ({answer.confidence.score:.3f})"
    )
    if answer.confidence.rationale:
        for r in deduplicate_list(answer.confidence.rationale):
            lines.append(f"  - {r}")
    lines.append("")

    # Assumptions
    lines.append("## Assumptions")
    clean_assumptions = deduplicate_list(answer.assumptions)
    if clean_assumptions:
        for a in clean_assumptions:
            lines.append(f"- {a}")
    else:
        lines.append("- None explicitly assumed.")
    lines.append("")

    # Limitations (deduplicate and cap at 6)
    lines.append("## Limitations")
    clean_limitations = deduplicate_list(answer.limitations)
    if clean_limitations:
        if len(clean_limitations) > 6:
            extra = len(clean_limitations) - 6
            clean_limitations = clean_limitations[:6] + [f"(+{extra} more)"]
        for lim in clean_limitations:
            lines.append(f"- {lim}")
    else:
        lines.append("- No critical limitations flagged.")
    lines.append("")

    # Next Steps (numbered with real incrementing numbers)
    lines.append("## Suggested Next Steps")
    clean_next_steps = deduplicate_list(answer.next_steps)
    if clean_next_steps:
        for idx, step in enumerate(clean_next_steps, 1):
            lines.append(f"{idx}. {step}")
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
            clean_fb = deduplicate_list(s.fallbacks_used)
            lines.append(f"- **Fallbacks Triggered:** {', '.join(clean_fb)}")
        if s.warnings:
            clean_warn = [w for w in deduplicate_list(s.warnings) if w and w.strip()]
            if clean_warn:
                lines.append(f"- **Warnings:** {', '.join(clean_warn)}")
        lines.append("")

    return "\n".join(lines)
