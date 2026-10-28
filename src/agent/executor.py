"""Executor module — runs the research pipeline per sub-question.

Coordinates:
- Concurrent sub-question execution bounded by semaphore
- Search, hit filtering, page fetching, injection scanning
- Evidence extraction & verification into the shared EvidenceLedger
- Structured tool execution (e.g. GitHub repo stats) when requested by plan
"""

from __future__ import annotations

import asyncio
import logging
import re

from agent.config import Budget, Settings
from agent.evidence import EvidenceLedger, extract_evidence_from_page
from agent.filtering import filter_and_rank_hits
from agent.safety import scan_untrusted
from agent.schemas import SearchHit, SubQuestion, ToolName
from agent.tools.fetch_page import fetch_page
from agent.tools.github_stats import github_repo_stats
from agent.tools.web_search import web_search

logger = logging.getLogger(__name__)

_GITHUB_URL_PATTERN = re.compile(r"github\.com/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)")


async def execute_sub_question(
    sub_q: SubQuestion,
    ledger: EvidenceLedger,
    *,
    settings: Settings,
    budget: Budget,
    semaphore: asyncio.Semaphore,
    chaos_mode: str = "off",
) -> None:
    """Execute search, fetch, scan, and extraction for a single sub-question."""
    async with semaphore:
        logger.info("Executing sub-question [%s]: %s", sub_q.id, sub_q.question)

        queries = sub_q.search_queries if sub_q.search_queries else [sub_q.question]

        # 1. Gather search hits across queries
        all_hits: list[SearchHit] = []
        for q in queries:
            tool_res = await web_search(
                q,
                max_results=6,
                settings=settings,
                budget=budget,
                chaos_mode=chaos_mode,
            )
            if tool_res.ok and isinstance(tool_res.data, list):
                for item in tool_res.data:
                    try:
                        all_hits.append(SearchHit.model_validate(item))
                    except Exception as e:
                        logger.warning("Could not validate search hit: %s", e)

        # 2. Filter and rank hits
        filtered_hits = await filter_and_rank_hits(
            all_hits,
            sub_q.question,
            settings=settings,
            budget=budget,
            top_k=4,
        )

        # 3. Fetch pages and extract evidence
        for hit in filtered_hits:
            fetch_res = await fetch_page(
                hit.url,
                settings=settings,
                budget=budget,
                chaos_mode=chaos_mode,
            )
            if not fetch_res.ok or not isinstance(fetch_res.data, dict):
                logger.info("Failed to fetch %s: %s", hit.url, fetch_res.error)
                continue

            page_data = fetch_res.data
            raw_text = page_data.get("text", "")

            # 4. Prompt injection scan
            scan_res = scan_untrusted(raw_text)
            if scan_res.flagged:
                logger.warning(
                    "Page %s flagged by injection scanner (score=%.2f, matches=%s)",
                    hit.url,
                    scan_res.score,
                    scan_res.matches,
                )
                if scan_res.score >= 0.7:
                    logger.warning("Quarantining high-risk page %s", hit.url)
                    continue

            trust_level = "low" if scan_res.flagged else "normal"

            # 5. Extract and verify evidence
            await extract_evidence_from_page(
                page_data=page_data,
                sub_question_id=sub_q.id,
                sub_question_text=sub_q.question,
                ledger=ledger,
                settings=settings,
                budget=budget,
                trust=trust_level,
            )

        # 6. Execute github_repo_stats if planned
        if ToolName.github_repo_stats in sub_q.tools:
            await _execute_github_stats_if_present(
                sub_q=sub_q,
                hits=all_hits,
                ledger=ledger,
                settings=settings,
                budget=budget,
            )


async def _execute_github_stats_if_present(
    sub_q: SubQuestion,
    hits: list[SearchHit],
    ledger: EvidenceLedger,
    *,
    settings: Settings,
    budget: Budget,
) -> None:
    """Identify GitHub repository paths from hits or question text, and fetch stats."""
    repos_found: set[str] = set()

    # Search in hits URLs
    for h in hits:
        m = _GITHUB_URL_PATTERN.search(h.url)
        if m:
            repo_path = m.group(1).rstrip(".git").strip("/")
            # Ignore general pages like github.com/features or github.com/topics
            parts = repo_path.split("/")
            if len(parts) == 2 and parts[0] not in (
                "features",
                "topics",
                "collections",
                "trending",
            ):
                repos_found.add(repo_path)

    # Also search sub_q question text
    for m in _GITHUB_URL_PATTERN.finditer(sub_q.question):
        repos_found.add(m.group(1).rstrip(".git").strip("/"))

    # Limit to top 3 repos per sub-question
    for repo in list(repos_found)[:3]:
        res = await github_repo_stats(repo, settings=settings, budget=budget)
        if res.ok and isinstance(res.data, dict):
            d = res.data
            summary = (
                f"Repository {repo}: {d.get('stars', 0):,} stars, "
                f"{d.get('forks', 0):,} forks, {d.get('open_issues', 0):,} open issues, "
                f"License: {d.get('license', 'Unknown')}, Last push: {d.get('pushed_at', 'Unknown')}"
            )
            ledger.add_structured_evidence(
                sub_question_id=sub_q.id,
                claim=f"GitHub metrics and adoption stats for {repo}",
                structured_data_str=summary,
                url=d.get("url", f"https://github.com/{repo}"),
                title=f"{repo} GitHub Stats",
            )


async def execute_plan(
    sub_questions: list[SubQuestion],
    ledger: EvidenceLedger,
    *,
    settings: Settings,
    budget: Budget,
    max_concurrency: int = 3,
    chaos_mode: str = "off",
) -> None:
    """Run all sub-questions concurrently with bounded concurrency."""
    sem = asyncio.Semaphore(max_concurrency)
    tasks = [
        execute_sub_question(
            sq,
            ledger,
            settings=settings,
            budget=budget,
            semaphore=sem,
            chaos_mode=chaos_mode,
        )
        for sq in sub_questions
    ]
    await asyncio.gather(*tasks, return_exceptions=True)
