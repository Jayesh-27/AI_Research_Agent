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
from agent.diagnostics import get_debug_collector
from agent.evidence import EvidenceLedger, extract_evidence_from_page
from agent.filtering import filter_and_rank_hits
from agent.safety import scan_untrusted
from agent.schemas import SearchHit, SubQuestion, ToolName
from agent.tools.fetch_page import fetch_page
from agent.tools.github_stats import github_repo_stats
from agent.tools.web_search import web_search

logger = logging.getLogger(__name__)

_GITHUB_URL_PATTERN = re.compile(r"github\.com/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)")

_JUNK_REPO_PATTERN = re.compile(
    r"(handbook|awesome|examples?|demo|tutorial|workshop|course|stars|cookbook|benchmark|docs?)",
    re.IGNORECASE,
)
_GITHUB_STOP_WORDS = {
    "github",
    "repo",
    "repository",
    "stats",
    "stars",
    "forks",
    "recent",
    "activity",
    "commits",
    "issues",
    "what",
    "where",
    "which",
    "with",
    "metrics",
    "adoption",
    "community",
    "open",
    "source",
    "vector",
    "database",
    "databases",
    "candidate",
    "candidates",
    "each",
    "compare",
    "comparison",
    "analysis",
}


def filter_and_rank_github_candidates(
    raw_candidates: list[str],
    query_text: str,
) -> tuple[list[str], list[dict[str, str]]]:
    """Extract owner/name, filter out junk/tutorial repos, and rank by entity match confidence."""
    # Extract entity tokens from query_text (words of >=3 chars not in stop words)
    words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", query_text.lower())
    entity_tokens = [w for w in words if w not in _GITHUB_STOP_WORDS]

    valid_candidates: list[tuple[int, str]] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()

    for cand in raw_candidates:
        cand_clean = cand.strip().removesuffix(".git").strip("/")
        if not cand_clean or cand_clean.lower() in seen:
            continue
        seen.add(cand_clean.lower())

        parts = cand_clean.split("/")
        if len(parts) != 2:
            rejected.append({"repo": cand_clean, "reason": "invalid_repo_format"})
            continue

        owner, name = parts[0].lower(), parts[1].lower()
        if owner in (
            "features",
            "topics",
            "collections",
            "trending",
            "site",
            "orgs",
            "blog",
            "pricing",
            "about",
            "search",
        ):
            rejected.append({"repo": cand_clean, "reason": "not_a_user_repo"})
            continue

        # Reject names containing junk keywords
        if _JUNK_REPO_PATTERN.search(name):
            rejected.append({"repo": cand_clean, "reason": "contains_junk_keyword"})
            continue

        # Score relevance against entity tokens
        # Priority 0: repo name exactly matches an entity token (e.g. qdrant/qdrant -> 'qdrant')
        # Priority 1: repo name or owner contains entity token (e.g. milvus-io/milvus -> 'milvus')
        # Priority 2: candidate without explicit entity token
        priority = 2
        if entity_tokens:
            if any(name == tok for tok in entity_tokens):
                priority = 0
            elif any(tok in name or tok in owner for tok in entity_tokens):
                priority = 1
            else:
                # If specific entity targets were in the query and this repo matches none of them, reject!
                rejected.append({"repo": cand_clean, "reason": "no_entity_token_match"})
                continue

        valid_candidates.append((priority, cand_clean))

    # Sort by priority (0 first, then 1, then 2)
    valid_candidates.sort(key=lambda x: x[0])
    chosen = [c for _, c in valid_candidates]
    return chosen, rejected


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

        raw_queries = sub_q.search_queries if sub_q.search_queries else [sub_q.question]
        queries: list[str] = []
        for rq in raw_queries:
            if "|" in rq:
                queries.extend([part.strip() for part in rq.split("|") if part.strip()])
            else:
                queries.append(rq.strip())
        # Cap to top 2 queries per sub-question to avoid search engine rate limits
        queries = queries[:2]

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
            top_k=1,
        )

        collector = get_debug_collector()
        if collector:
            collector.record_sub_question_search(
                sq_id=sub_q.id,
                question=sub_q.question,
                queries_run=queries,
                hits_returned=len(all_hits),
                hits_kept=[
                    {
                        "url": h.url,
                        "title": h.title,
                        "relevance_score": getattr(h, "relevance", None),
                        "domain_score": getattr(h, "domain_score", None),
                    }
                    for h in filtered_hits
                ],
            )

        # 3. Fetch pages and extract evidence
        for hit in filtered_hits:
            fetch_res = await fetch_page(
                hit.url,
                settings=settings,
                budget=budget,
                chaos_mode=chaos_mode,
            )
            if collector:
                collector.record_page_fetch(
                    sq_id=sub_q.id,
                    url=hit.url,
                    ok=fetch_res.ok,
                    error=fetch_res.error,
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

        if collector:
            evs = [
                {
                    "id": ev.id,
                    "sub_question_id": ev.sub_question_id,
                    "claim": ev.claim,
                    "quote": ev.quote[:100],
                    "domain": ev.domain,
                    "trust": ev.trust,
                }
                for ev in ledger.get_by_sub_question(sub_q.id)
            ]
            collector.finalize_sub_question_evidence(sub_q.id, evs)


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
            repo_path = m.group(1).removesuffix(".git").strip("/")
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
        repos_found.add(m.group(1).removesuffix(".git").strip("/"))

    collector = get_debug_collector()
    candidate_list = list(repos_found)

    # Filter junk repos and rank according to sub_question / query entity matches
    query_context = f"{sub_q.question} {' '.join(sub_q.search_queries)}"
    ranked_candidates, filter_rejected = filter_and_rank_github_candidates(
        candidate_list, query_context
    )

    chosen_repos: list[str] = []
    rejected_repos: list[dict[str, str]] = list(filter_rejected)

    # Limit to top 3 confident repos per sub-question
    for repo in ranked_candidates[:3]:
        res = await github_repo_stats(repo, settings=settings, budget=budget)
        if res.ok and isinstance(res.data, dict):
            d = res.data
            # Reject forks and archived repositories
            if d.get("archived"):
                rejected_repos.append({"repo": repo, "reason": "is_archived_repo"})
                continue
            if d.get("fork"):
                rejected_repos.append({"repo": repo, "reason": "is_fork_repo"})
                continue

            chosen_repos.append(repo)
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
        else:
            rejected_repos.append({"repo": repo, "reason": res.error or "fetch_failed"})

    for repo in ranked_candidates[3:]:
        rejected_repos.append({"repo": repo, "reason": "exceeded_max_repo_candidates_cap"})

    if collector:
        collector.record_github_decision(
            candidates=candidate_list,
            chosen=", ".join(chosen_repos) if chosen_repos else None,
            rejected=rejected_repos,
        )


async def execute_plan(
    sub_questions: list[SubQuestion],
    ledger: EvidenceLedger,
    *,
    settings: Settings,
    budget: Budget,
    max_concurrency: int = 2,
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
