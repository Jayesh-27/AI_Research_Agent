"""Filtering and ranking search hits.

Provides:
- URL canonicalization and deduplication (capped per domain)
- Deterministic domain quality scoring (boost high-reputation, penalize SEO/aggregators)
- Batched LLM relevance scoring (0-3) using SMALL model
- Weighted ranking: 0.6 * (relevance / 3) + 0.4 * domain_score
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from agent.config import Budget, Settings
from agent.llm import LLMOutputError, call_json
from agent.schemas import RelevanceBatch, SearchHit

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

# Known high-reputation domains & suffixes
_BOOSTED_DOMAINS = {
    "docs.python.org",
    "developer.mozilla.org",
    "arxiv.org",
    "stackoverflow.com",
    "en.wikipedia.org",
    "cloudflare.com",
    "aws.amazon.com",
    "cloud.google.com",
    "learn.microsoft.com",
}

_BOOSTED_TLDS = (".gov", ".edu", ".org")

# Low-quality or SEO-farm / content aggregator domains
_PENALIZED_DOMAINS = {
    "pinterest.com",
    "quora.com",
    "medium.com",  # often paywalled or unvetted listicles
    "geeksforgeeks.org",
    "tutorialspoint.com",
    "listverse.com",
    "buzzfeed.com",
    "reddit.com",
    "pastebin.com",
}


def canonicalize_url(url: str) -> str:
    """Normalize a URL for deduplication."""
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        # Strip trailing slash from path
        path = parsed.path.rstrip("/")
        # Discard query and fragment for deduplication where appropriate
        return urlunparse((scheme, netloc, path, "", "", ""))
    except Exception:
        return url.strip().lower()


def get_domain(url: str) -> str:
    """Extract registered/main domain from URL."""
    try:
        netloc = urlparse(url).netloc.lower()
        # Remove port if present
        return netloc.split(":")[0]
    except Exception:
        return ""


_JUNK_REPO_KEYWORDS = {
    "awesome",
    "curated",
    "tutorial",
    "tutorials",
    "demo",
    "demos",
    "example",
    "examples",
    "sample",
    "samples",
    "course",
    "workshop",
    "cookbook",
    "handbook",
    "cheatsheet",
    "cheat-sheet",
    "stars",
    "collection",
    "list",
    "lists",
}

_KNOWN_OFFICIAL_ORGS = {
    "google",
    "meta",
    "facebook",
    "microsoft",
    "apple",
    "apache",
    "openai",
    "anthropic",
    "amazon",
    "aws",
    "docker",
    "vercel",
    "cloudflare",
    "elastic",
    "redis",
    "qdrant",
    "milvus-io",
    "weaviate",
    "pinecone-io",
    "chroma-core",
    "langchain-ai",
    "huggingface",
    "pytorch",
    "tensorflow",
    "kubernetes",
    "pallets",
    "psf",
    "python",
    "golang",
    "rust-lang",
    "torvalds",
}


def score_github_url(url: str) -> float:
    """Score GitHub sources deterministically based on repo path.

    - Low quality (0.30): 'awesome-*' lists, curated lists, tutorials, demos, examples.
    - High quality (1.00): Official organization / primary project repositories.
    - Normal quality (0.70): Standard open-source repositories without junk keywords.
    """
    try:
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if not path:
            return 0.5

        parts = [p.lower() for p in path.split("/") if p]
        if not parts:
            return 0.5

        # Check for non-repo GitHub subpaths
        if parts[0] in {
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
            "explore",
        }:
            return 0.5

        owner = parts[0]
        repo = parts[1] if len(parts) > 1 else ""

        # Check for low-quality keywords in owner, repo, or path segments
        all_tokens = set(re.findall(r"[a-z0-9]+", path.lower()))
        if any(kw in all_tokens for kw in _JUNK_REPO_KEYWORDS) or repo.startswith("awesome"):
            return 0.30

        # If only owner is specified and matches known org
        if not repo:
            if owner in _KNOWN_OFFICIAL_ORGS:
                return 1.0
            return 0.5

        # Check for official org repository
        clean_owner = re.sub(
            r"[-_](io|ai|core|org|project|dev|official|hq|inc|corp|labs?|software|tech|team)$",
            "",
            owner,
        )
        clean_repo = re.sub(
            r"[-_](io|ai|core|org|project|dev|official|hq|inc|corp|labs?|software|tech|team)$",
            "",
            repo,
        )

        if (
            owner == repo
            or clean_owner == clean_repo
            or clean_repo in clean_owner
            or clean_owner in clean_repo
            or owner in _KNOWN_OFFICIAL_ORGS
        ):
            return 1.0

        return 0.70
    except Exception:
        return 0.5


def calculate_domain_score(url: str) -> float:
    """Calculate a deterministic 0-1 quality score for a source domain."""
    domain = get_domain(url)
    if not domain:
        return 0.5

    # Check penalties first
    for bad in _PENALIZED_DOMAINS:
        if domain == bad or domain.endswith("." + bad):
            return 0.2

    # Check GitHub repo scoring by path
    if domain == "github.com" or domain.endswith(".github.com"):
        return score_github_url(url)

    # Check exact boost
    for good in _BOOSTED_DOMAINS:
        if domain == good or domain.endswith("." + good):
            return 1.0

    # Check engineering blogs
    if "blog." in domain or domain.endswith(".github.io") or "tech." in domain:
        return 0.85

    # Check TLD boosts
    for tld in _BOOSTED_TLDS:
        if domain.endswith(tld):
            return 0.85

    return 0.5


def dedupe_and_cap_by_domain(hits: list[SearchHit], max_per_domain: int = 2) -> list[SearchHit]:
    """Deduplicate hits by canonical URL and cap maximum results per domain."""
    seen_urls: set[str] = set()
    domain_counts: dict[str, int] = {}
    filtered: list[SearchHit] = []

    for hit in hits:
        canon_url = canonicalize_url(hit.url)
        if canon_url in seen_urls:
            continue

        domain = get_domain(hit.url)
        count = domain_counts.get(domain, 0)
        if count >= max_per_domain:
            continue

        seen_urls.add(canon_url)
        domain_counts[domain] = count + 1
        filtered.append(hit)

    return filtered


async def filter_and_rank_hits(
    hits: list[SearchHit],
    sub_question: str,
    *,
    settings: Settings,
    budget: Budget | None = None,
    top_k: int = 4,
) -> list[SearchHit]:
    """Dedupe, compute domain score, score relevance with SMALL LLM, and rank."""
    if not hits:
        return []

    # 1. Dedupe and cap per domain
    deduped = dedupe_and_cap_by_domain(hits, max_per_domain=2)
    if not deduped:
        return []

    # 2. Assign deterministic domain scores
    for hit in deduped:
        hit.domain_score = calculate_domain_score(hit.url)

    # 3. Batched relevance scoring via SMALL model
    relevance_map: dict[str, int] = {}
    results_text = "\n".join(
        f"- URL: {h.url}\n  Title: {h.title}\n  Snippet: {h.snippet[:200]}" for h in deduped
    )

    prompt_path = _PROMPTS_DIR / "relevance_score.md"
    system_prompt = (
        prompt_path.read_text(encoding="utf-8")
        if prompt_path.exists()
        else "Rate relevance (0-3) of each search result to the sub-question."
    )
    user_prompt = f"Sub-question: {sub_question}\n\nSearch Results:\n{results_text}"

    try:
        batch = await call_json(
            tier="small",
            system=system_prompt,
            user=user_prompt,
            schema=RelevanceBatch,
            settings=settings,
            budget=budget,
            temperature=0.0,
            max_tokens=settings.MAX_TOKENS_RELEVANCE,
            span_name="relevance_scoring",
        )
        for item in batch.scores:
            relevance_map[item.url] = item.score
    except (LLMOutputError, Exception) as exc:
        warn_msg = f"Relevance scoring degraded: LLM call failed ({type(exc).__name__}: {exc}), defaulted scores to 2."
        logger.warning(warn_msg)
        if budget is not None and hasattr(budget, "warnings"):
            if warn_msg not in budget.warnings:
                budget.warnings.append(warn_msg)
        for h in deduped:
            relevance_map[h.url] = 2

    # 4. Populate relevance, drop relevance <= 1, and compute combined ranking score
    scored_hits: list[tuple[float, SearchHit]] = []
    for hit in deduped:
        rel = relevance_map.get(hit.url, 2)
        hit.relevance = rel
        if rel <= 1:
            continue  # Drop irrelevant / noisy items

        domain_sc = hit.domain_score if hit.domain_score is not None else 0.5
        # Ranking formula: 0.6 * normalized_relevance + 0.4 * domain_score
        combined_score = 0.6 * (rel / 3.0) + 0.4 * domain_sc
        scored_hits.append((combined_score, hit))

    # Sort descending by combined score
    scored_hits.sort(key=lambda x: x[0], reverse=True)

    # Return top_k hits
    return [hit for _, hit in scored_hits[:top_k]]
