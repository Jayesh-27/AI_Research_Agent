"""Safety utilities — URL/SSRF guard and prompt-injection scanner.

Provides:
- :func:`is_safe_url` — blocks private/loopback/link-local/metadata IPs,
  non-HTTP schemes, and credentials in URLs.
- :func:`wrap_untrusted` — wraps text in ``<untrusted_document>`` tags with
  sanitisation and truncation.
- :func:`scan_untrusted` — regex-based heuristic scanner that detects
  prompt-injection patterns in untrusted text.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from urllib.parse import urlparse

from agent.schemas import ScanResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_UNTRUSTED_LEN: int = 3_000

_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),  # loopback
    ipaddress.ip_network("10.0.0.0/8"),  # private class A
    ipaddress.ip_network("172.16.0.0/12"),  # private class B
    ipaddress.ip_network("192.168.0.0/16"),  # private class C
    ipaddress.ip_network("169.254.0.0/16"),  # link-local (incl. AWS metadata)
    ipaddress.ip_network("0.0.0.0/8"),  # "this" network
    ipaddress.ip_network("::1/128"),  # IPv6 loopback
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
    ipaddress.ip_network("fc00::/7"),  # IPv6 unique-local
]


# ---------------------------------------------------------------------------
# URL safety
# ---------------------------------------------------------------------------


def is_safe_url(url: str) -> bool:
    """Return ``True`` only if *url* is considered safe for fetching.

    Rejects:
    - Non-HTTP(S) schemes (``file://``, ``ftp://``, etc.)
    - URLs containing embedded credentials (``user:pass@host``)
    - Hosts that resolve to private, loopback, link-local, or metadata IPs
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    # --- Scheme check ---
    if parsed.scheme not in ("http", "https"):
        logger.debug("Blocked URL (scheme=%s): %s", parsed.scheme, url)
        return False

    # --- Credential check ---
    if parsed.username or parsed.password:
        logger.debug("Blocked URL (credentials present): %s", url)
        return False

    # --- Hostname presence ---
    hostname = parsed.hostname
    if not hostname:
        logger.debug("Blocked URL (no hostname): %s", url)
        return False

    # --- DNS resolution + IP blocklist ---
    try:
        addr_infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        logger.debug("Blocked URL (DNS resolution failed): %s", url)
        return False

    for family, _, _, _, sockaddr in addr_infos:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return False

        for network in _BLOCKED_NETWORKS:
            if ip in network:
                logger.debug("Blocked URL (resolved to %s in %s): %s", ip, network, url)
                return False

    return True


# ---------------------------------------------------------------------------
# Untrusted-text wrapping
# ---------------------------------------------------------------------------


def wrap_untrusted(
    text: str,
    source: str,
    max_length: int = _MAX_UNTRUSTED_LEN,
) -> str:
    """Wrap *text* in ``<untrusted_document>`` tags for safe inclusion in prompts.

    - Strips any occurrence of the closing tag inside *text* to prevent
      injection / early escape.
    - Truncates to *max_length* characters.
    """
    # Sanitise: remove closing tags that could let content escape the wrapper
    sanitised = text.replace("</untrusted_document>", "")

    # Truncate
    if len(sanitised) > max_length:
        sanitised = sanitised[:max_length] + "\n… [truncated]"

    return f'<untrusted_document source="{source}">\n{sanitised}\n</untrusted_document>'


# ---------------------------------------------------------------------------
# Prompt-injection scanning
# ---------------------------------------------------------------------------

# Each pattern tuple: (compiled regex, weight, human-readable label)
_INJECTION_PATTERNS: list[tuple[re.Pattern[str], float, str]] = [
    (
        re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.I),
        0.40,
        "ignore-previous-instructions",
    ),
    (
        re.compile(r"disregard\s+(the\s+)?above", re.I),
        0.35,
        "disregard-above",
    ),
    (
        re.compile(r"you\s+are\s+now\b", re.I),
        0.25,
        "you-are-now",
    ),
    (
        re.compile(r"system\s*prompt", re.I),
        0.20,
        "system-prompt-reference",
    ),
    (
        re.compile(r"\bact\s+as\b", re.I),
        0.15,
        "act-as",
    ),
    (
        re.compile(r"developer\s+message", re.I),
        0.20,
        "developer-message",
    ),
    (
        re.compile(r"reveal\s+your\s+(prompt|instructions)", re.I),
        0.30,
        "reveal-prompt",
    ),
    (
        # Tool-call / function-call syntax patterns
        re.compile(
            r"(<\s*/?function_call|<\s*/?tool_call|<\|im_start\|>|<\|endoftext\|>)",
            re.I,
        ),
        0.35,
        "tool-call-syntax",
    ),
    (
        # Long base64 blobs (>100 chars of contiguous base64)
        re.compile(r"[A-Za-z0-9+/=]{100,}"),
        0.20,
        "long-base64-blob",
    ),
    (
        # Hidden text hints (zero-width chars, invisible unicode)
        re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]{3,}"),
        0.25,
        "hidden-text",
    ),
]


def scan_untrusted(text: str) -> ScanResult:
    """Scan *text* for prompt-injection patterns using weighted regex heuristics.

    Returns a :class:`ScanResult` with a normalised score (0–1),
    a list of matched pattern labels, and a boolean ``flagged``
    field that is ``True`` when ``score >= 0.3``.
    """
    total_weight = 0.0
    matches: list[str] = []

    for pattern, weight, label in _INJECTION_PATTERNS:
        if pattern.search(text):
            total_weight += weight
            matches.append(label)

    # Clamp to [0, 1]
    score = min(total_weight, 1.0)
    flagged = score >= 0.3

    if flagged:
        logger.warning("Injection scan flagged (score=%.2f, matches=%s)", score, matches)

    return ScanResult(score=score, matches=matches, flagged=flagged)
