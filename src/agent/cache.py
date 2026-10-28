"""Disk-backed cache wrapper for agent tools.

Uses ``diskcache`` with SHA-256 keys derived from (tool_name + args_json).
Default TTL is 24 hours. Cache is stored in ``.cache/`` under the project root.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import diskcache

logger = logging.getLogger(__name__)

# Default TTL: 24 hours in seconds
DEFAULT_TTL_S: int = 86_400

# Cache directory relative to the project root
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cache"


class DiskCacheWrapper:
    """Thin wrapper around :pymod:`diskcache` with deterministic SHA-256 keys.

    Parameters
    ----------
    directory:
        Path to the cache directory.  Defaults to ``<project_root>/.cache/``.
    default_ttl_s:
        Default time-to-live for entries, in seconds.
    """

    def __init__(
        self,
        directory: str | Path | None = None,
        default_ttl_s: int = DEFAULT_TTL_S,
    ) -> None:
        self._dir = Path(directory) if directory else _CACHE_DIR
        self._ttl = default_ttl_s
        self._cache = diskcache.Cache(str(self._dir))

    # ------------------------------------------------------------------
    # Key derivation
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(tool_name: str, args_json: str) -> str:
        """Return a SHA-256 hex digest of ``tool_name + args_json``."""
        raw = f"{tool_name}:{args_json}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, tool_name: str, args_json: str) -> Any | None:
        """Look up a cached value.  Returns ``None`` on miss."""
        key = self.make_key(tool_name, args_json)
        value = self._cache.get(key, default=None)
        if value is not None:
            logger.debug("Cache HIT for %s (key=%s…)", tool_name, key[:12])
        return value

    def set(
        self,
        tool_name: str,
        args_json: str,
        value: Any,
        ttl_s: int | None = None,
    ) -> None:
        """Store a value with a TTL (defaults to ``default_ttl_s``)."""
        key = self.make_key(tool_name, args_json)
        self._cache.set(key, value, expire=ttl_s or self._ttl)
        logger.debug("Cache SET for %s (key=%s…, ttl=%ds)", tool_name, key[:12], ttl_s or self._ttl)

    def clear(self) -> None:
        """Purge all cached entries."""
        self._cache.clear()
        logger.info("Cache cleared (%s)", self._dir)

    def close(self) -> None:
        """Close the underlying diskcache database."""
        self._cache.close()
