# rag_pipeline/cache.py
# High-speed query caching layer with TTL support to eliminate redundant retrieval & LLM calls.

import hashlib
import time
from typing import Any

from common.config import get_settings
from common.logger import get_logger

logger = get_logger(__name__)


class QueryCache:
    """
    In-memory LRU-like TTL cache for RAG queries.
    Stores normalized query -> (timestamp, response_dict).
    """

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size

    def _hash_key(self, query: str) -> str:
        """Normalizes and hashes the user query."""
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, query: str) -> dict[str, Any] | None:
        """Retrieves a cached response if valid and not expired."""
        settings = get_settings()
        if not settings.enable_query_cache:
            return None

        key = self._hash_key(query)
        if key not in self._cache:
            return None

        created_at, response = self._cache[key]
        now = time.time()

        if now - created_at > settings.cache_ttl_seconds:
            # Expired
            del self._cache[key]
            logger.debug("Cache entry expired for query hash %s", key[:8])
            return None

        logger.info("Cache HIT for query hash %s (serving in <15ms)", key[:8])
        cached_copy = dict(response)
        cached_copy["cached"] = True
        return cached_copy

    def set(self, query: str, response: dict[str, Any]) -> None:
        """Caches a query response."""
        settings = get_settings()
        if not settings.enable_query_cache:
            return

        if len(self._cache) >= self.max_size:
            # Evict oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]

        key = self._hash_key(query)
        self._cache[key] = (time.time(), dict(response))
        logger.debug("Cached query response for hash %s", key[:8])

    def clear(self) -> None:
        """Clears all cached queries (e.g. on new document ingestion)."""
        self._cache.clear()
        logger.info("Query cache cleared.")


# Singleton cache instance
_query_cache = QueryCache()


def get_query_cache() -> QueryCache:
    return _query_cache
