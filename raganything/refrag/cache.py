"""Simple TTL cache for REFRAG embeddings and projections."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, TypeVar

T = TypeVar("T")


@dataclass
class CacheConfig:
    """Configuration for :class:`RefragCache`."""

    ttl_seconds: int = 3600
    store_embeddings: bool = True


class RefragCache(Generic[T]):
    """A minimal in-memory cache with TTL semantics."""

    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self._values: Dict[Any, tuple[float, T]] = {}

    def get_or_set(self, key: Any, factory: Callable[[], T]) -> T:
        if not self.config.store_embeddings:
            return factory()

        now = time.time()
        entry = self._values.get(key)
        if entry is not None:
            expires_at, value = entry
            if expires_at > now:
                return value

        value = factory()
        expires_at = now + max(1, self.config.ttl_seconds)
        self._values[key] = (expires_at, value)
        return value

    def clear(self) -> None:
        self._values.clear()

    def prune_expired(self) -> None:
        now = time.time()
        expired = [key for key, (expires_at, _) in self._values.items() if expires_at <= now]
        for key in expired:
            self._values.pop(key, None)


__all__ = ["CacheConfig", "RefragCache"]
