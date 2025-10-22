"""Simple TTL cache for REFRAG embeddings and projections."""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Generic, TypeVar, cast

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

    async def get_or_set(self, key: Any, factory: Callable[[], T | Awaitable[T]]) -> T:
        if not self.config.store_embeddings:
            return await _maybe_await(factory())

        now = time.time()
        entry = self._values.get(key)
        if entry is not None:
            expires_at, value = entry
            if expires_at > now:
                return value

        value = await _maybe_await(factory())
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


async def _maybe_await(value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        awaited = await cast(Awaitable[T], value)
        return awaited
    return cast(T, value)


__all__ = ["CacheConfig", "RefragCache"]
