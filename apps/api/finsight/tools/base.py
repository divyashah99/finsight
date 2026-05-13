"""Tool abstraction + reliability decorators.

Why we wrap MCP/HTTP tools with our own decorators:

- **Caching** sits in front of the tool so cache hits never burn an MCP call
  (or an API quota). Keyed on tool name + canonical args.
- **Rate limiting** uses the persisted token bucket — survives restarts so we
  don't accidentally blow the Alpha Vantage daily quota during dev.
- **Retry** wraps the *underlying* call only, not cache hits. Exponential
  backoff via tenacity on transient errors (5xx, network).

Decorator order (outermost first):
    cached -> rate_limited -> retry -> raw call
This makes cache hits free and ensures retries don't double-spend tokens.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from finsight.logging_setup import get_logger
from finsight.services import cache, rate_limit

log = get_logger(__name__)


# ─── Result envelope ────────────────────────────────────────────────────────


@dataclass
class ToolResult:
    """Uniform envelope for every tool call.

    `ok=False` lets agents decide whether to degrade gracefully (e.g. continue
    without news data) rather than crashing the whole graph.
    """

    ok: bool
    data: Any = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, data: Any, **meta: Any) -> "ToolResult":
        return cls(ok=True, data=data, meta=meta)

    @classmethod
    def failure(cls, error: str, **meta: Any) -> "ToolResult":
        return cls(ok=False, error=error, meta=meta)


class Tool(Protocol):
    """Every external integration implements this single async method.

    Args are kwargs only so the cache key derivation is unambiguous.
    """

    name: str

    async def call(self, **kwargs: Any) -> ToolResult: ...


# ─── Decorators ─────────────────────────────────────────────────────────────


def _canonical_key(prefix: str, kwargs: dict[str, Any]) -> str:
    """Deterministic cache key from sorted kwargs."""
    blob = json.dumps(kwargs, sort_keys=True, default=str)
    digest = hashlib.sha256(blob.encode()).hexdigest()[:16]
    return f"{prefix}:{digest}"


def cached(
    prefix: str, ttl_seconds: int
) -> Callable[
    [Callable[..., Awaitable[ToolResult]]], Callable[..., Awaitable[ToolResult]]
]:
    def deco(fn: Callable[..., Awaitable[ToolResult]]) -> Callable[..., Awaitable[ToolResult]]:
        async def wrapper(*args: Any, **kwargs: Any) -> ToolResult:
            key = _canonical_key(prefix, kwargs)
            hit = await cache.get(key)
            if hit is not None:
                return ToolResult(ok=True, data=hit, meta={"cache": "hit", "key": key})

            result = await fn(*args, **kwargs)
            if result.ok:
                await cache.set(key, result.data, ttl_seconds)
                result.meta["cache"] = "miss"
                result.meta["key"] = key
            return result

        return wrapper

    return deco


def rate_limited(
    provider: str, per_minute: int, per_day: int
) -> Callable[
    [Callable[..., Awaitable[ToolResult]]], Callable[..., Awaitable[ToolResult]]
]:
    def deco(fn: Callable[..., Awaitable[ToolResult]]) -> Callable[..., Awaitable[ToolResult]]:
        async def wrapper(*args: Any, **kwargs: Any) -> ToolResult:
            try:
                await rate_limit.acquire(provider, per_minute, per_day)
            except rate_limit.RateLimitExceeded as e:
                return ToolResult.failure(f"rate_limited: {e}")
            return await fn(*args, **kwargs)

        return wrapper

    return deco


def with_retry(
    attempts: int = 3,
) -> Callable[
    [Callable[..., Awaitable[ToolResult]]], Callable[..., Awaitable[ToolResult]]
]:
    """Retry on transient HTTP / network errors only.

    We do NOT retry on 4xx — those represent permanent client errors (bad
    ticker, bad API key) and retrying just burns rate-limit tokens.
    """

    def deco(fn: Callable[..., Awaitable[ToolResult]]) -> Callable[..., Awaitable[ToolResult]]:
        async def wrapper(*args: Any, **kwargs: Any) -> ToolResult:
            retryer = AsyncRetrying(
                reraise=True,
                stop=stop_after_attempt(attempts),
                wait=wait_exponential(multiplier=1, min=1, max=10),
                retry=retry_if_exception_type(
                    (httpx.TransportError, httpx.HTTPStatusError, httpx.RemoteProtocolError)
                ),
            )
            try:
                async for attempt in retryer:
                    with attempt:
                        result = await fn(*args, **kwargs)
                        if not result.ok and result.error and "5" in (result.meta.get("status", "")):
                            raise httpx.HTTPStatusError(
                                result.error, request=None, response=None  # type: ignore[arg-type]
                            )
                        return result
            except Exception as e:  # noqa: BLE001
                return ToolResult.failure(f"retry_exhausted: {e}")
            return ToolResult.failure("retry: unreachable")

        return wrapper

    return deco
