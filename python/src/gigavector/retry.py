"""Shared retry helpers for ingestion and external I/O."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


@dataclass
class RetryPolicy:
    """Retry transient failures with optional exponential backoff."""

    max_retries: int = 2
    base_delay: float = 0.1
    max_delay: float = 2.0
    exponential: bool = True

    @property
    def attempts(self) -> int:
        return max(1, self.max_retries + 1)

    @classmethod
    def for_vector(cls) -> RetryPolicy:
        return cls(max_retries=2, base_delay=0.05, max_delay=1.0)

    @classmethod
    def for_graph(cls) -> RetryPolicy:
        return cls(max_retries=2, base_delay=0.05, max_delay=1.0)

    @classmethod
    def for_network(cls) -> RetryPolicy:
        return cls(max_retries=3, base_delay=0.5, max_delay=8.0)


# Defaults used when callers do not pass an explicit policy.
VECTOR_RETRY = RetryPolicy.for_vector()
GRAPH_RETRY = RetryPolicy.for_graph()
NETWORK_RETRY = RetryPolicy.for_network()


def is_retryable(exc: BaseException) -> bool:
    """Return False for validation errors; retry most runtime/I/O failures."""
    if isinstance(exc, (ValueError, TypeError, KeyError, AttributeError)):
        return False
    return True


def call_with_retry(
    fn: Callable[[], T],
    policy: Optional[RetryPolicy] = None,
    *,
    operation: str = "operation",
    retry_if: Optional[Callable[[BaseException], bool]] = None,
) -> T:
    """Call ``fn`` up to ``policy.attempts`` times with backoff between retries."""
    active = policy or RetryPolicy(max_retries=0)
    should_retry = retry_if or is_retryable
    last: BaseException | None = None

    for attempt in range(active.attempts):
        try:
            return fn()
        except BaseException as exc:
            last = exc
            if not should_retry(exc) or attempt + 1 >= active.attempts:
                raise
            delay = active.base_delay
            if active.exponential:
                delay = min(active.max_delay, active.base_delay * (2 ** attempt))
            if delay > 0:
                time.sleep(delay)

    if last is not None:
        raise last
    raise RuntimeError(f"{operation} failed")
