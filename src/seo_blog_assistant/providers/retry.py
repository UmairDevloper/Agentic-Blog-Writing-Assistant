"""Reusable asynchronous retry support for provider adapters."""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class RetryDecision:
    """Retry timing selected by a provider-specific classifier."""

    delay_seconds: float | None = None


async def retry_async(
    operation: Callable[[], Awaitable[ResultT]],
    *,
    max_attempts: int,
    classify: Callable[[Exception], RetryDecision | None],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    logger: logging.Logger,
    label: str,
) -> ResultT:
    """Retry a provider operation when its classifier marks failures retryable."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")

    for attempt in range(1, max_attempts + 1):
        try:
            return await operation()
        except Exception as exc:
            decision = classify(exc)
            if decision is None or attempt == max_attempts:
                raise

            delay_seconds = decision.delay_seconds
            if delay_seconds is None:
                delay_seconds = min(30, 2 ** (attempt - 1)) + random.uniform(0, 1)
            logger.warning(
                "Retrying provider operation label=%s attempt=%s/%s "
                "delay_seconds=%.2f error_type=%s",
                label,
                attempt,
                max_attempts,
                delay_seconds,
                type(exc).__name__,
            )
            await sleep(delay_seconds)

    raise RuntimeError("retry loop exited without returning or raising")
