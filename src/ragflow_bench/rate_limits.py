from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from ragflow_bench.ragflow.errors import RagflowAPIError

T = TypeVar("T")

RATE_LIMIT_DELAY_MIN_SECONDS = 15.0
RATE_LIMIT_DELAY_MAX_SECONDS = 20.0
RATE_LIMIT_MAX_RETRIES = 2
RATE_LIMIT_MARKERS = (
    "litellm.RateLimitError",
    "RateLimitError",
    "rate limited",
    '"code":"1305"',
    '"code":1305',
    "too many requests",
    "retry-after",
)


@dataclass
class InlineRateLimitError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def is_rate_limit_error(value: Any) -> bool:
    if isinstance(value, RagflowAPIError):
        if value.status_code == 429:
            return True
        haystacks = [str(value), value.raw_body or ""]
    elif isinstance(value, BaseException):
        haystacks = [str(value)]
    else:
        haystacks = [str(value or "")]
    lowered = "\n".join(haystacks).lower()
    return any(marker.lower() in lowered for marker in RATE_LIMIT_MARKERS)


def sleep_before_rate_limit_retry() -> float:
    delay = random.uniform(RATE_LIMIT_DELAY_MIN_SECONDS, RATE_LIMIT_DELAY_MAX_SECONDS)
    time.sleep(delay)
    return delay


def run_with_rate_limit_retries(
    action: Callable[[], T],
    *,
    action_type: str,
    question_id: str | None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    should_retry_result: Callable[[T], str | None] | None = None,
    max_retries: int = RATE_LIMIT_MAX_RETRIES,
) -> T:
    max_attempts = max_retries + 1
    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = action()
            retry_reason = should_retry_result(result) if should_retry_result is not None else None
            if retry_reason:
                raise InlineRateLimitError(retry_reason)
            if progress_callback is not None and attempt > 1:
                progress_callback({
                    "type": "rate_limit_retry",
                    "action": action_type,
                    "question_id": question_id,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "status": "recovered",
                    "retry": False,
                })
            return result
        except BaseException as exc:  # noqa: BLE001
            if not is_rate_limit_error(exc):
                raise
            last_error = exc
            retry = attempt <= max_retries
            delay = sleep_before_rate_limit_retry() if retry else None
            if progress_callback is not None:
                progress_callback({
                    "type": "rate_limit_retry",
                    "action": action_type,
                    "question_id": question_id,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "status": "retrying" if retry else "exhausted",
                    "retry": retry,
                    "delay": delay,
                    "error": str(exc),
                })
            if not retry:
                raise
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Rate-limit retry loop ended unexpectedly for {action_type}")
