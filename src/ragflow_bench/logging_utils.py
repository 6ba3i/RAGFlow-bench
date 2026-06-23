from __future__ import annotations

import json
import logging
import os
import re
import sys
from typing import Any, Callable, Iterable

from rich.logging import RichHandler

_SECRET_PATTERNS = [
    re.compile(r"(Bearer\s+)([A-Za-z0-9._\-]{12,})", re.IGNORECASE),
    re.compile(r"(RAGFLOW_API_KEY=)([^\s]+)"),
]

ProgressCallback = Callable[[dict[str, Any]], None]


def redact_text(value: str, extra_secrets: Iterable[str] | None = None) -> str:
    text = value
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(r"\1***REDACTED***", text)
    for secret in extra_secrets or []:
        if secret:
            text = text.replace(secret, "***REDACTED***")
    return text


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        return redact_text(rendered)


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_text(record.msg)
        if record.args:
            record.args = tuple(redact_text(str(arg)) for arg in record.args)
        return True


LOGGER_NAME = "ragflow_bench"


def configure_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose or os.getenv("RAGFLOW_BENCH_DEBUG") else logging.INFO
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        logger.setLevel(level)
        return logger
    handler = RichHandler(rich_tracebacks=True, show_path=False)
    handler.setFormatter(RedactingFormatter("%(message)s"))
    handler.addFilter(RedactingFilter())
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def emit_progress(progress_callback: ProgressCallback | None, event: dict[str, Any]) -> None:
    if progress_callback is not None:
        progress_callback(event)


def default_progress_printer(event: dict[str, Any]) -> None:
    line = format_progress_event(event)
    if line:
        print(line, file=sys.stderr, flush=True)


def format_progress_event(event: dict[str, Any]) -> str:
    command = event.get("command") or event.get("type") or "progress"
    step = event.get("step")
    parts = [str(command)]
    if step:
        parts.append(f"step={step}")
    for key in (
        "status",
        "index",
        "total",
        "question_id",
        "document_id",
        "dataset_id",
        "chat_id",
        "session_id",
        "path",
        "output_dir",
        "count",
        "replaced",
        "remaining_errors",
        "failure_count",
        "status_code",
        "exception",
    ):
        if event.get(key) is not None:
            parts.append(f"{key}={_format_progress_value(event[key])}")
    if event.get("retry") is not None:
        parts.append(f"retry={str(bool(event.get('retry'))).lower()}")
    if event.get("delay") is not None:
        parts.append(f"delay={event.get('delay')}")
    if event.get("elapsed_seconds") is not None:
        parts.append(f"elapsed={_format_elapsed(float(event.get('elapsed_seconds') or 0.0))}")
    if event.get("error"):
        parts.append(f"error={_format_progress_value(str(event.get('error'))[:300])}")
    return redact_text(" ".join(parts))


def _format_progress_value(value: Any) -> str:
    text = str(value)
    if any(ch.isspace() for ch in text) or text == "":
        return json.dumps(text, ensure_ascii=False)
    return text


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m{remainder:.0f}s"
