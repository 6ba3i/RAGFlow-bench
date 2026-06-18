from __future__ import annotations

import logging
import os
import re
from typing import Iterable

from rich.logging import RichHandler

_SECRET_PATTERNS = [
    re.compile(r"(Bearer\s+)([A-Za-z0-9._\-]{12,})", re.IGNORECASE),
    re.compile(r"(RAGFLOW_API_KEY=)([^\s]+)"),
]


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
