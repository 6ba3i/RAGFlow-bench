from __future__ import annotations

import re


def exact_match(gold: str | None, pred: str | None) -> bool:
    return bool(gold is not None and pred is not None and gold == pred)


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s]", "", value)
    return value.strip()


def normalized_match(gold: str | None, pred: str | None) -> bool:
    if gold is None or pred is None:
        return False
    return normalize_text(gold) == normalize_text(pred)
