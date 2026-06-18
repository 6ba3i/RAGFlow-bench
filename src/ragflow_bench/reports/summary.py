from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable


def build_summary(*, benchmark: str, mode: str, rows: list[dict]) -> dict:
    total = len(rows)
    completed = sum(1 for row in rows if not row.get("error"))
    errors = sum(1 for row in rows if row.get("error"))
    exact = sum(1 for row in rows if row.get("exact_match"))
    normalized = sum(1 for row in rows if row.get("normalized_match"))
    avg_source_recall = sum(float(row.get("source_recall", 0.0)) for row in rows) / total if total else 0.0
    failures = Counter(row.get("failure_type", "") for row in rows)
    reasoning = defaultdict(lambda: {"count": 0, "exact_match": 0})
    for row in rows:
        for reason in row.get("reasoning_types", []):
            reasoning[reason]["count"] += 1
            reasoning[reason]["exact_match"] += int(bool(row.get("exact_match")))
    reasoning_accuracy = {
        key: {"count": value["count"], "exact_match_accuracy": (value["exact_match"] / value["count"] if value["count"] else 0.0)}
        for key, value in reasoning.items()
    }
    return {
        "benchmark": benchmark,
        "mode": mode,
        "total_questions": total,
        "completed_questions": completed,
        "errors": errors,
        "exact_match_accuracy": exact / total if total else 0.0,
        "normalized_match_accuracy": normalized / total if total else 0.0,
        "average_source_recall": avg_source_recall,
        "failure_type_counts": dict(failures),
        "accuracy_by_reasoning_type": reasoning_accuracy,
    }
