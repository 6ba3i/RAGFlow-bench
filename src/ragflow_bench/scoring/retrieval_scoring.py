from __future__ import annotations


def source_recall(expected_sources: list[str], retrieved_sources: list[str]) -> float:
    if not expected_sources:
        return 0.0
    expected = set(expected_sources)
    retrieved = set(retrieved_sources)
    return len(expected & retrieved) / len(expected)
