from __future__ import annotations


def classify_failure(*, error: str | None, ragflow_answer: str | None, exact_match: bool, source_recall: float) -> str:
    if error:
        return "error"
    if not ragflow_answer:
        return "empty_response"
    if exact_match:
        return "correct"
    if source_recall == 0.0:
        return "retrieval_failure"
    if ragflow_answer and not ragflow_answer.strip():
        return "format_failure"
    return "reasoning_failure"
