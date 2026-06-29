from __future__ import annotations

from typing import Any

_NOT_FOUND_MARKERS = (
    "not found",
    "no relevant",
    "unable to find",
    "does not contain",
    "not available",
    "cannot find",
    "can't find",
    "no information",
    "not in the provided",
)


def classify_failure(
    *,
    error: str | None,
    ragflow_answer: str | None,
    exact_match: bool,
    source_recall: float | None = None,
    raw_retrieval_shard_recall: float | None = None,
    final_context_shard_recall: float | None = None,
    citation_recall: float | None = None,
    judge_verdict: str | None = None,
    judge_excluded: bool | None = None,
    judge_exclusion_reason: str | None = None,
) -> str:
    """Classify benchmark failures using retrieval/context diagnostics when available."""
    if error:
        return "answer_generation_error"
    if judge_excluded and judge_exclusion_reason == "judge_error":
        return "judge_error"
    if not ragflow_answer or not str(ragflow_answer).strip():
        return "answer_generation_error"
    if exact_match:
        return "exact_match_correct"

    verdict = str(judge_verdict or "").strip().lower()
    raw_recall = _coalesce_float(raw_retrieval_shard_recall, source_recall, 0.0)
    final_recall = _coalesce_float(final_context_shard_recall, 0.0)
    citations_hit = citation_recall is not None and citation_recall > 0.0
    says_not_found = _looks_not_found(ragflow_answer)

    if says_not_found and (final_recall > 0.0 or citations_hit):
        return "not_found_false_negative"
    if raw_recall <= 0.0:
        return "expected_source_absent_from_raw_retrieval"
    if final_recall <= 0.0:
        if verdict == "correct":
            return "judge_correct_but_source_unverified"
        if not citations_hit:
            return "expected_source_present_raw_absent_final_context"
    if verdict == "correct":
        return "judge_correct_but_source_unverified"
    if verdict == "partial":
        return "expected_source_present_final_context_answer_partial"
    if verdict == "incorrect":
        if says_not_found:
            return "not_found_false_negative" if final_recall > 0.0 else "expected_source_present_final_context_answer_wrong"
        return "expected_source_present_final_context_answer_wrong" if final_recall > 0.0 else "possible_distractor_answer"
    if final_recall > 0.0:
        return "expected_source_present_final_context_answer_wrong"
    return "unknown_unclassified"


def classify_row(row: dict[str, Any]) -> str:
    return classify_failure(
        error=row.get("error"),
        ragflow_answer=row.get("ragflow_answer"),
        exact_match=bool(row.get("exact_match")),
        source_recall=row.get("source_recall"),
        raw_retrieval_shard_recall=row.get("raw_retrieval_shard_recall@20"),
        final_context_shard_recall=row.get("final_context_shard_recall@top_n"),
        citation_recall=row.get("citation_recall"),
        judge_verdict=row.get("judge_verdict"),
        judge_excluded=row.get("judge_excluded"),
        judge_exclusion_reason=row.get("judge_exclusion_reason"),
    )


def _coalesce_float(*values: Any) -> float:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _looks_not_found(answer: str | None) -> bool:
    text = str(answer or "").lower()
    return any(marker in text for marker in _NOT_FOUND_MARKERS)
