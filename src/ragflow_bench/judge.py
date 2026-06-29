from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import requests

from ragflow_bench.config import JudgeSettings
from ragflow_bench.logging_utils import redact_text
from ragflow_bench.rate_limits import run_with_rate_limit_retries
from ragflow_bench.ragflow.errors import RagflowAPIError, RagflowConfigError
from ragflow_bench.reports.writers import jsonl_to_csv, write_json, write_jsonl

EXCLUDED_INFRA_ERROR_MARKERS = (
    "**ERROR**:",
    "litellm.RateLimitError",
    "RateLimitError",
    "Read timed out",
    "timeout",
    "Timeout",
    "GENERIC_ERROR",
    "ConnectionError",
    "HTTPConnectionPool",
    "API error",
)

SCORE_TO_VERDICT = {
    4: "correct",
    2: "partial",
    0: "incorrect",
}
QUESTION_DELAY_MIN_SECONDS = 5.0
QUESTION_DELAY_MAX_SECONDS = 10.0


def is_excluded_infra_error(row: dict[str, Any]) -> bool:
    text = f"{row.get('error') or ''}\n{row.get('ragflow_answer') or ''}"
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in EXCLUDED_INFRA_ERROR_MARKERS)


def _normalize_score(value: Any) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 0
    return score if score in SCORE_TO_VERDICT else 0


def _normalize_confidence(value: Any) -> float | None:
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    if not isinstance(value, (int, float)):
        return None
    return max(0.0, min(1.0, float(value)))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def build_judge_messages(row: dict[str, Any], *, include_confidence: bool = True) -> list[dict[str, str]]:
    confidence_note = '"confidence": number between 0 and 1, ' if include_confidence else ""
    system = (
        "You are a strict evaluator for FRAMES-style QA results. "
        "Compare the candidate answer against the reference answer for semantic correctness, "
        "not exact wording. Return JSON only."
    )
    user = (
        "Evaluate whether the candidate answer correctly answers the question.\n\n"
        "Use this fixed 0/2/4 rule book:\n"
        "- 4 = fully correct: the candidate contains the same essential factual answer as the reference.\n"
        "- 2 = partially correct: the candidate includes some required facts but is incomplete, vague, "
        "or missing required disambiguating detail.\n"
        "- 0 = incorrect: the candidate gives a different entity, date, number, location, title, relation, "
        "cause, event, contradicts the reference, is irrelevant, refuses to answer, or is a non-answer.\n"
        "Accept paraphrases, equivalent date/number formats, aliases, reordered wording, and harmless extra context.\n"
        "Contradictory extra context prevents a score of 4.\n"
        "For multi-part answers, all required parts must be present for score 4.\n"
        "Judge only final answer correctness; do not reward retrieval quality or citations.\n\n"
        f"Question:\n{row.get('question') or ''}\n\n"
        f"Reference answer:\n{row.get('gold_answer') or ''}\n\n"
        f"Candidate answer:\n{row.get('ragflow_answer') or ''}\n\n"
        "Respond with a JSON object with keys: "
        f'"score" (one of 0, 2, 4), "verdict" (one of "correct", "partial", "incorrect"), {confidence_note}'
        '"reason", "matched_facts" (array of strings), "missing_or_wrong_facts" (array of strings).'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def summarize_judged_rows(*, rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    total = len(rows)
    scorable_rows = [row for row in rows if not row.get("judge_excluded") and isinstance(row.get("judge_score"), int)]
    scorable = len(scorable_rows)
    excluded_infra = sum(1 for row in rows if row.get("judge_exclusion_reason") == "infra_error")
    judge_errors = sum(1 for row in rows if row.get("judge_exclusion_reason") == "judge_error")
    excluded = sum(1 for row in rows if row.get("judge_excluded"))
    strict = sum(1 for row in scorable_rows if row.get("judge_score") == 4)
    partial_or_better = sum(1 for row in scorable_rows if int(row.get("judge_score", 0)) >= 2)
    score_sum = sum(float(row.get("judge_score_normalized", 0.0)) for row in scorable_rows)
    score_counts = Counter(str(row.get("judge_score")) for row in scorable_rows)
    verdict_counts = Counter(row.get("judge_verdict", "") for row in rows)
    confidences = [float(row["judge_confidence"]) for row in scorable_rows if isinstance(row.get("judge_confidence"), (int, float))]
    strict_accuracy = strict / scorable if scorable else 0.0
    return {
        "judge_provider": "zhipu",
        "judge_model": model,
        "total_questions": total,
        "scorable_questions": scorable,
        "excluded_questions": excluded,
        "excluded_infra_questions": excluded_infra,
        "judge_error_questions": judge_errors,
        "excluded_error_rate": (excluded / total if total else 0.0),
        "answer_accuracy": (score_sum / scorable if scorable else 0.0),
        "strict_accuracy": strict_accuracy,
        "partial_or_better_accuracy": (partial_or_better / scorable if scorable else 0.0),
        "judge_accuracy": strict_accuracy,
        "judge_score_counts": dict(score_counts),
        "judge_verdict_counts": dict(verdict_counts),
        "average_judge_confidence": (sum(confidences) / len(confidences) if confidences else None),
    }

class ZhipuJudgeClient:
    def __init__(self, settings: JudgeSettings, timeout: int | None = None):
        self.settings = settings
        self.base_url = settings.resolved_base_url().rstrip("/")
        self.api_key = settings.resolved_api_key()
        self.model = settings.resolved_model()
        self.temperature = settings.temperature
        self.timeout = timeout if timeout is not None else settings.resolved_timeout_seconds()
        self.max_retries = settings.resolved_max_retries()
        self.backoff_seconds = settings.resolved_backoff_seconds()
        self.max_backoff_seconds = settings.resolved_max_backoff_seconds()
        self.session = requests.Session()
        self.progress_callback = None
        if not self.base_url:
            raise RagflowConfigError("Judge base URL is required")
        if not self.api_key:
            raise RagflowConfigError("Judge API key is required")
        if not self.model:
            raise RagflowConfigError("Judge model is required")

    def _backoff_delay(self, attempt: int, response: requests.Response | None = None) -> float:
        retry_after = None
        if response is not None:
            retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = None
            else:
                return min(max(delay, 0.0), self.max_backoff_seconds)
        return min(self.backoff_seconds, self.max_backoff_seconds)

    @staticmethod
    def _should_retry_http(status_code: int | None) -> bool:
        return status_code == 429 or (status_code is not None and status_code >= 500)

    def _emit_log(self, event: dict[str, Any]) -> None:
        callback = getattr(self, "progress_callback", None)
        if callback is not None:
            callback(event)

    def _post_with_retries(self, payload: dict[str, Any], *, question_id: str | None = None) -> requests.Response:
        def _request_once() -> requests.Response:
            started = time.monotonic()
            try:
                response = self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.ReadTimeout as exc:
                elapsed = time.monotonic() - started
                self._emit_log({
                    "type": "judge_request",
                    "question_id": question_id,
                    "exception": exc.__class__.__name__,
                    "retry": False,
                    "elapsed_seconds": elapsed,
                    "error": str(exc),
                })
                raise
            elapsed = time.monotonic() - started
            if response.status_code < 400:
                self._emit_log({
                    "type": "judge_request",
                    "question_id": question_id,
                    "status_code": response.status_code,
                    "retry": False,
                    "elapsed_seconds": elapsed,
                })
                return response
            error = RagflowAPIError(
                f"Judge HTTP {response.status_code}",
                status_code=response.status_code,
                url=f"{self.base_url}/chat/completions",
                raw_body=redact_text(response.text, [self.api_key]),
            )
            self._emit_log({
                "type": "judge_request",
                "question_id": question_id,
                "status_code": response.status_code,
                "retry": False,
                "elapsed_seconds": elapsed,
                "error": str(error),
                "raw_body": redact_text(response.text, [self.api_key]),
            })
            raise error

        response = run_with_rate_limit_retries(
            _request_once,
            action_type="judge",
            question_id=question_id,
            progress_callback=self.progress_callback,
        )
        return response

    def judge_row(self, row: dict[str, Any]) -> dict[str, Any]:
        if is_excluded_infra_error(row):
            reason = str(row.get("error") or row.get("ragflow_answer") or "Infrastructure/provider error")
            return {
                "score": None,
                "score_normalized": None,
                "verdict": "excluded",
                "reason": reason,
                "confidence": None,
                "matched_facts": [],
                "missing_or_wrong_facts": [],
                "excluded": True,
                "exclusion_reason": "infra_error",
            }
        if not row.get("ragflow_answer"):
            return {
                "score": 0,
                "score_normalized": 0.0,
                "verdict": "incorrect",
                "reason": "No candidate answer to evaluate",
                "confidence": 1.0,
                "matched_facts": [],
                "missing_or_wrong_facts": ["No candidate answer was provided."],
                "excluded": False,
                "exclusion_reason": None,
            }

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": build_judge_messages(row, include_confidence=self.settings.include_confidence),
        }
        response = self._post_with_retries(payload, question_id=str(row.get("question_id") or ""))
        body = response.json()
        choice = ((body.get("choices") or [{}])[0]).get("message", {})
        content = choice.get("content") or "{}"
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RagflowAPIError(
                "Judge did not return valid JSON",
                raw_body=redact_text(content, [self.api_key]),
            ) from exc
        score = _normalize_score(parsed.get("score"))
        verdict = str(parsed.get("verdict") or SCORE_TO_VERDICT[score]).strip().lower()
        expected_verdict = SCORE_TO_VERDICT[score]
        if verdict not in {"correct", "partial", "incorrect"} or verdict != expected_verdict:
            verdict = expected_verdict
        return {
            "score": score,
            "score_normalized": score / 4,
            "verdict": verdict,
            "reason": str(parsed.get("reason") or "").strip(),
            "confidence": _normalize_confidence(parsed.get("confidence")),
            "matched_facts": _string_list(parsed.get("matched_facts")),
            "missing_or_wrong_facts": _string_list(parsed.get("missing_or_wrong_facts")),
            "excluded": False,
            "exclusion_reason": None,
        }


def _format_elapsed(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _make_judged_row(row: dict[str, Any], *, client: ZhipuJudgeClient, verdict: dict[str, Any]) -> dict[str, Any]:
    judged = dict(row)
    judged["judge_model"] = client.model
    judged["judge_score"] = verdict["score"]
    judged["judge_score_normalized"] = verdict["score_normalized"]
    judged["judge_verdict"] = verdict["verdict"]
    judged["judge_reason"] = verdict["reason"]
    judged["judge_confidence"] = verdict["confidence"]
    judged["judge_matched_facts"] = verdict["matched_facts"]
    judged["judge_missing_or_wrong_facts"] = verdict["missing_or_wrong_facts"]
    judged["judge_excluded"] = verdict["excluded"]
    judged["judge_exclusion_reason"] = verdict["exclusion_reason"]
    judged["judge_correct"] = verdict["score"] == 4
    return judged


def _judge_error_verdict(exc: Exception) -> dict[str, Any]:
    return {
        "score": None,
        "score_normalized": None,
        "verdict": "judge_error",
        "reason": str(exc),
        "confidence": None,
        "matched_facts": [],
        "missing_or_wrong_facts": [],
        "excluded": True,
        "exclusion_reason": "judge_error",
    }


def _load_existing_judged_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RagflowAPIError(f"Malformed existing judge output at {path}:{line_number}", raw_body=line) from exc
    return rows


def _emit_progress(progress_callback: Callable[[dict[str, Any]], None] | None, event: dict[str, Any]) -> None:
    if progress_callback is not None:
        progress_callback(event)


def _sleep_between_judge_rows() -> float:
    delay = random.uniform(QUESTION_DELAY_MIN_SECONDS, QUESTION_DELAY_MAX_SECONDS)
    time.sleep(delay)
    return delay


def _row_resume_key(row: dict[str, Any], index: int) -> str:
    question_id = row.get("question_id")
    if question_id is None or str(question_id) == "":
        return f"__row_index_{index}"
    return str(question_id)


def default_progress_printer(event: dict[str, Any]) -> None:
    if event.get("type") == "judge_request":
        parts = [
            "judge_request",
            f"qid={event.get('question_id')}",
            f"attempt={event.get('attempt')}/{event.get('max_attempts')}",
        ]
        if event.get("status_code") is not None:
            parts.append(f"status={event.get('status_code')}")
        if event.get("exception"):
            parts.append(f"exception={event.get('exception')}")
        parts.append(f"retry={str(bool(event.get('retry'))).lower()}")
        if event.get("delay") is not None:
            parts.append(f"delay={event.get('delay')}")
        if event.get("error"):
            parts.append(f"error={json.dumps(str(event.get('error'))[:300], ensure_ascii=False)}")
        print(" ".join(parts), file=sys.stderr, flush=True)
        return

    if event.get("type") == "judge_row":
        parts = [
            f"[{event.get('index')}/{event.get('total')}]",
            f"qid={event.get('question_id')}",
            f"status={event.get('status')}",
        ]
        if event.get("verdict"):
            parts.append(f"verdict={event.get('verdict')}")
        if event.get("score") is not None:
            parts.append(f"score={event.get('score')}")
        if event.get("score_normalized") is not None:
            parts.append(f"norm={float(event.get('score_normalized')):.2f}")
        if event.get("reason"):
            parts.append(f"reason={json.dumps(str(event.get('reason'))[:180], ensure_ascii=False)}")
        parts.extend([
            f"scorable={event.get('scorable_count')}",
            f"excluded={event.get('excluded_count')}",
            f"elapsed={_format_elapsed(float(event.get('elapsed_seconds') or 0.0))}",
        ])
        print(" ".join(parts), file=sys.stderr, flush=True)


def judge_results_file(
    *,
    results_path: str | Path,
    client: ZhipuJudgeClient,
    output_path: str | Path | None = None,
    resume: bool = True,
    force_question_ids: set[str] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    input_path = Path(results_path)
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    target = Path(output_path) if output_path else input_path.with_name("judge_results.jsonl")
    forced_ids = {str(question_id) for question_id in (force_question_ids or set())}

    existing_rows = _load_existing_judged_rows(target) if resume else []
    judged_by_id = {_row_resume_key(row, index): row for index, row in enumerate(existing_rows, start=1)}
    started = time.monotonic()
    previous_callback = getattr(client, "progress_callback", None)
    client.progress_callback = progress_callback

    mode = "a" if resume and target.exists() and not forced_ids else "w"
    try:
        with target.open(mode, encoding="utf-8") as handle:
            for index, row in enumerate(rows, start=1):
                question_id = str(row.get("question_id") or "")
                resume_key = _row_resume_key(row, index)
                should_force = bool(forced_ids) and resume_key in forced_ids
                if resume and resume_key in judged_by_id and not should_force:
                    existing = judged_by_id[resume_key]
                    _emit_progress(progress_callback, {
                        "type": "judge_row",
                        "index": index,
                        "total": len(rows),
                        "question_id": question_id,
                        "status": "skipped",
                        "verdict": existing.get("judge_verdict"),
                        "score": existing.get("judge_score"),
                        "score_normalized": existing.get("judge_score_normalized"),
                        "scorable_count": sum(1 for item in judged_by_id.values() if not item.get("judge_excluded") and isinstance(item.get("judge_score"), int)),
                        "excluded_count": sum(1 for item in judged_by_id.values() if item.get("judge_excluded")),
                        "elapsed_seconds": time.monotonic() - started,
                    })
                    if index < len(rows):
                        delay = _sleep_between_judge_rows()
                        _emit_progress(progress_callback, {
                            "type": "judge_row_delay",
                            "index": index,
                            "total": len(rows),
                            "question_id": question_id,
                            "delay": delay,
                        })
                    continue

                try:
                    verdict = client.judge_row(row)
                except Exception as exc:  # noqa: BLE001
                    verdict = _judge_error_verdict(exc)
                judged = _make_judged_row(row, client=client, verdict=verdict)
                judged_by_id[resume_key] = judged
                if not forced_ids:
                    handle.write(json.dumps(judged, ensure_ascii=False) + "\n")
                    handle.flush()
                status = "judge_error" if judged.get("judge_exclusion_reason") == "judge_error" else ("excluded" if judged.get("judge_excluded") else "judged")
                _emit_progress(progress_callback, {
                    "type": "judge_row",
                    "index": index,
                    "total": len(rows),
                    "question_id": question_id,
                    "status": status,
                    "verdict": judged.get("judge_verdict"),
                    "score": judged.get("judge_score"),
                    "score_normalized": judged.get("judge_score_normalized"),
                    "reason": judged.get("judge_exclusion_reason") or judged.get("judge_reason"),
                    "scorable_count": sum(1 for item in judged_by_id.values() if not item.get("judge_excluded") and isinstance(item.get("judge_score"), int)),
                    "excluded_count": sum(1 for item in judged_by_id.values() if item.get("judge_excluded")),
                    "elapsed_seconds": time.monotonic() - started,
                })
                if index < len(rows):
                    delay = _sleep_between_judge_rows()
                    _emit_progress(progress_callback, {
                        "type": "judge_row_delay",
                        "index": index,
                        "total": len(rows),
                        "question_id": question_id,
                        "delay": delay,
                    })
    finally:
        client.progress_callback = previous_callback

    final_rows = [judged_by_id[_row_resume_key(row, index)] for index, row in enumerate(rows, start=1) if _row_resume_key(row, index) in judged_by_id]
    if forced_ids:
        write_jsonl(target, final_rows)
    jsonl_to_csv(target, target.with_suffix(".csv"))
    summary = summarize_judged_rows(rows=final_rows, model=client.model)
    write_json(target.with_name("judge_summary.json"), summary)
    return summary
