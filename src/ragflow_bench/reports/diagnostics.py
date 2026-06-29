from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ragflow_bench.execution.benchmark_runner import audit_chunks, citation_chunk_ids, final_context_chunks_from_response
from ragflow_bench.reports.writers import load_jsonl, write_json, write_jsonl
from ragflow_bench.scoring.answer_scoring import exact_match, normalized_match
from ragflow_bench.scoring.failure_classification import classify_failure
from ragflow_bench.scoring.retrieval_scoring import retrieval_diagnostics, source_recall


def recompute_run_diagnostics(run_dir: str | Path, *, output_prefix: str = "diagnostics") -> dict[str, Any]:
    run_path = Path(run_dir)
    results_path = run_path / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.jsonl: {results_path}")
    result_rows = load_jsonl(results_path)
    judge_rows = load_jsonl(run_path / "judge_results.jsonl")
    judge_by_id = {str(row.get("question_id")): row for row in judge_rows if row.get("question_id") is not None}

    rows: list[dict[str, Any]] = []
    for row in result_rows:
        joined = dict(row)
        judge = judge_by_id.get(str(row.get("question_id")))
        if judge:
            for key, value in judge.items():
                if key.startswith("judge_"):
                    joined[key] = value
        rows.append(recompute_row_diagnostics(joined))

    summary = summarize_diagnostic_rows(rows)
    output_json = run_path / f"{output_prefix}.json"
    output_md = run_path / f"{output_prefix}.md"
    output_rows = run_path / f"{output_prefix}.rows.jsonl"
    write_json(output_json, summary)
    write_jsonl(output_rows, rows)
    output_md.write_text(render_diagnostics_markdown(summary), encoding="utf-8")
    return {**summary, "output_json": str(output_json), "output_md": str(output_md), "output_rows_jsonl": str(output_rows)}


def recompute_row_diagnostics(row: dict[str, Any]) -> dict[str, Any]:
    raw_retrieval = row.get("raw_retrieval") if isinstance(row.get("raw_retrieval"), dict) else {}
    raw_chunks = raw_retrieval.get("chunks", []) if isinstance(raw_retrieval, dict) else []
    raw_chunks = raw_chunks if isinstance(raw_chunks, list) else []
    final_chunks = row.get("final_context_chunks_raw") or final_context_chunks_from_response(row.get("raw_response") if isinstance(row.get("raw_response"), dict) else {})
    if not final_chunks and row.get("final_context_chunks"):
        final_chunks = row.get("final_context_chunks")
    final_chunks = final_chunks if isinstance(final_chunks, list) else []
    expected_sources = row.get("expected_sources") or []
    raw_diag = retrieval_diagnostics(expected_sources, raw_chunks, prefix="raw_retrieval")
    final_diag = retrieval_diagnostics(expected_sources, final_chunks, prefix="final_context")
    cited_ids = row.get("citation_chunk_ids") or citation_chunk_ids(row.get("ragflow_answer"), final_chunks)
    retrieved_source_uris = raw_diag["raw_retrieval_retrieved_source_uris"]
    exact = exact_match(row.get("gold_answer"), row.get("ragflow_answer"))
    normalized = normalized_match(row.get("gold_answer"), row.get("ragflow_answer"))
    recall = source_recall(expected_sources, retrieved_source_uris)
    failure = classify_failure(
        error=row.get("error"),
        ragflow_answer=row.get("ragflow_answer"),
        exact_match=exact,
        source_recall=recall,
        raw_retrieval_shard_recall=raw_diag["raw_retrieval_shard_recall"],
        final_context_shard_recall=final_diag["final_context_shard_recall"],
        citation_recall=None,
        judge_verdict=row.get("judge_verdict"),
        judge_excluded=row.get("judge_excluded"),
        judge_exclusion_reason=row.get("judge_exclusion_reason"),
    )
    final_ids = {str(chunk.get("chunk_id") or chunk.get("id")) for chunk in final_chunks if isinstance(chunk, dict) and (chunk.get("chunk_id") or chunk.get("id"))}
    out = dict(row)
    out.update(
        {
            "exact_match": exact,
            "normalized_match": normalized,
            "retrieved_source_uris": retrieved_source_uris,
            "source_recall": recall,
            "raw_retrieval_shard_recall@20": raw_diag["raw_retrieval_shard_recall"],
            "raw_retrieval_expected_rank": raw_diag["raw_retrieval_expected_rank"],
            "raw_retrieval_mrr": raw_diag["raw_retrieval_mrr"],
            "final_context_shard_recall@top_n": final_diag["final_context_shard_recall"],
            "final_context_expected_rank": final_diag["final_context_expected_rank"],
            "final_context_mrr": final_diag["final_context_mrr"],
            "retrieval_diagnostics": {**raw_diag, **final_diag},
            "raw_retrieval_chunks": row.get("raw_retrieval_chunks") or audit_chunks(raw_chunks, final_chunk_ids=final_ids),
            "final_context_chunks": row.get("final_context_chunks") or audit_chunks(final_chunks),
            "citation_chunk_ids": cited_ids,
            "prompt_chunk_count": row.get("prompt_chunk_count", len(final_chunks)),
            "failure_type": failure,
        }
    )
    return out


def summarize_diagnostic_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    scorable = [row for row in rows if not row.get("judge_excluded") and isinstance(row.get("judge_score"), int)]
    metric_rows = scorable if scorable else rows
    bucket_counts = Counter(row.get("failure_type") or "unknown_unclassified" for row in rows)
    representatives: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        bucket = row.get("failure_type") or "unknown_unclassified"
        if len(representatives[bucket]) < 10:
            representatives[bucket].append(str(row.get("question_id")))
    incorrect_scorable = [row for row in scorable if row.get("judge_verdict") == "incorrect" or row.get("judge_score") == 0]
    return {
        "total_rows": total,
        "scorable_rows": len(scorable),
        "raw_retrieval_shard_recall_average": _avg(metric_rows, "raw_retrieval_shard_recall@20"),
        "final_context_shard_recall_average": _avg(metric_rows, "final_context_shard_recall@top_n"),
        "raw_retrieval_mrr_average": _avg(metric_rows, "raw_retrieval_mrr"),
        "final_context_mrr_average": _avg(metric_rows, "final_context_mrr"),
        "raw_retrieval_expected_shard_hits": sum(1 for row in metric_rows if float(row.get("raw_retrieval_shard_recall@20") or 0.0) > 0.0),
        "final_context_expected_shard_hits": sum(1 for row in metric_rows if float(row.get("final_context_shard_recall@top_n") or 0.0) > 0.0),
        "incorrect_scorable_rows": len(incorrect_scorable),
        "incorrect_scorable_raw_retrieval_misses": sum(1 for row in incorrect_scorable if float(row.get("raw_retrieval_shard_recall@20") or 0.0) <= 0.0),
        "failure_bucket_counts": dict(bucket_counts),
        "representative_question_ids_by_failure_bucket": dict(representatives),
        "source_type_breakdown": _breakdown(rows, "source_types"),
        "reasoning_type_breakdown": _breakdown(rows, "reasoning_types"),
    }


def render_diagnostics_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RAGFlow Bench Diagnostics",
        "",
        f"- total rows: {summary['total_rows']}",
        f"- scorable rows: {summary['scorable_rows']}",
        f"- raw retrieval shard recall avg: {summary['raw_retrieval_shard_recall_average']:.4f}",
        f"- final context shard recall avg: {summary['final_context_shard_recall_average']:.4f}",
        f"- raw retrieval MRR avg: {summary['raw_retrieval_mrr_average']:.4f}",
        f"- final context MRR avg: {summary['final_context_mrr_average']:.4f}",
        f"- raw retrieval expected shard hits: {summary['raw_retrieval_expected_shard_hits']}",
        f"- final context expected shard hits: {summary['final_context_expected_shard_hits']}",
        f"- incorrect scorable raw retrieval misses: {summary['incorrect_scorable_raw_retrieval_misses']}/{summary['incorrect_scorable_rows']}",
        "",
        "## Failure buckets",
    ]
    for bucket, count in sorted(summary["failure_bucket_counts"].items()):
        reps = ", ".join(summary["representative_question_ids_by_failure_bucket"].get(bucket, []))
        lines.append(f"- {bucket}: {count} (examples: {reps})")
    lines.extend(["", "## Source-type breakdown", ""])
    lines.extend(_breakdown_lines(summary["source_type_breakdown"]))
    lines.extend(["", "## Reasoning-type breakdown", ""])
    lines.extend(_breakdown_lines(summary["reasoning_type_breakdown"]))
    return "\n".join(lines) + "\n"


def _avg(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key) or 0.0) for row in rows) / len(rows)


def _breakdown(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    out: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "raw_hits": 0, "final_hits": 0, "failure_buckets": Counter()})
    for row in rows:
        values = row.get(key) or _fallback_breakdown_values(row, key)
        if not isinstance(values, list):
            values = [values]
        for value in values or ["unknown"]:
            name = str(value or "unknown")
            item = out[name]
            item["count"] += 1
            item["raw_hits"] += int(float(row.get("raw_retrieval_shard_recall@20") or 0.0) > 0.0)
            item["final_hits"] += int(float(row.get("final_context_shard_recall@top_n") or 0.0) > 0.0)
            item["failure_buckets"][row.get("failure_type") or "unknown_unclassified"] += 1
    return {
        key: {**value, "failure_buckets": dict(value["failure_buckets"])}
        for key, value in sorted(out.items())
    }


def _fallback_breakdown_values(row: dict[str, Any], key: str) -> list[str]:
    if key == "source_types":
        values: list[str] = []
        for source in row.get("expected_sources") or []:
            text = str(source)
            if text.startswith("eragb-shard://"):
                rest = text[len("eragb-shard://") :]
                values.append(rest.split("/", 1)[0])
            elif "_shard_" in text:
                values.append(text.rsplit("/", 1)[-1].split("_shard_", 1)[0])
        if values:
            return list(dict.fromkeys(values))
    return ["unknown"]


def _breakdown_lines(breakdown: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for name, item in breakdown.items():
        lines.append(f"- {name}: count={item['count']}, raw_hits={item['raw_hits']}, final_hits={item['final_hits']}")
    return lines
