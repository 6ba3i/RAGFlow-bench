from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from uuid import uuid4

from ragflow_bench.benchmarks import CustomBenchmarkAdapter, EnterpriseRAGBenchAdapter, FramesAdapter
from ragflow_bench.benchmarks.base import BenchmarkAdapter
from ragflow_bench.config import AppConfig, BenchmarkKind, DatasetStrategy, dump_config
from ragflow_bench.execution.chat_runner import ensure_chat, run_chat
from ragflow_bench.execution.retrieval_runner import run_retrieval
from ragflow_bench.ingestion.document_registry import DocumentRegistry
from ragflow_bench.ingestion.ingest import ingest_documents, resolve_dataset_id
from ragflow_bench.logging_utils import ProgressCallback, emit_progress
from ragflow_bench.rate_limits import run_with_rate_limit_retries
from ragflow_bench.reports.summary import build_summary
from ragflow_bench.reports.writers import append_jsonl, jsonl_to_csv, write_json
from ragflow_bench.scoring import classify_failure, exact_match, normalized_match, retrieval_diagnostics, source_recall

QUESTION_DELAY_MIN_SECONDS = 5.0
QUESTION_DELAY_MAX_SECONDS = 10.0


def _sleep_between_questions() -> float:
    delay = random.uniform(QUESTION_DELAY_MIN_SECONDS, QUESTION_DELAY_MAX_SECONDS)
    time.sleep(delay)
    return delay


def _chat_result_rate_limit_reason(raw_response: dict) -> str | None:
    answer = raw_response.get("answer") or raw_response.get("content") or raw_response.get("data", {}).get("answer")
    if isinstance(answer, str) and "ratelimiterror" in answer.lower():
        return answer
    if isinstance(answer, str) and "rate limited" in answer.lower():
        return answer
    return None


def _chunk_id(chunk: dict) -> str | None:
    value = chunk.get("chunk_id") or chunk.get("id")
    return str(value) if value is not None else None


def audit_chunk(chunk: dict, *, rank: int, survived_final_context: bool | None = None) -> dict:
    return {
        "rank": rank,
        "chunk_id": _chunk_id(chunk),
        "doc_id": chunk.get("doc_id"),
        "document_id": chunk.get("document_id"),
        "document_name": chunk.get("document_name"),
        "document_keyword": chunk.get("document_keyword"),
        "docnm_kwd": chunk.get("docnm_kwd"),
        "source_uri": chunk.get("source_uri"),
        "similarity": chunk.get("similarity"),
        "score": chunk.get("score"),
        "term_similarity": chunk.get("term_similarity"),
        "vector_similarity": chunk.get("vector_similarity"),
        "positions": chunk.get("positions"),
        "metadata": chunk.get("metadata"),
        "meta_fields": chunk.get("meta_fields"),
        "document_metadata": chunk.get("document_metadata"),
        "survived_final_context": survived_final_context,
    }


def audit_chunks(chunks: list[dict], *, final_chunk_ids: set[str] | None = None) -> list[dict]:
    audited = []
    for rank, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            continue
        chunk_id = _chunk_id(chunk)
        survived = (chunk_id in final_chunk_ids) if final_chunk_ids is not None and chunk_id is not None else None
        audited.append(audit_chunk(chunk, rank=rank, survived_final_context=survived))
    return audited


def final_context_chunks_from_response(raw_response: dict) -> list[dict]:
    if not isinstance(raw_response, dict):
        return []
    reference = raw_response.get("reference") or raw_response.get("data", {}).get("reference") or {}
    chunks = reference.get("chunks", []) if isinstance(reference, dict) else []
    return chunks if isinstance(chunks, list) else []


def citation_chunk_ids(answer: str | None, final_chunks: list[dict]) -> list[str]:
    if not isinstance(answer, str):
        return []
    ids: list[str] = []
    for marker in re.findall(r"\[ID:(\d+)\]", answer):
        index = int(marker)
        if 0 <= index < len(final_chunks) and isinstance(final_chunks[index], dict):
            chunk_id = _chunk_id(final_chunks[index])
            if chunk_id and chunk_id not in ids:
                ids.append(chunk_id)
    return ids


def make_adapter(config: AppConfig) -> BenchmarkAdapter:
    if config.benchmark.kind == BenchmarkKind.FRAMES:
        return FramesAdapter(config)
    if config.benchmark.kind == BenchmarkKind.ENTERPRISE_RAG_BENCH:
        return EnterpriseRAGBenchAdapter(config)
    return CustomBenchmarkAdapter(config)


def run_benchmark(config: AppConfig, client, question_ids: set[str] | None = None, progress_callback: ProgressCallback | None = None) -> Path:
    started = time.monotonic()
    emit_progress(progress_callback, {"command": "run", "step": "start", "status": "start", "count": len(question_ids) if question_ids is not None else None})
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_config(output_dir / "config.resolved.yaml", config.resolved_for_output())
    emit_progress(progress_callback, {"command": "run", "step": "config_write", "status": "ok", "path": str(output_dir / "config.resolved.yaml"), "output_dir": str(output_dir)})
    adapter = make_adapter(config)
    emit_progress(progress_callback, {"command": "run", "step": "adapter", "status": "ok", "count": config.benchmark.kind.value})
    if config.dataset.strategy == DatasetStrategy.REUSE_EXISTING and config.document_registry_path:
        registry = DocumentRegistry.load(config.document_registry_path)
        dataset_id = registry.dataset_id
        emit_progress(progress_callback, {"command": "run", "step": "registry_load", "status": "ok", "path": str(config.document_registry_path), "dataset_id": dataset_id})
    elif config.dataset.strategy == DatasetStrategy.REUSE_EXISTING:
        dataset_id = config.dataset.dataset_id
        registry = DocumentRegistry(dataset_id=dataset_id)
        emit_progress(progress_callback, {"command": "run", "step": "dataset_reuse", "status": "ok", "dataset_id": dataset_id})
    else:
        registry = ingest_documents(config, client, adapter, output_dir, progress_callback=progress_callback)
        dataset_id = registry.dataset_id
    if not dataset_id:
        raise ValueError("No dataset_id available for benchmark run")
    questions = adapter.load_questions()
    emit_progress(progress_callback, {"command": "run", "step": "questions_load", "status": "ok", "count": len(questions), "dataset_id": dataset_id})
    selected_question_ids = {str(question_id) for question_id in question_ids} if question_ids is not None else None
    if selected_question_ids is not None:
        available_question_ids = {str(question.id) for question in questions}
        missing = selected_question_ids - available_question_ids
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Unknown question_id(s): {missing_list}")
        questions = [question for question in questions if str(question.id) in selected_question_ids]
        emit_progress(progress_callback, {"command": "run", "step": "questions_filter", "status": "ok", "count": len(questions)})
    run_id = uuid4().hex
    results_path = output_dir / "results.jsonl"
    chat = None
    if config.ragflow.resolved_llm_id():
        chat_started = time.monotonic()
        emit_progress(progress_callback, {"command": "run", "step": "chat_create", "status": "start", "dataset_id": dataset_id})
        chat = ensure_chat(client, config, dataset_id, f"ragflow-bench-{run_id}")
        emit_progress(progress_callback, {"command": "run", "step": "chat_create", "status": "ok", "dataset_id": dataset_id, "chat_id": chat.get("id"), "elapsed_seconds": time.monotonic() - chat_started})
    else:
        emit_progress(progress_callback, {"command": "run", "step": "chat_create", "status": "skipped", "dataset_id": dataset_id})
    rows: list[dict] = []
    for index, question in enumerate(questions, start=1):
        q_started = time.monotonic()
        emit_progress(progress_callback, {"command": "run", "step": "question", "status": "start", "index": index, "total": len(questions), "question_id": question.id})
        error = None
        raw_retrieval = {}
        raw_response = {}
        ragflow_answer = None
        try:
            retrieval_started = time.monotonic()
            emit_progress(progress_callback, {"command": "run", "step": "retrieval", "status": "start", "index": index, "total": len(questions), "question_id": question.id, "dataset_id": dataset_id})
            raw_retrieval = run_with_rate_limit_retries(
                lambda: run_retrieval(client, config, dataset_id, question),
                action_type="retrieval",
                question_id=str(question.id),
                progress_callback=progress_callback,
            )
            emit_progress(progress_callback, {"command": "run", "step": "retrieval", "status": "ok", "index": index, "total": len(questions), "question_id": question.id, "count": len(raw_retrieval.get("chunks", [])) if isinstance(raw_retrieval, dict) else None, "elapsed_seconds": time.monotonic() - retrieval_started})
            session_id = None
            if chat:
                if config.chat.fresh_session_per_question:
                    session_started = time.monotonic()
                    emit_progress(progress_callback, {"command": "run", "step": "session_create", "status": "start", "question_id": question.id, "chat_id": chat.get("id")})
                    session = client.create_session(chat["id"], name=f"{question.id}")
                    session_id = session.get("id") or session.get("session_id")
                    emit_progress(progress_callback, {"command": "run", "step": "session_create", "status": "ok", "question_id": question.id, "chat_id": chat.get("id"), "session_id": session_id, "elapsed_seconds": time.monotonic() - session_started})
                chat_started = time.monotonic()
                emit_progress(progress_callback, {"command": "run", "step": "chat", "status": "start", "index": index, "total": len(questions), "question_id": question.id, "chat_id": chat.get("id"), "session_id": session_id})
                raw_response = run_with_rate_limit_retries(
                    lambda: run_chat(client, config, chat["id"], question, session_id=session_id),
                    action_type="chat",
                    question_id=str(question.id),
                    progress_callback=progress_callback,
                    should_retry_result=_chat_result_rate_limit_reason,
                )
                ragflow_answer = raw_response.get("answer") or raw_response.get("content") or raw_response.get("data", {}).get("answer")
                emit_progress(progress_callback, {"command": "run", "step": "chat", "status": "ok", "index": index, "total": len(questions), "question_id": question.id, "chat_id": chat.get("id"), "session_id": session_id, "elapsed_seconds": time.monotonic() - chat_started})
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            emit_progress(progress_callback, {"command": "run", "step": "question", "status": "error", "index": index, "total": len(questions), "question_id": question.id, "exception": exc.__class__.__name__, "error": error, "elapsed_seconds": time.monotonic() - q_started})
        if error is None and isinstance(ragflow_answer, str) and ragflow_answer.strip().startswith("**ERROR**:"):
            error = ragflow_answer.strip()

        chunks = raw_retrieval.get("chunks", []) if isinstance(raw_retrieval, dict) else []
        chunks = chunks if isinstance(chunks, list) else []
        final_chunks = final_context_chunks_from_response(raw_response)
        final_chunk_ids = {_chunk_id(chunk) for chunk in final_chunks if isinstance(chunk, dict) and _chunk_id(chunk)}
        raw_retrieval_chunks = audit_chunks(chunks, final_chunk_ids=final_chunk_ids)
        final_context_chunks = audit_chunks(final_chunks)
        cited_chunk_ids = citation_chunk_ids(ragflow_answer, final_chunks)
        retrieved_document_ids = [chunk.get("doc_id") or chunk.get("document_id") for chunk in chunks if isinstance(chunk, dict)]
        retrieved_chunk_ids = [chunk.get("chunk_id") or chunk.get("id") for chunk in chunks if isinstance(chunk, dict)]
        retrieved_scores = [chunk.get("score") or chunk.get("similarity") for chunk in chunks if isinstance(chunk, dict)]
        raw_diag = retrieval_diagnostics(question.expected_sources, chunks, prefix="raw_retrieval")
        final_diag = retrieval_diagnostics(question.expected_sources, final_chunks, prefix="final_context")
        retrieved_source_uris = raw_diag["raw_retrieval_retrieved_source_uris"]
        em = exact_match(question.gold_answer, ragflow_answer)
        nm = normalized_match(question.gold_answer, ragflow_answer)
        recall = source_recall(question.expected_sources, retrieved_source_uris)
        failure = classify_failure(
            error=error,
            ragflow_answer=ragflow_answer,
            exact_match=em,
            source_recall=recall,
            raw_retrieval_shard_recall=raw_diag["raw_retrieval_shard_recall"],
            final_context_shard_recall=final_diag["final_context_shard_recall"],
        )
        emit_progress(progress_callback, {"command": "run", "step": "score", "status": failure, "index": index, "total": len(questions), "question_id": question.id, "count": len(retrieved_source_uris)})
        row = {
            "benchmark": config.benchmark.kind.value,
            "run_id": run_id,
            "question_id": question.id,
            "question": question.question,
            "gold_answer": question.gold_answer,
            "ragflow_answer": ragflow_answer,
            "exact_match": em,
            "normalized_match": nm,
            "expected_sources": question.expected_sources,
            "retrieved_document_ids": retrieved_document_ids,
            "retrieved_source_uris": retrieved_source_uris,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "retrieved_scores": retrieved_scores,
            "source_recall": recall,
            "raw_retrieval_shard_recall@20": raw_diag["raw_retrieval_shard_recall"],
            "raw_retrieval_expected_rank": raw_diag["raw_retrieval_expected_rank"],
            "raw_retrieval_mrr": raw_diag["raw_retrieval_mrr"],
            "final_context_shard_recall@top_n": final_diag["final_context_shard_recall"],
            "final_context_expected_rank": final_diag["final_context_expected_rank"],
            "final_context_mrr": final_diag["final_context_mrr"],
            "retrieval_diagnostics": {**raw_diag, **final_diag},
            "raw_retrieval_chunks": raw_retrieval_chunks,
            "final_context_chunks": final_context_chunks,
            "citation_chunk_ids": cited_chunk_ids,
            "prompt_chunk_count": len(final_context_chunks),
            "chat_top_n": config.chat.top_n,
            "retrieval_page_size": config.retrieval.page_size,
            "retrieval_rerank_id": config.retrieval.rerank_id,
            "chat_prompt_mode": config.chat.prompt_mode,
            "reasoning_types": question.reasoning_types,
            "failure_type": failure,
            "raw_retrieval": raw_retrieval,
            "raw_response": raw_response,
            "error": error,
        }
        append_jsonl(results_path, row)
        rows.append(row)
        emit_progress(progress_callback, {"command": "run", "step": "row_write", "status": "ok", "index": index, "total": len(questions), "question_id": question.id, "path": str(results_path), "elapsed_seconds": time.monotonic() - q_started})
        if index < len(questions):
            delay = _sleep_between_questions()
            emit_progress(progress_callback, {"command": "run", "step": "question_delay", "status": "ok", "index": index, "total": len(questions), "question_id": question.id, "delay": delay})
    jsonl_to_csv(results_path, output_dir / "results.csv")
    emit_progress(progress_callback, {"command": "run", "step": "csv_write", "status": "ok", "path": str(output_dir / "results.csv")})
    write_json(output_dir / "summary.json", build_summary(benchmark=config.benchmark.kind.value, mode=config.benchmark.mode.value, rows=rows))
    emit_progress(progress_callback, {"command": "run", "step": "summary_write", "status": "ok", "path": str(output_dir / "summary.json")})
    if registry:
        registry.save(output_dir / "document_registry.json")
        emit_progress(progress_callback, {"command": "run", "step": "registry_write", "status": "ok", "path": str(output_dir / "document_registry.json")})
    emit_progress(progress_callback, {"command": "run", "step": "complete", "status": "ok", "count": len(rows), "output_dir": str(output_dir), "elapsed_seconds": time.monotonic() - started})
    return output_dir
