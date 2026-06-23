from __future__ import annotations

import json
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
from ragflow_bench.reports.summary import build_summary
from ragflow_bench.reports.writers import append_jsonl, jsonl_to_csv, write_json
from ragflow_bench.scoring import classify_failure, exact_match, normalized_match, source_recall


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
            raw_retrieval = run_retrieval(client, config, dataset_id, question)
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
                raw_response = run_chat(client, config, chat["id"], question, session_id=session_id)
                ragflow_answer = raw_response.get("answer") or raw_response.get("content") or raw_response.get("data", {}).get("answer")
                emit_progress(progress_callback, {"command": "run", "step": "chat", "status": "ok", "index": index, "total": len(questions), "question_id": question.id, "chat_id": chat.get("id"), "session_id": session_id, "elapsed_seconds": time.monotonic() - chat_started})
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            emit_progress(progress_callback, {"command": "run", "step": "question", "status": "error", "index": index, "total": len(questions), "question_id": question.id, "exception": exc.__class__.__name__, "error": error, "elapsed_seconds": time.monotonic() - q_started})
        if error is None and isinstance(ragflow_answer, str) and ragflow_answer.strip().startswith("**ERROR**:"):
            error = ragflow_answer.strip()

        chunks = raw_retrieval.get("chunks", []) if isinstance(raw_retrieval, dict) else []
        retrieved_document_ids = [chunk.get("doc_id") or chunk.get("document_id") for chunk in chunks if isinstance(chunk, dict)]
        retrieved_chunk_ids = [chunk.get("chunk_id") or chunk.get("id") for chunk in chunks if isinstance(chunk, dict)]
        retrieved_scores = [chunk.get("score") or chunk.get("similarity") for chunk in chunks if isinstance(chunk, dict)]
        retrieved_source_uris = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            metadata = chunk.get("metadata") or chunk.get("meta_fields") or {}
            source_uri = metadata.get("source_uri") or chunk.get("source_uri")
            if source_uri:
                retrieved_source_uris.append(source_uri)
        em = exact_match(question.gold_answer, ragflow_answer)
        nm = normalized_match(question.gold_answer, ragflow_answer)
        recall = source_recall(question.expected_sources, retrieved_source_uris)
        failure = classify_failure(error=error, ragflow_answer=ragflow_answer, exact_match=em, source_recall=recall)
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
            "reasoning_types": question.reasoning_types,
            "failure_type": failure,
            "raw_retrieval": raw_retrieval,
            "raw_response": raw_response,
            "error": error,
        }
        append_jsonl(results_path, row)
        rows.append(row)
        emit_progress(progress_callback, {"command": "run", "step": "row_write", "status": "ok", "index": index, "total": len(questions), "question_id": question.id, "path": str(results_path), "elapsed_seconds": time.monotonic() - q_started})
    jsonl_to_csv(results_path, output_dir / "results.csv")
    emit_progress(progress_callback, {"command": "run", "step": "csv_write", "status": "ok", "path": str(output_dir / "results.csv")})
    write_json(output_dir / "summary.json", build_summary(benchmark=config.benchmark.kind.value, mode=config.benchmark.mode.value, rows=rows))
    emit_progress(progress_callback, {"command": "run", "step": "summary_write", "status": "ok", "path": str(output_dir / "summary.json")})
    if registry:
        registry.save(output_dir / "document_registry.json")
        emit_progress(progress_callback, {"command": "run", "step": "registry_write", "status": "ok", "path": str(output_dir / "document_registry.json")})
    emit_progress(progress_callback, {"command": "run", "step": "complete", "status": "ok", "count": len(rows), "output_dir": str(output_dir), "elapsed_seconds": time.monotonic() - started})
    return output_dir
