from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from ragflow_bench.benchmarks import CustomBenchmarkAdapter, EnterpriseRAGBenchAdapter, FramesAdapter
from ragflow_bench.benchmarks.base import BenchmarkAdapter
from ragflow_bench.config import AppConfig, BenchmarkKind, DatasetStrategy, dump_config
from ragflow_bench.execution.chat_runner import ensure_chat, run_chat
from ragflow_bench.execution.retrieval_runner import run_retrieval
from ragflow_bench.ingestion.document_registry import DocumentRegistry
from ragflow_bench.ingestion.ingest import ingest_documents, resolve_dataset_id
from ragflow_bench.reports.summary import build_summary
from ragflow_bench.reports.writers import append_jsonl, jsonl_to_csv, write_json
from ragflow_bench.scoring import classify_failure, exact_match, normalized_match, source_recall


def make_adapter(config: AppConfig) -> BenchmarkAdapter:
    if config.benchmark.kind == BenchmarkKind.FRAMES:
        return FramesAdapter(config)
    if config.benchmark.kind == BenchmarkKind.ENTERPRISE_RAG_BENCH:
        return EnterpriseRAGBenchAdapter(config)
    return CustomBenchmarkAdapter(config)


def run_benchmark(config: AppConfig, client) -> Path:
    output_dir = Path(config.output.output_dir or config.default_output_dir())
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_config(output_dir / "config.resolved.yaml", config.resolved_for_output())
    adapter = make_adapter(config)
    if config.dataset.strategy == DatasetStrategy.REUSE_EXISTING and config.document_registry_path:
        registry = DocumentRegistry.load(config.document_registry_path)
        dataset_id = registry.dataset_id
    elif config.dataset.strategy == DatasetStrategy.REUSE_EXISTING:
        dataset_id = config.dataset.dataset_id
        registry = DocumentRegistry(dataset_id=dataset_id)
    else:
        registry = ingest_documents(config, client, adapter, output_dir)
        dataset_id = registry.dataset_id
    if not dataset_id:
        raise ValueError("No dataset_id available for benchmark run")
    run_id = uuid4().hex
    results_path = output_dir / "results.jsonl"
    chat = ensure_chat(client, config, dataset_id, f"ragflow-bench-{run_id}") if config.ragflow.resolved_llm_id() else None
    rows: list[dict] = []
    for question in adapter.load_questions():
        error = None
        raw_retrieval = {}
        raw_response = {}
        ragflow_answer = None
        try:
            raw_retrieval = run_retrieval(client, config, dataset_id, question)
            session_id = None
            if chat:
                if config.chat.fresh_session_per_question:
                    session = client.create_session(chat["id"], name=f"{question.id}")
                    session_id = session.get("id") or session.get("session_id")
                raw_response = run_chat(client, config, chat["id"], question, session_id=session_id)
                ragflow_answer = raw_response.get("answer") or raw_response.get("content") or raw_response.get("data", {}).get("answer")
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
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
    jsonl_to_csv(results_path, output_dir / "results.csv")
    write_json(output_dir / "summary.json", build_summary(benchmark=config.benchmark.kind.value, mode=config.benchmark.mode.value, rows=rows))
    if registry:
        registry.save(output_dir / "document_registry.json")
    return output_dir
