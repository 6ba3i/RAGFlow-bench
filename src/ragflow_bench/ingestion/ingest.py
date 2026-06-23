from __future__ import annotations

import time
from pathlib import Path

from ragflow_bench.benchmarks.base import BenchmarkAdapter
from ragflow_bench.config import AppConfig, DatasetStrategy
from ragflow_bench.ingestion.document_registry import DocumentRegistry
from ragflow_bench.ingestion.parse_waiter import wait_for_parse
from ragflow_bench.logging_utils import ProgressCallback, emit_progress
from ragflow_bench.ragflow.client import RagflowClient


def resolve_dataset_id(config: AppConfig, client: RagflowClient, adapter: BenchmarkAdapter, progress_callback: ProgressCallback | None = None) -> str:
    if config.dataset.strategy == DatasetStrategy.REUSE_EXISTING:
        dataset_id = config.dataset.dataset_id or DocumentRegistry.load(config.document_registry_path).dataset_id  # type: ignore[arg-type]
        emit_progress(progress_callback, {"command": "dataset", "step": "reuse", "status": "ok", "dataset_id": dataset_id})
        return dataset_id
    started = time.monotonic()
    emit_progress(progress_callback, {"command": "dataset", "step": "create", "status": "start"})
    dataset = client.create_dataset(
        name=config.dataset.name or f"{config.benchmark.kind.value}-{config.benchmark.mode.value}",
        description=config.dataset.description,
        embedding_model=config.dataset.embedding_model,
        chunk_method=config.dataset.chunk_method,
        parser_config=config.dataset.parser_config.model_dump(mode="python"),
    )
    emit_progress(progress_callback, {"command": "dataset", "step": "create", "status": "ok", "dataset_id": dataset.get("id"), "elapsed_seconds": time.monotonic() - started})
    return dataset["id"]


def ingest_documents(config: AppConfig, client: RagflowClient, adapter: BenchmarkAdapter, output_dir: Path, progress_callback: ProgressCallback | None = None) -> DocumentRegistry:
    started = time.monotonic()
    emit_progress(progress_callback, {"command": "ingest", "step": "start", "status": "start", "output_dir": str(output_dir)})
    dataset_id = resolve_dataset_id(config, client, adapter, progress_callback)
    registry = DocumentRegistry(dataset_id=dataset_id)
    uploaded_ids: list[str] = []
    documents = list(adapter.iter_documents())
    emit_progress(progress_callback, {"command": "ingest", "step": "documents_loaded", "status": "ok", "count": len(documents), "dataset_id": dataset_id})
    for index, document in enumerate(documents, start=1):
        doc_started = time.monotonic()
        emit_progress(progress_callback, {"command": "ingest", "step": "upload", "status": "start", "index": index, "total": len(documents), "path": str(document.path), "dataset_id": dataset_id})
        try:
            uploaded = client.upload_document(dataset_id, document.path)
            if not uploaded:
                emit_progress(progress_callback, {"command": "ingest", "step": "upload", "status": "skipped", "index": index, "total": len(documents), "path": str(document.path), "dataset_id": dataset_id, "elapsed_seconds": time.monotonic() - doc_started})
                continue
            record = uploaded[0]
            document_id = record["id"]
            emit_progress(progress_callback, {"command": "ingest", "step": "upload", "status": "ok", "index": index, "total": len(documents), "document_id": document_id, "dataset_id": dataset_id, "elapsed_seconds": time.monotonic() - doc_started})
            meta_started = time.monotonic()
            emit_progress(progress_callback, {"command": "ingest", "step": "metadata", "status": "start", "document_id": document_id, "dataset_id": dataset_id})
            client.patch_document_metadata(
                dataset_id,
                document_id,
                meta_fields={
                    "source_uri": document.source_uri,
                    "title": document.title,
                    "source_type": document.source_type,
                    **(document.metadata or {}),
                },
            )
            emit_progress(progress_callback, {"command": "ingest", "step": "metadata", "status": "ok", "document_id": document_id, "dataset_id": dataset_id, "elapsed_seconds": time.monotonic() - meta_started})
            registry.register(
                document.source_uri,
                ragflow_document_id=document_id,
                title=document.title,
                source_type=document.source_type,
                metadata=document.metadata,
            )
            uploaded_ids.append(document_id)
        except Exception as exc:
            emit_progress(progress_callback, {"command": "ingest", "step": "document", "status": "error", "index": index, "total": len(documents), "path": str(document.path), "exception": exc.__class__.__name__, "error": str(exc), "elapsed_seconds": time.monotonic() - doc_started})
            raise
    if uploaded_ids:
        parse_started = time.monotonic()
        emit_progress(progress_callback, {"command": "ingest", "step": "parse_start", "status": "start", "count": len(uploaded_ids), "dataset_id": dataset_id})
        client.start_parse(dataset_id, uploaded_ids)
        emit_progress(progress_callback, {"command": "ingest", "step": "parse_start", "status": "ok", "count": len(uploaded_ids), "dataset_id": dataset_id, "elapsed_seconds": time.monotonic() - parse_started})
        wait_started = time.monotonic()
        emit_progress(progress_callback, {"command": "ingest", "step": "parse_wait", "status": "start", "count": len(uploaded_ids), "dataset_id": dataset_id})
        wait_for_parse(client, dataset_id, uploaded_ids)
        emit_progress(progress_callback, {"command": "ingest", "step": "parse_wait", "status": "ok", "count": len(uploaded_ids), "dataset_id": dataset_id, "elapsed_seconds": time.monotonic() - wait_started})
    registry_path = output_dir / "document_registry.json"
    registry.save(registry_path)
    emit_progress(progress_callback, {"command": "ingest", "step": "registry_write", "status": "ok", "path": str(registry_path), "dataset_id": dataset_id})
    emit_progress(progress_callback, {"command": "ingest", "step": "complete", "status": "ok", "count": len(uploaded_ids), "dataset_id": dataset_id, "elapsed_seconds": time.monotonic() - started})
    return registry
