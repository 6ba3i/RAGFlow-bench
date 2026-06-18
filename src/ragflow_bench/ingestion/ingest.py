from __future__ import annotations

from pathlib import Path

from ragflow_bench.benchmarks.base import BenchmarkAdapter
from ragflow_bench.config import AppConfig, DatasetStrategy
from ragflow_bench.ingestion.document_registry import DocumentRegistry
from ragflow_bench.ingestion.parse_waiter import wait_for_parse
from ragflow_bench.ragflow.client import RagflowClient


def resolve_dataset_id(config: AppConfig, client: RagflowClient, adapter: BenchmarkAdapter) -> str:
    if config.dataset.strategy == DatasetStrategy.REUSE_EXISTING:
        return config.dataset.dataset_id or DocumentRegistry.load(config.document_registry_path).dataset_id  # type: ignore[arg-type]
    dataset = client.create_dataset(
        name=config.dataset.name or f"{config.benchmark.kind.value}-{config.benchmark.mode.value}",
        description=config.dataset.description,
        embedding_model=config.dataset.embedding_model,
        chunk_method=config.dataset.chunk_method,
        parser_config=config.dataset.parser_config.model_dump(mode="python"),
    )
    return dataset["id"]


def ingest_documents(config: AppConfig, client: RagflowClient, adapter: BenchmarkAdapter, output_dir: Path) -> DocumentRegistry:
    dataset_id = resolve_dataset_id(config, client, adapter)
    registry = DocumentRegistry(dataset_id=dataset_id)
    uploaded_ids: list[str] = []
    for document in adapter.iter_documents():
        uploaded = client.upload_document(dataset_id, document.path)
        if not uploaded:
            continue
        record = uploaded[0]
        document_id = record["id"]
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
        registry.register(
            document.source_uri,
            ragflow_document_id=document_id,
            title=document.title,
            source_type=document.source_type,
            metadata=document.metadata,
        )
        uploaded_ids.append(document_id)
    if uploaded_ids:
        client.start_parse(dataset_id, uploaded_ids)
        wait_for_parse(client, dataset_id, uploaded_ids)
    registry.save(output_dir / "document_registry.json")
    return registry
