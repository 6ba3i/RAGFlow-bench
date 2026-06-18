from __future__ import annotations

from ragflow_bench.benchmarks.base import BenchmarkQuestion
from ragflow_bench.config import AppConfig
from ragflow_bench.ragflow.client import RagflowClient


def run_retrieval(client: RagflowClient, config: AppConfig, dataset_id: str, question: BenchmarkQuestion) -> dict:
    return client.retrieve(
        question=question.question,
        dataset_ids=[dataset_id],
        page_size=config.retrieval.page_size,
        similarity_threshold=config.retrieval.similarity_threshold,
        vector_similarity_weight=config.retrieval.vector_similarity_weight,
        top_k=config.retrieval.top_k,
    )
