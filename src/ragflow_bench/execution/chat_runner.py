from __future__ import annotations

from ragflow_bench.benchmarks.base import BenchmarkQuestion
from ragflow_bench.config import AppConfig
from ragflow_bench.ragflow.client import RagflowClient


def ensure_chat(client: RagflowClient, config: AppConfig, dataset_id: str, name: str) -> dict:
    llm_id = config.ragflow.resolved_llm_id()
    if not llm_id:
        raise ValueError("RAGFLOW_LLM_ID is required for chat runs")
    return client.create_chat(
        name=name,
        dataset_ids=[dataset_id],
        llm_id=llm_id,
        prompt_config={"quote": config.chat.quote, "refine_multiturn": config.chat.refine_multiturn},
        top_n=config.chat.top_n,
        top_k=config.retrieval.top_k,
        similarity_threshold=config.retrieval.similarity_threshold,
        vector_similarity_weight=config.retrieval.vector_similarity_weight,
    )


def run_chat(client: RagflowClient, config: AppConfig, chat_id: str, question: BenchmarkQuestion, *, session_id: str | None = None) -> dict:
    return client.ask_chat(
        question=question.question,
        chat_id=chat_id,
        session_id=session_id,
        llm_id=config.ragflow.resolved_llm_id(),
        quote=config.chat.quote,
        refine_multiturn=config.chat.refine_multiturn,
        stream=False,
        max_tokens=config.chat.max_tokens,
        temperature=config.chat.temperature,
        top_p=config.chat.top_p,
    )
