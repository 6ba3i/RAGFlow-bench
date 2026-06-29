from __future__ import annotations

from ragflow_bench.benchmarks.base import BenchmarkQuestion
from ragflow_bench.config import AppConfig
from ragflow_bench.ragflow.client import RagflowClient

EXACT_FACT_PROMPT = (
    "Use only the provided knowledge. Identify exact evidence for numbers, dates, "
    "metric names, thresholds, role lists, acceptance criteria, identifiers, and named entities. "
    "Resolve conflicts by relying on the most directly relevant cited evidence. "
    "Abstain only when exact evidence is absent. Produce concise final answers with citations. "
    "Avoid background explanation."
)


def build_prompt_config(config: AppConfig) -> dict:
    prompt_config = {"quote": config.chat.quote, "refine_multiturn": config.chat.refine_multiturn}
    if config.chat.prompt_mode == "exact_fact":
        prompt_config["system"] = EXACT_FACT_PROMPT
        prompt_config["prompt"] = EXACT_FACT_PROMPT
    return prompt_config


def ensure_chat(client: RagflowClient, config: AppConfig, dataset_id: str, name: str) -> dict:
    llm_id = config.ragflow.resolved_llm_id()
    if not llm_id:
        raise ValueError("RAGFLOW_LLM_ID is required for chat runs")
    return client.create_chat(
        name=name,
        dataset_ids=[dataset_id],
        llm_id=llm_id,
        prompt_config=build_prompt_config(config),
        top_n=config.chat.top_n,
        top_k=config.retrieval.top_k,
        similarity_threshold=config.retrieval.similarity_threshold,
        vector_similarity_weight=config.retrieval.vector_similarity_weight,
        rerank_id=config.retrieval.rerank_id,
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
