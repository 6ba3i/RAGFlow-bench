from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm, IntPrompt, Prompt

from ragflow_bench.config import BenchmarkKind, BenchmarkMode, DatasetStrategy, dump_config

console = Console()


def run_wizard() -> tuple[Path, bool]:
    benchmark = Prompt.ask("Which benchmark?", choices=[k.value for k in BenchmarkKind], default=BenchmarkKind.FRAMES.value)
    mode = Prompt.ask("Which mode?", choices=[m.value for m in BenchmarkMode], default=BenchmarkMode.SMOKE.value)
    question_choice = Prompt.ask("How many questions?", choices=["default smoke limit", "custom limit", "full"], default="default smoke limit")
    question_limit = 25 if question_choice == "default smoke limit" else (None if question_choice == "full" else IntPrompt.ask("Custom question limit", default=25))
    base_url = Prompt.ask("RAGFlow base URL", default="http://127.0.0.1:80")
    api_env = Prompt.ask("API key environment variable name", default="RAGFLOW_API_KEY")
    llm_env = Prompt.ask("LLM ID environment variable name", default="RAGFLOW_LLM_ID")
    strategy = Prompt.ask(
        "Dataset strategy",
        choices=[s.value for s in DatasetStrategy],
        default=DatasetStrategy.REUSE_EXISTING.value,
    )
    dataset_id = Prompt.ask("Existing dataset ID", default="") if strategy == DatasetStrategy.REUSE_EXISTING.value else ""
    dataset_name = Prompt.ask("New dataset name", default=f"{benchmark}-{mode}") if strategy != DatasetStrategy.REUSE_EXISTING.value else ""
    embedding_model = (
        Prompt.ask(
            "Dataset embedding model",
            default="",
        )
        if strategy != DatasetStrategy.REUSE_EXISTING.value
        else ""
    )
    chunk_size = IntPrompt.ask("Chunk size", default=512)
    delimiter = Prompt.ask("Delimiter", default="\n")
    top_k = IntPrompt.ask("Retrieval top_k", default=128)
    page_size = IntPrompt.ask("Retrieval page_size", default=20)
    similarity_threshold = float(Prompt.ask("Similarity threshold", default="0.05"))
    vector_similarity_weight = float(Prompt.ask("Vector similarity weight", default="0.3"))
    top_n = IntPrompt.ask("Chat top_n", default=8)
    temperature = float(Prompt.ask("Temperature", default="0.0"))
    top_p = float(Prompt.ask("Top p", default="0.1"))
    max_tokens = IntPrompt.ask("Max tokens", default=128)
    fresh = Confirm.ask("Fresh session per question?", default=True)
    quote = Confirm.ask("Quote references?", default=True)
    refine_multiturn = Confirm.ask("Refine multiturn?", default=False)
    default_output = f"outputs/{benchmark}_{mode}_<timestamp>"
    output_dir = Prompt.ask("Output folder", default=default_output)

    payload = {
        "benchmark": {"kind": benchmark, "mode": mode, "question_limit": question_limit},
        "ragflow": {
            "base_url": base_url,
            "api_key_env_var": api_env,
            "llm_id_env_var": llm_env,
        },
        "dataset": {
            "strategy": strategy,
            "dataset_id": dataset_id or None,
            "name": dataset_name or None,
            "embedding_model": embedding_model or None,
            "chunk_method": "naive",
            "parser_config": {
                "chunk_token_num": chunk_size,
                "delimiter": "\n" if delimiter == "\n" else delimiter,
                "raptor": {"use_raptor": False},
                "graphrag": {"use_graphrag": False},
            },
        },
        "retrieval": {
            "top_k": top_k,
            "page_size": page_size,
            "similarity_threshold": similarity_threshold,
            "vector_similarity_weight": vector_similarity_weight,
        },
        "chat": {
            "top_n": top_n,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "fresh_session_per_question": fresh,
            "quote": quote,
            "refine_multiturn": refine_multiturn,
        },
        "output": {"output_dir": output_dir},
    }
    if benchmark == BenchmarkKind.FRAMES.value:
        payload["benchmark"]["frames"] = {"split": "test", "mapping_path": "frames_mapping.json", "local_corpus_dir": "./frames_corpus"}
    elif benchmark == BenchmarkKind.ENTERPRISE_RAG_BENCH.value:
        payload["benchmark"]["enterprise_rag_bench"] = {"corpus_dir": "./corpus", "questions_path": "./questions.jsonl"}
    else:
        payload["benchmark"]["custom"] = {"corpus_dir": "./corpus", "questions_path": "./questions.jsonl"}

    target = Path(Prompt.ask("Config path", default=f"configs/{benchmark}_{mode}.yaml"))
    target.parent.mkdir(parents=True, exist_ok=True)
    dump_config(target, payload)
    command = f"ragflow-bench run --config {target}"
    console.print(f"[green]Wrote config:[/green] {target}")
    console.print(f"[bold]Run:[/bold] {command}")
    return target, Confirm.ask("Run immediately?", default=False)
