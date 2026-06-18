from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ragflow_bench.config import DatasetStrategy, RagflowConnectionConfig, ensure_local_paths_exist, load_config
from ragflow_bench.execution.benchmark_runner import make_adapter, run_benchmark
from ragflow_bench.ingestion.ingest import ingest_documents, resolve_dataset_id
from ragflow_bench.logging_utils import configure_logging
from ragflow_bench.ragflow import RagflowClient
from ragflow_bench.ragflow.errors import RagflowAPIError, RagflowConfigError
from ragflow_bench.reports.summary import build_summary
from ragflow_bench.reports.writers import write_json
from ragflow_bench.scoring import classify_failure, exact_match, normalized_match, source_recall
from ragflow_bench.wizard import run_wizard

app = typer.Typer(help="Standalone benchmark harness for evaluating RAGFlow over raw HTTP APIs")
console = Console()


def _load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


@app.callback()
def main_callback(verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging")):
    _load_dotenv()
    configure_logging(verbose=verbose)


@app.command()
def wizard() -> None:
    target, should_run = run_wizard()
    console.print(f"Saved wizard config to {target}")
    if should_run:
        cfg = load_config(target)
        output_dir = run_benchmark(cfg, RagflowClient(cfg.ragflow))
        console.print(f"Run completed: {output_dir}")


@app.command()
def doctor(config: str | None = typer.Option(None, help="Optional config path for local corpus checks")) -> None:
    issues: list[str] = []
    payload: dict[str, object] = {}
    cfg = load_config(config) if config else None
    client = RagflowClient(cfg.ragflow if cfg else RagflowConnectionConfig())
    requested_embedding_model = (
        cfg.dataset.embedding_model
        if cfg and cfg.dataset.strategy != DatasetStrategy.REUSE_EXISTING
        else None
    )
    try:
        payload["healthz"] = client.health_check()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Health check failed:[/red] {exc}")
        raise typer.Exit(code=1)
    api_key_ok = False
    dataset_id = None
    try:
        status = client.system_status()
        payload["system_status"] = status
        api_key_ok = True
    except RagflowConfigError as exc:
        issues.append(str(exc))
    except RagflowAPIError as exc:
        issues.append(f"Authenticated status probe failed: {exc}")
    if cfg:
        payload["doctor_config"] = {
            "dataset_strategy": cfg.dataset.strategy.value,
            "embedding_model": requested_embedding_model,
        }
        if cfg.dataset.strategy != DatasetStrategy.REUSE_EXISTING and not requested_embedding_model:
            issues.append(
                "dataset.embedding_model not set; new dataset creation will rely on RAGFlow server defaults"
            )
    if api_key_ok:
        try:
            models = client.list_models()
            payload["models_count"] = len(models)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Model listing failed: {exc}")
        try:
            dataset = client.create_dataset(
                name="ragflow-bench-doctor",
                embedding_model=requested_embedding_model,
            )
            dataset_id = dataset.get("id")
            payload["dataset_create"] = {
                "id": dataset_id,
                "requested_embedding_model": requested_embedding_model,
                "embedding_model": dataset.get("embedding_model"),
            }
            probe_file = Path(".omx") / "doctor-probe.txt"
            probe_file.parent.mkdir(exist_ok=True)
            probe_file.write_text("ragflow bench doctor probe", encoding="utf-8")
            uploaded = client.upload_document(dataset_id, probe_file)
            payload["upload_count"] = len(uploaded)
            doc_ids = [item["id"] for item in uploaded]
            if doc_ids:
                client.start_parse(dataset_id, doc_ids)
                docs = client.wait_for_documents_parsed(dataset_id, doc_ids, timeout=120)
                payload["parsed_docs"] = len(docs)
                retrieval = client.retrieve(question="What is this document about?", dataset_ids=[dataset_id])
                payload["retrieval_total"] = retrieval.get("total")
            llm_id = cfg.ragflow.resolved_llm_id() if cfg else None
            if llm_id:
                chat = client.create_chat(name="ragflow-bench-doctor", dataset_ids=[dataset_id], llm_id=llm_id)
                session = client.create_session(chat["id"], name="doctor")
                answer = client.ask_chat(question="Summarize the probe document.", chat_id=chat["id"], session_id=session.get("id") or session.get("session_id"), llm_id=llm_id)
                payload["chat_answer_keys"] = sorted(answer.keys()) if isinstance(answer, dict) else []
            else:
                issues.append("RAGFLOW_LLM_ID not set; skipped chat creation/completion probe")
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Doctor workflow probe failed: {exc}")
            if dataset_id:
                try:
                    docs = client.list_documents(dataset_id, page=1, page_size=20)
                    payload["doctor_documents"] = docs
                    failing = [doc for doc in docs.get("docs", []) if doc.get("run") == "FAIL"]
                    if failing:
                        issues.append(
                            "Live parse mismatch: dataset creation/upload succeeded but parsing failed on this server; "
                            + (failing[0].get("progress_msg") or "see doctor_documents payload")
                        )
                except Exception as inner_exc:  # noqa: BLE001
                    issues.append(f"Failed to inspect doctor documents after probe failure: {inner_exc}")
    if cfg:
        issues.extend(ensure_local_paths_exist(cfg))
    table = Table(title="ragflow-bench doctor")
    table.add_column("check")
    table.add_column("result")
    table.add_row("base_url", client.base_url)
    table.add_row("healthz", json.dumps(payload.get("healthz", {}), ensure_ascii=False))
    table.add_row("api_key", "ok" if api_key_ok else "missing/failed")
    table.add_row("issues", "none" if not issues else " | ".join(issues))
    console.print(table)
    if payload:
        console.print_json(json.dumps(payload, ensure_ascii=False))


@app.command()
def ingest(config: str = typer.Option(..., help="Config YAML path")) -> None:
    cfg = load_config(config)
    client = RagflowClient(cfg.ragflow)
    adapter = make_adapter(cfg)
    output_dir = Path(cfg.output.output_dir or cfg.default_output_dir())
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = ingest_documents(cfg, client, adapter, output_dir)
    console.print_json(json.dumps(registry.to_dict(), ensure_ascii=False))


@app.command()
def retrieve(config: str = typer.Option(..., help="Config YAML path")) -> None:
    cfg = load_config(config)
    client = RagflowClient(cfg.ragflow)
    adapter = make_adapter(cfg)
    dataset_id = resolve_dataset_id(cfg, client, adapter)
    question = adapter.load_questions()[0]
    payload = client.retrieve(
        question=question.question,
        dataset_ids=[dataset_id],
        page_size=cfg.retrieval.page_size,
        similarity_threshold=cfg.retrieval.similarity_threshold,
        vector_similarity_weight=cfg.retrieval.vector_similarity_weight,
        top_k=cfg.retrieval.top_k,
    )
    console.print_json(json.dumps(payload, ensure_ascii=False))


@app.command()
def run(config: str = typer.Option(..., help="Config YAML path")) -> None:
    cfg = load_config(config)
    output_dir = run_benchmark(cfg, RagflowClient(cfg.ragflow))
    console.print(f"Run completed: {output_dir}")


@app.command()
def score(results: str = typer.Option(..., help="Path to results.jsonl")) -> None:
    path = Path(results)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rescored = []
    for row in rows:
        row["exact_match"] = exact_match(row.get("gold_answer"), row.get("ragflow_answer"))
        row["normalized_match"] = normalized_match(row.get("gold_answer"), row.get("ragflow_answer"))
        row["source_recall"] = source_recall(row.get("expected_sources", []), row.get("retrieved_source_uris", []))
        row["failure_type"] = classify_failure(
            error=row.get("error"),
            ragflow_answer=row.get("ragflow_answer"),
            exact_match=row["exact_match"],
            source_recall=row["source_recall"],
        )
        rescored.append(row)
    summary = build_summary(benchmark=rows[0].get("benchmark", ""), mode="rescored", rows=rescored) if rows else {}
    write_json(path.with_name("summary.json"), summary)
    console.print_json(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
