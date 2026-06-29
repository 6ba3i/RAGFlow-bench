from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ragflow_bench.config import AppConfig, DatasetStrategy, JudgeSettings, RagflowConnectionConfig, ensure_local_paths_exist, load_config
from ragflow_bench.benchmarks import prepare_eragb_artifacts, prepare_frames_artifacts
from ragflow_bench.benchmarks.eragb_prep import ERAGBDownloadError
from ragflow_bench.execution.benchmark_runner import make_adapter, run_benchmark
from ragflow_bench.ingestion.ingest import ingest_documents, resolve_dataset_id
from ragflow_bench.judge import ZhipuJudgeClient, default_progress_printer as judge_progress_printer, is_excluded_infra_error, judge_results_file
from ragflow_bench.logging_utils import configure_logging, default_progress_printer, emit_progress
from ragflow_bench.ragflow import RagflowClient
from ragflow_bench.ragflow.errors import RagflowAPIError, RagflowConfigError
from ragflow_bench.reports.summary import build_summary
from ragflow_bench.reports.diagnostics import recompute_row_diagnostics, recompute_run_diagnostics
from ragflow_bench.reports.writers import jsonl_to_csv, load_jsonl, write_json, write_jsonl
from ragflow_bench.scoring import classify_failure, exact_match, normalized_match, source_recall
from ragflow_bench.wizard import run_wizard

app = typer.Typer(help="Standalone benchmark harness for evaluating RAGFlow over raw HTTP APIs")
console = Console()


def _unredact_config_for_retry(cfg: AppConfig) -> AppConfig:
    if cfg.ragflow.api_key == "***REDACTED***":
        cfg.ragflow.api_key = None
    if cfg.judge.api_key == "***REDACTED***":
        cfg.judge.api_key = None
    return cfg


def _backup_run_artifacts(run_dir: Path, backup_dir: Path) -> list[str]:
    backup_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in ("results.jsonl", "results.csv", "summary.json"):
        source = run_dir / name
        if source.exists():
            shutil.copy2(source, backup_dir / name)
            copied.append(name)
    return copied


def _row_needs_retry(row: dict) -> bool:
    return bool(row.get("error")) or is_excluded_infra_error(row)


def _auto_judge_run_dir(run_dir: str | Path, *, resume: bool = True, force_question_ids: set[str] | None = None) -> dict:
    run_path = Path(run_dir)
    results_path = run_path / "results.jsonl"
    config_path = run_path / "config.resolved.yaml"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing resolved config: {config_path}")
    cfg = _unredact_config_for_retry(load_config(config_path))
    return judge_results_file(
        results_path=results_path,
        client=ZhipuJudgeClient(cfg.judge),
        output_path=run_path / "judge_results.jsonl",
        resume=resume,
        force_question_ids=force_question_ids,
        progress_callback=judge_progress_printer,
    )


def _judge_error_question_ids(run_dir: str | Path) -> set[str]:
    rows = load_jsonl(Path(run_dir) / "judge_results.jsonl")
    return {str(row.get("question_id")) for row in rows if row.get("question_id") is not None and row.get("judge_exclusion_reason") == "judge_error"}


def _merge_retry_rows(original_rows: list[dict], retry_rows: list[dict]) -> tuple[list[dict], int]:
    original_ids = [str(row.get("question_id")) for row in original_rows]
    original_id_set = set(original_ids)
    retry_by_id: dict[str, dict] = {}
    for row in retry_rows:
        question_id = str(row.get("question_id"))
        if question_id not in original_id_set:
            raise ValueError(f"Retry produced unknown question_id: {question_id}")
        retry_by_id[question_id] = row
    merged = [retry_by_id.get(str(row.get("question_id")), row) for row in original_rows]
    return merged, len(retry_by_id)


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


def _exit_if_local_paths_missing(cfg) -> None:
    errors = ensure_local_paths_exist(cfg)
    if not errors:
        return
    console.print("[red]Missing required local benchmark files:[/red]")
    for error in errors:
        console.print(f"- {error}")
    if getattr(cfg.benchmark, "kind", None) and str(cfg.benchmark.kind.value) == "frames":
        console.print("[yellow]Hint:[/yellow] run [bold]ragflow-bench prepare-frames[/bold] first.")
    raise typer.Exit(code=1)


@app.callback()
def main_callback(verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging")):
    _load_dotenv()
    configure_logging(verbose=verbose)


@app.command()
def wizard() -> None:
    emit_progress(default_progress_printer, {"command": "wizard", "step": "start", "status": "start"})
    target, should_run = run_wizard()
    emit_progress(default_progress_printer, {"command": "wizard", "step": "config_write", "status": "ok", "path": str(target)})
    console.print(f"Saved wizard config to {target}")
    if should_run:
        cfg = load_config(target)
        _exit_if_local_paths_missing(cfg)
        output_dir = run_benchmark(cfg, RagflowClient(cfg.ragflow), progress_callback=default_progress_printer)
        _auto_judge_run_dir(output_dir)
        console.print(f"Run completed: {output_dir}")
    emit_progress(default_progress_printer, {"command": "wizard", "step": "complete", "status": "ok", "path": str(target)})


@app.command("prepare-frames")
def prepare_frames(
    split: str = typer.Option("test", help="FRAMES split to prepare"),
    question_limit: int | None = typer.Option(None, help="Optional question limit for a smaller local corpus"),
    output_dir: str = typer.Option("data/frames", help="Output directory for mapping, report, and corpus"),
    refresh: bool = typer.Option(False, help="Re-download and rewrite existing page files"),
) -> None:
    report = prepare_frames_artifacts(
        split=split,
        question_limit=question_limit,
        output_dir=output_dir,
        refresh=refresh,
        progress_callback=default_progress_printer,
    )
    table = Table(title="ragflow-bench prepare-frames")
    table.add_column("artifact")
    table.add_column("value")
    table.add_row("split", str(report["split"]))
    table.add_row("questions", str(report["question_count"]))
    table.add_row("mapped_questions", str(report["mapped_question_count"]))
    table.add_row("unique_wikipedia_urls", str(report["unique_wikipedia_url_count"]))
    table.add_row("downloaded_pages", str(report["downloaded_page_count"]))
    table.add_row("failures", str(report["failure_count"]))
    table.add_row("mapping_path", str(report["mapping_path"]))
    table.add_row("corpus_dir", str(report["corpus_dir"]))
    console.print(table)
    console.print_json(json.dumps(report, ensure_ascii=False))


@app.command("prepare-eragb")
def prepare_eragb(
    split: str = typer.Option("test", help="EnterpriseRAG-Bench split to prepare"),
    output_dir: str = typer.Option("data/eragb", help="Output directory for raw parquet, corpus, questions, manifest, and report"),
    document_limit: int | None = typer.Option(None, help="Optional document limit for smaller local corpora"),
    question_limit: int | None = typer.Option(None, help="Optional question limit for smaller local question sets"),
    refresh: bool = typer.Option(False, help="Re-download parquet files and rewrite generated artifacts"),
    hf_token_env_var: str = typer.Option("HF_TOKEN", help="Environment variable containing a Hugging Face token, if needed"),
    merge_documents: bool = typer.Option(False, help="Merge ERAGB rows into deterministic chunk-safe shard files"),
    merge_target_bytes: int = typer.Option(262144, help="Target maximum shard size in bytes when merging documents"),
    merge_max_docs: int = typer.Option(100, help="Maximum embedded documents per shard when merging documents"),
    filter_questions_with_missing_docs: bool = typer.Option(False, help="Drop questions whose expected docs are not present in the prepared corpus"),
    reference_granularity: str | None = typer.Option(None, help="Expected-source granularity: document, shard, or none. Defaults to shard in merged mode, document otherwise."),
) -> None:
    try:
        report = prepare_eragb_artifacts(
            split=split,
            output_dir=output_dir,
            document_limit=document_limit,
            question_limit=question_limit,
            refresh=refresh,
            hf_token_env_var=hf_token_env_var,
            merge_documents=merge_documents,
            merge_target_bytes=merge_target_bytes,
            merge_max_docs=merge_max_docs,
            filter_questions_with_missing_docs=filter_questions_with_missing_docs,
            reference_granularity=reference_granularity,
            progress_callback=default_progress_printer,
        )
    except ERAGBDownloadError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    table = Table(title="ragflow-bench prepare-eragb")
    table.add_column("artifact")
    table.add_column("value")
    table.add_row("split", str(report["split"]))
    table.add_row("documents", str(report["document_count"]))
    table.add_row("shards", str(report["shard_count"]))
    table.add_row("questions", str(report["question_count_written"]))
    table.add_row("reference_granularity", str(report["reference_granularity"]))
    table.add_row("corpus_dir", str(report["corpus_dir"]))
    table.add_row("questions_path", str(report["questions_path"]))
    table.add_row("documents_manifest", str(report["documents_manifest_path"]))
    console.print(table)
    console.print_json(json.dumps(report, ensure_ascii=False))


@app.command()
def doctor(config: str | None = typer.Option(None, help="Optional config path for local corpus checks")) -> None:
    emit_progress(default_progress_printer, {"command": "doctor", "step": "start", "status": "start", "path": config})
    issues: list[str] = []
    payload: dict[str, object] = {}
    cfg = load_config(config) if config else None
    emit_progress(default_progress_printer, {"command": "doctor", "step": "config_load", "status": "ok" if cfg else "skipped", "path": config})
    client = RagflowClient(cfg.ragflow if cfg else RagflowConnectionConfig())
    requested_embedding_model = (
        cfg.dataset.embedding_model
        if cfg and cfg.dataset.strategy != DatasetStrategy.REUSE_EXISTING
        else None
    )
    try:
        emit_progress(default_progress_printer, {"command": "doctor", "step": "health_check", "status": "start"})
        payload["healthz"] = client.health_check()
        emit_progress(default_progress_printer, {"command": "doctor", "step": "health_check", "status": "ok"})
    except Exception as exc:  # noqa: BLE001
        emit_progress(default_progress_printer, {"command": "doctor", "step": "health_check", "status": "error", "exception": exc.__class__.__name__, "error": str(exc)})
        console.print(f"[red]Health check failed:[/red] {exc}")
        raise typer.Exit(code=1)
    api_key_ok = False
    dataset_id = None
    try:
        emit_progress(default_progress_printer, {"command": "doctor", "step": "system_status", "status": "start"})
        status = client.system_status()
        payload["system_status"] = status
        api_key_ok = True
        emit_progress(default_progress_printer, {"command": "doctor", "step": "system_status", "status": "ok"})
    except RagflowConfigError as exc:
        emit_progress(default_progress_printer, {"command": "doctor", "step": "system_status", "status": "error", "exception": exc.__class__.__name__, "error": str(exc)})
        issues.append(str(exc))
    except RagflowAPIError as exc:
        emit_progress(default_progress_printer, {"command": "doctor", "step": "system_status", "status": "error", "exception": exc.__class__.__name__, "error": str(exc)})
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
            emit_progress(default_progress_printer, {"command": "doctor", "step": "models", "status": "start"})
            models = client.list_models()
            payload["models_count"] = len(models)
            emit_progress(default_progress_printer, {"command": "doctor", "step": "models", "status": "ok", "count": len(models)})
        except Exception as exc:  # noqa: BLE001
            emit_progress(default_progress_printer, {"command": "doctor", "step": "models", "status": "error", "exception": exc.__class__.__name__, "error": str(exc)})
            issues.append(f"Model listing failed: {exc}")
        try:
            emit_progress(default_progress_printer, {"command": "doctor", "step": "dataset_create", "status": "start"})
            dataset = client.create_dataset(
                name="ragflow-bench-doctor",
                embedding_model=requested_embedding_model,
            )
            dataset_id = dataset.get("id")
            emit_progress(default_progress_printer, {"command": "doctor", "step": "dataset_create", "status": "ok", "dataset_id": dataset_id})
            payload["dataset_create"] = {
                "id": dataset_id,
                "requested_embedding_model": requested_embedding_model,
                "embedding_model": dataset.get("embedding_model"),
            }
            probe_file = Path(".omx") / "doctor-probe.txt"
            probe_file.parent.mkdir(exist_ok=True)
            probe_file.write_text("ragflow bench doctor probe", encoding="utf-8")
            emit_progress(default_progress_printer, {"command": "doctor", "step": "upload", "status": "start", "dataset_id": dataset_id, "path": str(probe_file)})
            uploaded = client.upload_document(dataset_id, probe_file)
            payload["upload_count"] = len(uploaded)
            emit_progress(default_progress_printer, {"command": "doctor", "step": "upload", "status": "ok", "dataset_id": dataset_id, "count": len(uploaded)})
            doc_ids = [item["id"] for item in uploaded]
            if doc_ids:
                emit_progress(default_progress_printer, {"command": "doctor", "step": "parse_start", "status": "start", "dataset_id": dataset_id, "count": len(doc_ids)})
                client.start_parse(dataset_id, doc_ids)
                emit_progress(default_progress_printer, {"command": "doctor", "step": "parse_start", "status": "ok", "dataset_id": dataset_id, "count": len(doc_ids)})
                emit_progress(default_progress_printer, {"command": "doctor", "step": "parse_wait", "status": "start", "dataset_id": dataset_id, "count": len(doc_ids)})
                docs = client.wait_for_documents_parsed(dataset_id, doc_ids, timeout=120)
                payload["parsed_docs"] = len(docs)
                emit_progress(default_progress_printer, {"command": "doctor", "step": "parse_wait", "status": "ok", "dataset_id": dataset_id, "count": len(docs)})
                emit_progress(default_progress_printer, {"command": "doctor", "step": "retrieval", "status": "start", "dataset_id": dataset_id})
                retrieval = client.retrieve(question="What is this document about?", dataset_ids=[dataset_id])
                payload["retrieval_total"] = retrieval.get("total")
                emit_progress(default_progress_printer, {"command": "doctor", "step": "retrieval", "status": "ok", "dataset_id": dataset_id, "count": retrieval.get("total")})
            llm_id = cfg.ragflow.resolved_llm_id() if cfg else None
            if llm_id:
                emit_progress(default_progress_printer, {"command": "doctor", "step": "chat", "status": "start", "dataset_id": dataset_id})
                chat = client.create_chat(name="ragflow-bench-doctor", dataset_ids=[dataset_id], llm_id=llm_id)
                session = client.create_session(chat["id"], name="doctor")
                answer = client.ask_chat(question="Summarize the probe document.", chat_id=chat["id"], session_id=session.get("id") or session.get("session_id"), llm_id=llm_id)
                payload["chat_answer_keys"] = sorted(answer.keys()) if isinstance(answer, dict) else []
                emit_progress(default_progress_printer, {"command": "doctor", "step": "chat", "status": "ok", "dataset_id": dataset_id, "chat_id": chat.get("id"), "session_id": session.get("id") or session.get("session_id")})
            else:
                emit_progress(default_progress_printer, {"command": "doctor", "step": "chat", "status": "skipped", "dataset_id": dataset_id})
                issues.append("RAGFLOW_LLM_ID not set; skipped chat creation/completion probe")
        except Exception as exc:  # noqa: BLE001
            emit_progress(default_progress_printer, {"command": "doctor", "step": "workflow_probe", "status": "error", "dataset_id": dataset_id, "exception": exc.__class__.__name__, "error": str(exc)})
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
    emit_progress(default_progress_printer, {"command": "doctor", "step": "complete", "status": "ok" if not issues else "issues", "count": len(issues)})


@app.command()
def ingest(config: str = typer.Option(..., help="Config YAML path")) -> None:
    emit_progress(default_progress_printer, {"command": "ingest", "step": "config_load", "status": "start", "path": config})
    cfg = load_config(config)
    emit_progress(default_progress_printer, {"command": "ingest", "step": "config_load", "status": "ok", "path": config})
    _exit_if_local_paths_missing(cfg)
    client = RagflowClient(cfg.ragflow)
    adapter = make_adapter(cfg)
    output_dir = cfg.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = ingest_documents(cfg, client, adapter, output_dir, progress_callback=default_progress_printer)
    console.print_json(json.dumps(registry.to_dict(), ensure_ascii=False))


@app.command()
def retrieve(config: str = typer.Option(..., help="Config YAML path")) -> None:
    emit_progress(default_progress_printer, {"command": "retrieve", "step": "config_load", "status": "start", "path": config})
    cfg = load_config(config)
    emit_progress(default_progress_printer, {"command": "retrieve", "step": "config_load", "status": "ok", "path": config})
    _exit_if_local_paths_missing(cfg)
    client = RagflowClient(cfg.ragflow)
    adapter = make_adapter(cfg)
    if cfg.dataset.strategy == DatasetStrategy.CREATE_AND_INGEST:
        output_dir = cfg.resolved_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        registry = ingest_documents(cfg, client, adapter, output_dir, progress_callback=default_progress_printer)
        dataset_id = registry.dataset_id
    else:
        dataset_id = resolve_dataset_id(cfg, client, adapter, progress_callback=default_progress_printer)
    question = adapter.load_questions()[0]
    emit_progress(default_progress_printer, {"command": "retrieve", "step": "retrieval", "status": "start", "question_id": getattr(question, "id", None), "dataset_id": dataset_id})
    payload = client.retrieve(
        question=question.question,
        dataset_ids=[dataset_id],
        page_size=cfg.retrieval.page_size,
        similarity_threshold=cfg.retrieval.similarity_threshold,
        vector_similarity_weight=cfg.retrieval.vector_similarity_weight,
        top_k=cfg.retrieval.top_k,
        rerank_id=cfg.retrieval.rerank_id,
    )
    emit_progress(default_progress_printer, {"command": "retrieve", "step": "retrieval", "status": "ok", "question_id": getattr(question, "id", None), "dataset_id": dataset_id, "count": payload.get("total") if isinstance(payload, dict) else None})
    console.print_json(json.dumps(payload, ensure_ascii=False))


@app.command()
def run(config: str = typer.Option(..., help="Config YAML path")) -> None:
    emit_progress(default_progress_printer, {"command": "run", "step": "config_load", "status": "start", "path": config})
    cfg = load_config(config)
    emit_progress(default_progress_printer, {"command": "run", "step": "config_load", "status": "ok", "path": config})
    _exit_if_local_paths_missing(cfg)
    output_dir = run_benchmark(cfg, RagflowClient(cfg.ragflow), progress_callback=default_progress_printer)
    _auto_judge_run_dir(output_dir)
    console.print(f"Run completed: {output_dir}")


@app.command("retry-failed")
def retry_failed(run_dir: str = typer.Option(..., help="Existing run directory containing results.jsonl and config.resolved.yaml")) -> None:
    emit_progress(default_progress_printer, {"command": "retry-failed", "step": "start", "status": "start", "path": run_dir})
    run_path = Path(run_dir)
    results_path = run_path / "results.jsonl"
    config_path = run_path / "config.resolved.yaml"
    if not results_path.exists():
        console.print(f"[red]Missing results file:[/red] {results_path}")
        raise typer.Exit(code=1)
    if not config_path.exists():
        console.print(f"[red]Missing resolved config:[/red] {config_path}")
        raise typer.Exit(code=1)

    original_rows = load_jsonl(results_path)
    emit_progress(default_progress_printer, {"command": "retry-failed", "step": "load_results", "status": "ok", "path": str(results_path), "count": len(original_rows)})
    failed_ids = [str(row.get("question_id")) for row in original_rows if _row_needs_retry(row)]
    if not failed_ids:
        console.print("No failed rows found; nothing to retry.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    retry_dir = run_path / "retries" / f"retry_failed_{stamp}"
    backup_dir = run_path / "backups" / f"retry_failed_{stamp}"

    cfg = _unredact_config_for_retry(load_config(config_path))
    emit_progress(default_progress_printer, {"command": "retry-failed", "step": "select_failed", "status": "ok", "count": len(failed_ids)})
    cfg.output.output_dir = str(retry_dir)
    _exit_if_local_paths_missing(cfg)

    retry_output_dir = run_benchmark(cfg, RagflowClient(cfg.ragflow), question_ids=set(failed_ids), progress_callback=default_progress_printer)
    retry_rows = load_jsonl(retry_output_dir / "results.jsonl")
    emit_progress(default_progress_printer, {"command": "retry-failed", "step": "load_retry_results", "status": "ok", "path": str(retry_output_dir / "results.jsonl"), "count": len(retry_rows)})
    merged_rows, replaced = _merge_retry_rows(original_rows, retry_rows)
    emit_progress(default_progress_printer, {"command": "retry-failed", "step": "merge", "status": "ok", "count": len(merged_rows), "replaced": replaced})

    copied = _backup_run_artifacts(run_path, backup_dir)
    emit_progress(default_progress_printer, {"command": "retry-failed", "step": "backup", "status": "ok", "path": str(backup_dir), "count": len(copied)})
    write_jsonl(results_path, merged_rows)
    emit_progress(default_progress_printer, {"command": "retry-failed", "step": "results_write", "status": "ok", "path": str(results_path), "count": len(merged_rows)})
    jsonl_to_csv(results_path, run_path / "results.csv")
    emit_progress(default_progress_printer, {"command": "retry-failed", "step": "csv_write", "status": "ok", "path": str(run_path / "results.csv")})
    summary = build_summary(benchmark=merged_rows[0].get("benchmark", "") if merged_rows else "", mode=cfg.benchmark.mode.value, rows=merged_rows)
    write_json(run_path / "summary.json", summary)
    emit_progress(default_progress_printer, {"command": "retry-failed", "step": "summary_write", "status": "ok", "path": str(run_path / "summary.json")})
    judge_retry_ids = set(failed_ids) | _judge_error_question_ids(run_path)
    _auto_judge_run_dir(run_path, resume=True, force_question_ids=judge_retry_ids)

    successful_retries = sum(1 for row in retry_rows if not _row_needs_retry(row))
    remaining_errors = sum(1 for row in merged_rows if _row_needs_retry(row))
    report = {
        "run_dir": str(run_path),
        "retry_output_dir": str(retry_output_dir),
        "backup_dir": str(backup_dir),
        "backed_up_files": copied,
        "failed_rows_selected": len(failed_ids),
        "retry_rows": len(retry_rows),
        "replaced_rows": replaced,
        "successful_retries": successful_retries,
        "remaining_errors": remaining_errors,
        "judge_retry_rows_selected": len(judge_retry_ids),
    }
    write_json(retry_output_dir / "retry_report.json", report)
    emit_progress(default_progress_printer, {"command": "retry-failed", "step": "complete", "status": "ok", "count": len(retry_rows), "replaced": replaced, "remaining_errors": remaining_errors})
    console.print_json(json.dumps(report, ensure_ascii=False))


@app.command()
def score(results: str = typer.Option(..., help="Path to results.jsonl")) -> None:
    path = Path(results)
    emit_progress(default_progress_printer, {"command": "score", "step": "load_results", "status": "start", "path": str(path)})
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    emit_progress(default_progress_printer, {"command": "score", "step": "load_results", "status": "ok", "path": str(path), "count": len(rows)})
    rescored = []
    for index, row in enumerate(rows, start=1):
        emit_progress(default_progress_printer, {"command": "score", "step": "row", "status": "start", "index": index, "total": len(rows), "question_id": row.get("question_id")})
        row = recompute_row_diagnostics(row)
        rescored.append(row)
        emit_progress(default_progress_printer, {"command": "score", "step": "row", "status": row["failure_type"], "index": index, "total": len(rows), "question_id": row.get("question_id")})
    summary = build_summary(benchmark=rows[0].get("benchmark", ""), mode="rescored", rows=rescored) if rows else {}
    write_json(path.with_name("summary.json"), summary)
    emit_progress(default_progress_printer, {"command": "score", "step": "summary_write", "status": "ok", "path": str(path.with_name("summary.json")), "count": len(rescored)})
    console.print_json(json.dumps(summary, ensure_ascii=False))


@app.command("recompute-diagnostics")
def recompute_diagnostics(
    run_dir: str = typer.Argument(..., help="Existing run directory containing results.jsonl"),
    output_prefix: str = typer.Option("diagnostics", "--output-prefix", help="Output file prefix written inside the run directory"),
) -> None:
    summary = recompute_run_diagnostics(run_dir, output_prefix=output_prefix)
    console.print_json(json.dumps(summary, ensure_ascii=False))


@app.command()
def judge(
    results: str = typer.Option(..., help="Path to results.jsonl"),
    config: str | None = typer.Option(None, help="Optional config YAML path with judge settings"),
    model: str | None = typer.Option(None, help="Optional judge model override, e.g. glm-4-flash or glm-4.7-flash"),
    output: str | None = typer.Option(None, help="Optional output path for judged JSONL"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume existing judge_results.jsonl by default; use --no-resume to overwrite."),
) -> None:
    cfg = load_config(config) if config else None
    settings = cfg.judge if cfg else JudgeSettings()
    if model:
        settings.model = model
    summary = judge_results_file(
        results_path=results,
        client=ZhipuJudgeClient(settings),
        output_path=output,
        resume=resume,
        progress_callback=judge_progress_printer,
    )
    console.print_json(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
