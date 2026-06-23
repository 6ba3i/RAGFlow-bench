import json
from pathlib import Path

import pytest
import yaml

from ragflow_bench import cli
from ragflow_bench.config import AppConfig, BenchmarkConfig, BenchmarkKind, BenchmarkMode, DatasetConfig, DatasetStrategy


def test_exit_if_local_paths_missing_shows_friendly_message(capsys):
    cfg = AppConfig(
        benchmark=BenchmarkConfig(
            kind=BenchmarkKind.FRAMES,
            mode=BenchmarkMode.SMOKE,
            frames={
                "split": "test",
                "mapping_path": "frames_mapping.json",
                "local_corpus_dir": "./frames_corpus",
            },
        ),
        dataset=DatasetConfig(
            strategy=DatasetStrategy.REUSE_EXISTING,
            dataset_id="ds1",
        ),
    )

    with pytest.raises(cli.typer.Exit) as exc:
        cli._exit_if_local_paths_missing(cfg)

    captured = capsys.readouterr()
    assert exc.value.exit_code == 1
    assert "Missing required local benchmark files" in captured.out
    assert "missing path: frames_mapping.json" in captured.out
    assert "prepare-frames" in captured.out


class _FakeRetrieveClient:
    def __init__(self, connection):
        self.base_url = connection.resolved_base_url() if connection is not None else "http://example.test"
        self.retrieve_calls = []

    def retrieve(self, **kwargs):
        self.retrieve_calls.append(kwargs)
        return {"chunks": [], "total": 0}


class _FakeRegistry:
    def __init__(self, dataset_id):
        self.dataset_id = dataset_id

    def to_dict(self):
        return {"dataset_id": self.dataset_id, "documents": {}}


class _FakeQuestion:
    def __init__(self, question):
        self.question = question


class _FakeAdapter:
    def load_questions(self):
        return [_FakeQuestion("What is the question?")]


def _write_retrieve_config(tmp_path, strategy: str) -> str:
    config_path = tmp_path / "retrieve.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "kind": "custom",
                    "mode": "smoke",
                    "custom": {
                        "corpus_dir": ".",
                        "questions_path": "tests/test_cli.py",
                    },
                },
                "ragflow": {
                    "api_key": "secret",
                },
                "dataset": {
                    "strategy": strategy,
                    "dataset_id": "existing-ds" if strategy == DatasetStrategy.REUSE_EXISTING.value else None,
                    "name": "retrieve-dataset",
                },
                "output": {
                    "output_dir": str(tmp_path / "outputs"),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return str(config_path)


def test_retrieve_ingests_when_strategy_is_create_and_ingest(tmp_path, monkeypatch):
    config_path = _write_retrieve_config(tmp_path, DatasetStrategy.CREATE_AND_INGEST.value)
    client = _FakeRetrieveClient(None)
    ingest_calls = []
    resolve_calls = []

    def fake_client_factory(connection):
        return client

    def fake_ingest_documents(cfg, client_obj, adapter, output_dir, progress_callback=None):
        ingest_calls.append({"output_dir": output_dir, "client": client_obj, "adapter": adapter})
        return _FakeRegistry("ingested-ds")

    def fake_resolve_dataset_id(*args, **kwargs):
        resolve_calls.append((args, kwargs))
        return "resolved-ds"

    monkeypatch.setattr(cli, "RagflowClient", fake_client_factory)
    monkeypatch.setattr(cli, "make_adapter", lambda cfg: _FakeAdapter())
    monkeypatch.setattr(cli, "ingest_documents", fake_ingest_documents)
    monkeypatch.setattr(cli, "resolve_dataset_id", fake_resolve_dataset_id)

    cli.retrieve(config=config_path)

    assert len(ingest_calls) == 1
    assert ingest_calls[0]["output_dir"] == tmp_path / "outputs"
    assert resolve_calls == []
    assert client.retrieve_calls[0]["dataset_ids"] == ["ingested-ds"]


def test_retrieve_reuses_existing_dataset_without_ingest(tmp_path, monkeypatch):
    config_path = _write_retrieve_config(tmp_path, DatasetStrategy.REUSE_EXISTING.value)
    client = _FakeRetrieveClient(None)
    ingest_calls = []
    resolve_calls = []

    def fake_client_factory(connection):
        return client

    def fake_ingest_documents(*args, **kwargs):
        ingest_calls.append((args, kwargs))
        return _FakeRegistry("ingested-ds")

    def fake_resolve_dataset_id(*args, **kwargs):
        resolve_calls.append((args, kwargs))
        return "existing-ds"

    monkeypatch.setattr(cli, "RagflowClient", fake_client_factory)
    monkeypatch.setattr(cli, "make_adapter", lambda cfg: _FakeAdapter())
    monkeypatch.setattr(cli, "ingest_documents", fake_ingest_documents)
    monkeypatch.setattr(cli, "resolve_dataset_id", fake_resolve_dataset_id)

    cli.retrieve(config=config_path)

    assert ingest_calls == []
    assert len(resolve_calls) == 1
    assert client.retrieve_calls[0]["dataset_ids"] == ["existing-ds"]


def test_retrieve_creates_dataset_without_ingest_for_create_new(tmp_path, monkeypatch):
    config_path = _write_retrieve_config(tmp_path, DatasetStrategy.CREATE_NEW.value)
    client = _FakeRetrieveClient(None)
    ingest_calls = []
    resolve_calls = []

    def fake_client_factory(connection):
        return client

    def fake_ingest_documents(*args, **kwargs):
        ingest_calls.append((args, kwargs))
        return _FakeRegistry("ingested-ds")

    def fake_resolve_dataset_id(*args, **kwargs):
        resolve_calls.append((args, kwargs))
        return "created-ds"

    monkeypatch.setattr(cli, "RagflowClient", fake_client_factory)
    monkeypatch.setattr(cli, "make_adapter", lambda cfg: _FakeAdapter())
    monkeypatch.setattr(cli, "ingest_documents", fake_ingest_documents)
    monkeypatch.setattr(cli, "resolve_dataset_id", fake_resolve_dataset_id)

    cli.retrieve(config=config_path)

    assert ingest_calls == []
    assert len(resolve_calls) == 1
    assert client.retrieve_calls[0]["dataset_ids"] == ["created-ds"]



def _minimal_row(question_id, *, error=None, answer=None):
    return {
        "benchmark": "frames",
        "question_id": str(question_id),
        "question": f"Q{question_id}",
        "gold_answer": f"A{question_id}",
        "ragflow_answer": answer,
        "exact_match": bool(answer),
        "normalized_match": bool(answer),
        "expected_sources": [],
        "retrieved_document_ids": [],
        "retrieved_source_uris": [],
        "retrieved_chunk_ids": [],
        "retrieved_scores": [],
        "source_recall": 0.0,
        "reasoning_types": [],
        "failure_type": "error" if error else "correct",
        "raw_retrieval": {},
        "raw_response": {},
        "error": error,
    }


def test_merge_retry_rows_replaces_by_question_id_preserving_order():
    original = [_minimal_row("1"), _minimal_row("2", error="timeout"), _minimal_row("3")]
    retry = [_minimal_row("2", answer="A2")]

    merged, replaced = cli._merge_retry_rows(original, retry)

    assert replaced == 1
    assert [row["question_id"] for row in merged] == ["1", "2", "3"]
    assert merged[1]["ragflow_answer"] == "A2"
    assert merged[1]["error"] is None


def test_merge_retry_rows_rejects_unknown_question_id():
    with pytest.raises(ValueError, match="unknown question_id"):
        cli._merge_retry_rows([_minimal_row("1")], [_minimal_row("missing")])


def test_row_needs_retry_includes_litellm_error_answers():
    row = _minimal_row(
        "1",
        answer="**ERROR**: GENERIC_ERROR - litellm.RateLimitError: rate limited",
    )

    assert cli._row_needs_retry(row)


def test_row_needs_retry_includes_timeout_answers():
    row = _minimal_row(
        "1",
        answer="**ERROR**: GENERIC_ERROR - HTTPConnectionPool read timeout",
    )

    assert cli._row_needs_retry(row)


def test_row_needs_retry_ignores_normal_wrong_answers():
    row = _minimal_row("1", answer="not the gold answer")

    assert not cli._row_needs_retry(row)


def _write_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "benchmark": {"kind": "custom", "mode": "smoke", "custom": {"questions_path": "tests/test_cli.py"}},
                "ragflow": {"api_key": "***REDACTED***", "llm_id": None},
                "dataset": {"strategy": "reuse_existing_dataset", "dataset_id": "ds1"},
                "output": {"output_dir": str(run_dir)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    cli.write_jsonl(
        run_dir / "results.jsonl",
        [
            _minimal_row("1"),
            _minimal_row("2", error="timeout"),
            _minimal_row("3", error="timeout"),
            _minimal_row("4", answer="**ERROR**: GENERIC_ERROR - litellm.RateLimitError: rate limited"),
        ],
    )
    (run_dir / "results.csv").write_text("old csv", encoding="utf-8")
    (run_dir / "summary.json").write_text('{"old": true}', encoding="utf-8")
    return run_dir


def test_retry_failed_reruns_failed_ids_and_merges_in_place(tmp_path, monkeypatch):
    run_dir = _write_run_dir(tmp_path)
    observed = {}

    def fake_run_benchmark(cfg, client, question_ids=None, progress_callback=None):
        observed["output_dir"] = cfg.output.output_dir
        observed["api_key"] = cfg.ragflow.api_key
        observed["question_ids"] = question_ids
        retry_dir = Path(cfg.output.output_dir)
        retry_dir.mkdir(parents=True)
        cli.write_jsonl(
            retry_dir / "results.jsonl",
            [
                _minimal_row("2", answer="A2"),
                _minimal_row("3", error="timeout again"),
                _minimal_row("4", answer="A4"),
            ],
        )
        return retry_dir

    monkeypatch.setattr(cli, "RagflowClient", lambda connection: object())
    monkeypatch.setattr(cli, "run_benchmark", fake_run_benchmark)

    cli.retry_failed(run_dir=str(run_dir))

    assert observed["question_ids"] == {"2", "3", "4"}
    assert observed["api_key"] is None
    rows = cli.load_jsonl(run_dir / "results.jsonl")
    assert [row["question_id"] for row in rows] == ["1", "2", "3", "4"]
    assert len(rows) == 4
    assert rows[1]["ragflow_answer"] == "A2"
    assert rows[2]["error"] == "timeout again"
    assert rows[3]["ragflow_answer"] == "A4"
    assert list((run_dir / "backups").glob("retry_failed_*/results.jsonl"))
    retry_reports = list((run_dir / "retries").glob("retry_failed_*/retry_report.json"))
    assert retry_reports
    report = json.loads(retry_reports[0].read_text(encoding="utf-8"))
    assert report["remaining_errors"] == 1
    assert report["successful_retries"] == 2


def test_retry_failed_no_failed_rows_does_not_rewrite(tmp_path, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "benchmark": {"kind": "custom", "mode": "smoke", "custom": {"questions_path": "tests/test_cli.py"}},
                "ragflow": {"api_key": "secret"},
                "dataset": {"strategy": "reuse_existing_dataset", "dataset_id": "ds1"},
                "output": {"output_dir": str(run_dir)},
            }
        ),
        encoding="utf-8",
    )
    cli.write_jsonl(run_dir / "results.jsonl", [_minimal_row("1")])

    cli.retry_failed(run_dir=str(run_dir))

    assert "No failed rows found" in capsys.readouterr().out
    assert not (run_dir / "backups").exists()
    assert not (run_dir / "retries").exists()


def test_prepare_eragb_cli_prints_report(tmp_path, monkeypatch, capsys):
    observed = {}

    def fake_prepare(**kwargs):
        observed.update(kwargs)
        return {
            "split": "test",
            "document_count": 2,
            "shard_count": 1,
            "question_count": 1,
            "question_count_written": 1,
            "reference_granularity": "shard",
            "corpus_dir": str(tmp_path / "eragb" / "corpus"),
            "questions_path": str(tmp_path / "eragb" / "questions.jsonl"),
            "documents_manifest_path": str(tmp_path / "eragb" / "documents_manifest.json"),
        }

    monkeypatch.setattr(cli, "prepare_eragb_artifacts", fake_prepare)

    cli.prepare_eragb(
        output_dir=str(tmp_path / "eragb"),
        document_limit=2,
        question_limit=1,
        merge_documents=True,
        merge_target_bytes=123,
        merge_max_docs=4,
        filter_questions_with_missing_docs=True,
        reference_granularity="shard",
    )

    assert observed["output_dir"] == str(tmp_path / "eragb")
    assert observed["document_limit"] == 2
    assert observed["question_limit"] == 1
    assert observed["merge_documents"] is True
    assert observed["merge_target_bytes"] == 123
    assert observed["merge_max_docs"] == 4
    assert observed["filter_questions_with_missing_docs"] is True
    assert observed["reference_granularity"] == "shard"
    assert "prepare-eragb" in capsys.readouterr().out
