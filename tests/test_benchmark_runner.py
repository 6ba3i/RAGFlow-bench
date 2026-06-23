from ragflow_bench.config import AppConfig, BenchmarkConfig, BenchmarkKind, BenchmarkMode, DatasetConfig, DatasetStrategy
from ragflow_bench.execution import benchmark_runner
from ragflow_bench.reports.writers import load_jsonl


class _Question:
    def __init__(self, qid):
        self.id = qid
        self.question = f"Question {qid}?"
        self.gold_answer = f"Answer {qid}"
        self.expected_sources = []
        self.reasoning_types = []


class _Adapter:
    def load_questions(self):
        return [_Question("1"), _Question("2"), _Question("3")]


class _Client:
    def create_session(self, chat_id, name=None):
        return {"id": f"session-{name}"}


def _config(tmp_path):
    return AppConfig(
        benchmark=BenchmarkConfig(kind=BenchmarkKind.CUSTOM, mode=BenchmarkMode.SMOKE, custom={"questions_path": "tests/test_cli.py"}),
        ragflow={"api_key": "secret", "llm_id": None},
        dataset=DatasetConfig(strategy=DatasetStrategy.REUSE_EXISTING, dataset_id="ds1"),
        output={"output_dir": str(tmp_path / "out")},
    )


def test_run_benchmark_filters_question_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_runner, "make_adapter", lambda cfg: _Adapter())
    monkeypatch.setattr(benchmark_runner, "run_retrieval", lambda client, cfg, dataset_id, question: {"chunks": [], "total": 0})
    events = []

    output_dir = benchmark_runner.run_benchmark(_config(tmp_path), _Client(), question_ids={"2"}, progress_callback=events.append)

    rows = load_jsonl(output_dir / "results.jsonl")
    assert [row["question_id"] for row in rows] == ["2"]
    assert any(event.get("step") == "retrieval" and event.get("status") == "ok" for event in events)
    assert any(event.get("step") == "row_write" and event.get("question_id") == "2" for event in events)
    assert events[-1]["step"] == "complete"


def test_run_benchmark_rejects_unknown_question_id(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_runner, "make_adapter", lambda cfg: _Adapter())

    import pytest

    with pytest.raises(ValueError, match="Unknown question_id"):
        benchmark_runner.run_benchmark(_config(tmp_path), _Client(), question_ids={"missing"})


def test_run_benchmark_promotes_error_answers_to_error_field(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    cfg.ragflow.llm_id = "model"
    monkeypatch.setattr(benchmark_runner, "make_adapter", lambda cfg: _Adapter())
    monkeypatch.setattr(benchmark_runner, "run_retrieval", lambda client, cfg, dataset_id, question: {"chunks": [], "total": 0})
    monkeypatch.setattr(benchmark_runner, "ensure_chat", lambda client, cfg, dataset_id, name: {"id": "chat1"})
    monkeypatch.setattr(benchmark_runner, "run_chat", lambda client, cfg, chat_id, question, session_id=None: {"answer": "**ERROR**: GENERIC_ERROR - litellm.RateLimitError: rate limited"})

    output_dir = benchmark_runner.run_benchmark(cfg, _Client(), question_ids={"2"})

    row = load_jsonl(output_dir / "results.jsonl")[0]
    assert row["error"] == "**ERROR**: GENERIC_ERROR - litellm.RateLimitError: rate limited"
    assert row["failure_type"] == "error"


def test_run_benchmark_emits_chat_progress(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    cfg.ragflow.llm_id = "model"
    monkeypatch.setattr(benchmark_runner, "make_adapter", lambda cfg: _Adapter())
    monkeypatch.setattr(benchmark_runner, "run_retrieval", lambda client, cfg, dataset_id, question: {"chunks": [], "total": 0})
    monkeypatch.setattr(benchmark_runner, "ensure_chat", lambda client, cfg, dataset_id, name: {"id": "chat1"})
    monkeypatch.setattr(benchmark_runner, "run_chat", lambda client, cfg, chat_id, question, session_id=None: {"answer": "Answer 2"})
    events = []

    benchmark_runner.run_benchmark(cfg, _Client(), question_ids={"2"}, progress_callback=events.append)

    assert any(event.get("step") == "chat_create" and event.get("status") == "ok" for event in events)
    assert any(event.get("step") == "session_create" and event.get("status") == "ok" for event in events)
    assert any(event.get("step") == "chat" and event.get("question_id") == "2" and event.get("status") == "ok" for event in events)
