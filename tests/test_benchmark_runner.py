from ragflow_bench.config import AppConfig, BenchmarkConfig, BenchmarkKind, BenchmarkMode, DatasetConfig, DatasetStrategy
from ragflow_bench.execution import benchmark_runner
from ragflow_bench.ragflow.errors import RagflowAPIError
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
    assert row["failure_type"] == "answer_generation_error"


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


def test_run_benchmark_sleeps_between_nonfinal_questions(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_runner, "make_adapter", lambda cfg: _Adapter())
    monkeypatch.setattr(benchmark_runner, "run_retrieval", lambda client, cfg, dataset_id, question: {"chunks": [], "total": 0})
    delays = []
    events = []

    monkeypatch.setattr(benchmark_runner.random, "uniform", lambda a, b: 6.25)
    monkeypatch.setattr(benchmark_runner.time, "sleep", lambda seconds: delays.append(seconds))

    benchmark_runner.run_benchmark(_config(tmp_path), _Client(), progress_callback=events.append)

    assert delays == [6.25, 6.25]
    delay_events = [event for event in events if event.get("step") == "question_delay"]
    assert [event["delay"] for event in delay_events] == [6.25, 6.25]


def test_run_benchmark_retries_rate_limited_retrieval_before_continuing(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_runner, "make_adapter", lambda cfg: _Adapter())
    calls = {"count": 0}
    delays = []
    events = []

    def flaky_retrieval(client, cfg, dataset_id, question):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RagflowAPIError("HTTP 429 for /api/v1/retrieval", status_code=429)
        return {"chunks": [], "total": 0}

    monkeypatch.setattr(benchmark_runner, "run_retrieval", flaky_retrieval)
    monkeypatch.setattr(benchmark_runner.random, "uniform", lambda a, b: 17.0)
    monkeypatch.setattr(benchmark_runner.time, "sleep", lambda seconds: delays.append(seconds))

    output_dir = benchmark_runner.run_benchmark(_config(tmp_path), _Client(), question_ids={"2"}, progress_callback=events.append)

    assert calls["count"] == 2
    assert delays == [17.0]
    retry_events = [event for event in events if event.get("type") == "rate_limit_retry"]
    assert retry_events[0]["action"] == "retrieval"
    assert retry_events[0]["retry"] is True
    assert load_jsonl(output_dir / "results.jsonl")[0]["error"] is None


def test_run_benchmark_retries_rate_limited_chat_result(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    cfg.ragflow.llm_id = "model"
    monkeypatch.setattr(benchmark_runner, "make_adapter", lambda cfg: _Adapter())
    monkeypatch.setattr(benchmark_runner, "run_retrieval", lambda client, cfg, dataset_id, question: {"chunks": [], "total": 0})
    monkeypatch.setattr(benchmark_runner, "ensure_chat", lambda client, cfg, dataset_id, name: {"id": "chat1"})
    calls = {"count": 0}
    delays = []
    events = []

    def flaky_chat(client, cfg, chat_id, question, session_id=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"answer": "**ERROR**: GENERIC_ERROR - litellm.RateLimitError: rate limited"}
        return {"answer": "Answer 2"}

    monkeypatch.setattr(benchmark_runner, "run_chat", flaky_chat)
    monkeypatch.setattr(benchmark_runner.random, "uniform", lambda a, b: 16.0)
    monkeypatch.setattr(benchmark_runner.time, "sleep", lambda seconds: delays.append(seconds))

    output_dir = benchmark_runner.run_benchmark(cfg, _Client(), question_ids={"2"}, progress_callback=events.append)

    row = load_jsonl(output_dir / "results.jsonl")[0]
    assert calls["count"] == 2
    assert delays == [16.0]
    assert row["ragflow_answer"] == "Answer 2"
    assert row["error"] is None
    retry_events = [event for event in events if event.get("type") == "rate_limit_retry"]
    assert retry_events[0]["action"] == "chat"


def test_run_benchmark_exhausts_rate_limit_retries_and_records_error(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_runner, "make_adapter", lambda cfg: _Adapter())
    delays = []

    monkeypatch.setattr(benchmark_runner, "run_retrieval", lambda client, cfg, dataset_id, question: (_ for _ in ()).throw(RagflowAPIError("HTTP 429 for /api/v1/retrieval", status_code=429)))
    monkeypatch.setattr(benchmark_runner.random, "uniform", lambda a, b: 18.0)
    monkeypatch.setattr(benchmark_runner.time, "sleep", lambda seconds: delays.append(seconds))

    output_dir = benchmark_runner.run_benchmark(_config(tmp_path), _Client(), question_ids={"2"})

    row = load_jsonl(output_dir / "results.jsonl")[0]
    assert delays == [18.0, 18.0]
    assert "HTTP 429" in row["error"]
    assert row["failure_type"] == "answer_generation_error"


def test_audit_chunk_preserves_diagnostic_fields():
    from ragflow_bench.execution.benchmark_runner import audit_chunk

    chunk = {
        "id": "chunk1",
        "doc_id": "doc1",
        "document_name": "github_shard_000104.txt",
        "score": 0.9,
        "term_similarity": 0.7,
        "vector_similarity": 0.8,
        "positions": [[1, 2]],
        "metadata": {"source_uri": "src"},
    }

    audited = audit_chunk(chunk, rank=3, survived_final_context=True)

    assert audited["rank"] == 3
    assert audited["chunk_id"] == "chunk1"
    assert audited["document_name"] == "github_shard_000104.txt"
    assert audited["score"] == 0.9
    assert audited["term_similarity"] == 0.7
    assert audited["vector_similarity"] == 0.8
    assert audited["positions"] == [[1, 2]]
    assert audited["metadata"] == {"source_uri": "src"}
    assert audited["survived_final_context"] is True


def test_run_benchmark_persists_raw_and_final_context_diagnostics(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    cfg.ragflow.llm_id = "model"
    monkeypatch.setattr(benchmark_runner, "make_adapter", lambda cfg: _Adapter())
    monkeypatch.setattr(
        benchmark_runner,
        "run_retrieval",
        lambda client, cfg, dataset_id, question: {"chunks": [{"id": "raw1", "document_name": "source-2", "similarity": 0.9}]},
    )
    monkeypatch.setattr(benchmark_runner, "ensure_chat", lambda client, cfg, dataset_id, name: {"id": "chat1"})
    monkeypatch.setattr(
        benchmark_runner,
        "run_chat",
        lambda client, cfg, chat_id, question, session_id=None: {"answer": "Answer 2 [ID:0]", "reference": {"chunks": [{"id": "raw1", "document_name": "source-2"}]}},
    )

    output_dir = benchmark_runner.run_benchmark(cfg, _Client(), question_ids={"2"})

    row = load_jsonl(output_dir / "results.jsonl")[0]
    assert row["raw_retrieval_chunks"][0]["rank"] == 1
    assert row["final_context_chunks"][0]["chunk_id"] == "raw1"
    assert row["citation_chunk_ids"] == ["raw1"]
    assert row["prompt_chunk_count"] == 1
    assert row["chat_top_n"] == cfg.chat.top_n
    assert row["retrieval_page_size"] == cfg.retrieval.page_size


def test_exact_fact_prompt_mode_adds_prompt_without_budget_changes(tmp_path):
    from ragflow_bench.execution.chat_runner import build_prompt_config, EXACT_FACT_PROMPT

    cfg = _config(tmp_path)
    default_prompt = build_prompt_config(cfg)
    assert "system" not in default_prompt
    top_n = cfg.chat.top_n
    max_tokens = cfg.chat.max_tokens
    cfg.chat.prompt_mode = "exact_fact"
    exact_prompt = build_prompt_config(cfg)
    assert EXACT_FACT_PROMPT in exact_prompt["system"]
    assert cfg.chat.top_n == top_n
    assert cfg.chat.max_tokens == max_tokens
