import json

import pytest
import requests
import yaml

from ragflow_bench import cli
from ragflow_bench.config import JudgeSettings
from ragflow_bench.judge import ZhipuJudgeClient, build_judge_messages, is_excluded_infra_error, judge_results_file, summarize_judged_rows
from ragflow_bench.ragflow.errors import RagflowAPIError


class _FakeJudgeClient:
    def __init__(self, model="glm-4.7-flash"):
        self.model = model

    def judge_row(self, row):
        if is_excluded_infra_error(row):
            return {
                "score": None,
                "score_normalized": None,
                "verdict": "excluded",
                "reason": "infra error",
                "confidence": None,
                "matched_facts": [],
                "missing_or_wrong_facts": [],
                "excluded": True,
                "exclusion_reason": "infra_error",
            }
        if row.get("ragflow_answer") == "Paris":
            return {
                "score": 4,
                "score_normalized": 1.0,
                "verdict": "correct",
                "reason": "Same answer",
                "confidence": 0.93,
                "matched_facts": ["Paris"],
                "missing_or_wrong_facts": [],
                "excluded": False,
                "exclusion_reason": None,
            }
        if row.get("ragflow_answer") == "part of it":
            return {
                "score": 2,
                "score_normalized": 0.5,
                "verdict": "partial",
                "reason": "Partial answer",
                "confidence": 0.7,
                "matched_facts": ["one fact"],
                "missing_or_wrong_facts": ["missing fact"],
                "excluded": False,
                "exclusion_reason": None,
            }
        return {
            "score": 0,
            "score_normalized": 0.0,
            "verdict": "incorrect",
            "reason": "Wrong answer",
            "confidence": 0.22,
            "matched_facts": [],
            "missing_or_wrong_facts": ["wrong answer"],
            "excluded": False,
            "exclusion_reason": None,
        }


class _StubResponse:
    def __init__(self, status_code=200, payload=None, text=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text if text is not None else json.dumps(self._payload, ensure_ascii=False)
        self.headers = headers or {}

    def json(self):
        return self._payload


class _SequenceSession:
    def __init__(self, events):
        self.events = list(events)
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        event = self.events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event


RESULT_ROW = {
    "question": "Q",
    "gold_answer": "A",
    "ragflow_answer": "A",
}


def _judge_success_response():
    return _StubResponse(
        payload={
            "choices": [
                {
                    "message": {
                        "content": '{"score":4,"verdict":"correct","reason":"ok","confidence":1,"matched_facts":["A"],"missing_or_wrong_facts":[]}',
                    }
                }
            ]
        }
    )


def _make_client(**overrides):
    settings = JudgeSettings(api_key="secret", model="glm-4-flash", **overrides)
    client = ZhipuJudgeClient(settings)
    return client


def test_build_judge_messages_mentions_semantic_correctness():
    messages = build_judge_messages(
        {
            "question": "What is the capital of France?",
            "gold_answer": "Paris",
            "ragflow_answer": "The capital is Paris.",
        }
    )
    assert "semantic correctness" in messages[0]["content"]
    assert "0/2/4 rule book" in messages[1]["content"]
    assert "same essential factual answer" in messages[1]["content"]
    assert '"score" (one of 0, 2, 4)' in messages[1]["content"]


def test_judge_results_file_writes_artifacts(tmp_path):
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "benchmark": "frames",
                        "question": "What is the capital of France?",
                        "gold_answer": "Paris",
                        "ragflow_answer": "Paris",
                        "expected_sources": [],
                        "retrieved_source_uris": [],
                    }
                ),
                json.dumps(
                    {
                        "benchmark": "frames",
                        "question": "What is 2+2?",
                        "gold_answer": "4",
                        "ragflow_answer": "part of it",
                        "expected_sources": [],
                        "retrieved_source_uris": [],
                    }
                ),
                json.dumps(
                    {
                        "benchmark": "frames",
                        "question": "What failed?",
                        "gold_answer": "A",
                        "ragflow_answer": "**ERROR**: GENERIC_ERROR - litellm.RateLimitError: rate limited",
                        "expected_sources": [],
                        "retrieved_source_uris": [],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = judge_results_file(results_path=results_path, client=_FakeJudgeClient())

    judged_path = tmp_path / "judge_results.jsonl"
    summary_path = tmp_path / "judge_summary.json"
    assert judged_path.exists()
    assert (tmp_path / "judge_results.csv").exists()
    assert summary_path.exists()
    judged_rows = [json.loads(line) for line in judged_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert judged_rows[0]["judge_verdict"] == "correct"
    assert judged_rows[0]["judge_score"] == 4
    assert judged_rows[0]["judge_score_normalized"] == 1.0
    assert judged_rows[1]["judge_verdict"] == "partial"
    assert judged_rows[1]["judge_score"] == 2
    assert judged_rows[1]["judge_score_normalized"] == 0.5
    assert judged_rows[2]["judge_verdict"] == "excluded"
    assert judged_rows[2]["judge_excluded"] is True
    assert summary["answer_accuracy"] == 0.75
    assert summary["strict_accuracy"] == 0.5
    assert summary["partial_or_better_accuracy"] == 1.0
    assert summary["excluded_questions"] == 1
    assert summary["judge_model"] == "glm-4.7-flash"


def test_summarize_judged_rows_excludes_infra_errors_from_accuracy():
    summary = summarize_judged_rows(
        rows=[
            {"judge_score": 4, "judge_score_normalized": 1.0, "judge_verdict": "correct", "judge_confidence": 0.9, "judge_excluded": False},
            {"judge_score": 2, "judge_score_normalized": 0.5, "judge_verdict": "partial", "judge_confidence": 0.7, "judge_excluded": False},
            {"judge_score": 0, "judge_score_normalized": 0.0, "judge_verdict": "incorrect", "judge_confidence": 0.2, "judge_excluded": False},
            {"judge_score": None, "judge_score_normalized": None, "judge_verdict": "excluded", "judge_confidence": None, "judge_excluded": True},
        ],
        model="glm-4-flash",
    )
    assert summary["scorable_questions"] == 3
    assert summary["excluded_questions"] == 1
    assert summary["excluded_error_rate"] == 0.25
    assert summary["answer_accuracy"] == 0.5
    assert summary["strict_accuracy"] == 1 / 3
    assert summary["partial_or_better_accuracy"] == 2 / 3
    assert summary["judge_accuracy"] == summary["strict_accuracy"]
    assert summary["judge_score_counts"] == {"4": 1, "2": 1, "0": 1}
    assert summary["judge_model"] == "glm-4-flash"


def test_judge_command_uses_configured_model_and_allows_override(tmp_path, monkeypatch, capsys):
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(
        '{"benchmark":"frames","question":"Q","gold_answer":"A","ragflow_answer":"A","expected_sources":[],"retrieved_source_uris":[]}\n',
        encoding="utf-8",
    )
    config_path = tmp_path / "judge.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "kind": "custom",
                    "mode": "smoke",
                    "custom": {
                        "corpus_dir": ".",
                        "questions_path": "tests/test_judge.py",
                    },
                },
                "dataset": {
                    "strategy": "reuse_existing_dataset",
                    "dataset_id": "ds1",
                },
                "judge": {
                    "api_key": "secret",
                    "model": "glm-4-flash",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    created_models = []
    resume_values = []

    class _FakeCliJudgeClient:
        def __init__(self, settings):
            self.model = settings.resolved_model()
            created_models.append(self.model)

    def fake_judge_results_file(*, results_path, client, output_path=None, resume=True, progress_callback=None):
        resume_values.append(resume)
        assert progress_callback is not None
        return {"judge_model": client.model, "answer_accuracy": 1.0, "resume": resume}

    monkeypatch.setattr(cli, "ZhipuJudgeClient", _FakeCliJudgeClient)
    monkeypatch.setattr(cli, "judge_results_file", fake_judge_results_file)

    cli.judge(results=str(results_path), config=str(config_path), model=None, output=None, resume=True)
    cli.judge(results=str(results_path), config=str(config_path), model="glm-4.7-flash", output=None, resume=False)

    captured = capsys.readouterr()
    assert created_models == ["glm-4-flash", "glm-4.7-flash"]
    assert resume_values == [True, False]
    assert "answer_accuracy" in captured.out


def test_zhipu_judge_retries_read_timeout_then_succeeds(monkeypatch):
    client = _make_client(max_retries=2, backoff_seconds=0.01, max_backoff_seconds=0.01)
    session = _SequenceSession([requests.ReadTimeout("slow"), _judge_success_response()])
    sleeps = []
    client.session = session
    monkeypatch.setattr("ragflow_bench.judge.time.sleep", lambda seconds: sleeps.append(seconds))

    verdict = client.judge_row(RESULT_ROW)

    assert verdict["verdict"] == "correct"
    assert session.calls == 2
    assert sleeps == [0.01]


def test_zhipu_judge_uses_retry_after_when_present(monkeypatch):
    client = _make_client(max_retries=2, backoff_seconds=0.25, max_backoff_seconds=5.0)
    session = _SequenceSession([
        _StubResponse(status_code=429, text='{"error":{"code":"1305"}}', headers={"Retry-After": "3"}),
        _judge_success_response(),
    ])
    sleeps = []
    client.session = session
    monkeypatch.setattr("ragflow_bench.judge.time.sleep", lambda seconds: sleeps.append(seconds))

    verdict = client.judge_row(RESULT_ROW)

    assert verdict["verdict"] == "correct"
    assert session.calls == 2
    assert sleeps == [3.0]


def test_zhipu_judge_retries_429_then_succeeds(monkeypatch):
    client = _make_client(max_retries=2, backoff_seconds=0.25, max_backoff_seconds=0.25)
    session = _SequenceSession([
        _StubResponse(status_code=429, text='{"error":{"code":"1305"}}'),
        _judge_success_response(),
    ])
    sleeps = []
    client.session = session
    monkeypatch.setattr("ragflow_bench.judge.time.sleep", lambda seconds: sleeps.append(seconds))

    verdict = client.judge_row(RESULT_ROW)

    assert verdict["verdict"] == "correct"
    assert session.calls == 2
    assert sleeps == [0.25]


def test_zhipu_judge_raises_after_retry_exhaustion(monkeypatch):
    client = _make_client(max_retries=2, backoff_seconds=0.1, max_backoff_seconds=0.2)
    session = _SequenceSession([
        _StubResponse(status_code=503, text='{"error":"busy"}'),
        _StubResponse(status_code=503, text='{"error":"busy"}'),
        _StubResponse(status_code=503, text='{"error":"busy"}'),
    ])
    sleeps = []
    client.session = session
    monkeypatch.setattr("ragflow_bench.judge.time.sleep", lambda seconds: sleeps.append(seconds))

    with pytest.raises(RagflowAPIError) as exc:
        client.judge_row(RESULT_ROW)

    assert exc.value.status_code == 503
    assert session.calls == 3
    assert sleeps == [0.1, 0.1]


def test_zhipu_judge_does_not_retry_401(monkeypatch):
    client = _make_client(max_retries=3, backoff_seconds=0.1, max_backoff_seconds=0.1)
    session = _SequenceSession([_StubResponse(status_code=401, text='{"error":"bad auth"}')])
    sleeps = []
    client.session = session
    monkeypatch.setattr("ragflow_bench.judge.time.sleep", lambda seconds: sleeps.append(seconds))

    with pytest.raises(RagflowAPIError) as exc:
        client.judge_row(RESULT_ROW)

    assert exc.value.status_code == 401
    assert session.calls == 1
    assert sleeps == []


def test_is_excluded_infra_error_detects_error_answers_and_top_level_errors():
    assert is_excluded_infra_error({"ragflow_answer": "**ERROR**: GENERIC_ERROR - litellm.RateLimitError: rate limited"})
    assert is_excluded_infra_error({"error": "HTTPConnectionPool(host='x'): Read timed out."})
    assert not is_excluded_infra_error({"ragflow_answer": "The answer is Paris."})


def test_zhipu_judge_parses_partial_score_and_fact_lists(monkeypatch):
    client = _make_client()
    client.session = _SequenceSession([
        _StubResponse(
            payload={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({
                                "score": 2,
                                "verdict": "partial",
                                "reason": "One fact matched",
                                "confidence": "0.7",
                                "matched_facts": ["fact A"],
                                "missing_or_wrong_facts": ["fact B"],
                            })
                        }
                    }
                ]
            }
        )
    ])

    verdict = client.judge_row(RESULT_ROW)

    assert verdict["score"] == 2
    assert verdict["score_normalized"] == 0.5
    assert verdict["verdict"] == "partial"
    assert verdict["confidence"] == 0.7
    assert verdict["matched_facts"] == ["fact A"]
    assert verdict["missing_or_wrong_facts"] == ["fact B"]


def test_zhipu_judge_normalizes_invalid_score_to_zero(monkeypatch):
    client = _make_client()
    client.session = _SequenceSession([
        _StubResponse(
            payload={
                "choices": [
                    {
                        "message": {
                            "content": '{"score":3,"verdict":"correct","reason":"bad score","confidence":2}'
                        }
                    }
                ]
            }
        )
    ])

    verdict = client.judge_row(RESULT_ROW)

    assert verdict["score"] == 0
    assert verdict["score_normalized"] == 0.0
    assert verdict["verdict"] == "incorrect"
    assert verdict["confidence"] == 1.0


def test_zhipu_judge_excludes_infra_error_without_api_call():
    client = _make_client()
    client.session = _SequenceSession([])

    verdict = client.judge_row({"question": "Q", "gold_answer": "A", "ragflow_answer": "**ERROR**: GENERIC_ERROR - litellm.RateLimitError"})

    assert verdict["verdict"] == "excluded"
    assert verdict["excluded"] is True
    assert verdict["exclusion_reason"] == "infra_error"
    assert client.session.calls == 0



def test_judge_results_file_emits_progress_and_streams_rows(tmp_path):
    results_path = tmp_path / "results.jsonl"
    rows = [
        {"benchmark": "frames", "question_id": "1", "question": "Q1", "gold_answer": "A1", "ragflow_answer": "Paris"},
        {"benchmark": "frames", "question_id": "2", "question": "Q2", "gold_answer": "A2", "ragflow_answer": "part of it"},
    ]
    results_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    events = []

    summary = judge_results_file(results_path=results_path, client=_FakeJudgeClient(), progress_callback=events.append)

    judged_path = tmp_path / "judge_results.jsonl"
    judged_lines = [json.loads(line) for line in judged_path.read_text(encoding="utf-8").splitlines()]
    row_events = [event for event in events if event.get("type") == "judge_row"]
    assert [row["question_id"] for row in judged_lines] == ["1", "2"]
    assert [event["status"] for event in row_events] == ["judged", "judged"]
    assert summary["scorable_questions"] == 2


def test_judge_results_file_resumes_existing_rows(tmp_path):
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(
        "\n".join(
            [
                json.dumps({"benchmark": "frames", "question_id": "1", "question": "Q1", "gold_answer": "A1", "ragflow_answer": "Paris"}),
                json.dumps({"benchmark": "frames", "question_id": "2", "question": "Q2", "gold_answer": "A2", "ragflow_answer": "Paris"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    existing = {
        "question_id": "1",
        "judge_model": "glm-4.7-flash",
        "judge_score": 4,
        "judge_score_normalized": 1.0,
        "judge_verdict": "correct",
        "judge_reason": "old",
        "judge_confidence": 1.0,
        "judge_matched_facts": [],
        "judge_missing_or_wrong_facts": [],
        "judge_excluded": False,
        "judge_exclusion_reason": None,
        "judge_correct": True,
    }
    (tmp_path / "judge_results.jsonl").write_text(json.dumps(existing) + "\n", encoding="utf-8")
    events = []

    judge_results_file(results_path=results_path, client=_FakeJudgeClient(), resume=True, progress_callback=events.append)

    judged_rows = [json.loads(line) for line in (tmp_path / "judge_results.jsonl").read_text(encoding="utf-8").splitlines()]
    row_events = [event for event in events if event.get("type") == "judge_row"]
    assert [row["question_id"] for row in judged_rows] == ["1", "2"]
    assert row_events[0]["status"] == "skipped"
    assert row_events[1]["status"] == "judged"


def test_judge_results_file_no_resume_overwrites_existing_rows(tmp_path):
    results_path = tmp_path / "results.jsonl"
    results_path.write_text('{"benchmark":"frames","question_id":"1","question":"Q","gold_answer":"A","ragflow_answer":"Paris"}\n', encoding="utf-8")
    (tmp_path / "judge_results.jsonl").write_text('{"question_id":"old"}\n', encoding="utf-8")

    judge_results_file(results_path=results_path, client=_FakeJudgeClient(), resume=False)

    judged_rows = [json.loads(line) for line in (tmp_path / "judge_results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["question_id"] for row in judged_rows] == ["1"]


def test_judge_results_file_converts_client_exception_to_judge_error(tmp_path):
    class _FailingJudgeClient:
        model = "glm-4.7-flash"

        def judge_row(self, row):
            raise RagflowAPIError("Judge HTTP 429", status_code=429)

    results_path = tmp_path / "results.jsonl"
    results_path.write_text('{"benchmark":"frames","question_id":"1","question":"Q","gold_answer":"A","ragflow_answer":"A"}\n', encoding="utf-8")

    summary = judge_results_file(results_path=results_path, client=_FailingJudgeClient())

    judged = json.loads((tmp_path / "judge_results.jsonl").read_text(encoding="utf-8").strip())
    assert judged["judge_verdict"] == "judge_error"
    assert judged["judge_excluded"] is True
    assert judged["judge_exclusion_reason"] == "judge_error"
    assert summary["judge_error_questions"] == 1
    assert summary["scorable_questions"] == 0


def test_judge_results_file_rejects_malformed_resume_file(tmp_path):
    results_path = tmp_path / "results.jsonl"
    results_path.write_text('{"benchmark":"frames","question_id":"1","question":"Q","gold_answer":"A","ragflow_answer":"A"}\n', encoding="utf-8")
    (tmp_path / "judge_results.jsonl").write_text('{bad json}\n', encoding="utf-8")

    with pytest.raises(RagflowAPIError, match="Malformed existing judge output"):
        judge_results_file(results_path=results_path, client=_FakeJudgeClient(), resume=True)


def test_zhipu_judge_logs_429_attempts(monkeypatch):
    client = _make_client(max_retries=1, backoff_seconds=0.01, max_backoff_seconds=0.01)
    session = _SequenceSession([
        _StubResponse(status_code=429, text='{"error":{"code":"1305"}}'),
        _judge_success_response(),
    ])
    sleeps = []
    events = []
    client.session = session
    client.progress_callback = events.append
    monkeypatch.setattr("ragflow_bench.judge.time.sleep", lambda seconds: sleeps.append(seconds))

    verdict = client.judge_row({"question_id": "qid1", **RESULT_ROW})

    request_events = [event for event in events if event.get("type") == "judge_request"]
    assert verdict["verdict"] == "correct"
    assert request_events[0]["status_code"] == 429
    assert request_events[0]["retry"] is True
    assert request_events[0]["delay"] == 0.01
    assert request_events[1]["status_code"] == 200
    assert request_events[1]["retry"] is False
