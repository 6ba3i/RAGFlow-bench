import os

import yaml

from ragflow_bench import cli
from ragflow_bench.config import AppConfig, BenchmarkConfig, BenchmarkKind, BenchmarkMode, DatasetConfig, DatasetStrategy, JudgeSettings, RagflowConnectionConfig


def test_base_url_default_and_env_resolution(monkeypatch):
    monkeypatch.delenv("RAGFLOW_BASE_URL", raising=False)
    cfg = RagflowConnectionConfig()
    assert cfg.resolved_base_url() == "http://127.0.0.1:80"
    monkeypatch.setenv("RAGFLOW_BASE_URL", "http://example.com")
    assert cfg.resolved_base_url() == "http://example.com"


def test_dataset_id_required_for_reuse():
    try:
        AppConfig(
            benchmark=BenchmarkConfig(kind=BenchmarkKind.CUSTOM, mode=BenchmarkMode.SMOKE, custom={"corpus_dir":".","questions_path":"tests/test_config.py"}),
            dataset=DatasetConfig(strategy=DatasetStrategy.REUSE_EXISTING),
        )
    except ValueError as exc:
        assert "dataset.dataset_id" in str(exc)
    else:
        raise AssertionError("Expected validation failure")


def test_embedding_model_is_optional_and_preserved():
    cfg = AppConfig(
        benchmark=BenchmarkConfig(
            kind=BenchmarkKind.CUSTOM,
            mode=BenchmarkMode.SMOKE,
            custom={"corpus_dir": ".", "questions_path": "tests/test_config.py"},
        ),
        dataset=DatasetConfig(
            strategy=DatasetStrategy.CREATE_NEW,
            embedding_model="bge-m3",
        ),
    )

    assert cfg.dataset.embedding_model == "bge-m3"


def test_load_dotenv_populates_missing_env_vars(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        'RAGFLOW_API_KEY="secret-key"\nexport RAGFLOW_BASE_URL=http://example.test\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("RAGFLOW_API_KEY", raising=False)
    monkeypatch.delenv("RAGFLOW_BASE_URL", raising=False)

    cli._load_dotenv(env_path)

    assert os.environ["RAGFLOW_API_KEY"] == "secret-key"
    assert os.environ["RAGFLOW_BASE_URL"] == "http://example.test"


def test_load_dotenv_does_not_override_existing_env_vars(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("RAGFLOW_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("RAGFLOW_API_KEY", "from-shell")

    cli._load_dotenv(env_path)

    assert os.environ["RAGFLOW_API_KEY"] == "from-shell"


def test_resolved_for_output_is_yaml_serializable():
    cfg = AppConfig(
        benchmark=BenchmarkConfig(
            kind=BenchmarkKind.FRAMES,
            mode=BenchmarkMode.SMOKE,
            frames={"split": "test", "mapping_path": "frames_mapping.json", "local_corpus_dir": "."},
        ),
        dataset=DatasetConfig(
            strategy=DatasetStrategy.REUSE_EXISTING,
            dataset_id="ds1",
        ),
    )

    rendered = yaml.safe_dump(cfg.resolved_for_output(), sort_keys=False, allow_unicode=True)

    assert "frames" in rendered
    assert "smoke" in rendered


def test_resolved_output_dir_expands_timestamp_placeholder():
    cfg = AppConfig(
        benchmark=BenchmarkConfig(
            kind=BenchmarkKind.FRAMES,
            mode=BenchmarkMode.SMOKE,
            frames={"split": "test", "mapping_path": "frames_mapping.json", "local_corpus_dir": "."},
        ),
        dataset=DatasetConfig(
            strategy=DatasetStrategy.REUSE_EXISTING,
            dataset_id="ds1",
        ),
        output={"output_dir": "outputs/frames_smoke_<timestamp>"},
    )

    resolved = str(cfg.resolved_output_dir())

    assert resolved.startswith("outputs/frames_smoke_")
    assert "<timestamp>" not in resolved


def test_judge_settings_default_and_env_resolution(monkeypatch):
    for key in (
        "ZHIPU_BASE_URL",
        "ZHIPU_API_KEY",
        "ZHIPU_JUDGE_MODEL",
        "ZHIPU_JUDGE_TIMEOUT_SECONDS",
        "ZHIPU_JUDGE_MAX_RETRIES",
        "ZHIPU_JUDGE_BACKOFF_SECONDS",
        "ZHIPU_JUDGE_MAX_BACKOFF_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)
    cfg = JudgeSettings()
    assert cfg.resolved_base_url() == "https://open.bigmodel.cn/api/paas/v4"
    assert cfg.resolved_model() == "glm-4.7-flash"
    assert cfg.resolved_timeout_seconds() == 120
    assert cfg.resolved_max_retries() == 3
    assert cfg.resolved_backoff_seconds() == 10.0
    assert cfg.resolved_max_backoff_seconds() == 10.0
    monkeypatch.setenv("ZHIPU_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("ZHIPU_API_KEY", "secret")
    monkeypatch.setenv("ZHIPU_JUDGE_MODEL", "glm-4-flash")
    monkeypatch.setenv("ZHIPU_JUDGE_TIMEOUT_SECONDS", "180")
    monkeypatch.setenv("ZHIPU_JUDGE_MAX_RETRIES", "5")
    monkeypatch.setenv("ZHIPU_JUDGE_BACKOFF_SECONDS", "1.5")
    monkeypatch.setenv("ZHIPU_JUDGE_MAX_BACKOFF_SECONDS", "8.0")
    assert cfg.resolved_base_url() == "https://example.test/v1"
    assert cfg.resolved_api_key() == "secret"
    assert cfg.resolved_model() == "glm-4-flash"
    assert cfg.resolved_timeout_seconds() == 180
    assert cfg.resolved_max_retries() == 5
    assert cfg.resolved_backoff_seconds() == 1.5
    assert cfg.resolved_max_backoff_seconds() == 8.0


def test_rerank_id_defaults_to_absent_and_exact_fact_prompt_mode_default():
    from ragflow_bench.config import RetrievalSettings, ChatSettings

    assert RetrievalSettings().rerank_id is None
    assert ChatSettings().prompt_mode == "default"
