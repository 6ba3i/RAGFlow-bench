import os

from ragflow_bench import cli
from ragflow_bench.config import AppConfig, BenchmarkConfig, BenchmarkKind, BenchmarkMode, DatasetConfig, DatasetStrategy, RagflowConnectionConfig


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
