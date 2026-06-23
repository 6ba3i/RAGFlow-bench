from __future__ import annotations

import os
from copy import deepcopy
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class BenchmarkKind(str, Enum):
    FRAMES = "frames"
    ENTERPRISE_RAG_BENCH = "enterprise_rag_bench"
    CUSTOM = "custom"


class BenchmarkMode(str, Enum):
    SMOKE = "smoke"
    ORACLE = "oracle"
    GOLD_CORPUS = "gold-corpus"
    DISTRACTOR = "distractor"
    FULL = "full"


class DatasetStrategy(str, Enum):
    REUSE_EXISTING = "reuse_existing_dataset"
    CREATE_NEW = "create_new_dataset"
    CREATE_AND_INGEST = "create_new_dataset_and_ingest_documents"


class ParserSettings(BaseModel):
    chunk_token_num: int = 512
    delimiter: str = "\n"
    raptor: dict[str, Any] = Field(default_factory=lambda: {"use_raptor": False})
    graphrag: dict[str, Any] = Field(default_factory=lambda: {"use_graphrag": False})


class RetrievalSettings(BaseModel):
    top_k: int = 128
    page_size: int = 20
    similarity_threshold: float = 0.05
    vector_similarity_weight: float = 0.3


class ChatSettings(BaseModel):
    top_n: int = 8
    temperature: float = 0.0
    top_p: float = 0.1
    max_tokens: int = 128
    fresh_session_per_question: bool = True
    quote: bool = True
    refine_multiturn: bool = False


class JudgeSettings(BaseModel):
    provider: str = "zhipu"
    base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    base_url_env_var: str = "ZHIPU_BASE_URL"
    api_key: str | None = None
    api_key_env_var: str = "ZHIPU_API_KEY"
    model: str = "glm-4.7-flash"
    model_env_var: str = "ZHIPU_JUDGE_MODEL"
    temperature: float = 0.0
    include_confidence: bool = True
    timeout_seconds: int = 120
    timeout_env_var: str = "ZHIPU_JUDGE_TIMEOUT_SECONDS"
    max_retries: int = 3
    max_retries_env_var: str = "ZHIPU_JUDGE_MAX_RETRIES"
    backoff_seconds: float = 10.0
    backoff_env_var: str = "ZHIPU_JUDGE_BACKOFF_SECONDS"
    max_backoff_seconds: float = 10.0
    max_backoff_env_var: str = "ZHIPU_JUDGE_MAX_BACKOFF_SECONDS"

    def resolved_base_url(self) -> str:
        return os.getenv(self.base_url_env_var, self.base_url or "https://open.bigmodel.cn/api/paas/v4")

    def resolved_api_key(self) -> str | None:
        return self.api_key or os.getenv(self.api_key_env_var)

    def resolved_model(self) -> str:
        return os.getenv(self.model_env_var, self.model or "glm-4.7-flash")

    def resolved_timeout_seconds(self) -> int:
        return int(os.getenv(self.timeout_env_var, str(self.timeout_seconds)))

    def resolved_max_retries(self) -> int:
        return max(0, int(os.getenv(self.max_retries_env_var, str(self.max_retries))))

    def resolved_backoff_seconds(self) -> float:
        return max(0.0, float(os.getenv(self.backoff_env_var, str(self.backoff_seconds))))

    def resolved_max_backoff_seconds(self) -> float:
        return max(0.0, float(os.getenv(self.max_backoff_env_var, str(self.max_backoff_seconds))))


class RagflowConnectionConfig(BaseModel):
    base_url: str = "http://127.0.0.1:80"
    base_url_env_var: str = "RAGFLOW_BASE_URL"
    api_key: str | None = None
    api_key_env_var: str = "RAGFLOW_API_KEY"
    llm_id: str | None = None
    llm_id_env_var: str = "RAGFLOW_LLM_ID"

    def resolved_base_url(self) -> str:
        return os.getenv(self.base_url_env_var, self.base_url or "http://127.0.0.1:80")

    def resolved_api_key(self) -> str | None:
        return self.api_key or os.getenv(self.api_key_env_var)

    def resolved_llm_id(self) -> str | None:
        return self.llm_id or os.getenv(self.llm_id_env_var)


class DatasetConfig(BaseModel):
    strategy: DatasetStrategy = DatasetStrategy.REUSE_EXISTING
    dataset_id: str | None = None
    name: str | None = None
    description: str = ""
    embedding_model: str | None = None
    chunk_method: str = "naive"
    parser_config: ParserSettings = Field(default_factory=ParserSettings)


class FramesConfig(BaseModel):
    split: str = "test"
    mapping_path: str | None = None
    local_corpus_dir: str | None = None


class EnterpriseRAGBenchConfig(BaseModel):
    corpus_dir: str | None = None
    questions_path: str | None = None
    documents_manifest: str | None = None


class CustomBenchmarkConfig(BaseModel):
    corpus_dir: str | None = None
    questions_path: str | None = None
    documents_manifest: str | None = None


class BenchmarkConfig(BaseModel):
    kind: BenchmarkKind
    mode: BenchmarkMode = BenchmarkMode.SMOKE
    question_limit: int | None = None
    frames: FramesConfig | None = None
    enterprise_rag_bench: EnterpriseRAGBenchConfig | None = None
    custom: CustomBenchmarkConfig | None = None


class OutputConfig(BaseModel):
    output_dir: str | None = None


class AppConfig(BaseModel):
    benchmark: BenchmarkConfig
    ragflow: RagflowConnectionConfig = Field(default_factory=RagflowConnectionConfig)
    judge: JudgeSettings = Field(default_factory=JudgeSettings)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    chat: ChatSettings = Field(default_factory=ChatSettings)
    output: OutputConfig = Field(default_factory=OutputConfig)
    document_registry_path: str | None = None

    @model_validator(mode="after")
    def _validate_dataset_strategy(self) -> "AppConfig":
        if self.dataset.strategy == DatasetStrategy.REUSE_EXISTING and not self.dataset.dataset_id:
            if self.document_registry_path:
                return self
            raise ValueError("dataset.dataset_id is required when strategy=reuse_existing_dataset")
        return self

    def benchmark_section(self) -> BaseModel | None:
        if self.benchmark.kind == BenchmarkKind.FRAMES:
            return self.benchmark.frames
        if self.benchmark.kind == BenchmarkKind.ENTERPRISE_RAG_BENCH:
            return self.benchmark.enterprise_rag_bench
        return self.benchmark.custom

    def resolved_output_dir(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.output.output_dir:
            return Path(self.output.output_dir.replace("<timestamp>", stamp))
        return Path("outputs") / f"{self.benchmark.kind.value}_{self.benchmark.mode.value}_{stamp}"

    def default_output_dir(self) -> Path:
        return self.resolved_output_dir()

    def resolved_for_output(self) -> dict[str, Any]:
        payload = deepcopy(self.model_dump(mode="json"))
        payload["ragflow"]["base_url"] = self.ragflow.resolved_base_url()
        payload["ragflow"]["api_key"] = "***REDACTED***" if self.ragflow.resolved_api_key() else None
        payload["ragflow"]["llm_id"] = self.ragflow.resolved_llm_id()
        payload["judge"]["base_url"] = self.judge.resolved_base_url()
        payload["judge"]["api_key"] = "***REDACTED***" if self.judge.resolved_api_key() else None
        payload["judge"]["model"] = self.judge.resolved_model()
        payload["output"]["output_dir"] = str(self.resolved_output_dir())
        return payload


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    config = AppConfig.model_validate(raw)
    if config.output.output_dir is None:
        config.output.output_dir = str(config.default_output_dir())
    return config


def dump_config(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def ensure_local_paths_exist(config: AppConfig) -> list[str]:
    errors: list[str] = []
    section = config.benchmark_section()
    if section is None:
        return errors
    for field in ("mapping_path", "local_corpus_dir", "corpus_dir", "questions_path", "documents_manifest"):
        value = getattr(section, field, None)
        if value and not Path(value).exists():
            errors.append(f"missing path: {value}")
    if config.document_registry_path and not Path(config.document_registry_path).exists() and config.dataset.strategy == DatasetStrategy.REUSE_EXISTING:
        errors.append(f"missing document registry path: {config.document_registry_path}")
    return errors
