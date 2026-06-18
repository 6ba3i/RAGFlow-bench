from ragflow_bench.config import (
    AppConfig,
    BenchmarkConfig,
    BenchmarkKind,
    BenchmarkMode,
    DatasetConfig,
    DatasetStrategy,
)
from ragflow_bench.ingestion.ingest import resolve_dataset_id


class _DummyAdapter:
    pass


class _CapturingClient:
    def __init__(self):
        self.calls = []

    def create_dataset(self, **kwargs):
        self.calls.append(kwargs)
        return {"id": "ds-created"}


def test_resolve_dataset_id_passes_embedding_model_for_new_dataset():
    cfg = AppConfig(
        benchmark=BenchmarkConfig(
            kind=BenchmarkKind.CUSTOM,
            mode=BenchmarkMode.SMOKE,
            custom={"corpus_dir": ".", "questions_path": "tests/test_ingest.py"},
        ),
        dataset=DatasetConfig(
            strategy=DatasetStrategy.CREATE_NEW,
            name="bench-dataset",
            embedding_model="bge-m3",
        ),
    )
    client = _CapturingClient()

    dataset_id = resolve_dataset_id(cfg, client, _DummyAdapter())

    assert dataset_id == "ds-created"
    assert client.calls[0]["embedding_model"] == "bge-m3"

