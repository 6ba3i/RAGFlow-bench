from ragflow_bench.ingestion.document_registry import DocumentRegistry


def test_registry_round_trip(tmp_path):
    registry = DocumentRegistry(dataset_id="ds1")
    registry.register("https://example.com/a", ragflow_document_id="doc1", title="A", source_type="html", metadata={"x":1})
    path = tmp_path / "registry.json"
    registry.save(path)
    loaded = DocumentRegistry.load(path)
    assert loaded.dataset_id == "ds1"
    assert loaded.get_document_id("https://example.com/a") == "doc1"
