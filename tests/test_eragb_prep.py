import json
from pathlib import Path

import pandas as pd

from ragflow_bench.benchmarks.eragb_prep import prepare_eragb_artifacts
from ragflow_bench.benchmarks.enterprise_rag_bench import EnterpriseRAGBenchAdapter
from ragflow_bench.config import AppConfig, BenchmarkConfig, BenchmarkKind, BenchmarkMode, DatasetConfig, DatasetStrategy


def test_prepare_eragb_artifacts_downloads_and_converts_parquet(tmp_path, monkeypatch):
    source_docs = tmp_path / "source_documents.parquet"
    source_questions = tmp_path / "source_questions.parquet"
    pd.DataFrame(
        [
            {
                "doc_id": "dsid_1",
                "source_type": "github",
                "title": "Multipart upload PR",
                "content": "Default upload limit is 10 MiB.",
            },
            {
                "doc_id": "dsid_2",
                "source_type": "slack",
                "title": "Launch thread",
                "content": "Launch is on Friday.",
            },
        ]
    ).to_parquet(source_docs)
    pd.DataFrame(
        [
            {
                "question_id": "qst_1",
                "question_type": "basic",
                "source_types": ["github"],
                "question": "What is the limit?",
                "expected_doc_ids": ["dsid_1"],
                "gold_answer": "10 MiB",
                "answer_facts": ["The upload limit is 10 MiB."],
            }
        ]
    ).to_parquet(source_questions)

    def fake_download(*, repo_id, repo_type, filename, local_dir, local_dir_use_symlinks, force_download, token):
        assert repo_id == "onyx-dot-app/EnterpriseRAG-Bench"
        assert repo_type == "dataset"
        assert local_dir_use_symlinks is False
        assert token is None
        return str(source_docs if "documents" in filename else source_questions)

    monkeypatch.setattr("ragflow_bench.benchmarks.eragb_prep.hf_hub_download", fake_download)

    report = prepare_eragb_artifacts(output_dir=tmp_path / "eragb")

    corpus_file = tmp_path / "eragb" / "corpus" / "github" / "dsid_1.txt"
    assert corpus_file.exists()
    assert "Document ID: dsid_1" in corpus_file.read_text(encoding="utf-8")
    questions = [json.loads(line) for line in (tmp_path / "eragb" / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert questions[0]["id"] == "qst_1"
    assert questions[0]["expected_sources"] == ["dsid_1"]
    manifest = json.loads((tmp_path / "eragb" / "documents_manifest.json").read_text(encoding="utf-8"))
    assert manifest["github/dsid_1.txt"]["source_uri"] == "dsid_1"
    assert report["document_count"] == 2
    assert report["question_count"] == 1


def test_prepare_eragb_artifacts_reuses_existing_parquet_and_honors_limits(tmp_path, monkeypatch):
    raw_docs = tmp_path / "eragb" / "raw" / "documents"
    raw_questions = tmp_path / "eragb" / "raw" / "questions"
    raw_docs.mkdir(parents=True)
    raw_questions.mkdir(parents=True)
    pd.DataFrame(
        [
            {"doc_id": "dsid_1", "source_type": "github", "title": "One", "content": "one"},
            {"doc_id": "dsid_2", "source_type": "slack", "title": "Two", "content": "two"},
        ]
    ).to_parquet(raw_docs / "test.parquet")
    pd.DataFrame(
        [
            {"question_id": "qst_1", "question": "one?", "expected_doc_ids": ["dsid_1"], "gold_answer": "one"},
            {"question_id": "qst_2", "question": "two?", "expected_doc_ids": ["dsid_2"], "gold_answer": "two"},
        ]
    ).to_parquet(raw_questions / "test.parquet")

    def fail_download(**kwargs):
        raise AssertionError("should reuse existing parquet")

    monkeypatch.setattr("ragflow_bench.benchmarks.eragb_prep.hf_hub_download", fail_download)

    report = prepare_eragb_artifacts(output_dir=tmp_path / "eragb", document_limit=1, question_limit=1)

    assert report["document_count"] == 1
    assert report["question_count"] == 1
    assert not (tmp_path / "eragb" / "corpus" / "slack" / "dsid_2.txt").exists()


def test_enterprise_adapter_reads_onyx_and_manifest_relative_paths(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "github").mkdir(parents=True)
    (corpus / "github" / "dsid_1.txt").write_text("body", encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps(
            {
                "question_id": "qst_1",
                "question_type": "basic",
                "question": "What is the limit?",
                "expected_doc_ids": ["dsid_1"],
                "gold_answer": "10 MiB",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "documents_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "github/dsid_1.txt": {
                    "id": "dsid_1",
                    "source_uri": "dsid_1",
                    "title": "Multipart upload PR",
                    "source_type": "github",
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = AppConfig(
        benchmark=BenchmarkConfig(
            kind=BenchmarkKind.ENTERPRISE_RAG_BENCH,
            mode=BenchmarkMode.SMOKE,
            enterprise_rag_bench={
                "corpus_dir": str(corpus),
                "questions_path": str(questions),
                "documents_manifest": str(manifest),
            },
        ),
        dataset=DatasetConfig(strategy=DatasetStrategy.CREATE_NEW),
    )

    adapter = EnterpriseRAGBenchAdapter(cfg)
    loaded_questions = adapter.load_questions()
    documents = list(adapter.iter_documents())

    assert loaded_questions[0].id == "qst_1"
    assert loaded_questions[0].expected_sources == ["dsid_1"]
    assert loaded_questions[0].reasoning_types == ["basic"]
    assert documents[0].source_uri == "dsid_1"
    assert documents[0].title == "Multipart upload PR"


def test_prepare_eragb_merged_mode_writes_chunk_safe_shards_and_maps_questions(tmp_path, monkeypatch):
    source_docs = tmp_path / "source_documents.parquet"
    source_questions = tmp_path / "source_questions.parquet"
    pd.DataFrame(
        [
            {"doc_id": "dsid_b", "source_type": "slack", "title": "B", "content": "bravo"},
            {"doc_id": "dsid_a", "source_type": "slack", "title": "A", "content": "alpha"},
            {"doc_id": "dsid_c", "source_type": "github", "title": "C", "content": "charlie"},
        ]
    ).to_parquet(source_docs)
    pd.DataFrame(
        [
            {"question_id": "qst_1", "question": "alpha?", "expected_doc_ids": ["dsid_a"], "gold_answer": "alpha"},
            {"question_id": "qst_2", "question": "missing?", "expected_doc_ids": ["missing"], "gold_answer": "missing"},
        ]
    ).to_parquet(source_questions)

    monkeypatch.setattr(
        "ragflow_bench.benchmarks.eragb_prep.hf_hub_download",
        lambda **kwargs: str(source_docs if "documents" in kwargs["filename"] else source_questions),
    )

    report = prepare_eragb_artifacts(
        output_dir=tmp_path / "eragb",
        merge_documents=True,
        merge_max_docs=2,
        merge_target_bytes=10_000,
        filter_questions_with_missing_docs=True,
    )

    assert report["merge_documents"] is True
    assert report["reference_granularity"] == "shard"
    assert report["question_count_input"] == 2
    assert report["question_count_written"] == 1
    assert report["question_count_dropped_missing_docs"] == 1
    assert report["parser_delimiter"] == "`<<<ERAGB_DOC_BOUNDARY>>>`"
    parser_config = (tmp_path / "eragb" / "parser_config.merged.yaml").read_text(encoding="utf-8")
    assert 'delimiter: "`<<<ERAGB_DOC_BOUNDARY>>>`"' in parser_config

    slack_shard = tmp_path / "eragb" / "corpus" / "slack" / "slack_shard_000001.txt"
    assert slack_shard.exists()
    shard_text = slack_shard.read_text(encoding="utf-8")
    assert shard_text.count("<<<ERAGB_DOC_BOUNDARY>>>") == 2
    assert shard_text.index("Document ID: dsid_a") < shard_text.index("Document ID: dsid_b")
    assert "Content:\nalpha" in shard_text

    manifest = json.loads((tmp_path / "eragb" / "documents_manifest.json").read_text(encoding="utf-8"))
    assert manifest["slack/slack_shard_000001.txt"]["contained_doc_count"] == 2
    assert "contained_doc_ids" not in manifest["slack/slack_shard_000001.txt"]
    shard_manifest = json.loads((tmp_path / "eragb" / "shard_manifest.json").read_text(encoding="utf-8"))
    assert shard_manifest["slack/slack_shard_000001.txt"]["contained_doc_ids"] == ["dsid_a", "dsid_b"]
    doc_map = json.loads((tmp_path / "eragb" / "doc_id_to_shard.json").read_text(encoding="utf-8"))
    assert doc_map["dsid_a"] == "eragb-shard://slack/slack_shard_000001.txt"
    questions = [json.loads(line) for line in (tmp_path / "eragb" / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(questions) == 1
    assert questions[0]["expected_sources"] == ["eragb-shard://slack/slack_shard_000001.txt"]
    assert questions[0]["expected_doc_ids"] == ["dsid_a"]


def test_prepare_eragb_merged_mode_can_disable_reference_sources(tmp_path, monkeypatch):
    source_docs = tmp_path / "source_documents.parquet"
    source_questions = tmp_path / "source_questions.parquet"
    pd.DataFrame([{"doc_id": "dsid_1", "source_type": "slack", "title": "One", "content": "one"}]).to_parquet(source_docs)
    pd.DataFrame([{"question_id": "qst_1", "question": "one?", "expected_doc_ids": ["dsid_1"], "gold_answer": "one"}]).to_parquet(source_questions)
    monkeypatch.setattr(
        "ragflow_bench.benchmarks.eragb_prep.hf_hub_download",
        lambda **kwargs: str(source_docs if "documents" in kwargs["filename"] else source_questions),
    )

    prepare_eragb_artifacts(output_dir=tmp_path / "eragb", merge_documents=True, reference_granularity="none")

    questions = [json.loads(line) for line in (tmp_path / "eragb" / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert questions[0]["expected_sources"] == []
    assert questions[0]["reference_granularity"] == "none"


def test_prepare_eragb_rejects_shard_references_without_merge(tmp_path):
    try:
        prepare_eragb_artifacts(output_dir=tmp_path / "eragb", reference_granularity="shard")
    except ValueError as exc:
        assert "requires merge_documents" in str(exc)
    else:
        raise AssertionError("Expected validation failure")
