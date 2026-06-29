import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from ragflow_bench.benchmarks.eragb_prep import ERAGBDownloadError, _download_hf_file, prepare_eragb_artifacts
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

    monkeypatch.setattr(
        "ragflow_bench.benchmarks.eragb_prep.HfApi.list_repo_files",
        lambda self, **kwargs: ["data/documents/test.parquet", "data/questions/test.parquet"],
    )

    def fake_download(*, repo_id, repo_type, filename, local_dir, force_download, token):
        assert repo_id == "onyx-dot-app/EnterpriseRAG-Bench"
        assert repo_type == "dataset"
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
    assert report["include_only_question_docs"] is False
    assert report["required_doc_count"] == 1
    assert report["document_count_before_question_filter"] == 2
    assert report["document_count_after_question_filter"] == 2
    assert report["question_doc_filter_missing_count"] == 0


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


def test_prepare_eragb_include_only_question_docs_filters_to_selected_question_docs(tmp_path, monkeypatch):
    raw_docs = tmp_path / "eragb" / "raw" / "documents"
    raw_questions = tmp_path / "eragb" / "raw" / "questions"
    raw_docs.mkdir(parents=True)
    raw_questions.mkdir(parents=True)
    pd.DataFrame(
        [
            {"doc_id": "dsid_1", "source_type": "github", "title": "One", "content": "one"},
            {"doc_id": "dsid_2", "source_type": "slack", "title": "Two", "content": "two"},
            {"doc_id": "dsid_3", "source_type": "confluence", "title": "Three", "content": "three"},
        ]
    ).to_parquet(raw_docs / "test.parquet")
    pd.DataFrame(
        [
            {"question_id": "qst_1", "question": "two?", "expected_doc_ids": ["dsid_2"], "gold_answer": "two"},
            {"question_id": "qst_2", "question": "three?", "expected_doc_ids": ["dsid_3"], "gold_answer": "three"},
        ]
    ).to_parquet(raw_questions / "test.parquet")
    monkeypatch.setattr(
        "ragflow_bench.benchmarks.eragb_prep.hf_hub_download",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should reuse existing parquet")),
    )

    report = prepare_eragb_artifacts(
        output_dir=tmp_path / "eragb",
        question_limit=1,
        include_only_question_docs=True,
    )

    assert (tmp_path / "eragb" / "corpus" / "slack" / "dsid_2.txt").exists()
    assert not (tmp_path / "eragb" / "corpus" / "github" / "dsid_1.txt").exists()
    assert not (tmp_path / "eragb" / "corpus" / "confluence" / "dsid_3.txt").exists()
    assert report["include_only_question_docs"] is True
    assert report["required_doc_count"] == 1
    assert report["document_count_before_question_filter"] == 3
    assert report["document_count_after_question_filter"] == 1
    assert report["document_count"] == 1
    assert report["question_doc_filter_missing_count"] == 0
    questions = [json.loads(line) for line in (tmp_path / "eragb" / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(questions) == 1
    assert questions[0]["expected_doc_ids"] == ["dsid_2"]
    assert questions[0]["expected_sources"] == ["dsid_2"]


def test_prepare_eragb_include_only_question_docs_applies_document_limit_after_filter(tmp_path, monkeypatch):
    raw_docs = tmp_path / "eragb" / "raw" / "documents"
    raw_questions = tmp_path / "eragb" / "raw" / "questions"
    raw_docs.mkdir(parents=True)
    raw_questions.mkdir(parents=True)
    pd.DataFrame(
        [
            {"doc_id": "dsid_a", "source_type": "github", "title": "A", "content": "a"},
            {"doc_id": "dsid_b", "source_type": "github", "title": "B", "content": "b"},
            {"doc_id": "dsid_c", "source_type": "github", "title": "C", "content": "c"},
        ]
    ).to_parquet(raw_docs / "test.parquet")
    pd.DataFrame(
        [
            {
                "question_id": "qst_1",
                "question": "a and b?",
                "expected_doc_ids": ["dsid_a", "dsid_b"],
                "gold_answer": "a b",
            }
        ]
    ).to_parquet(raw_questions / "test.parquet")
    monkeypatch.setattr(
        "ragflow_bench.benchmarks.eragb_prep.hf_hub_download",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should reuse existing parquet")),
    )

    report = prepare_eragb_artifacts(
        output_dir=tmp_path / "eragb",
        include_only_question_docs=True,
        document_limit=1,
        filter_questions_with_missing_docs=True,
    )

    assert report["required_doc_count"] == 2
    assert report["document_count_before_question_filter"] == 3
    assert report["document_count_after_question_filter"] == 2
    assert report["document_count"] == 1
    assert report["question_count_written"] == 0
    assert report["question_count_dropped_missing_docs"] == 1
    assert report["question_doc_filter_missing_count"] == 0


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


def test_enterprise_adapter_uses_manifest_source_uri_with_relative_corpus_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    corpus = Path("relative_corpus")
    (corpus / "github").mkdir(parents=True)
    (corpus / "github" / "dsid_1.txt").write_text("body", encoding="utf-8")
    questions = Path("questions.jsonl")
    questions.write_text(json.dumps({"question_id": "qst_1", "question": "q?"}) + "\n", encoding="utf-8")
    manifest = Path("documents_manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "github/dsid_1.txt": {
                    "id": "dsid_1",
                    "source_uri": "manifest-source-uri",
                    "title": "Manifest title",
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

    documents = list(EnterpriseRAGBenchAdapter(cfg).iter_documents())

    assert documents[0].source_uri == "manifest-source-uri"
    assert documents[0].title == "Manifest title"


def test_enterprise_adapter_file_uri_fallback_allows_relative_corpus_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    corpus = Path("relative_corpus")
    corpus.mkdir()
    (corpus / "doc.txt").write_text("body", encoding="utf-8")
    questions = Path("questions.jsonl")
    questions.write_text(json.dumps({"question_id": "qst_1", "question": "q?"}) + "\n", encoding="utf-8")
    cfg = AppConfig(
        benchmark=BenchmarkConfig(
            kind=BenchmarkKind.ENTERPRISE_RAG_BENCH,
            mode=BenchmarkMode.SMOKE,
            enterprise_rag_bench={"corpus_dir": str(corpus), "questions_path": str(questions)},
        ),
        dataset=DatasetConfig(strategy=DatasetStrategy.CREATE_NEW),
    )

    documents = list(EnterpriseRAGBenchAdapter(cfg).iter_documents())

    assert documents[0].source_uri == (tmp_path / "relative_corpus" / "doc.txt").as_uri()


def test_enterprise_adapter_load_questions_handles_array_fields(tmp_path):
    questions = tmp_path / "questions.parquet"
    pd.DataFrame(
        [
            {
                "question_id": "qst_1",
                "question": "q?",
                "expected_sources": np.array(["source_1"]),
                "expected_doc_ids": np.array(["ignored_lower_precedence"]),
                "reasoning_types": np.array(["multi_hop"]),
                "gold_answer": "a",
            }
        ]
    ).to_parquet(questions)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    cfg = AppConfig(
        benchmark=BenchmarkConfig(
            kind=BenchmarkKind.ENTERPRISE_RAG_BENCH,
            mode=BenchmarkMode.SMOKE,
            enterprise_rag_bench={"corpus_dir": str(corpus), "questions_path": str(questions)},
        ),
        dataset=DatasetConfig(strategy=DatasetStrategy.CREATE_NEW),
    )

    loaded = EnterpriseRAGBenchAdapter(cfg).load_questions()

    assert loaded[0].expected_sources == ["source_1"]
    assert loaded[0].reasoning_types == ["multi_hop"]


def test_prepare_eragb_handles_array_expected_sources(tmp_path, monkeypatch):
    source_docs = tmp_path / "source_documents.parquet"
    source_questions = tmp_path / "source_questions.parquet"
    pd.DataFrame([{"doc_id": "dsid_1", "source_type": "slack", "title": "One", "content": "one"}]).to_parquet(source_docs)
    pd.DataFrame(
        [
            {
                "question_id": "qst_1",
                "question": "one?",
                "expected_sources": np.array(["dsid_1"]),
                "expected_doc_ids": np.array(["ignored_lower_precedence"]),
                "reasoning_types": np.array(["basic"]),
                "gold_answer": "one",
            }
        ]
    ).to_parquet(source_questions)
    monkeypatch.setattr(
        "ragflow_bench.benchmarks.eragb_prep.HfApi.list_repo_files",
        lambda self, **kwargs: ["data/documents/test.parquet", "data/questions/test.parquet"],
    )
    monkeypatch.setattr(
        "ragflow_bench.benchmarks.eragb_prep.hf_hub_download",
        lambda **kwargs: str(source_docs if "documents" in kwargs["filename"] else source_questions),
    )

    prepare_eragb_artifacts(output_dir=tmp_path / "eragb")

    questions = [json.loads(line) for line in (tmp_path / "eragb" / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert questions[0]["expected_doc_ids"] == ["dsid_1"]
    assert questions[0]["expected_sources"] == ["dsid_1"]
    assert questions[0]["reasoning_types"] == ["basic"]


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
        "ragflow_bench.benchmarks.eragb_prep.HfApi.list_repo_files",
        lambda self, **kwargs: ["data/documents/test.parquet", "data/questions/test.parquet"],
    )
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
        "ragflow_bench.benchmarks.eragb_prep.HfApi.list_repo_files",
        lambda self, **kwargs: ["data/documents/test.parquet", "data/questions/test.parquet"],
    )
    monkeypatch.setattr(
        "ragflow_bench.benchmarks.eragb_prep.hf_hub_download",
        lambda **kwargs: str(source_docs if "documents" in kwargs["filename"] else source_questions),
    )

    prepare_eragb_artifacts(output_dir=tmp_path / "eragb", merge_documents=True, reference_granularity="none")

    questions = [json.loads(line) for line in (tmp_path / "eragb" / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert questions[0]["expected_sources"] == []
    assert questions[0]["reference_granularity"] == "none"


def test_prepare_eragb_include_only_question_docs_filters_merged_shards(tmp_path, monkeypatch):
    source_docs = tmp_path / "source_documents.parquet"
    source_questions = tmp_path / "source_questions.parquet"
    pd.DataFrame(
        [
            {"doc_id": "gh1", "source_type": "github", "title": "GH", "content": "github"},
            {"doc_id": "sl1", "source_type": "slack", "title": "Slack 1", "content": "slack one"},
            {"doc_id": "sl2", "source_type": "slack", "title": "Slack 2", "content": "slack two"},
        ]
    ).to_parquet(source_docs)
    pd.DataFrame([{"question_id": "qst_1", "question": "slack two?", "expected_doc_ids": ["sl2"], "gold_answer": "slack two"}]).to_parquet(source_questions)
    monkeypatch.setattr(
        "ragflow_bench.benchmarks.eragb_prep.HfApi.list_repo_files",
        lambda self, **kwargs: ["data/documents/test.parquet", "data/questions/test.parquet"],
    )
    monkeypatch.setattr(
        "ragflow_bench.benchmarks.eragb_prep.hf_hub_download",
        lambda **kwargs: str(source_docs if "documents" in kwargs["filename"] else source_questions),
    )

    report = prepare_eragb_artifacts(
        output_dir=tmp_path / "eragb",
        merge_documents=True,
        include_only_question_docs=True,
        question_limit=1,
        merge_max_docs=10,
    )

    assert report["document_count"] == 1
    assert report["shard_count"] == 1
    assert report["document_count_before_question_filter"] == 3
    assert report["document_count_after_question_filter"] == 1
    slack_shard = tmp_path / "eragb" / "corpus" / "slack" / "slack_shard_000001.txt"
    assert slack_shard.exists()
    shard_text = slack_shard.read_text(encoding="utf-8")
    assert "Document ID: sl2" in shard_text
    assert "Document ID: gh1" not in shard_text
    assert "Document ID: sl1" not in shard_text
    doc_map = json.loads((tmp_path / "eragb" / "doc_id_to_shard.json").read_text(encoding="utf-8"))
    assert doc_map == {"sl2": "eragb-shard://slack/slack_shard_000001.txt"}
    shard_manifest = json.loads((tmp_path / "eragb" / "shard_manifest.json").read_text(encoding="utf-8"))
    assert shard_manifest["slack/slack_shard_000001.txt"]["contained_doc_ids"] == ["sl2"]
    questions = [json.loads(line) for line in (tmp_path / "eragb" / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert questions[0]["expected_sources"] == ["eragb-shard://slack/slack_shard_000001.txt"]


def test_prepare_eragb_verifies_required_hf_paths_before_download(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ragflow_bench.benchmarks.eragb_prep.HfApi.list_repo_files",
        lambda self, **kwargs: ["data/documents/test.parquet"],
    )

    with pytest.raises(ERAGBDownloadError) as exc_info:
        prepare_eragb_artifacts(output_dir=tmp_path / "eragb")

    message = str(exc_info.value)
    assert "onyx-dot-app/EnterpriseRAG-Bench" in message
    assert "data/questions/test.parquet" in message
    assert "required parquet files are missing" in message


def test_prepare_eragb_reports_download_failure_after_path_verification(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ragflow_bench.benchmarks.eragb_prep.HfApi.list_repo_files",
        lambda self, **kwargs: ["data/documents/test.parquet", "data/questions/test.parquet"],
    )

    def fail_download(**kwargs):
        raise LocalEntryNotFoundError("no local entry after verified repo path")

    def fail_resolve_download(**kwargs):
        raise OSError("resolve fallback failed")

    monkeypatch.setattr("ragflow_bench.benchmarks.eragb_prep.hf_hub_download", fail_download)
    monkeypatch.setattr("ragflow_bench.benchmarks.eragb_prep._download_hf_file_via_resolve_url", fail_resolve_download)

    with pytest.raises(ERAGBDownloadError) as exc_info:
        prepare_eragb_artifacts(output_dir=tmp_path / "eragb")

    message = str(exc_info.value)
    assert "file exists on Hugging Face but could not be downloaded" in message
    assert "Dataset: onyx-dot-app/EnterpriseRAG-Bench" in message
    assert "File: data/documents/test.parquet" in message
    assert str(tmp_path / "eragb" / "raw" / "documents" / "test.parquet") in message


def test_download_hf_file_falls_back_to_resolve_url(tmp_path, monkeypatch):
    def fail_hub_download(**kwargs):
        raise LocalEntryNotFoundError("hub client could not resolve local entry")

    def fake_resolve_download(*, repo_path, target, token):
        assert repo_path == "data/documents/test.parquet"
        assert token == "hf_test"
        target.write_bytes(b"parquet-bytes")
        return target

    monkeypatch.setattr("ragflow_bench.benchmarks.eragb_prep.hf_hub_download", fail_hub_download)
    monkeypatch.setattr("ragflow_bench.benchmarks.eragb_prep._download_hf_file_via_resolve_url", fake_resolve_download)

    downloaded = _download_hf_file(repo_path="data/documents/test.parquet", local_dir=tmp_path, refresh=False, token="hf_test")

    assert downloaded == tmp_path / "test.parquet"
    assert downloaded.read_bytes() == b"parquet-bytes"


def test_download_hf_file_reports_hub_and_fallback_errors(tmp_path, monkeypatch):
    def fail_hub_download(**kwargs):
        raise LocalEntryNotFoundError("hub failed")

    def fail_resolve_download(**kwargs):
        raise OSError("fallback failed")

    monkeypatch.setattr("ragflow_bench.benchmarks.eragb_prep.hf_hub_download", fail_hub_download)
    monkeypatch.setattr("ragflow_bench.benchmarks.eragb_prep._download_hf_file_via_resolve_url", fail_resolve_download)

    with pytest.raises(ERAGBDownloadError) as exc_info:
        _download_hf_file(repo_path="data/documents/test.parquet", local_dir=tmp_path, refresh=False, token=None)

    message = str(exc_info.value)
    assert "Hub client error: LocalEntryNotFoundError: hub failed" in message
    assert "Fallback error: OSError: fallback failed" in message


def test_prepare_eragb_rejects_shard_references_without_merge(tmp_path):
    try:
        prepare_eragb_artifacts(output_dir=tmp_path / "eragb", reference_granularity="shard")
    except ValueError as exc:
        assert "requires merge_documents" in str(exc)
    else:
        raise AssertionError("Expected validation failure")


def test_prepare_eragb_preserves_source_specific_metadata(tmp_path, monkeypatch):
    source_docs = tmp_path / "source_documents.parquet"
    source_questions = tmp_path / "source_questions.parquet"
    pd.DataFrame(
        [
            {"doc_id": "conf1", "source_type": "confluence", "title": "Conf", "content": "c", "page_title": "Page", "space": "ENG", "path": "/a", "heading_hierarchy": ["H1"]},
            {"doc_id": "slack1", "source_type": "slack", "title": "Slack", "content": "s", "channel": "#eng", "thread_timestamp": "1", "message_timestamp": "2", "user": "u"},
            {"doc_id": "fire1", "source_type": "fireflies", "title": "Fire", "content": "f", "meeting_title": "M", "date": "2026-01-01", "speaker": "Ada", "turn_index": 7},
        ]
    ).to_parquet(source_docs)
    pd.DataFrame([{"question_id": "q", "question": "q", "expected_doc_ids": ["conf1"], "gold_answer": "a"}]).to_parquet(source_questions)
    monkeypatch.setattr(
        "ragflow_bench.benchmarks.eragb_prep.HfApi.list_repo_files",
        lambda self, **kwargs: ["data/documents/test.parquet", "data/questions/test.parquet"],
    )
    monkeypatch.setattr(
        "ragflow_bench.benchmarks.eragb_prep.hf_hub_download",
        lambda **kwargs: str(source_docs if "documents" in kwargs["filename"] else source_questions),
    )

    prepare_eragb_artifacts(output_dir=tmp_path / "eragb", merge_documents=True, merge_max_docs=10)

    manifest = json.loads((tmp_path / "eragb" / "documents_manifest.json").read_text(encoding="utf-8"))
    all_docs = [doc for shard in manifest.values() for doc in shard["contained_documents"]]
    by_id = {doc["id"]: doc for doc in all_docs}
    assert by_id["conf1"]["metadata"]["space"] == "ENG"
    assert by_id["slack1"]["metadata"]["channel"] == "#eng"
    assert by_id["fire1"]["metadata"]["meeting_title"] == "M"
    assert by_id["conf1"]["canonical_shard_uri"].startswith("eragb-shard://confluence/")
