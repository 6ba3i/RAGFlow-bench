from ragflow_bench.scoring import (
    classify_failure,
    eragb_shard_uri_from_name,
    exact_match,
    normalized_match,
    retrieval_diagnostics,
    source_recall,
)


def test_exact_and_normalized_match():
    assert exact_match("Answer", "Answer") is True
    assert exact_match("Answer", "answer") is False
    assert normalized_match("Answer!", " answer ") is True


def test_source_recall():
    assert source_recall(["a", "b"], ["b", "c"]) == 0.5
    assert source_recall([], ["a"]) == 0.0


def test_failure_classification():
    assert classify_failure(error=None, ragflow_answer="ok", exact_match=True, source_recall=1.0) == "exact_match_correct"
    assert classify_failure(error="boom", ragflow_answer=None, exact_match=False, source_recall=0.0) == "answer_generation_error"
    assert classify_failure(error=None, ragflow_answer="", exact_match=False, source_recall=0.0) == "answer_generation_error"
    assert classify_failure(error=None, ragflow_answer="wrong", exact_match=False, source_recall=0.0) == "expected_source_absent_from_raw_retrieval"


def test_eragb_shard_uri_mapping_from_filename():
    assert eragb_shard_uri_from_name("github_shard_000104.txt") == "eragb-shard://github/github_shard_000104.txt"
    assert eragb_shard_uri_from_name("/tmp/confluence_shard_000191.txt") == "eragb-shard://confluence/confluence_shard_000191.txt"


def test_retrieval_diagnostics_matches_document_name_rank_and_mrr():
    diagnostics = retrieval_diagnostics(
        ["eragb-shard://github/github_shard_000104.txt"],
        [
            {"document_name": "github_shard_000001.txt", "id": "c1"},
            {"document_name": "github_shard_000104.txt", "id": "c2"},
        ],
        prefix="raw_retrieval",
    )

    assert diagnostics["raw_retrieval_shard_recall"] == 1.0
    assert diagnostics["raw_retrieval_expected_rank"] == 2
    assert diagnostics["raw_retrieval_mrr"] == 0.5


def test_failure_classification_specific_retrieval_context_and_not_found_labels():
    assert classify_failure(error=None, ragflow_answer="wrong", exact_match=False, raw_retrieval_shard_recall=0.0, final_context_shard_recall=0.0) == "expected_source_absent_from_raw_retrieval"
    assert classify_failure(error=None, ragflow_answer="wrong", exact_match=False, raw_retrieval_shard_recall=1.0, final_context_shard_recall=0.0) == "expected_source_present_raw_absent_final_context"
    assert classify_failure(error=None, ragflow_answer="wrong", exact_match=False, raw_retrieval_shard_recall=1.0, final_context_shard_recall=1.0, judge_verdict="incorrect") == "expected_source_present_final_context_answer_wrong"
    assert classify_failure(error=None, ragflow_answer="part", exact_match=False, raw_retrieval_shard_recall=1.0, final_context_shard_recall=1.0, judge_verdict="partial") == "expected_source_present_final_context_answer_partial"
    assert classify_failure(error=None, ragflow_answer="The provided context does not contain this information.", exact_match=False, raw_retrieval_shard_recall=1.0, final_context_shard_recall=1.0) == "not_found_false_negative"


def test_retrieval_diagnostics_preserves_non_eragb_source_candidates():
    diagnostics = retrieval_diagnostics(
        ["doc-1"],
        [{"doc_id": "doc-1", "metadata": {"source_uri": "file:///tmp/doc-1.txt"}}],
        prefix="raw_retrieval",
    )

    assert "doc-1" in diagnostics["raw_retrieval_retrieved_source_uris"]
    assert "file:///tmp/doc-1.txt" in diagnostics["raw_retrieval_retrieved_source_uris"]
    assert diagnostics["raw_retrieval_shard_recall"] == 1.0
    assert diagnostics["raw_retrieval_retrieved_shard_uris"] == []
