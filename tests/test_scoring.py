from ragflow_bench.scoring import classify_failure, exact_match, normalized_match, source_recall


def test_exact_and_normalized_match():
    assert exact_match("Answer", "Answer") is True
    assert exact_match("Answer", "answer") is False
    assert normalized_match("Answer!", " answer ") is True


def test_source_recall():
    assert source_recall(["a", "b"], ["b", "c"]) == 0.5
    assert source_recall([], ["a"]) == 0.0


def test_failure_classification():
    assert classify_failure(error=None, ragflow_answer="ok", exact_match=True, source_recall=1.0) == "correct"
    assert classify_failure(error="boom", ragflow_answer=None, exact_match=False, source_recall=0.0) == "error"
    assert classify_failure(error=None, ragflow_answer="", exact_match=False, source_recall=0.0) == "empty_response"
    assert classify_failure(error=None, ragflow_answer="wrong", exact_match=False, source_recall=0.0) == "retrieval_failure"
