from ragflow_bench.logging_utils import default_progress_printer, format_progress_event


def test_format_progress_event_redacts_and_quotes_values():
    line = format_progress_event(
        {
            "command": "run",
            "step": "retrieval",
            "status": "error",
            "question_id": "q 1",
            "error": "failed Bearer abcdefghijklmnop",
            "elapsed_seconds": 1.25,
        }
    )

    assert "run" in line
    assert "step=retrieval" in line
    assert 'question_id="q 1"' in line
    assert "Bearer ***REDACTED***" in line
    assert "elapsed=1.2s" in line


def test_default_progress_printer_writes_stderr(capsys):
    default_progress_printer({"command": "score", "step": "row", "status": "ok", "index": 1, "total": 2})

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "score step=row status=ok index=1 total=2" in captured.err
