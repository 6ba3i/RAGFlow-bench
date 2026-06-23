from pathlib import Path

from ragflow_bench.benchmarks.frames_prep import (
    build_page_filename,
    extract_reasoning_types,
    extract_wikipedia_urls,
    page_title_from_url,
    prepare_frames_artifacts,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append((url, params, timeout, headers))
        title = params["titles"]
        return _FakeResponse(
            {
                "query": {
                    "pages": {
                        "123": {
                            "pageid": 123,
                            "title": title,
                            "extract": f"{title} article body",
                        }
                    }
                }
            }
        )


def test_extract_wikipedia_urls_from_frames_row():
    row = {
        "wikipedia_link_1": "https://en.wikipedia.org/wiki/A",
        "wikipedia_link_2": "https://en.wikipedia.org/wiki/B",
        "wikipedia_link_3": "https://en.wikipedia.org/wiki/A",
    }

    assert extract_wikipedia_urls(row) == [
        "https://en.wikipedia.org/wiki/A",
        "https://en.wikipedia.org/wiki/B",
    ]


def test_extract_reasoning_types_splits_pipe_values():
    row = {"reasoning_types": "Numerical reasoning | Tabular reasoning | Multiple constraints"}

    assert extract_reasoning_types(row) == [
        "Numerical reasoning",
        "Tabular reasoning",
        "Multiple constraints",
    ]


def test_page_title_and_filename_normalization():
    assert page_title_from_url("https://en.wikipedia.org/wiki/Charlotte_Bront%C3%AB") == "Charlotte Brontë"
    assert build_page_filename("Charlotte Brontë", "123") == "Charlotte_Bront_123.txt"


def test_prepare_frames_artifacts_writes_mapping_and_dedupes_downloads(tmp_path, monkeypatch):
    rows = [
        {
            "Unnamed: 0": 0,
            "Prompt": "Q1",
            "Answer": "A1",
            "reasoning_types": "Multiple constraints",
            "wikipedia_link_1": "https://en.wikipedia.org/wiki/A",
            "wikipedia_link_2": "https://en.wikipedia.org/wiki/B",
        },
        {
            "Unnamed: 0": 1,
            "Prompt": "Q2",
            "Answer": "A2",
            "reasoning_types": "Tabular reasoning",
            "wikipedia_link_1": "https://en.wikipedia.org/wiki/B",
        },
    ]
    monkeypatch.setattr("ragflow_bench.benchmarks.frames_prep.load_dataset", lambda *args, **kwargs: rows)
    session = _FakeSession()

    report = prepare_frames_artifacts(output_dir=tmp_path, session=session)

    mapping_path = Path(report["mapping_path"])
    corpus_dir = Path(report["corpus_dir"])
    assert mapping_path.exists()
    assert corpus_dir.exists()
    assert report["question_count"] == 2
    assert report["downloaded_page_count"] == 2
    assert report["failure_count"] == 0
    assert len(session.calls) == 2
