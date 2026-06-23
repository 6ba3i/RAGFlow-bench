from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from datasets import load_dataset

from ragflow_bench.benchmarks.base import BenchmarkAdapter, BenchmarkDocument, BenchmarkQuestion
from ragflow_bench.config import AppConfig
from ragflow_bench.benchmarks.frames_prep import (
    extract_gold_answer,
    extract_question_text,
    extract_reasoning_types,
    extract_wikipedia_urls,
    question_id_for_row,
)


class FramesAdapter(BenchmarkAdapter):
    def __init__(self, config: AppConfig):
        self.config = config
        section = config.benchmark.frames
        if section is None:
            raise ValueError("frames config is required")
        self.section = section
        self.mapping = {}
        if section.mapping_path:
            self.mapping = json.loads(Path(section.mapping_path).read_text(encoding="utf-8"))
        self._questions: list[BenchmarkQuestion] | None = None

    def load_questions(self) -> list[BenchmarkQuestion]:
        if self._questions is not None:
            return self._questions
        dataset = load_dataset("google/frames-benchmark", split=self.section.split)
        questions: list[BenchmarkQuestion] = []
        limit = self.config.benchmark.question_limit
        for idx, row in enumerate(dataset):
            raw_row = dict(row)
            qid = question_id_for_row(raw_row, idx)
            mapping = self.mapping.get(qid, {})
            expected = [item.get("wikipedia_url") for item in mapping.get("local_files", []) if item.get("wikipedia_url")]
            if not expected:
                expected = extract_wikipedia_urls(raw_row)
            reasoning = extract_reasoning_types(raw_row)
            questions.append(
                BenchmarkQuestion(
                    id=qid,
                    question=extract_question_text(raw_row),
                    gold_answer=extract_gold_answer(raw_row),
                    expected_sources=expected,
                    reasoning_types=list(reasoning),
                    metadata={"raw": raw_row},
                )
            )
            if limit and len(questions) >= limit:
                break
        self._questions = questions
        return questions

    def iter_documents(self) -> Iterator[BenchmarkDocument]:
        for qid, payload in self.mapping.items():
            for idx, item in enumerate(payload.get("local_files", [])):
                path = Path(self.section.local_corpus_dir or ".") / item["path"]
                yield BenchmarkDocument(
                    id=f"{qid}:{idx}",
                    path=path,
                    source_uri=item.get("wikipedia_url") or path.as_uri(),
                    title=item.get("page_title") or path.stem,
                    source_type="wikipedia",
                    metadata={"question_id": qid, "pageid": item.get("pageid")},
                )
