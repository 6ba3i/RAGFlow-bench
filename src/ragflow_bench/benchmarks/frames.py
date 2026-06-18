from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from datasets import load_dataset

from ragflow_bench.benchmarks.base import BenchmarkAdapter, BenchmarkDocument, BenchmarkQuestion
from ragflow_bench.config import AppConfig


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
            qid = str(row.get("id", idx))
            mapping = self.mapping.get(qid, {})
            expected = [item.get("wikipedia_url") for item in mapping.get("local_files", []) if item.get("wikipedia_url")]
            reasoning = row.get("reasoning_types") or row.get("reasoning_type") or []
            if isinstance(reasoning, str):
                reasoning = [reasoning]
            questions.append(
                BenchmarkQuestion(
                    id=qid,
                    question=row.get("question") or row.get("prompt") or "",
                    gold_answer=row.get("answer") or row.get("gold_answer"),
                    expected_sources=expected,
                    reasoning_types=list(reasoning),
                    metadata={"raw": dict(row)},
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
