from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class BenchmarkQuestion:
    id: str
    question: str
    gold_answer: str | None
    expected_sources: list[str]
    reasoning_types: list[str]
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkDocument:
    id: str
    path: Path
    source_uri: str
    title: str
    source_type: str
    metadata: dict = field(default_factory=dict)


class BenchmarkAdapter:
    def load_questions(self) -> list[BenchmarkQuestion]:
        raise NotImplementedError

    def iter_documents(self) -> Iterator[BenchmarkDocument]:
        raise NotImplementedError

    def expected_sources_for_question(self, question_id: str) -> list[str]:
        return next((q.expected_sources for q in self.load_questions() if q.id == question_id), [])
