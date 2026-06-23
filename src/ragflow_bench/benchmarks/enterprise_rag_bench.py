from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pandas as pd

from ragflow_bench.benchmarks.base import BenchmarkAdapter, BenchmarkDocument, BenchmarkQuestion
from ragflow_bench.config import AppConfig


class EnterpriseRAGBenchAdapter(BenchmarkAdapter):
    def __init__(self, config: AppConfig):
        self.config = config
        section = config.benchmark.enterprise_rag_bench
        if section is None:
            raise ValueError("enterprise_rag_bench config is required")
        self.section = section

    def _load_questions_frame(self) -> pd.DataFrame:
        path = Path(self.section.questions_path or "")
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            return pd.read_json(path, lines=True)
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return pd.DataFrame(data)
        if suffix == ".csv":
            return pd.read_csv(path)
        return pd.read_parquet(path)

    def load_questions(self) -> list[BenchmarkQuestion]:
        df = self._load_questions_frame()
        limit = self.config.benchmark.question_limit
        results: list[BenchmarkQuestion] = []
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            expected_sources = _list_value(
                row.get("expected_sources")
                or row.get("expected_doc_ids")
                or row.get("expected_documents")
                or row.get("citations")
            )
            reasoning = _list_value(row.get("reasoning_types") or row.get("reasoning_type") or row.get("question_type") or row.get("source_types"))
            results.append(BenchmarkQuestion(
                id=str(row.get("id") or row.get("question_id") or idx),
                question=str(row.get("question") or ""),
                gold_answer=row.get("gold_answer") or row.get("answer"),
                expected_sources=expected_sources,
                reasoning_types=reasoning,
                metadata=row_dict,
            ))
            if limit and len(results) >= limit:
                break
        return results

    def iter_documents(self) -> Iterator[BenchmarkDocument]:
        corpus_dir = Path(self.section.corpus_dir or "")
        manifest = None
        if self.section.documents_manifest:
            manifest_path = Path(self.section.documents_manifest)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.suffix == ".json" else None
        for idx, path in enumerate(sorted(p for p in corpus_dir.rglob("*") if p.is_file())):
            meta = {}
            if isinstance(manifest, dict):
                rel_path = path.relative_to(corpus_dir).as_posix()
                meta = manifest.get(rel_path) or manifest.get(path.name, {})
            yield BenchmarkDocument(
                id=str(meta.get("id", idx)),
                path=path,
                source_uri=str(meta.get("source_uri", path.as_uri())),
                title=str(meta.get("title", path.stem)),
                source_type=str(meta.get("source_type", path.suffix.lstrip(".") or "file")),
                metadata=meta,
            )


def _list_value(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    if hasattr(value, "tolist"):
        return _list_value(value.tolist())
    text = str(value)
    return [text] if text else []
