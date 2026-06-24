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
            expected_sources = _list_value(_first_present_value(row_dict, ("expected_sources", "expected_doc_ids", "expected_documents", "citations")))
            reasoning = _list_value(_first_present_value(row_dict, ("reasoning_types", "reasoning_type", "question_type", "source_types")))
            results.append(BenchmarkQuestion(
                id=str(_first_present_value(row_dict, ("id", "question_id")) or idx),
                question=str(row.get("question") or ""),
                gold_answer=_first_present_value(row_dict, ("gold_answer", "answer")),
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
            source_uri = meta.get("source_uri")
            if source_uri is None:
                source_uri = path.resolve().as_uri()
            yield BenchmarkDocument(
                id=str(meta.get("id", idx)),
                path=path,
                source_uri=str(source_uri),
                title=str(meta.get("title", path.stem)),
                source_type=str(meta.get("source_type", path.suffix.lstrip(".") or "file")),
                metadata=meta,
            )


def _first_present_value(row: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in row and not _is_missing_value(row[key]):
            return row[key]
    return None


def _is_missing_value(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


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
