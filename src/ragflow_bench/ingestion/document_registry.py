from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DocumentRegistry:
    def __init__(self, dataset_id: str | None = None, documents: dict[str, Any] | None = None):
        self.dataset_id = dataset_id
        self.documents = documents or {}

    def register(self, source_uri: str, *, ragflow_document_id: str, title: str, source_type: str, metadata: dict[str, Any] | None = None) -> None:
        self.documents[source_uri] = {
            "ragflow_document_id": ragflow_document_id,
            "title": title,
            "source_type": source_type,
            "metadata": metadata or {},
        }

    def get_document_id(self, source_uri: str) -> str | None:
        payload = self.documents.get(source_uri)
        if not payload:
            return None
        return payload.get("ragflow_document_id")

    def to_dict(self) -> dict[str, Any]:
        return {"dataset_id": self.dataset_id, "documents": self.documents}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DocumentRegistry":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(dataset_id=payload.get("dataset_id"), documents=payload.get("documents") or {})
