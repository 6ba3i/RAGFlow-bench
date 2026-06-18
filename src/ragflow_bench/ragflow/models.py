from __future__ import annotations

from pydantic import BaseModel, Field


class APIEnvelope(BaseModel):
    code: int = 0
    message: str = ""
    data: object | None = None


class HealthStatus(BaseModel):
    db: str | None = None
    redis: str | None = None
    doc_engine: str | None = None
    storage: str | None = None
    status: str | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str | None = None
    document_id: str | None = Field(default=None, alias="doc_id")
    dataset_id: str | None = None
    content: str | None = None
    score: float | None = None
    source_uri: str | None = None
    metadata: dict = Field(default_factory=dict)
