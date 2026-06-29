from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

ERAGB_SHARD_PREFIX = "eragb-shard://"
_CHUNK_SOURCE_KEYS = (
    "source_uri",
    "canonical_source_uri",
    "eragb_shard_uri",
    "document_keyword",
    "document_name",
    "docnm_kwd",
    "doc_id",
    "document_id",
    "chunk_id",
    "id",
)
_METADATA_KEYS = ("metadata", "meta_fields", "document_metadata")


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _basename(value: str) -> str:
    text = value.replace(ERAGB_SHARD_PREFIX, "")
    return PurePosixPath(text).name


def eragb_shard_uri_from_name(value: str) -> str | None:
    """Map visible ERAGB shard file names to canonical shard URIs."""
    text = value.strip()
    if not text:
        return None
    if text.startswith(ERAGB_SHARD_PREFIX) and text.endswith(".txt"):
        return text
    name = _basename(text)
    if not name.endswith(".txt") or "_shard_" not in name:
        return None
    source_type = name.split("_shard_", 1)[0]
    if not source_type:
        return None
    return f"{ERAGB_SHARD_PREFIX}{source_type}/{name}"


def _iter_source_containers(chunk: dict[str, Any]):
    yield chunk
    for key in _METADATA_KEYS:
        value = chunk.get(key)
        if isinstance(value, dict):
            yield value


def canonical_source_candidates(chunk: dict[str, Any]) -> list[str]:
    """Return canonical and raw source candidates from common RAGFlow chunk fields."""
    raw_values: list[str] = []
    for container in _iter_source_containers(chunk):
        for key in _CHUNK_SOURCE_KEYS:
            text = _as_text(container.get(key))
            if text:
                raw_values.append(text)
    candidates: list[str] = []
    for value in raw_values:
        candidates.append(value)
        shard_uri = eragb_shard_uri_from_name(value)
        if shard_uri:
            candidates.append(shard_uri)
        base = _basename(value)
        if base and base != value:
            candidates.append(base)
    return _dedupe(candidates)


def source_recall(expected_sources: list[str], retrieved_sources: list[str]) -> float:
    if not expected_sources:
        return 0.0
    expected = set(expected_sources)
    retrieved = set(retrieved_sources)
    return len(expected & retrieved) / len(expected)


def shard_recall(expected_sources: list[str], chunks: list[dict[str, Any]]) -> tuple[float, int | None, float]:
    expected = _canonical_expected_sources(expected_sources)
    if not expected:
        return 0.0, None, 0.0
    expected_set = set(expected)
    hits: set[str] = set()
    first_rank: int | None = None
    for rank, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            continue
        matched = expected_set & set(canonical_source_candidates(chunk))
        if matched:
            hits.update(matched)
            if first_rank is None:
                first_rank = rank
    recall = len(hits) / len(expected_set)
    return recall, first_rank, (1.0 / first_rank if first_rank else 0.0)


def retrieval_diagnostics(expected_sources: list[str], chunks: list[dict[str, Any]], *, prefix: str) -> dict[str, Any]:
    normalized_chunks = [chunk for chunk in chunks if isinstance(chunk, dict)]
    candidate_groups = [canonical_source_candidates(chunk) for chunk in normalized_chunks]
    flat = [candidate for group in candidate_groups for candidate in group]
    recall, rank, mrr = shard_recall(expected_sources, normalized_chunks)
    return {
        f"{prefix}_source_candidates": candidate_groups,
        f"{prefix}_retrieved_source_uris": _dedupe(flat),
        f"{prefix}_retrieved_shard_uris": _dedupe(item for item in flat if str(item).startswith(ERAGB_SHARD_PREFIX)),
        f"{prefix}_shard_recall": recall,
        f"{prefix}_expected_rank": rank,
        f"{prefix}_mrr": mrr,
    }


def _canonical_expected_sources(expected_sources: list[str]) -> list[str]:
    canonical: list[str] = []
    for source in expected_sources or []:
        text = _as_text(source)
        if not text:
            continue
        shard_uri = eragb_shard_uri_from_name(text)
        canonical.append(shard_uri or text)
    return _dedupe(canonical)


def _dedupe(values) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered
