from __future__ import annotations

import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError, OfflineModeIsEnabled

from ragflow_bench.logging_utils import ProgressCallback, emit_progress
from ragflow_bench.reports.writers import write_json, write_jsonl

ERAGB_DATASET_ID = "onyx-dot-app/EnterpriseRAG-Bench"
DOCUMENTS_REPO_PATH = "data/documents/test.parquet"
QUESTIONS_REPO_PATH = "data/questions/test.parquet"
ERAGB_DOC_BOUNDARY = "<<<ERAGB_DOC_BOUNDARY>>>"
ERAGB_PARSER_DELIMITER = f"`{ERAGB_DOC_BOUNDARY}`"
ReferenceGranularity = Literal["document", "shard", "none"]
REQUIRED_REPO_PATHS = (DOCUMENTS_REPO_PATH, QUESTIONS_REPO_PATH)


class ERAGBDownloadError(RuntimeError):
    """Raised when EnterpriseRAG-Bench artifacts cannot be verified or downloaded."""


def prepare_eragb_artifacts(
    *,
    split: str = "test",
    output_dir: str | Path = "data/eragb",
    document_limit: int | None = None,
    question_limit: int | None = None,
    refresh: bool = False,
    hf_token_env_var: str = "HF_TOKEN",
    merge_documents: bool = False,
    merge_target_bytes: int = 262144,
    merge_max_docs: int = 100,
    filter_questions_with_missing_docs: bool = False,
    reference_granularity: ReferenceGranularity | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    emit_progress(progress_callback, {"command": "prepare-eragb", "step": "start", "status": "start", "output_dir": str(output_dir)})
    if split != "test":
        raise ValueError("EnterpriseRAG-Bench currently exposes only the 'test' split")
    reference_granularity = reference_granularity or ("shard" if merge_documents else "document")
    if reference_granularity not in {"document", "shard", "none"}:
        raise ValueError("reference_granularity must be one of: document, shard, none")
    if reference_granularity == "shard" and not merge_documents:
        raise ValueError("reference_granularity='shard' requires merge_documents=True")
    merge_target_bytes = max(1, int(merge_target_bytes))
    merge_max_docs = max(1, int(merge_max_docs))

    target_dir = Path(output_dir)
    raw_dir = target_dir / "raw"
    documents_raw_dir = raw_dir / "documents"
    questions_raw_dir = raw_dir / "questions"
    corpus_dir = target_dir / "corpus"
    target_dir.mkdir(parents=True, exist_ok=True)
    documents_raw_dir.mkdir(parents=True, exist_ok=True)
    questions_raw_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    token = os.getenv(hf_token_env_var) if hf_token_env_var else None
    if refresh or not (documents_raw_dir / Path(DOCUMENTS_REPO_PATH).name).exists() or not (questions_raw_dir / Path(QUESTIONS_REPO_PATH).name).exists():
        emit_progress(progress_callback, {"command": "prepare-eragb", "step": "verify_hf_repo", "status": "start", "dataset_id": ERAGB_DATASET_ID})
        _verify_hf_repo_paths(token=token)
        emit_progress(progress_callback, {"command": "prepare-eragb", "step": "verify_hf_repo", "status": "ok", "dataset_id": ERAGB_DATASET_ID})

    emit_progress(progress_callback, {"command": "prepare-eragb", "step": "download_documents", "status": "start", "path": DOCUMENTS_REPO_PATH})
    documents_parquet = _download_hf_file(
        repo_path=DOCUMENTS_REPO_PATH,
        local_dir=documents_raw_dir,
        refresh=refresh,
        token=token,
    )
    emit_progress(progress_callback, {"command": "prepare-eragb", "step": "download_documents", "status": "ok", "path": str(documents_parquet)})
    emit_progress(progress_callback, {"command": "prepare-eragb", "step": "download_questions", "status": "start", "path": QUESTIONS_REPO_PATH})
    questions_parquet = _download_hf_file(
        repo_path=QUESTIONS_REPO_PATH,
        local_dir=questions_raw_dir,
        refresh=refresh,
        token=token,
    )
    emit_progress(progress_callback, {"command": "prepare-eragb", "step": "download_questions", "status": "ok", "path": str(questions_parquet)})

    emit_progress(progress_callback, {"command": "prepare-eragb", "step": "read_parquet", "status": "start"})
    documents_df = pd.read_parquet(documents_parquet)
    questions_df = pd.read_parquet(questions_parquet)
    if document_limit is not None:
        documents_df = documents_df.head(max(0, document_limit))
    if question_limit is not None:
        questions_df = questions_df.head(max(0, question_limit))
    emit_progress(progress_callback, {"command": "prepare-eragb", "step": "read_parquet", "status": "ok", "count": int(len(documents_df)), "total": int(len(questions_df))})

    if merge_documents:
        emit_progress(progress_callback, {"command": "prepare-eragb", "step": "write_corpus", "status": "start", "count": int(len(documents_df))})
        corpus = _write_merged_document_corpus(
            documents_df,
            corpus_dir,
            merge_target_bytes=merge_target_bytes,
            merge_max_docs=merge_max_docs,
        )
    else:
        emit_progress(progress_callback, {"command": "prepare-eragb", "step": "write_corpus", "status": "start", "count": int(len(documents_df))})
        corpus = _write_document_corpus(documents_df, corpus_dir)
    emit_progress(progress_callback, {"command": "prepare-eragb", "step": "write_corpus", "status": "ok", "count": len(corpus["manifest"])})

    emit_progress(progress_callback, {"command": "prepare-eragb", "step": "normalize_questions", "status": "start", "count": int(len(questions_df))})
    normalized_questions, dropped_missing = _normalize_questions(
        questions_df,
        prepared_doc_ids=corpus["prepared_doc_ids"],
        doc_id_to_shard=corpus["doc_id_to_shard"],
        reference_granularity=reference_granularity,
        filter_questions_with_missing_docs=filter_questions_with_missing_docs,
    )
    emit_progress(progress_callback, {"command": "prepare-eragb", "step": "normalize_questions", "status": "ok", "count": len(normalized_questions), "failure_count": dropped_missing})

    manifest_path = target_dir / "documents_manifest.json"
    questions_path = target_dir / "questions.jsonl"
    report_path = target_dir / "prepare_report.json"
    doc_id_to_shard_path = target_dir / "doc_id_to_shard.json"
    shard_manifest_path = target_dir / "shard_manifest.json"
    parser_config_path = target_dir / "parser_config.merged.yaml"

    write_json(manifest_path, corpus["manifest"])
    emit_progress(progress_callback, {"command": "prepare-eragb", "step": "manifest_write", "status": "ok", "path": str(manifest_path), "count": len(corpus["manifest"])})
    write_jsonl(questions_path, normalized_questions)
    emit_progress(progress_callback, {"command": "prepare-eragb", "step": "questions_write", "status": "ok", "path": str(questions_path), "count": len(normalized_questions)})
    if merge_documents:
        write_json(doc_id_to_shard_path, corpus["doc_id_to_shard"])
        write_json(shard_manifest_path, corpus["shard_manifest"])
        parser_config_path.write_text(
            "chunk_token_num: 512\n"
            f"delimiter: \"{ERAGB_PARSER_DELIMITER}\"\n"
            "raptor:\n"
            "  use_raptor: false\n"
            "graphrag:\n"
            "  use_graphrag: false\n",
            encoding="utf-8",
        )

    report = {
        "dataset_id": ERAGB_DATASET_ID,
        "split": split,
        "documents_parquet_path": str(documents_parquet),
        "questions_parquet_path": str(questions_parquet),
        "document_count": int(len(documents_df)),
        "shard_count": len(corpus["manifest"]),
        "question_count_input": int(len(questions_df)),
        "question_count_written": len(normalized_questions),
        "question_count": len(normalized_questions),
        "question_count_dropped_missing_docs": dropped_missing,
        "reference_granularity": reference_granularity,
        "merge_documents": merge_documents,
        "merge_target_bytes": merge_target_bytes if merge_documents else None,
        "merge_max_docs": merge_max_docs if merge_documents else None,
        "parser_delimiter": ERAGB_PARSER_DELIMITER if merge_documents else None,
        "source_type_counts": corpus["source_type_counts"],
        "corpus_dir": str(corpus_dir),
        "questions_path": str(questions_path),
        "documents_manifest_path": str(manifest_path),
    }
    if merge_documents:
        report["doc_id_to_shard_path"] = str(doc_id_to_shard_path)
        report["shard_manifest_path"] = str(shard_manifest_path)
        report["parser_config_path"] = str(parser_config_path)
    write_json(report_path, report)
    emit_progress(progress_callback, {"command": "prepare-eragb", "step": "report_write", "status": "ok", "path": str(report_path)})
    emit_progress(progress_callback, {"command": "prepare-eragb", "step": "complete", "status": "ok", "count": len(corpus["manifest"]), "total": len(normalized_questions), "elapsed_seconds": time.monotonic() - started})
    return report


def _verify_hf_repo_paths(*, token: str | None) -> None:
    try:
        repo_files = set(HfApi().list_repo_files(repo_id=ERAGB_DATASET_ID, repo_type="dataset", token=token))
    except Exception as exc:
        raise ERAGBDownloadError(
            "Could not verify EnterpriseRAG-Bench files on Hugging Face. "
            f"Dataset: {ERAGB_DATASET_ID}. Expected files: {', '.join(REQUIRED_REPO_PATHS)}. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    missing = [path for path in REQUIRED_REPO_PATHS if path not in repo_files]
    if missing:
        raise ERAGBDownloadError(
            "EnterpriseRAG-Bench Hugging Face repo was found, but required parquet files are missing. "
            f"Dataset: {ERAGB_DATASET_ID}. Missing: {', '.join(missing)}. "
            f"Expected files: {', '.join(REQUIRED_REPO_PATHS)}."
        )


def _download_hf_file(*, repo_path: str, local_dir: Path, refresh: bool, token: str | None) -> Path:
    target = local_dir / Path(repo_path).name
    if target.exists() and not refresh:
        return target
    try:
        downloaded = hf_hub_download(
            repo_id=ERAGB_DATASET_ID,
            repo_type="dataset",
            filename=repo_path,
            local_dir=local_dir,
            force_download=refresh,
            token=token,
        )
    except (LocalEntryNotFoundError, OfflineModeIsEnabled, HfHubHTTPError, OSError) as exc:
        try:
            return _download_hf_file_via_resolve_url(repo_path=repo_path, target=target, token=token)
        except Exception as fallback_exc:
            raise ERAGBDownloadError(
                "EnterpriseRAG-Bench file exists on Hugging Face but could not be downloaded. "
                f"Dataset: {ERAGB_DATASET_ID}. File: {repo_path}. Local target: {target}. "
                "The Hugging Face hub client failed, and the direct resolve-url fallback also failed. "
                "Check access to huggingface.co and the signed CDN URL, set HF_TOKEN if needed, or unset HF_HUB_OFFLINE. "
                f"Hub client error: {type(exc).__name__}: {exc}. "
                f"Fallback error: {type(fallback_exc).__name__}: {fallback_exc}"
            ) from fallback_exc
    downloaded_path = Path(downloaded)
    if downloaded_path != target and downloaded_path.exists():
        target.write_bytes(downloaded_path.read_bytes())
    return target


def _download_hf_file_via_resolve_url(*, repo_path: str, target: Path, token: str | None) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_name(f"{target.name}.tmp")
    url = f"https://huggingface.co/datasets/{ERAGB_DATASET_ID}/resolve/main/{repo_path}"
    headers = {"User-Agent": "ragflow-bench"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response, tmp_target.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        tmp_target.replace(target)
        return target
    except (urllib.error.URLError, TimeoutError, OSError):
        tmp_target.unlink(missing_ok=True)
        raise


def _write_document_corpus(df: pd.DataFrame, corpus_dir: Path) -> dict[str, Any]:
    manifest: dict[str, dict[str, Any]] = {}
    source_type_counts: dict[str, int] = {}
    prepared_doc_ids: set[str] = set()
    for idx, row in enumerate(_document_rows(df)):
        doc_id = row["doc_id"] or str(idx)
        source_type = row["source_type"]
        title = row["title"] or doc_id
        content = row["content"]
        source_dir = corpus_dir / _safe_path_component(source_type)
        source_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{_safe_path_component(doc_id)}.txt"
        path = source_dir / filename
        path.write_text(_document_text(doc_id=doc_id, source_type=source_type, title=title, content=content), encoding="utf-8")
        relative_path = path.relative_to(corpus_dir).as_posix()
        manifest[relative_path] = {
            "id": doc_id,
            "source_uri": doc_id,
            "title": title,
            "source_type": source_type,
            "path": relative_path,
        }
        prepared_doc_ids.add(doc_id)
        source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
    return {
        "manifest": manifest,
        "doc_id_to_shard": {},
        "shard_manifest": {},
        "prepared_doc_ids": prepared_doc_ids,
        "source_type_counts": source_type_counts,
    }


def _write_merged_document_corpus(df: pd.DataFrame, corpus_dir: Path, *, merge_target_bytes: int, merge_max_docs: int) -> dict[str, Any]:
    manifest: dict[str, dict[str, Any]] = {}
    shard_manifest: dict[str, dict[str, Any]] = {}
    doc_id_to_shard: dict[str, str] = {}
    source_type_counts: dict[str, int] = {}
    prepared_doc_ids: set[str] = set()
    rows = sorted(_document_rows(df), key=lambda item: (item["source_type"], item["doc_id"]))
    current_source_type: str | None = None
    current_blocks: list[str] = []
    current_doc_ids: list[str] = []
    current_bytes = 0
    shard_index_by_source: dict[str, int] = {}

    def flush() -> None:
        nonlocal current_blocks, current_doc_ids, current_bytes, current_source_type
        if not current_blocks or current_source_type is None:
            return
        shard_index = shard_index_by_source.get(current_source_type, 0) + 1
        shard_index_by_source[current_source_type] = shard_index
        source_slug = _safe_path_component(current_source_type)
        source_dir = corpus_dir / source_slug
        source_dir.mkdir(parents=True, exist_ok=True)
        shard_id = f"{source_slug}_shard_{shard_index:06d}"
        relative_path = f"{source_slug}/{shard_id}.txt"
        shard_uri = f"eragb-shard://{relative_path}"
        path = corpus_dir / relative_path
        path.write_text("\n".join(current_blocks).rstrip() + "\n", encoding="utf-8")
        manifest[relative_path] = {
            "id": shard_id,
            "source_uri": shard_uri,
            "title": f"ERAGB {current_source_type} shard {shard_index:06d}",
            "source_type": current_source_type,
            "path": relative_path,
            "contained_doc_count": len(current_doc_ids),
        }
        shard_manifest[relative_path] = {
            **manifest[relative_path],
            "contained_doc_ids": list(current_doc_ids),
        }
        for doc_id in current_doc_ids:
            doc_id_to_shard[doc_id] = shard_uri
        current_blocks = []
        current_doc_ids = []
        current_bytes = 0

    for row in rows:
        doc_id = row["doc_id"]
        source_type = row["source_type"]
        block = _merged_document_block(**row)
        block_bytes = len(block.encode("utf-8"))
        if current_source_type != source_type:
            flush()
            current_source_type = source_type
        if current_blocks and (len(current_doc_ids) >= merge_max_docs or current_bytes + block_bytes > merge_target_bytes):
            flush()
            current_source_type = source_type
        current_blocks.append(block)
        current_doc_ids.append(doc_id)
        current_bytes += block_bytes
        prepared_doc_ids.add(doc_id)
        source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
    flush()
    return {
        "manifest": manifest,
        "doc_id_to_shard": doc_id_to_shard,
        "shard_manifest": shard_manifest,
        "prepared_doc_ids": prepared_doc_ids,
        "source_type_counts": source_type_counts,
    }


def _normalize_questions(
    questions_df: pd.DataFrame,
    *,
    prepared_doc_ids: set[str],
    doc_id_to_shard: dict[str, str],
    reference_granularity: ReferenceGranularity,
    filter_questions_with_missing_docs: bool,
) -> tuple[list[dict[str, Any]], int]:
    normalized: list[dict[str, Any]] = []
    dropped_missing = 0
    for row in questions_df.to_dict(orient="records"):
        expected_doc_ids = _list_value(row.get("expected_sources") or row.get("expected_doc_ids") or row.get("expected_documents") or row.get("citations"))
        missing = [doc_id for doc_id in expected_doc_ids if doc_id not in prepared_doc_ids]
        if filter_questions_with_missing_docs and missing:
            dropped_missing += 1
            continue
        normalized.append(
            _normalize_question_row(
                row,
                expected_doc_ids=expected_doc_ids,
                expected_sources=_expected_sources_for_granularity(expected_doc_ids, doc_id_to_shard, reference_granularity),
                missing_expected_doc_ids=missing,
                reference_granularity=reference_granularity,
            )
        )
    return normalized, dropped_missing


def _expected_sources_for_granularity(expected_doc_ids: list[str], doc_id_to_shard: dict[str, str], reference_granularity: ReferenceGranularity) -> list[str]:
    if reference_granularity == "none":
        return []
    if reference_granularity == "document":
        return expected_doc_ids
    return _dedupe_preserve_order(doc_id_to_shard[doc_id] for doc_id in expected_doc_ids if doc_id in doc_id_to_shard)


def _normalize_question_row(
    row: dict[str, Any],
    *,
    expected_doc_ids: list[str] | None = None,
    expected_sources: list[str] | None = None,
    missing_expected_doc_ids: list[str] | None = None,
    reference_granularity: str = "document",
) -> dict[str, Any]:
    question_id = str(row.get("id") or row.get("question_id") or "")
    raw_expected_doc_ids = expected_doc_ids if expected_doc_ids is not None else _list_value(row.get("expected_sources") or row.get("expected_doc_ids") or row.get("expected_documents") or row.get("citations"))
    source_types = _list_value(row.get("source_types"))
    reasoning = _list_value(row.get("reasoning_types") or row.get("reasoning_type") or row.get("question_type"))
    payload = {key: _jsonable(value) for key, value in row.items()}
    payload.update(
        {
            "id": question_id,
            "question": str(row.get("question") or ""),
            "gold_answer": row.get("gold_answer") or row.get("answer"),
            "expected_doc_ids": raw_expected_doc_ids,
            "expected_sources": expected_sources if expected_sources is not None else raw_expected_doc_ids,
            "missing_expected_doc_ids": missing_expected_doc_ids or [],
            "reference_granularity": reference_granularity,
            "reasoning_types": reasoning,
            "source_types": source_types,
        }
    )
    return payload


def _document_rows(df: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for idx, row in enumerate(df.to_dict(orient="records")):
        doc_id = str(row.get("doc_id") or row.get("id") or idx)
        rows.append(
            {
                "doc_id": doc_id,
                "source_type": str(row.get("source_type") or "document"),
                "title": str(row.get("title") or doc_id),
                "content": str(row.get("content") or ""),
            }
        )
    return rows


def _list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return _list_value(value.tolist())
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    text = str(value)
    if not text:
        return []
    return [text]


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _document_text(*, doc_id: str, source_type: str, title: str, content: str) -> str:
    return f"Title: {title}\nDocument ID: {doc_id}\nSource Type: {source_type}\n\n{content.strip()}\n"


def _merged_document_block(*, doc_id: str, source_type: str, title: str, content: str) -> str:
    return f"{ERAGB_DOC_BOUNDARY}\nDocument ID: {doc_id}\nSource Type: {source_type}\nTitle: {title}\nContent:\n{content.strip()}\n"


def _safe_path_component(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return slug or "item"


def _dedupe_preserve_order(values) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
