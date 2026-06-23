from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import requests
from datasets import load_dataset

from ragflow_bench.logging_utils import ProgressCallback, emit_progress
from ragflow_bench.reports.writers import write_json

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_USER_AGENT = "ragflow-bench/0.1.0 (https://github.com/openai/ragflow-bench; benchmarking tool)"
WIKIPEDIA_SHORT_HOST = "w.wiki"
WIKIPEDIA_URL_RE = re.compile(r"https?://[^\s\]\"]+")
WIKIPEDIA_URL_SEPARATOR_RE = re.compile(r",\s+(?=https?://)")


@dataclass
class FramesPageArtifact:
    wikipedia_url: str
    page_title: str
    filename: str
    pageid: str | None
    path: Path


def question_id_for_row(row: dict[str, Any], fallback_idx: int) -> str:
    value = row.get("id")
    if value is None:
        value = row.get("Unnamed: 0", fallback_idx)
    return str(value)


def extract_question_text(row: dict[str, Any]) -> str:
    return row.get("question") or row.get("Prompt") or row.get("prompt") or ""


def extract_gold_answer(row: dict[str, Any]) -> str | None:
    return row.get("answer") or row.get("Answer") or row.get("gold_answer")


def extract_reasoning_types(row: dict[str, Any]) -> list[str]:
    raw = row.get("reasoning_types") or row.get("reasoning_type") or []
    if isinstance(raw, str):
        return [item.strip() for item in raw.split("|") if item.strip()]
    return [str(item).strip() for item in raw if str(item).strip()]


def extract_wikipedia_urls(row: dict[str, Any]) -> list[str]:
    direct_fields: list[str] = []
    for key, value in row.items():
        if key.startswith("wikipedia_link_") and value:
            direct_fields.extend(split_wikipedia_urls(str(value)))
    if direct_fields:
        wiki_links = row.get("wiki_links")
        if wiki_links:
            direct_fields.extend(_extract_wiki_links_field(wiki_links))
        return _dedupe_preserve_order(direct_fields)

    wiki_links = row.get("wiki_links")
    return _dedupe_preserve_order(_extract_wiki_links_field(wiki_links))


def _extract_wiki_links_field(wiki_links: Any) -> list[str]:
    if isinstance(wiki_links, list):
        return [str(item) for item in wiki_links if item]
    if isinstance(wiki_links, str) and wiki_links.strip():
        try:
            parsed = json.loads(wiki_links.replace("'", '"'))
            if isinstance(parsed, list):
                return [url for item in parsed if item for url in split_wikipedia_urls(str(item))]
        except json.JSONDecodeError:
            return [
                url
                for piece in wiki_links.strip("[]").split(",")
                for url in split_wikipedia_urls(piece)
            ]
    return []


def split_wikipedia_urls(value: str) -> list[str]:
    """Return normalized URL tokens from a FRAMES link cell.

    The upstream FRAMES data sometimes stores multiple Wikipedia URLs in the
    final `wikipedia_link_11+` cell, separated by commas, and sometimes appends
    prose annotations after the actual URL. A regex pass keeps legitimate URL
    punctuation such as parentheses while dropping separators and notes.
    """

    urls: list[str] = []
    for piece in WIKIPEDIA_URL_SEPARATOR_RE.split(value):
        match = WIKIPEDIA_URL_RE.search(piece)
        if not match:
            continue
        url = clean_wikipedia_url_token(match.group(0))
        if url:
            urls.append(url)
    return urls


def clean_wikipedia_url_token(value: str) -> str:
    url = value.strip().strip("'\"").rstrip(",;")
    if url.endswith(")") and "(" not in url.rsplit("/", 1)[-1]:
        url = url[:-1]
    if " " in url:
        url = url.split(" ", 1)[0]
    return url


def default_frames_output_dir(base_dir: str | Path = "data/frames") -> Path:
    return Path(base_dir)


def prepare_frames_artifacts(
    *,
    split: str = "test",
    question_limit: int | None = None,
    output_dir: str | Path = "data/frames",
    refresh: bool = False,
    session: requests.Session | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    emit_progress(progress_callback, {"command": "prepare-frames", "step": "start", "status": "start", "output_dir": str(output_dir)})
    target_dir = default_frames_output_dir(output_dir)
    corpus_dir = target_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = target_dir / "frames_mapping.json"
    report_path = target_dir / "prepare_report.json"

    emit_progress(progress_callback, {"command": "prepare-frames", "step": "dataset_load", "status": "start", "count": question_limit})
    dataset = load_dataset("google/frames-benchmark", split=split)
    if question_limit:
        dataset = dataset.select(range(min(question_limit, len(dataset))))
    rows = [dict(row) for row in dataset]
    emit_progress(progress_callback, {"command": "prepare-frames", "step": "dataset_load", "status": "ok", "count": len(rows)})

    client = session or requests.Session()
    downloaded_pages: dict[str, FramesPageArtifact] = {}
    mapping: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []

    for idx, row in enumerate(rows):
        question_id = question_id_for_row(row, idx)
        emit_progress(progress_callback, {"command": "prepare-frames", "step": "question", "status": "start", "index": idx + 1, "total": len(rows), "question_id": question_id})
        local_files: list[dict[str, Any]] = []
        for wikipedia_url in extract_wikipedia_urls(row):
            try:
                artifact = downloaded_pages.get(wikipedia_url)
                if artifact is None:
                    page_started = time.monotonic()
                    emit_progress(progress_callback, {"command": "prepare-frames", "step": "download", "status": "start", "question_id": question_id, "path": wikipedia_url})
                    artifact = download_wikipedia_page(
                        wikipedia_url,
                        corpus_dir=corpus_dir,
                        refresh=refresh,
                        session=client,
                    )
                    downloaded_pages[wikipedia_url] = artifact
                    emit_progress(progress_callback, {"command": "prepare-frames", "step": "download", "status": "ok", "question_id": question_id, "path": artifact.filename, "elapsed_seconds": time.monotonic() - page_started})
                local_files.append(
                    {
                        "path": artifact.filename,
                        "wikipedia_url": artifact.wikipedia_url,
                        "page_title": artifact.page_title,
                        "pageid": artifact.pageid,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                emit_progress(progress_callback, {"command": "prepare-frames", "step": "download", "status": "error", "question_id": question_id, "path": wikipedia_url, "exception": exc.__class__.__name__, "error": str(exc)})
                failures.append(
                    {
                        "question_id": question_id,
                        "wikipedia_url": wikipedia_url,
                        "error": str(exc),
                    }
                )
        mapping[question_id] = {
            "question": extract_question_text(row),
            "gold_answer": extract_gold_answer(row),
            "reasoning_types": extract_reasoning_types(row),
            "local_files": local_files,
        }
        emit_progress(progress_callback, {"command": "prepare-frames", "step": "question", "status": "ok", "index": idx + 1, "total": len(rows), "question_id": question_id, "count": len(local_files)})

    mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    emit_progress(progress_callback, {"command": "prepare-frames", "step": "mapping_write", "status": "ok", "path": str(mapping_path), "count": len(mapping)})
    report = {
        "split": split,
        "question_count": len(rows),
        "mapped_question_count": sum(1 for item in mapping.values() if item.get("local_files")),
        "unique_wikipedia_url_count": len(downloaded_pages),
        "downloaded_page_count": len(downloaded_pages),
        "failure_count": len(failures),
        "mapping_path": str(mapping_path),
        "corpus_dir": str(corpus_dir),
        "failures": failures,
    }
    write_json(report_path, report)
    emit_progress(progress_callback, {"command": "prepare-frames", "step": "report_write", "status": "ok", "path": str(report_path), "failure_count": len(failures)})
    emit_progress(progress_callback, {"command": "prepare-frames", "step": "complete", "status": "ok", "count": len(downloaded_pages), "failure_count": len(failures), "elapsed_seconds": time.monotonic() - started})
    return report


def download_wikipedia_page(
    wikipedia_url: str,
    *,
    corpus_dir: Path,
    refresh: bool = False,
    session: requests.Session | None = None,
) -> FramesPageArtifact:
    client = session or requests.Session()
    normalized_url = normalize_wikipedia_url(wikipedia_url, session=client)
    page_title = page_title_from_url(normalized_url)
    response = client.get(
        WIKIPEDIA_API_URL,
        params={
            "action": "query",
            "prop": "extracts|info",
            "titles": page_title,
            "inprop": "url",
            "explaintext": 1,
            "exsectionformat": "plain",
            "format": "json",
            "redirects": 1,
        },
        headers={"User-Agent": WIKIPEDIA_USER_AGENT},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    pages = payload.get("query", {}).get("pages", {})
    if not pages:
        raise ValueError(f"No Wikipedia page returned for {wikipedia_url}")
    page = next(iter(pages.values()))
    if "missing" in page:
        raise ValueError(f"Wikipedia page missing for {normalized_url}")
    resolved_title = page.get("title") or page_title
    extract = (page.get("extract") or "").strip()
    if not extract:
        extract = fetch_wikipedia_wikitext(resolved_title, session=client).strip()
    if not extract:
        raise ValueError(f"Wikipedia page has no extract for {normalized_url}")
    filename = build_page_filename(resolved_title, str(page.get("pageid") or ""))
    path = corpus_dir / filename
    if refresh or not path.exists():
        path.write_text(build_page_text(resolved_title, normalized_url, extract), encoding="utf-8")
    return FramesPageArtifact(
        wikipedia_url=normalized_url,
        page_title=resolved_title,
        filename=filename,
        pageid=str(page.get("pageid")) if page.get("pageid") is not None else None,
        path=path,
    )


def normalize_wikipedia_url(wikipedia_url: str, *, session: requests.Session | None = None) -> str:
    url = clean_wikipedia_url_token(wikipedia_url)
    parsed = urlparse(url)
    if parsed.netloc == WIKIPEDIA_SHORT_HOST:
        client = session or requests.Session()
        response = client.get(url, headers={"User-Agent": WIKIPEDIA_USER_AGENT}, timeout=30, allow_redirects=True)
        response.raise_for_status()
        return response.url
    return url


def fetch_wikipedia_wikitext(title: str, *, session: requests.Session | None = None) -> str:
    client = session or requests.Session()
    response = client.get(
        WIKIPEDIA_API_URL,
        params={
            "action": "query",
            "prop": "revisions",
            "titles": title,
            "rvprop": "content",
            "rvslots": "main",
            "format": "json",
            "redirects": 1,
        },
        headers={"User-Agent": WIKIPEDIA_USER_AGENT},
        timeout=60,
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    revisions = page.get("revisions") or []
    if not revisions:
        return ""
    slots = revisions[0].get("slots") or {}
    main = slots.get("main") or {}
    return str(main.get("*") or main.get("content") or "")


def build_page_text(title: str, wikipedia_url: str, extract: str) -> str:
    return f"Title: {title}\nSource: {wikipedia_url}\n\n{extract.strip()}\n"


def page_title_from_url(wikipedia_url: str) -> str:
    parsed = urlparse(wikipedia_url)
    path = parsed.path
    query = parse_qs(parsed.query)
    if "/wiki/" in path:
        slug = path.split("/wiki/", 1)[1]
        return _decode_wikipedia_title(slug)
    if path.endswith("/w/index.php"):
        title = query.get("title", [""])[0]
        if title and title != "Special:Search":
            return _decode_wikipedia_title(title)
        search = query.get("search", [""])[0]
        if search:
            return _decode_wikipedia_title(search)
    raise ValueError(f"Unsupported Wikipedia URL: {wikipedia_url}")


def _decode_wikipedia_title(value: str) -> str:
    decoded = value
    for _ in range(3):
        next_decoded = unquote(decoded)
        if next_decoded == decoded:
            break
        decoded = next_decoded
    return decoded.replace("_", " ")


def build_page_filename(title: str, pageid: str | None) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_") or "page"
    if pageid:
        slug = f"{slug}_{pageid}"
    return f"{slug}.txt"


def _dedupe_preserve_order(values) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
