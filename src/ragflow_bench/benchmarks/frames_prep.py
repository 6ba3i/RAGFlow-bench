from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests
from datasets import load_dataset

from ragflow_bench.reports.writers import write_json

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_USER_AGENT = "ragflow-bench/0.1.0 (https://github.com/openai/ragflow-bench; benchmarking tool)"


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
            direct_fields.append(str(value))
    if direct_fields:
        return _dedupe_preserve_order(direct_fields)

    wiki_links = row.get("wiki_links")
    if isinstance(wiki_links, list):
        return _dedupe_preserve_order(str(item) for item in wiki_links if item)
    if isinstance(wiki_links, str) and wiki_links.strip():
        try:
            parsed = json.loads(wiki_links.replace("'", '"'))
            if isinstance(parsed, list):
                return _dedupe_preserve_order(str(item) for item in parsed if item)
        except json.JSONDecodeError:
            return _dedupe_preserve_order(
                piece.strip().strip("'").strip('"')
                for piece in wiki_links.strip("[]").split(",")
                if piece.strip()
            )
    return []


def default_frames_output_dir(base_dir: str | Path = "data/frames") -> Path:
    return Path(base_dir)


def prepare_frames_artifacts(
    *,
    split: str = "test",
    question_limit: int | None = None,
    output_dir: str | Path = "data/frames",
    refresh: bool = False,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    target_dir = default_frames_output_dir(output_dir)
    corpus_dir = target_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = target_dir / "frames_mapping.json"
    report_path = target_dir / "prepare_report.json"

    dataset = load_dataset("google/frames-benchmark", split=split)
    if question_limit:
        dataset = dataset.select(range(min(question_limit, len(dataset))))
    rows = [dict(row) for row in dataset]

    client = session or requests.Session()
    downloaded_pages: dict[str, FramesPageArtifact] = {}
    mapping: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []

    for idx, row in enumerate(rows):
        question_id = question_id_for_row(row, idx)
        local_files: list[dict[str, Any]] = []
        for wikipedia_url in extract_wikipedia_urls(row):
            try:
                artifact = downloaded_pages.get(wikipedia_url)
                if artifact is None:
                    artifact = download_wikipedia_page(
                        wikipedia_url,
                        corpus_dir=corpus_dir,
                        refresh=refresh,
                        session=client,
                    )
                    downloaded_pages[wikipedia_url] = artifact
                local_files.append(
                    {
                        "path": artifact.filename,
                        "wikipedia_url": artifact.wikipedia_url,
                        "page_title": artifact.page_title,
                        "pageid": artifact.pageid,
                    }
                )
            except Exception as exc:  # noqa: BLE001
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

    mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
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
    return report


def download_wikipedia_page(
    wikipedia_url: str,
    *,
    corpus_dir: Path,
    refresh: bool = False,
    session: requests.Session | None = None,
) -> FramesPageArtifact:
    page_title = page_title_from_url(wikipedia_url)
    client = session or requests.Session()
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
        raise ValueError(f"Wikipedia page missing for {wikipedia_url}")
    resolved_title = page.get("title") or page_title
    extract = (page.get("extract") or "").strip()
    if not extract:
        raise ValueError(f"Wikipedia page has no extract for {wikipedia_url}")
    filename = build_page_filename(resolved_title, str(page.get("pageid") or ""))
    path = corpus_dir / filename
    if refresh or not path.exists():
        path.write_text(build_page_text(resolved_title, wikipedia_url, extract), encoding="utf-8")
    return FramesPageArtifact(
        wikipedia_url=wikipedia_url,
        page_title=resolved_title,
        filename=filename,
        pageid=str(page.get("pageid")) if page.get("pageid") is not None else None,
        path=path,
    )


def build_page_text(title: str, wikipedia_url: str, extract: str) -> str:
    return f"Title: {title}\nSource: {wikipedia_url}\n\n{extract.strip()}\n"


def page_title_from_url(wikipedia_url: str) -> str:
    parsed = urlparse(wikipedia_url)
    path = parsed.path
    if "/wiki/" not in path:
        raise ValueError(f"Unsupported Wikipedia URL: {wikipedia_url}")
    slug = path.split("/wiki/", 1)[1]
    return unquote(slug).replace("_", " ")


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
