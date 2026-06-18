from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def append_jsonl(path: str | Path, row: dict) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def jsonl_to_csv(jsonl_path: str | Path, csv_path: str | Path) -> None:
    rows = [json.loads(line) for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if rows:
        pd.DataFrame(rows).to_csv(csv_path, index=False)


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
