# ragflow-bench

`ragflow-bench` is a standalone Python CLI for benchmarking a running RAGFlow instance over raw HTTP APIs.

It is **not** part of the RAGFlow repository, does **not** import RAGFlow source code, and does **not** depend on RAGFlow internals. It treats RAGFlow as an external service reachable through HTTP.

## What it does

- benchmark retrieval and answer generation against a live RAGFlow deployment
- support FRAMES, EnterpriseRAG-Bench, and future custom corpora
- upload and parse benchmark documents through RAGFlow HTTP endpoints
- write incremental JSONL results so partial runs survive failures
- score deterministic metrics only: exact match, normalized exact match, source recall, and failure type

## Requirements

- Python 3.11+
- a reachable RAGFlow instance
- a valid RAGFlow API key
- a valid RAGFlow chat model ID for chat-based runs

Default base URL:

```bash
http://127.0.0.1:80
```

## Installation

```bash
pip install -e .
```

## Environment variables

```bash
export RAGFLOW_BASE_URL="http://127.0.0.1:80"
export RAGFLOW_API_KEY="..."
export RAGFLOW_LLM_ID="..."
```

Rules:
- `RAGFLOW_BASE_URL` defaults to `http://127.0.0.1:80`
- `RAGFLOW_API_KEY` is required unless supplied directly in config
- `RAGFLOW_LLM_ID` is required for chat creation and answer generation
- `ragflow-bench` auto-loads `.env` from the current working directory without overriding already-exported shell variables
- `.env` and `.env.*` are ignored by git
- `.env.example` contains placeholders only

## CLI

```bash
ragflow-bench --help
ragflow-bench wizard
ragflow-bench doctor
ragflow-bench ingest --config configs/frames_smoke.yaml
ragflow-bench retrieve --config configs/frames_smoke.yaml
ragflow-bench run --config configs/frames_smoke.yaml
ragflow-bench score --results outputs/<run>/results.jsonl
```

## Doctor

`doctor` verifies the live RAGFlow connection and probes the HTTP workflow:

- base URL reachable
- API key accepted
- docs-backed health assumptions match the live server where possible
- dataset creation works
- document upload works
- parsing can be started and polled
- retrieval works
- chat creation/session/completion work when `RAGFLOW_LLM_ID` is set

If `RAGFLOW_LLM_ID` is missing, doctor reports partial success and lists available models when possible.

When a config creates a new dataset, `doctor` also warns if `dataset.embedding_model` is omitted, because parsing/indexing will then depend on RAGFlow's server-side default dataset behavior.

## Wizard usage

`ragflow-bench wizard` interactively writes a YAML config, prints the exact run command, and can optionally run it immediately.

## Config usage

Configs live under `configs/` and can be adapted for local corpora or FRAMES mappings.

For new datasets, set `dataset.embedding_model` explicitly so RAGFlow uses the intended embedding model during parsing/indexing:

```yaml
dataset:
  strategy: create_new_dataset_and_ingest_documents
  name: frames-oracle
  embedding_model: your-ragflow-embedding-model-id
  chunk_method: naive
```

If you reuse an existing dataset, `ragflow-bench` does not override that dataset's embedding model.

Every run writes:
- `results.jsonl`
- `results.csv`
- `summary.json`
- `config.resolved.yaml`
- `document_registry.json` when ingestion is used

## FRAMES setup

FRAMES uses the Hugging Face dataset:

```python
from datasets import load_dataset
load_dataset("google/frames-benchmark", split="test")
```

This repo does **not** download Wikipedia content. Provide a local mapping file from benchmark question IDs to local files and source URLs instead.

## EnterpriseRAG-Bench setup

Provide:
- a local corpus directory
- a local questions file (`json`, `jsonl`, `csv`, or `parquet`)
- optional document metadata manifest if available

The adapter is designed so source-type-specific parsing can be extended later.

## Known limitations

- no built-in Wikipedia downloader for FRAMES in this iteration
- no LLM-as-judge scoring yet
- live server behavior wins over docs when they disagree
- chat validation still depends on the selected live `RAGFLOW_LLM_ID`
- if parsing fails on a newly created dataset, verify `dataset.embedding_model` and the corresponding provider binding in the live RAGFlow server

## Standalone guarantee

This project does **not** need to live inside the RAGFlow repository.
It only speaks to RAGFlow through HTTP.
