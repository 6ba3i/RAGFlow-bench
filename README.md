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
export ZHIPU_API_KEY="..."
export ZHIPU_JUDGE_MODEL="glm-4-flash"
export ZHIPU_JUDGE_TIMEOUT_SECONDS="120"
export ZHIPU_JUDGE_MAX_RETRIES="6"
export ZHIPU_JUDGE_BACKOFF_SECONDS="2.0"
export ZHIPU_JUDGE_MAX_BACKOFF_SECONDS="30.0"
```

Rules:
- `RAGFLOW_BASE_URL` defaults to `http://127.0.0.1:80`
- `RAGFLOW_API_KEY` is required unless supplied directly in config
- `RAGFLOW_LLM_ID` is required for chat creation and answer generation
- `ZHIPU_API_KEY` is required for LLM-judge scoring unless supplied directly in config
- `ZHIPU_JUDGE_TIMEOUT_SECONDS`, `ZHIPU_JUDGE_MAX_RETRIES`, `ZHIPU_JUDGE_BACKOFF_SECONDS`, and `ZHIPU_JUDGE_MAX_BACKOFF_SECONDS` control judge timeout and transient-failure retries
- `ragflow-bench` auto-loads `.env` from the current working directory without overriding already-exported shell variables
- `.env` and `.env.*` are ignored by git
- `.env.example` contains placeholders only

## CLI

```bash
ragflow-bench --help
ragflow-bench wizard
ragflow-bench doctor
ragflow-bench prepare-frames
ragflow-bench ingest --config configs/frames_smoke.yaml
ragflow-bench retrieve --config configs/frames_smoke.yaml
ragflow-bench run --config configs/frames_smoke.yaml
ragflow-bench retry-failed --run-dir outputs/<run>
ragflow-bench score --results outputs/<run>/results.jsonl
ragflow-bench judge --results outputs/<run>/results.jsonl --config configs/frames_oracle.yaml
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

`ragflow-bench retry-failed --run-dir outputs/<run>` reruns rows with non-empty `error` using the saved resolved config, stores raw retry attempts under `retries/`, backs up current result artifacts under `backups/`, replaces matching `question_id` rows in-place, and rebuilds `results.csv` plus `summary.json`.

## LLM judge scoring

Use the standalone `judge` command after a run to score free-form answers with a Zhipu GLM model:

```bash
ragflow-bench judge --results outputs/<run>/results.jsonl --config configs/frames_oracle.yaml
```

Switch models just by changing the configured or overridden model name:

```bash
ragflow-bench judge --results outputs/<run>/results.jsonl --config configs/frames_oracle.yaml --model glm-4-flash
ragflow-bench judge --results outputs/<run>/results.jsonl --config configs/frames_oracle.yaml --model glm-4.7-flash
```

The judge command writes:
- `judge_results.jsonl`
- `judge_results.csv`
- `judge_summary.json`

Judge config shape:

```yaml
judge:
  provider: zhipu
  api_key_env_var: ZHIPU_API_KEY
  model: glm-4-flash
  temperature: 0.0
  timeout_seconds: 120
  max_retries: 6
  backoff_seconds: 2.0
  max_backoff_seconds: 30.0
```

The judge evaluates semantic correctness against the gold answer, so different wording can still be marked correct.

Judge retries only transient failures: HTTP 429, HTTP 5xx, and `ReadTimeout`. When the provider sends `Retry-After`, the judge honors it up to the configured max backoff. Authentication and malformed-response errors still fail fast.

## FRAMES setup

FRAMES uses the Hugging Face dataset:

```python
from datasets import load_dataset
load_dataset("google/frames-benchmark", split="test")
```

FRAMES questions and gold answers come from Hugging Face, but RAGFlow still needs a local corpus to ingest. Use:

```bash
ragflow-bench prepare-frames
```

By default this creates:

- `data/frames/frames_mapping.json`
- `data/frames/corpus/`
- `data/frames/prepare_report.json`

The command reads the FRAMES question rows, extracts the referenced Wikipedia links, downloads page text through Wikipedia's HTTP API, deduplicates repeated pages, and writes the mapping file expected by the FRAMES adapter.

Then run either:

```bash
ragflow-bench ingest --config configs/frames_oracle.yaml
ragflow-bench run --config configs/frames_oracle.yaml
```

or retrieve directly, with behavior determined by `dataset.strategy`:

```bash
ragflow-bench retrieve --config configs/frames_smoke.yaml
```

- `reuse_existing_dataset` → retrieve from the configured existing dataset
- `create_new_dataset` → create a fresh empty dataset, then retrieve
- `create_new_dataset_and_ingest_documents` → create a fresh dataset, ingest local corpus documents, wait for parsing, then retrieve

## EnterpriseRAG-Bench setup

Prepare the Onyx EnterpriseRAG-Bench artifacts from Hugging Face:

```bash
ragflow-bench prepare-eragb
```

By default this creates:
- `data/eragb/corpus/`
- `data/eragb/questions.jsonl`
- `data/eragb/documents_manifest.json`
- `data/eragb/prepare_report.json`

The command downloads the questions/documents parquet files, converts document rows into local text files, and preserves Onyx `doc_id` values as document `source_uri` metadata for source-recall scoring. For small local tests, pass `--document-limit` and `--question-limit`.

For lower file-count runs where exact source-reference accuracy is not required, use merged shard mode:

```bash
ragflow-bench prepare-eragb --merge-documents --filter-questions-with-missing-docs
```

Merged mode writes deterministic source-type shards, `doc_id_to_shard.json`, `shard_manifest.json`, and `parser_config.merged.yaml`. Shard text contains only a boundary delimiter plus original content to minimize retrieval bias; doc IDs, titles, and source types stay in sidecar manifests. It maps question references to shard URIs by default and labels `reference_granularity` as `shard`; do not compare shard recall with document-level leaderboard source recall.

## Known limitations

- LLM-as-judge quality depends on the selected judge model and prompt; validate cheaper judge models before using them as final benchmark authority
- live server behavior wins over docs when they disagree
- chat validation still depends on the selected live `RAGFLOW_LLM_ID`
- if parsing fails on a newly created dataset, verify `dataset.embedding_model` and the corresponding provider binding in the live RAGFlow server

## Standalone guarantee

This project does **not** need to live inside the RAGFlow repository.
It only speaks to RAGFlow through HTTP.
