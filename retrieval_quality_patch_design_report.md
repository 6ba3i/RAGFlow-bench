# RAGFlow EnterpriseRAG-Bench Patch Design Report

## 1. Executive Summary

**Direct opinion:** implement benchmark-harness diagnostics first, not RAGFlow core changes and not larger context. The first five changes should be:

1. **Diff 1: source/shard mapping + recall metrics** — evaluation fix, diagnostic improvement. It fixes the current `source_recall=0.0` measurement bug.
2. **Diff 2: final-context and citation diagnostics** — evaluation/context-selection diagnostic improvement. It separates raw retrieval misses from final prompt/context misses.
3. **Diff 4: historical artifact recompute command** — evaluation fix, diagnostic improvement. It lets the June 26 run be rescored without a new full benchmark.
4. **Diff 3: failure classification rewrite** — evaluation fix, diagnostic improvement. It stops collapsing all non-exact zero-source rows into `retrieval_failure`.
5. **Diff 11: judge/infra stabilization** — infra/evaluation fix. It reduces the 297/500 excluded-row problem before trusting any quality deltas.

**Benchmark-harness fixes:** Diffs 1, 2, 3, 4, 6, 7, 8, and 11 are primarily in `/home/idriss/RAGFlow-bench` and should land before any full rerun.

**RAGFlow or ingestion improvements:** Diff 5 is an ingestion-side benchmark improvement that requires reingestion. RAGFlow core candidates in Section 12 should wait until the harness proves the bottleneck remains after benchmark-side fixes.

**Prompt/synthesis improvements:** Diff 7 adds an exact-fact prompt mode targeted at rows where the expected shard is already in final context but the answer is wrong or partial.

**Reject as score inflation:** do **not** raise `chat.top_n`, `retrieval.page_size`, or `chat.max_tokens` as the primary fix. Those can be later controlled experiments, but only at fixed-budget or with explicit inflation labels.

External evidence used for patch design:

- RAGFlow HTTP/API and source expose retrieval/chat knobs including `top_k`, `top_n`, `similarity_threshold`, `vector_similarity_weight`, `quote`, and `rerank_id`: <https://ragflow.io/docs/http_api_reference>, <https://ragflow.io/docs/start_chat>, and local RAGFlow source paths cited below.
- RAGFlow retrieval-test docs describe hybrid retrieval using keyword similarity, vector similarity, and reranker score when a reranker is configured: <https://ragflow.io/docs/run_retrieval_test>.
- RAGFlow child chunking docs support the principle that smaller retrievable child chunks can improve recall while preserving broader parent context: <https://ragflow.io/docs/configure_child_chunking_strategy>.
- Azure AI Search chunking guidance supports fixed, measured chunking choices rather than arbitrary context expansion: <https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-chunk-documents>.
- Elasticsearch RRF documentation supports rank-fusion experiments because RRF combines ranked lists without requiring comparable score scales: <https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion>.
- Ragas context recall docs support separating context recall from answer correctness: <https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/>.

## 2. Prior Evidence Revalidated

Local artifacts rechecked:

- `outputs/enterprise_rag_bench_full_20260626_101748/summary.json`
- `outputs/enterprise_rag_bench_full_20260626_101748/judge_summary.json`
- `outputs/enterprise_rag_bench_full_20260626_101748/results.jsonl`
- `outputs/enterprise_rag_bench_full_20260626_101748/judge_results.jsonl`
- `outputs/enterprise_rag_bench_full_20260626_101748/config.resolved.yaml`
- `outputs/enterprise_rag_bench_full_20260626_101748/document_registry.json`

Confirmed counts:

- 500 total questions.
- 203 scorable judged rows.
- 67 correct / 40 partial / 96 incorrect.
- Strict accuracy: 67/203 = 33.0%.
- Partial-credit accuracy: 42.9%.
- Partial-or-better: 107/203 = 52.7%.
- 297/500 excluded rows: 99 answer/infra-error rows and 198 judge-error rows.
- `summary.json` reports `average_source_recall=0.0` and `failure_type_counts.retrieval_failure=420`, but this is a harness mapping bug: expected sources are canonical ERAGB shard URIs while retrieved chunks expose `document_id`, `document_keyword`, `document_name`, chunk IDs, and scores rather than canonical `source_uri`.
- Over 203 scorable rows, expected shard basename matching reproduced:
  - expected shard in raw top-20 for 97/203;
  - expected shard in final context for 90/203;
  - 78/96 incorrect rows missing expected shard from raw top-20;
  - 7/203 rows with expected shard in raw retrieval but not final context;
  - 13 incorrect rows with expected shard in final context;
  - 25 partial rows with expected shard in final context.
- Weak source types by scorable breakdown remain Confluence, Slack, and Fireflies. Revalidated counts by expected-source membership:
  - Confluence: 54 rows, 7 correct, 19 partial, 28 incorrect.
  - Slack: 33 rows, 6 correct, 11 partial, 16 incorrect.
  - Fireflies: 11 rows, 2 correct, 1 partial, 8 incorrect.

## 3. Patch Design Principles

1. **Fixed-budget quality before larger context.** Any change that only improves score by adding more chunks or token budget is potential score inflation.
2. **Source/shard mapping before new full runs.** The current `source_recall=0.0` invalidates source-quality conclusions.
3. **Raw retrieval recall and final-context recall must be separate.** The 7 raw-present/final-absent rows prove context selection can fail after broad retrieval succeeds.
4. **No `top_n`/`max_tokens` inflation as a primary fix.** Larger budgets may be useful later as labeled ablations, not baseline fixes.
5. **Every change must have proof and validation.** Unit tests, artifact-level recomputation, a small diagnostic slice, or a manual invariant is required.
6. **Benchmark-side proof before RAGFlow core patches.** RAGFlow search/prompt patches carry higher risk and should follow harness diagnostics.

## 4. Proposed Diff 1: Source/Shard Mapping and Recall Metrics

**Labels:** evaluation fix; diagnostic improvement.  
**Target functions/classes:** `source_recall`, new `canonical_source_candidates`, `retrieval_diagnostics`, and `run_benchmark` scoring section.

### Proposed unified diff

```diff
diff --git a/src/ragflow_bench/scoring/retrieval_scoring.py b/src/ragflow_bench/scoring/retrieval_scoring.py
index 0c0f111..8d2a901 100644
--- a/src/ragflow_bench/scoring/retrieval_scoring.py
+++ b/src/ragflow_bench/scoring/retrieval_scoring.py
@@ -1,9 +1,99 @@
 from __future__ import annotations
+
+from pathlib import PurePosixPath
+from typing import Any
+
+
+ERAGB_SHARD_PREFIX = "eragb-shard://"
+
+
+def _as_text(value: Any) -> str | None:
+    if value is None:
+        return None
+    text = str(value).strip()
+    return text or None
+
+
+def _basename(value: str) -> str:
+    return PurePosixPath(value.replace(ERAGB_SHARD_PREFIX, "")).name
+
+
+def eragb_shard_uri_from_name(value: str) -> str | None:
+    text = value.strip()
+    name = _basename(text)
+    if not name.endswith(".txt") or "_shard_" not in name:
+        return None
+    source_type = name.split("_shard_", 1)[0]
+    return f"{ERAGB_SHARD_PREFIX}{source_type}/{name}"
+
+
+def canonical_source_candidates(chunk: dict[str, Any]) -> list[str]:
+    """Return canonical and audit-friendly source candidates for a RAGFlow chunk."""
+    metadata = chunk.get("metadata") or chunk.get("meta_fields") or chunk.get("document_metadata") or {}
+    raw_values: list[str] = []
+    for container in (chunk, metadata):
+        if not isinstance(container, dict):
+            continue
+        for key in (
+            "source_uri",
+            "canonical_source_uri",
+            "eragb_shard_uri",
+            "document_keyword",
+            "document_name",
+            "docnm_kwd",
+            "doc_id",
+            "document_id",
+            "chunk_id",
+            "id",
+        ):
+            text = _as_text(container.get(key))
+            if text:
+                raw_values.append(text)
+    candidates: list[str] = []
+    for value in raw_values:
+        candidates.append(value)
+        shard_uri = eragb_shard_uri_from_name(value)
+        if shard_uri:
+            candidates.append(shard_uri)
+        base = _basename(value)
+        if base != value:
+            candidates.append(base)
+    seen: set[str] = set()
+    ordered: list[str] = []
+    for item in candidates:
+        if item not in seen:
+            seen.add(item)
+            ordered.append(item)
+    return ordered
+
+
+def source_recall(expected_sources: list[str], retrieved_sources: list[str]) -> float:
+    if not expected_sources:
+        return 0.0
+    expected = set(expected_sources)
+    retrieved = set(retrieved_sources)
+    return len(expected & retrieved) / len(expected)
+
+
+def shard_recall(expected_sources: list[str], chunks: list[dict[str, Any]]) -> tuple[float, int | None, float]:
+    expected = [src for src in expected_sources if str(src).startswith(ERAGB_SHARD_PREFIX)]
+    if not expected:
+        return 0.0, None, 0.0
+    expected_set = set(expected)
+    first_rank: int | None = None
+    hits: set[str] = set()
+    for rank, chunk in enumerate(chunks, start=1):
+        if not isinstance(chunk, dict):
+            continue
+        candidates = set(canonical_source_candidates(chunk))
+        matched = expected_set & candidates
+        if matched:
+            hits.update(matched)
+            first_rank = first_rank or rank
+    recall = len(hits) / len(expected_set)
+    mrr = 1.0 / first_rank if first_rank else 0.0
+    return recall, first_rank, mrr
+
+
+def retrieval_diagnostics(expected_sources: list[str], chunks: list[dict[str, Any]], *, prefix: str) -> dict[str, Any]:
+    candidates = [canonical_source_candidates(chunk) for chunk in chunks if isinstance(chunk, dict)]
+    flat = [item for group in candidates for item in group]
+    recall, rank, mrr = shard_recall(expected_sources, chunks)
+    return {
+        f"{prefix}_source_candidates": candidates,
+        f"{prefix}_retrieved_source_uris": [item for item in flat if item.startswith(ERAGB_SHARD_PREFIX)],
+        f"{prefix}_shard_recall": recall,
+        f"{prefix}_expected_rank": rank,
+        f"{prefix}_mrr": mrr,
+    }
-
-
-def source_recall(expected_sources: list[str], retrieved_sources: list[str]) -> float:
-    if not expected_sources:
-        return 0.0
-    expected = set(expected_sources)
-    retrieved = set(retrieved_sources)
-    return len(expected & retrieved) / len(expected)
diff --git a/src/ragflow_bench/execution/benchmark_runner.py b/src/ragflow_bench/execution/benchmark_runner.py
index 6df3dfd..13d77e7 100644
--- a/src/ragflow_bench/execution/benchmark_runner.py
+++ b/src/ragflow_bench/execution/benchmark_runner.py
@@ -18,7 +18,7 @@ from ragflow_bench.rate_limits import run_with_rate_limit_retries
 from ragflow_bench.reports.summary import build_summary
 from ragflow_bench.reports.writers import append_jsonl, jsonl_to_csv, write_json
-from ragflow_bench.scoring import classify_failure, exact_match, normalized_match, source_recall
+from ragflow_bench.scoring import classify_failure, exact_match, normalized_match, retrieval_diagnostics, source_recall
@@ -135,19 +135,23 @@ def run_benchmark(config: AppConfig, client, question_ids: set[str] | None = Non
         chunks = raw_retrieval.get("chunks", []) if isinstance(raw_retrieval, dict) else []
+        final_chunks = (raw_response.get("reference") or {}).get("chunks", []) if isinstance(raw_response, dict) else []
         retrieved_document_ids = [chunk.get("doc_id") or chunk.get("document_id") for chunk in chunks if isinstance(chunk, dict)]
         retrieved_chunk_ids = [chunk.get("chunk_id") or chunk.get("id") for chunk in chunks if isinstance(chunk, dict)]
         retrieved_scores = [chunk.get("score") or chunk.get("similarity") for chunk in chunks if isinstance(chunk, dict)]
-        retrieved_source_uris = []
-        for chunk in chunks:
-            if not isinstance(chunk, dict):
-                continue
-            metadata = chunk.get("metadata") or chunk.get("meta_fields") or {}
-            source_uri = metadata.get("source_uri") or chunk.get("source_uri")
-            if source_uri:
-                retrieved_source_uris.append(source_uri)
+        raw_diag = retrieval_diagnostics(question.expected_sources, chunks, prefix="raw_retrieval")
+        final_diag = retrieval_diagnostics(question.expected_sources, final_chunks, prefix="final_context")
+        retrieved_source_uris = raw_diag["raw_retrieval_retrieved_source_uris"]
         em = exact_match(question.gold_answer, ragflow_answer)
         nm = normalized_match(question.gold_answer, ragflow_answer)
         recall = source_recall(question.expected_sources, retrieved_source_uris)
-        failure = classify_failure(error=error, ragflow_answer=ragflow_answer, exact_match=em, source_recall=recall)
+        failure = classify_failure(
+            error=error,
+            ragflow_answer=ragflow_answer,
+            exact_match=em,
+            source_recall=recall,
+            raw_retrieval_shard_recall=raw_diag["raw_retrieval_shard_recall"],
+            final_context_shard_recall=final_diag["final_context_shard_recall"],
+        )
@@ -165,7 +169,13 @@ def run_benchmark(config: AppConfig, client, question_ids: set[str] | None = Non
             "retrieved_scores": retrieved_scores,
             "source_recall": recall,
+            "raw_retrieval_shard_recall@20": raw_diag["raw_retrieval_shard_recall"],
+            "raw_retrieval_expected_rank": raw_diag["raw_retrieval_expected_rank"],
+            "raw_retrieval_mrr": raw_diag["raw_retrieval_mrr"],
+            "final_context_shard_recall@top_n": final_diag["final_context_shard_recall"],
+            "final_context_expected_rank": final_diag["final_context_expected_rank"],
+            "final_context_mrr": final_diag["final_context_mrr"],
             "reasoning_types": question.reasoning_types,
             "failure_type": failure,
+            "retrieval_diagnostics": {**raw_diag, **final_diag},
             "raw_retrieval": raw_retrieval,
```

### Before/after behavior

Before: `retrieved_source_uris` only used `metadata.source_uri`, `meta_fields.source_uri`, or top-level `source_uri`. The June 26 chunks had `document_keyword`/`document_name`, so recall was always zero.

After: the scorer extracts candidates from RAGFlow-visible fields and maps names like `github_shard_000104.txt` to `eragb-shard://github/github_shard_000104.txt`. It reports raw retrieval recall, final-context recall, expected rank, and MRR separately.

### Proof

The June 26 scorable rows include RAGFlow chunks with keys `document_keyword`, `document_id`, `similarity`, `term_similarity`, `vector_similarity` in raw retrieval and `document_name`, `document_metadata`, `similarity`, `term_similarity`, `vector_similarity` in final context. Expected sources are `eragb-shard://...`. The exact-set metric cannot match these without canonicalization.

### Validation

```diff
diff --git a/tests/test_scoring.py b/tests/test_scoring.py
index 7b1de88..63eedde 100644
--- a/tests/test_scoring.py
+++ b/tests/test_scoring.py
@@ -1,5 +1,6 @@
-from ragflow_bench.scoring import classify_failure, exact_match, normalized_match, source_recall
+from ragflow_bench.scoring import classify_failure, exact_match, normalized_match, retrieval_diagnostics, source_recall
@@
 def test_source_recall():
     assert source_recall(["a", "b"], ["b", "c"]) == 0.5
+
+
+def test_eragb_shard_recall_maps_ragflow_document_keyword():
+    diag = retrieval_diagnostics(
+        ["eragb-shard://github/github_shard_000104.txt"],
+        [{"document_keyword": "github_shard_000104.txt", "similarity": 0.9}],
+        prefix="raw_retrieval",
+    )
+    assert diag["raw_retrieval_shard_recall"] == 1.0
+    assert diag["raw_retrieval_expected_rank"] == 1
+    assert diag["raw_retrieval_mrr"] == 1.0
```

Artifact validation: run the read-only recompute command from Diff 4 against the June 26 output and confirm raw hits become 97/203 and final hits become 90/203 instead of `source_recall=0.0` for every row.

## 5. Proposed Diff 2: Persist Final Context and Citation Diagnostics

**Labels:** evaluation fix + context-selection diagnostic; diagnostic improvement.  
**Target functions/classes:** `run_benchmark`, new helpers in `benchmark_runner.py` or `retrieval_scoring.py`.

### Proposed unified diff

```diff
diff --git a/src/ragflow_bench/execution/benchmark_runner.py b/src/ragflow_bench/execution/benchmark_runner.py
index 13d77e7..ea82b30 100644
--- a/src/ragflow_bench/execution/benchmark_runner.py
+++ b/src/ragflow_bench/execution/benchmark_runner.py
@@ -39,6 +39,35 @@ def make_adapter(config: AppConfig) -> BenchmarkAdapter:
     return CustomBenchmarkAdapter(config)
+
+
+def _chunk_audit_record(chunk: dict, *, rank: int, survived_final_context: bool | None = None) -> dict:
+    metadata = chunk.get("metadata") or chunk.get("meta_fields") or chunk.get("document_metadata") or {}
+    return {
+        "rank": rank,
+        "chunk_id": chunk.get("chunk_id") or chunk.get("id"),
+        "doc_id": chunk.get("doc_id") or chunk.get("document_id"),
+        "document_name": chunk.get("document_name") or chunk.get("document_keyword") or chunk.get("docnm_kwd"),
+        "source_uri": metadata.get("source_uri") or chunk.get("source_uri"),
+        "similarity": chunk.get("score") or chunk.get("similarity"),
+        "term_similarity": chunk.get("term_similarity"),
+        "vector_similarity": chunk.get("vector_similarity"),
+        "positions": chunk.get("positions"),
+        "metadata": metadata,
+        "survived_final_context": survived_final_context,
+    }
+
+
+def _citation_chunk_ids(answer: str | None, final_chunks: list[dict]) -> list[str]:
+    if not answer:
+        return []
+    ids: list[str] = []
+    for match in re.finditer(r"\[ID:(\d+)\]", answer):
+        idx = int(match.group(1))
+        if 0 <= idx < len(final_chunks):
+            chunk = final_chunks[idx]
+            ids.append(str(chunk.get("id") or chunk.get("chunk_id") or idx))
+    return ids
@@ -135,6 +164,12 @@ def run_benchmark(config: AppConfig, client, question_ids: set[str] | None = Non
         chunks = raw_retrieval.get("chunks", []) if isinstance(raw_retrieval, dict) else []
         final_chunks = (raw_response.get("reference") or {}).get("chunks", []) if isinstance(raw_response, dict) else []
+        final_chunk_ids = {str(chunk.get("id") or chunk.get("chunk_id")) for chunk in final_chunks if isinstance(chunk, dict)}
+        raw_retrieval_audit = [
+            _chunk_audit_record(chunk, rank=rank, survived_final_context=str(chunk.get("id") or chunk.get("chunk_id")) in final_chunk_ids)
+            for rank, chunk in enumerate(chunks, start=1) if isinstance(chunk, dict)
+        ]
+        final_context_audit = [_chunk_audit_record(chunk, rank=rank, survived_final_context=True) for rank, chunk in enumerate(final_chunks, start=1) if isinstance(chunk, dict)]
@@ -170,6 +205,12 @@ def run_benchmark(config: AppConfig, client, question_ids: set[str] | None = Non
             "raw_retrieval_mrr": raw_diag["raw_retrieval_mrr"],
             "final_context_shard_recall@top_n": final_diag["final_context_shard_recall"],
             "final_context_expected_rank": final_diag["final_context_expected_rank"],
             "final_context_mrr": final_diag["final_context_mrr"],
+            "raw_retrieval_chunks": raw_retrieval_audit,
+            "final_context_chunks": final_context_audit,
+            "citation_chunk_ids": _citation_chunk_ids(ragflow_answer, final_chunks),
+            "prompt_chunk_count": len(final_context_audit),
+            "chat_top_n": config.chat.top_n,
+            "retrieval_page_size": config.retrieval.page_size,
             "reasoning_types": question.reasoning_types,
```

Also add `import re` at the top of `benchmark_runner.py`.

### Before/after behavior

Before: the full raw payloads are present, but summary/classification must repeatedly rediscover field shapes. Final context exists only buried inside `raw_response.reference.chunks`; citation references are not normalized.

After: every row has stable audit arrays for raw retrieval and final context, with rank, IDs, document name, similarity, term/vector similarity, metadata, and final-context survival. Citation IDs are extracted from `[ID:n]` markers when quote mode provides them.

### Proof

The prior findings depend on distinguishing 78 raw misses, 7 raw-present/final-absent rows, and 13 wrong-answer/final-present rows. Persisting normalized audit fields makes that distinction first-class and reproducible.

### Validation

Unit-test design:

```diff
diff --git a/tests/test_benchmark_runner.py b/tests/test_benchmark_runner.py
@@
+def test_chunk_audit_record_preserves_scores_and_document_name():
+    from ragflow_bench.execution.benchmark_runner import _chunk_audit_record
+
+    record = _chunk_audit_record(
+        {"id": "c1", "document_keyword": "github_shard_000104.txt", "similarity": 0.7,
+         "term_similarity": 0.8, "vector_similarity": 0.6},
+        rank=3,
+        survived_final_context=False,
+    )
+    assert record["rank"] == 3
+    assert record["document_name"] == "github_shard_000104.txt"
+    assert record["term_similarity"] == 0.8
+    assert record["survived_final_context"] is False
```

Artifact validation: recompute on June 26 artifacts and assert `raw_retrieval_chunks[0].document_name` matches existing `raw_retrieval.chunks[0].document_keyword`, and final context count remains 8 for scorable rows.

## 6. Proposed Diff 3: Failure Classification Rewrite

**Labels:** evaluation fix; diagnostic improvement.  
**Target functions/classes:** `classify_failure`, `build_summary`, future recompute command.

### Proposed unified diff

```diff
diff --git a/src/ragflow_bench/scoring/failure_classification.py b/src/ragflow_bench/scoring/failure_classification.py
index bfd8a66..9ac03e8 100644
--- a/src/ragflow_bench/scoring/failure_classification.py
+++ b/src/ragflow_bench/scoring/failure_classification.py
@@ -1,15 +1,70 @@
 from __future__ import annotations
+
+
+NOT_FOUND_MARKERS = ("not found", "no relevant", "unable to find", "does not contain", "not available")
 
 
-def classify_failure(*, error: str | None, ragflow_answer: str | None, exact_match: bool, source_recall: float) -> str:
+def classify_failure(
+    *,
+    error: str | None,
+    ragflow_answer: str | None,
+    exact_match: bool,
+    source_recall: float,
+    raw_retrieval_shard_recall: float | None = None,
+    final_context_shard_recall: float | None = None,
+    cited_shard_recall: float | None = None,
+    judge_verdict: str | None = None,
+) -> str:
     if error:
-        return "error"
+        return "answer_generation_error"
     if not ragflow_answer:
-        return "empty_response"
+        return "answer_generation_error"
     if exact_match:
-        return "correct"
-    if source_recall == 0.0:
-        return "retrieval_failure"
-    if ragflow_answer and not ragflow_answer.strip():
-        return "format_failure"
-    return "reasoning_failure"
+        return "exact_match_correct"
+    if judge_verdict == "judge_error":
+        return "judge_error"
+
+    raw_recall = raw_retrieval_shard_recall if raw_retrieval_shard_recall is not None else source_recall
+    final_recall = final_context_shard_recall if final_context_shard_recall is not None else source_recall
+    cited_recall = cited_shard_recall if cited_shard_recall is not None else 0.0
+    answer_lower = ragflow_answer.lower()
+
+    if raw_recall <= 0.0:
+        if any(marker in answer_lower for marker in NOT_FOUND_MARKERS):
+            return "not_found_false_negative"
+        return "expected_source_absent_from_raw_retrieval"
+    if final_recall <= 0.0:
+        return "expected_source_present_raw_absent_final_context"
+    if judge_verdict == "correct":
+        return "judge_correct_but_source_unverified" if cited_recall <= 0.0 else "exact_match_correct"
+    if judge_verdict == "partial":
+        return "expected_source_present_final_context_answer_partial"
+    if judge_verdict == "incorrect":
+        if any(marker in answer_lower for marker in NOT_FOUND_MARKERS):
+            return "not_found_false_negative"
+        return "expected_source_present_final_context_answer_wrong"
+    return "unknown_unclassified"
+
+
+def classify_judged_row(row: dict) -> str:
+    if row.get("judge_exclusion_reason") == "judge_error":
+        return "judge_error"
+    return classify_failure(
+        error=row.get("error"),
+        ragflow_answer=row.get("ragflow_answer"),
+        exact_match=bool(row.get("exact_match")),
+        source_recall=float(row.get("source_recall") or 0.0),
+        raw_retrieval_shard_recall=float(row.get("raw_retrieval_shard_recall@20") or 0.0),
+        final_context_shard_recall=float(row.get("final_context_shard_recall@top_n") or 0.0),
+        cited_shard_recall=float(row.get("cited_shard_recall") or 0.0),
+        judge_verdict=row.get("judge_verdict"),
+    )
```

### Logic

- `answer_generation_error`: runner/chat/API failure or empty answer.
- `judge_error`: judge failed independently of answer generation.
- `expected_source_absent_from_raw_retrieval`: raw retrieval did not surface expected shard.
- `expected_source_present_raw_absent_final_context`: broad retrieval found expected shard, final context lost it.
- `expected_source_present_final_context_answer_wrong`: evidence present, answer wrong.
- `expected_source_present_final_context_answer_partial`: evidence present, incomplete answer.
- `not_found_false_negative`: model claims absence while raw/final evidence says otherwise, or no raw source but answer is an unsupported not-found.
- `possible_distractor_answer`: optional later refinement when cited or top-ranked non-expected source is strong while expected source is absent.
- `exact_match_correct`: exact match or trusted correct with citation recall.
- `judge_correct_but_source_unverified`: answer correct but source/citation path unproven.
- `unknown_unclassified`: fallback.

### Proof

Old `retrieval_failure` collapsed 420 completed rows because `source_recall=0.0` was itself broken. That obscured the observed buckets: 78/96 incorrect raw misses, 7 raw-present/final-absent, 13 final-present/wrong, and 25 final-present/partial.

### Validation

```diff
diff --git a/tests/test_scoring.py b/tests/test_scoring.py
@@
 def test_failure_classification():
-    assert classify_failure(error="boom", ragflow_answer=None, exact_match=False, source_recall=0.0) == "error"
+    assert classify_failure(error="boom", ragflow_answer=None, exact_match=False, source_recall=0.0) == "answer_generation_error"
+    assert classify_failure(error=None, ragflow_answer="x", exact_match=False, source_recall=0.0, raw_retrieval_shard_recall=0.0, final_context_shard_recall=0.0, judge_verdict="incorrect") == "expected_source_absent_from_raw_retrieval"
+    assert classify_failure(error=None, ragflow_answer="x", exact_match=False, source_recall=0.0, raw_retrieval_shard_recall=1.0, final_context_shard_recall=0.0, judge_verdict="incorrect") == "expected_source_present_raw_absent_final_context"
+    assert classify_failure(error=None, ragflow_answer="wrong", exact_match=False, source_recall=0.0, raw_retrieval_shard_recall=1.0, final_context_shard_recall=1.0, judge_verdict="incorrect") == "expected_source_present_final_context_answer_wrong"
+    assert classify_failure(error=None, ragflow_answer="some facts", exact_match=False, source_recall=0.0, raw_retrieval_shard_recall=1.0, final_context_shard_recall=1.0, judge_verdict="partial") == "expected_source_present_final_context_answer_partial"
+    assert classify_failure(error=None, ragflow_answer="not found", exact_match=False, source_recall=0.0, raw_retrieval_shard_recall=1.0, final_context_shard_recall=1.0, judge_verdict="incorrect") == "not_found_false_negative"
```

## 7. Proposed Diff 4: Historical Artifact Recompute Command

**Labels:** evaluation fix; diagnostic improvement.  
**Target functions/classes:** new `reports/diagnostics.py`, new `cli.py` command.

### Proposed unified diff

```diff
diff --git a/src/ragflow_bench/reports/diagnostics.py b/src/ragflow_bench/reports/diagnostics.py
new file mode 100644
index 0000000..5d2aeee
--- /dev/null
+++ b/src/ragflow_bench/reports/diagnostics.py
@@ -0,0 +1,118 @@
+from __future__ import annotations
+
+import json
+from collections import Counter, defaultdict
+from pathlib import Path
+from typing import Any
+
+from ragflow_bench.scoring.failure_classification import classify_judged_row
+from ragflow_bench.scoring.retrieval_scoring import retrieval_diagnostics
+
+
+def _load_jsonl(path: Path) -> list[dict[str, Any]]:
+    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
+
+
+def _source_types(row: dict[str, Any]) -> list[str]:
+    values: list[str] = []
+    for source in row.get("expected_sources") or []:
+        text = str(source)
+        if text.startswith("eragb-shard://"):
+            values.append(text.split("//", 1)[1].split("/", 1)[0])
+    return sorted(set(values)) or ["unknown"]
+
+
+def recompute_row_diagnostics(row: dict[str, Any]) -> dict[str, Any]:
+    raw_chunks = (row.get("raw_retrieval") or {}).get("chunks") or []
+    final_chunks = ((row.get("raw_response") or {}).get("reference") or {}).get("chunks") or []
+    raw = retrieval_diagnostics(row.get("expected_sources") or [], raw_chunks, prefix="raw_retrieval")
+    final = retrieval_diagnostics(row.get("expected_sources") or [], final_chunks, prefix="final_context")
+    out = dict(row)
+    out.update({
+        "raw_retrieval_shard_recall@20": raw["raw_retrieval_shard_recall"],
+        "raw_retrieval_expected_rank": raw["raw_retrieval_expected_rank"],
+        "raw_retrieval_mrr": raw["raw_retrieval_mrr"],
+        "final_context_shard_recall@top_n": final["final_context_shard_recall"],
+        "final_context_expected_rank": final["final_context_expected_rank"],
+        "final_context_mrr": final["final_context_mrr"],
+    })
+    out["failure_type_recomputed"] = classify_judged_row(out)
+    return out
+
+
+def build_diagnostic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
+    scorable = [row for row in rows if not row.get("judge_excluded") and isinstance(row.get("judge_score"), int)]
+    failure_counts = Counter(row.get("failure_type_recomputed") for row in rows)
+    source_breakdown: dict[str, Counter] = defaultdict(Counter)
+    reasoning_breakdown: dict[str, Counter] = defaultdict(Counter)
+    representatives: dict[str, list[str]] = defaultdict(list)
+    for row in rows:
+        bucket = row.get("failure_type_recomputed") or "unknown_unclassified"
+        if len(representatives[bucket]) < 10:
+            representatives[bucket].append(str(row.get("question_id")))
+        for source_type in _source_types(row):
+            source_breakdown[source_type][bucket] += 1
+            source_breakdown[source_type]["count"] += 1
+        for reason in row.get("reasoning_types") or ["unknown"]:
+            reasoning_breakdown[str(reason)][bucket] += 1
+            reasoning_breakdown[str(reason)]["count"] += 1
+    return {
+        "total_rows": len(rows),
+        "scorable_rows": len(scorable),
+        "raw_retrieval_shard_recall@20": sum(float(r.get("raw_retrieval_shard_recall@20") or 0.0) for r in scorable) / len(scorable) if scorable else 0.0,
+        "final_context_shard_recall@top_n": sum(float(r.get("final_context_shard_recall@top_n") or 0.0) for r in scorable) / len(scorable) if scorable else 0.0,
+        "raw_retrieval_mrr": sum(float(r.get("raw_retrieval_mrr") or 0.0) for r in scorable) / len(scorable) if scorable else 0.0,
+        "failure_bucket_counts": dict(failure_counts),
+        "source_type_breakdown": {k: dict(v) for k, v in source_breakdown.items()},
+        "reasoning_type_breakdown": {k: dict(v) for k, v in reasoning_breakdown.items()},
+        "representative_question_ids": dict(representatives),
+    }
+
+
+def recompute_run_diagnostics(run_dir: str | Path, *, output_prefix: str = "diagnostics") -> dict[str, Any]:
+    run_path = Path(run_dir)
+    results = _load_jsonl(run_path / "results.jsonl")
+    judge_path = run_path / "judge_results.jsonl"
+    rows = _load_jsonl(judge_path) if judge_path.exists() else results
+    recomputed = [recompute_row_diagnostics(row) for row in rows]
+    summary = build_diagnostic_summary(recomputed)
+    (run_path / f"{output_prefix}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
+    (run_path / f"{output_prefix}.rows.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in recomputed) + "\n", encoding="utf-8")
+    md = ["# Retrieval Diagnostics", "", "```json", json.dumps(summary, indent=2, ensure_ascii=False), "```"]
+    (run_path / f"{output_prefix}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
+    return summary
diff --git a/src/ragflow_bench/cli.py b/src/ragflow_bench/cli.py
index 3f613a7..fd837f4 100644
--- a/src/ragflow_bench/cli.py
+++ b/src/ragflow_bench/cli.py
@@ -22,6 +22,7 @@ from ragflow_bench.ragflow.errors import RagflowAPIError, RagflowConfigError
 from ragflow_bench.reports.summary import build_summary
+from ragflow_bench.reports.diagnostics import recompute_run_diagnostics
 from ragflow_bench.reports.writers import jsonl_to_csv, load_jsonl, write_json, write_jsonl
@@
+@app.command("recompute-diagnostics")
+def recompute_diagnostics(
+    run_dir: str = typer.Argument(..., help="Existing benchmark output directory"),
+    output_prefix: str = typer.Option("diagnostics", help="Output basename inside run_dir"),
+) -> None:
+    """Recompute retrieval/context/failure diagnostics from saved artifacts only."""
+    summary = recompute_run_diagnostics(run_dir, output_prefix=output_prefix)
+    console.print_json(json.dumps(summary, ensure_ascii=False))
+
```

### Usage example

```bash
uv run ragflow-bench recompute-diagnostics \
  outputs/enterprise_rag_bench_full_20260626_101748 \
  --output-prefix eragb_june26_diagnostics
```

Outputs:

- `eragb_june26_diagnostics.json`
- `eragb_june26_diagnostics.md`
- `eragb_june26_diagnostics.rows.jsonl`

### Proof

The prior reports were produced by ad-hoc scans. A stable command is needed to recompute diagnostics from the already-paid June 26 artifacts without rerunning RAGFlow or the judge.

### Validation

- Unit: build two synthetic rows and assert bucket counts, source-type counts, and representatives.
- Artifact: run against the June 26 directory and assert 203 scorable rows, 97 raw hits, 90 final hits, 78 incorrect raw misses, and 7 raw-present/final-absent.
- Manual invariant: command must not call `RagflowClient`, `run_benchmark`, or judge APIs.

## 8. Proposed Diff 5: Source-Boundary-Aware ERAGB Ingestion

**Labels:** ingestion fix + retrieval fix; real improvement.  
**Reingestion required:** yes.

### Proposed unified diff

```diff
diff --git a/src/ragflow_bench/benchmarks/eragb_prep.py b/src/ragflow_bench/benchmarks/eragb_prep.py
index 40f199d..737a5bb 100644
--- a/src/ragflow_bench/benchmarks/eragb_prep.py
+++ b/src/ragflow_bench/benchmarks/eragb_prep.py
@@ -13,6 +13,7 @@ ERAGB_DOC_BOUNDARY = "<<<ERAGB_DOC_BOUNDARY>>>"
 ERAGB_PARSER_DELIMITER = f"`{ERAGB_DOC_BOUNDARY}`"
 ReferenceGranularity = Literal["document", "shard", "none"]
 REQUIRED_REPO_PATHS = (DOCUMENTS_REPO_PATH, QUESTIONS_REPO_PATH)
+SOURCE_METADATA_KEYS = ("space", "path", "channel", "thread_ts", "message_ts", "user", "speaker", "meeting_date", "turn_index")
@@
-def _document_rows(df: pd.DataFrame) -> list[dict[str, str]]:
-    rows: list[dict[str, str]] = []
+def _document_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
+    rows: list[dict[str, Any]] = []
     for idx, row in enumerate(df.to_dict(orient="records")):
         doc_id = str(row.get("doc_id") or row.get("id") or idx)
+        metadata = _source_metadata(row)
         rows.append(
             {
                 "doc_id": doc_id,
                 "source_type": str(row.get("source_type") or "document"),
                 "title": str(row.get("title") or doc_id),
                 "content": str(row.get("content") or ""),
+                "metadata": metadata,
             }
         )
     return rows
+
+
+def _source_metadata(row: dict[str, Any]) -> dict[str, Any]:
+    metadata: dict[str, Any] = {}
+    raw_metadata = row.get("metadata") or row.get("meta") or {}
+    if isinstance(raw_metadata, dict):
+        metadata.update({str(k): _jsonable(v) for k, v in raw_metadata.items() if not _is_missing_value(v)})
+    for key in SOURCE_METADATA_KEYS:
+        if key in row and not _is_missing_value(row[key]):
+            metadata[key] = _jsonable(row[key])
+    source_type = str(row.get("source_type") or "")
+    if source_type == "confluence":
+        for key in ("page_title", "space", "path", "heading_hierarchy"):
+            if key in row and not _is_missing_value(row[key]):
+                metadata[key] = _jsonable(row[key])
+    elif source_type == "slack":
+        for key in ("channel", "thread_ts", "message_ts", "speaker", "user"):
+            if key in row and not _is_missing_value(row[key]):
+                metadata[key] = _jsonable(row[key])
+    elif source_type == "fireflies":
+        for key in ("meeting_title", "date", "speaker", "turn_index"):
+            if key in row and not _is_missing_value(row[key]):
+                metadata[key] = _jsonable(row[key])
+    return metadata
@@
-        manifest[relative_path] = {
+        manifest[relative_path] = {
             "id": doc_id,
             "source_uri": doc_id,
             "title": title,
             "source_type": source_type,
             "path": relative_path,
+            "metadata": row.get("metadata") or {},
         }
@@
-        shard_manifest[relative_path] = {
+        shard_manifest[relative_path] = {
             **manifest[relative_path],
             "contained_doc_ids": list(current_doc_ids),
+            "contained_documents": current_doc_records,
         }
@@
-    current_doc_ids: list[str] = []
+    current_doc_ids: list[str] = []
+    current_doc_records: list[dict[str, Any]] = []
@@
-        current_doc_ids = []
+        current_doc_ids = []
+        current_doc_records = []
@@
         current_blocks.append(block)
         current_doc_ids.append(doc_id)
+        current_doc_records.append({
+            "doc_id": doc_id,
+            "source_type": source_type,
+            "title": row.get("title"),
+            "canonical_shard_uri": f"eragb-shard://{_safe_path_component(source_type)}/{_safe_path_component(source_type)}_shard_{shard_index_by_source.get(source_type, 0) + 1:06d}.txt",
+            "metadata": row.get("metadata") or {},
+        })
@@
-def _document_text(*, doc_id: str, source_type: str, title: str, content: str) -> str:
-    return f"Title: {title}\nDocument ID: {doc_id}\nSource Type: {source_type}\n\n{content.strip()}\n"
+def _document_text(*, doc_id: str, source_type: str, title: str, content: str, metadata: dict[str, Any] | None = None) -> str:
+    metadata_lines = _metadata_header(metadata or {})
+    return f"Title: {title}\nDocument ID: {doc_id}\nSource Type: {source_type}{metadata_lines}\n\n{content.strip()}\n"
@@
-def _merged_document_block(*, doc_id: str, source_type: str, title: str, content: str) -> str:
-    return f"{ERAGB_DOC_BOUNDARY}\nDocument ID: {doc_id}\nSource Type: {source_type}\nTitle: {title}\nContent:\n{content.strip()}\n"
+def _merged_document_block(*, doc_id: str, source_type: str, title: str, content: str, metadata: dict[str, Any] | None = None) -> str:
+    metadata_lines = _metadata_header(metadata or {})
+    return f"{ERAGB_DOC_BOUNDARY}\nDocument ID: {doc_id}\nSource Type: {source_type}\nTitle: {title}{metadata_lines}\nContent:\n{content.strip()}\n"
+
+
+def _metadata_header(metadata: dict[str, Any]) -> str:
+    if not metadata:
+        return ""
+    safe = {k: v for k, v in metadata.items() if k in SOURCE_METADATA_KEYS or k in {"page_title", "heading_hierarchy", "meeting_title"}}
+    if not safe:
+        return ""
+    return "\nMetadata: " + json.dumps(safe, ensure_ascii=False, sort_keys=True)
```

Note: the final implementation should fix the closure bookkeeping carefully; the diff shows the intended shape. In particular, `current_doc_records` must be declared `nonlocal` inside `flush()`, and canonical shard URI should be assigned after the shard id is finalized. This is why this is a design report, not an applied patch.

### Proof

The 78/96 incorrect raw-retrieval-miss bucket means ranking often fails before final context. Merged source-type shards plus naive parser and blank-line delimiter are likely losing original document boundaries. RAGFlow supports metadata enrichment in chat (`dialog_service.py` has reference metadata enrichment helpers), and the benchmark already has `patch_document_metadata()` in `ingest.py`.

RAGFlow child chunking docs support preserving parent context while making smaller child chunks retrievable, and Azure chunking guidance supports chunking by document shape/density rather than blind size increases.

### Validation

```diff
diff --git a/tests/test_eragb_prep.py b/tests/test_eragb_prep.py
@@
+def test_prepare_eragb_preserves_source_specific_metadata(tmp_path, monkeypatch):
+    # Build a tiny documents dataframe with confluence/slack/fireflies metadata.
+    # Assert documents_manifest.json and shard_manifest.json keep doc_id,
+    # canonical shard URI, source_type, title, and source-specific metadata.
+    ...
```

Diagnostic slice: prepare a 50-document Confluence/Slack/Fireflies subset, reingest into a new dataset, run 20 fixed questions, and compare raw retrieval shard recall@20 at the same `top_k`, `page_size`, and `top_n`. This targets the raw-retrieval-miss bucket and requires reingestion.

## 9. Proposed Diff 6: Fixed-Budget Reranker / Context Quality Experiment Support

**Labels:** context-selection fix + retrieval diagnostic; real improvement if fixed-budget rank/recall improves, potential score inflation if only top-N grows.  
**Reingestion required:** no, if the RAGFlow tenant has a reranker model configured.

### Proposed unified diff

```diff
diff --git a/src/ragflow_bench/config.py b/src/ragflow_bench/config.py
index 648f1d9..54f910c 100644
--- a/src/ragflow_bench/config.py
+++ b/src/ragflow_bench/config.py
@@ -27,6 +27,7 @@ class RetrievalSettings(BaseModel):
     page_size: int = 20
     similarity_threshold: float = 0.05
     vector_similarity_weight: float = 0.3
+    rerank_id: str | None = None
@@ -37,6 +38,8 @@ class ChatSettings(BaseModel):
     quote: bool = True
     refine_multiturn: bool = False
+    fixed_budget_experiment: bool = True
+    prompt_mode: str = "default"
diff --git a/src/ragflow_bench/ragflow/client.py b/src/ragflow_bench/ragflow/client.py
index aebd313..a81b16a 100644
--- a/src/ragflow_bench/ragflow/client.py
+++ b/src/ragflow_bench/ragflow/client.py
@@ -145,7 +145,7 @@ class RagflowClient:
-    def create_chat(self, *, name: str, dataset_ids: list[str], llm_id: str, prompt_config: dict[str, Any] | None = None, top_n: int = 8, top_k: int = 128, similarity_threshold: float = 0.05, vector_similarity_weight: float = 0.3) -> dict[str, Any]:
+    def create_chat(self, *, name: str, dataset_ids: list[str], llm_id: str, prompt_config: dict[str, Any] | None = None, top_n: int = 8, top_k: int = 128, similarity_threshold: float = 0.05, vector_similarity_weight: float = 0.3, rerank_id: str | None = None) -> dict[str, Any]:
@@
         if prompt_config is not None:
             body["prompt_config"] = prompt_config
+        if rerank_id:
+            body["rerank_id"] = rerank_id
         payload = self._request("POST", "/api/v1/chats", json=body)
@@ -164,7 +166,7 @@ class RagflowClient:
-    def retrieve(self, *, question: str, dataset_ids: list[str] | None = None, document_ids: list[str] | None = None, page_size: int = 20, similarity_threshold: float = 0.05, vector_similarity_weight: float = 0.3, top_k: int = 128) -> dict[str, Any]:
+    def retrieve(self, *, question: str, dataset_ids: list[str] | None = None, document_ids: list[str] | None = None, page_size: int = 20, similarity_threshold: float = 0.05, vector_similarity_weight: float = 0.3, top_k: int = 128, rerank_id: str | None = None) -> dict[str, Any]:
@@
         if document_ids:
             body["document_ids"] = document_ids
+        if rerank_id:
+            body["rerank_id"] = rerank_id
         payload = self._request("POST", "/api/v1/retrieval", json=body)
diff --git a/src/ragflow_bench/execution/chat_runner.py b/src/ragflow_bench/execution/chat_runner.py
index 1a6d44e..b29a44d 100644
--- a/src/ragflow_bench/execution/chat_runner.py
+++ b/src/ragflow_bench/execution/chat_runner.py
@@ -18,6 +18,7 @@ def ensure_chat(client: RagflowClient, config: AppConfig, dataset_id: str, name:
         top_k=config.retrieval.top_k,
         similarity_threshold=config.retrieval.similarity_threshold,
         vector_similarity_weight=config.retrieval.vector_similarity_weight,
+        rerank_id=config.retrieval.rerank_id,
     )
diff --git a/src/ragflow_bench/execution/retrieval_runner.py b/src/ragflow_bench/execution/retrieval_runner.py
index 6a052e5..1cae654 100644
--- a/src/ragflow_bench/execution/retrieval_runner.py
+++ b/src/ragflow_bench/execution/retrieval_runner.py
@@ -14,4 +14,5 @@ def run_retrieval(client: RagflowClient, config: AppConfig, dataset_id: str, que
         vector_similarity_weight=config.retrieval.vector_similarity_weight,
         top_k=config.retrieval.top_k,
+        rerank_id=config.retrieval.rerank_id,
     )
```

### Proof

RAGFlow `dataset_api_service.search()` reads `rerank_id` from request/search config and constructs `rerank_mdl`; `rag/nlp/search.py` calls `rerank_by_model()` when `rerank_mdl` is present. The current harness has no `rerank_id` field, so fixed-budget reranker experiments cannot be configured.

### Validation

- Unit: fake `RagflowClient._request` and assert `rerank_id` appears in `/api/v1/retrieval` and `/api/v1/chats` request bodies.
- Config: assert `config.resolved.yaml` persists `retrieval.rerank_id`.
- Experiment: run a 20-question diagnostic slice at `top_n=8`, `max_tokens=128`, `top_k=128`, `page_size=20` with and without reranker. Success means raw/final expected rank or MRR improves at fixed budgets.

## 10. Proposed Diff 7: Exact-Fact Extraction Prompt Variant

**Labels:** synthesis fix; real improvement if final-context-present rows improve at fixed context.  
**Reingestion required:** no.

### Proposed unified diff

```diff
diff --git a/src/ragflow_bench/execution/chat_runner.py b/src/ragflow_bench/execution/chat_runner.py
index b29a44d..964bde3 100644
--- a/src/ragflow_bench/execution/chat_runner.py
+++ b/src/ragflow_bench/execution/chat_runner.py
@@ -5,6 +5,24 @@ from ragflow_bench.config import AppConfig
 from ragflow_bench.ragflow.client import RagflowClient
+
+
+EXACT_FACT_SYSTEM_PROMPT = """You are an exact-fact enterprise QA assistant.
+Use only the provided knowledge.
+Before answering, identify the exact evidence snippets for requested numbers,
+dates, metric names, thresholds, role lists, acceptance criteria, identifiers,
+and named entities. If multiple values conflict, state the conflict and choose
+the value from the most directly relevant cited snippet. If exact evidence is
+absent, answer that the exact value is not found in the provided knowledge.
+Return a concise answer with citations. Do not add background explanation."""
+
+
+def _prompt_config(config: AppConfig) -> dict:
+    payload = {"quote": config.chat.quote, "refine_multiturn": config.chat.refine_multiturn}
+    if config.chat.prompt_mode == "exact_fact":
+        payload.update({
+            "system": EXACT_FACT_SYSTEM_PROMPT + "\n\n{knowledge}",
+            "empty_response": "The exact answer is not found in the provided knowledge.",
+        })
+    return payload
@@
-        prompt_config={"quote": config.chat.quote, "refine_multiturn": config.chat.refine_multiturn},
+        prompt_config=_prompt_config(config),
```

### Proof

The June 26 artifacts show 13 incorrect rows and 25 partial rows where the expected shard was already in final context. These rows are not solved by raw retrieval alone; they need better exact extraction, conflict handling, and concise answer synthesis.

### Validation

- Unit: assert `ensure_chat()` sends `prompt_config.system` when `chat.prompt_mode=exact_fact` and does not change `top_n` or `max_tokens`.
- Slice: run only final-context-present wrong/partial representatives, e.g. `qst_0020`, `qst_0021`, `qst_0023`, `qst_0097`, `qst_0102`, `qst_0195`, at fixed `top_n=8` and `max_tokens=128`.
- Pass condition: strict or partial score improves on this slice without raw/final context recall changing. That proves synthesis improvement, not retrieval inflation.

## 11. Proposed Diff 8: Judge/Infra Stabilization

**Labels:** infra fix + evaluation fix; diagnostic improvement.  
**Reingestion required:** no.

### Proposed unified diff

```diff
diff --git a/src/ragflow_bench/judge.py b/src/ragflow_bench/judge.py
index 5a96a91..d32c118 100644
--- a/src/ragflow_bench/judge.py
+++ b/src/ragflow_bench/judge.py
@@ -156,15 +156,16 @@ class ZhipuJudgeClient:
             else:
                 return min(max(delay, 0.0), self.max_backoff_seconds)
-        return min(self.backoff_seconds, self.max_backoff_seconds)
+        return min(self.backoff_seconds * (2 ** max(0, attempt - 1)), self.max_backoff_seconds)
@@
-            except requests.ReadTimeout as exc:
+            except (requests.ReadTimeout, requests.ConnectionError) as exc:
                 elapsed = time.monotonic() - started
                 self._emit_log({
@@
-                    "retry": False,
+                    "retry": True,
@@
                 })
                 raise
@@
         response = run_with_rate_limit_retries(
             _request_once,
             action_type="judge",
             question_id=question_id,
             progress_callback=self.progress_callback,
+            max_retries=self.max_retries,
+            base_delay_seconds=self.backoff_seconds,
+            max_delay_seconds=self.max_backoff_seconds,
         )
         return response
@@
-            result = self.judge_row(row)
+            result = self.judge_row(row)
+            judged_payload = {**row, "raw_judge_response": result.get("raw_response")}
```

If `run_with_rate_limit_retries()` currently does not accept explicit retry parameters, add them there rather than hard-coding judge behavior. The design intent is exponential backoff and persistent raw response capture, not a second retry system.

### Proof

297/500 rows were excluded, including 198 judge-error rows. That makes benchmark quality conclusions selection-biased. This improves evaluation trustworthiness only; it is not a retrieval-quality improvement.

### Validation

```diff
diff --git a/tests/test_judge.py b/tests/test_judge.py
@@
-def test_zhipu_judge_does_not_retry_read_timeout(monkeypatch):
+def test_zhipu_judge_retries_read_timeout(monkeypatch):
+    # First call raises ReadTimeout, second returns valid JSON.
+    # Assert two calls and a non-excluded judged result.
+    ...
@@
 def test_zhipu_judge_retries_429_then_succeeds(monkeypatch):
     ...
+
+def test_judge_summary_reports_exclusion_bias_by_source_type():
+    # Synthetic rows with source_types/reasoning_types and excluded rows.
+    # Assert excluded_by_source_type and excluded_by_reasoning_type counts.
+    ...
```

Also add summary fields:

```diff
diff --git a/src/ragflow_bench/judge.py b/src/ragflow_bench/judge.py
@@
 def summarize_judged_rows(*, rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
@@
+    excluded_by_source_type = Counter()
+    excluded_by_reasoning_type = Counter()
+    for row in rows:
+        if not row.get("judge_excluded"):
+            continue
+        for source in row.get("expected_sources") or []:
+            if str(source).startswith("eragb-shard://"):
+                excluded_by_source_type[str(source).split("//", 1)[1].split("/", 1)[0]] += 1
+        for reason in row.get("reasoning_types") or []:
+            excluded_by_reasoning_type[str(reason)] += 1
@@
         "average_judge_confidence": (sum(confidences) / len(confidences) if confidences else None),
+        "excluded_by_source_type": dict(excluded_by_source_type),
+        "excluded_by_reasoning_type": dict(excluded_by_reasoning_type),
     }
```

## 12. RAGFlow Core Code Patch Candidates

Benchmark-side fixes should come first. RAGFlow core patches are higher risk because they affect product behavior outside this benchmark.

### Candidate A: expose prompt-packing diagnostics in chat reference

**Label:** speculative RAGFlow core patch; context-selection diagnostic; low/medium risk.  
**Grounded code:** `/home/idriss/commits/ragflow/api/db/services/dialog_service.py` around `kb_prompt(kbinfos, max_tokens)` and yielded final reference.

```diff
diff --git a/api/db/services/dialog_service.py b/api/db/services/dialog_service.py
index 5b4a19b..8f0aeda 100644
--- a/api/db/services/dialog_service.py
+++ b/api/db/services/dialog_service.py
@@ -755,7 +755,15 @@ async def chat(...):
-    knowledges = kb_prompt(kbinfos, max_tokens)
+    pre_prompt_chunk_count = len(kbinfos.get("chunks", []))
+    knowledges = kb_prompt(kbinfos, max_tokens)
+    prompt_chunk_count = len(knowledges)
+    kbinfos["prompt_packing"] = {
+        "pre_prompt_chunk_count": pre_prompt_chunk_count,
+        "prompt_chunk_count": prompt_chunk_count,
+        "dropped_chunk_count": max(0, pre_prompt_chunk_count - prompt_chunk_count),
+        "max_tokens": max_tokens,
+    }
```

**Proof:** RAGFlow currently truncates knowledge in `rag/prompts/generator.py::kb_prompt()` when token budget is exceeded, but the benchmark has to infer whether chunks survived final packing.  
**Validation:** chat API response with `quote=true` includes `reference.prompt_packing`; no answer text changes.

### Candidate B: RRF/rank-fusion experiment in search

**Label:** speculative RAGFlow core patch; retrieval/context-selection fix; high risk.  
**Grounded code:** `/home/idriss/commits/ragflow/rag/nlp/search.py::retrieval()` combines scores by magnitude. RRF could avoid score-scale mismatch.

Do **not** implement yet. Elasticsearch RRF docs support the concept, but the local bottleneck must first be measured with fixed-budget reranker and shard recall diagnostics.

### Candidate C: metadata propagation into chunks

**Label:** speculative RAGFlow core patch; ingestion/retrieval diagnostic; medium risk.  
**Grounded code:** `dialog_service.py` already imports reference metadata utilities and can enrich chunks with `document_metadata`; benchmark ingestion already calls `patch_document_metadata()`.  
**Recommendation:** prefer benchmark-side metadata and `reference_metadata` request/config first. RAGFlow core patch only if metadata is not reliably exposed in retrieval or chat references.

## 13. Patch Backlog Table

| Patch ID | Target files | Failure mode addressed | Proof from artifacts/code | Expected strict-accuracy impact | Complexity | Reingestion | Risk | Validation | Priority |
|---|---|---|---|---:|---|---|---|---|---:|
| D1 | `retrieval_scoring.py`, `benchmark_runner.py` | broken `source_recall=0.0` | expected shard URIs vs chunk `document_keyword`/`document_name` | none direct | medium | no | low | unit + recompute | P0 |
| D2 | `benchmark_runner.py` | retrieval vs final-context ambiguity | 7 raw-present/final-absent rows | none direct | medium | no | low | row audit tests | P0 |
| D4 | `reports/diagnostics.py`, `cli.py` | no stable artifact recompute | prior ad-hoc scans | none direct | medium | no | low | June 26 recompute | P0 |
| D3 | `failure_classification.py`, summary | collapsed `retrieval_failure` | 420 retrieval failures in summary | none direct | low | no | low | classification tests | P0 |
| D8 | `judge.py` | 297 excluded rows | 198 judge errors, 99 infra errors | indirect | medium | no | medium | retry tests + resume | P1 |
| D5 | `eragb_prep.py`, ingestion metadata | raw retrieval misses | 78/96 incorrect raw misses; weak Confluence/Slack/Fireflies | medium/high if recall improves | high | yes | medium | small reingest slice | P1 |
| D6 | config/client/runners | fixed-budget reranker tests impossible | RAGFlow accepts `rerank_id`; harness lacks it | medium if MRR improves | low/medium | no | low | request-body tests + slice | P2 |
| D7 | `chat_runner.py`, config | final-context-present wrong/partial | 13 wrong + 25 partial final-present rows | medium on that slice | low | no | low | exact-fact slice | P2 |
| RC-A | RAGFlow `dialog_service.py` | prompt packing opaque | `kb_prompt` truncates silently except log | none direct | low | no | medium | API response invariant | P3 |
| RC-B | RAGFlow `search.py` | score-scale/ranking | local hybrid score path; RRF docs | unknown | high | no | high | offline fixed-query eval | P4 |

## 14. Tests and Validation Plan

For every proposed diff:

- **D1:** unit tests for shard URI canonicalization; artifact recompute must show non-zero raw/final shard recall on June 26 rows.
- **D2:** unit tests for chunk audit records and citation extraction; artifact recompute must preserve ranks and scores from saved payloads.
- **D3:** at least five representative classification tests: raw absent, raw present/final absent, final present wrong, final present partial, not-found false negative.
- **D4:** unit tests for read-only recompute; assert no RAGFlow/judge client calls; June 26 recompute must reproduce known counts.
- **D5:** ERAGB prep tests for metadata and sidecars; small source-type-specific reingest slice; metric to improve is raw retrieval shard recall@20, not only answer score.
- **D6:** request-body tests for `rerank_id`; fixed-budget 20-question experiment; metrics: raw/final MRR, expected rank, final-context recall. `top_n` and `max_tokens` must not change.
- **D7:** prompt-config tests; fixed final-context-present wrong/partial slice; metrics: strict/partial answer score on targeted rows, no context-budget increase.
- **D8:** judge retry tests for 429/read timeout/5xx and no retry for 401; resume tests; exclusion-bias summary tests.
- **RAGFlow core candidates:** only after benchmark harness proves need; add API-level tests and no-answer-change invariants for diagnostic-only fields.

## 15. What Must Not Be Implemented Yet

- `top_n` inflation as a primary fix.
- `max_tokens` inflation as a primary fix.
- Random LLM/model/embedding swaps.
- GraphRAG/RAPTOR for this exact-fact ERAGB work.
- A new full benchmark before diagnostics are fixed and recomputed.
- RAGFlow core search changes before benchmark-side evidence still points there.
- Reranker provider comparisons without fixed candidate and fixed final-context budgets.

## 16. Final Recommended Patch Sequence

1. **Harness diagnostics and source mapping:** implement D1 and D2.
2. **Historical artifact recompute:** implement D4 and recompute June 26 diagnostics.
3. **Failure classification rewrite:** implement D3 after D1/D2 fields exist.
4. **Judge/retry stabilization:** implement D8, then retry judge-error rows only.
5. **Source-boundary-aware ingestion slice:** implement D5, reingest a small Confluence/Slack/Fireflies slice, validate raw recall@20.
6. **Fixed-budget reranker/context experiment support:** implement D6, run small fixed-budget reranker slice.
7. **Exact-fact prompt variant:** implement D7, run final-context-present wrong/partial validation slice.
8. **RAGFlow core patches:** only if benchmark-side evidence still isolates a RAGFlow search or prompt-packing bottleneck.

