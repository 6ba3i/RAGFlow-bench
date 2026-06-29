# RAGFlow EnterpriseRAG-Bench Retrieval Quality Investigation

## 1. Executive Summary

The current result is genuinely underwhelming, but the headline `average_source_recall: 0.0` and the runner's `retrieval_failure` labels are not reliable evidence that retrieval always misses. The strongest confirmed bottleneck is a combination of **retrieval/context selection quality plus evaluation bookkeeping**, with answer budget and synthesis failures as secondary contributors.

Confirmed facts from the June 26 run:

- 500 total questions were attempted; only 203 were scorable by the judge.
- On those 203, strict accuracy was 67/203 = **33.0%**, partial-credit score average was **42.9%**, and partial-or-better was 107/203 = **52.7%**.
- 297/500 rows were excluded from judged accuracy: 99 infra-error rows and 198 judge-error rows.
- Every row has `source_recall = 0.0`, but the benchmark only computes recall from `retrieved_source_uris`, and this run's retrieval payloads populate `document_id` / `document_keyword`, not `source_uri`. Thus the metric is broken for this run.
- A diagnostic scan of the saved artifacts found the expected shard filename in the top-20 raw retrieval `document_keyword` for **97/203 scorable rows**: 53 correct, 26 partial, and 18 incorrect. It found the expected shard in the chat prompt titles for **90/203 scorable rows**: 52 correct, 25 partial, and 13 incorrect. This proves retrieval sometimes works and final context/answer synthesis still fails for some incorrect rows.

Opinion: this is fixable, but not by one blind parameter tweak. The next work should first repair diagnostics, then run small ablations. The highest-upside safe experiments are: increase `chat.top_n`, increase generation budget, test reranking, and test source-boundary-aware / section-aware chunking. Code-level work is likely needed for trustworthy source recall and possibly for better final-context packing, but the first quality improvements can be explored through parameter sweeps once diagnostics are trustworthy.

## 2. Evidence Reviewed

### Local benchmark artifacts

- `outputs/enterprise_rag_bench_full_20260626_101748/summary.json`
- `outputs/enterprise_rag_bench_full_20260626_101748/judge_summary.json`
- `outputs/enterprise_rag_bench_full_20260626_101748/results.jsonl`
- `outputs/enterprise_rag_bench_full_20260626_101748/judge_results.jsonl`
- `outputs/enterprise_rag_bench_full_20260626_101748/config.resolved.yaml`
- `outputs/enterprise_rag_bench_full_20260626_101748/document_registry.json`

### Local benchmark code reviewed

- `src/ragflow_bench/execution/benchmark_runner.py` — benchmark loop, retrieval/chat calls, result row writing, recall extraction.
- `src/ragflow_bench/execution/retrieval_runner.py` — retrieval request wrapper.
- `src/ragflow_bench/execution/chat_runner.py` — chat creation and completion request wrapper.
- `src/ragflow_bench/ragflow/client.py` — `/api/v1/retrieval`, `/api/v1/chats`, `/api/v1/chat/completions` request bodies.
- `src/ragflow_bench/scoring/retrieval_scoring.py` — source recall formula.
- `src/ragflow_bench/scoring/failure_classification.py` — failure labels.
- `src/ragflow_bench/benchmarks/enterprise_rag_bench.py` — question loading and expected-source fields.
- `src/ragflow_bench/benchmarks/eragb_prep.py` — document/shard expected source mapping.
- `src/ragflow_bench/judge.py` — judge messages, exclusion logic, judged summary.

### RAGFlow source code reviewed

- `/home/idriss/commits/ragflow/api/apps/services/dataset_api_service.py` — dataset search semantics and retrieval parameter forwarding.
- `/home/idriss/commits/ragflow/api/apps/restful_apis/dataset_api.py` — dataset search API shape.
- `/home/idriss/commits/ragflow/api/apps/restful_apis/chat_api.py` — chat completion API entry point and defaults.
- `/home/idriss/commits/ragflow/api/db/services/dialog_service.py` — chat retrieval, `top_n`, context packing, quote/citation handling, generation-token handling.
- `/home/idriss/commits/ragflow/rag/nlp/search.py` — retrieval candidate window, hybrid scoring, thresholding, reranking path, returned chunk fields.
- `/home/idriss/commits/ragflow/rag/app/naive.py` — naive parser behavior for text/markdown and parser config use.
- `/home/idriss/commits/ragflow/rag/nlp/__init__.py` — `naive_merge` and custom delimiter behavior.

### Online sources reviewed

- RAGFlow HTTP API reference: https://ragflow.io/docs/http_api_reference — retrieval and chat API parameter surface (`top_k`, `top_n`, `similarity_threshold`, `vector_similarity_weight`, `quote`, `rerank_id`).
- Azure AI Search hybrid search overview: https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview — hybrid vector + keyword retrieval, RRF, semantic ranker benefits for grounding.
- Azure AI Search chunking guidance: https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-chunk-documents — start around 512-token chunks with overlap; adapt to document shape/density.
- Elasticsearch RRF documentation: https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html — reciprocal rank fusion combines result sets without score calibration and has rank-window tradeoffs.
- Cohere reranking documentation: https://docs.cohere.com/docs/reranking-with-cohere — rerankers take candidate documents and select more relevant top-N results for the query.

## 3. Current Benchmark Result

From `judge_summary.json`:

| Metric | Value |
|---|---:|
| Total questions | 500 |
| Scorable questions | 203 |
| Excluded questions | 297 |
| Infra-error exclusions | 99 |
| Judge-error exclusions | 198 |
| Correct | 67 |
| Partial | 40 |
| Incorrect | 96 |
| Strict accuracy | 33.0% |
| Partial-credit accuracy | 42.9% |
| Partial-or-better accuracy | 52.7% |
| Average judge confidence | 0.971 |

From `summary.json`, the runner-level metrics are misleading for this run:

- `completed_questions = 420`, `errors = 80`.
- `average_source_recall = 0.0`.
- `failure_type_counts.retrieval_failure = 420`.

That last pair should be treated as a metric bug symptom, not a quality diagnosis. The scoring path labels every non-exact answer with zero source recall as `retrieval_failure`; because source recall is always zero, the label collapses all non-exact non-error answers into retrieval failure.

## 4. Actual Pipeline Reconstruction

Confirmed pipeline:

1. **Question loading**
   - `EnterpriseRAGBenchAdapter.load_questions()` loads `questions_path` as JSONL/JSON/CSV/parquet and maps `expected_sources` from the first available of `expected_sources`, `expected_doc_ids`, `expected_documents`, or `citations` (`src/ragflow_bench/benchmarks/enterprise_rag_bench.py:21-47`).

2. **Dataset selection**
   - This run used `dataset.strategy: reuse_existing_dataset` with dataset id `e2cd5edc6f7a11f19f83d591be0be0c1` (`config.resolved.yaml`).

3. **Broad retrieval probe**
   - For each question, `run_benchmark()` calls `run_retrieval()` and stores the returned payload verbatim in `raw_retrieval` (`src/ragflow_bench/execution/benchmark_runner.py:100-109`, `169`).
   - `RagflowClient.retrieve()` sends `question`, `dataset_ids`, `page_size`, `similarity_threshold`, `vector_similarity_weight`, and `top_k` to `/api/v1/retrieval` (`src/ragflow_bench/ragflow/client.py:164-177`).

4. **Chat setup**
   - If an answer LLM is configured, `ensure_chat()` creates one RAGFlow chat with `top_n`, `top_k`, `similarity_threshold`, `vector_similarity_weight`, and prompt config including `quote`/`refine_multiturn` (`src/ragflow_bench/execution/chat_runner.py:8-20`; request body in `src/ragflow_bench/ragflow/client.py:145-158`).

5. **Per-question session and answer**
   - With `fresh_session_per_question: true`, the runner creates a fresh session for each question (`src/ragflow_bench/execution/benchmark_runner.py:111-117`).
   - `run_chat()` calls `/api/v1/chat/completions` with `quote`, `max_tokens`, `temperature`, and `top_p` (`src/ragflow_bench/execution/chat_runner.py:24-35`; `src/ragflow_bench/ragflow/client.py:179-195`).

6. **RAGFlow chat retrieval/context selection**
   - In RAGFlow, chat retrieval calls `retriever.retrieval(..., page=1, page_size=dialog.top_n, similarity_threshold=dialog.similarity_threshold, vector_similarity_weight=dialog.vector_similarity_weight, top=dialog.top_k, rerank_mdl=rerank_mdl)` (`api/db/services/dialog_service.py:717-731`).
   - Returned chunks are packed into prompt knowledge via `kb_prompt(kbinfos, max_tokens)` before final message fitting (`api/db/services/dialog_service.py:756-784`).

7. **Answer generation and citation decoration**
   - RAGFlow appends citation instructions when both prompt config and request `quote` are true (`api/db/services/dialog_service.py:780-782`).
   - If the generated answer lacks citation markers, it may hydrate chunk vectors and insert citations using the same text/vector weights (`api/db/services/dialog_service.py:803-817`).

8. **Benchmark scoring**
   - The runner extracts `retrieved_document_ids`, `retrieved_chunk_ids`, `retrieved_scores`, and `retrieved_source_uris` from `raw_retrieval` chunks (`src/ragflow_bench/execution/benchmark_runner.py:135-146`).
   - It computes exact/normalized match and source recall, then writes `results.jsonl` (`src/ragflow_bench/execution/benchmark_runner.py:147-173`).

9. **Judge**
   - The Zhipu judge scores final answer correctness only; the prompt explicitly says not to reward retrieval quality or citations (`src/ragflow_bench/judge.py:72-97`).
   - `summarize_judged_rows()` counts only non-excluded rows with integer judge scores as scorable (`src/ragflow_bench/judge.py:104-130`).

## 5. Config Parameter Semantics

| Parameter | Current value | Code/docs-grounded meaning | Likely effect | Risk |
|---|---:|---|---|---|
| `retrieval.top_k` | 128 | Sent to `/api/v1/retrieval` and chat creation (`RagflowClient.retrieve/create_chat`). RAGFlow clamps search `top_k` to 1..2048 and passes it as `top`/`topk` candidate limit (`dataset_api_service.py:958-960`; `rag/nlp/search.py:603-611`). | Broad candidate-pool size before final page slicing/rerank. | More candidates can improve recall but increase latency/noise. |
| `retrieval.page_size` | 20 | Sent to retrieval endpoint as `page_size`; RAGFlow retrieval uses it as final page size after scoring/threshold (`rag/nlp/search.py:579-583`, `701-704`). | Number of chunks returned in raw retrieval artifact. | Does not by itself change chat context; chat uses `top_n`. |
| `similarity_threshold` | 0.05 | RAGFlow includes it in search request and filters final scored chunks with `sim >= threshold`; threshold is disabled only when vector weight <= 0 (`rag/nlp/search.py:603-612`, `690-699`). | Lower threshold admits more weak chunks; higher threshold prunes. | Too low adds distractors; too high drops sparse-but-relevant exact facts. |
| `vector_similarity_weight` | 0.3 | Hybrid fusion weight; RAGFlow computes full-text/term weight as `1 - vector_similarity_weight` (`rag/nlp/search.py:629-638`; API docs list the field). | 0.3 favors lexical/term scoring over vector. | Increasing may improve semantic recall but hurt exact IDs/numbers; decreasing may miss paraphrases. |
| `chat.top_n` | 8 | Stored on chat; RAGFlow chat passes it as retrieval `page_size` for answer context (`dialog_service.py:717-724`). RAGFlow docs list it as a chat body parameter. | Final number of chunks available to prompt packing. | More chunks improve coverage but add distractors and token pressure. |
| `chat.max_tokens` | 128 | Sent to chat completion; RAGFlow uses `max_tokens` as the total message-fit budget for system+knowledge+query and caps generation config to remaining budget if present (`dialog_service.py:756-791`). | Controls context fit and possibly output budget. Current 128 is tight for multi-fact answers. | Raising cost/latency; may still not help if context is wrong. |
| `quote` | true | Prompt config and request flag; RAGFlow adds citation prompt and may insert citations if absent (`dialog_service.py:780-817`). RAGFlow docs say quote controls whether source text is displayed. | Helps inspect cited chunks and references. | Citation insertion can add overhead; not a substitute for recall mapping. |
| `chunk_token_num` | 512 | Naive parser config; text and markdown paths read this value, and `naive_merge` merges/splits sections around the token target (`rag/app/naive.py:1005-1021`, `1170-1176`; `rag/nlp/__init__.py:1070-1138`). | Chunk granularity. 512 is a reasonable starting point, but merged shard contamination may dominate. | Too large mixes many facts; too small loses context. |
| `delimiter` | blank-line string in config | Parser delimiter passed to text/markdown/naive merge. Custom delimiters wrapped in backticks have special behavior: each segment is its own chunk and ignores `chunk_token_num` (`rag/nlp/__init__.py:1105-1124`). | Current blank-line delimiter does not preserve ERAGB document boundaries unless actual text structure aligns. | Boundary contamination when merged shards include multiple documents. |
| parser/chunk method | naive | RAGFlow naive parser supports docx/pdf/excel/txt/markdown/etc.; successive text is sliced by delimiter and merged to max token count (`rag/app/naive.py:839-845`). | General-purpose chunking, not source-type-specific enterprise parsing. | Can mix logical records and sources in merged shards. |
| reranker | not configured | RAGFlow retrieval accepts `rerank_id`; when present, `retriever.retrieval()` uses `rerank_by_model()` (`dataset_api_service.py:1019-1024`; `rag/nlp/search.py:641-649`). | Can improve final ranking among retrieved candidates. | Requires available rerank model; latency/cost; needs ablation. |
| RAPTOR / GraphRAG | disabled | Config sets both false. RAGFlow code has KG/GraphRAG paths, but current question set is exact-fact enterprise QA. | Not first-line for exact shard facts. | Likely expensive and may blur source provenance. |

## 6. Failure Mode Taxonomy

### Retrieval miss

Confirmed for many incorrect rows: the expected shard filename was not present in the top-20 raw retrieval `document_keyword` for 106/203 scorable rows. Among incorrect rows, only 18/96 had the expected shard in raw top-20, so many incorrect answers are likely true retrieval/ranking misses.

### Retrieved but not selected

Confirmed as possible and measurable: expected shard in raw top-20 but not necessarily in the chat prompt/context. The saved prompt titles contained expected shards in 90 scorable rows versus 97 raw-retrieval hits, so some broad-retrieval hits are lost before final prompt.

### Selected but wrong extraction

Confirmed: 13 incorrect rows had the expected shard filename in the chat prompt titles. That means final context sometimes includes the likely gold source but the model extracts the wrong value or is distracted by conflicting nearby chunks.

Example: `qst_0001` asks for default multipart upload limits. The answer gives 25MB/25MB while the gold answer is 10 MiB per file and 50 MiB total. Its raw retrieval top title was `github_shard_000013.txt`, not expected `github_shard_000104.txt`, so this example is primarily retrieval miss / distractor selection.

### Partial due to incomplete synthesis

Strongly suggested: project-related rows have low strict accuracy but high partial-or-better. On scorable rows, `project_related` strict was 5/28 = 17.9%, but partial-or-better was 24/28 = 85.7%. This pattern indicates the system often finds some relevant facts but does not aggregate all required details.

### Output truncation / answer budget

Likely contributor, not the dominant confirmed failure. `chat.max_tokens` is only 128, while RAGFlow uses `max_tokens` for message fitting and generation budget handling (`dialog_service.py:756-791`). Many answers are longer than 128 whitespace words in the saved output, so the exact downstream model semantics may differ, but the code path still makes the budget suspiciously tight for multi-part ERAGB answers.

### Source-recall bookkeeping issue

Confirmed. See Section 7.

### Judge/infra instability

Confirmed and severe. 297/500 rows are excluded: 99 infra errors and 198 judge errors. Common runner errors include HTTP read timeouts and LLM rate-limit strings. This reduces confidence in full-run quality conclusions and makes scorable-only breakdowns subject to selection bias.

## 7. Source Recall Metric Investigation

The current `source_recall=0.0` is unreliable and should be treated as broken for this run.

Code evidence:

- `source_recall()` computes exact set overlap between expected sources and retrieved sources (`src/ragflow_bench/scoring/retrieval_scoring.py:4-9`).
- The runner populates `retrieved_source_uris` only from `chunk.metadata.source_uri`, `chunk.meta_fields.source_uri`, or top-level `chunk.source_uri` (`src/ragflow_bench/execution/benchmark_runner.py:139-146`).
- The saved retrieval chunks in this run have fields such as `document_id`, `document_keyword`, `id`, `similarity`, `term_similarity`, and `vector_similarity`; they do not expose `source_uri`.
- Therefore `retrieved_source_uris` is empty for all 500 rows, so source recall is forced to 0.0 regardless of actual retrieval.

Mapping evidence:

- ERAGB prep can translate expected document IDs to shard URIs like `eragb-shard://github/github_shard_000104.txt` when `reference_granularity` is `shard` (`src/ragflow_bench/benchmarks/eragb_prep.py:381-408`).
- RAGFlow retrieval for this run exposes the shard filename in `document_keyword` (for example `github_shard_000013.txt`), not in `source_uri`.
- Diagnostic scan: expected shard filename appeared in raw retrieval `document_keyword` for 97/203 scorable rows and in the chat prompt titles for 90/203 scorable rows. If the metric were using shard filename matching, recall would not be zero.

Proposed diagnostic metric design, not implemented here:

1. Preserve three IDs for every chunk:
   - RAGFlow `document_id`.
   - RAGFlow visible document name / `document_keyword` / prompt title.
   - Benchmark canonical source URI (`eragb-shard://...`) or original doc ID.
2. Build a run-local lookup from `document_keyword` basename to expected shard URI.
3. Report separate metrics:
   - `raw_retrieval_shard_recall@20` from `/api/v1/retrieval` chunks.
   - `chat_context_shard_recall@top_n` from `raw_response.reference.chunks` or prompt titles.
   - `answer_citation_shard_recall` from cited reference chunks when `quote=true`.
4. Keep document-level recall separate from shard-level recall; do not compare shard-mode recall against document-level leaderboard metrics.
5. Store unmatched expected source IDs and unmatched retrieved document keywords for audit.

## 8. Source-Type and Reasoning-Type Findings

Because `source_types` were not persisted directly in result rows, this pass inferred source type from `expected_sources` URI prefixes. Rows with multiple expected sources count in multiple source-type buckets.

### Source-type breakdown on 203 scorable rows

| Expected source type | Count | Correct | Partial | Incorrect | Strict | Partial-or-better |
|---|---:|---:|---:|---:|---:|---:|
| confluence | 85 | 8 | 36 | 41 | 9.4% | 51.8% |
| jira | 48 | 15 | 19 | 14 | 31.2% | 70.8% |
| github | 36 | 13 | 15 | 8 | 36.1% | 77.8% |
| slack | 33 | 6 | 11 | 16 | 18.2% | 51.5% |
| google_drive | 28 | 7 | 8 | 13 | 25.0% | 53.6% |
| gmail | 27 | 10 | 4 | 13 | 37.0% | 51.9% |
| linear | 22 | 6 | 9 | 7 | 27.3% | 68.2% |
| fireflies | 18 | 2 | 8 | 8 | 11.1% | 55.6% |
| hubspot | 10 | 2 | 2 | 6 | 20.0% | 40.0% |

Interpretation:

- Confluence is the largest drag: many partials imply some relevant context but incomplete extraction/synthesis. Confluence-like pages often need section-aware splitting and title/path preservation.
- Slack and Fireflies have low strict accuracy; conversational data is noisy, temporally ordered, and speaker-dependent, so naive chunks may mix distracting facts.
- GitHub and Jira do better, likely because structured issue/PR/release-note language aligns with lexical retrieval and exact identifiers.
- Project/completeness questions need multi-source aggregation; single top chunks are often insufficient.

### Reasoning-type breakdown on 203 scorable rows

| Reasoning type | Count | Correct | Partial | Incorrect | Strict | Partial-or-better |
|---|---:|---:|---:|---:|---:|---:|
| basic | 75 | 25 | 9 | 41 | 33.3% | 45.3% |
| semantic | 49 | 11 | 7 | 31 | 22.4% | 36.7% |
| project_related | 28 | 5 | 19 | 4 | 17.9% | 85.7% |
| intra_document_reasoning | 18 | 9 | 2 | 7 | 50.0% | 61.1% |
| conflicting_info | 8 | 3 | 1 | 4 | 37.5% | 50.0% |
| high_level | 7 | 5 | 0 | 2 | 71.4% | 71.4% |
| completeness | 6 | 1 | 1 | 4 | 16.7% | 33.3% |
| constrained | 4 | 2 | 1 | 1 | 50.0% | 75.0% |
| miscellaneous | 4 | 2 | 0 | 2 | 50.0% | 50.0% |
| info_not_found | 4 | 4 | 0 | 0 | 100.0% | 100.0% |

Caveat: counts are scorable-only; 297 excluded rows may change the distribution.

## 9. Parameter-Only Optimization Hypotheses

| Rank | Experiment | Expected upside | Risk | Why this should be tested |
|---:|---|---|---|---|
| 1 | Increase `chat.top_n` from 8 to 16, then 24/32 | Improve final context recall when broad retrieval finds gold but top-8 misses it. | More distractors; context budget pressure. | Chat uses `top_n` as retrieval page size for answer context; raw top-20 had more gold hits than prompt context. |
| 2 | Increase `chat.max_tokens` from 128 to 256/384/512 | Better multi-fact answers and less context/message fitting pressure. | More cost/latency; potential verbosity. | ERAGB has many multi-part/project/completeness questions; RAGFlow uses this budget in prompt fitting and generation config. |
| 3 | Enable/test a reranker if a rerank model is available | Improve ordering of semantically relevant candidates before prompt packing. | Latency/cost; model availability. | RAGFlow has a `rerank_id` path, and external reranker docs support reranking candidates before top-N selection. |
| 4 | Test vector weights 0.5 and 0.7, plus maybe 0.2 | Improve semantic/paraphrase retrieval; discover lexical/vector balance. | Higher vector weight may hurt exact numeric/entity matching. | Current 0.3 favors lexical. Azure notes hybrid retrieval helps combine exact and conceptual matching; RAGFlow weight directly controls term/vector fusion. |
| 5 | Test similarity thresholds 0.0, 0.03, 0.1 | Understand whether threshold prunes relevant low-scoring chunks or admits too many distractors. | Too low adds noise; too high kills recall. | RAGFlow filters by final similarity after fusion. Current top scores are often around 0.35-0.40, but per-query distribution matters. |
| 6 | Chunk size alternatives: 256/384 with overlap or section-aware splitting | Reduce multi-document contamination and improve exact-fact localization. | Smaller chunks may lose context and increase index size. | Azure suggests 512 tokens as a starting point, not universal; ERAGB merged shards and conversations need boundary preservation. |
| 7 | Use custom ERAGB boundary delimiter during ingestion | Strong source-boundary preservation. | RAGFlow custom delimiter behavior can ignore `chunk_token_num`, creating very large per-document chunks if documents are long. | RAGFlow `naive_merge` treats backtick-wrapped custom delimiters specially; this can prevent cross-document mixing but must be tested carefully. |
| 8 | RAPTOR/GraphRAG | Low priority for exact-fact QA. | Expensive, more moving parts, source attribution complexity. | No local evidence yet that graph summarization is the bottleneck. |

## 10. Code-Level Optimization Hypotheses

| Rank | Recommendation | Expected benefit | Evidence / rationale | Risk |
|---:|---|---|---|---|
| 1 | Fix source ID mapping and recall diagnostics | Trustworthy retrieval/context metrics. | Current recall compares expected shard URIs to empty `retrieved_source_uris`. | Low; evaluation-only, but must avoid mixing doc/shard granularity. |
| 2 | Record chat-context chunks separately from broad retrieval chunks | Distinguish retrieval miss from top-N/context loss. | RAGFlow answer prompt can include fewer/different chunks than raw retrieval. | Medium if relying on prompt parsing; better to use structured `reference`. |
| 3 | Source-aware chunk packing | Preserve one chunk per source/section and avoid redundant distractors. | Prompt can include many similar distractor chunks; top_n increase may worsen this. | Requires RAGFlow/eval code changes. |
| 4 | Final-context reranking / MMR diversity | Improve final selected context, not just broad candidate pool. | Incorrect rows sometimes have gold in broad retrieval or prompt but lose extraction. | Needs careful ablation to avoid reducing exact recall. |
| 5 | Exact-fact extraction before synthesis | Improve numeric/date/entity questions. | Incorrect answers often give plausible but wrong numbers from distractor chunks. | Can overfit if too benchmark-specific. |
| 6 | Anti-distractor answer prompt | Force quote-grounded answer and conflict resolution. | ERAGB has conflicting-info and exact-fact questions. | Prompt changes may reduce recall or increase abstentions. |
| 7 | Answer abstention rules | Improve `info_not_found` and reduce hallucinated answers. | Current prompt already has an empty-response instruction; still needs evaluation. | Can lower partial-or-better if too conservative. |
| 8 | Confluence-specific section/path parsing | Improve largest weak source type. | Confluence has 85 scorable source mentions with only 9.4% strict. | Ingestion changes may require reindexing. |
| 9 | Multi-source aggregation diagnostics | Improve project_related/completeness questions. | Project-related partial-or-better is high but strict is low. | More complex scoring and prompt context. |

## 11. Proposed Next Experiment Matrix

Do not launch these until diagnostics are fixed enough to measure raw-retrieval recall and chat-context recall.

| Experiment | Change | Keep fixed | Primary metric expected to improve | Secondary metric |
|---|---|---|---|---|
| Baseline re-score diagnostics | No retrieval/chat config change; recompute diagnostic shard recall from saved artifacts or a small controlled rerun | Same dataset/model | Metric trust | raw vs context recall gap |
| Top-N 16 | `chat.top_n=16` | top_k=128, max_tokens baseline initially | chat-context recall | partial-or-better, strict |
| Top-N 24/32 | `chat.top_n=24/32` | best previous max_tokens | chat-context recall | distractor rate, latency |
| Max tokens 256/384/512 | raise `chat.max_tokens` | retrieval settings fixed | strict on multi-part/project rows | answer length, judge errors |
| Vector weight sweep | `vector_similarity_weight=0.5/0.7`, maybe 0.2 | top_n/max_tokens chosen | raw retrieval recall by source type | exact numeric/entity accuracy |
| Threshold sweep | `similarity_threshold=0.0/0.03/0.1` | vector weight fixed | raw retrieval recall and noise | prompt distractor rate |
| Reranker on/off | add `rerank_id` if available | same candidate pool | context recall@top_n and strict accuracy | latency/cost |
| Chunk size 256/384 | reingest controlled subset | same retrieval/chat config | source recall and exact-fact extraction | chunk count/cost |
| Boundary-aware ingestion | use section/source boundary strategy | same question subset | shard/doc boundary precision | context contamination |

Recommended subset for fast iteration: use a fixed stratified sample of scorable plus formerly excluded rows across Confluence, Slack, Fireflies, GitHub/Jira, and project/completeness reasoning. Avoid using only the 203 scorable rows for final conclusions.

## 12. What Not To Do Yet

- Do not run another full 500-question benchmark before fixing source recall diagnostics and judge/infra instability.
- Do not interpret `average_source_recall=0.0` as true retrieval failure.
- Do not randomly swap embedding or answer models before measuring retrieval/context recall; that would obscure the bottleneck.
- Do not jump to RAPTOR/GraphRAG before simpler reranking, top-N, token budget, and boundary-aware chunking tests.
- Do not optimize only against the 203 scorable rows; the excluded 297 rows are too large a slice to ignore.
- Do not compare shard-level recall from merged corpus mode to document-level recall without a clear mapping.
- Do not treat exact-match metrics as meaningful for this dataset; judge-based semantic scoring is more appropriate, but judge reliability must improve.

## 13. Final Recommendation

Prioritized plan for the next run:

1. **Fix evaluation diagnostics first**: compute raw-retrieval and chat-context shard recall from `document_keyword` / prompt/reference chunks, not `source_uri` only.
2. **Stabilize judge/infra**: reduce judge errors and API timeouts so more than 203/500 rows are scorable.
3. **Run a small controlled ablation**, not a full benchmark: baseline, `top_n=16`, `max_tokens=256/384`, and one reranker-on run if a reranker is configured.
4. **Then test retrieval balance**: vector weight and threshold sweeps with source-type breakdowns.
5. **Then test ingestion/chunking**: smaller/section-aware chunks and source-boundary-preserving ingestion, especially for Confluence, Slack, and Fireflies.
6. **Only after diagnostics and small ablations** consider RAGFlow code-level improvements to source-aware context packing, exact-fact extraction, and multi-source aggregation.

### Follow-up checklist

- [ ] Add diagnostic metrics for raw-retrieval shard recall and chat-context shard recall.
- [ ] Preserve mapping from RAGFlow `document_keyword` / document name to ERAGB shard URI.
- [ ] Separate shard-level and document-level recall in reports.
- [ ] Re-judge or retry excluded rows enough to reduce the 59.4% exclusion rate.
- [ ] Run a small top-N sweep: 8 vs 16 vs 24/32.
- [ ] Run a small max-token sweep: 128 vs 256/384/512.
- [ ] Test reranker on/off if a RAGFlow rerank model is available.
- [ ] Run vector weight and threshold sweeps after diagnostic recall is trustworthy.
- [ ] Test boundary-aware or section-aware ingestion on a subset before reingesting the full corpus.
- [ ] Report results by source type and reasoning type, not only aggregate accuracy.
