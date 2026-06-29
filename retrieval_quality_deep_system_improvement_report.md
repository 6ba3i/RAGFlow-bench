# RAGFlow EnterpriseRAG-Bench Deep System Improvement Investigation

## 1. Executive Summary

**Direct opinion:** the current EnterpriseRAG-Bench bottleneck is **not one thing**. The strongest code-and-artifact-grounded diagnosis is:

1. **Evaluation is currently untrustworthy for source quality** because `source_recall=0.0` is a harness mapping bug, and 297/500 rows were excluded from judged accuracy.
2. **Retrieval/ranking is the largest answer-quality bottleneck among scorable failures**: 78/96 incorrect scorable rows did not have the expected shard in the saved raw top-20 retrieval artifact.
3. **Final context selection is a real but smaller bottleneck**: 7/203 scorable rows had the expected shard in raw retrieval but not in final chat context; 5 of those were incorrect.
4. **Synthesis/exact-fact extraction is also confirmed**: 13 incorrect rows had the expected shard in the final chat context but still answered incorrectly.
5. **Ingestion/source-boundary quality is likely upstream of both retrieval and extraction**: the run used merged source-type shards, RAGFlow `naive` text chunking, blank-line delimiter, and top-level shard filenames as visible document titles. This preserves some shard identity, but it does not preserve original enterprise-record boundaries as first-class searchable metadata.

**What is most worth fixing first:** do not start with a full rerun or a blind `top_n/max_tokens` increase. The first implementation sequence should be: **evaluation/source mapping fixes**, then **small diagnostic slices by failure bucket/source type**, then **source-boundary-aware ingestion**, then **final-context reranking/diversity**, then **exact-fact extraction/validation prompting**, and only then parameter sweeps.

**Strict-accuracy upside is most likely from:**

- fixing retrieval misses for exact identifiers/numbers/titles, especially Confluence, Slack, and Fireflies;
- preserving original document/source boundaries and metadata during ingestion;
- adding a second-stage reranker or score fusion that improves top-8 context quality without merely expanding it;
- adding quote-first/evidence-table exact-value extraction for rows where gold context is already present.

## 2. Prior Report Review

### What the prior report got right

Confirmed from artifacts:

- `judge_summary.json` reports 500 total questions, 203 scorable questions, 297 exclusions, 99 infra exclusions, 198 judge errors, 67 correct, 40 partial, 96 incorrect, strict accuracy 0.3300, partial-credit answer accuracy 0.4286, and partial-or-better 0.5271.
- `summary.json` reports `average_source_recall: 0.0` and `failure_type_counts.retrieval_failure: 420`, but this is not trustworthy as retrieval diagnosis.
- The harness computes source recall from `retrieved_source_uris`, while saved RAGFlow chunks expose fields such as `document_id`, `document_keyword`/`document_name`, and chunk IDs rather than canonical `source_uri`.
- Some wrong answers include the likely gold shard in the final context, so failures are not only retrieval misses.

### What the prior report left under-proven or too parameter-focused

- It correctly suggested diagnostic experiments such as larger `chat.top_n` and `max_tokens`, but those can inflate judge score by brute-force context rather than improving retrieval/context quality.
- It did not deeply separate raw retrieval miss, raw-to-final-context loss, final context wrong extraction, distractor use, partial multi-fact synthesis, and evaluation ambiguity.
- It did not inspect RAGFlow’s actual final-context flow deeply enough: chat uses `dialog.top_n` as retrieval `page_size`, then `kb_prompt(kbinfos, max_tokens)` token-packs the returned chunks before message fitting.
- It did not emphasize enough that RAGFlow’s ES path fuses locally computed term similarity with a second KNN-only score, rather than using rank-based fusion such as RRF.

### What this deeper pass adds

- A refined scorable-row taxonomy with counts and representative question IDs.
- A code-grounded path from benchmark runner to RAGFlow retrieval to prompt packing and citation decoration.
- A stricter distinction between score inflation and real retrieval/context/synthesis/evaluation improvements.
- Source-type deep dives for Confluence and Slack/Fireflies.
- A ranked backlog that labels every recommendation as real improvement or score inflation.

## 3. Definition: Score Inflation vs Real Improvement

### Score inflation

A change is **score inflation** when it increases final judge score primarily by giving the answer model more raw context or output space without improving the quality, rank, provenance, or diagnostic trustworthiness of retrieved evidence. Examples:

- raising `chat.top_n` from 8 to 32 without measuring final-context recall, redundancy, and distractor rate;
- raising `max_tokens` until the prompt can carry many more chunks, while leaving ranking and chunking unchanged;
- optimizing only partial-or-better accuracy on the 203 scorable rows;
- rerunning the full benchmark before fixing source recall and judge/infra exclusions.

### Retrieval improvement

A **retrieval improvement** increases the probability that the correct source/chunk appears in the raw candidate set at a fixed candidate budget, or improves its rank at fixed `top_k/page_size`. It should improve `raw_retrieval_shard_recall@k`, `raw_retrieval_source_recall@k`, MRR, or rank of expected shard.

### Context-selection improvement

A **context-selection improvement** increases the probability that the best evidence from a fixed raw candidate pool reaches final prompt context at a fixed `top_n` and fixed prompt budget. Examples: reranking, MMR/diversity, source balancing, exact-token boosts, metadata boosts, and redundancy pruning.

### Synthesis/extraction improvement

A **synthesis/extraction improvement** increases correctness when the gold evidence is already in final context. It should improve rows where `final_context_shard_recall=true`, especially exact numeric/entity/date/enum questions, without requiring more chunks.

### Evaluation-quality improvement

An **evaluation-quality improvement** makes benchmark measurements reflect the real pipeline. It does not directly improve RAGFlow, but it prevents wrong conclusions. Examples: source mapping, final-context persistence, citation mapping, judge retry stabilization, and metrics split by failure stage.

## 4. Evidence Reviewed

### Local artifacts

- `/home/idriss/RAGFlow-bench/retrieval_quality_investigation.md`
- `/home/idriss/RAGFlow-bench/outputs/enterprise_rag_bench_full_20260626_101748/summary.json`
- `/home/idriss/RAGFlow-bench/outputs/enterprise_rag_bench_full_20260626_101748/judge_summary.json`
- `/home/idriss/RAGFlow-bench/outputs/enterprise_rag_bench_full_20260626_101748/results.jsonl`
- `/home/idriss/RAGFlow-bench/outputs/enterprise_rag_bench_full_20260626_101748/judge_results.jsonl`
- `/home/idriss/RAGFlow-bench/outputs/enterprise_rag_bench_full_20260626_101748/config.resolved.yaml`
- `/home/idriss/RAGFlow-bench/outputs/enterprise_rag_bench_full_20260626_101748/document_registry.json`

### Benchmark code

- `src/ragflow_bench/execution/benchmark_runner.py:49-187`
- `src/ragflow_bench/execution/retrieval_runner.py:8-16`
- `src/ragflow_bench/execution/chat_runner.py:8-35`
- `src/ragflow_bench/ragflow/client.py:145-195`
- `src/ragflow_bench/scoring/retrieval_scoring.py:4-9`
- `src/ragflow_bench/scoring/failure_classification.py:4-15`
- `src/ragflow_bench/benchmarks/enterprise_rag_bench.py:21-74`
- `src/ragflow_bench/benchmarks/eragb_prep.py:330-386`
- `src/ragflow_bench/judge.py:19-30`, `src/ragflow_bench/judge.py:72-134`

### RAGFlow source code

- `/home/idriss/commits/ragflow/api/apps/services/dataset_api_service.py:927-1087`
- `/home/idriss/commits/ragflow/api/apps/restful_apis/dataset_api.py:263-276`
- `/home/idriss/commits/ragflow/api/db/services/dialog_service.py:717-835`
- `/home/idriss/commits/ragflow/rag/nlp/search.py:573-771`
- `/home/idriss/commits/ragflow/rag/prompts/generator.py:135-169`
- `/home/idriss/commits/ragflow/rag/app/naive.py:1005-1021`, `/home/idriss/commits/ragflow/rag/app/naive.py:1170-1176`
- `/home/idriss/commits/ragflow/rag/nlp/__init__.py:1070-1138`

### External/upstream evidence

Additional current upstream evidence gathered for this pass:

- RAGFlow's run-retrieval-test docs describe retrieval as hybrid search: keyword similarity plus vector cosine without a reranker, or keyword similarity plus reranking score when a reranker is selected.
- RAGFlow's chat/start docs expose `top_n`, `top_k`, `similarity_threshold`, `vector_similarity_weight`, `rerank_id`, `quote`, and `empty_response` as chat-side controls.
- RAGFlow's parent-child chunking docs describe a newer strategy where smaller child chunks improve recall while parent chunks preserve broader context; this is directly relevant to ERAGB source-boundary fixes.
- RAGFlow's auto-keyword/auto-question, tag-set, and PageRank docs provide upstream-supported retrieval-enrichment levers beyond brute-force context expansion.
- Upstream issue https://github.com/infiniflow/ragflow/issues/15428 flags reranker score-scale mismatch across providers, so reranker experiments should compare fixed-candidate/fixed-top-N quality and not assume cross-provider score calibration.

- RAGFlow HTTP API reference: https://ragflow.io/docs/http_api_reference
- RAGFlow retrieval test docs: https://ragflow.io/docs/run_retrieval_test
- RAGFlow Python API parser config docs: https://ragflow.io/docs/python_api_reference
- RAGFlow dataset configuration/chunking docs: https://ragflow.io/docs/configure_knowledge_base
- RAGFlow retrieval component docs: https://ragflow.io/docs/retrieval_component
- RAGFlow Token chunker component docs: https://ragflow.io/docs/chunker_token_component
- RAGFlow run retrieval test docs: https://ragflow.io/docs/run_retrieval_test
- RAGFlow start chat docs: https://ragflow.io/docs/start_chat
- RAGFlow child chunking docs: https://ragflow.io/docs/configure_child_chunking_strategy
- RAGFlow auto-keyword/auto-question docs: https://ragflow.io/docs/autokeyword_autoquestion
- RAGFlow tag set docs: https://ragflow.io/docs/use_tag_sets
- RAGFlow PageRank docs: https://ragflow.io/docs/set_page_rank
- RAGFlow upstream reranker scale issue: https://github.com/infiniflow/ragflow/issues/15428
- Azure AI Search hybrid search overview: https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview
- Azure AI Search hybrid search scoring/RRF: https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking
- Azure AI Search chunking guidance: https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-chunk-documents
- Azure AI Search semantic/document-layout chunking: https://learn.microsoft.com/en-us/azure/search/search-how-to-semantic-chunking
- Elasticsearch RRF docs: https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion
- Cohere rerank docs: https://docs.cohere.com/docs/rerank and https://docs.cohere.com/reference/rerank
- Ragas metrics docs: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/ and https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/
- TruLens RAG triad: https://www.trulens.org/getting_started/core_concepts/rag_triad/
- LlamaIndex evaluation docs: https://developers.llamaindex.ai/python/framework/module_guides/evaluating/

## 5. Pipeline Deep Dive

### 5.1 Benchmark question and source loading

Confirmed fact: `EnterpriseRAGBenchAdapter.load_questions()` reads JSONL/JSON/CSV/parquet and chooses `expected_sources` from `expected_sources`, `expected_doc_ids`, `expected_documents`, or `citations` (`enterprise_rag_bench.py:21-47`). ERAGB prep can emit shard-level expected sources by mapping expected doc IDs through `doc_id_to_shard` (`eragb_prep.py:381-386`).

This run’s config used `reference_granularity` indirectly through prepared questions: expected sources appear as strings like `eragb-shard://confluence/confluence_shard_000191.txt` in results.

### 5.2 Dataset and ingestion state

Confirmed fact: the run reused an existing RAGFlow dataset (`dataset.strategy: reuse_existing_dataset`, dataset id `e2cd5edc6f7a11f19f83d591be0be0c1`) and did not persist a useful document registry (`document_registry.json` is only 73 bytes). Therefore, the run artifacts can map by visible shard filename but not by a full benchmark registry of original doc IDs to RAGFlow document IDs.

The config says the dataset was created with RAGFlow `chunk_method: naive`, `chunk_token_num: 512`, and delimiter equal to blank-line text (`config.resolved.yaml`). RAGFlow’s dataset API exposes multiple chunk methods including `naive`, `email`, `book`, `one`, `qa`, etc. (`dataset_api.py:263-270`), and official docs describe built-in chunking templates for different layouts.

### 5.3 Raw retrieval probe

Confirmed fact: for each question, `run_benchmark()` calls `run_retrieval()` and stores `raw_retrieval` verbatim (`benchmark_runner.py:100-109`, `169`). `RagflowClient.retrieve()` sends `question`, `dataset_ids`, `page_size`, `similarity_threshold`, `vector_similarity_weight`, and `top_k` to `/api/v1/retrieval` (`client.py:164-177`).

RAGFlow’s dataset search reads `size`, `similarity_threshold`, `vector_similarity_weight`, `top_k`, optional `rerank_id`, optional keyword expansion, and optional metadata filters (`dataset_api_service.py:953-1069`).

### 5.4 Chat setup and answer call

Confirmed fact: `ensure_chat()` creates one RAGFlow chat with `top_n`, `top_k`, `similarity_threshold`, `vector_similarity_weight`, and prompt config `{quote, refine_multiturn}` (`chat_runner.py:8-21`; `client.py:145-158`). `run_chat()` calls `/api/v1/chat/completions` with the question, `quote`, `max_tokens`, temperature, and top-p (`chat_runner.py:24-35`; `client.py:179-195`).

### 5.5 RAGFlow final context selection

Confirmed fact: RAGFlow chat retrieval calls `retriever.retrieval()` using `page_size=dialog.top_n`, `similarity_threshold=dialog.similarity_threshold`, `vector_similarity_weight=dialog.vector_similarity_weight`, `top=dialog.top_k`, and optional `rerank_mdl` (`dialog_service.py:717-731`). Then it calls `retrieval_by_children()` and passes `kbinfos` to `kb_prompt(kbinfos, max_tokens)` (`dialog_service.py:736`, `dialog_service.py:756`).

`kb_prompt()` takes chunks in current order, accumulates chunk token counts, and truncates the knowledge list once cumulative tokens exceed `max_tokens * 0.97` (`generator.py:135-148`). It then renders each retained chunk with ID, title, URL, metadata, and content (`generator.py:157-167`). Afterward, `message_fit_in()` is called with `int(max_tokens * 0.95)` (`dialog_service.py:784`).

Important implication: `chat.top_n` controls how many chunks are requested, but `max_tokens` controls how many of those chunks survive prompt packing. This makes blind top_n increases especially risky: they can add candidates that never fit, or fit only by crowding out useful text.

### 5.6 Hybrid scoring and ranking

Confirmed fact: RAGFlow retrieval builds a candidate request with `topk=top`, vector enabled, and `similarity=similarity_threshold` (`search.py:603-612`). It computes `term_similarity_weight = 1 - vector_similarity_weight` (`search.py:629`). If a reranker exists, it calls `rerank_by_model()`; otherwise ES uses `_knn_scores()` plus `rerank_with_knn()` to combine term and vector scores (`search.py:641-680`). It sorts by combined similarity and filters below threshold unless vector weight is zero (`search.py:687-704`). Returned chunks include `docnm_kwd`, `doc_id`, similarities, content, positions, and metadata-like fields (`search.py:721-739`).

Probable hypothesis: because RAGFlow’s ES path combines score magnitudes from term and KNN paths, score calibration can bury exact identifiers/numbers if vector or term score distributions are poorly matched for a query. This is not the same as RRF. Azure and Elasticsearch docs describe RRF as merging ranked lists without requiring comparable score scales, which is one reason RRF is attractive for hybrid retrieval.

### 5.7 Citation and reference behavior

Confirmed fact: when quote mode is enabled, RAGFlow appends citation instructions and later inserts or repairs citations (`dialog_service.py:780-824`). It also prunes `doc_aggs` to cited docs when citations exist, or falls back to all doc aggs if no cited docs remain (`dialog_service.py:826-830`). The saved `raw_response.reference.chunks` are the best local artifact for final context, not the final answer citations alone.

## 6. Failure Mode Counts and Representative Rows

### 6.1 Scorable-row taxonomy

The following counts are over **203 scorable judged rows**. Matching used expected shard basename against saved raw retrieval chunk fields and final `raw_response.reference.chunks` fields.

| Bucket | Count | Correct | Partial | Incorrect | Interpretation |
|---|---:|---:|---:|---:|---|
| Expected shard absent from raw retrieval top-20 | 106 | 14 | 14 | 78 | Main retrieval/ranking miss bucket. |
| Expected shard present in raw retrieval but absent from final chat context | 7 | 1 | 1 | 5 | Raw-to-final selection/packing loss. |
| Expected shard present in final context but answer wrong | 13 | 0 | 0 | 13 | Extraction/distractor/conflict/synthesis failure. |
| Expected shard present in final context but answer partial | 25 | 0 | 25 | 0 | Multi-fact completeness/synthesis failure. |
| Expected shard present in final context and answer correct | 52 | 52 | 0 | 0 | Happy path. |

Caveat: “expected shard absent” does not prove no relevant chunk was retrieved, because a question can sometimes be answerable from another shard or source. The 14 correct and 14 partial rows in the absent bucket prove that shard-basename recall is useful but not sufficient as the only metric.

### 6.2 Representative failed row IDs

| Row | Verdict | Source type | Diagnosed failure mode | Evidence summary |
|---|---|---|---|---|
| `qst_0001` | incorrect | GitHub | expected shard absent from raw retrieval | Asked multipart limits; answer gave wrong 25MB/25MB instead of 10 MiB/50 MiB. |
| `qst_0013` | incorrect | Confluence | raw present but final context absent | Expected Confluence shard in raw retrieval, not final reference chunks; answer gave generic canary/cutover but missed two-phase protocol details. |
| `qst_0020` | incorrect | GitHub | final context present but wrong exact behavior | Expected shard present in context; answer said null preserved instead of correct normalizer behavior. |
| `qst_0021` | incorrect | Confluence | final context present but wrong exact value | Expected shard present; answer gave 5 business days, gold is 3 business days. |
| `qst_0023` | incorrect | Confluence | final context present but incomplete/wrong enumerated facts | Expected shard present; answer summarized docs review generally but missed exact required roles/names. |
| `qst_0038` | incorrect | Fireflies | expected shard absent from raw retrieval | Asked median/p95 latency; answer not found. |
| `qst_0043` | incorrect | Slack | expected shard absent plus distractor | Answer used a plausible unrelated latency cause; gold was tenant-32 long-context burst/KV-cache pressure. |
| `qst_0097` | incorrect | GitHub | final context present but incomplete exact mechanism | Expected shard present; answer discussed seed handling generally but missed storage/normalization specifics. |
| `qst_0102` | incorrect | Confluence | final context present but false not-found | Expected shard present; answer claimed handbook lacked thresholds; gold has p-value and sample-size guidelines. |
| `qst_0104` | incorrect | Confluence | raw present but final context absent | Asked buddy time/day; raw had shard but final context did not. |
| `qst_0176` | partial | Slack | final context present but exact date off/incomplete | Answer gave 2026-04-06; gold asks Monday 2026-04-07 09:00 UTC. |
| `qst_0195` | incorrect | Google Drive | final context present but wrong exact value extraction | Expected shard present; answer referenced correct lab/shard but did not give the correct requested cost. |
| `qst_0224` | incorrect | Slack | raw present but final context absent / wrong identifier | Raw contained expected shard; final context absent; answer gave `end_user_pseudo_42`, gold `tokenized_actor_id`. |
| `qst_0245` | incorrect | HubSpot | raw present but final context absent / distractor source | Answer named the wrong prospect. |
| `qst_0450` | partial | Fireflies | raw present but final context absent / aggregation failure | Asked count of transcripts mentioning residency; answer said 6, gold 8. |

### 6.3 Source-type scorable breakdown

Rows with multiple expected source types count in each type.

| Source type | Count | Correct | Partial | Incorrect | Strict | Partial-or-better |
|---|---:|---:|---:|---:|---:|---:|
| confluence | 54 | 7 | 19 | 28 | 13.0% | 48.1% |
| jira | 43 | 14 | 18 | 11 | 32.6% | 74.4% |
| slack | 33 | 6 | 11 | 16 | 18.2% | 51.5% |
| github | 32 | 13 | 11 | 8 | 40.6% | 75.0% |
| google_drive | 27 | 7 | 7 | 13 | 25.9% | 51.9% |
| gmail | 25 | 10 | 3 | 12 | 40.0% | 52.0% |
| linear | 20 | 6 | 7 | 7 | 30.0% | 65.0% |
| fireflies | 11 | 2 | 1 | 8 | 18.2% | 27.3% |
| hubspot | 10 | 2 | 2 | 6 | 20.0% | 40.0% |

The source-type counts differ from the prior report because this pass counted source types from `expected_sources` in the scorable result rows available in this checkout. The conclusion is directionally unchanged: Confluence is the largest weak footprint; Slack and Fireflies are weak conversational types; GitHub/Jira are comparatively stronger.

## 7. Final Context Selection Analysis

### Confirmed facts

- The run used `retrieval.page_size=20`, `retrieval.top_k=128`, and `chat.top_n=8`.
- Saved final references have exactly 8 chunks for all 203 scorable rows. This indicates chat retrieval returned eight reference chunks per scorable row.
- The rendered prompt often has more than 8 `Title:` lines because the system prompt includes title occurrences inside chunk content, not because more than 8 reference chunks were selected. Therefore final context should be measured from `raw_response.reference.chunks`, not prompt title regex alone.
- The expected shard was present in raw top-20 but absent from final context for 7 scorable rows, showing final selection loss exists but is smaller than raw retrieval miss.
- The expected shard was present in final context for 90 scorable rows: 52 correct, 25 partial, 13 incorrect.

### Ranking and reranking

RAGFlow supports an optional rerank model path: `dataset_api_service.py:1044-1048` builds `rerank_mdl` from `rerank_id`, and `search.py:641-649` calls `rerank_by_model()` when present. This run’s config did not include a reranker.

External support: RAGFlow retrieval docs state that without a reranker it combines weighted keyword similarity and weighted vector cosine similarity; with a reranker it combines weighted keyword similarity and weighted reranking score. Cohere’s rerank docs describe rerankers as sorting candidate texts by query relevance, and the API returns ordered relevance scores. This supports a **real context-selection improvement**: rerank a fixed candidate pool into a fixed final top-N, rather than increasing top-N.

### Redundancy and source domination

In saved final context chunks, duplicate first-200-character prefixes were not observed among scorable rows, and no row had one document occupying 50%+ of the 8 final chunks by visible document name. This argues against simple duplicate-chunk collapse as the main problem in this run.

However, this does **not** rule out semantic redundancy or distractor domination, because many ERAGB shards contain similarly named enterprise artifacts and some answers cite plausible but wrong sources. A better metric should compute document/source-type diversity, near-duplicate similarity, and distractor overlap against expected sources.

### Prompt packing and token budget

RAGFlow `kb_prompt(kbinfos, max_tokens)` cuts the knowledge list based on cumulative chunk token count (`generator.py:135-148`). Then `message_fit_in()` fits system+knowledge+query into `max_tokens * 0.95` (`dialog_service.py:784`). The run set `chat.max_tokens=128`, which is suspiciously small relative to 512-token chunks. But raising it blindly is still score inflation unless paired with fixed-`top_n` evidence-quality metrics.

## 8. Ingestion and Source-Boundary Analysis

### Confirmed local behavior

ERAGB prep merges documents into source-type shard files. The merge loop groups by source type and flushes based on target bytes and max docs (`eragb_prep.py:330-343`). It maps original document IDs to shard URIs when `reference_granularity=shard` (`eragb_prep.py:381-386`). Merged blocks include `<<<ERAGB_DOC_BOUNDARY>>>`, document ID, source type, title, and content; this was visible in saved RAGFlow chunks.

RAGFlow text parsing for `.txt` uses `TxtParser()` with `chunk_token_num` and delimiter (`naive.py:1005-1008`). Later, chunks are produced by `naive_merge()` (`naive.py:1170-1176`). `naive_merge()` splits by delimiters and merges sections up to token budget (`__init__.py:1070-1138`). For backtick-wrapped custom delimiters, it treats each segment as its own chunk and ignores `chunk_token_num` (`__init__.py:1105-1124`).

### Source-boundary risk

Probable hypothesis: the current blank-line delimiter does not treat `<<<ERAGB_DOC_BOUNDARY>>>` as a hard boundary. Saved final chunks usually contain one boundary marker, but some final chunks contain two, and many chunks are slices of a shard file rather than first-class original source records. This can cause:

- unrelated records to share nearby chunk context;
- title/path/source metadata to be embedded as plain text instead of structured fields;
- weak field-specific retrieval for source titles, issue IDs, dates, and participant names;
- Confluence pages to lose section hierarchy;
- Slack/Fireflies transcripts to lose thread/speaker/time structure;
- GitHub/Jira issue context to fragment or mix fields.

### External best-practice support

- Azure chunking guidance recommends starting around 512 tokens with overlap, but explicitly treats this as a starting point that should be adapted to content shape.
- Azure semantic/document-layout chunking docs emphasize preserving headings and semantically coherent units.
- RAGFlow docs describe multiple chunk templates for different file layouts and a newer ingestion pipeline for customized ingestion/cleansing workflows.

### Real ingestion improvements

- Reingest with source-boundary-aware splitting that makes each original ERAGB document/record a first-class unit before chunking.
- Consider RAGFlow parent-child chunking for ERAGB: child chunks can target recall while parent chunks preserve answer context, avoiding the false choice between tiny exact chunks and huge contaminated chunks.
- Store original doc ID, shard filename, source type, title, timestamp/thread/speaker fields, and path as structured metadata, not only inline text.
- Use source-type-specific serializers: Confluence title/heading/body; Slack channel/thread/timestamp/speaker/message; Fireflies meeting title/date/speaker/utterance; GitHub/Jira structured fields.
- Avoid making one giant chunk per original doc if using custom delimiter; combine boundary splitting with secondary token chunking per original record.

## 9. Hybrid Retrieval and Ranking Mechanics

### Code-grounded mechanics

- `vector_similarity_weight` is read from the request and `term_similarity_weight = 1 - vector_similarity_weight` (`dataset_api_service.py:958-960`; `search.py:629`).
- `top_k` is clamped to 1..2048 in dataset search (`dataset_api_service.py:960`).
- Chat uses `top_k` for candidate generation but `top_n` as final retrieval `page_size` (`dialog_service.py:717-728`).
- RAGFlow computes scores, stable-sorts by descending combined similarity, filters by threshold, and then slices the requested page (`search.py:687-704`).
- Returned chunks include term, vector, and combined similarity scores (`search.py:721-733`).
- Similarity threshold is effectively disabled only when vector weight is zero (`search.py:690-693`).

### Ranking weaknesses likely relevant to ERAGB

Confirmed failure signal: 106/203 scorable rows lacked the expected shard in raw top-20; 78/96 incorrect rows were in that bucket. Therefore, any later generation improvements will be capped unless raw retrieval improves.

Probable ranking issues:

- Exact values and identifiers can be buried when dense semantic similarity favors plausible but wrong neighbors.
- Term/vector score magnitudes may be poorly calibrated across query types and source types.
- Titles and source metadata are not visibly used as boosted fields in the benchmark harness; RAGFlow returns document title as `docnm_kwd`, but the expected shard/source title is not a structured query constraint.
- Source-type-specific fields are not weighted differently; a Slack timestamp, a GitHub PR title, and a Confluence section heading are all mostly flattened into chunk text.

### Real ranking improvements to test later

- **Rerank fixed candidates into fixed top-N:** use RAGFlow rerank path with a known rerank model and evaluate at fixed `top_n=8` before increasing top-N. Because upstream issue #15428 reports reranker score-scale mismatch across providers, compare candidate order/recall rather than raw reranker scores across vendors.
- **Rank-based hybrid fusion / RRF experiment:** compare current weighted-score fusion to RRF-like fusion for lexical and vector candidate lists. Azure and Elasticsearch docs support RRF for combining heterogeneous rankers without score calibration. Treat RRF as an experimental comparison, not documented current RAGFlow behavior.
- **Exact-token boosts:** boost code identifiers, metric names, numbers, dates, limits, file names, source titles, issue IDs, and quoted phrases.
- **Metadata boosts:** boost title/source/path matches and source-type-specific fields.
- **Query decomposition/expansion for multi-fact questions:** rewrite into exact subqueries, retrieve per subquery, then merge with diversity constraints.
- **Source-type-aware retrieval:** use separate field weights or retrieval profiles for Confluence, Slack/Fireflies, GitHub/Jira, and CRM/email.

## 10. Exact-Fact Extraction and Anti-Distractor Analysis

### Confirmed extraction failures

At least 13 incorrect scorable rows had the expected shard in final context. Examples:

- `qst_0021`: expected Confluence context present; answer gave 5 business days; gold says 3 business days.
- `qst_0102`: expected Confluence context present; answer said thresholds were not explicitly listed; gold contains p-value/sample-size guidelines.
- `qst_0195`: expected Google Drive context present; answer identified the right lab/shard but failed the requested exact cost value.
- `qst_0176`: expected Slack context present; answer gave an adjacent date but missed exact timestamp/date detail.

### Prompt and answer behavior

RAGFlow’s default system prompt in saved responses asks the model to “summarize the content of the dataset,” “list the data,” and answer in detail. That is not an exact-fact extraction prompt. It encourages broad summarization and can reward plausible synthesis when the benchmark asks for a precise number, date, role list, or threshold.

RAGFlow quote mode adds citation instructions (`dialog_service.py:780-824`), but citation insertion is not the same as evidence-first extraction. If the model answers without quoting the exact evidence first, the answer can cite a nearby chunk while still choosing the wrong value.

### Real synthesis improvements

- **Quote-first prompt:** require the model to quote the exact evidence span(s) before final answer.
- **Evidence table:** for each required fact, produce `fact_needed`, `source ID`, `quoted evidence`, `extracted value`, `confidence` before final answer.
- **Conflict-aware template:** if multiple conflicting values appear, list conflicts and choose only the one from the highest-ranked or exact-title-matched source.
- **Exact-value validation pass:** after draft answer, verify every number/date/entity/role against retrieved evidence; abstain if unsupported.
- **Source-constrained answer:** when benchmark expects a source, answer only from chunks with matching source/title metadata if available.
- **Multi-part completeness checklist:** decompose enumerated questions and ensure every requested slot is filled or explicitly marked not found.

These are real synthesis improvements if evaluated on rows where final context already contains gold evidence at fixed context size.

## 11. Confluence Deep Dive

### Failure footprint

Confluence is the largest weak source type in this scorable set: 54 source-type memberships, 7 correct, 19 partial, 28 incorrect.

Representative Confluence failures:

- `qst_0013`: raw expected shard present, final context absent. The answer described canary/cutover generally but missed the prescribed two-phase credential rotation details.
- `qst_0021`: final context present, wrong exact turnaround value.
- `qst_0023`: final context present, answer gave a broad docs summary but missed exact review roles/names.
- `qst_0102`: final context present, answer falsely said significance/sample-size defaults were not listed.
- `qst_0104`: raw present, final context absent; exact buddy time not retrieved into final context.
- `qst_0221`: expected shard absent from raw retrieval; answer not found.
- `qst_0234`: expected shard absent; answer gave generic rollout ramp schedules from distractors.

### Likely root causes

Confirmed/probable mix:

- **Retrieval miss:** many Confluence failures lack expected shard in raw top-20.
- **Section contamination:** Confluence pages can have multiple sections with similar operational language; naive chunking flattens section hierarchy.
- **Exact role/threshold extraction weakness:** rows ask for approval roles, default thresholds, liability language, and SLA values.
- **Title/path metadata weakness:** Confluence page titles and headings should be strong retrieval fields, not just inline chunk text.

### Confluence-specific improvements

- Preserve page title, space, path, and heading hierarchy as metadata fields.
- Chunk by heading/section with parent title included, not by blank-line/size alone.
- Boost title/heading matches for exact policy/SOP names.
- Add exact-value extraction prompts for SLA/time/threshold/role-list questions.
- For pages with repeated policy patterns, use source-title exact match and section-heading rerank before final top-N.

## 12. Slack and Fireflies Deep Dive

### Slack failures

Slack scorable footprint: 33 memberships, 6 correct, 11 partial, 16 incorrect. Representative rows:

- `qst_0043`: expected shard absent; answer used a plausible but wrong p99-latency cause.
- `qst_0099`: expected shard absent; not found for MTU/MSS mitigation.
- `qst_0100`: expected shard absent; answer gave rollback heuristic instead of disabling continuous batching/tightened limits.
- `qst_0176`: expected shard present in context but exact booking date/time was wrong/incomplete.
- `qst_0224`: raw expected shard present but final context absent; wrong identifier extracted.

### Fireflies failures

Fireflies scorable footprint: 11 memberships, 2 correct, 1 partial, 8 incorrect. Representative rows:

- `qst_0038`: expected shard absent; not found for median/p95 latency.
- `qst_0101`: expected shard absent; answer discussed unrelated companies and missed utilization threshold.
- `qst_0156`: expected shard absent; answer conflated health-score thresholds with error-budget burn rates.
- `qst_0450`: raw expected shard present but final context absent; aggregation count was partial/wrong.

### Conversational-source root causes

Probable:

- Speaker/time/thread boundaries are flattened into prose.
- Similar operational incidents and customer calls create high distractor density.
- Exact temporal facts and speaker-attributed statements require metadata-aware retrieval.
- Aggregation questions need multiple relevant records, not a single best chunk.

### Recommended conversational-source improvements

- Serialize Slack as `channel`, `thread_ts`, `message_ts`, `speaker`, `message`, and include thread parent context.
- Serialize Fireflies as `meeting_title`, `date`, `speaker`, `turn_index`, `utterance`; preserve agenda/summary if available.
- Chunk conversations by thread or meeting segment with overlap on speaker/time context.
- Add metadata filters/boosts for dates, customers, regions, and named participants.
- For count/aggregation questions, retrieve per subquery and deduplicate by original transcript/thread ID before answering.

## 13. Evaluation Harness Fixes

### Confirmed evaluation issues

1. **Source URI bug:** `source_recall()` compares expected sources to `retrieved_source_uris` (`retrieval_scoring.py:4-9`), but the runner only extracts `source_uri` from metadata/top-level fields (`benchmark_runner.py:139-146`). Saved RAGFlow chunks expose document names and IDs, not canonical source URIs.
2. **Failure type collapse:** `classify_failure()` labels every non-exact answer with `source_recall == 0.0` as `retrieval_failure` (`failure_classification.py:11-12`). Because recall is always zero, runner failure labels are mostly unusable.
3. **Judge exclusion rate:** `judge_summary.json` excludes 297/500 rows, including 198 judge errors. This can bias source-type and reasoning-type conclusions.
4. **Final-context metrics absent:** the harness stores raw response references, but summary metrics do not compute final context shard recall, citation recall, redundancy, or distractor rate.
5. **Reasoning/source-type summary mismatch:** `summary.json` reports runner exact-match by `reasoning_type`, but scorable judge analysis needs judge rows joined to source types and expected sources.

### Proposed robust metrics

Persist and summarize:

- `raw_retrieval_source_recall@k`
- `raw_retrieval_shard_recall@k`
- `raw_retrieval_expected_rank` / MRR
- `final_context_source_recall@top_n`
- `final_context_shard_recall@top_n`
- `cited_source_recall`
- `answer_correct_given_gold_context`
- `answer_correct_given_retrieved_context`
- `distractor_rate`: final chunks not matching expected source but high lexical/entity overlap
- `redundancy_rate`: near-duplicate final chunks or same source repeated
- `context_utilization`: fraction of final answer claims supported by retrieved chunks
- `source_type`, `reasoning_type`, and exclusion reason breakdowns for all 500 rows

### What to persist per row

- Canonical expected doc IDs, expected shard URIs, expected source type.
- For every raw retrieval chunk: RAGFlow chunk ID, document ID, document name, canonical mapped shard/source URI, rank, term/vector/combined score, metadata, source type.
- For every final context chunk: same fields plus prompt order and whether it survived token packing.
- Citation IDs mapped to final context chunks and canonical sources.
- Judge raw output, retry count, exclusion reason, and parse errors.
- A normalized “not found” flag and answer length/token count.

## 14. Ranked Improvement Backlog

| Priority | Improvement | Target failure mode | Real improvement or score inflation | Expected strict-accuracy impact | Complexity | Risk | Code/config area | Reingest? |
|---:|---|---|---|---|---|---|---|---|
| 1 | Fix source/shard mapping metrics and failure classification | Evaluation trust | Real evaluation-quality improvement | Indirect high | Medium | Low | `benchmark_runner.py`, scoring/reporting | No |
| 2 | Persist final-context/citation recall and per-row ranks | Evaluation trust and diagnosis | Real evaluation-quality improvement | Indirect high | Medium | Low | benchmark harness | No |
| 3 | Stabilize judge/retry accounting and reduce exclusions | Evaluation trust | Real evaluation-quality improvement | Indirect medium | Medium | Low | `judge.py`, retry workflow | No |
| 4 | Source-boundary-aware ERAGB ingestion with original doc metadata | Raw retrieval misses, distractors | Real retrieval + ingestion improvement | High | Medium-high | Medium | `eragb_prep.py`, ingestion config/pipeline | Yes |
| 5 | Confluence heading/title/path-aware serialization | Confluence retrieval/extraction | Real retrieval/context improvement | High for Confluence | Medium-high | Medium | prep/ingestion; RAGFlow metadata | Yes |
| 6 | Slack/Fireflies thread/speaker/time-aware serialization | Conversational retrieval | Real retrieval/context improvement | Medium-high for conversational | Medium-high | Medium | prep/ingestion | Yes |
| 7 | Fixed-candidate reranker into fixed top-8 | Final context quality | Real context-selection improvement | Medium-high | Medium | Medium latency | RAGFlow `rerank_id` path/chat config | No, if reranker available |
| 8 | RRF/rank-based hybrid fusion experiment | Score calibration | Real retrieval improvement | Medium | High | Medium | `rag/nlp/search.py` | No |
| 9 | Exact-token/title/number metadata boosts | Exact identifiers and values | Real retrieval improvement | Medium | Medium-high | Medium | search ranking/index fields | Maybe |
| 10 | Query decomposition for multi-fact questions | Partial/incomplete answers | Real retrieval + synthesis improvement | Medium | Medium | Medium | benchmark/RAGFlow prompt/retrieval layer | No |
| 11 | Quote-first/evidence-table answer prompt | Wrong exact value with context present | Real synthesis improvement | Medium | Low-medium | Low | chat prompt config | No |
| 12 | Numeric/entity validation pass against evidence | Wrong exact values | Real synthesis improvement | Medium | Medium | Low-medium | answer postprocessor or prompt | No |
| 13 | Increase `chat.top_n` only with fixed diagnostics | Raw-to-context misses | Mostly score inflation unless reranked/diversified | Low-medium | Low | High distractor/cost | config | No |
| 14 | Increase `max_tokens` only with fixed diagnostics | Prompt truncation/multi-fact answers | Mostly score inflation unless evidence quality controlled | Low-medium | Low | Medium cost | config | No |
| 15 | Random embedding/model swaps | Unknown | Not recommended; likely noisy | Unknown | Medium | High | config/reingest | Usually yes |
| 16 | GraphRAG/RAPTOR before simpler evidence | Sparse/high-level retrieval | Not first-line; may inflate/complicate | Unknown-low for exact facts | High | High provenance risk | RAGFlow config | Yes/Maybe |

## 15. Minimal Future Experiment Design

Run small, controlled slices before any full benchmark.

### Stage 1: evaluation-only repair

- Recompute historical June 26 metrics from existing artifacts using shard-name mapping.
- Produce per-row `raw_retrieval_shard_recall@20`, `final_context_shard_recall@8`, expected rank, and source type.
- Validate on at least the representative rows in this report.

### Stage 2: fixed-context diagnostic slices

Use 30-60 rows, stratified by bucket:

- expected absent from raw retrieval;
- raw present/final absent;
- final present/wrong;
- final present/partial;
- Confluence failures;
- Slack/Fireflies failures.

### Stage 3: controls that detect context inflation

For every proposed change compare:

- raw recall@20 and recall@8;
- final_context_recall@8 at fixed `top_n`;
- redundancy and distractor rate;
- strict accuracy split by gold-source absent/present in final context;
- source-type accuracy;
- answer correctness with fixed gold context if possible;
- answer correctness with fixed retrieved context if possible.

### Stage 4: experiment order

1. Evaluation repair on old artifacts.
2. Prompt-only exact extraction test on final-context-present wrong rows.
3. Reranker test at fixed `top_n=8`, `max_tokens` controlled.
4. Source-boundary reingestion test on one or two weak source types.
5. Only after those, parameter sweeps for `top_n`, `max_tokens`, threshold, and vector weight.

## 16. What Not To Do

- Do not treat `chat.top_n` increase as the solution. It may reveal context-selection weakness, but it is not a system-quality fix unless paired with better ranking/diversity/evidence selection.
- Do not blindly increase `max_tokens`. It may hide poor retrieval by letting more distractors through.
- Do not run a new full benchmark before source recall, final-context recall, and judge exclusions are fixed.
- Do not randomly swap embeddings or answer models before measuring where the pipeline fails.
- Do not jump to GraphRAG/RAPTOR before simpler source-boundary, reranking, and exact-extraction evidence.
- Do not optimize only partial-or-better; strict correctness is the target for exact-fact EnterpriseRAG-Bench.
- Do not overfit to the 203 scorable rows; 297 exclusions mean the analyzed set is likely biased.
- Do not use source filename matching as the only long-term truth; it is a recovery metric for this run, not a robust source mapping design.

## 17. Final Recommendation

Recommended sequence for the later implementation/planning phase:

1. **Evaluation fixes first.** Repair source/shard mapping, final-context recall, citation recall, and failure classification. Recompute historical June 26 diagnostics from existing artifacts.
2. **Ingestion/source-boundary fixes second.** Reingest a small source-type slice with first-class original doc metadata and source-specific serialization. Start with Confluence plus Slack/Fireflies.
3. **Final-context reranking/diversity fixes third.** Test reranking and/or rank-based hybrid fusion at fixed `top_n` before expanding context.
4. **Exact-fact extraction fixes fourth.** Add quote-first/evidence-table/conflict-aware prompt variants and numeric/entity validation, targeting rows where gold context is already present.
5. **Parameter sweeps last.** Only sweep `top_n`, `max_tokens`, similarity threshold, and vector weight after diagnostics can distinguish real quality from context inflation.

## Confirmed Facts, Probable Hypotheses, Open Questions

### Confirmed facts

- 203/500 rows are scorable; strict accuracy is 33.0%; partial-or-better is 52.7%.
- Source recall is broken for this run because expected canonical sources are compared to empty `retrieved_source_uris`.
- 78/96 incorrect scorable rows lacked expected shard in raw top-20 by shard-basename matching.
- 13 incorrect scorable rows had expected shard in final context and still answered wrong.
- RAGFlow chat uses `top_n` as retrieval `page_size` and then token-packs chunks via `kb_prompt()`.
- RAGFlow supports an optional reranker path, but this run did not configure one.

### Probable hypotheses

- Merged shard ingestion and naive chunking weaken original source boundaries.
- Confluence failures are driven by a mix of retrieval miss, section hierarchy loss, and exact-value extraction failure.
- Slack/Fireflies failures are driven by flattened speaker/time/thread context and high distractor density.
- RRF/rank-based fusion or reranking may improve fixed-budget final context quality more than increasing context size.

### Open questions

- What exact RAGFlow version/commit and embedding model were used to build the reused dataset?
- Was any parser/custom delimiter different at actual ingestion time from the resolved config?
- How many excluded judge/infra rows would be scorable after retry stabilization, and do they shift source-type conclusions?
- Does RAGFlow’s current index contain structured metadata for source type/title/original doc ID, or only inline text plus document name?

## Checklist of Next Actions

- [ ] Repair harness source/shard mapping and failure labels.
- [ ] Recompute historical diagnostics from the June 26 artifacts without a new full benchmark.
- [ ] Persist raw retrieval rank, final context rank, citation mapping, and source type per row.
- [ ] Build a 30-60 row diagnostic slice by failure bucket and source type.
- [ ] Test quote-first/evidence-table prompting on final-context-present wrong rows.
- [ ] Test reranker at fixed `top_n=8` and fixed prompt budget.
- [ ] Design source-boundary-aware Confluence and Slack/Fireflies ingestion slices.
- [ ] Only after the above, run controlled parameter sweeps.
- [ ] Feed this report into a later `$ralplan` before any implementation or configuration changes.
- [ ] Do not present any implementation as complete until a later implementation workflow runs and validates it.
