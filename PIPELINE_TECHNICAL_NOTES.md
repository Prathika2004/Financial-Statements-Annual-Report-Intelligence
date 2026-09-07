# Pipeline Technical Notes

This note captures the technical issues discovered while running Stage 1
ingestion (PDF parse -> clean -> metadata and section extraction).

## Current status

- Stages 1-6 (parse -> clean -> extract metadata -> chunk -> embed -> load
  into Qdrant) are complete and verified across the full dataset: 50 PDFs,
  10 companies (Amazon, Apple, Berkshire Hathaway, Coca-Cola, Intel,
  Johnson & Johnson, Microsoft, Nvidia, Tesla, Visa), 5 fiscal years each.
- Verified invariants across all 50 `metadata.json` / `chunks.json` pairs:
  zero null identity fields, zero leaked Markdown artifacts in
  `company_name`, exactly 50 distinct (ticker, fiscal_year) combinations with
  no collisions, zero duplicate `chunk_id`s, every chunk has a
  `parent_section_id` + offsets. 29,887 total chunks (12,088 table, 17,799
  text).
- Stage 5 (`src/embeddings/embedder.py`, `BAAI/bge-small-en-v1.5` via an
  OpenVINO-backed CPU inference path, cached locally after first export):
  29,887/29,887 chunks embedded, all 384-dim, all L2-normalized, row order
  verified against `chunks.json` for every filing. Took ~73 minutes
  wall-clock for the full batch on CPU -- a one-time cost.
- Stage 6 (`src/vectorstore/qdrant_client.py`, Qdrant running locally via
  `docker/docker-compose.yml`): all 29,887 points uploaded to the
  `sec_10k_filings` collection. Verified: point count matches chunk count
  exactly, a semantic query for "risk factors" correctly retrieves Item 1A
  chunks, payload filtering by (ticker, fiscal_year) returns exact counts
  matching the source `chunks.json`, and re-running the upload script is
  idempotent (deterministic UUID5 point ids from chunk_id -- see
  `qdrant_client.py`'s module docstring).
- Stage 7 (`src/retrieval/`: `dense_retriever.py`, `bm25_retriever.py`,
  `hybrid_fusion.py`, `retriever.py`): dense search via Qdrant, BM25 sparse
  search (index built in-process from all 29,887 chunks, ~4.3s), and
  Reciprocal Rank Fusion to merge them. Verified end-to-end with
  `scripts/run_retrieval.py`: a cybersecurity-related query correctly
  surfaces Item 1C sections across six different companies, all found by
  both dense and BM25; metadata filtering by (ticker, fiscal_year) returns
  only matching-year Tesla chunks for a revenue-growth query.
  `retrieve()`'s signature takes a list of query strings (not a single
  string) specifically so Query Transformation (multi-query, decomposition
  -- not built yet) can generate several query variants and pass them
  straight through without this function's signature changing; see its
  module docstring for how a HyDE-style strategy should bypass it and call
  `dense_search()` directly with a pre-computed vector instead.
- Qdrant's collection already uses HNSW indexing (Module 12-style ANN)
  automatically -- confirmed via `GET /collections/sec_10k_filings`
  (`hnsw_config: {m: 16, ef_construct: 100}`, Qdrant's defaults). No
  separate indexing work was needed or done; at ~30k vectors this dataset
  is far below the scale where HNSW tuning or an alternative (IVF+PQ,
  DiskANN) would matter.
- Stage 8 (`src/reranking/flashrank_reranker.py`): cross-encoder reranking
  via FlashRank's `ms-marco-MiniLM-L-12-v2` (12-layer, INT8-quantized ONNX;
  chosen over FlashRank's other zoo models -- see the module's docstring
  for the full comparison). `retriever.retrieve_and_rerank()` composes
  Stage 7 + 8: hybrid-retrieves a wider RERANK_CANDIDATE_K=25 pool, then
  reranks down to RERANK_TOP_K=5. Verified end-to-end via
  `scripts/run_reranking.py`: reranking correctly promoted a Tesla Item 1C
  chunk (found only by dense search, low pre-rerank rrf_score) to a
  near-top rerank_score of 0.996, ahead of several chunks that ranked
  higher pre-rerank -- a real instance of reranking fixing an
  under-ranked-but-relevant result, not just a theoretical benefit.
- Query Transformation (`src/query_transformation/`: `query_rewriting.py`,
  `multi_query.py`, `decomposition.py`, `hyde.py`) implemented as pure
  logic against a swappable `llm_generate: Callable[[str], str]` parameter
  (see `llm_interface.py`) -- deliberately built ahead of Stage 9 (no LLM
  provider chosen yet) so the parsing/fallback logic is tested now via
  `FakeLLM`, and Stage 9 only needs to supply a real callable matching that
  signature, with zero changes to this package. `generate_multi_queries()`
  and `decompose_query()` both return `List[str]`, usable directly as
  `retriever.retrieve()`'s `queries` parameter -- confirming the
  extensibility that parameter was built for during Stage 7.
  `hyde_search()` instead bypasses `retrieve()` and calls
  `dense_retriever.dense_search()` directly with a hypothetical-document
  vector (embedded with `is_query=False`, never the BGE query prefix).
- Stage 9 (`src/generation/`: `llm_router.py`, `prompt_templates.py`,
  `numeric_verifier.py`, `rag_pipeline.py`): generation via Qwen2.5 7B
  served locally by Ollama (already installed/running on this machine;
  model pulled mid-session, 4.7GB at Ollama's default quantization).
  `llm_router.generate()`/`chat()` implement query_transformation's
  `LLMGenerate` contract for real, over Ollama's `/api/chat`.
  `prompt_templates.py` builds numbered, source-labeled prompts
  (`[Source N: TICKER FY20XX, Item X]`) with an explicit
  say-"I don't know" instruction as the primary anti-hallucination
  defense; `numeric_verifier.py` is the secondary, structural one --
  it flags any number in the generated answer that doesn't appear
  anywhere in the cited sources' text.
- Verified end-to-end via `scripts/run_generation.py` (full pipeline:
  ingest -> chunk -> embed -> Qdrant -> hybrid retrieve -> rerank ->
  prompt -> Qwen2.5 7B -> numeric check) on a real cybersecurity-risk
  question: the answer correctly cited 5 sources by their `[Source N]`
  label (matching KO/TSLA/INTC Item 1C sections, the same set Stage 8's
  reranking test surfaced), and `numeric_check.unverified` was empty
  (no fabricated figures -- this particular answer had no numeric claims
  to check, so this run validates the clean-pass path; the flagging path
  itself is covered by `test_numeric_verifier.py`).
- Bug caught during this test: `llm_router.py`'s original 120s read
  timeout was too short for CPU inference on a 7B model processing a full
  prompt (system instructions + 5 reranked source chunks, 1500-3000+
  tokens) and generating a multi-paragraph answer -- the very first real
  end-to-end run timed out. Raised to 300s.
- A single question through the full pipeline takes on the order of a few
  minutes end-to-end on this CPU-only machine, dominated by Ollama
  generation time (retrieval + reranking together are well under a
  second, per Stages 7-8's own timing notes) -- expected and consistent
  with running a 7B model without a GPU; not itself a bug, but worth
  knowing before treating a still-running request as hung.
- 75/75 unit tests pass.
- Query Transformation re-verified against the real LLM (Qwen2.5 7B) via
  `scripts/run_query_transformation.py`, now that one exists -- it had only
  ever been tested against FakeLLM's canned responses. 4 of 5 techniques
  worked correctly on the first real run: `rewrite_query` reproduced the
  exact "Explain AI" -> "Explain artificial intelligence" example from the
  original spec; `generate_multi_queries` produced 3 genuinely distinct,
  cleanly-parsed phrasings; `decompose_query` correctly split a compound
  question into 2 independent sub-questions; `hyde_search` worked fully
  end-to-end (generated a plausible hypothetical 10-K passage, embedded it
  as a passage per `is_query=False`, and dense search returned genuinely
  relevant real filings -- Intel's Cybersecurity section, J&J's Item 1A).
  `step_back_query` was the exception: the code path was correct (LLM
  called, output parsed, no fallback needed) but the model just reworded
  the question instead of generalizing it ("What was Tesla's Q3 2023
  revenue?" -> "What was Tesla's revenue for the third quarter of 2023?").
  Fixed by adding a one-shot example to `STEP_BACK_PROMPT_TEMPLATE`
  (`query_rewriting.py`) demonstrating the specific->general transformation
  concretely; re-tested and it now correctly produces "How has Tesla's
  revenue trended over time?". A reminder that logic tested only against a
  canned FakeLLM validates control flow, not prompt quality -- those are
  separate concerns and both need checking once a real model exists.
- Stage 10 (`src/api/`: `main.py`, `routes.py`, `schemas.py`): FastAPI
  server exposing `GET /health`, `POST /ask` (blocking), and
  `POST /ask/stream` (Server-Sent Events). Everything expensive (embedding
  model, Qdrant client, 29,887-chunk BM25 index, reranker) loads once at
  startup via a `lifespan` context, not per-request -- same
  load-once-reuse pattern as `RetrievalContext` (Stage 7). Route handlers
  are plain `def`, not `async def`, since every call inside them (Qdrant,
  BM25, Ollama) is synchronous/blocking -- FastAPI runs plain `def` routes
  in a threadpool automatically, and Starlette does the same for a plain
  generator passed to `StreamingResponse`, so nothing here blocks the
  server's event loop.
- `/ask/stream`'s two-endpoint split exists because `numeric_verifier.py`
  needs the *complete* answer text to check -- it can't run mid-stream.
  Tokens stream as they're generated; sources + the numeric check are sent
  as one final SSE event once the full text is assembled server-side.
- Verified for real, not just "it returned 200": `GET /health` correctly
  reports live Qdrant/Ollama reachability; `POST /ask/stream` was checked
  with a small timestamped Python client (not just curl, since a shell
  redirect's own buffering can make a truly-streamed response look like it
  arrived all at once) -- confirmed genuine incremental delivery: first
  token at 65s (retrieval + reranking + Ollama's prompt-evaluation time for
  ~5 chunks of context), then subsequent tokens ~0.17-0.18s apart, not a
  buffered dump at the end. The final event's `sources`/`numeric_check`
  payload matched the same shape and content `/ask`'s blocking response
  and `scripts/run_generation.py`'s CLI output already produce.
- 78/78 unit tests pass. Deferred as explicit follow-ups (not forgotten):
  a query-transformation toggle on `/ask` (wire `generate_multi_queries`/
  `hyde_search`/etc. in as an opt-in per-request technique) and
  ticker/fiscal-year listing endpoints for a future frontend's dropdowns.
- Stages 1-10 are now complete and independently verified end-to-end --
  ingest -> clean -> extract -> chunk -> embed -> store -> retrieve ->
  rerank -> transform queries -> generate -> verify -> serve over HTTP.
  Remaining unbuilt: RAG evaluation (`eval/`, RAGAS is already a
  dependency), guardrails, and the other Module 18-28 items analyzed in
  this session but deliberately not yet implemented.

## 15. Model weight files: where they live and what runs them

Asked mid-project; worth keeping current as more stages add models.

| Model | Location | Format | Inference engine |
|---|---|---|---|
| Embedding (`bge-small-en-v1.5`) | `sec-rag-project/models/BAAI__bge-small-en-v1.5-openvino/openvino/` | OpenVINO IR | OpenVINO (`optimum-intel`) |
| Reranker (`ms-marco-MiniLM-L-12-v2`) | `sec-rag-project/models/flashrank/ms-marco-MiniLM-L-12-v2/` | ONNX (pre-quantized by FlashRank) | ONNX Runtime, `CPUExecutionProvider` (confirmed via `ort.get_available_providers()` -- no GPU/OpenVINO provider is compiled into the installed `onnxruntime` package, so FlashRank always runs on plain CPU regardless of OpenVINO being installed for the embedder) |

Both are project-local under `models/` (gitignored). Separately, there is
also a machine-wide Hugging Face cache at `~/.cache/huggingface/hub/`
(55GB, shared across every project ever run on this machine) which holds
`bge-small-en-v1.5`'s original 129MB PyTorch download (from before
`optimum-intel` converted it to OpenVINO) -- that copy is unused at
runtime; only `models/` is ever read by this project's code.

## 14b. Company name as a cover-page logo image (Apple, Intel, Visa)

### What happened

`GET /filings` (Stage 10 follow-up #3) surfaced `company_name:
"<!-- image -->"` for 9 of 50 filings (Apple x4, Intel x2, Visa x3). These
filers render their cover-page company name as a stylized logo image
rather than plain text; Docling represents an image it can't extract text
from as an HTML comment placeholder in the Markdown, and
`extract_company_name_from_text()`'s backward-scan from the "(Exact name
of Registrant...)" anchor accepted that placeholder verbatim -- it passed
every existing validation (non-empty, >2 chars, doesn't start with "(")
because none of those checks anticipated an HTML comment.

### Resolution

Added `IMAGE_PLACEHOLDER_PATTERN = re.compile(r"^<!--.*-->$")` and reject
any candidate matching it, in `metadata_extractor.py`. Confirmed via the
actual line layout (Apple 2021: `<!-- image -->` sits 2 lines above the
anchor with only blank lines otherwise nearby) that the real name
genuinely isn't present as text within the existing 3-line backward search
window -- rejecting the placeholder correctly makes the function return
`None`, which the existing `run_ingestion.py`/`backfill_identity.py`
fallback (`identity.company_name or registry_info["company_name"]`)
already handles by using the EDGAR-registry name instead. No new fallback
logic was needed, just removing the bad acceptance.

### An unexpected cascade this fix exposed (and how it was handled)

Fixing `company_name` and re-running `backfill_identity.py` +
`run_chunking.py` changed the total chunk count from 29,887 to 29,886 --
one fewer, not zero difference. Root cause: `table_chunker.py`'s fallback
caption (used when no natural preceding sentence exists) is
`f"Table from {section_item} ({section_title}), {company_name} FY{fiscal_year}"`,
and `_split_table_by_rows()`'s per-part character budget is computed as
`MAX_TABLE_CHUNK_CHARS - len(caption) - 30` -- so the *length* of
`company_name` directly affects how many row-split parts an oversized
table needs. Confirmed concretely: Intel's Heading 231 "Key Terms" table's
caption changed from embedding `<!-- image -->` (13 chars) to `INTEL CORP`
(10 chars), shifting its part count. This is correct, expected behavior,
not a regression -- but it meant the fix couldn't stop at re-chunking:

1. The table chunks using the fallback caption now have different *text*
   (the caption is prepended into the chunk's embedded text) -- their
   embeddings were stale. Re-ran `scripts/run_embedding.py --folder X` for
   each of the 9 affected folders (not all 50 -- targeted, since
   re-embedding is the expensive step).
2. `chunk_id` is a per-document sequential counter (see
   `section_chunker.py`), so every chunk after a table whose part count
   changed gets a different chunk_id than before -- and therefore a
   different Qdrant point id (`chunk_id_to_point_id()`'s UUID5 is
   deterministic *from* chunk_id). Upserting new chunks alone never
   deletes a point sitting at an old chunk_id the new output no longer
   produces, which would have left silent orphaned stale points in Qdrant.
   Added `qdrant_client.delete_by_source_filename()` and wired
   `run_qdrant_upload.py` to delete each document's existing points before
   upserting it, unconditionally -- not just for this one-off fix, so any
   future re-chunk-triggered renumbering is handled automatically instead
   of needing another manual audit like this one.

Verified clean afterward: Qdrant's point count (29,886) exactly matches
the corrected total chunk count with no leftover orphans, and a live
`scroll()` query against Qdrant confirms `company_name: "Apple Inc."` in
the actual served payload, not just in the source JSON files.

### Takeaway

A metadata field that looks like free text (company_name) fed into a
*derived, size-sensitive* computation (a caption's length affecting a
table-splitting budget) elsewhere in the pipeline -- fixing the field at
its source doesn't stay contained to that one field. Re-running "the next
stage down" isn't automatically sufficient after a Stage 1-3 metadata fix;
check whether anything downstream's behavior depends on the *value*, not
just the *presence*, of what changed.

## 14c. RAGAS evaluation: judge model swap, and a lost-progress bug

### Judge model: gpt-oss:20b was abandoned, Llama 3.1 8B used instead

`gpt-oss:20b` (already pulled, unused elsewhere) was the first choice for
RAGAS's independent judge -- different model than Qwen2.5 (the answering
model), so the pipeline isn't graded by the same model that answered. It
failed with `tensor "blk.0.ffn_down_exps.weight" size overflow`,
reproduced via a raw `curl` straight to Ollama (not our code, not RAGAS).
A full `ollama rm` + fresh 13GB re-pull did not fix it, and got stuck for
over 40 minutes in what should have been a sub-minute checksum-verify step
after the download itself finished -- abandoned per the user's own call
after a bounded extra wait. `gpt-oss:120b` was not tried as a fallback: it
shares the same MoE (`ffn_down_exps`) tensor layout, so it's a likely
repeat of the same bug, not a safe alternative. Switched to **Llama 3.1
8B** (dense, non-MoE architecture, different lineage from Qwen2.5) --
pulled cleanly, verified working via an isolated Faithfulness check before
committing to a real run. `RAGAS_JUDGE_MODEL_NAME` in `config/settings.py`
carries the full reasoning.

### A real bug: batching the results write until the end lost 4 completed evaluations

The first full 8-question run completed 4 questions successfully (real
RAGAS scores each time) then crashed on question 5 -- Llama 3.1 8B
returned malformed structured output for the `AnswerRelevancy` metric's
internal prompt (a Pydantic validation error, `instructor` retries
exhausted). `run_ragas.py` only wrote `ragas_results.json` once, after the
entire loop finished -- so the crash didn't just fail question 5, it threw
away all 4 already-completed, already-paid-for results (each took
6-10 minutes; nothing about this pipeline is cheap enough to lose casually).

Fixed with two changes: (1) each question's pipeline+scoring is wrapped in
try/except -- a single question's judge-model hiccup is recorded as a
failure entry and the run continues, instead of taking the whole batch
down; (2) results are written to disk after every question, not once at
the end. Added `--skip N` to resume a partial run without re-scoring (and
re-paying the LLM-call cost for) questions already completed -- the 4
already-done results were reconstructed from the crashed run's log output
and used to resume from question 5 rather than restarting from 1.

One casualty of the reconstruction: the log only ever printed each
question's *scores*, never its generated *answer* text, so 3 of the 4
recovered entries have `"answer": null` -- genuinely lost, not recoverable
without re-running them. `jnj_worldwide_sales_fy2023` also scored
`faithfulness: 0.0`, a stronger signal than the citation-sentence artifact
seen elsewhere (see the eval summary for what a 0.5-ish score usually
means) -- worth a closer look once its answer text exists again, e.g. by
re-running that one question specifically.

## 14d. Full diagnostic of the first real 8-question eval run

The resumed run finished: 5 scored, 3 failed. All 5 problems (2 low
faithfulness scores + 3 outright failures) were root-caused individually
rather than treated as one fuzzy "eval is flaky" bucket.

### faithfulness=0.0 on multi-company/synthesized answers (root-caused, fixed)

`cybersecurity_risk_factors_general`'s answer correctly attributed claims
to companies by name ("Microsoft and Tesla mention..." / "NVIDIA also
notes...") using the `[Source N]` labels the generation prompt provides --
but `run_ragas.py` passed RAGAS's Faithfulness judge only the *bare chunk
text*, no labels. Reproduced the exact retrieval for this question and
confirmed directly: none of the actual Microsoft/Tesla/NVIDIA chunks
mention that company's name anywhere in their raw text (company identity
is metadata, not repeated inline in the prose). A judge with no company
label in its context literally cannot verify "Microsoft mentions X" is
supported -- it isn't lying about faithfulness, it's being handed less
information than the answering model had. Almost certainly the same
mechanism behind `jnj_worldwide_sales_fy2023`'s faithfulness=0.0, though
that answer's text was lost in the earlier crash and couldn't be directly
confirmed. Fixed: `run_pipeline()` now labels contexts for RAGAS the same
way `build_context_block()` labels them for generation (`format_source_label()`
prefix on each chunk), so the judge sees what the answering model saw.

### faithfulness=0.5 on single-fact answers (root-caused, fixed)

`tesla_net_income_fy2023` answered correctly but appended a standalone
sentence, "This information is cited from [Source 1]." -- Faithfulness
decomposes the answer into statements and checks each independently; that
sentence isn't a claim from the source at all, so it fails verification
regardless of how correct the actual answer was, costing exactly 1 of 2
statements (0.5). Fixed: tightened `SYSTEM_PROMPT` rule 2 to require the
citation label attached to the sentence containing the claim, explicitly
forbidding a separate citation-only sentence.

### 2 of 3 failures: Llama 3.1 8B structured-output flakiness (documented, not "fixed")

`visa_cybersecurity_approach` and `unanswerable_out_of_dataset_company`
both failed with `2 validation errors for AnswerRelevanceOutput` --
`instructor` (the structured-output library RAGAS's `llm_factory` uses)
couldn't coerce Llama 3.1 8B's response into the required
`{question, noncommittal}` JSON shape, even after its own internal
retries. Notably, `cybersecurity_risk_factors_general` hit this exact
error in the *first* run attempt but succeeded on this run -- same
question, same model, different outcome, confirming this is inherent
sampling flakiness in the judge model's instruction-following for
structured JSON, not a deterministic bug reachable by a code fix. Smaller
local models are measurably less reliable at strict schema-constrained
output than large hosted ones; this is a real, accepted limitation of
using an 8B local model as judge, not something eliminated here.

### 1 of 3 failures: RAGAS can't score a deliberately-empty context (root-caused, fixed by redesigning the check, not chasing the metric)

`unanswerable_wrong_fiscal_year` failed differently: `retrieved_contexts is
missing`. This question deliberately filters to `fiscal_year=2030`
(doesn't exist in the dataset) specifically so retrieval returns *nothing*
-- exactly the scenario meant to prove the system says "I don't know"
instead of hallucinating. But RAGAS's context-based metrics require
non-empty `retrieved_contexts` and raise rather than degrade gracefully,
and conceptually they don't even fit here: Context Precision/Recall
compare retrieved context against a reference, but a refusal question's
"reference" describes an absence, not real content to match. Trying to
force this question through the same 4 metrics as answerable questions was
the wrong tool, not a bug to patch around. Fixed by adding a dedicated
path (`score_refusal_question()`/`check_refusal()`) for any
`expect_refusal` question: skip RAGAS's context metrics entirely, check
the answer directly for refusal language instead. Simpler, doesn't crash
on empty context, and actually measures the thing this question class is
for.

### Re-verified with a targeted 3-question re-run, not a full re-run

Added `--ids` (comma-separated question ids, merges into the existing
results file by id rather than requiring a full re-run) specifically to
re-check just the 3 questions these fixes touched, at a fraction of the
~1.5-2 hour full-set cost. Result: 2 of 3 confirmed cleanly --
`cybersecurity_risk_factors_general` faithfulness rose from 0.0 to 0.625
(the labeled-context fix), and `unanswerable_wrong_fiscal_year` completed
with `refused=True` instead of crashing (the refusal-path fix).

`tesla_net_income_fy2023` initially looked like a regression --
faithfulness dropped from 0.5 to **0.0** even though the answer text
confirmed the citation fix worked exactly as intended ("...was $15.00
billion [Source 1]." -- one clean sentence, no separate citation-only
sentence like before). Investigated by re-scoring the *exact same*
answer + context through Faithfulness in isolation: it returned **1.0**.
Identical input, identical metric, identical judge model, different
result. This confirms the 0.0 was pure judge non-determinism, not a real
problem with the fix -- and extends the "Llama 3.1 8B is flaky as a local
judge" finding from structured-output parsing (already documented above)
to the core Faithfulness NLI judgment itself. Practical implication for
reading any future result from this harness: a single score from this
judge setup is a noisy point estimate, not a precise measurement --
treat one low score as a prompt to re-check before treating it as ground
truth, the same way this one was.

## 15. Guardrails, caching, observability, confidence gating, and API auth (Modules 19/21/22/25/26)

### What was built

- **Confidence gating** (`rag_pipeline.check_confidence()`, Module 19): if
  the top reranked score is below `CONFIDENCE_THRESHOLD` (0.3, a
  conservative starting point -- see its comment in `config/settings.py`),
  the pipeline refuses to generate rather than force an answer from weak
  context. Costs nothing extra (the score already exists from reranking);
  short-circuits before the expensive generation call.
- **Input guardrails** (`src/guardrails/input_guardrails.py`, Module 21):
  heuristic prompt-injection pattern matching + PII detection on the
  question. Injection blocks the request; PII is flagged but not blocking
  (a user's own question containing an email isn't an attack). Explicitly
  NOT a robust defense against a determined attacker -- see the module's
  own honesty note about what pattern-matching does and doesn't catch.
- **Output guardrails** (`src/guardrails/output_guardrails.py`, Module 19/21):
  citation enforcement -- flags an answer that cites no source at all
  (unless it's a correct refusal) or cites a source number that was never
  given (a fabricated citation, structurally the same failure mode
  `numeric_verifier.py` already catches for fabricated numbers).
- **Response cache** (`src/caching/response_cache.py`, Module 25): SQLite,
  not Redis -- a single-user local tool doesn't need a cache server, and
  it would be one more process that can silently stop running (see note
  #14). Keyed by (question, filters, technique). Verified for real: a
  repeat question that took 208s the first time returned in 0.006s cached.
- **Observability** (`src/observability/tracer.py`, Module 26): one JSON
  line per request (question, filters, technique, status, scores, elapsed
  time) to a local file -- not LangSmith/Phoenix/OpenTelemetry, which are
  built for a team debugging a deployed multi-user service this project
  isn't.
- **API auth** (`src/api/auth.py`, Module 22): an optional
  `Authorization: Bearer <key>` check on `/ask`/`/ask/stream`, off by
  default (`SEC_RAG_API_KEY` unset). Full OAuth/JWT/RBAC would solve a
  multi-user problem this project doesn't have; this is the one piece of
  Module 22 that's actually proportionate for "local tool that might one
  day be reachable from somewhere other than localhost."

### Architecture fix that came free with this

`src/api/routes.py` had drifted into reimplementing the pipeline inline
(retrieve -> prompt -> generate -> verify) rather than calling
`rag_pipeline.answer_question()`, which existed but had gone unused once
the `technique` parameter was added directly to the routes. Fixed by
splitting `rag_pipeline.py` into `screen_and_prepare()` (guardrails, cache,
retrieval, confidence gate -- no LLM call) and `finalize_answer()`
(numeric + citation checks, cache write -- runs after a complete answer
string exists), so `/ask` (blocking) and `/ask/stream` (SSE) now share
every cross-cutting concern exactly once instead of each needing its own
copy. `answer_question()` is the blocking convenience wrapper composing
both, for scripts/tests that just want one call.

### A real bug this surfaced: numpy float32 isn't JSON-serializable

First live test of the confidence-gate path crashed with `TypeError:
Object of type float32 is not JSON serializable`. Root cause:
`flashrank_reranker.rerank()` assigns `passage["score"]` directly from
FlashRank's sigmoid/softmax output -- a numpy float32, not a native Python
float. This had been silently working through every Pydantic-validated API
response (`SourceOut.score: float` coerces it automatically) since Stage 8,
so it was never visible until a *raw* `json.dumps()` call -- the new
tracer, and the streaming endpoint's inline SSE events -- hit it with no
such coercion. Fixed at the source: `rerank()` now casts every score to
`float(...)` before returning, so every downstream consumer (API
responses, tracing, streaming, tests, the UI) always gets a native float,
rather than requiring every JSON-serialization call site to defensively
handle numpy types. A regression test (`test_rerank_score_is_a_native_float_not_a_numpy_scalar`)
uses a fake ranker that specifically returns `np.float32` (the existing
fake used a plain Python float and would never have caught this) and
asserts `json.dumps()` succeeds on the result.

### Deliberately not built (Module 22's other pieces)

OAuth/JWT/SSO/RBAC/ABAC, encryption-at-rest, Key Vault/KMS/Vault secrets
management -- all solve problems a localhost-only, single-user, no-cloud-secrets
project doesn't have. Building them would be security theater, not
security. Revisit if this ever becomes a multi-user or network-exposed
deployment.

## 14. Docker Desktop does not stay running between sessions

### What happened

Mid-session, a previously-working Qdrant connection started failing with
`ConnectError: [WinError 10061] No connection could be made because the
target machine actively refused it`. `docker ps` then failed too, with the
same "cannot find dockerDesktopLinuxEngine pipe" error seen before Docker
Desktop was ever started -- the whole Docker Desktop application had
stopped, not just the container.

### Resolution

Relaunching `Docker Desktop.exe` and polling `docker ps` until it responds
brings the daemon back; `docker compose up -d` from `docker/` then restarts
the `sec_rag_qdrant` container. Data was not lost -- `qdrant_storage` is a
named Docker volume, independent of the container's lifecycle, and
`points_count` read back as 29887 immediately after the restart.

### Takeaway

Any stage that depends on Qdrant being reachable (retrieval, and later
generation/API serving) should check for this failure mode specifically
rather than treating a connection error as fatal -- Docker Desktop on this
machine is not guaranteed to still be running from an earlier point in the
same working session.

## 1. Two virtual environments caused package confusion

### What happened

There are two environments:

| Location | Docling |
| --- | --- |
| `C:\\Users\\Prathika\\Documents\\SEC\\.venv` | Not installed |
| `C:\\Users\\Prathika\\Documents\\SEC\\sec-rag-project\\.venv` | Installed |

Earlier commands used the workspace-level environment. That caused the parser
to log `Docling not installed` and fall back to `pdfplumber`.

### Resolution

Activate the project environment before running project commands:

```powershell
C:\\Users\\Prathika\\Documents\\SEC\\sec-rag-project\\.venv\\Scripts\\Activate.ps1
```

### Prevention

Configure the IDE interpreter to:

```text
sec-rag-project/.venv/Scripts/python.exe
```

## 2. Windows default encoding could not write SEC characters

### What happened

The original write of `raw_docling.md` used the Windows default `cp1252`
encoding. SEC cover pages contain Unicode symbols such as `☒`, causing:

```text
UnicodeEncodeError: 'charmap' codec can't encode character '\\u2612'
```

### Resolution

All text and JSON outputs in `scripts/run_ingestion.py` now explicitly use
UTF-8 (`encoding="utf-8"` and `ensure_ascii=False`).

## 3. Docling failed before document export

### What happened

Docling initially raised a Pydantic circular-reference serialization error:

```text
PydanticSerializationError: Circular reference detected (id repeated)
```

The trace showed this happened inside `DocumentConverter.convert()` while
Docling serialized pipeline options to create an internal pipeline-cache key.
It was therefore not initially a `document.export_to_dict()` error.

### Resolution

The old fixed `pydantic==2.9.2` requirement was relaxed and the project virtual
environment was upgraded to a supported newer Pydantic `2.x` version. A
follow-up single-file run initialized the Docling PDF pipeline, converted the
document, and wrote both Markdown and structured JSON.

### Safeguard retained

`parse_with_docling()` still treats `document.export_to_dict()` as optional.
If only the structured JSON export fails for a particular PDF, ingestion keeps
the successful Docling Markdown instead of needlessly switching to
`pdfplumber`.

## 4. Section detection mistook a narrative reference for a heading

### What happened

In Amazon 2023--2025 filings, the extractor treated this kind of prose as a
new section:

```text
Item 7 of Part II, "Management's Discussion ..."
```

This split `Item 1A - Risk Factors` too early and assigned much of its text to
a false `Item 7` section.

### Root cause

The matcher gave an uppercase `ITEM` keyword a strong body-heading score. It
did not verify that the text after the item number resembled the expected SEC
section title.

### Resolution

`metadata_extractor.py` now checks whether the candidate title begins with the
canonical title for that SEC Item. This distinguishes:

- a real heading: `Item 7. Management's Discussion ...`;
- a table-of-contents entry: the same title with a trailing page number; and
- a prose reference: `Item 7 of Part II ...`.

The metadata unit tests pass after this change. The regenerated full batch
should be reviewed for section counts and boundary quality across several
companies, not only Amazon.

## 5. Dependency resolution is slow and needs tighter version management

### What happened

`pip install -r requirements.txt` backtracked through several LangChain
versions because multiple packages were unpinned. `pip check` also reported
LangChain/LangGraph version mismatches in the environment during setup.

### Impact

Installation can take a long time, and installing another package later may
silently change a transitive dependency required by RAG evaluation components.

### Follow-up

After the full ingestion run, capture tested versions in a lock/constraints
file and make `pip check` pass with no broken requirements. Keep the Docling
and Pydantic versions that were validated by the successful ingestion run.

## 6. Full Docling ingestion is CPU-intensive

The successful Amazon 2021 conversion took about 6 minutes 22 seconds on CPU.
At the current size (100 PDFs), a sequential full run is expected to take
roughly 9--11 hours. Keep the machine powered, the project environment active,
and the terminal open until the run reports its final summary.

## 7. Intel uses a nontraditional 10-K structure

### What happened

The five Intel filings (2021--2025) converted successfully with Docling but
reported zero sections. Intel organizes the report with descriptive headings
such as `Fundamentals of Our Business` and `Management's Discussion and
Analysis`, rather than placing `Item 1`, `Item 1A`, and similar SEC labels at
the start of its body sections.

### Resolution

When no standard SEC Item headings exist, the metadata extractor now falls
back to substantive Docling Markdown headings. These are stored as
`Heading 1`, `Heading 2`, and so on, instead of fabricating SEC Item numbers.
Cover-page and table-of-contents headings are skipped, as are very short
heading fragments. Existing Intel metadata was backfilled from `cleaned.md`,
so no slow PDF conversion was required.

## 8. Chunker produced non-unique IDs, reordered tables, and no parent links

### What happened

Before the first full 50-filing chunking run, review of `section_chunker.py`
found three defects that a full run would have baked into every chunk:
`chunk_id` was only counter-based (`{ticker}_{fiscal_year}_{counter:04d}`,
colliding across sections that reset the counter differently), tables were
extracted first and appended after all prose (so a table appearing at the end
of a section showed up before the section's opening prose in chunk order),
and chunks carried no reference back to their parent section.

### Resolution

`chunk_id` now includes the source filename stem
(`{ticker}_{fy}_{filename_stem}_{counter:04d}`), guaranteeing document-scoped
uniqueness. Table and prose chunks are now interleaved by walking the
section text once and emitting prose-before-table-before-prose in original
document order (see `chunk_section()`). Every chunk now carries
`parent_section_id`, `parent_section_start`, `parent_section_end` so
retrieval can expand a small chunk back to its full source section without
duplicating the entire section text in every Qdrant payload.

### Alternative considered

Storing the full section text on every chunk (denormalized) instead of a
parent reference -- rejected because it multiplies Qdrant payload size by
the average chunks-per-section ratio for no retrieval benefit; a reference
plus one lookup into `cleaned.md`/`chunks.json` is cheap and keeps the vector
DB lean.

## 9. Oversized table chunks

### What happened

A real-file check on Amazon 2021 showed several table chunks at 4,500-7,750
characters -- roughly 3-5x the ~400-token prose target -- risking silent
truncation by the embedding model's token limit.

### Resolution

`table_chunker._split_table_by_rows()` splits an oversized Markdown table
only at row boundaries (never mid-row, so numbers are never corrupted),
repeating the header + separator row in every part and appending a
`(part N/M)` suffix to the caption.

### Residual limitation (accepted, not fixed)

One Coca-Cola 2021 table chunk still reaches 5,197 characters because a
single row -- Docling misdetected an auditor's narrative "Critical Audit
Matter" writeup, formatted as a 2-column Markdown table -- is itself 1,708
characters of prose. Rows are never split mid-row by design, so this floor
can't be lowered without a smarter table-vs-prose classifier. Left as a
documented limitation rather than special-cased, since it is a single
outlier chunk out of 12,088 table chunks.

## 10. Leaked Markdown heading marker in `company_name`

### What happened

`extract_company_name_from_text()` walks backward from the "(Exact name of
Registrant...)" anchor line to find the company name. Docling renders that
line as a Markdown heading (`## AMAZON.COM, INC.`), and the extractor
returned it verbatim, leaking `##` into 27 of 50 filings' `company_name` --
and therefore into every one of their chunks' metadata.

### Resolution

Route the candidate line through the existing `_clean_heading_line()`
helper (already used for section-heading cleanup) before returning it.

### Alternative considered

A one-off `.lstrip("#").strip()` at the call site -- rejected in favor of
reusing `_clean_heading_line()`, which already strips bold/italic/bullet
decoration too, so the same fix also protects against decoration forms this
26-heading-marker count didn't happen to include.

## 11. NVDA 2024/2025 identity fields all null

### What happened

`extract_reporting_period_end()` and `extract_company_name_from_text()`
only searched the first 5,000 / 4,000 characters of each document for
cover-page fields. NVIDIA's cover page has enough front matter that its
"For the fiscal year ended January 28, 2024" declaration lands at character
~5,091 -- just past the cutoff -- so in-document extraction silently
returned `None` for exactly the 2024 and 2025 filings. Because `fiscal_year`
is derived from `reporting_period_end`, the registry cross-check (which
needs `fiscal_year` to look up the matching filing) also failed, leaving
`filing_date` null too, even though the CIK/ticker resolved fine.

### Resolution

Introduced a shared `COVER_PAGE_SEARCH_CHARS = 8000` constant and widened
all three identity-field search windows to it. Chosen value comes from
measuring the actual furthest-back anchor offset across all 50 filings
(4,881 for `company_name`, 5,091 for the fiscal-year declaration) plus a
safety margin, rather than guessing a round number.

### Alternative considered

Searching the entire document text unbounded -- rejected because the anchor
phrases ("Exact name of Registrant", "fiscal year ended") can appear again
later in the document (e.g. in a subsidiary's registrant description or a
historical comparison table), so an unbounded search risks matching the
wrong occurrence. A generous but bounded cover-page window is safer.

## 12. J&J's two 2023-labeled filings collided on `fiscal_year`

### What happened

`johnson__jnj_10K_2023` (period end 2023-01-01) and `johnson__jnj_10K_2024`
(period end 2023-12-31) both derived `fiscal_year = 2023` under the original
`int(reporting_period_end[:4])` rule, and `lookup_filing()` matched EDGAR
registry entries by `reporting_period_end.startswith(str(fiscal_year))` --
so both filings' registry lookups matched the *same* wrong-or-right entry,
giving both an identical (and for one of them, wrong) `filing_date`.

### Root cause

J&J's fiscal year follows a 52/53-week calendar ending on the Sunday nearest
December 31, which lands on January 1-3 in most years. J&J's own filing text
confirms the convention: the document with period end 2023-01-01 repeatedly
calls itself "fiscal year 2022" ("In the fiscal year 2022, the Company
recorded...") and mentions "fiscal year 2023" only as a future event. NVIDIA
has the opposite convention for a similar-looking case: its period ending
2024-01-28 (late January, not early) is self-labeled "fiscal year 2024" --
the same calendar year the date falls in, not shifted back.

### Resolution

Two changes, addressing two different failure points:

1. `fiscal_year_from_reporting_period_end()` in `metadata_extractor.py`
   shifts the derived year back by one when the period end falls in the
   first 7 days of January, leaving late-January dates (like NVIDIA's)
   unshifted.
2. `lookup_filing()` in `sec_registry.py` now matches registry entries by
   **exact** `reporting_period_end` first, falling back to the
   fiscal-year-prefix heuristic only when no in-document date is available.
   This was necessary even after fix (1): matching by a derived fiscal-year
   integer is inherently ambiguous whenever a company has two filings whose
   period-end dates round to years that collide under any single rule.

### Alternative considered

A per-company fiscal-year-end override table (like the existing
`FOLDER_NAME_TICKER_OVERRIDES`) -- rejected as a first resort because it
doesn't scale (every new 52/53-week filer added to the dataset would need a
manual entry) and doesn't fix the underlying `lookup_filing` ambiguity,
which would still misfire for any future filer with the same pattern. The
day-of-month heuristic plus exact-date registry matching generalizes without
per-company maintenance; a manual override table remains the right fallback
only for a filer whose convention doesn't fit either bucket.

### Verification

Confirmed via a script that reads every `reporting_period_end` +
`fiscal_year` pair across all 50 filings before and after: only the 6
January-dated filings (3 JNJ, 3 NVDA) were affected, and after the fix all
50 filings resolve to exactly 50 distinct (ticker, fiscal_year) pairs with
zero collisions.

## 13. Self-inflicted regression: wrong folder-name hint in the identity backfill

### What happened

To apply fixes 10-12 without re-running multi-hour Docling PDF conversion,
a new `scripts/backfill_identity.py` was written to re-derive identity
fields from the already-parsed `cleaned.md` files. Its first version passed
the full *processed*-folder name (e.g. `"intel__intc_10K_2021"`) into
`resolve_ticker_from_folder_name()`, but `run_ingestion.py`'s original
`folder_hint` is the *raw PDF's parent folder name* (`"intel"`) --
processed folders are named `"{raw_folder}__{pdf_stem}"`. The longer,
wrong string fed into the registry's fuzzy company-name matcher and silently
resolved Intel's folder to ticker `TSLA` (wrong CIK, wrong filing_date
written to disk) and failed to resolve NVDA's folders at all (nulling out
ticker/cik/filing_date that had been correct before the backfill).

### Resolution

Fixed `folder_hint = folder.name.split("__", 1)[0]` to match
`run_ingestion.py`'s convention, verified the corrected hint resolves both
companies correctly in isolation, then re-ran the backfill across all 50
filings and re-ran the full audit to confirm no other fields were disturbed.

### How this was caught

A blanket integrity audit (null-field counts, distinct ticker/fiscal_year
combinations, spot-checking the two affected companies by name) was run
immediately after the backfill and before re-chunking or moving to the next
stage -- this is why the corruption was caught and fully reversible rather
than propagating into `chunks.json` or, later, embeddings.

### Takeaway

Any script that re-derives identity fields from already-processed output
must reconstruct the exact same `folder_hint` convention the original
ingestion script used, not infer a plausible-looking substitute. Re-running
a full audit immediately after any bulk metadata edit is the actual safety
net, not care taken while writing the script.

## Output validation checklist after the full run

1. Confirm the final runner summary reports no unexpected skips.
2. Sample several `metadata.json` files and check `parser_used` is `docling`.
3. Check `raw_docling.md`, `cleaned.md`, and `metadata.json` exist for every
   processed filing; `raw_docling.json` is expected when structured export
   succeeds.
4. Review a few long sections, especially `Item 1A` and `Item 7`, to confirm
   their boundaries do not start at narrative references.
5. Run `python -m pip check` after batch work and resolve any remaining
   dependency conflicts before starting embedding, retrieval, or RAGAS work.
