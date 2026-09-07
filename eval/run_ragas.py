"""
Scores the pipeline against the eval set using RAGAS.

Save as: sec-rag-project/eval/run_ragas.py

--- Why gpt-oss:20b judges, not Qwen2.5 (the answering model) ---
Having a model grade its own answers is a weaker signal than an independent
judge -- gpt-oss:20b is already pulled in Ollama (used for nothing else in
this project yet) and is a different, larger model than Qwen2.5 7B, so it
isn't scoring its own homework. See PIPELINE_TECHNICAL_NOTES.md's
Module 24 (LLM-as-Judge) discussion for why this pairing was chosen.

--- Why this doesn't use ragas.evaluate() / a Dataset object ---
RAGAS 0.4.3's newer `ragas.metrics.collections` API scores one sample at a
time via each metric's async `.ascore(...)`/sync `.score(...)` method with
named arguments (user_input, response, retrieved_contexts, reference) --
confirmed by inspecting the installed package directly rather than
assumed from possibly-stale docs/memory, since this API differs across
ragas versions. This is more verbose than the classic
`evaluate(dataset, metrics=[...])` one-liner, but avoids depending on an
API surface this specific installed version has already deprecated
(importing metric classes from `ragas.metrics` directly logs a
DeprecationWarning pointing at this collections API).

--- Why RAGAS gets its own OpenAI/HuggingFace clients, not our own wrappers ---
RAGAS's llm_factory()/HuggingFaceEmbeddings expect an OpenAI-compatible
client and a plain model-name string respectively -- Ollama exposes an
OpenAI-compatible endpoint at /v1, so `OpenAI(base_url=...)` works directly
with no new dependency (the `openai` and `instructor` packages RAGAS needs
for this were already present as transitive dependencies). This does mean
RAGAS loads its own separate copy of bge-small for AnswerRelevancy rather
than reusing embedder.py's OpenVINO-cached one -- an accepted inefficiency
for an infrequent eval run, not worth the plumbing to share.

--- Cost warning ---
Every question makes ~5 LLM calls total: one Qwen2.5 answer-generation
call (same multi-minute-on-CPU cost as any other request -- see
llm_router.py), plus up to 4 gpt-oss:20b judge calls (Faithfulness,
AnswerRelevancy, ContextPrecisionWithReference, ContextRecall), and
gpt-oss:20b is a larger model than Qwen2.5 so its calls are not necessarily
faster. Use --limit while testing; a full run of the default eval set is a
genuinely long operation on this CPU-only machine, not a quick check.

Usage:
    python eval/run_ragas.py                # full eval set
    python eval/run_ragas.py --limit 2       # first 2 questions only (for testing the wiring)
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import (
    EVAL_SET_DIR, EMBEDDING_MODEL_NAME, RAGAS_JUDGE_MODEL_NAME, OLLAMA_OPENAI_COMPATIBLE_URL,
)
from eval.build_eval_set import build_eval_set
from src.retrieval.retriever import build_retrieval_context, retrieve_and_rerank
from src.reranking.flashrank_reranker import load_reranker
from src.generation.prompt_templates import assemble_messages, format_source_label, is_refusal_answer
from src.generation.llm_router import chat, is_ollama_running

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                     datefmt="%H:%M:%S")
logger = logging.getLogger("run_ragas")


def build_judge():
    """Ollama's OpenAI-compatible endpoint, wrapped for RAGAS's llm_factory.
    api_key is required by the OpenAI client but ignored by Ollama. Must be
    AsyncOpenAI, not OpenAI -- every metric here is scored via .ascore(),
    and RAGAS's agenerate() raises TypeError against a sync client
    (confirmed empirically: the sync client builds without error but fails
    at the first actual .ascore() call, not at construction time)."""
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    client = AsyncOpenAI(base_url=OLLAMA_OPENAI_COMPATIBLE_URL, api_key="ollama")
    return llm_factory(RAGAS_JUDGE_MODEL_NAME, provider="openai", client=client)


def build_ragas_embeddings():
    from ragas.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(EMBEDDING_MODEL_NAME)


def load_eval_questions() -> List[dict]:
    path = EVAL_SET_DIR / "eval_questions.json"
    if not path.exists():
        path = build_eval_set()
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(question: dict, context, ranker) -> Dict:
    """Runs retrieval+reranking+generation for one eval question (plain
    retrieval, technique="none" -- this eval set measures the baseline
    pipeline, not query transformation variants).

    contexts are labeled the same way build_context_block() labels them for
    the generation prompt (see prompt_templates.py) -- NOT bare chunk text.
    A real run against this exact eval set proved why this matters: chunk
    text routinely doesn't repeat the company name inline (confirmed by
    reproducing the retrieval for "cybersecurity_risk_factors_general" and
    inspecting the actual chunks -- none of the Microsoft/Tesla/NVIDIA
    chunks say "Microsoft"/"Tesla"/"NVIDIA" anywhere in their text). The
    answer correctly attributes claims to companies using the [Source N]
    labels the generation prompt gave it, but a Faithfulness judge handed
    only bare text has no way to verify that attribution and marks the
    (actually correct) claim unfaithful -- this is what produced
    faithfulness=0.0 on that question. Labeling contexts for RAGAS the same
    way they were labeled for generation gives the judge the same
    information the answering model had to work from."""
    sources = retrieve_and_rerank([question["question"]], context, ranker, filters=question.get("filters"))
    messages = assemble_messages(question["question"], sources)
    answer = chat(messages)
    labeled_contexts = [f"{format_source_label(s, i)}\n{s['text']}" for i, s in enumerate(sources, start=1)]
    return {"answer": answer, "contexts": labeled_contexts, "sources": sources}


async def score_refusal_question(question: dict, pipeline_result: Dict) -> Dict:
    """
    expect_refusal questions (see build_eval_set.py) test whether the
    system correctly declines rather than hallucinates when the answer
    isn't in scope -- RAGAS's context-based metrics don't meaningfully
    apply here (Context Precision/Recall compare retrieved context against
    a reference, but the "reference" for these questions describes an
    ABSENCE of information, not real content to match against) and, worse,
    can flat-out crash: retrieved_contexts can legitimately be empty (e.g.
    a filter combination matching zero real chunks, by design for one of
    these questions), which RAGAS's metrics reject outright rather than
    handling gracefully. A direct refusal-language check is the correct
    tool for this question type, not a workaround for a metric that
    doesn't fit.
    """
    return {"refused": is_refusal_answer(pipeline_result["answer"])}


async def score_one_question(question: dict, pipeline_result: Dict, judge, embeddings) -> Dict:
    from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecisionWithReference, ContextRecall

    user_input = question["question"]
    response = pipeline_result["answer"]
    retrieved_contexts = pipeline_result["contexts"]
    reference = question["reference"]

    faithfulness = await Faithfulness(llm=judge).ascore(
        user_input=user_input, response=response, retrieved_contexts=retrieved_contexts)
    answer_relevancy = await AnswerRelevancy(llm=judge, embeddings=embeddings).ascore(
        user_input=user_input, response=response)
    context_precision = await ContextPrecisionWithReference(llm=judge).ascore(
        user_input=user_input, reference=reference, retrieved_contexts=retrieved_contexts)
    context_recall = await ContextRecall(llm=judge).ascore(
        user_input=user_input, retrieved_contexts=retrieved_contexts, reference=reference)

    return {
        "faithfulness": faithfulness.value,
        "answer_relevancy": answer_relevancy.value,
        "context_precision": context_precision.value,
        "context_recall": context_recall.value,
    }


def _write_results(results, output_path: Path):
    """Overwrites the results file after every question, not just at the
    end -- a run that dies partway through (a judge-model hiccup, Ollama
    restarting, anything) must not lose already-completed, already-paid-for
    results. This is the direct fix for a real failure: the first full run
    completed 4/8 questions correctly, then crashed on question 5's
    AnswerRelevancy call, and because results were only ever written after
    the whole loop finished, all 4 completed results were lost with it."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions.")
    parser.add_argument("--skip", type=int, default=0,
                         help="Skip the first N questions -- use to resume after a partial run "
                              "without re-scoring (and re-paying for) questions already completed.")
    parser.add_argument("--ids", type=str, default=None,
                         help="Comma-separated question ids to run -- for re-checking specific "
                              "questions (e.g. after a fix) without paying for a full re-run. "
                              "Existing results for OTHER ids are preserved untouched; results for "
                              "these ids are overwritten with the fresh run.")
    args = parser.parse_args()

    if not is_ollama_running():
        raise SystemExit("Ollama isn't reachable -- is it running?")

    all_questions = load_eval_questions()
    if args.ids:
        wanted_ids = set(args.ids.split(","))
        questions = [q for q in all_questions if q["id"] in wanted_ids]
        missing = wanted_ids - {q["id"] for q in questions}
        if missing:
            raise SystemExit(f"Unknown question id(s): {sorted(missing)}")
    else:
        questions = all_questions
        if args.skip:
            questions = questions[args.skip:]
        if args.limit:
            questions = questions[:args.limit]
    logger.info("Loaded %d eval question(s) to run.", len(questions))

    output_path = EVAL_SET_DIR / "ragas_results.json"
    # Always merge into whatever's already on disk, keyed by id -- running a
    # subset (via --ids or --skip) must update just those entries and leave
    # every other already-scored question's result exactly as it was, not
    # discard them.
    results_by_id = {}
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for r in json.load(f):
                results_by_id[r["id"]] = r
        logger.info("Loaded %d existing result(s) from %s to merge into.", len(results_by_id), output_path)

    # Defined here too (not just inside the loop) so an empty `questions`
    # list (e.g. --skip larger than the eval set) still leaves `results`
    # defined for the summary section below.
    results = [results_by_id[q["id"]] for q in all_questions if q["id"] in results_by_id]

    logger.info("Building retrieval context and reranker...")
    context = build_retrieval_context()
    ranker = load_reranker()

    logger.info("Setting up RAGAS judge (%s) and embeddings...", RAGAS_JUDGE_MODEL_NAME)
    judge = build_judge()
    embeddings = build_ragas_embeddings()

    for i, question in enumerate(questions, start=1):
        t0 = time.time()
        logger.info("[%d/%d] %s: running pipeline...", i, len(questions), question["id"])
        try:
            pipeline_result = run_pipeline(question, context, ranker)
            is_refusal_question = question.get("expect_refusal", False)

            if is_refusal_question:
                logger.info("[%d/%d] %s: checking refusal language (no RAGAS -- see "
                            "score_refusal_question's docstring)...", i, len(questions), question["id"])
                scores = asyncio.run(score_refusal_question(question, pipeline_result))
            else:
                logger.info("[%d/%d] %s: scoring with RAGAS...", i, len(questions), question["id"])
                scores = asyncio.run(score_one_question(question, pipeline_result, judge, embeddings))

            elapsed = time.time() - t0
            results_by_id[question["id"]] = {
                "id": question["id"],
                "question": question["question"],
                "answer": pipeline_result["answer"],
                "reference": question["reference"],
                "expect_refusal": is_refusal_question,
                "scores": scores,
                "elapsed_seconds": round(elapsed, 1),
            }
            logger.info("[%d/%d] %s done in %.1fs: %s", i, len(questions), question["id"], elapsed, scores)
        except Exception as e:
            # A single question's judge-model hiccup (e.g. malformed
            # structured output) must not take the whole run down with it --
            # record the failure and keep going, so a flaky question costs
            # only itself, not everything scored before it.
            logger.error("[%d/%d] %s FAILED: %s", i, len(questions), question["id"], e)
            results_by_id[question["id"]] = {
                "id": question["id"],
                "question": question["question"],
                "error": str(e),
                "elapsed_seconds": round(time.time() - t0, 1),
            }

        # Preserve the original eval-set order in the file regardless of
        # which subset just ran.
        results = [results_by_id[q["id"]] for q in all_questions if q["id"] in results_by_id]
        _write_results(results, output_path)

    print(f"\nResults written to {output_path}\n")
    print(f"{'id':<35} {'faithfulness':>12} {'relevancy':>10} {'precision':>10} {'recall':>8}")
    scored = [r for r in results if "scores" in r and not r.get("expect_refusal")]
    refusals = [r for r in results if "scores" in r and r.get("expect_refusal")]
    failed = [r for r in results if "scores" not in r]
    for r in scored:
        s = r["scores"]
        print(f"{r['id']:<35} {s['faithfulness']:>12.3f} {s['answer_relevancy']:>10.3f} "
              f"{s['context_precision']:>10.3f} {s['context_recall']:>8.3f}")
    for r in failed:
        print(f"{r['id']:<35} {'FAILED':>12}")

    n = len(scored)
    if n:
        avg = {k: sum(r["scores"][k] for r in scored) / n for k in scored[0]["scores"]}
        print(f"\n{'AVERAGE (' + str(n) + ' scored)':<35} {avg['faithfulness']:>12.3f} "
              f"{avg['answer_relevancy']:>10.3f} {avg['context_precision']:>10.3f} {avg['context_recall']:>8.3f}")

    if refusals:
        print(f"\nRefusal check ({len(refusals)} question(s), scored separately -- see "
              f"score_refusal_question's docstring for why):")
        for r in refusals:
            print(f"  {r['id']:<35} refused={r['scores']['refused']}")

    if failed:
        print(f"\n{len(failed)} question(s) failed to score -- see log for details.")


if __name__ == "__main__":
    main()
