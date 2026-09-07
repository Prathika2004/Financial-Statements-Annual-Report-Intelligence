"""
Final prompt assembly: System Prompt -> Retrieved Context -> User Question
-> Instructions, as {role, content} messages for llm_router.chat().

Save as: sec-rag-project/src/generation/prompt_templates.py

--- Why sources are numbered and labeled with identity metadata ---
Each chunk already carries ticker/fiscal_year/section_item/chunk_type
(set all the way back in Stage 4 chunking specifically for this kind of
downstream use). Putting "[Source N: TICKER FY20XX, Item X]" directly above
each chunk's text, then instructing the model to cite "[Source N]" in its
answer, gives every claim a traceable link back to a specific filing and
section -- the whole point of doing RAG over financial filings instead of
just asking an LLM from its training memory. numeric_verifier.py depends on
being able to associate a claim with its source chunk after generation --
this citation format is what makes that association possible.

--- Why the system prompt explicitly permits "I don't know" ---
A model that always produces a confident-sounding answer regardless of
whether the retrieved context actually supports it is the single biggest
hallucination risk in a RAG system over financial data. The instruction to
say so plainly when the context is insufficient is not a nicety -- it is
the primary anti-hallucination mechanism at the prompt level (numeric_verifier.py
is the secondary, structural one).
"""

import sys
from pathlib import Path
from typing import Dict, List

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

SYSTEM_PROMPT = """You are a financial research assistant that answers questions about \
companies' SEC 10-K filings using ONLY the retrieved excerpts provided to you.

Rules:
1. Base your answer strictly on the provided sources. Never use outside knowledge \
or numbers you were trained on -- only what appears in the sources below.
2. Cite the source(s) your answer relies on using their bracketed labels, e.g. "[Source 2]", \
placed right after the claim they support, in the same sentence as that claim. Do not add a \
separate sentence that only states a citation (e.g. "This is cited from [Source 1]") -- the \
label must be attached to the factual claim itself, not stated on its own.
3. If the sources do not contain enough information to answer the question, say so \
plainly (e.g. "The provided filing excerpts do not contain this information") rather \
than guessing or extrapolating.
4. When stating a financial figure (a dollar amount, percentage, or count), reproduce \
it exactly as written in the source -- do not round, recompute, or approximate it.
5. If sources disagree or refer to different fiscal years/companies, point that out \
explicitly rather than silently picking one."""


# Phrases rule 3's "say so plainly" instruction would produce. Shared by
# eval/run_ragas.py's refusal-question scoring and
# src/guardrails/output_guardrails.py's citation check (a refusal correctly
# cites nothing, so it must not be flagged as a missing-citation problem) --
# both need to recognize the same "did the model actually refuse" signal,
# so it's defined once here rather than drifting into two copies.
REFUSAL_PHRASES = (
    "cannot be answered", "can't be answered", "do not contain", "does not contain",
    "no information", "not contain this information", "not available in", "not part of this dataset",
    "cannot determine", "insufficient information", "outside the scope", "i don't know", "i do not know",
)


def is_refusal_answer(answer: str) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)


def format_source_label(chunk: Dict, index: int) -> str:
    kind = " (table)" if chunk.get("chunk_type") == "table" else ""
    return (f"[Source {index}: {chunk.get('company_name') or chunk.get('ticker')} "
            f"FY{chunk.get('fiscal_year')}, {chunk.get('section_item')}{kind}]")


def build_context_block(chunks: List[Dict]) -> str:
    """Numbers sources starting at 1, in the order given (callers should
    pass chunks already ranked best-first, e.g. from retrieve_and_rerank())."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"{format_source_label(chunk, i)}\n{chunk['text']}")
    return "\n\n".join(blocks)


def build_user_message(question: str, chunks: List[Dict]) -> str:
    context_block = build_context_block(chunks) if chunks else "(No sources were retrieved.)"
    return (
        f"Sources:\n\n{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Instructions: Answer the question using only the sources above, citing them "
        f"by their [Source N] label as described in your system instructions."
    )


def assemble_messages(question: str, chunks: List[Dict]) -> List[Dict[str, str]]:
    """Returns the {role, content} message list ready for llm_router.chat()."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(question, chunks)},
    ]
