"""
HyDE (Hypothetical Document Embeddings): generates a hypothetical answer to
the question, embeds that answer (not the question), and searches with it.

Save as: sec-rag-project/src/query_transformation/hyde.py

--- Why this bypasses retriever.retrieve() entirely ---
Every other technique in this package produces query STRINGS that get
embedded with BGE's query-instruction prefix (see embedder.py) and searched
normally through retriever.retrieve(). HyDE is structurally different: the
point is to embed passage-like text (a hypothetical answer, written in the
same register as the documents being searched) instead of the literal
question, on the theory that a fake answer sits closer in embedding space
to a real answer than the bare question does. That means the hypothetical
document must be embedded with is_query=False (see embedder.embed_texts)
and searched via dense_retriever.dense_search() directly with the
resulting vector -- exactly the split dense_search()/dense_search_from_text()
was built for (see dense_retriever.py's module docstring).

This has no BM25 equivalent: BM25 is keyword-based, and a hypothetical
document's exact wording isn't a meaningful sparse-search signal the way a
real question's keywords are -- so HyDE here only ever contributes a dense
ranked list. Feed that list into hybrid_fusion.reciprocal_rank_fusion()
alongside whatever else a caller runs (e.g. a plain retrieve() call on the
real question, for a "HyDE + normal hybrid search" ensemble).
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import DENSE_CANDIDATE_K
from src.query_transformation.llm_interface import LLMGenerate

logger = logging.getLogger(__name__)

HYDE_PROMPT_TEMPLATE = (
    "Write a short, plausible passage from a company's SEC 10-K filing that "
    "would answer the following question. Write it in the same factual, "
    "formal style as an actual filing -- it does not need to be accurate, "
    "it only needs to plausibly resemble a real answer.\n\nQuestion: {question}"
)


def generate_hypothetical_document(question: str, llm_generate: LLMGenerate) -> str:
    """Falls back to the literal question if generation returns nothing
    usable, so a caller downstream always has *something* to embed."""
    result = llm_generate(HYDE_PROMPT_TEMPLATE.format(question=question)).strip()
    return result or question


def hyde_search(question: str, llm_generate: LLMGenerate, embedding_model, qdrant_client,
                top_k: int = DENSE_CANDIDATE_K, filters: Optional[Dict] = None) -> List[dict]:
    """
    Full HyDE pipeline: generate a hypothetical answer, embed it as a
    passage (not a query), and dense-search with that vector. Returns a
    ranked list in the same shape dense_search() always returns.
    """
    from src.embeddings.embedder import embed_texts
    from src.retrieval.dense_retriever import dense_search

    hypothetical_doc = generate_hypothetical_document(question, llm_generate)
    vector = embed_texts([hypothetical_doc], embedding_model, is_query=False)[0]
    return dense_search(vector, qdrant_client, top_k=top_k, filters=filters)
