"""
API routes: GET /health, POST /ask (blocking), POST /ask/stream (SSE).

Save as: sec-rag-project/src/api/routes.py

--- Why two /ask endpoints instead of one ---
numeric_verifier.py needs the COMPLETE answer text to check numbers against
sources -- it cannot run mid-stream. /ask/stream sends the answer as it's
generated (for a responsive UI, since generation takes minutes on this CPU
-- see llm_router.py's REQUEST_TIMEOUT_SECONDS comment) and only sends
sources + the numeric/citation checks once the full text is assembled, as
a final event. /ask is the simpler blocking version for callers that just
want one JSON response (curl, tests, non-UI callers). Both call
rag_pipeline.screen_and_prepare()/finalize_answer() -- the same input
guardrails, cache lookup, confidence gate, numeric verification, citation
check, and cache write, in the same order -- so results are identical
modulo streaming; see rag_pipeline.py's module docstring for why the
pipeline is split at that specific seam.

--- Why the route functions are plain `def`, not `async def` ---
Every call inside them (Qdrant, BM25, Ollama) is a blocking, synchronous
call -- none of it is `await`-able. FastAPI runs plain `def` routes in a
threadpool automatically, so the server's event loop isn't blocked while
one request is mid-retrieval or mid-generation. Starlette's
StreamingResponse does the equivalent for a plain (sync) generator function
passed to it, which is why event_stream() below is also a plain generator,
not an async one.

--- Why /health exists ---
This project has already had Docker Desktop silently stop mid-session once
(see PIPELINE_TECHNICAL_NOTES.md, note #14) -- a cheap endpoint that
reports whether Qdrant and Ollama are actually reachable right now is
directly motivated by that real incident, not a generic nicety.
"""

import json
import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from config.settings import OLLAMA_URL, PROCESSED_DIR
from src.api.auth import require_api_key
from src.api.schemas import AskRequest, AskResponse, SourceOut, NumericCheckOut, CitationCheckOut
from src.generation.prompt_templates import assemble_messages
from src.generation.llm_router import chat_stream, is_ollama_running
from src.generation.rag_pipeline import (
    screen_and_prepare, finalize_answer, answer_question,
    LOW_CONFIDENCE_MESSAGE, BLOCKED_MESSAGE,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def load_available_filings(processed_dir: Path = PROCESSED_DIR) -> List[dict]:
    """
    Reads {ticker, fiscal_year, company_name} out of every filing's
    metadata.json -- called once at server startup (see main.py's
    lifespan), not per-request, and cached on app.state.filings. Reading
    50 small JSON files at startup is negligible next to everything else
    startup already does (loading the embedding model, building the BM25
    index); querying Qdrant for distinct payload values on every /filings
    request would cost more for the same, static answer.
    """
    filings = []
    for meta_path in sorted(processed_dir.glob("*/metadata.json")):
        doc = json.loads(meta_path.read_text(encoding="utf-8"))["document"]
        if doc["status"] == "ok":
            filings.append({
                "ticker": doc["ticker"],
                "fiscal_year": doc["fiscal_year"],
                "company_name": doc["company_name"],
            })
    return filings


def _build_filters(req: AskRequest):
    filters = {}
    if req.ticker:
        filters["ticker"] = req.ticker.upper()
    if req.fiscal_year:
        filters["fiscal_year"] = req.fiscal_year
    if req.chunk_type:
        filters["chunk_type"] = req.chunk_type
    return filters or None


@router.get("/health")
def health(request: Request):
    qdrant_ok = True
    try:
        request.app.state.retrieval_context.qdrant_client.get_collections()
    except Exception:
        qdrant_ok = False

    return {"qdrant": qdrant_ok, "ollama": is_ollama_running(OLLAMA_URL)}


@router.get("/tickers")
def tickers(request: Request):
    return sorted({f["ticker"] for f in request.app.state.filings})


@router.get("/filings")
def filings(request: Request):
    return sorted(request.app.state.filings, key=lambda f: (f["ticker"], f["fiscal_year"]))


@router.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def ask(req: AskRequest, request: Request):
    state = request.app.state
    result = answer_question(req.question, state.retrieval_context, state.reranker,
                              filters=_build_filters(req), technique=req.technique)

    return AskResponse(
        answer=result["answer"],
        sources=[SourceOut(**s) for s in result["sources"]],
        numeric_check=NumericCheckOut(**result["numeric_check"]),
        citation_check=CitationCheckOut(**result["citation_check"]) if result.get("citation_check") else None,
        cached=result["cached"],
        blocked=result["blocked"],
        low_confidence=result["low_confidence"],
    )


@router.post("/ask/stream", dependencies=[Depends(require_api_key)])
def ask_stream(req: AskRequest, request: Request):
    state = request.app.state
    filters = _build_filters(req)
    prep = screen_and_prepare(req.question, state.retrieval_context, state.reranker,
                               filters=filters, technique=req.technique)

    def event_stream():
        if prep["status"] == "blocked":
            yield f"data: {json.dumps({'type': 'token', 'content': BLOCKED_MESSAGE})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'sources': [], 'numeric_check': {'checked': [], 'unverified': []}, 'blocked': True, 'reasons': prep['reasons']})}\n\n"
            return

        if prep["status"] == "cached":
            r = prep["result"]
            yield f"data: {json.dumps({'type': 'token', 'content': r['answer']})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'sources': [SourceOut(**s).model_dump() for s in r['sources']], 'numeric_check': r['numeric_check'], 'citation_check': r.get('citation_check'), 'cached': True})}\n\n"
            return

        if prep["status"] == "low_confidence":
            yield f"data: {json.dumps({'type': 'token', 'content': LOW_CONFIDENCE_MESSAGE})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'sources': [SourceOut(**s).model_dump() for s in prep['sources']], 'numeric_check': {'checked': [], 'unverified': []}, 'low_confidence': True, 'top_score': prep['top_score']})}\n\n"
            return

        sources = prep["sources"]
        messages = assemble_messages(req.question, sources)
        pieces = []
        for piece in chat_stream(messages):
            pieces.append(piece)
            yield f"data: {json.dumps({'type': 'token', 'content': piece})}\n\n"

        full_answer = "".join(pieces)
        result = finalize_answer(req.question, sources, full_answer, filters, req.technique)
        final_payload = {
            "type": "done",
            "sources": [SourceOut(**s).model_dump() for s in sources],
            "numeric_check": result["numeric_check"],
            "citation_check": result["citation_check"],
        }
        yield f"data: {json.dumps(final_payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
