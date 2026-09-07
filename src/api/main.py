"""
FastAPI application entry point: loads the retrieval context, reranker, and
checks Ollama once at startup, then serves the routes in routes.py.

Save as: sec-rag-project/src/api/main.py

Run with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Or directly:
    python src/api/main.py

--- Why everything expensive loads once at startup, not per-request ---
The embedding model, Qdrant client, BM25 index (built from all 29,887
chunks), and reranker are all "load once, reuse forever" objects -- the
same pattern retriever.RetrievalContext was built around back in Stage 7,
specifically so a server never rebuilds any of them per request.
Rebuilding the BM25 index alone (~4.3s) on every request would be a
meaningful, entirely avoidable latency tax on top of the multi-minute
generation time that's already unavoidable on this CPU.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import OLLAMA_URL
from src.retrieval.retriever import build_retrieval_context
from src.reranking.flashrank_reranker import load_reranker
from src.generation.llm_router import is_ollama_running
from src.api.routes import router, load_available_filings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Building retrieval context (embedding model, Qdrant, BM25 index)...")
    app.state.retrieval_context = build_retrieval_context()

    logger.info("Loading reranker...")
    app.state.reranker = load_reranker()

    app.state.filings = load_available_filings()
    logger.info("Loaded %d available filing(s) for /tickers and /filings.", len(app.state.filings))

    if is_ollama_running():
        logger.info("Ollama is reachable at %s.", OLLAMA_URL)
    else:
        logger.warning("Ollama is NOT reachable at %s -- /ask requests will fail until it's running.", OLLAMA_URL)

    yield


app = FastAPI(title="SEC 10-K RAG API", lifespan=lifespan)
app.include_router(router)

# Mounted last and deliberately: Starlette matches routes in registration
# order, so /ask, /ask/stream, /health, /tickers, /filings (registered
# above) all take precedence over this catch-all. html=True serves
# static/index.html for "/" and for any other unmatched path, so the
# frontend and the API share one origin -- no CORS configuration needed,
# since a browser page fetching /ask/stream from the same origin that
# served it isn't a cross-origin request at all.
_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
