"""
End-to-end runner for Stage 5: Embedding.

Save as: sec-rag-project/scripts/run_embedding.py

For every filing folder under data/processed/ that has chunks.json, produces:

  data/processed/<company>__<filename>/
      ...(existing Stage 1/4 files)...
      embeddings.npy       # NEW: float32 array, shape (num_chunks, EMBEDDING_DIM)
      embeddings_meta.json # NEW: {model_name, dim, chunk_ids} -- see below

Row i of embeddings.npy is the vector for chunks.json's i-th chunk. That
alignment is the whole contract between this file and Stage 6 (Qdrant
upload); embeddings_meta.json's chunk_ids list exists so a loader can assert
the alignment still holds rather than trusting it silently.

Usage:
    python scripts/run_embedding.py                # process everything
    python scripts/run_embedding.py --limit 5
    python scripts/run_embedding.py --folder amazon__amzn_10K_2022
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import PROCESSED_DIR, EMBEDDING_MODEL_NAME
from src.embeddings.embedder import load_embedding_model, embed_texts

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_embedding")


def process_one_folder(folder: Path, model) -> int:
    chunks_path = folder / "chunks.json"

    if not chunks_path.exists():
        logger.warning("%s: no chunks.json found -- skipping.", folder.name)
        return 0

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        logger.info("%s: chunks.json is empty -- skipping.", folder.name)
        return 0

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts, model)

    np.save(folder / "embeddings.npy", vectors)
    with open(folder / "embeddings_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_name": EMBEDDING_MODEL_NAME,
            "dim": vectors.shape[1],
            "num_vectors": vectors.shape[0],
            "chunk_ids": [c["chunk_id"] for c in chunks],
        }, f, indent=2, ensure_ascii=False)

    return len(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--folder", type=str, default=None, help="Process a single named folder only")
    args = parser.parse_args()

    if args.folder:
        folders = [PROCESSED_DIR / args.folder]
    else:
        folders = sorted([p for p in PROCESSED_DIR.iterdir() if p.is_dir()])
        if args.limit:
            folders = folders[: args.limit]

    if not folders:
        logger.error("No processed filing folders found under %s. Run Stage 4 chunking first.", PROCESSED_DIR)
        return

    logger.info("Found %d filing folder(s) to embed.", len(folders))

    model = load_embedding_model()

    total_vectors = 0
    zero_vector_folders = []

    for folder in tqdm(folders, desc="Embedding filings"):
        if not folder.exists():
            logger.error("Missing folder: %s", folder)
            continue
        n = process_one_folder(folder, model)
        total_vectors += n
        if n == 0:
            zero_vector_folders.append(folder.name)

    print(f"\nDone. {total_vectors} total vectors produced across {len(folders)} filings.")
    if zero_vector_folders:
        print(f"{len(zero_vector_folders)} folder(s) produced zero vectors (check logs above): "
              f"{zero_vector_folders[:10]}{'...' if len(zero_vector_folders) > 10 else ''}")


if __name__ == "__main__":
    main()
