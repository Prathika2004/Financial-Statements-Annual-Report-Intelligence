"""
End-to-end runner for Stage 6: Qdrant upload.

Save as: sec-rag-project/scripts/run_qdrant_upload.py

For every filing folder under data/processed/ that has chunks.json +
embeddings.npy, upserts its chunks (text + metadata as payload, vector as
the point's embedding) into the Qdrant collection defined in
config/settings.py. Idempotent: re-running is safe and overwrites the same
points rather than duplicating them (see qdrant_client.py's point-id scheme).

Usage:
    python scripts/run_qdrant_upload.py                # process everything
    python scripts/run_qdrant_upload.py --limit 5
    python scripts/run_qdrant_upload.py --folder amazon__amzn_10K_2022
    python scripts/run_qdrant_upload.py --recreate      # drop + recreate the collection first
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import PROCESSED_DIR, QDRANT_COLLECTION_NAME
from src.vectorstore.qdrant_client import (
    get_client, ensure_collection, drop_collection, upsert_chunks, count_points, delete_by_source_filename,
)

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_qdrant_upload")


def process_one_folder(folder: Path, client) -> int:
    chunks_path = folder / "chunks.json"
    embeddings_path = folder / "embeddings.npy"
    meta_path = folder / "embeddings_meta.json"

    if not chunks_path.exists() or not embeddings_path.exists():
        logger.warning("%s: missing chunks.json or embeddings.npy -- skipping.", folder.name)
        return 0

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    vectors = np.load(embeddings_path)

    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta["chunk_ids"] != [c["chunk_id"] for c in chunks]:
            logger.error(
                "%s: chunks.json order no longer matches embeddings_meta.json's chunk_ids "
                "-- re-run Stage 5 embedding before uploading. Skipping to avoid uploading "
                "vectors against the wrong chunk text.",
                folder.name,
            )
            return 0

    if not chunks:
        return 0

    # Delete this document's existing points first, not just upsert on top:
    # if the chunk count for this document changed since the last upload
    # (e.g. a table's row-split count shifted -- see qdrant_client.py's
    # delete_by_source_filename docstring), chunk_ids after that point
    # renumber, and upserting alone would leave old, now-orphaned points
    # behind at chunk_ids the new output no longer produces.
    delete_by_source_filename(client, chunks[0]["source_filename"])
    return upsert_chunks(client, chunks, vectors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--folder", type=str, default=None, help="Process a single named folder only")
    parser.add_argument("--recreate", action="store_true",
                         help="Drop and recreate the collection before uploading (DISCARDS existing points)")
    args = parser.parse_args()

    if args.folder:
        folders = [PROCESSED_DIR / args.folder]
    else:
        folders = sorted([p for p in PROCESSED_DIR.iterdir() if p.is_dir()])
        if args.limit:
            folders = folders[: args.limit]

    if not folders:
        logger.error("No processed filing folders found under %s. Run Stage 5 embedding first.", PROCESSED_DIR)
        return

    client = get_client()

    if args.recreate:
        try:
            drop_collection(client)
        except Exception as e:
            logger.info("Nothing to drop (%s).", e)

    ensure_collection(client)

    logger.info("Found %d filing folder(s) to upload.", len(folders))

    total_points = 0
    for folder in tqdm(folders, desc="Uploading filings"):
        if not folder.exists():
            logger.error("Missing folder: %s", folder)
            continue
        total_points += process_one_folder(folder, client)

    final_count = count_points(client)
    print(f"\nDone. {total_points} points upserted this run. "
          f"Collection '{QDRANT_COLLECTION_NAME}' now holds {final_count} points total.")


if __name__ == "__main__":
    main()
