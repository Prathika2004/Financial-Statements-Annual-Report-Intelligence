"""
End-to-end runner for Stage 4: Chunking.

Save as: sec-rag-project/scripts/run_chunking.py

For every filing folder under data/processed/ that has metadata.json +
cleaned.md, produces chunks.json alongside them:

  data/processed/<company>__<filename>/
      ...(existing Stage 1 files)...
      chunks.json    # NEW: list of chunk dicts ready for Stage 5 embedding

Usage:
    python scripts/run_chunking.py                # process everything
    python scripts/run_chunking.py --limit 5
    python scripts/run_chunking.py --folder amazon__amzn_10K_2022
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import PROCESSED_DIR
from src.chunking.section_chunker import chunk_document

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_chunking")


def process_one_folder(folder: Path) -> int:
    metadata_path = folder / "metadata.json"
    cleaned_path = folder / "cleaned.md"

    if not metadata_path.exists():
        logger.warning("%s: no metadata.json found -- skipping.", folder.name)
        return 0
    if not cleaned_path.exists():
        logger.warning("%s: no cleaned.md found -- skipping.", folder.name)
        return 0

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    if metadata["document"]["status"] != "ok":
        logger.info("%s: document status is '%s', not 'ok' -- skipping chunking.",
                     folder.name, metadata["document"]["status"])
        return 0

    cleaned_text = Path(cleaned_path).read_text(encoding="utf-8")

    chunks = chunk_document(metadata, cleaned_text)

    with open(folder / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

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
        logger.error("No processed filing folders found under %s. Run Stage 1 ingestion first.", PROCESSED_DIR)
        return

    logger.info("Found %d filing folder(s) to chunk.", len(folders))

    total_chunks = 0
    zero_chunk_folders = []

    for folder in tqdm(folders, desc="Chunking filings"):
        if not folder.exists():
            logger.error("Missing folder: %s", folder)
            continue
        n = process_one_folder(folder)
        total_chunks += n
        if n == 0:
            zero_chunk_folders.append(folder.name)

    print(f"\nDone. {total_chunks} total chunks produced across {len(folders)} filings.")
    if zero_chunk_folders:
        print(f"{len(zero_chunk_folders)} folder(s) produced zero chunks (check logs above): "
              f"{zero_chunk_folders[:10]}{'...' if len(zero_chunk_folders) > 10 else ''}")


if __name__ == "__main__":
    main()