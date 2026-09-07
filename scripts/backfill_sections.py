"""Regenerate section metadata from existing cleaned documents.

Use this after improving section detection. It avoids re-running slow PDF
conversion when the raw and cleaned outputs are already valid.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import PROCESSED_DIR
from src.ingestion.metadata_extractor import extract_sections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        required=True,
        help="Processed-folder prefix to update, for example 'intel__'.",
    )
    args = parser.parse_args()

    folders = sorted(path for path in PROCESSED_DIR.glob(f"{args.prefix}*") if path.is_dir())
    if not folders:
        raise SystemExit(f"No processed folders found with prefix: {args.prefix}")

    for folder in folders:
        cleaned_path = folder / "cleaned.md"
        metadata_path = folder / "metadata.json"
        if not cleaned_path.exists() or not metadata_path.exists():
            print(f"Skipping {folder.name}: cleaned.md or metadata.json is missing")
            continue

        cleaned_text = cleaned_path.read_text(encoding="utf-8")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        sections = extract_sections(cleaned_text)

        metadata["sections"] = [section.to_dict() for section in sections]
        metadata["statistics"]["num_sections"] = len(sections)
        metadata["statistics"]["full_cleaned_text_char_count"] = len(cleaned_text)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"{folder.name}: {len(sections)} sections")


if __name__ == "__main__":
    main()
