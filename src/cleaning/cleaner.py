"""
Cleans raw extracted filing text: strips repeated running headers (e.g. the
"Table of Contents" header that appears on nearly every page), standalone
page-number lines, and normalizes excess whitespace.

Save as: sec-rag-project/src/cleaning/cleaner.py

Design note: this is deliberately conservative. It does NOT strip repeated
table headers like "Year Ended December 31," or "2023 2022 2021" -- those
look repetitive in a frequency count too, but they're real content that
appears once per table, not page boilerplate. Confirmed by checking a real
Tesla 10-K: "Table of Contents" appeared 26 times purely as a running page
header with no other content on its line, which is what BOILERPLATE_LINES
is matching against.
"""

import re
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import BOILERPLATE_LINES

PAGE_NUMBER_ONLY_PATTERN = re.compile(r"^\s*\d{1,4}\s*$")
MULTI_BLANK_LINE_PATTERN = re.compile(r"\n{3,}")


def is_boilerplate_line(line: str) -> bool:
    stripped = line.strip().lower()
    return stripped in BOILERPLATE_LINES


def is_page_number_line(line: str) -> bool:
    return bool(PAGE_NUMBER_ONLY_PATTERN.match(line))


def clean_text(raw_text: str) -> str:
    """
    Line-by-line cleaning pass:
      1. Drop lines that are exactly a known running header (case-insensitive).
      2. Drop lines that are exactly a standalone page number.
      3. Collapse 3+ consecutive blank lines down to a single blank line.
    """
    lines = raw_text.split("\n")
    kept_lines = []

    for line in lines:
        if is_boilerplate_line(line):
            continue
        if is_page_number_line(line):
            continue
        kept_lines.append(line)

    cleaned = "\n".join(kept_lines)
    cleaned = MULTI_BLANK_LINE_PATTERN.sub("\n\n", cleaned)
    return cleaned.strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("text_file", help="Path to a .txt file of extracted filing text")
    parser.add_argument("--out", default=None, help="Optional path to save cleaned text")
    args = parser.parse_args()

    raw = Path(args.text_file).read_text()
    cleaned = clean_text(raw)

    print(f"Raw length:     {len(raw):,} chars")
    print(f"Cleaned length: {len(cleaned):,} chars")
    print(f"Removed:        {len(raw) - len(cleaned):,} chars "
          f"({(len(raw) - len(cleaned)) / len(raw) * 100:.1f}%)")

    if args.out:
        Path(args.out).write_text(cleaned)
        print(f"Saved cleaned text to {args.out}")