"""
Diagnostic script for Tesla PDF - writes to file to avoid encoding issues.
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.parser import parse_with_docling

OUT = project_root / "scripts" / "docling_output_tesla.txt"

def analyze_pdf(pdf_path: Path, label: str):
    out_lines = []
    out_lines.append("=" * 80)
    out_lines.append(f"ANALYZING: {label} ({pdf_path.name})")
    out_lines.append("=" * 80)

    text = parse_with_docling(pdf_path)

    if text is None:
        out_lines.append("ERROR: Docling failed to parse this PDF")
        OUT.write_text("\n".join(out_lines), encoding="utf-8")
        return

    out_lines.append(f"\nTotal text length: {len(text):,} characters")
    out_lines.append(f"Total lines: {len(text.splitlines()):,}")

    # First 5000 chars
    out_lines.append("\n" + "-" * 60)
    out_lines.append("FIRST 5000 CHARACTERS:")
    out_lines.append("-" * 60)
    out_lines.append(text[:5000])

    # Lines with Item
    out_lines.append("\n" + "-" * 60)
    out_lines.append("LINES CONTAINING 'ITEM' (case-insensitive) - First 50:")
    out_lines.append("-" * 60)
    lines = text.splitlines()
    item_matches = []
    for i, line in enumerate(lines):
        if 'item' in line.lower():
            item_matches.append((i + 1, line))
    for line_num, line in item_matches[:50]:
        out_lines.append(f"  Line {line_num:4d}: {line[:150]}")
    out_lines.append(f"\n  Total lines matching 'item': {len(item_matches)}")

    # All Item N variations
    import re
    out_lines.append("\n" + "-" * 60)
    out_lines.append("ALL 'ITEM <N>' VARIATIONS (any case, with or without markdown):")
    out_lines.append("-" * 60)
    relaxed = re.compile(r"(?:^#+\s*)?item\s+(\d{1,2}[a-z]?)\.?\s+(.+)", re.IGNORECASE)
    relaxed_matches = []
    for i, line in enumerate(lines):
        m = relaxed.match(line.strip())
        if m:
            relaxed_matches.append((i + 1, m.group(1), m.group(2), line))
    for line_num, item_num, rest, line in relaxed_matches[:40]:
        out_lines.append(f"  Line {line_num:4d}: [{line[:150]}]")
    out_lines.append(f"\n  Total relaxed matches: {len(relaxed_matches)}")

    # Current regex
    out_lines.append("\n" + "-" * 60)
    out_lines.append("CURRENT REGEX MATCHES:")
    out_lines.append("-" * 60)
    pattern = re.compile(r"^ITEM\s+(\d{1,2}[A]?)\.\s+(.+)$")
    regex_matches = []
    for i, line in enumerate(lines):
        m = pattern.match(line.strip())
        if m:
            regex_matches.append((i + 1, m.group(1), m.group(2), line))
    for line_num, item_num, title, line in regex_matches[:20]:
        out_lines.append(f"  Line {line_num:4d}: ITEM {item_num}. {title[:80]}")
    out_lines.append(f"\n  Total regex matches: {len(regex_matches)}")

    # Markdown headers with Item
    out_lines.append("\n" + "-" * 60)
    out_lines.append("MARKDOWN HEADERS (first 50):")
    out_lines.append("-" * 60)
    md_headers = []
    for i, line in enumerate(lines):
        if line.strip().startswith('#'):
            md_headers.append((i + 1, line))
    for line_num, line in md_headers[:50]:
        out_lines.append(f"  Line {line_num:4d}: {line[:150]}")
    out_lines.append(f"\n  Total markdown headers: {len(md_headers)}")

    OUT.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Done! Output written to {OUT}")

if __name__ == "__main__":
    tesla_pdf = project_root / "data" / "raw_pdfs" / "sec_pdfs" / "tesla" / "tesla_10K_2022.pdf"
    analyze_pdf(tesla_pdf, "Tesla 10K 2022")
