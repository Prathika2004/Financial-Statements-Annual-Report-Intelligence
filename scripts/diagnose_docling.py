"""
Diagnostic script to examine Docling markdown output format.
Writes results to a file to avoid Windows encoding issues.
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.parser import parse_with_docling

OUT = project_root / "scripts" / "docling_output.txt"

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

    # 1. First 5000 characters
    out_lines.append("\n" + "-" * 60)
    out_lines.append("FIRST 5000 CHARACTERS:")
    out_lines.append("-" * 60)
    out_lines.append(text[:5000])

    # 2. Last 1000 characters
    out_lines.append("\n" + "-" * 60)
    out_lines.append("LAST 1000 CHARACTERS:")
    out_lines.append("-" * 60)
    out_lines.append(text[-1000:])

    # 3. Lines containing "Item" or "ITEM"
    out_lines.append("\n" + "-" * 60)
    out_lines.append("LINES CONTAINING 'ITEM' (case-insensitive) - First 50 matches:")
    out_lines.append("-" * 60)
    lines = text.splitlines()
    item_matches = []
    for i, line in enumerate(lines):
        if 'item' in line.lower():
            item_matches.append((i + 1, line))

    for line_num, line in item_matches[:50]:
        out_lines.append(f"  Line {line_num:4d}: {line[:150]}")

    out_lines.append(f"\n  Total lines matching 'item': {len(item_matches)}")

    # 4. Lines around "Item 1" or "Item 1A" appearances (section starts)
    out_lines.append("\n" + "-" * 60)
    out_lines.append("CONTEXT AROUND 'ITEM 1' or 'ITEM 1A' LINES:")
    out_lines.append("-" * 60)

    import re
    # Match Item 1, Item 1A, Item 2, etc. but in context of a section heading
    item1_pattern = re.compile(r"item\s+1[a]?\b", re.IGNORECASE)
    item1_matches = []
    for i, line in enumerate(lines):
        if item1_pattern.search(line):
            item1_matches.append(i)

    for idx in item1_matches[:10]:
        start = max(0, idx - 5)
        end = min(len(lines), idx + 6)
        out_lines.append(f"\n  --- Context around line {idx + 1} ---")
        for j in range(start, end):
            marker = ">>>" if j == idx else "   "
            out_lines.append(f"  {marker} Line {j + 1:4d}: {lines[j][:150]}")

    # 5. Markdown headers
    out_lines.append("\n" + "-" * 60)
    out_lines.append("MARKDOWN HEADER PATTERNS (lines starting with #):")
    out_lines.append("-" * 60)
    md_headers = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#'):
            md_headers.append((i + 1, line))

    for line_num, line in md_headers[:50]:
        out_lines.append(f"  Line {line_num:4d}: {line[:150]}")

    out_lines.append(f"\n  Total markdown headers: {len(md_headers)}")

    # 6. Current regex pattern matches
    out_lines.append("\n" + "-" * 60)
    out_lines.append("CURRENT REGEX PATTERN MATCHES (r'^ITEM\\s+(\\d{1,2}[A]?)\\.\\s+(.+)$'):")
    out_lines.append("-" * 60)
    pattern = re.compile(r"^ITEM\s+(\d{1,2}[A]?)\.\s+(.+)$")
    regex_matches = []
    for i, line in enumerate(lines):
        m = pattern.match(line.strip())
        if m:
            regex_matches.append((i + 1, m.group(1), m.group(2), line))

    for line_num, item_num, title, line in regex_matches[:30]:
        out_lines.append(f"  Line {line_num:4d}: ITEM {item_num}. {title[:80]}")
        out_lines.append(f"         Raw: {line[:120]}")

    out_lines.append(f"\n  Total regex matches: {len(regex_matches)}")

    # 7. Relaxed pattern search - show all variations of "Item" followed by number
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

    OUT.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Done! Output written to {OUT}")

if __name__ == "__main__":
    amazon_pdf = project_root / "data" / "raw_pdfs" / "sec_pdfs" / "amazon" / "amzn_10K_2022.pdf"
    analyze_pdf(amazon_pdf, "Amazon 10K 2022")
