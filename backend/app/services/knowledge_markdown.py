import re
from typing import List


_TABLE_SEPARATOR_RE = re.compile(r":?-{3,}:?")


def markdown_table_rows(markdown: str) -> List[List[str]]:
    """Return data rows from Markdown pipe tables, excluding headers/separators."""
    lines = markdown.splitlines()
    rows: List[List[str]] = []
    for index, line in enumerate(lines):
        cells = _table_cells(line)
        if not cells or _is_separator_row(cells):
            continue
        next_cells = _table_cells(lines[index + 1]) if index + 1 < len(lines) else []
        if next_cells and _is_separator_row(next_cells):
            continue
        rows.append(cells)
    return rows


def _table_cells(line: str) -> List[str]:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _is_separator_row(cells: List[str]) -> bool:
    return bool(cells) and all(_TABLE_SEPARATOR_RE.fullmatch(cell) for cell in cells)
