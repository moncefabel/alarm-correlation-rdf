"""Help documentation source: the SAP HANA Cloud SQL reference, in Markdown.

SAP publishes the Markdown source of this documentation on GitHub under
CC-BY-4.0 (https://github.com/SAP-docs/sap-hana-cloud-hana-database), so it
can be cloned and reused with attribution, with no scraping:

    git clone --depth 1 https://github.com/SAP-docs/sap-hana-cloud-hana-database \
        supportkg/data/sap-hana-cloud-hana-database

Three kinds of page are read from docs/sql-ref:
  - monitoring views (M_*): what an administrator queries when an alert fires;
  - other system views (USERS, TABLES...);
  - SQL statements, with their category ("System Management", "Access Control").
For a view: its description, its columns (name, data type, description) and
the views listed under "Related Information". Every record keeps the GitHub
URL of its page at the cloned commit, so each fact points to a fixed source.
"""

from __future__ import annotations

import html
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO_URL = "https://github.com/SAP-docs/sap-hana-cloud-hana-database"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
VIEWS_DIR = "docs/sql-ref/020-System-Views-Reference"
STATEMENTS_DIR = "docs/sql-ref/010-SQL-Reference/012-SQL-Statements"

_TITLE = re.compile(r"^#\s+(.+?)\s*$", re.M)
_STATEMENT = re.compile(r"^(.+?) Statement \((.+)\)$")
_ROW = re.compile(r"<tr>(.*?)</tr>", re.S)
_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S)
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+\.md)(?:\s+\"[^\"]*\")?\)")


@dataclass
class DocRecord:
    key: str                     # file name without extension, stable across runs
    kind: str                    # "monitoring_view" | "system_view" | "sql_statement"
    name: str                    # M_SAVEPOINTS, ALTER SYSTEM CLEAR TRACES
    description: str
    url: str
    category: str = ""           # statements only
    columns: list[tuple[str, str, str]] = field(default_factory=list)
    related: list[str] = field(default_factory=list)   # keys of related pages


def unescape(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"\\(.)", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def _description(body: str) -> str:
    """First plain paragraph after the title."""
    after = body.split("\n", 1)[1] if "\n" in body else ""
    para: list[str] = []
    for ln in after.splitlines():
        s = ln.strip()
        if not s:
            if para:
                break
            continue
        if s.startswith(("<", "#", "```", "|", "!")):
            if para:
                break
            continue
        para.append(s)
    return unescape(" ".join(para))


def _columns(text: str) -> list[tuple[str, str, str]]:
    # the column table, under "## Structure" or, on some pages, with no heading;
    # *_RESET views have none and point to their main view instead
    m = re.search(r"##\s+Structure(.*?)</table>", text, re.S)
    table = m.group(1) if m else next(
        (t for t in re.findall(r"<table>(.*?)</table>", text, re.S)
         if re.search(r"<th[^>]*>\s*Column", t)), "")
    out = []
    for row in _ROW.findall(table):
        if "<th" in row:
            continue
        cells = [unescape(c) for c in _CELL.findall(row)]
        if len(cells) >= 2 and cells[0]:
            out.append((cells[0], cells[1], cells[2] if len(cells) > 2 else ""))
    return out


def _related(text: str) -> list[str]:
    i = text.find("Related Information")
    if i < 0:
        return []
    return list(dict.fromkeys(Path(t).stem for _, t in _LINK.findall(text[i:])))


def parse_page(path: Path, kind_hint: str, url: str) -> DocRecord | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = _TITLE.search(text)
    if not m:
        return None
    title = unescape(m.group(1))
    body = text[m.start():]
    if kind_hint == "view":
        if not title.endswith(" System View"):
            return None
        name = title[: -len(" System View")].strip()
        kind = "monitoring_view" if "022-Monitoring-Views" in str(path) else "system_view"
        return DocRecord(path.stem, kind, name, _description(body), url,
                         columns=_columns(text), related=_related(text))
    sm = _STATEMENT.match(title)
    if not sm:                                   # overview pages, lists
        return None
    return DocRecord(path.stem, "sql_statement", sm.group(1).strip(), _description(body), url,
                     category=sm.group(2).strip(), related=_related(text))


def commit_of(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "main"


def load(repo: Path) -> list[DocRecord]:
    repo = Path(repo)
    if not (repo / VIEWS_DIR).is_dir():
        raise SystemExit(f"{repo} is not a clone of {REPO_URL}; see supportkg/README.md")
    commit = commit_of(repo)
    out: list[DocRecord] = []
    for sub, hint in ((VIEWS_DIR, "view"), (STATEMENTS_DIR, "statement")):
        for p in sorted((repo / sub).rglob("*.md")):
            rel = p.relative_to(repo).as_posix()
            rec = parse_page(p, hint, f"{REPO_URL}/blob/{commit}/{rel}")
            if rec:
                out.append(rec)
    keys = {r.key for r in out}
    for r in out:                                 # keep links to pages we know
        r.related = [k for k in r.related if k in keys and k != r.key]
    return out
