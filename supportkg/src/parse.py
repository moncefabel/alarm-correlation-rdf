"""Parse a public SAP knowledge-article preview page into a record.

The preview of an SAP Note or KBA shows a title ("1977220 - How to handle HANA
Alert 54: ..."), then labelled sections: Symptom, Environment, Product,
Keywords, and a notice that the full text needs a login. The page layout is
not documented and may change, so the parser does not depend on CSS classes
or element ids: it flattens the page to lines and splits them on the known
section headings. If a page cannot be read that way, `parse` says so instead
of returning an empty record silently.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from bs4 import BeautifulSoup

SECTIONS = ("Symptom", "Environment", "Product", "Keywords", "Reproducing the Issue",
            "Cause", "Resolution", "See Also", "Attributes", "Other Terms",
            "Other Components", "About this page")
_HEAD = re.compile(r"^\s*(%s)\s*:?\s*$" % "|".join(re.escape(s) for s in SECTIONS), re.I)
_TITLE = re.compile(r"^\s*(\d{6,7})\s*[-–—:]\s*(.+?)\s*$")
_CANON = {s.lower(): s for s in SECTIONS}


@dataclass
class Record:
    number: str
    title: str
    url: str = ""
    kind: str = ""                    # "KBA", "SAPNote" or "" when the page does not say
    symptom: str = ""
    environment: str = ""
    products: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    sections: dict[str, str] = field(default_factory=dict)

    def text(self) -> str:
        """Everything the extractor may read: title and visible sections."""
        return "\n".join([self.title, self.symptom, self.environment,
                          *self.products, ", ".join(self.keywords)])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Record":
        return cls(**d)


class ParseError(ValueError):
    pass


def _lines(html: str) -> tuple[list[str], str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    page_title = soup.title.get_text(" ", strip=True) if soup.title else ""
    text = soup.get_text("\n")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln], page_title


def _split_list(value: str, seps: str) -> list[str]:
    parts = re.split(seps, value)
    return [p.strip(" .") for p in parts if p.strip(" .")]


def parse(html: str, number: str | None = None, url: str = "") -> Record:
    lines, page_title = _lines(html)

    title_num, title = None, ""
    for cand in [page_title, *lines[:60]]:
        m = _TITLE.match(cand or "")
        if m and (number is None or m.group(1) == number):
            title_num, title = m.group(1), m.group(2)
            break

    sections: dict[str, list[str]] = {}
    current = None
    for ln in lines:
        m = _HEAD.match(ln)
        if m:
            current = _CANON[m.group(1).lower()]
            sections.setdefault(current, [])
            continue
        if current:
            sections[current].append(ln)
    body = {k: "\n".join(v).strip() for k, v in sections.items()}

    num = title_num or number
    if not num or not title or not body:
        raise ParseError(
            f"page {number or '?'}: no '<number> - <title>' line or no known section "
            f"heading found; the layout may have changed or the page needs JavaScript")

    about = body.get("About this page", "")
    kind = ("KBA" if re.search(r"Knowledge Base Article", about, re.I)
            else "SAPNote" if re.search(r"\bSAP Note\b", about, re.I) else "")
    return Record(
        number=num, title=title, url=url, kind=kind,
        symptom=body.get("Symptom", ""),
        environment=body.get("Environment", ""),
        products=_split_list(body.get("Product", ""), r"\s*[;\n]\s*"),
        keywords=_split_list(body.get("Keywords", ""), r"\s*[,\n]\s*"),
        sections={k: v for k, v in body.items()
                  if k not in ("Symptom", "Environment", "Product", "Keywords")},
    )
