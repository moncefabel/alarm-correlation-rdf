"""Information extraction from knowledge-article previews.

A spaCy pipeline finds the entities, rules turn them into relations, and every
fact records the method that produced it, so the evaluation can score each
method separately and the graph can say how a fact was obtained.

Entities (spaCy):
  COMPONENT    application component codes (HAN-DB-MON): a custom tokenizer
               rule keeps a code in one token, an EntityRuler matches its shape
  TOOL         gazetteer from tools.ttl (PhraseMatcher on lower case)
  PRODUCT      gazetteer from ppms_standin.ttl labels and alternative labels
  ALERT_REF    "Alert 54", "alert ID 39", "alert (ID=75)": regex spans
  ARTICLE_REF  "SAP Note 1758890", "KBA 2000002": regex spans

Relations (rules):
  addresses      the title is a how-to for alert N ("How to handle HANA Alert N")
  mentionsAlert  any alert reference in the title or symptom
  cooccursWith   two alerts reported in the same article
  assignedTo     component codes in the Keywords section
  appliesTo      Product section strings linked to the PPMS stand-in
  references     article numbers cited in the visible text
  usesTool       tools named in the visible text
Alert names come from the title pattern; component names from the keyword
that follows the code (SAP lists a code next to its description).

A Hugging Face embedding linker (link.py) adds the links rules miss: product
strings with no exact match, and articles about an alert they never number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import spacy
from rdflib import Graph, RDF, RDFS, URIRef
from rdflib.namespace import SKOS
from spacy.language import Language
from spacy.tokens import Doc, Span

ONTO = Path(__file__).resolve().parents[1] / "ontology"
SUP = "http://example.org/sapsupport#"
PPMS = "http://example.org/ppms#"

CODE_RE = r"[A-Z][A-Z0-9]{1,5}(?:-[A-Z0-9]{1,8})+"
_CODE = re.compile(rf"^{CODE_RE}$")
# Codes that look like component codes but are not, seen in support texts
NOT_CODES = {"SAP-HANA", "S-USER", "NON-HANA", "E-MAIL"}

_ALERT = re.compile(
    r"\balerts?\s*(?:\(\s*)?(?:id\s*(?:=|:)?\s*)?#?(\d{1,4})\b\)?"
    r"((?:\s*(?:,|and|&)\s*#?\d{1,4}\b)*)", re.I)
_HOWTO = re.compile(
    r"^(?:how[- ]to:?\s*)?(?:how\s+to\s+)?(?:handle|resolve|interpret)\w*\s+"
    r"(?:sap\s+)?hana\s+alert\s*(?:\(\s*)?(?:id\s*=?\s*)?(\d{1,4})\)?\s*(?:[:\-–]\s*)?(.*)$",
    re.I)
_ARTICLE = re.compile(
    r"\b(?:SAP\s+)?(?:Notes?|KBAs?|Knowledge\s+Base\s+Articles?)\s*(?:#|No\.?|number)?\s*"
    r"(\d{6,7})((?:\s*(?:,|and|or|&)\s*\d{6,7}\b)*)", re.I)
_QUOTES = "'\"‘’“”"
_GENERIC_KEYWORDS = {"kba", "how to", "how-to", "problem", "sap note", "note", "faq"}


@dataclass(frozen=True)
class Fact:
    s: tuple            # ("article", "1977220"), ("alert", 54), ("component", "HAN-DB")...
    p: str              # local name of a sup: or ppms: property, or "label"
    o: object           # tuple key, IRI string, or literal
    method: str         # "title-rule", "keyword-rule", "gazetteer", "regex", "embedding"...
    confidence: float = 1.0


@dataclass
class Extraction:
    number: str
    facts: list[Fact] = field(default_factory=list)
    unlinked_products: list[str] = field(default_factory=list)

    def add(self, *a, **k) -> None:
        f = Fact(*a, **k)
        if f not in self.facts:
            self.facts.append(f)


# ------------------------------------------------------------- gazetteers

@lru_cache(maxsize=1)
def gazetteers() -> dict[str, dict[str, str]]:
    """Label -> IRI for tools and product items, from the ontology files."""
    g = Graph()
    for f in ("tools.ttl", "ppms_standin.ttl"):
        g.parse(ONTO / f)
    out: dict[str, dict[str, str]] = {"TOOL": {}, "PRODUCT": {}}
    kinds = {URIRef(SUP + "Tool"): "TOOL", URIRef(PPMS + "Product"): "PRODUCT",
             URIRef(PPMS + "ProductVersion"): "PRODUCT"}
    for cls, kind in kinds.items():
        for s in g.subjects(RDF.type, cls):
            for p in (RDFS.label, SKOS.altLabel):
                for lbl in g.objects(s, p):
                    out[kind][normalise(str(lbl))] = str(s)
    return out


def normalise(s: str) -> str:
    s = s.lower().replace(" ", " ")
    s = re.sub(r",?\s*all versions\b", "", s)
    s = re.sub(r"[^\w.]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------- spaCy pipeline

@Language.component("numbered_refs")
def numbered_refs(doc: Doc) -> Doc:
    """Alert and article references as entity spans, from regex matches."""
    spans = list(doc.ents)
    taken = {i for e in spans for i in range(e.start, e.end)}
    for label, rx in (("ALERT_REF", _ALERT), ("ARTICLE_REF", _ARTICLE)):
        for m in rx.finditer(doc.text):
            sp = doc.char_span(m.start(), m.end(), label=label, alignment_mode="expand")
            if sp is not None and not taken & set(range(sp.start, sp.end)):
                spans.append(sp)
                taken |= set(range(sp.start, sp.end))
    doc.ents = sorted(spans, key=lambda e: e.start)
    return doc


@lru_cache(maxsize=1)
def nlp() -> Language:
    """A blank English pipeline: no model download, fully deterministic."""
    pipe = spacy.blank("en")
    code = re.compile(rf"^{CODE_RE}[.,;:)]?$")
    pipe.tokenizer.token_match = code.match          # keep HAN-DB-MON in one token
    ruler = pipe.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER"})
    patterns = [{"label": "COMPONENT", "pattern": [{"TEXT": {"REGEX": rf"^{CODE_RE}$"}}]}]
    for kind, table in gazetteers().items():
        for lbl in table:
            patterns.append({"label": kind, "pattern": lbl})
    ruler.add_patterns(patterns)
    pipe.add_pipe("numbered_refs", after="entity_ruler")
    return pipe


# ------------------------------------------------------------------ rules

def _numbers(first: str, more: str) -> list[str]:
    return [first, *re.findall(r"\d+", more or "")]


def article_refs(text: str, exclude: set[str] | frozenset = frozenset()) -> list[str]:
    out = []
    for m in _ARTICLE.finditer(text or ""):
        out += [n for n in _numbers(m.group(1), m.group(2)) if n not in exclude]
    return list(dict.fromkeys(out))


def alert_refs(text: str) -> list[int]:
    out = []
    for m in _ALERT.finditer(text or ""):
        out += [int(n) for n in _numbers(m.group(1), m.group(2))]
    return list(dict.fromkeys(out))


def title_alert(title: str) -> tuple[int, str] | None:
    """(number, name) when the title is a how-to for one alert."""
    m = _HOWTO.match(title.strip())
    if not m:
        return None
    name = m.group(2).strip().strip(_QUOTES).strip()
    name = re.sub(r"\s+", " ", name)
    return int(m.group(1)), name


def component_codes(keywords: list[str]) -> list[tuple[str, str]]:
    """(code, label) pairs; the label is the keyword right after the code."""
    out = []
    for i, kw in enumerate(keywords):
        k = kw.strip()
        if _CODE.match(k) and k not in NOT_CODES:
            label = ""
            if i + 1 < len(keywords):
                nxt = keywords[i + 1].strip()
                if not _CODE.match(nxt) and nxt.lower() not in _GENERIC_KEYWORDS \
                        and not nxt.startswith("#"):
                    label = nxt
            out.append((k, label))
    return out


def parents(code: str) -> list[str]:
    parts = code.split("-")
    return ["-".join(parts[:i]) for i in range(len(parts) - 1, 0, -1)]


def extract(rec, linker=None) -> Extraction:
    """Facts about one record. `linker` (link.EmbeddingLinker) is optional."""
    ex = Extraction(rec.number)
    art = ("article", rec.number)
    ex.add(art, "label", rec.title, "parser")
    if rec.url:
        ex.add(art, "sourceUrl", rec.url, "parser")
    if rec.kind:
        ex.add(art, "type", rec.kind, "parser")

    # alerts: how-to title, then every reference in the title and symptom
    main = title_alert(rec.title)
    if main:
        n, name = main
        ex.add(art, "addresses", ("alert", n), "title-rule")
        ex.add(art, "mentionsAlert", ("alert", n), "title-rule")
        if name:
            ex.add(("alert", n), "label", name, "title-rule")
    doc = nlp()(rec.title + "\n" + rec.symptom)
    mentioned = []
    for ent in doc.ents:
        if ent.label_ == "ALERT_REF":
            for n in alert_refs(ent.text):
                mentioned.append(n)
                ex.add(art, "mentionsAlert", ("alert", n), "regex")
    alerts = list(dict.fromkeys(([main[0]] if main else []) + mentioned))
    for i, a in enumerate(alerts):
        for b in alerts[i + 1:]:
            ex.add(("alert", min(a, b)), "cooccursWith", ("alert", max(a, b)), "same-article")

    # components: Keywords section, with the hierarchy implied by the code
    for code, label in component_codes(rec.keywords):
        ex.add(art, "assignedTo", ("component", code), "keyword-rule")
        if label:
            ex.add(("component", code), "label", label, "keyword-rule")
        child = code
        for parent in parents(code):
            ex.add(("component", child), "subComponentOf", ("component", parent), "code-prefix")
            child = parent

    # products: the Product section, linked to the PPMS stand-in
    table = gazetteers()["PRODUCT"]
    for p in rec.products:
        key = normalise(p)
        if key in table:
            ex.add(art, "appliesTo", table[key], "gazetteer")
        elif linker is not None and (hit := linker.link_product(p)):
            iri, score = hit
            ex.add(art, "appliesTo", iri, "embedding", round(score, 3))
        else:
            ex.unlinked_products.append(p)

    # tools and cited articles, anywhere in the visible text
    full = nlp()(rec.text())
    for ent in full.ents:
        if ent.label_ == "TOOL":
            ex.add(art, "usesTool", gazetteers()["TOOL"][normalise(ent.text)], "gazetteer")
    for ref in article_refs(rec.text(), exclude={rec.number}):
        ex.add(art, "references", ("article", ref), "regex")

    # embedding linker: an article about alerts it never numbers
    if linker is not None and not alerts:
        for n, score in linker.link_alerts(rec.title + ". " + rec.symptom[:400]):
            ex.add(art, "mentionsAlert", ("alert", n), "embedding", round(score, 3))
    return ex
