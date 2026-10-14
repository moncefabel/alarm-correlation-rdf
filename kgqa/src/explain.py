"""Explainable answers: the triples that justify an answer, and where each
one comes from.

An answer to a SPARQL query is justified by the triples its pattern matched.
To find them, the query's WHERE clause is run again as SELECT *, so every
variable is bound, not only the projected ones; each triple pattern of the
query is filled in with those bindings, and the filled-in triples that exist
in the graph are the evidence. Patterns under FILTER NOT EXISTS and MINUS are
left out: they justify by absence, and an absent triple has no source.

Each evidence triple then gets its origin, from what the support graph
builder recorded:

  extracted      a statement node in provenance.ttl: the page(s) it was
                 extracted from, the method (title rule, regex, linker) and a
                 confidence
  inferred       listed in inferred.ttl: added by an OWL axiom (inverse or
                 symmetric property); its premise is the triple read the other
                 way, explained in turn
  page           a fact stated on one page: a documentation page (a view's
                 columns, a statement's category) or, for an alert, the
                 knowledge article whose title names it
  stand-in       the hand-written PPMS stand-in, which has no page

This is what "explainable retrieval" means here: every answer can be traced
to public pages, and an inferred link says which axiom produced it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from rdflib import OWL, RDF, Graph, Literal, Namespace, URIRef, Variable
from rdflib.paths import InvPath, MulPath
from rdflib.plugins.sparql import prepareQuery

SUP = Namespace("http://example.org/sapsupport#")
PROV = Namespace("http://www.w3.org/ns/prov#")
PPMS = "http://example.org/ppms#"
MAX_EVIDENCE = 40


@dataclass
class Evidence:
    triple: tuple
    kind: str                       # extracted | inferred | page | stand-in | unknown
    sources: list[str] = field(default_factory=list)
    method: str = ""
    confidence: float | None = None
    premise: "Evidence | None" = None

    def sourced(self) -> bool:
        if self.kind == "inferred":
            return bool(self.premise and self.premise.sourced())
        return bool(self.sources) or self.kind == "stand-in"

    def as_dict(self) -> dict:
        d = {"triple": [str(x) for x in self.triple], "kind": self.kind,
             "sources": self.sources, "method": self.method, "confidence": self.confidence}
        if self.premise is not None:
            d["premise"] = self.premise.as_dict()
        return d

    def render(self, indent: str = "  ", label: str = "") -> str:
        s, p, o = (_short(x) for x in self.triple)
        tag = {"extracted": f"[{self.method}, confidence {self.confidence}]",
               "inferred": f"[inferred: {self.method}]",
               "page": "[stated on the page]",
               "stand-in": "[PPMS stand-in, hand-written]"}.get(self.kind, "[no source]")
        lines = [f"{indent}{label}{s} {p} {o}  {tag}"]
        lines += [f"{indent}{' ' * len(label)}    {u}" for u in self.sources]
        if self.premise is not None:
            lines.append(self.premise.render(indent + "    ", "from: "))
        return "\n".join(lines)


def _short(x) -> str:
    if isinstance(x, Literal):
        return f'"{x}"'
    return str(x).split("#")[-1]


# ------------------------------------------------------------------ query side

def _match_brace(text: str, start: int) -> int:
    """Index of the brace closing the one at `start`, skipping IRIs and
    string literals (which may contain braces)."""
    depth, i, n = 0, start, len(text)
    while i < n:
        c = text[i]
        if c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            i = j
        elif c == "<" and i + 1 < n and not text[i + 1].isspace() and text[i + 1] not in "=":
            j = text.find(">", i)
            i = j if j > 0 and " " not in text[i:j] else i
        elif c == "#":
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unbalanced braces")


def bindings_query(query: str) -> str:
    """The query's WHERE clause as SELECT *: every variable bound."""
    import re
    m = re.search(r"\b(SELECT|ASK)\b", query, re.I)
    if not m:
        raise ValueError("not a SELECT or ASK query")
    start = query.index("{", m.end())
    end = _match_brace(query, start)
    return query[:m.start()] + "SELECT * WHERE " + query[start:end + 1] + " LIMIT 100"


def patterns(query: str) -> list[tuple]:
    """Triple patterns that must MATCH for a solution (not those under
    FILTER NOT EXISTS or MINUS)."""
    out: list[tuple] = []

    def walk(node):
        if not hasattr(node, "name"):
            return
        if node.name == "BGP":
            out.extend(tuple(t) for t in node.triples)
            return
        if node.name == "Minus":
            walk(node.get("p1"))
            return
        for key in ("p", "p1", "p2"):
            child = node.get(key) if hasattr(node, "get") else None
            if child is not None:
                walk(child)

    walk(prepareQuery(query).algebra)
    return out


# ------------------------------------------------------------------ graph side

class Explainer:
    def __init__(self, graph: Graph | str, provenance: Graph | str | None = None,
                 inferred: Graph | str | None = None):
        self.g = graph if isinstance(graph, Graph) else _load(graph)
        self.prov = provenance if isinstance(provenance, Graph) else _load(provenance)
        self.inferred = inferred if isinstance(inferred, Graph) else _load(inferred)
        self.inverse: dict = {}
        for p, q in self.g.subject_objects(OWL.inverseOf):
            self.inverse[p] = q
            self.inverse[q] = p
        self.symmetric = set(self.g.subjects(RDF.type, OWL.SymmetricProperty))

    @classmethod
    def for_graph(cls, graph_path: str | Path) -> "Explainer":
        """The graph file and, next to it, provenance.ttl and inferred.ttl."""
        d = Path(graph_path).parent
        return cls(str(graph_path), _maybe(d / "provenance.ttl"), _maybe(d / "inferred.ttl"))

    # ---------------------------------------------------------------- triples
    def evidence_triples(self, query: str) -> list[tuple]:
        pats = patterns(query)
        try:
            sols = list(self.g.query(bindings_query(query)))
        except Exception:                                    # noqa: BLE001
            return []
        found: list[tuple] = []
        seen = set()
        for sol in sols:
            b = sol.asdict()
            for s, p, o in pats:
                s2 = b.get(str(s), s) if isinstance(s, Variable) else s
                o2 = b.get(str(o), o) if isinstance(o, Variable) else o
                if isinstance(s2, Variable) or isinstance(o2, Variable):
                    continue
                for t in self._ground(s2, p, o2):
                    if t not in seen:
                        seen.add(t)
                        found.append(t)
                if len(found) >= MAX_EVIDENCE:
                    return found
        return found

    def _ground(self, s, p, o) -> list[tuple]:
        if isinstance(p, URIRef):
            return [(s, p, o)] if (s, p, o) in self.g else []
        if isinstance(p, InvPath) and isinstance(p.arg, URIRef):
            return [(o, p.arg, s)] if (o, p.arg, s) in self.g else []
        if isinstance(p, MulPath) and isinstance(p.path, URIRef):
            return self._chain(s, p.path, o)
        return []

    def _chain(self, s, p, o) -> list[tuple]:
        """The shortest chain s p ... p o, for a p+ or p* pattern."""
        prev = {s: None}
        todo = deque([s])
        while todo:
            x = todo.popleft()
            if x == o and x != s:
                break
            for y in self.g.objects(x, p):
                if y not in prev:
                    prev[y] = x
                    todo.append(y)
        if o not in prev or o == s:
            return []
        chain, y = [], o
        while prev[y] is not None:
            chain.append((prev[y], p, y))
            y = prev[y]
        return chain[::-1]

    # ---------------------------------------------------------------- origin
    def origin(self, t: tuple, depth: int = 0) -> Evidence:
        s, p, o = t
        if self.prov is not None:
            for st in self.prov.subjects(RDF.subject, s):
                if (st, RDF.predicate, p) in self.prov and (st, RDF.object, o) in self.prov:
                    conf = self.prov.value(st, SUP.confidence)
                    return Evidence(t, "extracted",
                                    sorted(str(u) for u in self.prov.objects(st, PROV.wasDerivedFrom)),
                                    str(self.prov.value(st, SUP.extractionMethod) or ""),
                                    float(conf) if conf is not None else None)
        if self.inferred is not None and t in self.inferred and depth < 2:
            if p in self.symmetric:
                return Evidence(t, "inferred", method=f"{_short(p)} is symmetric",
                                premise=self.origin((o, p, s), depth + 1))
            q = self.inverse.get(p)
            if q is not None:
                return Evidence(t, "inferred", method=f"{_short(p)} owl:inverseOf {_short(q)}",
                                premise=self.origin((o, q, s), depth + 1))
        if str(s).startswith(PPMS) or str(o).startswith(PPMS):
            return Evidence(t, "stand-in")
        page = self._page(s) or (self._page(o) if isinstance(o, URIRef) else [])
        if page:
            return Evidence(t, "page", page)
        return Evidence(t, "unknown")

    def _page(self, node) -> list[str]:
        url = self.g.value(node, SUP.sourceUrl)
        if url is not None:
            return [str(url)]
        view = self.g.value(node, SUP.columnOf)              # a column: its view's page
        if view is not None and self.g.value(view, SUP.sourceUrl) is not None:
            return [str(self.g.value(view, SUP.sourceUrl))]
        for p in (SUP.addresses, SUP.mentionsAlert):         # an alert: the articles naming it
            arts = sorted(str(u) for a in self.g.subjects(p, node)
                          for u in self.g.objects(a, SUP.sourceUrl))
            if arts:
                return arts
        return []

    # ---------------------------------------------------------------- answer
    def explain(self, query: str) -> list[Evidence]:
        return [self.origin(t) for t in self.evidence_triples(query)]


def summary(evidence: list[Evidence]) -> dict:
    return {"evidence_triples": len(evidence),
            "sourced": sum(e.sourced() for e in evidence),
            "inferred": sum(e.kind == "inferred" for e in evidence),
            "kinds": sorted({e.kind for e in evidence})}


def _load(path) -> Graph | None:
    if path is None:
        return None
    g = Graph()
    g.parse(str(path), format="turtle")
    return g


def _maybe(path: Path) -> str | None:
    return str(path) if path.exists() else None
