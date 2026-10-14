"""Semantic check: does each triple of a query go the way the ontology says?

The symbolic gate (agents.hallucinated) catches IRIs that do not exist. It
cannot catch a query whose IRIs all exist but which reads a property the wrong
way round: `sup:View_M_PASSWORD_POLICY sup:columnOf ?c` is valid SPARQL, every
IRI is known, and it returns nothing. Allemang and Sequeda (2024) call this an
"incorrect path" and repair it with an ontology-based query check; this module
is that check for this project's graphs.

Two signals:

1. Types. For each triple pattern with a known property, the entities named
   in it (and variables typed with `?x a C` in the same query) are checked
   against the property's rdfs:domain and rdfs:range. The domain and range of
   an inverse property are read off its inverse when not declared. A subject
   that fits the range and not the domain is a reversed triple, and the
   message says so and names the inverse property when there is one.

2. Emptiness (used by the agent, not here). A query that runs and returns no
   rows to an in-scope question gets one repair with a hint about direction.
   This is the only signal when both ends have the same type, as with
   net:causallyLinkedTo (Alarm to Alarm): no type check can tell a cause from
   an effect.
"""

from __future__ import annotations

from rdflib import OWL, RDF, RDFS, Graph, URIRef, Variable

from explain import patterns

EMPTY_HINT = ("The query ran without error but returned no rows, and the question "
              "expects an answer. Check the direction of every triple: which entity "
              "is the subject and which the object of each property, given its domain "
              "and range, and whether an inverse property reads more naturally. If the "
              "answer really is empty, return the same query unchanged.")


ZERO_HINT = ("The query ran without error but its answer is a count of 0, and the "
             "question expects a non-empty answer. Check the direction of every triple: "
             "which entity is the subject and which the object of each property, given "
             "its domain and range, and whether an inverse property or the property "
             "actually used in the data reads more naturally. If the true count is "
             "zero, return the same query unchanged.")


def zero_count(rows: list[dict]) -> bool:
    """One row whose every value is the number 0: the shape of COUNT over nothing."""
    if len(rows) != 1 or not rows[0]:
        return False
    try:
        return all(float(str(v)) == 0 for v in rows[0].values())
    except ValueError:
        return False


class TypeChecker:
    def __init__(self, g: Graph):
        self.g = g
        sup: dict = {}
        for c, d in g.subject_objects(RDFS.subClassOf):
            sup.setdefault(c, set()).add(d)
        self._sup = sup
        self.inverse: dict = {}
        for p, q in g.subject_objects(OWL.inverseOf):
            self.inverse[p] = q
            self.inverse[q] = p
        self.domain = {p: d for p, d in g.subject_objects(RDFS.domain) if isinstance(d, URIRef)}
        self.range = {p: r for p, r in g.subject_objects(RDFS.range) if isinstance(r, URIRef)}
        for p, q in list(self.inverse.items()):
            if p not in self.domain and q in self.range:
                self.domain[p] = self.range[q]
            if p not in self.range and q in self.domain:
                self.range[p] = self.domain[q]
        self.classes = set(g.subjects(RDF.type, OWL.Class)) | set(g.subjects(RDF.type, RDFS.Class))
        self._types: dict = {}

    def closure(self, classes) -> set:
        out, todo = set(classes), list(classes)
        while todo:
            for d in self._sup.get(todo.pop(), ()):
                if d not in out:
                    out.add(d)
                    todo.append(d)
        return out

    def types(self, node) -> set:
        if node not in self._types:
            self._types[node] = self.closure(set(self.g.objects(node, RDF.type)))
        return self._types[node]

    def check(self, query: str) -> list[str]:
        """Problems found, as messages for the repair prompt. Empty = passes."""
        try:
            pats = patterns(query)
        except Exception:                                    # noqa: BLE001
            return []                                        # syntax is checked elsewhere
        declared: dict = {}
        for s, p, o in pats:
            if p == RDF.type and isinstance(s, Variable) and isinstance(o, URIRef):
                declared.setdefault(s, set()).update(self.closure({o}))

        def kinds(x) -> set:
            if isinstance(x, URIRef):
                return self.types(x)
            if isinstance(x, Variable):
                return declared.get(x, set())
            return set()

        problems = []
        for s, p, o in pats:
            if not isinstance(p, URIRef) or p == RDF.type:
                continue
            d, r = self.domain.get(p), self.range.get(p)
            ts, to = kinds(s), kinds(o)
            s_bad = bool(d and ts and d not in ts)
            o_bad = bool(r in self.classes and to and r not in to) if r else False
            if not (s_bad or o_bad):
                continue
            reversed_ = (not ts or (r in ts)) and (not to or (d in to))
            name = lambda x: x.n3(self.g.namespace_manager)       # noqa: E731
            if reversed_:
                inv = self.inverse.get(p)
                fix = (f"swap subject and object, or use {name(inv)} "
                       f"({name(s)} {name(inv)} {name(o)})" if inv else "swap subject and object")
                problems.append(f"{name(s)} {name(p)} {name(o)} reads the property the wrong "
                                f"way round: {name(p)} goes from a {name(d)} to a {name(r)}. "
                                f"Fix: {fix}.")
            else:
                where = name(s) if s_bad else name(o)
                need = name(d) if s_bad else name(r)
                problems.append(f"In {name(s)} {name(p)} {name(o)}, {where} is not a {need}, "
                                f"which {name(p)} requires on that side.")
        return problems
