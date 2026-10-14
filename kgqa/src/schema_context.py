"""Schema and entity linking.

The single biggest driver of text-to-SPARQL quality in the literature is what
context goes in the prompt. Dumping the whole ontology does not scale and hurts
precision; SPARQL-LLM (SIB) retrieves endpoint metadata, MetaboT resolves
entities against authoritative identifiers first.

Three kinds of items are indexed:
  - classes and properties (declared in the ontology),
  - properties USED in the data but never declared (deviceType, mitreId here:
    real graphs are never perfectly declared, and a model that never sees these
    predicates cannot answer questions that need them),
  - instances with their rdfs:label, so "firewall FW1" can be linked to
    net:Firewall_Edge. Without this the model has to guess entity IRIs, which is
    the main source of hallucination this project measures.

Properties whose values are a small set of strings (severity levels, alarm
types) also show those values. Without them a model asked for "critical"
alarms has to guess the spelling ("critical", "Critical", "CRITICAL") and an
exact-match filter on the wrong one returns nothing. This is value linking, the
literal counterpart of entity linking.

Retrieval is lexical and deterministic so the ablation
"full context vs retrieved context" stays clean. Swapping in embeddings is
itself an ablation, not a default.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from rdflib import OWL, RDF, RDFS, SKOS, XSD, Graph, Literal, URIRef
from rdflib.collection import Collection

# Standard vocabularies a query may legitimately use without them being part of
# the domain ontology.
STANDARD_NS = (str(RDF), str(RDFS), str(OWL), str(XSD))

_TOKEN = re.compile(r"[a-z0-9]{2,}")


@dataclass
class SchemaItem:
    uri: str
    kind: str          # "class" | "property" | "instance"
    label: str
    comment: str
    domain: str = ""
    range_: str = ""
    type_: str = ""    # rdf:type of an instance
    values: tuple[str, ...] = ()   # small closed set of literal values

    def render(self) -> str:
        bits = [f"{self.kind}: <{self.uri}>"]
        if self.label:
            bits.append(f'label="{self.label}"')
        if self.type_:
            bits.append(f"type=<{self.type_}>")
        if self.domain or self.range_:
            bits.append(f"domain={self.domain or '?'} range={self.range_ or '?'}")
        if self.values:
            bits.append("values=" + " | ".join(f'"{v}"' for v in self.values))
        if self.comment:
            bits.append(f"# {self.comment}")
        return "  ".join(bits)

    def bag(self) -> set[str]:
        local = self.uri.split("#")[-1].split("/")[-1]
        text = " ".join([_decamel(local).replace("_", " "), self.label, self.comment,
                         " ".join(self.values)])
        return set(_TOKEN.findall(text.lower()))


def _decamel(s: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)


class SchemaIndex:
    RETRIEVALS = ("lexical", "tbox_entities")

    def __init__(self, ontology_path: str, retrieval: str = "lexical"):
        if retrieval not in self.RETRIEVALS:
            raise ValueError(f"retrieval must be one of {self.RETRIEVALS}")
        self.retrieval = retrieval
        g = Graph()
        g.parse(ontology_path, format="turtle")
        self.g = g
        self.items: list[SchemaItem] = []
        self.prefixes = {p: str(n) for p, n in g.namespaces() if p}
        self._build()
        for it in self.items:
            if it.kind == "property":
                it.values = self._values(it.uri)
        self._known = self._all_iris()
        self._entity_index = None

    def _lit(self, subj, pred) -> str:
        val = self.g.value(subj, pred)
        return str(val) if val else ""

    def _build(self) -> None:
        classes = {c for c in set(self.g.subjects(RDF.type, OWL.Class))
                   | set(self.g.subjects(RDF.type, RDFS.Class))
                   if isinstance(c, URIRef)}
        for cls in sorted(classes):
            self.items.append(SchemaItem(str(cls), "class",
                                         self._lit(cls, RDFS.label),
                                         self._lit(cls, RDFS.comment)))

        declared: set[str] = set()
        for pt in (OWL.ObjectProperty, OWL.DatatypeProperty, RDF.Property,
                   OWL.TransitiveProperty):
            for prop in sorted(self.g.subjects(RDF.type, pt)):
                if not isinstance(prop, URIRef) or str(prop) in declared:
                    continue
                declared.add(str(prop))
                self.items.append(SchemaItem(
                    str(prop), "property",
                    self._lit(prop, RDFS.label), self._lit(prop, RDFS.comment),
                    self._lit(prop, RDFS.domain), self._lit(prop, RDFS.range)))

        # Predicates used in the data but never declared.
        for p in sorted(set(self.g.predicates())):
            sp = str(p)
            if sp in declared or sp.startswith(STANDARD_NS):
                continue
            declared.add(sp)
            self.items.append(SchemaItem(sp, "property", "", "used in data, undeclared"))

        # Instances: typed by a domain class.
        for cls in sorted(classes):
            for inst in sorted(self.g.subjects(RDF.type, cls)):
                if isinstance(inst, URIRef):
                    self.items.append(SchemaItem(str(inst), "instance",
                                                 self._lit(inst, RDFS.label), "",
                                                 type_=str(cls)))

    MAX_VALUES = 8

    def _values(self, prop: str) -> tuple[str, ...]:
        """Distinct plain string values of a property, when they form a small
        closed set. Dates, numbers and free text are left out."""
        vals = set()
        for o in self.g.objects(predicate=URIRef(prop)):
            if not isinstance(o, Literal) or o.datatype not in (None, XSD.string):
                return ()
            vals.add(str(o))
            if len(vals) > self.MAX_VALUES:
                return ()
        return tuple(sorted(vals)) if all(len(v) <= 40 for v in vals) else ()

    def _all_iris(self) -> set[str]:
        out: set[str] = set()
        for s, p, o in self.g:
            for node in (s, p, o):
                if isinstance(node, URIRef):
                    out.add(str(node))
        return out

    # ---------------------------------------------------------------- prompts
    def prefix_header(self) -> str:
        return "\n".join(f"PREFIX {p}: <{n}>" for p, n in sorted(self.prefixes.items()))

    def relevant(self, question: str, k: int = 25) -> str:
        if self.retrieval == "tbox_entities":
            return self.tbox_entities(question)
        return self.lexical(question, k)

    def lexical(self, question: str, k: int = 25) -> str:
        """Lexical overlap retrieval over schema AND entities."""
        qbag = set(_TOKEN.findall(question.lower()))
        scored = sorted(
            self.items,
            key=lambda it: len(qbag & it.bag()) + 0.01 * (it.kind == "property"),
            reverse=True,
        )
        keep = [it for it in scored if qbag & it.bag()][:k]
        # Always give the model the schema backbone, even when no word matches.
        backbone = [it for it in self.items if it.kind in ("class", "property")]
        for it in backbone:
            if len(keep) >= k:
                break
            if it not in keep:
                keep.append(it)
        return "\n".join(it.render() for it in keep)

    # ------------------------------------------------- schema + entity linking
    def tbox_entities(self, question: str, k: int = 15) -> str:
        """Every class and property, then the entities the question names.

        On a graph with thousands of entities the lexical retrieval fills its
        25 slots with entities sharing a common word ("view", "column") and
        pushes out the properties the query needs. Here the schema is always
        complete (it is small) and only entities are retrieved, in two tiers:

        1. exact: a name of the entity appears in the question as a phrase. The
           names are its label and alternative labels, the values of the keys the ontology declares for
           its class (owl:hasKey: article number, alert number, view name), and
           "<class word> <key>" ("alert 54"). A key of fewer than 4 digits only
           counts with its class word, so "2" in "HANA 2.0" is not alert 2.
        2. soft: label words shared with the question, weighted by rarity (IDF),
           so "column", shared by 6 490 labels, weighs almost nothing.
        """
        names, bags, idf = self._entities()
        ql = " " + question.lower() + " "
        qbag = set(_TOKEN.findall(question.lower()))
        exact, soft = [], []
        for it in self.items:
            if it.kind != "instance":
                continue
            hit = max((len(n) for n in names[it.uri] if _phrase(n, ql)), default=0)
            if hit:
                exact.append((-hit, len(it.label), it.uri, it))
                continue
            sc = sum(idf[t] for t in qbag & bags[it.uri])
            if sc > 0:
                soft.append((-sc, len(it.label), it.uri, it))
        picked = [x[3] for x in sorted(exact)][:k]
        picked += [x[3] for x in sorted(soft)][:max(0, k - len(picked))]
        backbone = [it for it in self.items if it.kind in ("class", "property")]
        return "\n".join(it.render() for it in backbone + picked)

    def _entities(self):
        if self._entity_index is not None:
            return self._entity_index
        sub_of: dict[str, set[str]] = {}
        for c, sup in self.g.subject_objects(RDFS.subClassOf):
            sub_of.setdefault(str(c), set()).add(str(sup))

        def ancestors(c: str) -> set[str]:
            out, todo = {c}, [c]
            while todo:
                for p in sub_of.get(todo.pop(), ()):
                    if p not in out:
                        out.add(p)
                        todo.append(p)
            return out

        keys: dict[str, list[URIRef]] = {}
        for c, lst in self.g.subject_objects(OWL.hasKey):
            keys[str(c)] = list(Collection(self.g, lst))
        names: dict[str, set[str]] = {}
        bags: dict[str, set[str]] = {}
        df: Counter = Counter()
        for it in self.items:
            if it.kind != "instance":
                continue
            node = URIRef(it.uri)
            ns = {it.label.lower()} if it.label else set()
            ns.update(str(v).lower() for v in self.g.objects(node, SKOS.altLabel))
            classes = ancestors(it.type_)
            words = {self._lit(URIRef(c), RDFS.label).lower().split()[-1]
                     for c in classes if self._lit(URIRef(c), RDFS.label)}
            for c in classes:
                for kp in keys.get(c, ()):
                    for v in self.g.objects(node, kp):
                        v = str(v).lower()
                        if not v.isdigit() or len(v) >= 4:
                            ns.add(v)
                        ns.update(f"{w} {v}" for w in words)
            names[it.uri] = {n for n in ns if n}
            bags[it.uri] = set(_TOKEN.findall(it.label.lower().replace("_", " ")))
            df.update(bags[it.uri])
        n = max(1, len(bags))
        idf = {t: math.log(n / c) for t, c in df.items()}
        self._entity_index = (names, bags, idf)
        return self._entity_index

    def full(self) -> str:
        return "\n".join(it.render() for it in self.items)

    # ----------------------------------------------------------- symbolic gate
    def known_uris(self) -> set[str]:
        """Every IRI that exists in the graph. A query IRI outside this set is
        a hallucination, whether it is a class, a property or an entity."""
        return self._known

    def is_known(self, iri: str) -> bool:
        return iri in self._known or iri.startswith(STANDARD_NS)


def _phrase(name: str, text: str) -> bool:
    """`name` occurs in `text` as a whole phrase (both lower case)."""
    i = text.find(name)
    while i >= 0:
        before = text[i - 1] if i > 0 else " "
        after = text[i + len(name)] if i + len(name) < len(text) else " "
        if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
            return True
        i = text.find(name, i + 1)
    return False
