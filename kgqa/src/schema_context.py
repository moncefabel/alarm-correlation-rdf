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

import re
from dataclasses import dataclass

from rdflib import OWL, RDF, RDFS, XSD, Graph, Literal, URIRef

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
    def __init__(self, ontology_path: str):
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

    def full(self) -> str:
        return "\n".join(it.render() for it in self.items)

    # ----------------------------------------------------------- symbolic gate
    def known_uris(self) -> set[str]:
        """Every IRI that exists in the graph. A query IRI outside this set is
        a hallucination, whether it is a class, a property or an entity."""
        return self._known

    def is_known(self, iri: str) -> bool:
        return iri in self._known or iri.startswith(STANDARD_NS)
