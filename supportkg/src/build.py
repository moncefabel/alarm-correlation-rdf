"""Build the support knowledge graph from parsed article previews.

    python supportkg/src/build.py                       # rules only
    python supportkg/src/build.py --linker hf           # + Hugging Face embedding linker

Steps:
  1. extract facts from every record (extract.py), in two passes: the first
     collects alert names from how-to titles, so the embedding linker of the
     second pass knows which alerts exist;
  2. entity resolution: one IRI per article number, alert number and component
     code, whichever article mentions them; product and tool mentions resolve to
     the PPMS stand-in and tools.ttl;
  3. materialise the inverse and symmetric axioms of the ontology (an OWL RL
     subset), so a SPARQL query reading a relation in either direction finds it
     without a reasoner in the store;
  4. provenance: every extracted triple gets a statement node in
     provenance.ttl with its source page, extraction method and confidence;
  5. SHACL validation of the result.

Outputs in supportkg/out/: graph.ttl (ontology, PPMS stand-in, tools and
facts, ready for kgqa), provenance.ttl, report.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import DCTERMS, OWL, PROV, SKOS, XSD

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from extract import Extraction, extract, gazetteers  # noqa: E402
from parse import Record  # noqa: E402

ROOT = HERE.parent
ONTO = ROOT / "ontology"
SUP = Namespace("http://example.org/sapsupport#")
PPMS = Namespace("http://example.org/ppms#")
PREVIEW = "https://userapps.support.sap.com/sap/support/knowledge/en/{n}"
ONTOLOGY_FILES = ("support.ttl", "ppms_standin.ttl", "tools.ttl")


def iri(key) -> URIRef:
    if isinstance(key, str):
        return URIRef(key)
    kind, val = key
    if kind == "article":
        return SUP[f"Article_{val}"]
    if kind == "alert":
        return SUP[f"Alert_{val}"]
    if kind == "component":
        return SUP["Component_" + str(val).replace("-", "_")]
    raise ValueError(key)


def ontology() -> Graph:
    g = Graph()
    for f in ONTOLOGY_FILES:
        g.parse(ONTO / f)
    return g


def extract_all(records: list[Record], linker_factory=None) -> list[Extraction]:
    first = [extract(r) for r in records]
    if linker_factory is None:
        return first
    names = {}
    for ex in first:
        for f in ex.facts:
            if f.s[0] == "alert" and f.p == "label":
                names.setdefault(f.s[1], f.o)
    linker = linker_factory(gazetteers()["PRODUCT"], names)
    return [extract(r, linker) for r in records]


def materialise(g: Graph, onto: Graph) -> int:
    """owl:inverseOf (both directions) and owl:SymmetricProperty, to a fixpoint
    of one step each, which is all these two rules need."""
    added = 0
    pairs = [(p, q) for p, q in onto.subject_objects(OWL.inverseOf)]
    pairs += [(q, p) for p, q in pairs]
    sym = set(onto.subjects(RDF.type, OWL.SymmetricProperty))
    for p, q in pairs:
        for s, o in list(g.subject_objects(p)):
            if (o, q, s) not in g:
                g.add((o, q, s)); added += 1
    for p in sym:
        for s, o in list(g.subject_objects(p)):
            if (o, p, s) not in g:
                g.add((o, p, s)); added += 1
    return added


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def add_docs(data: Graph, docs) -> dict[str, URIRef]:
    """Help-documentation pages as typed nodes, with columns and related links.
    Provenance is per page (sup:sourceUrl at a fixed commit) and per source
    (dct:source, carrying the CC-BY-4.0 licence)."""
    by_key: dict[str, URIRef] = {}
    kinds = {"monitoring_view": SUP.MonitoringView, "system_view": SUP.SystemView,
             "sql_statement": SUP.SQLStatement}
    for d in docs:
        node = SUP[("Statement_" if d.kind == "sql_statement" else "View_") + slug(d.name)]
        by_key[d.key] = node
        data.add((node, RDF.type, kinds[d.kind]))
        data.add((node, RDFS.label, Literal(d.name)))
        if d.description:
            data.add((node, RDFS.comment, Literal(d.description)))
        data.add((node, SUP.sourceUrl, Literal(d.url, datatype=XSD.anyURI)))
        data.add((node, DCTERMS.source, SUP.Source_HANACloudDocs))
        if d.kind == "sql_statement":
            data.add((node, SUP.statementCategory, Literal(d.category)))
            continue
        data.add((node, SUP.viewName, Literal(d.name)))
        for col, dtype, desc in d.columns:
            c = SUP[f"Column_{slug(d.name)}__{slug(col)}"]
            data.add((c, RDF.type, SUP.Column))
            data.add((c, SUP.columnName, Literal(col)))
            data.add((c, SUP.dataType, Literal(dtype or "unspecified")))
            data.add((c, RDFS.label, Literal(f"{col} column of {d.name}")))
            if desc:
                data.add((c, RDFS.comment, Literal(desc)))
            data.add((c, SUP.columnOf, node))
    for d in docs:
        for k in d.related:
            data.add((by_key[d.key], SUP.seeAlso, by_key[k]))
    return by_key


def build_graph(records: list[Record], linker_factory=None, materialise_axioms: bool = True,
                docs=None, view_linker_factory=None):
    onto = ontology()
    data = Graph()
    prov = Graph()
    for g in (data, prov):
        g.bind("sup", SUP); g.bind("ppms", PPMS); g.bind("prov", PROV); g.bind("skos", SKOS)

    extractions = extract_all(records, linker_factory)
    by_number = {r.number: r for r in records}
    alert_names: dict[int, Counter] = defaultdict(Counter)
    comp_labels: dict[str, Counter] = defaultdict(Counter)
    articles, alerts, components = set(), set(), set()
    report = {"articles_collected": len(records), "facts": 0, "by_method": Counter(),
              "by_relation": Counter(), "unlinked_products": Counter()}

    def note(key):
        if isinstance(key, tuple):
            {"article": articles, "alert": alerts, "component": components}[key[0]].add(key[1])

    for ex in extractions:
        rec = by_number[ex.number]
        src = URIRef(rec.url or PREVIEW.format(n=rec.number))
        for p in ex.unlinked_products:
            report["unlinked_products"][p] += 1
        for f in ex.facts:
            note(f.s); note(f.o)
            s = iri(f.s)
            if f.p == "label":
                if f.s[0] == "alert":
                    alert_names[f.s[1]][f.o] += 1
                elif f.s[0] == "component":
                    comp_labels[f.s[1]][f.o] += 1
                else:
                    data.add((s, RDFS.label, Literal(f.o)))
                continue
            if f.p == "sourceUrl":
                data.add((s, SUP.sourceUrl, Literal(f.o, datatype=XSD.anyURI)))
                continue
            if f.p == "type":
                data.add((s, RDF.type, SUP.KBA if f.o == "KBA" else SUP.SAPNote))
                continue
            o = iri(f.o) if isinstance(f.o, (tuple, str)) else Literal(f.o)
            triple = (s, SUP[f.p], o)
            data.add(triple)
            st = URIRef(SUP + "stmt_" + hashlib.sha1(
                f"{s}|{f.p}|{o}|{ex.number}|{f.method}".encode()).hexdigest()[:16])
            prov.add((st, RDF.type, RDF.Statement))
            prov.add((st, RDF.subject, s)); prov.add((st, RDF.predicate, SUP[f.p]))
            prov.add((st, RDF.object, o))
            prov.add((st, PROV.wasDerivedFrom, src))
            prov.add((st, SUP.extractionMethod, Literal(f.method)))
            prov.add((st, SUP.confidence, Literal(f.confidence, datatype=XSD.decimal)))
            report["facts"] += 1
            report["by_method"][f.method] += 1
            report["by_relation"][f.p] += 1

    # entity resolution: one node per key, typed, with an identifier and a name
    for n in articles:
        a = iri(("article", n))
        if (a, RDF.type, SUP.KBA) not in data and (a, RDF.type, SUP.SAPNote) not in data:
            data.add((a, RDF.type, SUP.KnowledgeArticle))
        data.add((a, SUP.articleNumber, Literal(n)))
        if n not in by_number:                       # cited, not collected
            data.add((a, RDFS.label, Literal(f"SAP Note or KBA {n} (cited, not collected)")))
            data.add((a, SUP.sourceUrl, Literal(PREVIEW.format(n=n), datatype=XSD.anyURI)))
    for n in alerts:
        a = iri(("alert", n))
        data.add((a, RDF.type, SUP.Alert))
        data.add((a, SUP.alertNumber, Literal(int(n), datatype=XSD.integer)))
        names = [nm for nm, _ in alert_names[n].most_common()]
        data.add((a, RDFS.label, Literal(names[0] if names else f"SAP HANA alert {n}")))
        for alt in names[1:]:
            data.add((a, SKOS.altLabel, Literal(alt)))
    for code in components:
        c = iri(("component", code))
        data.add((c, RDF.type, SUP.ApplicationComponent))
        data.add((c, SUP.componentCode, Literal(code)))
        labels = [lb for lb, _ in comp_labels[code].most_common()]
        data.add((c, RDFS.label, Literal(labels[0] if labels else code)))
        data.add((c, SKOS.notation, Literal(code)))

    # help documentation, then the cross-source link alert -> view
    report["entities"] = {"articles": len(articles), "alerts": len(alerts),
                          "components": len(components)}
    if docs:
        add_docs(data, docs)
        views = {d.name: d.description for d in docs if d.kind != "sql_statement"}
        report["entities"].update(
            views=len(views), statements=sum(d.kind == "sql_statement" for d in docs),
            columns=sum(len(d.columns) for d in docs))
        if view_linker_factory is not None:
            linker = view_linker_factory(views)
            urls = {d.name: d.url for d in docs}
            alert_srcs: dict[int, list[str]] = defaultdict(list)   # every article naming it
            for ex in extractions:
                for f in ex.facts:
                    if f.p == "addresses":
                        alert_srcs[f.o[1]].append(by_number[ex.number].url
                                                  or PREVIEW.format(n=ex.number))
            for n in sorted(alerts):
                label = data.value(iri(("alert", n)), RDFS.label)
                for view, score in linker.link(str(label)):
                    s, o = iri(("alert", n)), SUP["View_" + slug(view)]
                    data.add((s, SUP.investigatedWith, o))
                    st = URIRef(SUP + "stmt_" + hashlib.sha1(
                        f"{s}|investigatedWith|{o}|{linker.method}".encode()).hexdigest()[:16])
                    prov.add((st, RDF.type, RDF.Statement))
                    prov.add((st, RDF.subject, s)); prov.add((st, RDF.predicate, SUP.investigatedWith))
                    prov.add((st, RDF.object, o))
                    for src in alert_srcs.get(n, []):
                        prov.add((st, PROV.wasDerivedFrom, URIRef(src)))
                    prov.add((st, PROV.wasDerivedFrom, URIRef(urls[view])))
                    prov.add((st, SUP.extractionMethod, Literal(f"{linker.method}-linker")))
                    prov.add((st, SUP.confidence, Literal(round(score, 3), datatype=XSD.decimal)))
                    report["facts"] += 1
                    report["by_method"][f"{linker.method}-linker"] += 1
                    report["by_relation"]["investigatedWith"] += 1

    report["inferred_triples"] = materialise(data, onto) if materialise_axioms else 0
    full = onto + data
    for p, ns in (("sup", SUP), ("ppms", PPMS), ("skos", SKOS), ("prov", PROV), ("dct", DCTERMS)):
        full.bind(p, ns)
    return full, prov, report


def validate(g: Graph) -> tuple[bool, int, str]:
    import pyshacl
    ok, results, text = pyshacl.validate(g, shacl_graph=str(ONTO / "shapes.ttl"),
                                         inference="none", advanced=False)
    n = len(list(results.subjects(RDF.type, URIRef("http://www.w3.org/ns/shacl#ValidationResult"))))
    return ok, n, text


def load_records(path: Path) -> list[Record]:
    return [Record.from_dict(json.loads(ln)) for ln in path.read_text().splitlines() if ln.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", default=str(ROOT / "data" / "parsed" / "articles.jsonl"))
    ap.add_argument("--docs", default=str(ROOT / "data" / "sap-hana-cloud-hana-database"),
                    help="Clone of SAP-docs/sap-hana-cloud-hana-database; 'none' to leave docs out.")
    ap.add_argument("--out", default=str(ROOT / "out"))
    ap.add_argument("--linker", choices=["none", "hf"], default="none",
                    help="Embedding linker for products and unnumbered alerts in articles.")
    ap.add_argument("--view-linker", choices=["none", "lexical", "hf"], default="lexical",
                    help="How alerts are linked to the views that investigate them.")
    ap.add_argument("--no-materialise", action="store_true")
    args = ap.parse_args()

    records = load_records(Path(args.parsed))
    factory = None
    if args.linker == "hf":
        from link import EmbeddingLinker
        factory = lambda products, alerts: EmbeddingLinker(products, alerts)  # noqa: E731
    docs = None
    if args.docs != "none":
        from docs import load
        docs = load(Path(args.docs))
    view_factory = None
    if docs and args.view_linker != "none":
        from link import ViewLinker
        method = "lexical" if args.view_linker == "lexical" else "embedding"
        view_factory = lambda views: ViewLinker(views, method=method)  # noqa: E731
    graph, prov, report = build_graph(records, factory, not args.no_materialise,
                                      docs=docs, view_linker_factory=view_factory)
    ok, n, text = validate(graph)
    report.update(shacl_conforms=ok, shacl_violations=n, triples=len(graph),
                  linker=args.linker, view_linker=args.view_linker if docs else "none")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    graph.serialize(out / "graph.ttl", format="turtle")
    prov.serialize(out / "provenance.ttl", format="turtle")
    (out / "report.json").write_text(json.dumps(report, indent=1, default=dict))
    if not ok:
        (out / "shacl_report.txt").write_text(text)
    print(json.dumps(report, indent=1, default=dict))


if __name__ == "__main__":
    main()
