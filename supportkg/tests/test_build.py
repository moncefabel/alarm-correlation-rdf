"""Graph construction: resolution, OWL materialisation, provenance, SHACL,
and the link to the embedding linker and to the kgqa agents."""

from __future__ import annotations

import sys

import pytest
from rdflib import Literal, RDF, URIRef
from rdflib.namespace import XSD

from conftest import ROOT, bow_encoder

SUP = "http://example.org/sapsupport#"
P = "PREFIX sup: <http://example.org/sapsupport#>\nPREFIX ppms: <http://example.org/ppms#>\n" \
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"


@pytest.fixture(scope="module")
def built(records):
    from build import build_graph
    return build_graph(records)


def ask(g, q):
    return bool(g.query(P + q).askAnswer)


def rows(g, q):
    return {tuple(str(v) for v in r) for r in g.query(P + q)}


def test_graph_conforms_to_the_shapes(built):
    from build import validate
    ok, n, text = validate(built[0])
    assert ok, text


def test_one_node_per_key_whoever_mentions_it(built):
    g = built[0]
    # HAN-DB is implied by two different articles' codes: still one node
    assert rows(g, "SELECT ?c WHERE { ?c sup:componentCode 'HAN-DB' }") == {(SUP + "Component_HAN_DB",)}
    assert (URIRef(SUP + "Alert_54"), URIRef(SUP + "alertNumber"), Literal(54, datatype=XSD.integer)) in g


def test_inverse_and_symmetric_axioms_are_materialised(built):
    g = built[0]
    assert ask(g, "ASK { sup:Alert_54 sup:addressedBy sup:Article_9900001 }")
    assert ask(g, "ASK { sup:Alert_54 sup:cooccursWith sup:Alert_39 }")
    assert ask(g, "ASK { sup:Alert_39 sup:cooccursWith sup:Alert_54 }")
    assert ask(g, "ASK { ppms:PV_HANA_Platform_2_0 sup:hasApplicableArticle sup:Article_9900002 }")
    assert ask(g, "ASK { sup:Article_9900002 sup:referencedBy sup:Article_9900001 }")


def test_materialisation_can_be_switched_off_for_ablation(records):
    from build import build_graph
    g, _, report = build_graph(records, materialise_axioms=False)
    assert report["inferred_triples"] == 0
    assert not ask(g, "ASK { sup:Alert_54 sup:addressedBy ?a }")


def test_component_hierarchy_is_transitive_through_a_property_path(built):
    g = built[0]
    found = rows(g, "SELECT ?a WHERE { ?a sup:assignedTo/sup:subComponentOf* sup:Component_HAN_DB }")
    assert found == {(SUP + "Article_9900001",), (SUP + "Article_9900002",), (SUP + "Article_9900003",)}


def test_cited_articles_become_nodes_that_say_so(built):
    g = built[0]
    lbl = rows(g, "SELECT ?l WHERE { sup:Article_9900004 rdfs:label ?l }")
    assert lbl == {("SAP Note or KBA 9900004 (cited, not collected)",)}


def test_every_extracted_triple_has_provenance(built):
    g, prov, report = built
    stmts = list(prov.subjects(RDF.type, RDF.Statement))
    assert len(stmts) == report["facts"] > 0
    methods = {str(o) for o in prov.objects(None, URIRef(SUP + "extractionMethod"))}
    assert {"title-rule", "regex", "keyword-rule", "gazetteer", "code-prefix"} <= methods


def test_embedding_linker_adds_what_rules_miss(records):
    from build import build_graph
    from link import EmbeddingLinker

    def factory(products, alerts):
        return EmbeddingLinker(products, {**alerts, 1: "Host physical memory usage",
                                          43: "Memory usage of services"},
                               encoder=bow_encoder(), product_threshold=0.6, alert_threshold=0.3)
    g, prov, report = build_graph(records, factory)
    # the bag-of-words stand-in cannot tell "2" from the product name; a real
    # sentence encoder is scored on the annotated set, not here
    linked = rows(g, "SELECT ?p WHERE { sup:Article_9900003 sup:appliesTo ?p }")
    assert linked and linked <= {("http://example.org/ppms#PV_HANA_Platform_2_0",),
                                 ("http://example.org/ppms#Product_HANA_Platform",)}
    mentioned = rows(g, "SELECT ?a WHERE { sup:Article_9900003 sup:mentionsAlert ?a }")
    assert (SUP + "Alert_43",) in mentioned and (SUP + "Alert_1",) in mentioned
    assert report["by_method"]["embedding"] >= 3
    assert not report["unlinked_products"]


def test_kgqa_agents_can_read_the_graph(built, tmp_path):
    sys.path.insert(0, str(ROOT.parent / "kgqa" / "src"))
    from schema_context import SchemaIndex
    path = tmp_path / "graph.ttl"
    built[0].serialize(path, format="turtle")
    idx = SchemaIndex(str(path))
    ctx = idx.relevant("Which article addresses the savepoint duration alert?")
    assert "addresses" in ctx and "Savepoint Duration" in ctx
    assert idx.is_known(SUP + "addressedBy")
