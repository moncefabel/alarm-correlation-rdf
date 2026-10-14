"""Explainable answers: evidence triples and their sources."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kgqa" / "src"))
sys.path.insert(0, str(ROOT / "kgqa" / "tests"))

from explain import Explainer, bindings_query, patterns, summary  # noqa: E402

GRAPH = ROOT / "supportkg" / "out" / "graph.ttl"
NOINV = ROOT / "supportkg" / "out_noinv" / "graph.ttl"
needs = pytest.mark.skipif(not GRAPH.exists(), reason="support graph not built")
QUESTIONS = yaml.safe_load((ROOT / "supportkg" / "benchmark" / "questions.yaml").read_text())["questions"]
GOLD = {q["id"]: q["gold_sparql"] for q in QUESTIONS}
P = "PREFIX sup: <http://example.org/sapsupport#>\n"


@pytest.fixture(scope="module")
def ex():
    return Explainer.for_graph(GRAPH)


def test_bindings_query_keeps_the_where_clause_whole():
    q = 'PREFIX x: <http://e.org/a#>\nSELECT (COUNT(?a) AS ?n) WHERE { ?a x:p "a } b" ; x:q <http://e.org/a#b> . } GROUP BY ?a'
    assert bindings_query(q) == ('PREFIX x: <http://e.org/a#>\nSELECT * WHERE '
                                 '{ ?a x:p "a } b" ; x:q <http://e.org/a#b> . } LIMIT 100')


def test_patterns_skip_what_must_be_absent():
    pats = patterns(GOLD["N01"])            # ?art a KnowledgeArticle . FILTER NOT EXISTS {...}
    assert [str(p).split("#")[-1] for _, p, _ in pats] == ["type"]


@needs
@pytest.mark.parametrize("graph", [GRAPH, NOINV], ids=["materialised", "no_inverses"])
def test_every_gold_answer_is_fully_sourced(graph):
    e = Explainer.for_graph(graph)
    for q in QUESTIONS:
        if q["type"] == "out_of_scope":
            continue
        s = summary(e.explain(q["gold_sparql"]))
        assert s["evidence_triples"] > 0 and s["sourced"] == s["evidence_triples"], (q["id"], s)


@needs
def test_a_cross_source_answer_cites_the_article_and_the_documentation(ex):
    ev = {str(e.triple[1]).split("#")[-1]: e for e in ex.explain(GOLD["M01"])}
    assert ev["addresses"].kind == "extracted" and ev["addresses"].method == "title-rule"
    assert ev["addresses"].sources == ["https://userapps.support.sap.com/sap/support/knowledge/en/1977207"]
    link = ev["investigatedWith"]
    assert link.method == "lexical-linker" and 0 < link.confidence < 1
    assert any("github.com/SAP-docs" in u for u in link.sources)
    assert any("1977207" in u for u in link.sources)


@needs
def test_an_inferred_triple_names_its_axiom_and_its_premise(ex):
    q = P + "SELECT ?c WHERE { sup:View_M_PASSWORD_POLICY sup:hasColumn ?c }"
    ev = ex.explain(q)
    assert len(ev) == 2 and all(e.kind == "inferred" for e in ev)
    assert ev[0].method == "hasColumn owl:inverseOf columnOf"
    assert ev[0].premise.kind == "page" and "m-password-policy" in ev[0].premise.sources[0]
    # same question on the graph without inverses: nothing matches, nothing to explain
    assert Explainer.for_graph(NOINV).explain(q) == []


def test_a_transitive_path_is_explained_link_by_link():
    """No provenance on the alarm graph, but the chain itself is the evidence."""
    alarm = yaml.safe_load((ROOT / "kgqa" / "benchmark" / "questions.yaml").read_text())["questions"]
    c02 = next(q for q in alarm if q["id"] == "C02")["gold_sparql"]
    e = Explainer(str(ROOT / "ontology" / "alarm_ontology.ttl"))
    chain = [(s.split("#")[-1], o.split("#")[-1]) for s, _, o in e.evidence_triples(c02)]
    assert chain == [("Alarm_FirewallSpike", "Alarm_CPU"), ("Alarm_CPU", "Alarm_PacketDrop"),
                     ("Alarm_PacketDrop", "Alarm_WebTimeout")]


@needs
def test_evaluate_reports_how_many_correct_answers_are_explained():
    from agents import register_schema
    from domains import SUPPORT
    from evaluate import evaluate
    from graph_store import GraphStore
    from schema_context import SchemaIndex
    from test_domains import support_oracle
    schema = SchemaIndex(str(GRAPH), retrieval=SUPPORT.retrieval)
    register_schema(schema)
    res = evaluate("orchestrator", QUESTIONS, schema, GraphStore(ttl_path=str(GRAPH)),
                   support_oracle, explainer=Explainer.for_graph(GRAPH), domain=SUPPORT)
    assert res["summary"]["explained_correct_rate"] == 1.0
    m01 = next(t for t in res["traces"] if t["question"].startswith("Which view should be queried"))
    assert {e["kind"] for e in m01["evidence"]} == {"extracted"}
