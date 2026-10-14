"""The semantic check arm: type check before execution, empty-answer repair
after, and the guarantee that a wrong check never costs an answer."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml
from rdflib import Graph

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kgqa" / "src"))
sys.path.insert(0, str(ROOT / "kgqa" / "tests"))

from agents import register_schema, run  # noqa: E402
from domains import ALARMS, SUPPORT  # noqa: E402
from evaluate import build_agent, evaluate  # noqa: E402
from graph_store import GraphStore  # noqa: E402
from schema_context import SchemaIndex  # noqa: E402
from semantic_check import EMPTY_HINT, TypeChecker  # noqa: E402

SUPPORT_GRAPH = ROOT / "supportkg" / "out_noinv" / "graph.ttl"    # no inverses: reversals bite
ALARM_GRAPH = ROOT / "ontology" / "alarm_ontology.ttl"
needs = pytest.mark.skipif(not SUPPORT_GRAPH.exists(), reason="support graph not built")
P = "PREFIX sup: <http://example.org/sapsupport#>\n"


def _q(user):
    return re.search(r"Question: (.+?)(?:\n|$)", user).group(1).strip()


@pytest.fixture(scope="module")
def support():
    schema = SchemaIndex(str(SUPPORT_GRAPH), retrieval=SUPPORT.retrieval)
    register_schema(schema)
    return schema, GraphStore(ttl_path=str(SUPPORT_GRAPH))


@needs
def test_reversed_triples_are_found_with_their_fix():
    g = Graph()
    g.parse(SUPPORT_GRAPH)
    tc = TypeChecker(g)
    [msg] = tc.check(P + "SELECT ?c WHERE { sup:View_M_PASSWORD_POLICY sup:columnOf ?c }")
    assert "wrong way round" in msg and "sup:hasColumn" in msg
    assert tc.check(P + "SELECT ?c WHERE { ?c sup:columnOf sup:View_M_PASSWORD_POLICY }") == []
    # a variable typed in the query is checked too
    [msg] = tc.check(P + "SELECT ?a WHERE { ?a a sup:Alert . ?a sup:addresses sup:Article_1977207 }")
    assert "sup:addressedBy" in msg


@needs
def test_no_gold_query_is_flagged():
    for graph, qfile in [(SUPPORT_GRAPH, ROOT / "supportkg" / "benchmark" / "questions.yaml"),
                         (ALARM_GRAPH, ROOT / "kgqa" / "benchmark" / "questions.yaml")]:
        g = Graph()
        g.parse(graph)
        tc = TypeChecker(g)
        for q in yaml.safe_load(qfile.read_text())["questions"]:
            assert tc.check(q["gold_sparql"]) == [], q["id"]


@needs
def test_a_reversed_query_is_repaired_only_by_the_checked_arm(support):
    schema, store = support
    question = "Which columns does the monitoring view M_PASSWORD_POLICY have?"
    wrong = P + "SELECT ?c WHERE { sup:View_M_PASSWORD_POLICY sup:columnOf ?c }"
    right = P + "SELECT ?c WHERE { ?c sup:columnOf sup:View_M_PASSWORD_POLICY }"

    def model(system, user):
        if system.startswith("Classify"):
            return "lookup"
        return right if "Type check:" in user else wrong

    plain = run(build_agent("orchestrator", model, schema, store, domain=SUPPORT),
                question, "orchestrator", store)
    assert plain.ok and plain.rows == []                  # valid, known IRIs, wrong way round
    checked = run(build_agent("orchestrator_checked", model, schema, store, domain=SUPPORT),
                  question, "orchestrator_checked", store)
    assert len(checked.rows) == 2 and checked.repairs == 1 and "type_error" in checked.steps


def test_an_empty_answer_gets_one_repair_on_the_alarm_graph():
    """Cause and effect are both alarms: only emptiness can reveal the reversal."""
    schema = SchemaIndex(str(ALARM_GRAPH))
    register_schema(schema)
    store = GraphStore(ttl_path=str(ALARM_GRAPH))
    q = "What directly caused the latency spike?"
    n = "PREFIX net: <http://example.org/network#>\n"
    effects = n + "SELECT ?c WHERE { net:Alarm_Latency net:causallyLinkedTo ?c }"   # reversed
    causes = n + "SELECT ?c WHERE { ?c net:causallyLinkedTo net:Alarm_Latency }"

    def model(system, user):
        if system.startswith("Classify"):
            return "causal"
        return causes if EMPTY_HINT[:40] in user else effects

    tr = run(build_agent("orchestrator_checked", model, schema, store), q,
             "orchestrator_checked", store)
    assert [r["c"].split("#")[-1] for r in tr.rows] == ["Alarm_CPU"]
    assert "empty_answer" in tr.steps and tr.repairs == 1

    def stubborn(system, user):                       # keeps the empty query: asked once only
        return "causal" if system.startswith("Classify") else effects
    tr = run(build_agent("orchestrator_checked", stubborn, schema, store), q,
             "orchestrator_checked", store)
    assert tr.rows == [] and tr.repairs == 1


@needs
def test_a_wrong_type_check_never_costs_the_answer(support, monkeypatch):
    """If the check flags a correct query on every attempt, the last attempt
    is executed anyway."""
    import semantic_check
    monkeypatch.setattr(semantic_check.TypeChecker, "check", lambda self, q: ["always wrong"])
    schema, store = support
    right = P + "SELECT ?c WHERE { ?c sup:columnOf sup:View_M_PASSWORD_POLICY }"
    tr = run(build_agent("orchestrator_checked",
                         lambda s, u: "lookup" if s.startswith("Classify") else right,
                         schema, store, domain=SUPPORT),
             "Which columns does the monitoring view M_PASSWORD_POLICY have?",
             "orchestrator_checked", store)
    assert len(tr.rows) == 2


@needs
def test_checked_arm_scores_perfectly_with_a_perfect_model(support):
    from test_domains import QUESTIONS, support_oracle
    schema, store = support
    s = evaluate("orchestrator_checked", QUESTIONS, schema, store, support_oracle,
                 domain=SUPPORT)["summary"]
    assert s["execution_accuracy"] == 1.0 and s["refusal_accuracy"] == 1.0
    assert s["mean_repairs"] == 0.0                   # no false alarm on any gold query


def test_checked_arm_on_the_alarm_benchmark_with_a_perfect_model():
    import test_topologies as T
    schema = SchemaIndex(T.ONTOLOGY)
    register_schema(schema)
    s = evaluate("orchestrator_checked", T.QUESTIONS, schema, GraphStore(ttl_path=T.ONTOLOGY),
                 T.oracle, domain=ALARMS)["summary"]
    assert s["execution_accuracy"] == 1.0 and s["mean_repairs"] == 0.0


@needs
def test_a_count_of_zero_is_an_empty_answer_only_when_asked(support):
    """Failure analysis of the first Haiku runs: on the graph without inverses,
    `COUNT(?c) WHERE { view hasColumn ?c }` returns one row, "0", so the
    empty-answer repair never fired. The zero-count arm sends it back once."""
    schema, store = support
    q = "How many columns does the monitoring view M_SAVEPOINTS have?"
    wrong = P + "SELECT (COUNT(?c) AS ?n) WHERE { sup:View_M_SAVEPOINTS sup:hasColumn ?c }"
    right = P + "SELECT (COUNT(?c) AS ?n) WHERE { ?c sup:columnOf sup:View_M_SAVEPOINTS }"

    def model(system, user):
        if system.startswith("Classify"):
            return "aggregation"
        return right if "count of 0" in user else wrong

    plain = run(build_agent("orchestrator_checked", model, schema, store, domain=SUPPORT), q,
                "orchestrator_checked", store)
    assert plain.rows[0]["n"] == "0" and plain.repairs == 0           # the gap
    zero = run(build_agent("orchestrator_checked_zero", model, schema, store, domain=SUPPORT), q,
               "orchestrator_checked_zero", store)
    assert zero.rows[0]["n"] == "27" and zero.repairs == 1 and "empty_answer" in zero.steps


def test_zero_count_shape():
    from semantic_check import zero_count
    assert zero_count([{"n": "0"}]) and zero_count([{"n": "0.0"}])
    assert not zero_count([]) and not zero_count([{"n": "3"}])
    assert not zero_count([{"n": "0"}, {"n": "0"}]) and not zero_count([{"v": "M_DISKS"}])
