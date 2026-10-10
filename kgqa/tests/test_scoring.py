"""Scoring, replayed on queries a real model (gpt-oss-20b on Groq) wrote.

The first smoke run scored a correct answer zero because the model projected
rdfs:label instead of the IRI, and halved precision when it projected both.
These tests pin the fix: an entity counts once, however the query names it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kgqa" / "src"))

from evaluate import canonical, label_index, prf, score  # noqa: E402
from graph_store import GraphStore  # noqa: E402

ONTOLOGY = str(ROOT / "ontology" / "alarm_ontology.ttl")
GOLD = {q["id"]: q for q in
        yaml.safe_load((ROOT / "kgqa" / "benchmark" / "questions.yaml").read_text())["questions"]}
P = ("PREFIX net: <http://example.org/network#>\n"
     "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n")


@pytest.fixture(scope="module")
def env():
    store = GraphStore(ttl_path=ONTOLOGY)
    return store, label_index(store)


def f1(env, qid, query):
    store, labels = env
    pred, gold = store.select(P + query), store.select(GOLD[qid]["gold_sparql"])
    assert pred.ok, pred.error
    return prf(canonical(pred.rows, labels), canonical(gold.rows, labels))[2]


def test_labels_instead_of_iris_score_as_correct(env):
    # F01, verbatim from the smoke run: right alarm, given by its label
    q = """SELECT ?alarmLabel WHERE { ?alarm a net:Alarm ; rdfs:label ?alarmLabel ;
                                      net:affectsDevice net:Firewall_Edge . }"""
    assert f1(env, "F01", q) == 1.0


def test_projecting_iri_and_label_is_not_penalised(env):
    q = "SELECT ?a ?l WHERE { ?a net:affectsDevice net:Firewall_Edge ; rdfs:label ?l . }"
    assert f1(env, "F01", q) == 1.0


def test_non_transitive_answer_stays_partial(env):
    # C04, verbatim: direct links only, misses the rest of the chain.
    # The fix must not turn a genuinely incomplete answer into a correct one.
    q = """SELECT ?alarm ?label WHERE { net:Alarm_FirewallSpike net:causallyLinkedTo ?alarm .
                                        ?alarm rdfs:label ?label . }"""
    assert 0.0 < f1(env, "C04", q) < 1.0


def test_wrong_entity_still_scores_zero(env):
    q = "SELECT ?l WHERE { net:Router_Core rdfs:label ?l . }"
    assert f1(env, "K03", q) == 0.0


def test_ambiguous_labels_are_not_mapped(env):
    _, labels = env
    store, _ = env
    rows = store.select(P + "SELECT ?s ?l WHERE { ?s rdfs:label ?l }").rows
    counts: dict[str, int] = {}
    for r in rows:
        counts[str(r["l"]).lower()] = counts.get(str(r["l"]).lower(), 0) + 1
    assert all(counts[lbl] == 1 for lbl in labels)


def test_literals_such_as_counts_are_untouched(env):
    q = 'SELECT (COUNT(?a) AS ?n) WHERE { ?a net:hasSeverity "CRITICAL" . }'
    assert f1(env, "F02", q) == 1.0


def test_interrupted_run_resumes_without_repeating_questions(env, tmp_path):
    """A provider outage or an exhausted daily quota stops a pass halfway.
    Scored questions are checkpointed, and the rerun asks the model only for
    the remaining ones, with the same final scores as an uninterrupted run."""
    from agents import REFUSAL, register_schema
    from evaluate import evaluate
    from schema_context import SchemaIndex

    store, _ = env
    schema = SchemaIndex(ONTOLOGY)
    register_schema(schema)
    qs = [GOLD[i] for i in ("F01", "F02", "C04", "S01")]
    by_text = {q["question"]: q for q in qs}

    def oracle(system, user):
        q = next(v for k, v in by_text.items() if k in user)
        return REFUSAL if q["type"] == "out_of_scope" else q["gold_sparql"]

    calls = []

    def flaky(system, user):
        calls.append(user)
        if len(calls) > 2:
            raise RuntimeError("provider down")
        return oracle(system, user)

    ckpt = tmp_path / "react_run0.partial.jsonl"
    with pytest.raises(RuntimeError):
        evaluate("react", qs, schema, store, flaky, checkpoint=ckpt)
    assert len(ckpt.read_text().splitlines()) == 2

    asked = []

    def counting(system, user):
        asked.append(user)
        return oracle(system, user)

    res = evaluate("react", qs, schema, store, counting, checkpoint=ckpt)
    assert len(asked) == 2                               # only C04 and S01
    assert [r["id"] for r in res["rows"]] == ["F01", "F02", "C04", "S01"]
    assert res["summary"]["execution_accuracy"] == 1.0
    assert res["summary"]["refusal_accuracy"] == 1.0



# ------------------------------------------- pilot run (gpt-oss-120b, Groq)
# Queries below are verbatim from the first full pass. Each test pins how the
# revised scorer reads them, in both directions: correct answers that were
# penalised by form, and wrong answers that must stay wrong.

def lenient(env, qid, query):
    store, labels = env
    pred, gold = store.select(P + query), store.select(GOLD[qid]["gold_sparql"])
    assert pred.ok, pred.error
    return score(pred.rows, gold.rows, labels)


def test_extra_time_column_no_longer_penalised(env):
    q = """SELECT ?alarm ?time WHERE { ?alarm a net:Alarm ; net:triggeredAt ?time . }
           ORDER BY ASC(?time) LIMIT 1"""
    p, r, f, strict = lenient(env, "F06", q)
    assert f == 1.0 and not strict          # lenient credits it, strict does not


def test_extra_count_column_no_longer_penalised(env):
    q = """SELECT ?device (COUNT(?alarm) AS ?alarmCount) WHERE {
             ?alarm a net:Alarm ; net:affectsDevice ?device . }
           GROUP BY ?device ORDER BY DESC(?alarmCount) LIMIT 1"""
    assert lenient(env, "F05", q)[2] == 1.0


def test_empty_select_answers_a_yes_no_question_with_no(env):
    # K01 (react): lists cause/effect pairs, finds none: "no" is the right answer
    q = """SELECT ?cause ?effect WHERE {
             { net:Alarm_Latency net:causallyLinkedTo net:Alarm_PacketDrop
               BIND(net:Alarm_Latency AS ?cause) BIND(net:Alarm_PacketDrop AS ?effect) }
             UNION
             { net:Alarm_PacketDrop net:causallyLinkedTo net:Alarm_Latency
               BIND(net:Alarm_PacketDrop AS ?cause) BIND(net:Alarm_Latency AS ?effect) } }"""
    assert lenient(env, "K01", q)[2] == 1.0


def test_count_answers_a_yes_no_question(env):
    # C06 (react): counts paths instead of ASK; 1 path means yes
    q = "SELECT (COUNT(*) AS ?n) WHERE { net:Alarm_CPU net:causallyLinkedTo+ net:Alarm_WebTimeout . }"
    assert lenient(env, "C06", q)[2] == 1.0
    # and a zero count means no, it is not read as "something was returned"
    q0 = "SELECT (COUNT(*) AS ?n) WHERE { net:Alarm_Latency net:causallyLinkedTo+ net:Alarm_WebTimeout . }"
    assert lenient(env, "C06", q0)[2] == 0.0


def test_yes_no_literal_is_read_as_boolean(env):
    # C07 (react) answered with IF(EXISTS {...}, "yes", "no"). rdflib cannot
    # execute EXISTS with a property path inside IF (Fuseki can), so the value
    # is produced directly here; the point is how "no" is read.
    q = 'SELECT ?causes WHERE { VALUES ?causes { "no" } }'
    assert lenient(env, "C07", q)[2] == 1.0
    q = 'SELECT ?causes WHERE { VALUES ?causes { "yes" } }'
    assert lenient(env, "C07", q)[2] == 0.0


def test_revised_x06_has_natural_polarity(env):
    q = "ASK { net:Alarm_Latency net:causallyLinkedTo+ net:Alarm_WebTimeout . }"
    assert lenient(env, "X06", q)[2] == 1.0


def test_wrong_answers_stay_wrong(env):
    # C04: direct links only, the chain is incomplete
    q = """SELECT ?alarm ?label WHERE { net:Alarm_FirewallSpike net:causallyLinkedTo ?alarm .
                                        ?alarm rdfs:label ?label . }"""
    assert 0.0 < lenient(env, "C04", q)[2] < 1.0
    # F02: wrong spelling of the severity, the count is 0, not 1
    q = 'SELECT (COUNT(?a) AS ?n) WHERE { ?a a net:Alarm ; net:hasSeverity "critical" . }'
    assert lenient(env, "F02", q)[2] == 0.0
    # a yes/no question answered with the wrong boolean
    assert lenient(env, "C06", "ASK { net:Alarm_Latency net:causallyLinkedTo+ net:Alarm_WebTimeout }")[2] == 0.0


def test_dumping_the_graph_is_not_rewarded(env):
    q = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"
    assert lenient(env, "F01", q)[2] < 0.1


def test_severity_values_are_in_the_model_context():
    from schema_context import SchemaIndex
    ctx = SchemaIndex(ONTOLOGY).relevant("How many alarms have critical severity?")
    line = next(l for l in ctx.splitlines() if "hasSeverity" in l)
    assert '"CRITICAL"' in line and '"HIGH"' in line
