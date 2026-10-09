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

from evaluate import canonical, label_index, prf  # noqa: E402
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
