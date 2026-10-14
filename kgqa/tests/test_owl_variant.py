"""The OWL variant of the alarm graph: same gold answers, plus the two
entailments the direction experiment is about."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kgqa" / "src"))
sys.path.insert(0, str(ROOT / "kgqa" / "scripts"))

from graph_store import GraphStore  # noqa: E402
from make_owl_variant import owl_variant  # noqa: E402
from schema_context import SchemaIndex  # noqa: E402

BASE = ROOT / "ontology" / "alarm_ontology.ttl"
VARIANT = ROOT / "ontology" / "alarm_ontology_owl.ttl"
QUESTIONS = yaml.safe_load((ROOT / "kgqa" / "benchmark" / "questions.yaml").read_text())["questions"]
P = "PREFIX net: <http://example.org/network#>\n"


@pytest.fixture(scope="module")
def store():
    return GraphStore(ttl_path=str(VARIANT))


def test_committed_variant_is_what_the_script_builds():
    built, added = owl_variant(BASE)
    committed = GraphStore(ttl_path=str(VARIANT)).graph
    assert set(built) - {t for t in built if t[0].__class__.__name__ == "BNode"} == \
        set(committed) - {t for t in committed if t[0].__class__.__name__ == "BNode"}
    assert added == 5                    # 4 causedBy + 1 reversed correlatedWith


@pytest.mark.parametrize("q", QUESTIONS, ids=[q["id"] for q in QUESTIONS])
def test_gold_answers_are_unchanged_on_the_variant(store, q):
    res = store.select(q["gold_sparql"])
    assert res.ok, res.error
    # as a set: K02 reads correlatedWith both ways, and now finds the pair twice
    assert sorted({v.split("#")[-1] for r in res.rows for v in r.values()}) == sorted(q["expected"])


def test_questions_can_now_be_written_in_the_direction_they_are_asked(store):
    # C03 "What directly caused the latency spike?"
    assert [r["c"] for r in store.select(P + "SELECT ?c WHERE { net:Alarm_Latency net:causedBy ?c }").rows] \
        == ["http://example.org/network#Alarm_CPU"]
    # K02: the correlation is readable from the packet drop's side
    assert store.select(P + "ASK { net:Alarm_PacketDrop net:correlatedWith net:Alarm_Latency }").rows \
        == [{"ask": "true"}]


def test_the_gate_knows_the_new_property_only_on_the_variant():
    assert SchemaIndex(str(VARIANT)).is_known("http://example.org/network#causedBy")
    assert not SchemaIndex(str(BASE)).is_known("http://example.org/network#causedBy")
