"""Both topologies, end to end through LangGraph, with a scripted LLM.

No API key and no network: the LLM is replaced by a function that answers from
the benchmark. This exercises the control flow, the symbolic gate, the repair
loop and the refusal path, which is everything the comparison depends on except
the model itself.

The key test is test_invented_uri_*: an invented IRI does not make SPARQL fail,
it makes it return nothing. A loop that only repairs on execution errors
therefore accepts the empty answer. The orchestrator's gate rejects the IRI
before execution and triggers a repair. That difference is the mechanism the
benchmark is designed to measure on a real model.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kgqa" / "src"))

from agents import (REFUSAL, build_orchestrator, build_react,  # noqa: E402
                    extract_uris, hallucinated, register_schema, run)
from evaluate import evaluate  # noqa: E402
from graph_store import GraphStore  # noqa: E402
from schema_context import SchemaIndex  # noqa: E402

ONTOLOGY = str(ROOT / "ontology" / "alarm_ontology.ttl")
QUESTIONS = yaml.safe_load((ROOT / "kgqa" / "benchmark" / "questions.yaml").read_text())["questions"]
BY_TEXT = {q["question"]: q for q in QUESTIONS}

INVENTED = """PREFIX net: <http://example.org/network#>
SELECT ?a WHERE { ?a net:raisedOn net:FW1 . }"""


@pytest.fixture(scope="module")
def env():
    schema = SchemaIndex(ONTOLOGY)
    register_schema(schema)
    return schema, GraphStore(ttl_path=ONTOLOGY)


def _question(user: str) -> str:
    m = re.search(r"Question: (.+?)(?:\n|$)", user)
    return m.group(1).strip() if m else ""


def oracle(system: str, user: str) -> str:
    """Perfect model: right class, gold query, refusal when out of scope."""
    q = BY_TEXT[_question(user)]
    if system.startswith("Classify"):
        return q["type"] if q["type"] != "contrast" else "causal"
    return REFUSAL if q["type"] == "out_of_scope" else q["gold_sparql"]


def first_invents(system: str, user: str) -> str:
    """Invents IRIs on the first attempt, repairs correctly once told."""
    if system.startswith("Classify"):
        return "factual"
    if "Failed query" in user or "Previous attempt failed" in user:
        return BY_TEXT[_question(user)]["gold_sparql"]
    return INVENTED


# ------------------------------------------------------------------ gate

def test_gate_expands_prefixed_names(env):
    schema, _ = env
    assert extract_uris(INVENTED, schema.prefixes) == {
        "http://example.org/network#raisedOn", "http://example.org/network#FW1"}
    assert hallucinated(INVENTED, schema) == [
        "http://example.org/network#FW1", "http://example.org/network#raisedOn"]


def test_gate_accepts_real_entities_and_standard_vocabularies(env):
    schema, _ = env
    q = """PREFIX net: <http://example.org/network#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?l WHERE { ?a net:affectsDevice net:Firewall_Edge ; rdfs:label ?l .
                      FILTER(?l != "net:notAnIri") }  # net:alsoIgnored"""
    assert hallucinated(q, schema) == []


def test_gate_catches_invented_full_iri(env):
    schema, _ = env
    q = "SELECT ?a WHERE { ?a <http://example.org/network#causedBy> ?b }"
    assert hallucinated(q, schema) == ["http://example.org/network#causedBy"]


# ------------------------------------------------------------ topologies

@pytest.mark.parametrize("topology", ["react", "orchestrator"])
def test_oracle_scores_perfectly_on_both_topologies(env, topology):
    schema, store = env
    res = evaluate(topology, QUESTIONS, schema, store, oracle)
    s = res["summary"]
    assert s["execution_accuracy"] == 1.0
    assert s["refusal_accuracy"] == 1.0
    assert s["false_refusal_rate"] == 0.0
    assert s["hallucinated_uri_rate"] == 0.0


def test_invented_uri_is_silently_accepted_by_react(env):
    schema, store = env
    q = "Which alarms were raised on the firewall FW1?"
    tr = run(build_react(first_invents, schema, store), q, "react", store)
    assert tr.ok and tr.rows == []          # executed fine, answered nothing
    assert tr.repairs == 0                  # no error, so no repair
    assert tr.hallucinated_uris             # and the metric records it


def test_invented_uri_is_caught_and_repaired_by_orchestrator(env):
    schema, store = env
    q = "Which alarms were raised on the firewall FW1?"
    tr = run(build_orchestrator(first_invents, schema, store), q, "orchestrator", store)
    assert tr.repairs == 1
    assert "refine" in tr.steps and tr.steps.index("verify") < tr.steps.index("refine")
    assert [r["a"].split("#")[-1] for r in tr.rows] == ["Alarm_FirewallSpike"]
    assert tr.hallucinated_uris == []


def test_refusal_is_available_to_both_topologies(env):
    """Otherwise the out-of-scope gap would be built into the design rather
    than measured."""
    schema, store = env
    q = "What is the capital of Slovenia?"
    for build, name in ((build_react, "react"), (build_orchestrator, "orchestrator")):
        tr = run(build(oracle, schema, store), q, name, store)
        assert tr.refused, name
        assert tr.rows == []


def test_classify_stage_refuses_before_any_generation(env):
    schema, store = env
    tr = run(build_orchestrator(oracle, schema, store),
             "Who won the 2022 football World Cup?", "orchestrator", store)
    assert tr.refused
    assert tr.steps == ["classify"]
    assert tr.llm_calls == 1


def always_broken(system: str, user: str) -> str:
    return "factual" if system.startswith("Classify") else "SELECT ?a WHERE { ?a"


@pytest.mark.parametrize("build,name", [(build_react, "react"),
                                        (build_orchestrator, "orchestrator")])
def test_repair_loop_stops_at_cap(env, build, name):
    """Regression: the ReAct router used to increment the repair counter, a
    write LangGraph discards, so the cap never applied and the graph looped
    until LangGraph's recursion limit."""
    from agents import MAX_REPAIRS
    schema, store = env
    tr = run(build(always_broken, schema, store),
             "Which alarms affect the core router R1?", name, store)
    assert not tr.ok
    assert tr.repairs == MAX_REPAIRS


# ------------------------------------------------- real-model answer shapes

from agents import strip_fences  # noqa: E402

Q = "PREFIX net: <http://example.org/network#>\nSELECT ?a WHERE { ?a net:hasSeverity \"HIGH\" . }"


@pytest.mark.parametrize("raw", [
    Q,
    f"```sparql\n{Q}\n```",
    f"```\n{Q}\n```",
    f"Here is the query:\n\n```sparql\n{Q}\n```\nIt returns high-severity alarms.",
    f"Sure. The answer uses hasSeverity.\n{Q}",
], ids=["bare", "fenced", "fence-no-lang", "prose-around-fence", "prose-before"])
def test_query_is_extracted_from_real_model_answer_shapes(raw):
    assert strip_fences(raw) == Q


@pytest.mark.parametrize("raw", ["OUT_OF_SCOPE", "OUT_OF_SCOPE.",
                                 "This graph has no such data. OUT_OF_SCOPE"])
def test_refusal_is_recognised_with_noise(raw):
    assert strip_fences(raw) == REFUSAL
