"""Domain profiles: the alarm prompts are unchanged, the support profile runs
end to end on the SAP support graph, and its retrieval brings every IRI the
gold queries need into the prompt."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kgqa" / "src"))
sys.path.insert(0, str(ROOT / "kgqa" / "tests"))
sys.path.insert(0, str(ROOT / "kgqa" / "scripts"))

from agents import REFUSAL, register_schema, run  # noqa: E402
from domains import ALARMS, SUPPORT  # noqa: E402
from evaluate import build_agent, evaluate  # noqa: E402
from graph_store import GraphStore  # noqa: E402
from schema_context import SchemaIndex  # noqa: E402

ARMS = ["react", "orchestrator", "react_typed", "orchestrator_untyped"]

# sha256 of every (system, user) pair the four arms send on the alarm
# benchmark, with a perfect model and with one that invents a URI first.
# Recorded on the code that produced the published alarm results (36c2575).
ALARM_PROMPTS_SHA256 = "101e7d238e3e530fedd91d8be9d0aab3dbedf1a691f6772a48e14b2c65f292de"


def test_alarm_prompts_are_byte_identical_to_the_published_runs():
    import test_topologies as T
    schema = SchemaIndex(T.ONTOLOGY)
    register_schema(schema)
    store = GraphStore(ttl_path=T.ONTOLOGY)
    seen = []

    def rec(base):
        def llm(s, u):
            seen.append([s, u])
            return base(s, u)
        return llm
    for arm in ARMS:
        for q in T.QUESTIONS:
            run(build_agent(arm, rec(T.oracle), schema, store), q["question"], arm, store)
            run(build_agent(arm, rec(T.first_invents), schema, store), q["question"], arm, store)
    assert hashlib.sha256(json.dumps(seen).encode()).hexdigest() == ALARM_PROMPTS_SHA256


def test_profiles_name_their_domain():
    assert "network devices, alarms" in ALARMS.gen_system
    assert "SAP HANA support" in SUPPORT.gen_system and "alarm" not in SUPPORT.gen_system.split("Scope")[0]
    assert SUPPORT.classify_system.endswith("lookup, multihop, aggregation, negation, or out_of_scope.")
    assert set(SUPPORT.hints) <= set(SUPPORT.types)


# ------------------------------------------------------------ support graph

GRAPH = ROOT / SUPPORT.graph
needs_graph = pytest.mark.skipif(not GRAPH.exists(), reason="support graph not built")
QUESTIONS = yaml.safe_load((ROOT / SUPPORT.questions).read_text())["questions"]
BY_TEXT = {q["question"]: q for q in QUESTIONS}
CLASS_OF = {"inverse": "lookup"}           # benchmark bucket -> classifier label


@pytest.fixture(scope="module")
def support():
    schema = SchemaIndex(str(GRAPH), retrieval=SUPPORT.retrieval)
    register_schema(schema)
    return schema, GraphStore(ttl_path=str(GRAPH))


def support_oracle(system: str, user: str) -> str:
    q = BY_TEXT[re.search(r"Question: (.+?)(?:\n|$)", user).group(1).strip()]
    if system.startswith("Classify"):
        return CLASS_OF.get(q["type"], q["type"])
    return REFUSAL if q["type"] == "out_of_scope" else q["gold_sparql"]


@needs_graph
@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_scores_perfectly_on_the_support_benchmark_with_a_perfect_model(support, arm):
    schema, store = support
    s = evaluate(arm, QUESTIONS, schema, store, support_oracle, domain=SUPPORT)["summary"]
    assert s["execution_accuracy"] == 1.0 and s["refusal_accuracy"] == 1.0
    assert s["hallucinated_uri_rate"] == 0.0


@needs_graph
def test_support_prompts_carry_the_support_profile(support):
    schema, store = support
    seen = []

    def llm(s, u):
        seen.append((s, u))
        return support_oracle(s, u)
    q = next(q for q in QUESTIONS if q["type"] == "aggregation")
    run(build_agent("orchestrator", llm, schema, store, domain=SUPPORT), q["question"],
        "orchestrator", store)
    (cls_sys, cls_user), (gen_sys, gen_user) = seen[:2]
    assert cls_sys == SUPPORT.classify_system and "SAP HANA support" in cls_user
    assert gen_sys == SUPPORT.gen_system and SUPPORT.hints["aggregation"] in gen_user


@needs_graph
def test_schema_plus_entities_retrieval_finds_every_gold_iri(support):
    """Measured by scripts/retrieval_recall.py: the original lexical retrieval
    misses 39 % of the IRIs the gold queries need on this graph."""
    from retrieval_recall import recall
    schema, _ = support
    res = recall(schema, QUESTIONS)
    assert res["iri_recall"] == 1.0, [r for r in res["rows"] if r["missing"]]


@needs_graph
def test_entity_names_come_from_keys_labels_and_class_words(support):
    schema, _ = support
    first = lambda c: next(l for l in c.splitlines() if l.startswith("instance"))  # noqa: E731
    assert "#Alert_54>" in first(schema.relevant("Which article addresses alert 54?"))
    assert "#Article_1977220>" in schema.relevant("What does SAP Note 1977220 address?")
    # "2" inside "2.0" is not alert 2, the version is found by its alternative label
    ctx = schema.relevant("Which components are in SAP HANA 2.0?")
    assert "#PV_HANA_Platform_2_0>" in ctx and "#Alert_2>" not in ctx
