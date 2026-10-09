"""The benchmark checks itself.

A gold query that runs but returns the wrong answer scores every prediction
zero without raising anything. Each gold query is executed here and compared to
the answer written by hand in `expected`.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kgqa" / "src"))

from graph_store import GraphStore  # noqa: E402

ONTOLOGY = ROOT / "ontology" / "alarm_ontology.ttl"
QUESTIONS = yaml.safe_load((ROOT / "kgqa" / "benchmark" / "questions.yaml").read_text())["questions"]


@pytest.fixture(scope="module")
def store() -> GraphStore:
    return GraphStore(ttl_path=str(ONTOLOGY))


def _local(v: str) -> str:
    return v.split("#")[-1]


@pytest.mark.parametrize("q", QUESTIONS, ids=[q["id"] for q in QUESTIONS])
def test_gold_query_returns_expected_answer(store, q):
    res = store.select(q["gold_sparql"])
    assert res.ok, res.error
    got = sorted(_local(v) for row in res.rows for v in row.values())
    assert got == sorted(q["expected"])


def test_ids_are_unique():
    ids = [q["id"] for q in QUESTIONS]
    assert len(ids) == len(set(ids))


def test_every_bucket_is_represented():
    counts = Counter(q["type"] for q in QUESTIONS)
    for bucket in ("factual", "causal", "contrast", "counterfactual", "out_of_scope"):
        assert counts[bucket] >= 3, f"{bucket}: {counts[bucket]} questions"


def test_contrast_bucket_separates_correlation_from_causation(store):
    """The co-location trap, checked on the data itself: the device with the
    most alarms is NOT the device hosting the root cause."""
    by_id = {q["id"]: q for q in QUESTIONS}
    most_alarms = store.select(by_id["F05"]["gold_sparql"]).rows[0]["d"]
    root_device = store.select(by_id["K03"]["gold_sparql"]).rows[0]["d"]
    assert _local(most_alarms) == "Router_Core"
    assert _local(root_device) == "Firewall_Edge"
    assert most_alarms != root_device
