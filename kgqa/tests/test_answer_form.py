"""Answer-form scoring: a name counts for the gold entity it names, nothing else."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kgqa" / "src"))

from answer_form import name_aliases, score_by_name  # noqa: E402
from evaluate import label_index, score  # noqa: E402
from graph_store import GraphStore  # noqa: E402

GRAPH = ROOT / "supportkg" / "out" / "graph.ttl"
needs = pytest.mark.skipif(not GRAPH.exists(), reason="support graph not built")
NS = "http://example.org/sapsupport#"
GOLD = [{"c": NS + "Column_M_PASSWORD_POLICY__PROPERTY"}, {"c": NS + "Column_M_PASSWORD_POLICY__VALUE"}]


@pytest.fixture(scope="module")
def store():
    return GraphStore(ttl_path=str(GRAPH))


@needs
def test_column_names_count_for_the_columns_they_name(store):
    labels = label_index(store)
    pred = [{"name": "PROPERTY"}, {"name": "VALUE"}]
    assert score(pred, GOLD, labels)[2] == 0.0                      # the original scorer
    assert score_by_name(pred, GOLD, labels, store)[2] == 1.0


@needs
def test_a_name_of_something_else_earns_nothing(store):
    labels = label_index(store)
    pred = [{"name": "PROPERTY"}, {"name": "CONNECTION_ID"}]       # second is not in the gold set
    assert score_by_name(pred, GOLD, labels, store)[2] == pytest.approx(0.5)
    assert "connection_id" not in name_aliases(store, GOLD)


@needs
def test_a_wrong_answer_is_still_wrong(store):
    labels = label_index(store)
    assert score_by_name([{"name": "UNLOAD_TIME"}], GOLD, labels, store)[2] == 0.0
