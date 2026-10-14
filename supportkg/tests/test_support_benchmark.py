"""The support benchmark checks itself, on both graphs of the OWL experiment.

Each gold query must return its hand-written `expected` answer on the default
graph (inverses and symmetric links materialised) AND on the graph built with
--no-materialise. Otherwise a difference between the two arms could come from
the gold, not from the agents.
"""

from __future__ import annotations

from collections import Counter

import pytest
import yaml
from rdflib import BNode, Graph

from conftest import ROOT

QUESTIONS = yaml.safe_load((ROOT / "benchmark" / "questions.yaml").read_text())["questions"]
GRAPHS = {"materialised": ROOT / "out" / "graph.ttl",
          "no_inverses": ROOT / "out_noinv" / "graph.ttl"}
SUP = "http://example.org/sapsupport#"


@pytest.fixture(scope="module", params=sorted(GRAPHS))
def graph(request):
    path = GRAPHS[request.param]
    if not path.exists():
        pytest.skip(f"{path} not built")
    g = Graph()
    g.parse(path)
    return g


def answers(g: Graph, query: str) -> list[str]:
    res = g.query(query)
    return sorted(str(v).split("#")[-1] for row in res for v in row)


@pytest.mark.parametrize("q", QUESTIONS, ids=[q["id"] for q in QUESTIONS])
def test_gold_query_returns_expected_answer(graph, q):
    assert answers(graph, q["gold_sparql"]) == sorted(q["expected"])


def test_ids_unique_and_every_bucket_represented():
    ids = [q["id"] for q in QUESTIONS]
    assert len(ids) == len(set(ids))
    counts = Counter(q["type"] for q in QUESTIONS)
    for bucket in ("lookup", "inverse", "multihop", "aggregation", "negation", "out_of_scope"):
        assert counts[bucket] >= 3, f"{bucket}: {counts[bucket]}"


def test_the_two_graphs_differ_exactly_by_the_owl_inferences():
    """What the experiment varies: the no-inverse graph lacks the inverse
    direction (hasColumn, addressedBy, helpsInvestigate...) and the reversed
    seeAlso links, and nothing else."""
    if not all(p.exists() for p in GRAPHS.values()):
        pytest.skip("graphs not built")
    full, bare = Graph(), Graph()
    full.parse(GRAPHS["materialised"])
    bare.parse(GRAPHS["no_inverses"])
    ground = lambda g: {t for t in g if not any(isinstance(x, BNode) for x in t)}  # noqa: E731
    assert ground(bare) <= ground(full)            # blank nodes (OWL lists) aside
    extra = Counter(str(p).split("#")[-1] for _, p, _ in ground(full) - ground(bare))
    assert set(extra) == {"hasColumn", "addressedBy", "mentionedIn", "helpsInvestigate",
                          "seeAlso", "hasVersion", "hasSupportPackage", "includedIn",
                          "lineHasProduct"}
    assert extra["hasColumn"] == 6490
    # the inverse direction is what the `inverse` questions ask for
    ask = f"ASK {{ <{SUP}View_M_PASSWORD_POLICY> <{SUP}hasColumn> ?c }}"
    assert full.query(ask).askAnswer and not bare.query(ask).askAnswer


def test_inferred_ttl_lists_exactly_what_owl_added():
    """explain.py reads out/inferred.ttl to tell an inferred triple from an
    extracted one; it must be the difference between the two builds."""
    paths = [*GRAPHS.values(), ROOT / "out" / "inferred.ttl"]
    if not all(p.exists() for p in paths):
        pytest.skip("graphs not built")
    full, bare, inferred = Graph(), Graph(), Graph()
    full.parse(paths[0]); bare.parse(paths[1]); inferred.parse(paths[2])
    ground = lambda g: {t for t in g if not any(isinstance(x, BNode) for x in t)}  # noqa: E731
    assert ground(inferred) == ground(full) - ground(bare)
