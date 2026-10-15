"""The no-SPARQL baseline: pages carry the graph's facts, retrieval finds named
pages, answers are scored like the agents' answers."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kgqa" / "src"))

from rag_baseline import RagBaseline, evaluate_rag, parse_answer  # noqa: E402

GRAPH = ROOT / "supportkg" / "out" / "graph.ttl"
needs = pytest.mark.skipif(not GRAPH.exists(), reason="support graph not built")
QUESTIONS = yaml.safe_load((ROOT / "supportkg" / "benchmark" / "questions.yaml").read_text())["questions"]


@pytest.fixture(scope="module")
def rag():
    return RagBaseline(str(GRAPH), k=5)


@needs
def test_pages_carry_columns_and_links_both_ways(rag):
    view = rag.page_text["View_M_CS_UNLOADS"]
    assert "UNLOAD_TIME, TIMESTAMP" in view and "[ID Column_M_CS_UNLOADS__UNLOAD_TIME]" in view
    assert "[ID Alert_55]" in view                                   # view -> alert
    assert "[ID View_M_CS_UNLOADS]" in rag.page_text["Alert_55"]     # alert -> view
    assert "[ID Article_1977207]" in rag.page_text["Alert_55"]       # alert -> article


@needs
def test_a_named_page_is_retrieved_even_when_long(rag):
    q = "How many columns does the monitoring view M_CONNECTIONS have?"
    assert "View_M_CONNECTIONS" in [rag.pages[i]["id"] for i in rag.index.top(q, 5)]


def test_answers_are_parsed_from_json_or_lines():
    assert parse_answer('["View_M_DISKS", "View_M_DISK_USAGE"]') == ["View_M_DISKS", "View_M_DISK_USAGE"]
    assert parse_answer('The answer is ["27"].') == ["27"]
    assert parse_answer("OUT_OF_SCOPE") is None and parse_answer('["OUT_OF_SCOPE"]') is None
    assert parse_answer("- BIGINT\n") == ["BIGINT"]


@needs
def test_a_model_that_reads_the_pages_right_scores_perfectly(rag):
    by_text = {q["question"]: q for q in QUESTIONS}

    def reader(system, user):
        q = by_text[re.search(r"Question: (.+?)(?:\n|$)", user).group(1).strip()]
        if q["type"] == "out_of_scope":
            return "OUT_OF_SCOPE"
        return '["' + '", "'.join(q["expected"]) + '"]'
    s = evaluate_rag(rag, QUESTIONS, reader)["summary"]
    assert s["execution_accuracy"] == 1.0 and s["refusal_accuracy"] == 1.0


@needs
def test_closed_book_sends_no_pages():
    seen = []
    rag0 = RagBaseline(str(GRAPH), k=0)
    rag0.ask(lambda s, u: seen.append(u) or '["x"]', "What is the name of SAP HANA alert 54?")
    assert seen == ["Question: What is the name of SAP HANA alert 54?"]
