"""The direction audit, on queries the models actually wrote."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kgqa" / "src"))

from direction import inverted_entities  # noqa: E402

GOLD = {q["id"]: q["gold_sparql"] for q in
        yaml.safe_load((ROOT / "kgqa" / "benchmark" / "questions.yaml").read_text())["questions"]}


def test_c03_inversion_is_found():
    # orchestrator, run 0: asks for the effects of the latency spike
    q = """SELECT ?cause WHERE { net:Alarm_Latency net:causallyLinkedTo ?cause . }"""
    assert inverted_entities(q, GOLD["C03"]) == ["net:Alarm_Latency"]


def test_c02_inversion_through_a_path_is_found():
    # orchestrator, run 1: walks upstream with the forward property
    q = """SELECT ?root WHERE {
             net:Alarm_WebTimeout net:causallyLinkedTo* ?root .
             FILTER NOT EXISTS { ?upstream net:causallyLinkedTo ?root }
           }"""
    assert inverted_entities(q, GOLD["C02"]) == ["net:Alarm_WebTimeout"]


def test_parenthesised_path_is_read():
    q = "SELECT ?r WHERE { net:Alarm_WebTimeout (net:causallyLinkedTo)+ ?r . }"
    assert inverted_entities(q, GOLD["C02"]) == ["net:Alarm_WebTimeout"]


def test_correct_direction_is_not_flagged():
    q = """SELECT ?root WHERE {
             ?root (net:causallyLinkedTo)+ net:Alarm_WebTimeout .
             FILTER NOT EXISTS { ?x net:causallyLinkedTo ?root }
           }"""
    assert inverted_entities(q, GOLD["C02"]) == []
    assert inverted_entities("SELECT ?c WHERE { ?c net:causallyLinkedTo net:Alarm_Latency }",
                             GOLD["C03"]) == []


def test_one_direction_of_a_symmetric_gold_is_not_an_inversion():
    # K02's gold looks both ways; a query that looks one way is incomplete,
    # which the scorer already penalises, not inverted
    q = "SELECT ?a WHERE { net:Alarm_PacketDrop net:correlatedWith ?a . " \
        "FILTER NOT EXISTS { net:Alarm_PacketDrop net:causallyLinkedTo+ ?a } }"
    assert inverted_entities(q, GOLD["K02"]) == []


def test_comments_are_ignored():
    q = """SELECT ?c WHERE {
             # net:Alarm_Latency net:causallyLinkedTo ?c  would be wrong
             ?c net:causallyLinkedTo net:Alarm_Latency .
           }"""
    assert inverted_entities(q, GOLD["C03"]) == []


def test_every_gold_query_agrees_with_itself():
    assert all(inverted_entities(g, g) == [] for g in GOLD.values())
