"""Extraction scoring, on the synthetic pages with a hand-written gold set
that disagrees with the extractor on purpose in two places."""

from __future__ import annotations

import pytest
import yaml

GOLD = yaml.safe_load("""
articles:
  "9900001":
    addresses: [54]
    mentionsAlert: [54, 39]
    assignedTo: [HAN-DB-PER, SV-PERF]
    appliesTo: [ppms:Product_HANA_Platform]
    references: ["9900002", "9900004", "9900005"]
    usesTool: [sup:Tool_HANA_Studio, sup:Tool_DBACOCKPIT]
  "9900003":
    addresses: []
    mentionsAlert: [1, 43]                     # the article is about memory alerts
    assignedTo: [HAN-DB-ENG-MM]
    appliesTo: [ppms:PV_HANA_Platform_2_0]     # "HANA 2 platform"
    references: []
    usesTool: []
  "9900002":
    skip: true
""")


def test_rules_are_scored_per_relation(records):
    from build import extract_all
    from evaluate_extraction import gold_sets, predicted, score
    gold = gold_sets(GOLD)
    assert set(gold) == {"9900001", "9900003"}                 # skipped page left out
    table = score(predicted(extract_all(records)), gold)
    assert table["addresses"]["f1"] == 1.0
    assert table["assignedTo"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 3, "fp": 0, "fn": 0}
    # rules cannot know 9900003 is about alerts 1 and 43, nor link "HANA 2 platform"
    assert table["mentionsAlert"]["recall"] == pytest.approx(2 / 4)
    assert table["appliesTo"]["recall"] == pytest.approx(1 / 2)
    assert table["appliesTo"]["precision"] == 1.0


def test_errors_list_names_the_missing_objects(records):
    from build import extract_all
    from evaluate_extraction import errors, gold_sets, predicted
    lines = errors(predicted(extract_all(records)), gold_sets(GOLD))
    assert "9900003 mentionsAlert: missing ['1', '43'] extra []" in lines


def test_template_never_overwrites_annotations(records, tmp_path):
    from evaluate_extraction import template
    p = tmp_path / "gold.yaml"
    template(records, 2, p)
    form = yaml.safe_load(p.read_text())
    assert len(form["articles"]) == 2 and all(v["addresses"] == [] for v in form["articles"].values())
    with pytest.raises(SystemExit):
        template(records, 2, p)
