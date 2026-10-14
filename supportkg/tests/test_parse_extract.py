"""Parser and rule-based extraction, on synthetic pages and on real title
spellings taken from the seed list."""

from __future__ import annotations

import pytest

from conftest import FIXTURES


def facts(ex, p):
    return {(f.s, f.o) for f in ex.facts if f.p == p}


def test_two_different_layouts_parse_the_same_fields(records):
    a, b, c = records
    assert a.title == "How to handle HANA Alert 54: 'Savepoint Duration'"
    assert a.kind == "KBA" and c.kind == "SAPNote"
    assert a.products == ["SAP HANA, platform edition, all versions"]
    assert b.products == ["SAP HANA 1.0, platform edition", "SAP HANA, platform edition 2.0"]
    assert "HAN-DB-MON" in b.keywords and "SAP HANA Monitoring" in b.keywords
    assert "tracking" not in a.symptom          # script content is dropped


def test_a_page_without_content_is_an_error_not_an_empty_record():
    from parse import ParseError, parse
    with pytest.raises(ParseError):
        parse((FIXTURES / "shell.html").read_text(), number="9900009")


@pytest.mark.parametrize("title,expected", [
    ("How to handle HANA Alert 54: Savepoint Duration", (54, "Savepoint Duration")),
    ("How to handle HANA Alert 37 - 'Age of most recent data backup'",
     (37, "Age of most recent data backup")),
    ("Handle HANA Alert 69: Enablement of automatic log backup",
     (69, "Enablement of automatic log backup")),
    ("How to handle HANA alert 36: 'Status of Most Recent Data Backup'",
     (36, "Status of Most Recent Data Backup")),
    ("How-To: Handle HANA Alert 50: 'Number of Diagnosis files'", (50, "Number of Diagnosis files")),
    ("How to handle HANA Alert 4: ‘Restarted services’", (4, "Restarted services")),
    ("How to handle HANA alert (ID=75) : Row Store Version Space Skew",
     (75, "Row Store Version Space Skew")),
])
def test_how_to_titles_name_the_alert_they_address(title, expected):
    from extract import title_alert
    assert title_alert(title) == expected


@pytest.mark.parametrize("title", [
    "Alert 25 not providing the expected result in SAP HANA Cloud",   # about, not how-to
    "How-To: Interpreting and Resolving SAP HANA Alerts",             # no single alert
    "Alerts related to HANA memory consumption",
])
def test_other_titles_address_no_single_alert(title):
    from extract import title_alert
    assert title_alert(title) is None


def test_alert_and_article_references():
    from extract import alert_refs, article_refs
    assert alert_refs("see Alert ID 39 and alert (ID=75); alerts 1, 2 and 43") == [39, 75, 1, 2, 43]
    assert article_refs("SAP Note 1758890, KBAs 2000002 and 2000003; Note 12345",
                        exclude={"2000003"}) == ["1758890", "2000002"]


def test_component_codes_take_the_next_keyword_as_name():
    from extract import component_codes, parents
    kws = ["HANA Alert", "KBA", "HAN-DB-MON", "SAP HANA Monitoring", "SV-PERF", "KBA", "SAP-HANA"]
    assert component_codes(kws) == [("HAN-DB-MON", "SAP HANA Monitoring"), ("SV-PERF", "")]
    assert parents("HAN-DB-ENG-MM") == ["HAN-DB-ENG", "HAN-DB", "HAN"]


def test_extraction_of_a_how_to_article(records):
    from extract import extract
    ex = extract(records[0])
    art = ("article", "9900001")
    assert facts(ex, "addresses") == {(art, ("alert", 54))}
    assert facts(ex, "mentionsAlert") == {(art, ("alert", 54)), (art, ("alert", 39))}
    assert facts(ex, "cooccursWith") == {(("alert", 39), ("alert", 54))}
    assert facts(ex, "assignedTo") == {(art, ("component", "HAN-DB-PER")),
                                       (art, ("component", "SV-PERF"))}
    assert facts(ex, "appliesTo") == {(art, "http://example.org/ppms#Product_HANA_Platform")}
    assert {o for _, o in facts(ex, "usesTool")} == {
        "http://example.org/sapsupport#Tool_HANA_Studio", "http://example.org/sapsupport#Tool_DBACOCKPIT"}
    assert {o[1] for _, o in facts(ex, "references")} == {"9900002", "9900004", "9900005"}


def test_product_strings_link_to_versions_and_unmatched_ones_are_reported(records):
    from extract import extract
    two = extract(records[1])
    assert {o for _, o in facts(two, "appliesTo")} == {
        "http://example.org/ppms#PV_HANA_Platform_1_0", "http://example.org/ppms#PV_HANA_Platform_2_0"}
    three = extract(records[2])
    assert not facts(three, "appliesTo") and three.unlinked_products == ["HANA 2 platform"]


def test_every_fact_records_its_method(records):
    from extract import extract
    for r in records:
        assert all(f.method for f in extract(r).facts)
