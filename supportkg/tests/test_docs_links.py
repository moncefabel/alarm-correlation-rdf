"""Help-documentation source, non-scraping article sources, the alert -> view
linker and the gold list."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from conftest import FIXTURES, ROOT, bow_encoder

DOCS = FIXTURES / "docs-repo"
SUP = "http://example.org/sapsupport#"
P = "PREFIX sup: <http://example.org/sapsupport#>\nPREFIX prov: <http://www.w3.org/ns/prov#>\n"


@pytest.fixture(scope="module")
def docs():
    from docs import load
    return {d.name: d for d in load(DOCS)}


def test_docs_pages_become_typed_records(docs):
    assert set(docs) == {"M_SAVEPOINTS", "USERS", "ALTER SYSTEM SAVEPOINT"}   # overview skipped
    sp, users, stmt = docs["M_SAVEPOINTS"], docs["USERS"], docs["ALTER SYSTEM SAVEPOINT"]
    assert sp.kind == "monitoring_view" and users.kind == "system_view"
    assert stmt.kind == "sql_statement" and stmt.category == "System Management"
    assert sp.description.startswith("Test description: savepoint history")
    assert sp.columns == [("HOST", "NVARCHAR(64)", "Test text: host."),
                          ("DURATION", "BIGINT", "Test text: how long the savepoint took.")]
    assert users.columns[0][0] == "USER_NAME"            # table without a heading
    assert sp.related == ["alter-system-savepoint-statement-system-management-t03"]  # unknown page dropped
    assert sp.url.startswith("https://github.com/SAP-docs/sap-hana-cloud-hana-database/blob/")


def test_a_folder_that_is_not_the_docs_clone_is_refused(tmp_path):
    from docs import load
    with pytest.raises(SystemExit):
        load(tmp_path)


def test_title_records_come_from_the_seed_list():
    from collect import records_from_titles
    recs = records_from_titles(ROOT / "seeds" / "articles.txt")
    assert len(recs) == 39
    r = next(x for x in recs if x.number == "1977220")
    assert r.title.startswith("How to handle HANA Alert 54") and r.url.endswith("/1977220")


def test_saved_pages_are_read_and_bad_ones_reported():
    from collect import records_from_saved
    recs, bad = records_from_saved(FIXTURES)
    assert sorted(r.number for r in recs) == ["9900001", "9900002", "9900003"]
    assert len(bad) == 1 and bad[0].startswith("shell.html")


def test_lexical_linker_weights_rare_words_and_view_names():
    from link import ViewLinker, tokens
    assert tokens("Memory Usage of Services") == {"memory", "usage", "service"}
    assert tokens("M_CS_LOB_SPACE_RECLAIMS") == {"cs", "lob", "space", "reclaim"}
    views = {"M_SAVEPOINTS": "Savepoint history.", "M_DISK_USAGE": "Disk usage per host.",
             "M_EXPENSIVE_STATEMENTS": "Statements whose duration exceeds a threshold."}
    linker = ViewLinker(views, method="lexical")
    assert linker.link("Savepoint Duration")[0][0] == "M_SAVEPOINTS"
    assert linker.link("Unrelated words entirely") == []


def test_embedding_linker_uses_the_encoder():
    from link import ViewLinker
    views = {"M_TRACEFILES": "Trace and diagnosis files on each host.",
             "M_LICENSE": "The current license key."}
    linker = ViewLinker(views, method="embedding", encoder=bow_encoder(), threshold=0.2)
    assert linker.link("Number of diagnosis files")[0][0] == "M_TRACEFILES"


def test_graph_joins_articles_and_documentation(records, docs):
    from build import build_graph, validate
    from link import ViewLinker
    from parse import Record
    recs = records + [Record(number="1977220", title="How to handle HANA Alert 54: Savepoint Duration",
                             url="https://userapps.support.sap.com/sap/support/knowledge/en/1977220")]
    g, prov, report = build_graph(recs, docs=list(docs.values()),
                                  view_linker_factory=lambda v: ViewLinker(v, method="lexical", threshold=0.2))
    ok, _, text = validate(g)
    assert ok, text
    q = lambda s: bool(g.query(P + s).askAnswer)  # noqa: E731
    assert q("ASK { sup:Alert_54 sup:investigatedWith sup:View_M_SAVEPOINTS }")
    assert q("ASK { sup:View_M_SAVEPOINTS sup:helpsInvestigate sup:Alert_54 }")        # inverse
    assert q("ASK { sup:View_M_SAVEPOINTS sup:hasColumn sup:Column_M_SAVEPOINTS__DURATION }")
    assert q("ASK { sup:Statement_ALTER_SYSTEM_SAVEPOINT sup:seeAlso sup:View_M_SAVEPOINTS }")  # symmetric
    assert report["entities"]["views"] == 2 and report["entities"]["columns"] == 3
    # the predicted link names both sources it joins
    srcs = {str(r[0]) for r in prov.query(P + """SELECT ?src WHERE { ?st a <http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement> ;
            <http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate> sup:investigatedWith ; prov:wasDerivedFrom ?src }""")}
    assert any("1977220" in s for s in srcs) and any("github.com/SAP-docs" in s for s in srcs)


def test_gold_list_is_well_formed():
    from evaluate_links import alert_names
    gold = yaml.safe_load((ROOT / "gold" / "alert_views.yaml").read_text())
    names = alert_names(ROOT / "seeds" / "articles.txt")
    assert set(gold["alerts"]) == set(names)                # one entry per seeded alert
    assert all(v and all(isinstance(x, str) for x in v) for v in gold["alerts"].values())


@pytest.mark.skipif(not (Path(os.environ.get("SUPPORTKG_DOCS", ROOT / "data" / "sap-hana-cloud-hana-database"))
                         / "docs").is_dir(), reason="documentation clone not present")
def test_gold_views_exist_in_the_real_documentation():
    from docs import load
    repo = Path(os.environ.get("SUPPORTKG_DOCS", ROOT / "data" / "sap-hana-cloud-hana-database"))
    names = {d.name for d in load(repo)}
    gold = yaml.safe_load((ROOT / "gold" / "alert_views.yaml").read_text())["alerts"]
    assert {v for vs in gold.values() for v in vs} <= names
