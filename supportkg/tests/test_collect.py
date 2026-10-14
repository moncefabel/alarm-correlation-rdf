"""Collector behaviour, offline: robots.txt, cache, following citations."""

from __future__ import annotations

import json

from conftest import FIXTURES


class Robots:
    def __init__(self, disallowed):
        self.disallowed = disallowed

    def can_fetch(self, ua, url):
        return not any(url.endswith(n) for n in self.disallowed)


def fake_fetcher(calls):
    def fetch(url):
        n = url.rsplit("/", 1)[-1]
        calls.append(n)
        path = FIXTURES / f"{n}.html"
        return path.read_text() if path.exists() else (FIXTURES / "shell.html").read_text()
    return fetch


def test_collect_respects_robots_caches_and_follows_citations(tmp_path):
    from collect import collect
    calls = []
    raw, parsed = tmp_path / "raw", tmp_path / "parsed.jsonl"
    stats = collect(["9900001"], raw, parsed, delay=0, follow=1, max_pages=10,
                    rp=Robots({"9900005"}), fetcher=fake_fetcher(calls))
    # 9900001 cites 9900002, 9900004 and 9900005; 9900005 is disallowed
    assert calls == ["9900001", "9900002", "9900004"]
    assert stats["disallowed"] == ["9900005"]
    assert stats["parsed"] == 2 and len(stats["unparsed"]) == 1      # 9900004 is a shell page
    assert [json.loads(l)["number"] for l in parsed.read_text().splitlines()] == ["9900001", "9900002"]

    calls.clear()
    again = collect(["9900001"], raw, parsed, delay=0, follow=1, max_pages=10,
                    rp=Robots({"9900005"}), fetcher=fake_fetcher(calls))
    assert calls == [] and again["cached"] == 3                      # nothing fetched twice


def test_seed_file_reader_ignores_comments_and_duplicates(tmp_path):
    from collect import read_seeds
    p = tmp_path / "s.txt"
    p.write_text("# header\n1977220   # alert 54\n\n1977220\nabc\n2445867\n")
    assert read_seeds(p) == ["1977220", "2445867"]


def test_shipped_seed_list_is_well_formed():
    from collect import read_seeds
    from conftest import ROOT
    seeds = read_seeds(ROOT / "seeds" / "articles.txt")
    assert len(seeds) == len(set(seeds)) >= 35
