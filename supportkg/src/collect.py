"""Collect public SAP knowledge-article previews.

    python supportkg/src/collect.py                          # seeds/articles.txt
    python supportkg/src/collect.py --follow 1 --max 120     # also articles they cite

Runs on your own machine. It is polite and conservative by construction:
  - reads robots.txt first and skips any path it disallows;
  - one request every --delay seconds (default 3), identified by its user agent;
  - caches every page in data/raw/, so a rerun fetches nothing twice;
  - stores raw pages and parsed text under data/ only, which git ignores:
    SAP's texts are SAP's property and stay out of the repository. The graph
    keeps numbers, titles, codes and links to the source page.

With --follow 1, article numbers cited in a fetched preview are queued too
(one level), which grows the graph along the references SAP itself makes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
import urllib.robotparser
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from extract import article_refs  # noqa: E402
from parse import ParseError, parse  # noqa: E402

BASE = "https://userapps.support.sap.com"
URL = BASE + "/sap/support/knowledge/en/{n}"
UA = "supportkg-research/0.1 (PhD application prototype; polite, cached, robots-aware)"
ROOT = HERE.parents[0]


def read_seeds(path: Path) -> list[str]:
    out = []
    for ln in path.read_text().splitlines():
        ln = ln.split("#", 1)[0].strip()
        if re.fullmatch(r"\d{6,7}", ln):
            out.append(ln)
    return list(dict.fromkeys(out))


def robots(base: str = BASE) -> urllib.robotparser.RobotFileParser:
    rp = urllib.robotparser.RobotFileParser(base + "/robots.txt")
    try:
        rp.read()
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"cannot read {base}/robots.txt ({exc}); not collecting")
    return rp


def fetch(url: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode(r.headers.get_content_charset() or "utf-8", "replace")


def collect(numbers: list[str], raw_dir: Path, parsed_path: Path, delay: float,
            follow: int, max_pages: int, rp=None, fetcher=fetch) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    queue = [(n, 0) for n in numbers]
    seen: set[str] = set()
    stats = {"fetched": 0, "cached": 0, "parsed": 0, "unparsed": [], "disallowed": [], "failed": []}
    records = []
    while queue and len(seen) < max_pages:
        n, depth = queue.pop(0)
        if n in seen:
            continue
        seen.add(n)
        url = URL.format(n=n)
        cache = raw_dir / f"{n}.html"
        if cache.exists():
            html = cache.read_text()
            stats["cached"] += 1
        else:
            if rp is not None and not rp.can_fetch(UA, url):
                stats["disallowed"].append(n)
                continue
            try:
                html = fetcher(url)
            except (urllib.error.URLError, OSError) as exc:
                stats["failed"].append(f"{n}: {exc}")
                continue
            cache.write_text(html)
            stats["fetched"] += 1
            time.sleep(delay)
        try:
            rec = parse(html, number=n, url=url)
        except ParseError as exc:
            stats["unparsed"].append(str(exc))
            continue
        records.append(rec)
        stats["parsed"] += 1
        if depth < follow:
            for ref in article_refs(rec.text(), exclude={n}):
                if ref not in seen:
                    queue.append((ref, depth + 1))
    with parsed_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=str(ROOT / "seeds" / "articles.txt"))
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--parsed", default=str(ROOT / "data" / "parsed" / "articles.jsonl"))
    ap.add_argument("--delay", type=float, default=3.0)
    ap.add_argument("--follow", type=int, default=0, help="Levels of cited articles to queue.")
    ap.add_argument("--max", type=int, default=150, help="Upper bound on pages visited.")
    args = ap.parse_args()

    rp = robots()
    stats = collect(read_seeds(Path(args.seeds)), Path(args.raw), Path(args.parsed),
                    args.delay, args.follow, args.max, rp=rp)
    print(json.dumps({k: v if not isinstance(v, list) else (len(v), v[:5]) for k, v in stats.items()},
                     indent=1))
    if stats["unparsed"]:
        print("\nSome pages could not be parsed. Send one file from data/raw/ "
              "to adapt the parser to the current layout.")


if __name__ == "__main__":
    main()
