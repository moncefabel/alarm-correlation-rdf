"""Knowledge-article sources, three ways, none of which scrapes a site that
forbids it.

    python supportkg/src/collect.py                               # titles (default)
    python supportkg/src/collect.py --mode saved --saved-dir DIR  # + pages you saved
    python supportkg/src/collect.py --mode web                    # robots.txt permitting

titles  Number and title of each article in seeds/articles.txt, as public
        search-engine listings show them. Enough for the alert each how-to
        article addresses and the alert's name.
saved   Preview pages a person opened in a browser and saved (File > Save
        page as, HTML only) into a folder. robots.txt governs automated
        crawlers, not a person reading pages; the folder stays local and git
        ignores it. Saved pages add symptom, product and keyword fields; for
        numbers with no saved page the title record is kept.
web     Automated download. It reads robots.txt first and skips every
        disallowed path. SAP's support portal disallows all crawlers except
        named search engines (checked 2026-10-14), so this mode collects
        nothing there; it is kept for sources that allow it.

Raw pages and parsed text stay under supportkg/data/, which git ignores: SAP's
texts are SAP's property. The graph keeps numbers, titles, codes and links.
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


def records_from_titles(path: Path) -> list:
    """Records from the seed list: '1977220   # How to handle HANA Alert 54: ...'."""
    from parse import Record
    out = []
    for ln in path.read_text().splitlines():
        num, _, title = ln.partition("#")
        num, title = num.strip(), title.strip()
        if re.fullmatch(r"\d{6,7}", num) and title:
            out.append(Record(number=num, title=title, url=URL.format(n=num)))
    return out


def records_from_saved(folder: Path) -> tuple[list, list[str]]:
    """Records from preview pages saved by hand; the number comes from the page."""
    recs, bad = [], []
    for f in sorted(list(folder.glob("*.html")) + list(folder.glob("*.htm"))):
        try:
            rec = parse(f.read_text(errors="replace"))
        except ParseError as exc:
            bad.append(f"{f.name}: {exc}")
            continue
        rec.url = URL.format(n=rec.number)
        recs.append(rec)
    return recs, bad


def write_records(records, parsed_path: Path) -> None:
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    with parsed_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["titles", "saved", "web"], default="titles")
    ap.add_argument("--seeds", default=str(ROOT / "seeds" / "articles.txt"))
    ap.add_argument("--saved-dir", default=str(ROOT / "data" / "saved"))
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--parsed", default=str(ROOT / "data" / "parsed" / "articles.jsonl"))
    ap.add_argument("--delay", type=float, default=3.0)
    ap.add_argument("--follow", type=int, default=0, help="web mode: levels of cited articles.")
    ap.add_argument("--max", type=int, default=150, help="web mode: upper bound on pages.")
    args = ap.parse_args()

    if args.mode == "web":
        rp = robots()
        stats = collect(read_seeds(Path(args.seeds)), Path(args.raw), Path(args.parsed),
                        args.delay, args.follow, args.max, rp=rp)
        print(json.dumps({k: v if not isinstance(v, list) else (len(v), v[:5])
                          for k, v in stats.items()}, indent=1))
        if stats["disallowed"]:
            print("\nrobots.txt disallows these pages for automated tools: nothing was "
                  "fetched. Use --mode titles, or --mode saved with pages saved by hand.")
        return

    records = {r.number: r for r in records_from_titles(Path(args.seeds))}
    report = {"from_titles": len(records)}
    if args.mode == "saved":
        saved, bad = records_from_saved(Path(args.saved_dir))
        for r in saved:
            records[r.number] = r                 # a saved page has more fields
        report.update(from_saved_pages=len(saved), unparsed_saved_pages=bad)
    write_records(list(records.values()), Path(args.parsed))
    report["records_written"] = len(records)
    report["file"] = args.parsed
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
