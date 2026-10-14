"""Score extraction against hand annotations.

    python supportkg/src/evaluate_extraction.py --template 25   # write the annotation form
    python supportkg/src/evaluate_extraction.py                 # rules only
    python supportkg/src/evaluate_extraction.py --linker hf     # rules + embedding linker

Annotation protocol (written before any score is computed):
  - the form lists a fixed random sample of collected articles (seed 0), each
    with the link to its public preview and empty fields;
  - the annotator reads the page in a browser and fills the fields, without
    looking at the extractor's output, so the gold set is not anchored on it;
  - what to annotate, per article: the alert it is written to resolve
    (addresses), every alert it refers to (mentionsAlert), its application
    components (assignedTo), the PPMS stand-in items its Product section
    names (appliesTo), the article numbers it cites (references), the tools
    it names (usesTool).

Scores: micro precision, recall and F1 per relation over the annotated
articles, the set of predicted objects against the set of gold objects.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build import extract_all, load_records  # noqa: E402

ROOT = HERE.parent
RELATIONS = ("addresses", "mentionsAlert", "assignedTo", "appliesTo", "references", "usesTool")
PREFIXES = {"ppms:": "http://example.org/ppms#", "sup:": "http://example.org/sapsupport#"}


def _norm(rel: str, v) -> str:
    """One spelling per object: alert numbers as ints, IRIs expanded."""
    if rel in ("addresses", "mentionsAlert"):
        return str(int(v[1] if isinstance(v, tuple) else v))
    if isinstance(v, tuple):
        return str(v[1])
    v = str(v)
    for p, ns in PREFIXES.items():
        if v.startswith(p):
            return ns + v[len(p):]
    return v


def predicted(extractions) -> dict[str, dict[str, set[str]]]:
    out: dict[str, dict[str, set[str]]] = {}
    for ex in extractions:
        rels = out.setdefault(ex.number, {r: set() for r in RELATIONS})
        for f in ex.facts:
            if f.p in rels and f.s == ("article", ex.number):
                rels[f.p].add(_norm(f.p, f.o))
    return out


def gold_sets(gold: dict) -> dict[str, dict[str, set[str]]]:
    out = {}
    for num, ann in (gold.get("articles") or {}).items():
        if not ann or ann.get("skip"):
            continue
        out[str(num)] = {r: {_norm(r, v) for v in (ann.get(r) or [])} for r in RELATIONS}
    return out


def score(pred: dict, gold: dict) -> dict[str, dict]:
    table = {}
    for rel in RELATIONS:
        tp = fp = fn = 0
        for num, g in gold.items():
            p = pred.get(num, {}).get(rel, set())
            tp += len(p & g[rel]); fp += len(p - g[rel]); fn += len(g[rel] - p)
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        table[rel] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
                      "tp": tp, "fp": fp, "fn": fn}
    return table


def errors(pred: dict, gold: dict) -> list[str]:
    lines = []
    for num, g in sorted(gold.items()):
        for rel in RELATIONS:
            p = pred.get(num, {}).get(rel, set())
            if p != g[rel]:
                lines.append(f"{num} {rel}: missing {sorted(g[rel] - p)} extra {sorted(p - g[rel])}")
    return lines


def template(records, n: int, path: Path) -> None:
    nums = sorted(r.number for r in records)
    random.Random(0).shuffle(nums)
    lines = ["# Gold annotations. Read each page in a browser and fill the fields.",
             "# Do not look at the extractor output while annotating.",
             "# Alerts as numbers; components as codes; products as ppms: IRIs from",
             "# supportkg/ontology/ppms_standin.ttl; tools as sup: IRIs from tools.ttl.",
             "# Put `skip: true` on a page that does not load.", "annotator: \"\"", "articles:"]
    for num in nums[:n]:
        lines += [f"  \"{num}\":   # https://userapps.support.sap.com/sap/support/knowledge/en/{num}",
                  *[f"    {r}: []" for r in RELATIONS]]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"{path} exists; not overwriting annotations")
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path} with {min(n, len(nums))} articles")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", default=str(ROOT / "data" / "parsed" / "articles.jsonl"))
    ap.add_argument("--gold", default=str(ROOT / "gold" / "annotations.yaml"))
    ap.add_argument("--linker", choices=["none", "hf"], default="none")
    ap.add_argument("--template", type=int, default=0, help="Write an annotation form for N articles.")
    ap.add_argument("--show-errors", action="store_true")
    args = ap.parse_args()

    records = load_records(Path(args.parsed))
    if args.template:
        template(records, args.template, Path(args.gold))
        return
    factory = None
    if args.linker == "hf":
        from link import EmbeddingLinker
        factory = lambda products, alerts: EmbeddingLinker(products, alerts)  # noqa: E731
    pred = predicted(extract_all(records, factory))
    gold = gold_sets(yaml.safe_load(Path(args.gold).read_text()))
    table = score(pred, gold)
    print(f"{len(gold)} annotated articles, linker={args.linker}\n")
    print(f"| relation | precision | recall | F1 | tp | fp | fn |\n|---|---|---|---|---|---|---|")
    for rel, s in table.items():
        print(f"| {rel} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} "
              f"| {s['tp']} | {s['fp']} | {s['fn']} |")
    out = ROOT / "out" / f"extraction_scores_{args.linker}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1))
    if args.show_errors:
        print("\n" + "\n".join(errors(pred, gold)))


if __name__ == "__main__":
    main()
