"""Score the cross-source link alert -> views against the gold list.

    python supportkg/src/evaluate_links.py                   # lexical baseline
    python supportkg/src/evaluate_links.py --method hf       # Hugging Face embeddings
    python supportkg/src/evaluate_links.py --method both --show

Settings fixed before any embedding result was seen: each method keeps its
top k = 2 views above its threshold (lexical 0.35, embedding 0.5). Scores:
micro precision, recall and F1 over (alert, view) pairs, and hit@2, the share
of alerts with at least one correct view among the two proposed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from collect import records_from_titles  # noqa: E402
from docs import load  # noqa: E402
from extract import title_alert  # noqa: E402
from link import ViewLinker  # noqa: E402

ROOT = HERE.parent


def alert_names(seeds: Path) -> dict[int, str]:
    out = {}
    for r in records_from_titles(seeds):
        hit = title_alert(r.title)
        if hit:
            out.setdefault(hit[0], hit[1])
    return out


def score(linker: ViewLinker, names: dict[int, str], gold: dict[int, list[str]]):
    tp = fp = fn = hits = 0
    rows = []
    for a, g in sorted(gold.items()):
        pred = [v for v, _ in linker.link(names[a])]
        p, gs = set(pred), set(g)
        tp += len(p & gs); fp += len(p - gs); fn += len(gs - p)
        hits += bool(p & gs)
        rows.append((a, names[a], pred, sorted(gs)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
            "hit@2": round(hits / len(gold), 3), "tp": tp, "fp": fp, "fn": fn,
            "alerts": len(gold)}, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=str(ROOT / "data" / "sap-hana-cloud-hana-database"))
    ap.add_argument("--seeds", default=str(ROOT / "seeds" / "articles.txt"))
    ap.add_argument("--gold", default=str(ROOT / "gold" / "alert_views.yaml"))
    ap.add_argument("--method", choices=["lexical", "hf", "both"], default="lexical")
    ap.add_argument("--show", action="store_true", help="Print every alert's proposals.")
    args = ap.parse_args()

    g = yaml.safe_load(Path(args.gold).read_text())
    gold = {int(k): v for k, v in g["alerts"].items()}
    if g.get("status") != "checked":
        print("NOTE: the gold list is still a draft; check it before reporting these scores.\n")
    docs = load(Path(args.docs))
    views = {d.name: d.description for d in docs if d.kind != "sql_statement"}
    missing = sorted({v for vs in gold.values() for v in vs} - set(views))
    if missing:
        raise SystemExit(f"gold views not in the documentation: {missing}")
    names = alert_names(Path(args.seeds))
    methods = ["lexical", "embedding"] if args.method == "both" else \
              ["embedding" if args.method == "hf" else "lexical"]
    results = {}
    for m in methods:
        table, rows = score(ViewLinker(views, method=m), names, gold)
        results[m] = table
        print(f"{m:9}  P={table['precision']:.3f}  R={table['recall']:.3f}  "
              f"F1={table['f1']:.3f}  hit@2={table['hit@2']:.3f}  ({table['alerts']} alerts)")
        if args.show:
            for a, name, pred, gs in rows:
                mark = "ok " if set(pred) & set(gs) else "-- "
                print(f"   {mark}alert {a:>3} {name[:40]:40} -> {pred}   gold {gs}")
    out = ROOT / "out" / "link_scores.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
