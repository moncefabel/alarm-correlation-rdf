"""How much of what the gold query needs reaches the prompt. No model is called.

For every in-scope question, the IRIs of the gold query (classes, properties,
entities) are compared with the schema context the agents would receive. A
gold IRI missing from the context can only be guessed, so this recall is an
upper bound on what any model can answer without inventing a URI.

    python kgqa/scripts/retrieval_recall.py --domain support --show \
        --out kgqa/results_support/retrieval_recall.json
    python kgqa/scripts/retrieval_recall.py --domain support --graph supportkg/out_noinv/graph.ttl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kgqa" / "src"))

from agents import extract_uris  # noqa: E402
from domains import DOMAINS  # noqa: E402
from schema_context import STANDARD_NS, SchemaIndex  # noqa: E402


def recall(schema: SchemaIndex, questions: list[dict]) -> dict:
    rows = []
    for q in questions:
        if q["type"] == "out_of_scope":
            continue
        need = {u for u in extract_uris(q["gold_sparql"], schema.prefixes)
                if not u.startswith(STANDARD_NS)}
        ctx = schema.relevant(q["question"])
        missing = sorted(u.split("#")[-1] for u in need if f"<{u}>" not in ctx)
        rows.append({"id": q["id"], "type": q["type"], "needed": len(need),
                     "missing": missing, "context_chars": len(ctx)})
    full = sum(not r["missing"] for r in rows)
    return {"retrieval": schema.retrieval, "questions": len(rows),
            "all_gold_iris_in_context": full,
            "iri_recall": round(1 - sum(len(r["missing"]) for r in rows)
                                / max(1, sum(r["needed"] for r in rows)), 3),
            "mean_context_chars": round(sum(r["context_chars"] for r in rows) / max(1, len(rows))),
            "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="support", choices=sorted(DOMAINS))
    ap.add_argument("--graph", default=None)
    ap.add_argument("--questions", default=None)
    ap.add_argument("--show", action="store_true", help="List the missing IRIs per question.")
    ap.add_argument("--out", default=None, help="Also save the numbers as JSON here.")
    args = ap.parse_args()
    d = DOMAINS[args.domain]
    graph = str(ROOT / (args.graph or d.graph))
    questions = yaml.safe_load((ROOT / (args.questions or d.questions)).read_text())["questions"]
    out = {}
    for mode in SchemaIndex.RETRIEVALS:
        res = recall(SchemaIndex(graph, retrieval=mode), questions)
        out[mode] = {k: v for k, v in res.items() if k != "rows"}
        print(f"{mode:<14} gold IRIs in context: {res['iri_recall']:.3f}   "
              f"questions with every gold IRI: {res['all_gold_iris_in_context']}/{res['questions']}   "
              f"mean context: {res['mean_context_chars']} chars")
        if args.show:
            for r in res["rows"]:
                if r["missing"]:
                    print(f"   {r['id']:<4} missing {', '.join(r['missing'])}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
