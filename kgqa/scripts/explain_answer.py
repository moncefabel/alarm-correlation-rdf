"""Show why an answer is an answer: evidence triples and their sources.

No model is called. The query is a benchmark gold query, or one you give.

    python kgqa/scripts/explain_answer.py --id M01
    python kgqa/scripts/explain_answer.py --id I01 --graph supportkg/out_noinv/graph.ttl
    python kgqa/scripts/explain_answer.py --query my_query.rq
    python kgqa/scripts/explain_answer.py --all        # coverage over the benchmark
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kgqa" / "src"))

from domains import SUPPORT  # noqa: E402
from explain import Explainer, summary  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=str(ROOT / SUPPORT.graph))
    ap.add_argument("--questions", default=str(ROOT / SUPPORT.questions))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--id")
    g.add_argument("--query", help="File holding a SPARQL SELECT or ASK query.")
    g.add_argument("--all", action="store_true")
    args = ap.parse_args()

    ex = Explainer.for_graph(args.graph)
    questions = yaml.safe_load(Path(args.questions).read_text())["questions"]
    if args.all:
        full = 0
        answered = [q for q in questions if q["type"] != "out_of_scope"]
        for q in answered:
            s = summary(ex.explain(q["gold_sparql"]))
            ok = s["evidence_triples"] > 0 and s["sourced"] == s["evidence_triples"]
            full += ok
            print(f"{q['id']:<4} {s['evidence_triples']:>3} triples  {s['sourced']:>3} sourced  "
                  f"{s['inferred']:>2} inferred  {'/'.join(s['kinds'])}")
        print(f"\nfully sourced answers: {full}/{len(answered)}")
        return
    if args.id:
        q = next(q for q in questions if q["id"] == args.id)
        print(f"Question: {q['question']}\nAnswer:   {', '.join(map(str, q['expected']))}\n")
        query = q["gold_sparql"]
    else:
        query = Path(args.query).read_text()
    evidence = ex.explain(query)
    if not evidence:
        print("No evidence: the query matches nothing in this graph.")
    print("Because:")
    for e in evidence:
        print(e.render())


if __name__ == "__main__":
    main()
