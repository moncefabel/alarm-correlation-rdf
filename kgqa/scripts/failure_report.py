"""Show what the agents wrote for the questions they got wrong. No model is called.

Reads the run files an evaluation saved, and for every question that was not
answered exactly in at least one run prints the question, the expected answer,
and for each run the query the agent wrote and the rows it returned.

    python kgqa/scripts/failure_report.py kgqa/results_support --domain support
    python kgqa/scripts/failure_report.py kgqa/results_support --ids I01,M02 --arm orchestrator
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kgqa" / "src"))

from domains import DOMAINS  # noqa: E402


def short(v: object) -> str:
    return str(v).split("#")[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="A results directory, e.g. kgqa/results_support")
    ap.add_argument("--domain", default="support", choices=sorted(DOMAINS))
    ap.add_argument("--questions", default=None, help="Default: the domain's benchmark.")
    ap.add_argument("--arm", default=None, help="Only this arm, e.g. orchestrator.")
    ap.add_argument("--ids", default="", help="Comma-separated question ids; default: every failure.")
    ap.add_argument("--rows", type=int, default=6, help="Rows shown per answer.")
    args = ap.parse_args()
    d = DOMAINS[args.domain]
    questions = {q["id"]: q for q in yaml.safe_load((ROOT / (args.questions or d.questions)).read_text())["questions"]}
    wanted = [i for i in args.ids.split(",") if i]
    runs = sorted(Path(args.results).glob("*_run*.json"))
    if args.arm:
        runs = [r for r in runs if r.name.rsplit("_run", 1)[0] == args.arm]
    data = {r: json.loads(r.read_text()) for r in runs}
    failed: dict[str, int] = {}
    for blob in data.values():
        for row in blob["rows"]:
            if not row["exact"]:
                failed[row["id"]] = failed.get(row["id"], 0) + 1
    for qid in (wanted or sorted(failed)):
        q = questions[qid]
        print("=" * 78)
        print(f"{qid} [{q['type']}]  {q['question']}")
        print(f"expected: {q['expected']}")
        print("gold:")
        print("   " + q["gold_sparql"].strip().replace("\n", "\n   "))
        for path, blob in data.items():
            row = next(r for r in blob["rows"] if r["id"] == qid)
            trace = next(t for t in blob["traces"] if t["question"] == q["question"])
            print(f"--- {path.stem}: {'exact' if row['exact'] else 'WRONG'} (F1 {row['f1']})")
            if row["exact"]:
                continue
            print("   " + (trace.get("query") or "(no query)").strip().replace("\n", "\n   "))
            rows = trace.get("rows") or []
            print(f"   returned {len(rows)} row(s): "
                  + "; ".join(", ".join(short(v) for v in r.values()) for r in rows[:args.rows])
                  + (" ..." if len(rows) > args.rows else ""))
            if trace.get("error"):
                print(f"   error: {trace['error']}")


if __name__ == "__main__":
    main()
