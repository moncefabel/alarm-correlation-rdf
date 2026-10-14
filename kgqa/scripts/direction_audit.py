"""Count causal-direction inversions in every run file of a results folder.

    python kgqa/scripts/direction_audit.py                 # kgqa/results
    python kgqa/scripts/direction_audit.py --out other/dir

For each arm: wrong answers over all runs, and how many of them traverse
`net:causallyLinkedTo` the wrong way round (see kgqa/src/direction.py).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kgqa" / "src"))

from direction import inverted_entities  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "kgqa" / "results"))
    ap.add_argument("--questions", default=str(ROOT / "kgqa" / "benchmark" / "questions.yaml"))
    args = ap.parse_args()

    gold = {q["id"]: q.get("gold_sparql", "")
            for q in yaml.safe_load(Path(args.questions).read_text())["questions"]}
    stats: dict[str, dict] = defaultdict(lambda: {"runs": 0, "answers": 0, "wrong": 0,
                                                  "inverted": 0, "inverted_wrong": 0,
                                                  "where": defaultdict(int)})
    for f in sorted(Path(args.out).glob("*_run*.json")):
        if f.name.endswith(".partial.jsonl"):
            continue
        arm = re.sub(r"_run\d+\.json$", "", f.name)
        data = json.loads(f.read_text())
        s = stats[arm]
        s["runs"] += 1
        for row, tr in zip(data["rows"], data["traces"]):
            s["answers"] += 1
            wrong = not row["exact"]
            s["wrong"] += wrong
            if not row.get("refused") and inverted_entities(tr.get("query", ""), gold[row["id"]]):
                s["inverted"] += 1
                s["inverted_wrong"] += wrong
                s["where"][row["id"]] += 1

    print(f"{'arm':22} {'runs':>4} {'wrong':>6} {'inverted':>9} {'of which wrong':>15}  questions")
    for arm in ("react", "react_typed", "orchestrator_untyped", "orchestrator",
                *sorted(set(stats) - {"react", "react_typed", "orchestrator_untyped", "orchestrator"})):
        if arm not in stats:
            continue
        s = stats[arm]
        where = ", ".join(f"{q}x{n}" for q, n in sorted(s["where"].items()))
        print(f"{arm:22} {s['runs']:>4} {s['wrong']:>6} {s['inverted']:>9} {s['inverted_wrong']:>15}  {where}")


if __name__ == "__main__":
    main()
