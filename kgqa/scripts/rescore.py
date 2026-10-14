"""Score saved runs again, with and without answer-form scoring. No model is called.

    python kgqa/scripts/rescore.py kgqa/results_support
    python kgqa/scripts/rescore.py kgqa/results_support_noinv --graph supportkg/out_noinv/graph.ttl

For every run file, the answers the agent returned are read from the saved
traces and scored against the gold queries twice: with the original scorer
(the one the numbers in the README were computed with) and with the
answer-form scorer of answer_form.py. Questions whose result changes are
listed, so the change can be audited by hand.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kgqa" / "src"))

from answer_form import score_by_name  # noqa: E402
from domains import DOMAINS  # noqa: E402
from evaluate import label_index, score  # noqa: E402
from graph_store import GraphStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--domain", default="support", choices=sorted(DOMAINS))
    ap.add_argument("--graph", default=None)
    args = ap.parse_args()
    d = DOMAINS[args.domain]
    store = GraphStore(ttl_path=str(ROOT / (args.graph or d.graph)))
    labels = label_index(store)
    questions = {q["id"]: q for q in yaml.safe_load((ROOT / d.questions).read_text())["questions"]}
    arms: dict[str, list[tuple[float, float]]] = {}
    changed: dict[str, dict[str, int]] = {}
    for path in sorted(Path(args.results).glob("*_run*.json")):
        arm = path.name.rsplit("_run", 1)[0]
        blob = json.loads(path.read_text())
        v1 = v2 = 0
        for row, trace in zip(blob["rows"], blob["traces"]):
            q = questions[row["id"]]
            if q["type"] == "out_of_scope" or row["refused"]:
                a = b = 1.0 if row["exact"] else 0.0
            else:
                gold = store.select(q["gold_sparql"]).rows
                a = score(trace["rows"], gold, labels)[2]
                b = score_by_name(trace["rows"], gold, labels, store)[2]
            v1 += a == 1.0
            v2 += b == 1.0
            if (a == 1.0) != (b == 1.0):
                changed.setdefault(arm, {}).setdefault(row["id"], 0)
                changed[arm][row["id"]] += 1
        n = len(blob["rows"])
        arms.setdefault(arm, []).append((v1 / n, v2 / n))
    print(f"{'arm':<28} runs  original  answer-form   questions that changed (runs)")
    for arm, vals in arms.items():
        o = statistics.mean(v[0] for v in vals)
        f = statistics.mean(v[1] for v in vals)
        ch = ", ".join(f"{k} x{v}" for k, v in sorted(changed.get(arm, {}).items())) or "-"
        print(f"{arm:<28} {len(vals):<5} {o:<9.4f} {f:<12.4f}  {ch}")


if __name__ == "__main__":
    main()
