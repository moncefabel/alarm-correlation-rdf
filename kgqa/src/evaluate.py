"""Evaluation harness.

Metrics follow what the text-to-SPARQL literature reports, so numbers stay
comparable, plus two it under-reports: hallucinated-IRI rate and repair
efficiency. Refusals are scored explicitly on out-of-scope questions.

Usage (from the repository root):
    python kgqa/src/evaluate.py --topology both --runs 3

Any OpenAI-compatible endpoint works: set OPENAI_BASE_URL to point at a
self-hosted model (vLLM, Ollama, Mistral) and the comparison stays
model-controlled, since both topologies call the same function.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as stats
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from agents import build_orchestrator, build_react, register_schema, run  # noqa: E402
from graph_store import GraphStore, answer_set  # noqa: E402
from schema_context import SchemaIndex  # noqa: E402


# --------------------------------------------------------------------------
# Plug your provider here. Kept behind one function so the topology
# comparison is model-controlled: both arms call exactly this.
# --------------------------------------------------------------------------
def make_llm(model: str = "gpt-4o", seed: int | None = None,
             reasoning_effort: str | None = None, max_retries: int = 8):
    """One function, called by both topologies.

    Works with any OpenAI-compatible endpoint (OPENAI_BASE_URL): OpenAI, Groq,
    Mistral, vLLM, Ollama. The client retries rate-limit (429) and server errors
    with exponential backoff and honours the provider's retry-after header,
    which matters on free tiers where a long run would otherwise stop halfway.
    """
    from openai import BadRequestError, OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
                    base_url=os.environ.get("OPENAI_BASE_URL"),
                    max_retries=max_retries, timeout=120)
    extra: dict = {}
    if seed is not None:
        extra["seed"] = seed
    if reasoning_effort:
        extra["reasoning_effort"] = reasoning_effort

    def llm(system: str, user: str) -> str:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        try:
            r = client.chat.completions.create(model=model, temperature=0,
                                               messages=messages, **extra)
        except BadRequestError as exc:
            # Some providers reject seed or reasoning_effort: drop them once,
            # keep going, and say so rather than failing the whole run.
            if not extra:
                raise
            print(f"  [llm] provider rejected {sorted(extra)}: {exc}. Retrying without.")
            extra.clear()
            r = client.chat.completions.create(model=model, temperature=0,
                                               messages=messages)
        return r.choices[0].message.content or ""
    return llm


def prf(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred or not gold:
        return 0.0, 0.0, 0.0
    tp = len(pred & gold)
    p = tp / len(pred)
    r = tp / len(gold)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def label_index(store: GraphStore) -> dict[str, str]:
    """Label to IRI, for labels that name exactly one node in the graph."""
    res = store.select("PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
                       "SELECT ?s ?l WHERE { ?s rdfs:label ?l }")
    seen: dict[str, set[str]] = {}
    for r in res.rows:
        seen.setdefault(str(r["l"]).strip().lower(), set()).add(str(r["s"]))
    return {lbl: next(iter(iris)) for lbl, iris in seen.items() if len(iris) == 1}


def canonical(rows: list[dict], labels: dict[str, str]) -> set[str]:
    """Answer set where an entity counts once, whether the query projected its
    IRI, its rdfs:label, or both. Without this, a correct answer given as labels
    scores zero, and projecting ?alarm and ?label together halves precision.
    Applied identically to both topologies and to the gold answers."""
    return {labels.get(v.lower(), v) for v in answer_set(rows)}


def evaluate(topology: str, questions: list[dict], schema: SchemaIndex,
             store: GraphStore, llm, **flags) -> dict:
    agent = (build_react(llm, schema, store, **flags) if topology == "react"
             else build_orchestrator(llm, schema, store, **flags))
    rows, traces = [], []
    labels = label_index(store)
    for q in questions:
        tr = run(agent, q["question"], topology, store)
        qtype = q.get("type", "factual")
        if qtype == "out_of_scope":
            # Correct behaviour is to refuse. An executed query that happens to
            # return nothing is not a refusal and is not credited as one.
            p = r = f = 1.0 if tr.refused else 0.0
        elif tr.refused:
            p = r = f = 0.0                     # refused an answerable question
        else:
            gold = store.select(q["gold_sparql"])
            p, r, f = prf(canonical(tr.rows, labels), canonical(gold.rows, labels))
        rows.append({
            "id": q["id"], "type": qtype, "refused": tr.refused,
            "valid": tr.ok, "exact": f == 1.0, "f1": f, "precision": p, "recall": r,
            "repairs": tr.repairs, "llm_calls": tr.llm_calls, "seconds": tr.seconds,
            "hallucinated": len(tr.hallucinated_uris),
            "provenance_triples": len(tr.provenance),
        })
        traces.append(tr.__dict__)
        print(f"  [{topology}] {q['id']:<8} F1={f:.2f} valid={tr.ok} "
              f"repairs={tr.repairs} calls={tr.llm_calls}")

    def mean(k):
        return round(stats.mean([r[k] for r in rows]), 4) if rows else 0.0

    summary = {
        "topology": topology, "flags": flags, "n": len(rows),
        "execution_accuracy": mean("exact"),        # strict: answer set identical
        "answer_f1": mean("f1"),
        "query_validity_rate": mean("valid"),       # executed without error
        "hallucinated_uri_rate": round(
            sum(r["hallucinated"] > 0 for r in rows) / max(len(rows), 1), 4),
        "mean_repairs": mean("repairs"),
        "mean_llm_calls": mean("llm_calls"),        # the cost axis mKGQAgent reports
        "mean_seconds": mean("seconds"),
    }
    by_type: dict[str, list[float]] = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r["f1"])
    summary["f1_by_type"] = {k: round(stats.mean(v), 4) for k, v in by_type.items()}
    oos = [r for r in rows if r["type"] == "out_of_scope"]
    ins = [r for r in rows if r["type"] != "out_of_scope"]
    summary["refusal_accuracy"] = round(
        sum(r["refused"] for r in oos) / len(oos), 4) if oos else None
    summary["false_refusal_rate"] = round(
        sum(r["refused"] for r in ins) / len(ins), 4) if ins else None
    return {"summary": summary, "rows": rows, "traces": traces}


def aggregate(runs: list[dict]) -> dict:
    """Mean and standard deviation over repeated runs. A single run's gap
    between topologies is usually noise at this benchmark size."""
    keys = [k for k, v in runs[0].items() if isinstance(v, (int, float)) and k != "n"]
    out = {"topology": runs[0]["topology"], "runs": len(runs), "n": runs[0]["n"]}
    for k in keys:
        vals = [r[k] for r in runs if r.get(k) is not None]
        out[k] = round(stats.mean(vals), 4)
        out[k + "_std"] = round(stats.stdev(vals), 4) if len(vals) > 1 else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="kgqa/benchmark/questions.yaml")
    ap.add_argument("--ontology", default="ontology/alarm_ontology.ttl")
    ap.add_argument("--topology", default="both", choices=["react", "orchestrator", "both"])
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--runs", type=int, default=1, help="Repeated runs, seeds 0..runs-1.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only the first N questions (smoke test, saves quota).")
    ap.add_argument("--ids", default="",
                    help="Comma-separated question ids, e.g. F01,C01,K03,S01.")
    ap.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"],
                    help="For reasoning models (gpt-oss): low saves tokens on free tiers.")
    ap.add_argument("--no-schema-retrieval", action="store_true",
                    help="Ablation: dump the full ontology instead of retrieving.")
    ap.add_argument("--out", default="kgqa/results")
    args = ap.parse_args()

    schema = SchemaIndex(args.ontology)
    register_schema(schema)
    store = GraphStore(ttl_path=args.ontology)
    print(f"graph: {store.mode}\nschema: {len(schema.items)} items")

    questions = yaml.safe_load(Path(args.questions).read_text())["questions"]
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
        questions = [q for q in questions if q["id"] in wanted]
    if args.limit:
        questions = questions[:args.limit]
    print(f"questions: {len(questions)} | model: {args.model} | "
          f"endpoint: {os.environ.get('OPENAI_BASE_URL', 'OpenAI default')}")
    flags = {"retrieve_schema": not args.no_schema_retrieval}

    Path(args.out).mkdir(parents=True, exist_ok=True)
    topos = ["react", "orchestrator"] if args.topology == "both" else [args.topology]
    final = []
    for t in topos:
        per_run = []
        for seed in range(args.runs):
            print(f"\n=== {t} | run {seed} ===")
            llm = make_llm(args.model, seed=seed if args.runs > 1 else None,
                           reasoning_effort=args.reasoning_effort)
            res = evaluate(t, questions, schema, store, llm, **flags)
            per_run.append(res["summary"])
            tag = f"{t}{'_fullschema' if args.no_schema_retrieval else ''}_run{seed}"
            Path(args.out, f"{tag}.json").write_text(json.dumps(res, indent=2, default=str))
        final.append(aggregate(per_run))

    Path(args.out, "summary.json").write_text(json.dumps(final, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
