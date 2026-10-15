"""Baseline without SPARQL: retrieve documentation pages, let the model read them.

The question it answers: does the query step earn its place, or would a model
reading the right pages do as well? To isolate the query mechanism, the pages
are written from the same facts the agents query: every view, statement,
article, alert and product item of the materialised graph becomes one page
listing its description, its columns (name, data type, short description) and
its links to other entities, in both directions. The baseline therefore has
all the information the agents have, including the predicted alert-to-view
links, but must find it by text retrieval and read it rather than query it.
This is the most favourable fair setting for text retrieval; a baseline over
the raw documentation would also lose the extracted links.

Retrieval is BM25 over the pages (the lexical baseline beat an off-the-shelf
embedding model on SAP identifiers in the cross-source link experiment). The
model answers with a JSON list: entity IDs as printed on the pages, or values.
Answers are scored with the same two scorers as the agents. --k 0 gives the
closed-book baseline: the model answers from what it knows, with no pages.

    python kgqa/src/rag_baseline.py --model claude-haiku-5-5 --runs 3
    python kgqa/src/rag_baseline.py --model claude-haiku-5-5 --runs 3 \\
        --questions supportkg/benchmark/heldout.yaml --out kgqa/results_rag_heldout
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from rdflib import RDF, RDFS, Graph, Literal, URIRef

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents import REFUSAL  # noqa: E402
from answer_form import score_by_name  # noqa: E402
from evaluate import label_index, make_llm, score  # noqa: E402
from graph_store import GraphStore  # noqa: E402

PAGE_TYPES = {"MonitoringView": "monitoring view", "SystemView": "system view",
              "SQLStatement": "SQL statement", "KnowledgeArticle": "knowledge article",
              "Alert": "SAP HANA alert", "Product": "product", "ProductVersion": "product version",
              "SoftwareComponent": "software component", "SupportPackage": "support package",
              "ProductLine": "product line", "Tool": "tool"}
SKIP = {"comment", "label", "sourceUrl", "source", "type", "altLabel"}
COLUMN_NOTE = 80          # characters of each column description kept on a page

SYSTEM = """You answer questions about SAP HANA support using ONLY the documentation excerpts given.
Each excerpt starts with a line "ID: ...".
Answer with a JSON list of strings and nothing else.
- For an entity (view, column, SQL statement, knowledge article, alert, product item), give its ID exactly as written in the excerpts, e.g. "View_M_DISKS" or "Column_M_DISKS__HOST".
- For a value (a data type, a category, a name, a number), give the value.
- For a count, give the number, e.g. ["12"].
- If the excerpts do not settle the answer, give your best answer from them.
- If the question is not about SAP HANA support at all, answer exactly OUT_OF_SCOPE."""

CLOSED_BOOK = """You answer questions about SAP HANA support from your own knowledge.
Answer with a JSON list of strings and nothing else: names, numbers or values.
If the question is not about SAP HANA support at all, answer exactly OUT_OF_SCOPE."""


def local(u) -> str:
    return str(u).split("#")[-1].split("/")[-1]


def build_pages(g: Graph) -> list[dict]:
    label = {s: str(o) for s, o in g.subject_objects(RDFS.label)}
    plabel = {p: label.get(p, local(p)) for p in set(g.predicates())}
    pages = []
    for s, t in g.subject_objects(RDF.type):
        kind = PAGE_TYPES.get(local(t))
        if kind is None or not isinstance(s, URIRef):
            continue
        lines = [f"ID: {local(s)}", f"{label.get(s, local(s))} ({kind})"]
        for c in g.objects(s, RDFS.comment):
            lines.append(str(c))
        links, columns = defaultdict(list), []
        for p, o in sorted(g.predicate_objects(s)):
            name = local(p)
            if name in SKIP:
                continue
            if name == "hasColumn":
                cname = next((str(v) for v in g.objects(o, URIRef(str(p).rsplit("#", 1)[0] + "#columnName"))), local(o))
                dtype = next((str(v) for v in g.objects(o, URIRef(str(p).rsplit("#", 1)[0] + "#dataType"))), "")
                note = next((str(v) for v in g.objects(o, RDFS.comment)), "")
                note = note[:COLUMN_NOTE] + ("..." if len(note) > COLUMN_NOTE else "")
                columns.append(f"- {cname}, {dtype}: {note} [ID {local(o)}]")
            elif isinstance(o, Literal):
                links[plabel[p]].append(str(o))
            elif o in label:
                links[plabel[p]].append(f"{label[o]} [ID {local(o)}]")
        for name, vals in sorted(links.items()):
            lines.append(f"{name}: " + "; ".join(sorted(vals)))
        if columns:
            lines.append("Columns:")
            lines.extend(sorted(columns))
        pages.append({"id": local(s), "text": "\n".join(lines)})
    return sorted(pages, key=lambda p: p["id"])


def tokens(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[A-Za-z0-9_.()]+", text.lower()):
        w = w.strip(".()")
        if not w:
            continue
        out.append(w)
        if "_" in w:
            out.extend(x for x in w.split("_") if x)
    return out


class BM25:
    """BM25, plus a fixed bonus when every token of a page's title occurs in
    the question: a question naming M_CONNECTIONS gets the M_CONNECTIONS page
    even though that page is long and the name also occurs on many others."""

    def __init__(self, docs: list[str], titles: list[str] | None = None,
                 k1: float = 1.5, b: float = 0.75, title_bonus: float = 10.0):
        self.titles = [set(tokens(t)) for t in (titles or [""] * len(docs))]
        self.title_bonus = title_bonus
        self.toks = [Counter(tokens(d)) for d in docs]
        self.len = [sum(t.values()) for t in self.toks]
        self.avg = sum(self.len) / max(1, len(self.len))
        df = Counter(w for t in self.toks for w in t)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.k1, self.b = k1, b

    def top(self, query: str, k: int) -> list[int]:
        q = set(tokens(query))
        scores = []
        for i, t in enumerate(self.toks):
            s = 0.0
            for w in q:
                f = t.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            if self.titles[i] and self.titles[i] <= q:
                s += self.title_bonus
            scores.append(s)
        return sorted(range(len(scores)), key=lambda i: -scores[i])[:k]


def parse_answer(text: str) -> list[str] | None:
    """None means a refusal."""
    t = text.strip()
    if t.upper().startswith(REFUSAL) or t.strip('"[] ').upper() == REFUSAL:
        return None
    m = re.search(r"\[.*\]", t, re.S)
    if m:
        try:
            vals = json.loads(m.group(0))
            return [str(v).strip() for v in vals if str(v).strip()]
        except ValueError:
            pass
    return [x.strip(" -*\"'") for x in t.splitlines() if x.strip(" -*\"'")]


class RagBaseline:
    def __init__(self, graph_path: str, k: int = 5):
        g = Graph()
        g.parse(graph_path)
        self.pages = build_pages(g)
        self.index = BM25([p["text"] for p in self.pages],
                          [p["text"].splitlines()[1].rsplit(" (", 1)[0] for p in self.pages])
        self.page_text = {p["id"]: p["text"] for p in self.pages}
        self.k = k
        self.store = GraphStore(ttl_path=graph_path)
        self.labels = label_index(self.store)
        # local name -> IRI, to read an answer given as a page ID (the store's
        # select caps its rows, so the graph is read directly)
        self.iri = {local(x): str(x) for x in set(g.subjects()) if isinstance(x, URIRef)}

    def ask(self, llm, question: str) -> dict:
        if self.k:
            hits = [self.pages[i] for i in self.index.top(question, self.k)]
            context = "\n\n".join(p["text"] for p in hits)
            raw = llm(SYSTEM, f"Documentation excerpts:\n\n{context}\n\nQuestion: {question}")
        else:
            hits = []
            raw = llm(CLOSED_BOOK, f"Question: {question}")
        vals = parse_answer(raw)
        rows = [] if vals is None else [{"a": self.iri.get(v, v)} for v in vals]
        return {"refused": vals is None, "rows": rows, "raw": raw,
                "retrieved": [p["id"] for p in hits]}


def evaluate_rag(rag: RagBaseline, questions: list[dict], llm) -> dict:
    usage = getattr(llm, "usage", None)
    rows, traces = [], []
    for q in questions:
        before = usage["total"] if usage else 0
        t0 = time.time()
        tr = rag.ask(llm, q["question"])
        if q["type"] == "out_of_scope":
            f = fa = 1.0 if tr["refused"] else 0.0
        elif tr["refused"]:
            f = fa = 0.0
        else:
            gold = rag.store.select(q["gold_sparql"]).rows
            f = score(tr["rows"], gold, rag.labels)[2]
            fa = score_by_name(tr["rows"], gold, rag.labels, rag.store)[2]
        gold_rows = rag.store.select(q["gold_sparql"]).rows if q["type"] != "out_of_scope" else []
        needed = {local(v) for r in gold_rows for v in r.values() if str(v).startswith("http")}
        seen = "\n".join(rag.page_text[i] for i in tr["retrieved"])
        rows.append({"id": q["id"], "type": q["type"], "refused": tr["refused"], "valid": True,
                     "exact": f == 1.0, "exact_answer_form": fa == 1.0, "f1": f,
                     "gold_entities_retrieved": all(n in seen for n in needed),
                     "tokens": (usage["total"] - before) if usage else 0,
                     "seconds": round(time.time() - t0, 2)})
        traces.append({"question": q["question"], "query": "RAG: " + ", ".join(tr["retrieved"]),
                       "rows": tr["rows"], "refused": tr["refused"], "raw": tr["raw"]})
        print(f"  [rag k={rag.k}] {q['id']:<4} exact={rows[-1]['exact']!s:<5} "
              f"answer-form={rows[-1]['exact_answer_form']!s:<5} tokens={rows[-1]['tokens']}", flush=True)
    n = len(rows)
    inscope = [r for r in rows if r["type"] != "out_of_scope"]
    oos = [r for r in rows if r["type"] == "out_of_scope"]
    summary = {"topology": f"rag_k{rag.k}", "n": n,
               "execution_accuracy": round(sum(r["exact"] for r in rows) / n, 4),
               "execution_accuracy_answer_form": round(sum(r["exact_answer_form"] for r in rows) / n, 4),
               "refusal_accuracy": round(sum(r["refused"] for r in oos) / max(1, len(oos)), 4),
               "false_refusal_rate": round(sum(r["refused"] for r in inscope) / max(1, len(inscope)), 4),
               "gold_entities_retrieved_rate": round(sum(r["gold_entities_retrieved"] for r in inscope)
                                                     / max(1, len(inscope)), 4),
               "mean_tokens": round(sum(r["tokens"] for r in rows) / n, 1)}
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r["exact_answer_form"])
    summary["answer_form_by_type"] = {t: round(sum(v) / len(v), 3) for t, v in by_type.items()}
    return {"summary": summary, "rows": rows, "traces": traces}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--questions", default="supportkg/benchmark/questions.yaml")
    ap.add_argument("--graph", default="supportkg/out/graph.ttl")
    ap.add_argument("--k", type=int, default=5, help="Pages retrieved; 0 = closed book.")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="kgqa/results_rag")
    args = ap.parse_args()
    questions = yaml.safe_load((root / args.questions).read_text())["questions"]
    if args.limit:
        questions = questions[:args.limit]
    rag = RagBaseline(str(root / args.graph), k=args.k)
    print(f"pages: {len(rag.pages)} | k: {args.k} | questions: {len(questions)} | model: {args.model}")
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in range(args.runs):
        print(f"\n=== rag_k{args.k} | run {seed} ===")
        res = evaluate_rag(rag, questions, make_llm(args.model))
        (out / f"rag_k{args.k}_run{seed}.json").write_text(json.dumps(res, indent=2, default=str))
        summaries.append(res["summary"])
    keys = ["execution_accuracy", "execution_accuracy_answer_form", "refusal_accuracy",
            "false_refusal_rate", "gold_entities_retrieved_rate", "mean_tokens"]
    final = {"topology": f"rag_k{args.k}", "runs": len(summaries), "n": len(questions),
             **{k: round(statistics.mean(s[k] for s in summaries), 4) for k in keys}}
    (out / "summary.json").write_text(json.dumps(final, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
