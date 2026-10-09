"""Two agent topologies over identical primitives.

This is the contribution. ARUQULA is ReAct. MetaboT is a hierarchical
supervisor with specialised sub-agents. mKGQAgent is plan/action/feedback.
They are evaluated on different KGs, with different models, against different
benchmarks, so the contribution of the *topology* has never been isolated.

Here both topologies call the SAME llm(), the SAME SchemaIndex and the SAME
GraphStore. Only the control flow differs. That makes the comparison clean.

Everything downstream of `generate` is shared, so any difference in the
numbers is attributable to control flow, not to prompt or tooling drift.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, TypedDict

from langgraph.graph import END, StateGraph

from graph_store import GraphStore, QueryResult
from schema_context import SchemaIndex

MAX_REPAIRS = int(os.getenv("MAX_REPAIRS", "3"))

LLM = Callable[[str, str], str]          # (system, user) -> completion


# --------------------------------------------------------------------------
# Shared primitives
# --------------------------------------------------------------------------

GEN_SYSTEM = """You write SPARQL 1.1 SELECT queries against an RDF knowledge graph.
Rules:
- Use ONLY the classes and properties given in the schema. Never invent a URI.
- Always include the PREFIX declarations you use.
- Return ONLY the query, no prose, no markdown fences.
- For causal questions use the causal properties, not mere co-occurrence.
- Scope: the graph describes network devices, alarms, incidents, attack patterns
  and the causal links between alarms. Questions about root causes, where to
  intervene, what a prevented or blocked alarm would have avoided, or which
  alarm or device matters most ARE in scope: answer them from the causal links.
- Return exactly OUT_OF_SCOPE only when the question is about something this
  graph does not describe at all (general knowledge, another domain)."""

REPAIR_SYSTEM = """You repair a SPARQL query that failed. Return ONLY the corrected query.
Do not explain. Never invent URIs outside the provided schema."""


_FENCED = re.compile(r"```(?:sparql|SPARQL)?\s*\n(.*?)```", re.S)
_QUERY_START = re.compile(r"^\s*(PREFIX|BASE|SELECT|ASK|CONSTRUCT|DESCRIBE)\b", re.I | re.M)


def strip_fences(text: str) -> str:
    """Extract the query from a model answer.

    Real models often wrap the query in a fence or add a sentence before it,
    despite the instruction. Left as is, that sentence becomes a syntax error and
    burns repair attempts on formatting rather than on the query. The same
    extraction is applied to both topologies, so it does not bias the comparison.
    """
    text = (text or "").strip()
    fenced = _FENCED.search(text)
    if fenced:
        text = fenced.group(1).strip()
    m = _QUERY_START.search(text)
    if m:
        return text[m.start():].strip()
    if REFUSAL in text:
        return REFUSAL
    return text


REFUSAL = "OUT_OF_SCOPE"

_PREFIX_DECL = re.compile(r"PREFIX\s+([A-Za-z][\w-]*)?:\s*<([^>]+)>", re.I)
_LITERAL = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')
_FULL_IRI = re.compile(r"<(https?://[^>]+)>")
_PNAME = re.compile(r"(?<![\w?$<])([A-Za-z][\w-]*)?:([A-Za-z_][\w-]*)")


def extract_uris(query: str, default_prefixes: dict[str, str] | None = None) -> set[str]:
    """Every IRI a query references, full or prefixed.

    LLMs almost always write prefixed names (net:causallyLinkedTo). A gate that
    only inspects <http://...> forms never fires on them, which silently turns
    the symbolic check into a no-op. Prefixes declared in the query win over the
    schema defaults; an undeclared prefix is reported as-is, since it is itself
    an invented term.
    """
    prefixes = dict(default_prefixes or {})
    for p, ns in _PREFIX_DECL.findall(query):
        prefixes[p or ""] = ns
    body = _PREFIX_DECL.sub(" ", query)
    body = _LITERAL.sub(" ", body)
    # Full IRIs before comments: an IRI contains '#', and stripping comments
    # first would cut it in half and hide an invented term.
    out = set(_FULL_IRI.findall(body))
    body = _FULL_IRI.sub(" ", body)
    body = re.sub(r"#[^\n]*", " ", body)
    for p, local in _PNAME.findall(body):
        ns = prefixes.get(p or "")
        out.add(ns + local if ns else f"{p}:{local}")
    return out


def hallucinated(query: str, schema: SchemaIndex) -> list[str]:
    return sorted(u for u in extract_uris(query, schema.prefixes) if not schema.is_known(u))


@dataclass
class Trace:
    topology: str = ""
    question: str = ""
    query: str = ""
    repairs: int = 0
    llm_calls: int = 0
    seconds: float = 0.0
    ok: bool = False
    refused: bool = False
    error: str | None = None
    rows: list[dict] = field(default_factory=list)
    hallucinated_uris: list[str] = field(default_factory=list)
    provenance: list[tuple] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)


class S(TypedDict, total=False):
    question: str
    qtype: Literal["factual", "causal", "counterfactual", "out_of_scope"]
    schema: str
    query: str
    result: QueryResult
    repairs: int
    llm_calls: int
    steps: list[str]
    error: str
    refused: bool


# --------------------------------------------------------------------------
# Topology A, ReAct: one loop, model decides when to stop
# --------------------------------------------------------------------------

def build_react(llm: LLM, schema: SchemaIndex, store: GraphStore, retrieve_schema: bool = True):
    def think_act(s: S) -> S:
        ctx = schema.relevant(s["question"]) if retrieve_schema else schema.full()
        failed = bool(s.get("result")) and not s["result"].ok
        prior = f"\n\nPrevious attempt failed:\n{s['query']}\nError: {s['result'].error}" \
            if failed else ""
        if failed:
            # State is written in nodes only: LangGraph discards writes made
            # inside a conditional-edge function.
            s["repairs"] = s.get("repairs", 0) + 1
        user = (f"{schema.prefix_header()}\n\nSchema:\n{ctx}\n\n"
                f"Question: {s['question']}{prior}")
        out = strip_fences(llm(GEN_SYSTEM if not prior else REPAIR_SYSTEM, user))
        s["query"] = out
        s["schema"] = ctx
        s["llm_calls"] = s.get("llm_calls", 0) + 1
        s.setdefault("steps", []).append("think_act")
        return s

    def observe(s: S) -> S:
        if s["query"].strip() == REFUSAL:
            s["refused"] = True
            s["result"] = QueryResult(ok=True)
            s.setdefault("steps", []).append("refuse")
            return s
        s["result"] = store.select(s["query"])
        s.setdefault("steps", []).append("observe")
        return s

    def again(s: S) -> str:
        if s["result"].ok or s.get("repairs", 0) >= MAX_REPAIRS:
            return END
        return "think_act"

    g = StateGraph(S)
    g.add_node("think_act", think_act)
    g.add_node("observe", observe)
    g.set_entry_point("think_act")
    g.add_edge("think_act", "observe")
    g.add_conditional_edges("observe", again, {"think_act": "think_act", END: END})
    return g.compile()


# --------------------------------------------------------------------------
# Topology B, hierarchical orchestrator: specialised stages, MetaboT-style
# --------------------------------------------------------------------------

def build_orchestrator(llm: LLM, schema: SchemaIndex, store: GraphStore,
                       retrieve_schema: bool = True, scope_check: bool = True):
    def classify(s: S) -> S:
        s.setdefault("steps", []).append("classify")
        if not scope_check:
            s["qtype"] = "factual"
            return s
        out = llm(
            "Classify the question. Answer with exactly one word: "
            "factual, causal, counterfactual, or out_of_scope.",
            f"Graph domain: network alarms, incidents, causal dependencies.\n"
            f"Question: {s['question']}",
        ).strip().lower()
        s["qtype"] = next(
            (t for t in ("counterfactual", "causal", "out_of_scope", "factual") if t in out),
            "factual",
        )
        s["refused"] = s["qtype"] == "out_of_scope"
        s["llm_calls"] = s.get("llm_calls", 0) + 1
        return s

    def link(s: S) -> S:
        s["schema"] = schema.relevant(s["question"]) if retrieve_schema else schema.full()
        s.setdefault("steps", []).append("link")
        return s

    def generate(s: S) -> S:
        hint = {
            "causal": "This is a ROOT-CAUSE question. Traverse causal properties "
                      "transitively; do not rely on co-occurrence.",
            "counterfactual": "This is a COUNTERFACTUAL question. Identify the "
                              "downstream set that would not occur if the named "
                              "node were removed or blocked.",
        }.get(s.get("qtype", "factual"), "")
        user = (f"{schema.prefix_header()}\n\nSchema:\n{s['schema']}\n\n"
                f"{hint}\n\nQuestion: {s['question']}")
        s["query"] = strip_fences(llm(GEN_SYSTEM, user))
        s["llm_calls"] = s.get("llm_calls", 0) + 1
        s.setdefault("steps", []).append("generate")
        return s

    def verify(s: S) -> S:
        """Symbolic gate before execution: no invented URIs, valid syntax."""
        s.setdefault("steps", []).append("verify")
        if s["query"].strip() == REFUSAL:
            s["refused"] = True
            s["error"] = ""
            return s
        bad = hallucinated(s["query"], schema)
        ok, err = store.validate_syntax(s["query"])
        if bad:
            s["error"] = f"Unknown URIs not in ontology: {bad}"
        elif not ok:
            s["error"] = f"SyntaxError: {err}"
        else:
            s["error"] = ""
        return s

    def execute(s: S) -> S:
        s["result"] = store.select(s["query"])
        s.setdefault("steps", []).append("execute")
        return s

    def refine(s: S) -> S:
        s["repairs"] = s.get("repairs", 0) + 1
        err = s.get("error") or (s["result"].error if s.get("result") else "unknown")
        user = (f"{schema.prefix_header()}\n\nSchema:\n{s['schema']}\n\n"
                f"Question: {s['question']}\n\nFailed query:\n{s['query']}\n\nError: {err}")
        s["query"] = strip_fences(llm(REPAIR_SYSTEM, user))
        s["llm_calls"] = s.get("llm_calls", 0) + 1
        s.setdefault("steps", []).append("refine")
        return s

    def after_classify(s: S) -> str:
        return END if s.get("refused") else "link"

    def after_verify(s: S) -> str:
        if s.get("refused"):
            return END
        if s.get("error"):
            return "refine" if s.get("repairs", 0) < MAX_REPAIRS else END
        return "execute"

    def after_execute(s: S) -> str:
        if s["result"].ok:
            return END
        return "refine" if s.get("repairs", 0) < MAX_REPAIRS else END

    g = StateGraph(S)
    for name, fn in [("classify", classify), ("link", link), ("generate", generate),
                     ("verify", verify), ("execute", execute), ("refine", refine)]:
        g.add_node(name, fn)
    g.set_entry_point("classify")
    g.add_conditional_edges("classify", after_classify, {"link": "link", END: END})
    g.add_edge("link", "generate")
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", after_verify,
                            {"execute": "execute", "refine": "refine", END: END})
    g.add_conditional_edges("execute", after_execute, {"refine": "refine", END: END})
    g.add_edge("refine", "verify")
    return g.compile()


# --------------------------------------------------------------------------

def run(agent, question: str, topology: str, store: GraphStore,
        with_provenance: bool = True) -> Trace:
    t0 = time.time()
    final: S = agent.invoke({"question": question, "repairs": 0, "llm_calls": 0,
                             "steps": [], "refused": False})
    refused = bool(final.get("refused"))
    res: QueryResult = final.get("result") or QueryResult(ok=refused,
                                                          error=None if refused else "no execution")
    tr = Trace(
        topology=topology, question=question, query=final.get("query", ""),
        repairs=final.get("repairs", 0), llm_calls=final.get("llm_calls", 0),
        seconds=round(time.time() - t0, 2), ok=res.ok or refused, refused=refused,
        error=res.error, rows=[] if refused else res.rows, steps=final.get("steps", []),
    )
    if _SCHEMA is not None and tr.query and not refused:
        tr.hallucinated_uris = hallucinated(tr.query, _SCHEMA)
    if with_provenance and res.ok and res.rows and not refused:
        seeds = [v for r in res.rows[:5] for v in r.values() if str(v).startswith("http")]
        tr.provenance = store.describe_subgraph(seeds[:5], hops=1)
    return tr


_SCHEMA: SchemaIndex | None = None


def register_schema(schema: SchemaIndex) -> None:
    """Shared by both topologies so the hallucination metric is computed
    against exactly the same reference."""
    global _SCHEMA
    _SCHEMA = schema
