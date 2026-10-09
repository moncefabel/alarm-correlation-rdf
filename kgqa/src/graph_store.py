"""Graph access layer.

Switches between a local in-memory rdflib graph (fast tests) and a real
SPARQL endpoint (Apache Jena Fuseki) via the SPARQL_ENDPOINT env var.

The Fuseki path is the one the benchmark runs against; the in-memory path is
kept for tests and quick iteration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery
from rdflib.plugins.stores import sparqlstore


@dataclass
class QueryResult:
    ok: bool
    rows: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None
    n_rows: int = 0


class GraphStore:
    def __init__(self, ttl_path: str | None = None, endpoint: str | None = None):
        self.endpoint = endpoint or os.getenv("SPARQL_ENDPOINT")
        if self.endpoint:
            store = sparqlstore.SPARQLStore(self.endpoint, returnFormat="json")
            self.graph = Graph(store=store)
            self.mode = f"fuseki:{self.endpoint}"
        else:
            self.graph = Graph()
            if ttl_path:
                self.graph.parse(ttl_path, format="turtle")
            self.mode = f"in-memory ({len(self.graph)} triples)"

    def validate_syntax(self, query: str) -> tuple[bool, str | None]:
        """Cheap pre-flight check. Catches malformed SPARQL before execution
        so the repair loop gets a precise parser error, not a 500."""
        try:
            prepareQuery(query)
            return True, None
        except Exception as exc:  # noqa: BLE001 - we want the message verbatim
            return False, str(exc)

    def select(self, query: str, limit: int = 200) -> QueryResult:
        ok, err = self.validate_syntax(query)
        if not ok:
            return QueryResult(ok=False, error=f"SyntaxError: {err}")
        try:
            result = self.graph.query(query)
            # ASK returns a boolean, not bindings. Normalised to one row so the
            # scorer compares it like any other answer set.
            if result.type == "ASK":
                rows = [{"ask": str(bool(result.askAnswer)).lower()}]
                return QueryResult(ok=True, rows=rows, n_rows=1)
            rows: list[dict[str, str]] = []
            for i, row in enumerate(result):
                if i >= limit:
                    break
                rows.append({str(k): str(v) for k, v in row.asdict().items()})
            return QueryResult(ok=True, rows=rows, n_rows=len(rows))
        except Exception as exc:  # noqa: BLE001
            return QueryResult(ok=False, error=f"ExecutionError: {exc}")

    def describe_subgraph(self, uris: list[str], hops: int = 1) -> list[tuple[str, str, str]]:
        """Provenance support: pull the neighbourhood that justifies an answer.

        This is what gets returned alongside the query as the *explanation*.
        """
        triples: list[tuple[str, str, str]] = []
        seen = set(uris)
        frontier = list(uris)
        for _ in range(hops):
            nxt: list[str] = []
            for uri in frontier:
                q = f"""
                SELECT ?s ?p ?o WHERE {{
                  {{ <{uri}> ?p ?o . BIND(<{uri}> AS ?s) }}
                  UNION
                  {{ ?s ?p <{uri}> . BIND(<{uri}> AS ?o) }}
                }} LIMIT 50
                """
                res = self.select(q)
                for r in res.rows:
                    t = (r.get("s", ""), r.get("p", ""), r.get("o", ""))
                    triples.append(t)
                    for node in (t[0], t[2]):
                        if node.startswith("http") and node not in seen:
                            seen.add(node)
                            nxt.append(node)
            frontier = nxt
        return triples


def answer_set(rows: list[dict[str, Any]]) -> set[str]:
    """Order-insensitive, projection-insensitive answer set for scoring."""
    out: set[str] = set()
    for row in rows:
        for v in row.values():
            out.add(str(v).strip())
    return out
