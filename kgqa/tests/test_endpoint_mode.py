"""GraphStore in endpoint mode, over HTTP.

Fuseki speaks the SPARQL 1.1 Protocol. This test stands up a minimal local
server implementing that protocol on top of rdflib, points GraphStore at it
through SPARQL_ENDPOINT, and checks that every gold query returns the same
answer as the in-memory path.

It validates the client side of the Fuseki switch (protocol, result parsing,
ASK handling). The server side is validated against a real Fuseki with
`docker compose up` and scripts/load_fuseki.sh, see kgqa/README.md.
"""

from __future__ import annotations

import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
import yaml
from rdflib import Graph

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kgqa" / "src"))

from graph_store import GraphStore, answer_set  # noqa: E402

ONTOLOGY = str(ROOT / "ontology" / "alarm_ontology.ttl")
QUESTIONS = yaml.safe_load((ROOT / "kgqa" / "benchmark" / "questions.yaml").read_text())["questions"]


def _handler(graph: Graph):
    class SparqlProtocol(BaseHTTPRequestHandler):
        def _answer(self, query: str) -> None:
            body = graph.query(query).serialize(format="json")
            self.send_response(200)
            self.send_header("Content-Type", "application/sparql-results+json")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            self._answer(parse_qs(urlparse(self.path).query)["query"][0])

        def do_POST(self):  # noqa: N802
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
            ctype = self.headers.get("Content-Type", "")
            query = raw if "sparql-query" in ctype else parse_qs(raw)["query"][0]
            self._answer(query)

        def log_message(self, *args):  # silence test output
            pass
    return SparqlProtocol


@pytest.fixture(scope="module")
def endpoint():
    g = Graph().parse(ONTOLOGY, format="turtle")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(g))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/alarms/query"
    server.shutdown()


def test_env_var_switches_to_endpoint_mode(endpoint, monkeypatch):
    monkeypatch.setenv("SPARQL_ENDPOINT", endpoint)
    assert GraphStore(ttl_path=ONTOLOGY).mode.startswith("fuseki:")


@pytest.mark.parametrize("q", QUESTIONS, ids=[q["id"] for q in QUESTIONS])
def test_endpoint_matches_in_memory(endpoint, q):
    remote = GraphStore(endpoint=endpoint).select(q["gold_sparql"])
    local = GraphStore(ttl_path=ONTOLOGY).select(q["gold_sparql"])
    assert remote.ok, remote.error
    assert answer_set(remote.rows) == answer_set(local.rows)
