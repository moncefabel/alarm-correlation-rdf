"""The LLM client against a local OpenAI-compatible server.

Groq, Mistral, vLLM and Ollama all expose the OpenAI chat-completions API. This
test stands up a minimal server speaking that API and checks the two behaviours
a free-tier run depends on: retrying after a rate limit (HTTP 429), and carrying
on when the provider rejects an optional parameter such as `seed`.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("openai")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kgqa" / "src"))

from evaluate import make_llm  # noqa: E402


def _completion(content: str) -> bytes:
    return json.dumps({
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }).encode()


class FakeProvider:
    def __init__(self, rate_limit_first: int = 0, reject_seed: bool = False):
        self.rate_limit_first = rate_limit_first
        self.reject_seed = reject_seed
        self.bodies: list[dict] = []

    def handler(self):
        provider = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                provider.bodies.append(body)
                if len(provider.bodies) <= provider.rate_limit_first:
                    self._send(429, json.dumps({"error": {"message": "rate limited"}}).encode(),
                               {"retry-after-ms": "10"})
                elif provider.reject_seed and "seed" in body:
                    self._send(400, json.dumps({"error": {
                        "message": "seed is not supported",
                        "type": "invalid_request_error"}}).encode())
                else:
                    self._send(200, _completion("SELECT ?a WHERE { ?a ?p ?o }"))

            def _send(self, code, payload, headers=None):
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass
        return H


@pytest.fixture
def serve(monkeypatch):
    servers = []

    def start(provider: FakeProvider) -> FakeProvider:
        srv = ThreadingHTTPServer(("127.0.0.1", 0), provider.handler())
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{srv.server_port}/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        return provider

    yield start
    for srv in servers:
        srv.shutdown()


def test_retries_after_rate_limit(serve):
    p = serve(FakeProvider(rate_limit_first=2))
    out = make_llm("openai/gpt-oss-120b", max_retries=4)("sys", "user")
    assert out.startswith("SELECT")
    assert len(p.bodies) == 3                 # two 429s, then success


def test_drops_rejected_optional_parameter(serve):
    p = serve(FakeProvider(reject_seed=True))
    out = make_llm("openai/gpt-oss-120b", seed=0)("sys", "user")
    assert out.startswith("SELECT")
    assert "seed" in p.bodies[0] and "seed" not in p.bodies[-1]


def test_sends_model_temperature_and_reasoning_effort(serve):
    p = serve(FakeProvider())
    make_llm("openai/gpt-oss-120b", reasoning_effort="low")("sys", "user")
    body = p.bodies[0]
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["temperature"] == 0
    assert body["reasoning_effort"] == "low"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
