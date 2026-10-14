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
    def __init__(self, rate_limit_first: int = 0, reject_seed: bool = False,
                 reject_system: bool = False, answer: str = "SELECT ?a WHERE { ?a ?p ?o }",
                 slow_first: float = 0.0, reject_low_temperature: bool = False):
        self.reject_low_temperature = reject_low_temperature
        self.answer = answer
        self.slow_first = slow_first
        self.rate_limit_first = rate_limit_first
        self.reject_seed = reject_seed
        self.reject_system = reject_system
        self.bodies: list[dict] = []

    def handler(self):
        provider = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                provider.bodies.append(body)
                if provider.slow_first and len(provider.bodies) == 1:
                    import time as _t
                    _t.sleep(provider.slow_first)
                if len(provider.bodies) <= provider.rate_limit_first:
                    self._send(429, json.dumps({"error": {"message": "rate limited"}}).encode(),
                               {"retry-after-ms": "10"})
                elif provider.reject_system and any(
                        m["role"] == "system" for m in body["messages"]):
                    self._send(400, json.dumps({"error": {
                        "message": "Developer instruction is not enabled for models/gemma",
                        "type": "invalid_request_error"}}).encode())
                elif provider.reject_low_temperature and body.get("temperature", 1) < 1:
                    self._send(400, json.dumps({"error": {
                        "message": "temperature: values below 1 are not supported for this model",
                        "type": "invalid_request_error"}}).encode())
                elif provider.reject_seed and "seed" in body:
                    self._send(400, json.dumps({"error": {
                        "message": "seed is not supported",
                        "type": "invalid_request_error"}}).encode())
                else:
                    self._send(200, _completion(provider.answer))

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


def test_counts_tokens_from_provider_usage(serve):
    serve(FakeProvider())
    llm = make_llm("openai/gpt-oss-120b")
    llm("sys", "user")
    llm("sys", "user")
    assert llm.usage == {"prompt": 2, "completion": 2, "total": 4}


def test_runs_spread_over_several_days_are_aggregated(serve, tmp_path, monkeypatch):
    """A 200k tokens/day quota fits about one pass a day: --seed runs one pass,
    and summary.json always covers every run file present in --out."""
    import json as _json

    import evaluate
    serve(FakeProvider())
    base = ["evaluate.py", "--questions", str(ROOT / "kgqa/benchmark/questions.yaml"),
            "--ontology", str(ROOT / "ontology/alarm_ontology.ttl"),
            "--ids", "F01,S01", "--out", str(tmp_path)]
    for seed in ("0", "1"):                      # day 1, then day 2
        monkeypatch.setattr(sys, "argv", base + ["--seed", seed])
        evaluate.main()
    names = sorted(p.name for p in tmp_path.glob("*_run*.json"))
    assert names == ["orchestrator_run0.json", "orchestrator_run1.json",
                     "react_run0.json", "react_run1.json"]
    summary = _json.loads((tmp_path / "summary.json").read_text())
    assert [s["runs"] for s in summary] == [2, 2]
    assert all(s["mean_tokens"] > 0 for s in summary)

    (tmp_path / "summary.json").unlink()        # rebuild without calling a model
    monkeypatch.setattr(sys, "argv", base + ["--aggregate"])
    evaluate.main()
    assert _json.loads((tmp_path / "summary.json").read_text()) == summary


def test_model_without_system_role_gets_instructions_in_user_turn(serve):
    """Gemma on the Gemini API rejects the system role. The client merges the
    instructions into the user turn once, then keeps doing so for the run."""
    p = serve(FakeProvider(reject_system=True, reject_seed=True))
    llm = make_llm("gemma", seed=0)
    assert llm("Rules here.", "Question: x").startswith("SELECT")
    assert llm("Rules here.", "Question: y").startswith("SELECT")
    last = p.bodies[-1]
    assert [m["role"] for m in last["messages"]] == ["user"]
    assert last["messages"][0]["content"].startswith("Rules here.\n\nQuestion: y")
    assert "seed" not in last


def test_reasoning_block_in_the_answer_is_removed(serve):
    """Gemma on the Gemini API returns <thought>...</thought> inside the content.
    A draft query or OUT_OF_SCOPE inside it must not be taken as the answer."""
    final = "PREFIX net: <http://example.org/network#>\nSELECT ?a WHERE { ?a ?p ?o }"
    serve(FakeProvider(answer="<thought>Maybe OUT_OF_SCOPE?\nSELECT ?draft WHERE {}\n"
                              "No, it is answerable.</thought>" + final))
    out = make_llm("gemma")("sys", "user")
    assert out == final

    from agents import strip_fences
    assert strip_fences(out) == final


def test_timeout_is_retried_and_announced(serve, capsys):
    """A slow free-tier model must not look like a frozen terminal."""
    p = serve(FakeProvider(slow_first=1.5))
    out = make_llm("gemma", timeout=0.5, max_retries=2)("sys", "user")
    assert out.startswith("SELECT")
    assert len(p.bodies) == 2
    assert "timeout after 0s; retry 1/2" in capsys.readouterr().out


def test_provider_specific_fields_are_sent_from_the_environment(serve, monkeypatch):
    # OpenRouter: pin the serving provider and set the reasoning effort
    monkeypatch.setenv("LLM_EXTRA_BODY", '{"provider": {"order": ["groq"], "allow_fallbacks": false}, '
                                         '"reasoning": {"effort": "low"}}')
    p = serve(FakeProvider())
    make_llm("openai/gpt-oss-120b", seed=0)("sys", "user")
    body = p.bodies[0]
    assert body["provider"] == {"order": ["groq"], "allow_fallbacks": False}
    assert body["reasoning"] == {"effort": "low"} and body["seed"] == 0


def test_no_extra_fields_by_default(serve, monkeypatch):
    monkeypatch.delenv("LLM_EXTRA_BODY", raising=False)
    p = serve(FakeProvider())
    make_llm("openai/gpt-oss-120b")("sys", "user")
    assert "provider" not in p.bodies[0] and "reasoning" not in p.bodies[0]


def test_model_refusing_temperature_zero_runs_at_its_default(serve, capsys):
    # recent Claude models on the OpenAI-compatible endpoint
    p = serve(FakeProvider(reject_low_temperature=True))
    llm = make_llm("claude-haiku-5-5", seed=0)
    assert llm("sys", "user").startswith("SELECT")
    assert p.bodies[0]["temperature"] == 0 and "temperature" not in p.bodies[-1]
    assert p.bodies[-1]["seed"] == 0                  # other settings are kept
    llm("sys", "again")
    assert "temperature" not in p.bodies[-1]          # decided once, not per call
    assert "rejected temperature 0" in capsys.readouterr().out
