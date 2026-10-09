# kgqa: agent topologies for natural language to SPARQL

Does the **control flow** of an LLM agent change how well it translates
questions into SPARQL over a knowledge graph?

Recent text-to-SPARQL agents use different topologies: a ReAct loop (ARUQULA),
a hierarchical supervisor with specialised sub-agents (MetaboT), a
plan/action/feedback cycle (mKGQAgent). Each is evaluated on its own graph,
with its own model, against its own benchmark. The contribution of the topology
itself has never been isolated.

This module isolates it. Both topologies call the **same** `llm()`, the **same**
schema index and the **same** graph store, with the **same** prompts. Only the
control flow differs, so any gap in the numbers is attributable to the topology.

## The two topologies

| | ReAct | Hierarchical orchestrator |
|---|---|---|
| Control flow | one loop: generate, execute, retry on error | classify, link, generate, verify, execute, refine |
| Repair trigger | execution error | symbolic gate before execution, then execution error |
| Scope handling | may answer `OUT_OF_SCOPE` | dedicated classification stage, may also answer `OUT_OF_SCOPE` |

Refusal is available to **both** arms through the shared system prompt.
Otherwise the out-of-scope gap would be built into the design instead of
measured.

## The mechanism under test

An invented IRI does not make a SPARQL query fail. It makes it return nothing.

```sparql
SELECT ?a WHERE { ?a net:raisedOn net:FW1 . }   # both IRIs invented: 0 rows, no error
```

A loop that repairs only on execution errors accepts that empty answer as
final. The orchestrator's gate expands every prefixed name, compares it to the
IRIs that actually exist in the graph, and triggers a repair **before**
execution. `tests/test_topologies.py` reproduces both behaviours with a
scripted model; the benchmark measures how often it happens with a real one.

## Benchmark

32 questions on the DDoS incident graph (`ontology/alarm_ontology.ttl`):

| Bucket | n | What it tests |
|---|---|---|
| factual | 11 | lookups, counts, ordering, date filters |
| causal | 8 | transitive traversal of `net:causallyLinkedTo` |
| contrast | 3 | correlation and causation give **different** answers |
| counterfactual | 6 | what would not have happened if X were blocked |
| out_of_scope | 4 | must be refused |

The **contrast** bucket encodes the co-location trap: router R1 concentrates the
most alarms, yet the root cause sits on firewall FW1. A co-occurrence reading
picks the wrong device.

Every question carries its gold SPARQL **and** a hand-written expected answer.
`tests/test_benchmark.py` checks that each gold query returns exactly that
answer, since a broken gold query would score every prediction zero silently.

## Metrics

- `execution_accuracy`: answer set identical to the gold answer set
- `answer_f1`: set-level F1 over entities: an IRI and its `rdfs:label` count
  once, so projecting labels, or IRI and label together, is not penalised
- `query_validity_rate`: executed without error (refusals count as valid)
- `hallucinated_uri_rate`: share of questions whose final query references an
  IRI absent from the graph
- `mean_repairs`, `mean_llm_calls` (cost proxy), `mean_seconds`
- `refusal_accuracy` on out-of-scope questions, `false_refusal_rate` on the rest
- every metric reported as mean and standard deviation over `--runs`

## Run

From the repository root:

```bash
pip install -r kgqa/requirements.txt

# offline tests: no API key, no network (96 tests)
pytest kgqa/tests -q

# benchmark against any OpenAI-compatible endpoint (OpenAI, Groq, Mistral, vLLM, Ollama)
cp kgqa/.env.example kgqa/.env       # then put your key in kgqa/.env (git-ignored)
set -a; source kgqa/.env; set +a
python kgqa/src/evaluate.py --model openai/gpt-oss-120b --topology both --runs 3

# ablation: full ontology in the prompt instead of retrieved context
python kgqa/src/evaluate.py --topology both --runs 3 --no-schema-retrieval
```

### Running on a free tier

One pass over the 32 questions with both topologies is about 112 requests and
160k to 230k tokens, depending on how much the model reasons. Rate limits (429)
are retried with backoff, honouring the provider's retry-after header.

```bash
# smoke test on 5 questions, one per bucket
python kgqa/src/evaluate.py --model openai/gpt-oss-20b --ids F01,C04,K03,X02,S01 \
    --reasoning-effort low --out /tmp/kgqa-smoke
# full benchmark
python kgqa/src/evaluate.py --model openai/gpt-oss-120b --runs 3 --reasoning-effort low
```

`gpt-oss-120b` and `gpt-oss-20b` are open-weight models, so results can be
reproduced on self-hosted infrastructure; running both also gives a model-size
axis at no extra engineering cost.

### Apache Jena Fuseki

```bash
docker compose -f kgqa/docker-compose.yml up -d
bash kgqa/scripts/load_fuseki.sh               # creates the dataset, loads 102 triples
export SPARQL_ENDPOINT=http://localhost:3030/alarms/query
python kgqa/src/evaluate.py --topology both
```
Validated against Apache Jena Fuseki (stain/jena-fuseki): the dataset loads
102 triples and all 32 gold queries return the expected answers.

Without `SPARQL_ENDPOINT` the graph is held in memory by rdflib.
`tests/test_endpoint_mode.py` checks the endpoint path over HTTP against a
local server implementing the SPARQL 1.1 Protocol: all 32 gold queries return
the same answers as the in-memory graph.

## Layout

```
kgqa/
├── src/
│   ├── graph_store.py      rdflib in memory or Fuseki over HTTP, SELECT and ASK
│   ├── schema_context.py   schema, undeclared predicates and entity linking
│   ├── agents.py           the two topologies over shared primitives
│   └── evaluate.py         harness, metrics, repeated runs
├── benchmark/questions.yaml
├── tests/                  benchmark self-check, topologies, endpoint mode, LLM client
├── scripts/load_fuseki.sh
├── .env.example            provider settings template (copy to .env)
└── docker-compose.yml
```

## Status

The pipeline, the benchmark and the offline tests are complete. **No LLM
results are reported yet**: the topology comparison is the next step, and the
numbers will be added here when three runs per topology are done.

Known limits:
- the graph is small (102 triples); the benchmark validates the method, not
  scale
- entity linking is lexical; embedding retrieval is planned as an ablation
- strict execution accuracy penalises extra projected variables, which is why
  answer F1 is reported next to it
