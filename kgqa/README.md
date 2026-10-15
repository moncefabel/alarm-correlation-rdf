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

- `execution_accuracy`: the answer is correct. For entity answers, the
  best-matching projected column must equal the gold set, so an extra `?time`
  or `?count` column next to a correct `?alarm` column is not penalised. For
  yes/no questions (gold is ASK) the prediction is read as a boolean: an ASK,
  a yes/no or true/false value, a count (0 means no), or whether a SELECT
  returns anything.
- `strict_execution_accuracy`: classic execution match, the union of every
  projected value equals the gold set. Reported next to the lenient metric.
- `answer_f1`: set-level F1 on the same best-matching column. An IRI and its
  `rdfs:label` count once.
- `query_validity_rate`: executed without error (refusals count as valid)
- `hallucinated_uri_rate`: share of questions whose final query references an
  IRI absent from the graph
- `mean_repairs`, `mean_llm_calls`, `mean_tokens` and `total_tokens` (cost),
  `mean_seconds`
- `refusal_accuracy` on out-of-scope questions, `false_refusal_rate` on the rest
- every metric reported as mean and standard deviation over `--runs`

### Pilot run and benchmark revision

A first full pass (gpt-oss-120b, one run) was used as a pilot and is not
reported. Inspecting every failed question showed three defects shared by both
topologies, which were fixed before the reported runs:

1. Value linking: the context did not show literal values, so "critical
   severity" was queried as `"critical"` while the data says `"CRITICAL"`.
   Properties with a small closed set of string values now list them.
2. Form over content: correct answers with an extra projected column, and
   yes/no questions answered with a SELECT or a count, were scored zero. The
   lenient metric above fixes this; the strict one is kept alongside.
3. X06 had inverted polarity ("would the timeout still happen"); it now asks
   "would the timeout have been avoided".

The scoring rules are fixed in `tests/test_scoring.py`, which replays the pilot
queries verbatim, including wrong answers that must stay wrong.

## Run

From the repository root:

```bash
pip install -r kgqa/requirements.txt

# offline tests: no API key, no network (201 tests)
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

When the daily token quota fits only one pass, spread the runs over several
days. `--seed` runs one pass and `summary.json` always covers every run file
already in `--out`; `--aggregate` rebuilds it without calling a model.

```bash
python kgqa/src/evaluate.py --model openai/gpt-oss-120b --seed 0 --reasoning-effort low   # day 1
python kgqa/src/evaluate.py --model openai/gpt-oss-120b --seed 1 --reasoning-effort low   # day 2
python kgqa/src/evaluate.py --aggregate
```

Every scored question is checkpointed in `<topology>_run<seed>.partial.jsonl`.
If a run stops (provider outage, exhausted daily quota, Ctrl+C), rerunning the
same command resumes from the next question; the file is removed once the
topology completes.

Tokens are recorded per question from the provider's usage field
(`mean_tokens`, `total_tokens`), so cost per answered question is reported
next to accuracy.

Gemma (open weights) on the Gemini API free tier also works through
`OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`. It is
served without a system role; the client then prepends the instructions to the
user turn, identically for both topologies. Gemma also returns its reasoning
inside the answer (`<thought>...</thought>`); reasoning blocks are removed
before the query is extracted, for every model. Every retry (timeout, rate
limit) is printed with its wait, and `--timeout` sets the per-answer limit.

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

Without `SPARQL_ENDPOINT` the graph is held in memory by rdflib. rdflib
rejects some valid SPARQL that Fuseki runs (for instance `IF(EXISTS { ... })`
with a property path), which would count a correct query as invalid. Reported
runs therefore use Fuseki.
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
│   ├── evaluate.py         harness, metrics, repeated runs
│   ├── domains.py          what the prompts say the graph is about (alarms, support)
│   ├── explain.py          evidence triples of an answer and their sources
│   ├── semantic_check.py   domain/range check of each triple, empty-answer hint
│   └── direction.py        audit: causal property traversed the wrong way round
├── benchmark/questions.yaml
├── tests/                  benchmark self-check, topologies, endpoint mode, LLM client,
│                           audit, domains, OWL variant, explanations
├── scripts/load_fuseki.sh
├── scripts/direction_audit.py
├── scripts/retrieval_recall.py   gold IRIs that reach the prompt, no model called
├── scripts/make_owl_variant.py   alarm graph with causedBy and symmetric correlatedWith
├── scripts/explain_answer.py     why an answer is an answer, with sources
├── .env.example            provider settings template (copy to .env)
└── docker-compose.yml
```

## Results

gpt-oss-120b (Groq, `reasoning_effort=low`, temperature 0), Apache Jena Fuseki,
benchmark v2, 3 runs per topology. Mean and standard deviation over runs.

| | ReAct | Orchestrator |
|---|---|---|
| Execution accuracy | 0.750 ± 0.031 | **0.885 ± 0.018** |
| Strict execution accuracy | 0.479 ± 0.036 | **0.635 ± 0.036** |
| Answer F1 | 0.819 ± 0.034 | **0.923 ± 0.028** |
| Query validity | 1.00 | 1.00 |
| Hallucinated-IRI rate | 0.00 | 0.00 |
| Refusal accuracy / false refusals | 1.00 / 0.00 | 1.00 / 0.00 |
| LLM calls per question | 1.01 | 1.92 |
| Tokens per question | 1 546 | 1 601 (+3.6 %) |

Execution accuracy by question type (mean over the 3 runs):

| Bucket | n | ReAct | Orchestrator |
|---|---|---|---|
| factual | 11 | 1.00 | 1.00 |
| causal | 8 | 0.71 | 0.79 |
| contrast | 3 | 0.44 | 0.89 |
| counterfactual | 6 | 0.33 | 0.72 |
| out_of_scope | 4 | 1.00 | 1.00 |

Raw per-question results: `results/react_run{0,1,2}.json`,
`results/orchestrator_run{0,1,2}.json`; aggregate in `results/summary.json`.

### Reading

- The whole gap is on multi-hop questions. On factual lookups and
  out-of-scope refusals the two topologies are identical.
- Averaged over runs, the orchestrator scores higher on 8 questions, lower on
  1 (C03) and the same on 23. An exact sign test on the 9 non-tied questions
  gives p ≈ 0.04: consistent, but measured on 32 questions.
- The typical ReAct failure is stopping at the first causal hop (C04, X01:
  direct `causallyLinkedTo` where the chain needs `causallyLinkedTo+`).
- The extra cost is small: the classification call is about 150 tokens, so the
  orchestrator spends 3.6 % more tokens for almost twice the calls.
- The symbolic gate never fired: no invented IRI in any run, for either
  topology, once entities and literal values are in the context. The
  advantage therefore does not come from verification but from the
  classification stage and the type-specific instruction it adds. The
  ablation below separates the two: the instruction carries the gain.
- Latency is not reported: on the free tier, rate-limit waits dominate wall
  time and fall unevenly on the two topologies.

### Ablation: instruction or control flow?

The orchestrator differs from ReAct in two ways at once: its control flow
(classification, symbolic gate, refine) and a type-specific instruction added
to the generation prompt ("This is a ROOT-CAUSE question. Traverse causal
properties transitively..."). Since the gate never fired, the gap could come
from the instruction alone. Two arms separate the causes:

| Arm | Control flow | Type-specific instruction |
|---|---|---|
| `react` | ReAct | no |
| `react_typed` | ReAct, plus the same classification call | **yes**, verbatim |
| `orchestrator_untyped` | orchestrator | no |
| `orchestrator` | orchestrator | yes |

In `react_typed` the class only selects the instruction; it never routes, so
ReAct keeps its own refusal path. Prompts of the original two arms are
unchanged character for character (checked on all 32 questions), so the
existing runs stay comparable.

```bash
python kgqa/src/evaluate.py --model openai/gpt-oss-120b --topology ablation --seed 0 --reasoning-effort low
```

**Result** (same model, settings and benchmark, 3 runs per arm):

| Arm | Execution accuracy | Strict | Answer F1 | Tokens / question |
|---|---|---|---|---|
| `react` | 0.750 ± 0.031 | 0.479 | 0.819 | 1 546 |
| `orchestrator_untyped` | 0.750 ± 0.031 | 0.500 | 0.807 | 1 556 |
| `react_typed` | 0.823 ± 0.036 | 0.542 | 0.862 | 1 767 |
| `orchestrator` | **0.885 ± 0.018** | **0.635** | **0.923** | 1 601 |

| Bucket | n | react | orchestrator_untyped | react_typed | orchestrator |
|---|---|---|---|---|---|
| factual | 11 | 1.00 | 1.00 | 1.00 | 1.00 |
| causal | 8 | 0.71 | 0.67 | 0.71 | 0.79 |
| contrast | 3 | 0.44 | 0.44 | 0.56 | 0.89 |
| counterfactual | 6 | 0.33 | 0.39 | 0.67 | 0.72 |
| out_of_scope | 4 | 1.00 | 1.00 | 1.00 | 1.00 |

Reading:

- **The control flow alone adds nothing.** Without the instruction, the
  orchestrator scores exactly what ReAct scores (0.750), bucket by bucket
  within 0.06.
- **The instruction carries the gain.** It adds 0.073 to ReAct and 0.135 to
  the orchestrator, on the questions it targets: C04, X01, X02 and X03 need a
  transitive traversal, and both instructed arms find it where neither
  uninstructed arm does. Per question, `react_typed` beats `react` on 5 and
  loses on 2 (sign test p ≈ 0.45); `orchestrator` beats `orchestrator_untyped`
  on 6 and loses on 1 (p ≈ 0.13). Consistent in direction, not significant on
  32 questions.
- **The 0.063 left between `react_typed` and `orchestrator` is not evidence for
  the control flow.** For questions classified causal or counterfactual the
  two arms send the same generation call, byte for byte (same system prompt,
  same user message, same instruction; for other classes they differ by two
  blank lines). The gate never fired and repairs are rare (0.03 and 0.01 per
  question), so the routing has almost nothing to act on. The differences sit
  on C02, K02 and X01, which change outcome between arms with identical
  prompts. The arms ran in different sessions, days apart: at temperature 0
  with fixed seeds, variation between sessions is as large as this residual.
  The next comparison runs all arms interleaved in one session.

**Where the remaining errors come from.** `kgqa/scripts/direction_audit.py`
compares the side of `net:causallyLinkedTo` on which each named alarm appears
in the predicted query and in the gold query (`kgqa/src/direction.py`):

| Arm (3 runs) | Wrong answers | Direction inverted | Where |
|---|---|---|---|
| `react` | 24 | 1 | C03 |
| `orchestrator_untyped` | 24 | 5 | C02 x2, C03 x3 |
| `react_typed` | 17 | 6 | C02 x2, C03 x3, X01 |
| `orchestrator` | 11 | 5 | C02 x2, C03 x3 |

Every inverted query is a wrong answer, and inversions are almost half of the
orchestrator's errors. Plain ReAct inverts once in 96 answers; each of the
three other arms inverts 5 or 6 times, with or without the instruction. The query is valid and uses only known IRIs, so neither
the syntax check nor the symbolic gate can catch it; it returns zero rows
without an error, and no repair is attempted. C03 ("what directly caused the
latency spike?") is the clearest case: plain ReAct answers it in 2 runs of 3,
every other arm in 0 of 3, including `orchestrator_untyped`, whose generation
prompt differs from ReAct's only by two blank lines. The inversion is
therefore not caused by the instruction. Across C02 and C03 together, ReAct
inverts in 1 of 6 answers and `orchestrator_untyped` in 5 of 6, with
generation prompts two blank lines apart. On two questions and three runs this
is suggestive, not conclusive, but it points at prompt brittleness rather than
reasoning: a layout change that no one would review moves the model from one
reading of the property to the other.

K02 fails in 10 of 12 runs across the arms for a different, schema-level
reason: `correlatedWith` is stored in one direction only
(`Alarm_Latency correlatedWith Alarm_PacketDrop`), and a query that looks the
other way finds nothing. The gold query needs a UNION over both directions.

Three points follow.

- The direction of `causallyLinkedTo` is stated only in its `rdfs:comment`;
  the name does not encode it (`causes` / `causedBy` would). Ontology naming
  is part of what an LLM agent reads.
- Two OWL axioms target the remaining errors directly: an inverse property
  (`causedBy owl:inverseOf causallyLinkedTo`) for C02 and C03, and
  `correlatedWith a owl:SymmetricProperty` for K02. Testing them, with the
  inferred triples materialised, is the next experiment (below).
- An empty answer to a question that presupposes one ("what caused...") is a
  cheap signal for a repair, and a direction check like the audit could run
  before execution. That is semantic verification, beyond the symbolic gate.

### Next experiment: the two OWL axioms

`scripts/make_owl_variant.py` writes `ontology/alarm_ontology_owl.ttl`: the
same graph plus `net:causedBy owl:inverseOf net:causallyLinkedTo` and
`net:correlatedWith a owl:SymmetricProperty`, with the 5 entailed triples
materialised. Every gold answer is unchanged on it
(`tests/test_owl_variant.py`), so the two graphs can be compared question by
question. Prediction, written before the run: C03 and K02 improve on every
arm, and the direction audit counts fewer inversions; if the inversions only
move to `causedBy`, naming was not the cause.

```bash
python kgqa/scripts/make_owl_variant.py
python kgqa/src/evaluate.py --model openai/gpt-oss-120b --reasoning-effort low \
    --ontology ontology/alarm_ontology_owl.ttl --topology both --seed 0 \
    --out kgqa/results_owl
python kgqa/scripts/direction_audit.py --out kgqa/results_owl
```

### Scoring and arm added after the first SAP results

Reading the queries behind the wrong answers of the first Claude Haiku runs
(`scripts/failure_report.py`) found two things, both added after seeing the
results and so reported next to the original numbers, never instead of them:

- `src/answer_form.py` and `scripts/rescore.py`: an agent that returns the
  names of the gold entities (PROPERTY, VALUE) instead of the entities is
  right, and the first scorer called it wrong. The rescorer reads the saved
  traces and scores them with and without the answer-form rule; no model call.
- `orchestrator_checked_zero`: a lone `COUNT` of 0 counts as an empty answer
  for the semantic check's repair. Exploratory; see `supportkg/README.md`.

### Next experiment: a semantic check

The OWL variant removes the need to choose a direction; a check can instead
catch the wrong choice. `orchestrator_checked` (`--topology checked` runs it
next to `orchestrator`) adds two steps to the orchestrator, everything else
unchanged:

- before execution, every triple is checked against the `rdfs:domain` and
  `rdfs:range` of its property (`src/semantic_check.py`; for an inverse
  property they are read off its inverse). A reversed triple goes back to the
  repair step with the fix, e.g. "sup:View_M_PASSWORD_POLICY sup:columnOf ?c
  reads the property the wrong way round: sup:columnOf goes from a
  sup:Column to a sup:SystemView. Fix: swap subject and object, or use
  sup:hasColumn". This is the ontology-based query check of Allemang and
  Sequeda (2024), whose "incorrect path" is our direction error;
- after execution, an empty answer to an in-scope question goes back once with
  a hint about direction. On the alarm graph this is the only signal: cause and
  effect are both `net:Alarm`, so no type check can tell them apart.

A wrong check cannot cost an answer: on the last attempt a query that only
fails the type check is executed anyway. No gold query of either benchmark is
flagged. Limit: neither benchmark has an in-scope question whose true answer is
empty, so the empty-answer repair is never penalised here; a real deployment
needs such questions in its test set.

```bash
python kgqa/src/evaluate.py --model openai/gpt-oss-120b --reasoning-effort low \
    --topology orchestrator_checked --seed 0
python kgqa/scripts/direction_audit.py
```

### Other graphs

The prompts' description of the graph is a domain profile
(`src/domains.py`): `--domain alarms` (default, byte-identical to the runs
above, checked by a hash of every prompt in `tests/test_domains.py`) or
`--domain support` for the SAP support graph of [`supportkg/`](../supportkg/),
with its own benchmark and question types. See
[`supportkg/README.md`](../supportkg/README.md#question-answering-over-the-graph).

### Limits

- one model; a second model family is needed before generalising
- small graph (102 triples) and 32 questions: the benchmark validates the
  method, not scale
- direction inversions (C02, C03) and a one-way symmetric relation (K02)
  cause most remaining errors (see above)
- arms ran in different sessions; session-to-session variation is of the
  order of the smallest differences reported
- entity linking is lexical; on the 7 000-entity support graph the lexical
  retrieval misses 39 % of the IRIs the gold queries need, and a schema-first
  retrieval with entity names from `owl:hasKey` replaces it there
  (`supportkg/README.md`)
