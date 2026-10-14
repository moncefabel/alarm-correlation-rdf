# supportkg: an SAP support knowledge graph from open and public sources

A small version of the build side of an SAP support knowledge graph: combine
several sources into one RDF graph with a formal ontology, link them across
sources, keep the provenance of every fact, validate the result. The query
side is [`kgqa/`](../kgqa/), whose agents read this graph unchanged.

```
SAP HANA Cloud SQL reference (CC-BY-4.0, git clone) ─► docs.py ─┐
knowledge-article titles (public listings) ──────────► extract ─┤
pages saved by hand (optional) ──────────────────────► extract ─┤
PPMS-shaped stand-in, tools list ───────────────────────────────┤
support.ttl (OWL) ──────────────────────────────────────────────┴─► build.py
      entity resolution · alert → view linker (lexical | Hugging Face)
      OWL inverse/symmetric materialisation · PROV-O provenance · SHACL
                                         │
                              out/graph.ttl ──► Fuseki ──► kgqa agents
```

## Sources

| Source | Gives | Licence and access |
|---|---|---|
| [SAP HANA Cloud, SAP HANA Database documentation](https://github.com/SAP-docs/sap-hana-cloud-hana-database), SQL reference | 516 system and monitoring views (description, 6 490 columns with types, related pages) and 182 SQL statements with their category | CC-BY-4.0, © SAP SE; cloned with git, nothing scraped. Each node keeps the page URL at the cloned commit |
| Knowledge-article titles (`seeds/articles.txt`) | number and title of 39 SAP Notes and KBAs, 32 of them the "How to handle HANA Alert N" series: which alert each addresses, and the alert's name | titles as listed publicly by search engines |
| Preview pages saved by hand (optional) | symptom, product, keywords (component codes) for the articles you save | a person reading public pages in a browser; kept local |
| PPMS-shaped stand-in (`ontology/ppms_standin.ttl`) | ProductLine, Product, ProductVersion, SoftwareComponent, SupportPackage for SAP HANA | hand-written; **not PPMS data**, PPMS is internal |
| Tools list (`ontology/tools.ttl`) | HANA studio, cockpit, DBACOCKPIT, hdbcons... | hand-curated |

Why not download the article previews automatically: SAP's support portal
robots.txt disallows every crawler except named search engines (checked
2026-10-14). `collect.py --mode web` reads robots.txt first and, for this
site, collects nothing. That is the intended behaviour.

## The cross-source link

No source says which monitoring view to query when an alert fires: alerts
come from the knowledge articles, views from the documentation. `build.py`
predicts `sup:investigatedWith` (alert → view) and records how:

- **lexical** baseline: the alert name's words found in the view's name and
  description, weighted by rarity (IDF), name matches counted double;
- **embedding**: cosine similarity of sentence embeddings
  (`sentence-transformers/all-MiniLM-L6-v2`, Hugging Face, PyTorch).

Both keep the top 2 views above a threshold fixed before any embedding result
was seen (lexical 0.35, embedding 0.5). They are scored against
`gold/alert_views.yaml`: 32 alerts, 43 (alert, view) pairs, drafted from SAP
HANA administration practice and checked by the author against the view
descriptions in the documentation.

| Method | Precision | Recall | F1 | hit@2 |
|---|---|---|---|---|
| lexical (IDF-weighted word overlap) | **0.469** | **0.349** | **0.400** | **0.438** |
| embedding (`all-MiniLM-L6-v2`) | 0.091 | 0.116 | 0.102 | 0.156 |

hit@2: share of alerts with at least one correct view among the two proposed.

**The general-purpose embedding model loses to the word-overlap baseline by a
factor of four.** The errors show why:

- **The query template matches the wrong thing.** Alerts are embedded as
  "SAP HANA alert: <name>". The view whose description is closest to that
  phrase is `M_TIMEZONE_ALERTS` ("alerts relating to the status of SAP HANA
  internal timezone conversion"), proposed first for 7 of 32 alerts, from
  savepoint duration to automatic log backup. Views whose description says
  "SAP HANA" (`M_SHARED_MEMORY`, `M_TRACEFILE_CONTENTS`) are pulled up the
  same way. The words shared by every query dominate the similarity.
- **Domain abbreviations are opaque to it.** `M_CS_TABLES` (CS: column store)
  is the view for four alerts about column-store tables; its description
  says only "runtime data for column tables".
- **Domain jargon is rare in its training data**: savepoint, delta storage,
  mergedog, pagedump.

The lexical baseline is immune to the first two: rare words dominate its
score, and a word in the view's name counts double. The general lesson for
enterprise knowledge graphs: off-the-shelf sentence embeddings are not a safe
default for linking technical identifiers, and a strong lexical baseline has
to be reported next to them.

Next experiments, reported separately as post-hoc since they are designed after
seeing these errors: the query without the template, a hybrid of the two
scores, a domain-adapted or larger encoder, and LLM re-ranking of the lexical
candidates.

The lexical baseline finds links whose words match (`Disk usage` →
`M_DISK_USAGE`) and misses paraphrases (`Number of diagnosis files` →
`M_TRACEFILES`), which is what a better semantic method has to close.

## Ontology

`ontology/support.ttl` (OWL 2), aligned with the PPMS vocabulary instead of
redefining it. It builds in the two lessons of the kgqa error analysis:

- **every directed relation has a named inverse**: `addresses` /
  `addressedBy`, `investigatedWith` / `helpsInvestigate`, `hasColumn` /
  `columnOf`, `assignedTo` / `hasArticle`, `appliesTo` /
  `hasApplicableArticle`, `references` / `referencedBy`, `subComponentOf` /
  `hasSubComponent`;
- **undirected relations are symmetric**: `cooccursWith` between alerts,
  `seeAlso` between documentation pages.

Also: class hierarchy MonitoringView ⊑ SystemView ⊑ DocumentationTopic,
disjointness between kinds of thing, keys (`owl:hasKey`) on article number,
alert number, component code and view name, and a source node carrying the
documentation's CC-BY-4.0 licence (`dct:license`). `ontology/shapes.ttl` holds
the SHACL shapes the graph must pass.

## Extraction from articles

| Relation | Method |
|---|---|
| `addresses` | title rule: "How to handle HANA Alert 54: ..." and its variants |
| `mentionsAlert` | regex in title and symptom ("Alert ID 39", "alerts 1, 2 and 43"); embedding linker for articles that never give a number |
| `cooccursWith` | alerts reported in the same article |
| `assignedTo`, `subComponentOf` | component codes in the keywords; hierarchy from the code (HAN-DB-MON → HAN-DB → HAN) |
| `appliesTo` | Product section strings linked to the PPMS stand-in (gazetteer, then embeddings) |
| `references`, `usesTool` | cited article numbers; tool names (spaCy PhraseMatcher) |

The last four need saved pages: titles alone give `addresses`,
`mentionsAlert` and the alert names. spaCy runs as a blank English pipeline
(no model download) with a custom tokenizer rule and an EntityRuler built from
the ontology's own labels.

Every extracted or predicted triple gets a statement node in
`out/provenance.ttl`: the page(s) it came from (`prov:wasDerivedFrom`), the
method and a confidence. A predicted alert → view link cites both the
article that names the alert and the documentation page of the view.

## Run

```bash
pip install -r supportkg/requirements.txt
git clone --depth 1 https://github.com/SAP-docs/sap-hana-cloud-hana-database \
    supportkg/data/sap-hana-cloud-hana-database
python supportkg/src/collect.py                     # article records from the seed titles
python supportkg/src/build.py                       # -> supportkg/out/graph.ttl, provenance.ttl, inferred.ttl, report.json
python supportkg/src/evaluate_links.py --show       # lexical alert -> view links against the gold list

pip install sentence-transformers                   # optional: Hugging Face models (PyTorch)
python supportkg/src/evaluate_links.py --method both --show
python supportkg/src/build.py --view-linker hf      # graph with embedding links
python supportkg/src/build.py --no-materialise --out supportkg/out_noinv   # OWL experiment

# load into Fuseki next to the alarm graph
docker compose -f kgqa/docker-compose.yml up -d
DATASET=support bash kgqa/scripts/load_fuseki.sh supportkg/out/graph.ttl
```

With pages saved by hand in `supportkg/data/saved/`:
`python supportkg/src/collect.py --mode saved`, then `build.py` again, and
`evaluate_extraction.py --template 25` for the annotation form of the
article-level extraction.

Tests (offline, no SAP page, no model download; one test also checks the gold
views against the real documentation when the clone is present; the benchmark
tests read the two committed graphs):

```bash
pytest supportkg/tests -q
```

## Question answering over the graph

The kgqa agents run on this graph unchanged, with a domain profile that tells
them what the graph holds (`kgqa/src/domains.py`, `--domain support`).

**Benchmark** (`benchmark/questions.yaml`): 32 questions, each with a gold
SPARQL query and an answer written by hand, in six buckets: lookup, inverse,
multi-hop across sources (article → alert → view → column or documentation
link; PPMS version → component → support package), aggregation, negation,
out of scope. `tests/test_support_benchmark.py` runs every gold query on both
graphs of the OWL experiment below and checks the answer.

**Schema retrieval at this scale.** The graph has 7 278 entities, 6 490 of
them columns. The alarm graph's retrieval (top 25 schema items and entities
by shared words) brings only 61 % of the IRIs the gold queries need into the
prompt here, and every one of the 29 in-scope questions needs all of them:
"columns" and "view" are words shared by thousands of column labels, which
push out the properties. The support profile therefore uses a schema-first
retrieval: every class and property (the T-box is small), then the entities
the question names, found by their label, alternative labels and the values
of the keys the ontology declares (`owl:hasKey`: article number, alert
number, view name), then by rare shared words. Measured without calling any
model (`kgqa/scripts/retrieval_recall.py`):

| Retrieval | Gold IRIs in the prompt | Questions with all of them | Prompt size |
|---|---|---|---|
| lexical top 25 (alarm default) | 0.607 | 11 / 29 | 4 600 chars |
| schema + entities by key | **1.000** | **29 / 29** | 11 400 chars |

On the alarm graph both reach 1.000, which is why the problem did not show
there. The cost is a prompt about 2.5 times longer.

**The OWL experiment.** Does materialising the ontology's inverse and
symmetric axioms help an LLM agent? Two graphs, same data:

- `out/graph.ttl`: inverses and symmetric links materialised (53 358 triples);
- `out_noinv/graph.ttl`: built with `--no-materialise` (45 885 triples). Each
  fact exists only in the direction it was extracted: `columnOf` but no
  `hasColumn`, `addresses` but no `addressedBy`, `seeAlso` one way.

The ontology, and so the prompt's schema, is the same in both: the agent is
told `hasColumn` exists in both cases, and in the second graph it returns
nothing. The `inverse` bucket asks questions against the extracted direction
("which columns does M_PASSWORD_POLICY have?"). Prediction, written before the
run: the inverse bucket drops on the no-inverse graph, the other buckets do
not move, and the checked arm recovers most of the drop.

```bash
set -a; . ~/.config/claude.env; set +a          # or groq.env, see kgqa/.env.example
unset SPARQL_ENDPOINT                            # both graphs run in memory
python kgqa/src/evaluate.py --domain support --model claude-haiku-5-5 \
    --topology both --runs 3                                  # -> kgqa/results_support
python kgqa/src/evaluate.py --domain support --model claude-haiku-5-5 \
    --topology both --runs 3 --ontology supportkg/out_noinv/graph.ttl \
    --out kgqa/results_support_noinv
# the third arm: the orchestrator with the semantic check, on both graphs
python kgqa/src/evaluate.py --domain support --model claude-haiku-5-5 \
    --topology checked --runs 3 --out kgqa/results_support_checked
python kgqa/src/evaluate.py --domain support --model claude-haiku-5-5 \
    --topology checked --runs 3 --ontology supportkg/out_noinv/graph.ttl \
    --out kgqa/results_support_noinv_checked
# what the agents wrote for the questions they got wrong
python kgqa/scripts/failure_report.py kgqa/results_support --ids I01,M02,M03
```

Two ways to the same goal, compared on the same questions: materialise the
inverses so either direction works, or keep the graph lean and let the
semantic check (`kgqa/src/semantic_check.py`) send a reversed triple back with
its fix. On this graph the domains and ranges differ, so the type check sees
every reversal.

One run of one arm is about 200 000 tokens (32 questions, about 6 400 each,
the schema is in every prompt): about one cent with Claude Haiku 5.5, so the
whole experiment below cost under 25 cents. Haiku 5.5 refuses the
`temperature` setting, so the runs are not deterministic (the evaluation
prints a notice and uses the provider's default); that is why each arm ran
three times.

### Results (Claude Haiku 5.5, 3 runs, n = 32 questions)

Execution accuracy, lenient, averaged over runs. One question is 3.1 points.
Haiku 5.5 does not accept `temperature: 0`, so runs are not deterministic: the
same configuration, run in two batches of three, differs by up to 4 points
(orchestrator on the graph without inverses: 0.781, 0.823, 0.802). Runs from
different batches (same code, same model) are pooled; the plain orchestrator
was run in three batches (9 runs), the checked arms in two (6 runs), react
once (3 runs).

With the original scorer (the rule written before the runs), and with the
answer-form scorer (`kgqa/src/answer_form.py`, written after the failure
analysis below: a name counts for the gold entity it names; applied to every
arm, no model call, `kgqa/scripts/rescore.py`):

| arm | runs | with inverses: original / answer-form | without inverses: original / answer-form |
|---|---|---|---|
| react | 3 | 0.896 / 0.969 | 0.802 / 0.823 |
| orchestrator | 9 | 0.889 / 0.969 | 0.802 / 0.802 |
| orchestrator_checked | 6 | 0.875 / 0.969 | 0.833 / 0.932 |
| orchestrator_checked_zero (exploratory) | 6 | not run | 0.854 / 0.922 |

The answer-form scorer helps the checked arms most because the plain agents on
the graph without inverses fail by returning nothing, which no scorer can
credit, while the checked arms repair the query and then return correct
answers as names.

What the numbers say, and what they do not:

- **Materialising the inverses helps, by about 17 points** with the
  answer-form scorer (0.97 against 0.80) and about 9 with the original. The
  direction is the same in every batch.
- **A query-time semantic check recovers about three quarters of that on the
  lean graph:** 0.93 against 0.80 for the plain orchestrator (answer-form
  scorer), out of the 0.17 gap. On the graph with inverses it adds nothing
  (0.969 for both). The original scorer hides this, because the repaired
  answers come back as names. The checked arms cost about 16 % more tokens
  (7 300 against 6 300 per question).
- **The zero-count repair did not help once replicated.** In the first batch
  it looked better (0.938 against 0.917); in the second it was worse (0.906
  against 0.948). Pooled it is 0.92 against 0.93: no difference. The
  mechanism works (the unit test, and A01 and A03 passing in the first batch),
  but the check without it already gets there on most runs. Reported as a
  negative result.
- **The prediction written above was half right.** The checked arm does recover
  most of the loss. It was wrong that the `inverse` bucket would drop (0.89 on
  both graphs, react arm, original scorer): the loss on the lean graph is in
  lookup, aggregation and negation questions, for the reasons below.
- **The orchestrator does not beat react here.** It adds one call per question
  and no accuracy.
- **Strict accuracy is about 0.40 and lenient about 0.90.** Strict requires
  every projected value to equal the gold set; Haiku adds a name or label
  column next to the entity, which the lenient rule (best column) does not
  penalise. Both are reported.
- **Safety numbers are clean:** every query valid, no invented URI, refusal
  accuracy 1.0 on the three out-of-scope questions, at most one false refusal
  in 96 answers, and every answer came with its evidence triples
  (`explained_correct_rate` 1.0, one run 0.987).

Limits that matter: one model, one 32-question benchmark. The answer-form
scorer and the zero-count repair were both written after seeing the failures
on this same benchmark, so the numbers they produce are in-sample. They show
what the failure analysis found; a held-out question set would be needed to
claim they generalise. The original-scorer column is kept so nothing is lost
by the change.

Share of exact answers by question bucket, react arm, original scorer
(L lookup, I inverse, M multihop, A aggregation, N negation, O out of scope):

| graph | L | I | M | A | N | O |
|---|---|---|---|---|---|---|
| with inverses | 1.00 | 0.89 | 0.67 | 1.00 | 1.00 | 1.00 |
| without inverses | 0.83 | 0.89 | 0.75 | 0.67 | 0.75 | 1.00 |

#### Why the answers fail

Read from the saved queries (`kgqa/scripts/failure_report.py`). Four different
causes, only one of which is a plain model error.

1. **The answer is right, the form is not (I01, M02).** The agent returns the
   column *names* (PROPERTY, VALUE, UNLOAD_TIME) where the gold answer is the
   column *entities*. The original scorer maps an `rdfs:label` to its entity,
   but the label is "PROPERTY column of M_PASSWORD_POLICY" and the bare name
   PROPERTY is shared by many columns. A benchmark artefact worth 2 of 32
   questions in every run, fixed by the answer-form scorer.
2. **The data does not hold what the ontology promises (L06, A01, A03 without
   inverses).** The agent reads the ontology, correctly uses `hasColumn`, and
   gets nothing back, because on the lean graph only `columnOf` has triples.
   A COUNT over nothing returns one row, "0", so the empty-answer repair never
   fired on A01. `orchestrator_checked_zero` treats a lone COUNT of 0 as an
   empty answer: A01 and A03 then passed in all three runs of the first batch. Over six runs it makes no measurable difference to the total.
3. **A symmetric property that is not materialised (N04 without inverses).**
   `seeAlso` is symmetric in the ontology but stored in one direction. The
   agent trusts the ontology and checks one direction only, so it returns
   SET_TRANSACTION_AUTOCOMMIT_DDL as having no link when the link exists the
   other way (F1 0.8). This is the case materialisation exists for, and the
   check does not catch it: the wrong answer is not empty.
4. **A real ontology defect (M03).** `seeAlso` is declared with domain and
   range `DocumentationTopic`, but the data links views and statements. The
   agent honours the declaration, adds `?page a sup:DocumentationTopic`, and
   gets nothing; one run wrote a comment noting the gap. The fix is in the
   ontology (widen the domain and range, or make views and statements
   subclasses of `DocumentationTopic`), not in the agent. Not fixed: it changes
   the graph, so it needs a rebuild and fresh runs. With inverses, it is the
   one systematic error left.

```bash
python kgqa/scripts/rescore.py kgqa/results_support
python kgqa/scripts/rescore.py kgqa/results_support_noinv --graph supportkg/out_noinv/graph.ttl
python kgqa/scripts/rescore.py kgqa/results_support_checked
python kgqa/scripts/rescore.py kgqa/results_support_noinv_checked --graph supportkg/out_noinv/graph.ttl
python kgqa/scripts/rescore.py kgqa/results_support_noinv_zero --graph supportkg/out_noinv/graph.ttl
# the replication batch (checked and zero arms on the graph without inverses)
python kgqa/scripts/rescore.py kgqa/results_support_rep
python kgqa/scripts/rescore.py kgqa/results_support_noinv_rep --graph supportkg/out_noinv/graph.ttl
python kgqa/src/evaluate.py --domain support --model claude-haiku-5-5 \
    --topology orchestrator_checked_zero --runs 3 --ontology supportkg/out_noinv/graph.ttl \
    --out kgqa/results_support_noinv_zero
```

Not shown: other models, a held-out benchmark, a baseline without the
knowledge graph (a vector-RAG baseline is the first thing to add).

**Explainable answers** (`kgqa/src/explain.py`). Every answer the agents
return on this graph is stored with the triples that justify it and where each
one comes from. The query's WHERE clause is run again with every variable
bound; the triples its patterns matched are the evidence (patterns under
FILTER NOT EXISTS are left out: an absence has no source; a `p+` path is
expanded link by link). Each evidence triple is then traced:

| Origin | Shown as |
|---|---|
| extracted from an article, or predicted by the linker | page(s), method, confidence (`provenance.ttl`) |
| added by an OWL axiom (`out/inferred.ttl`, written by the build) | the axiom, then the triple it was inferred from, traced in turn |
| stated on a documentation page or in an article title | that page |
| PPMS stand-in | "hand-written", since PPMS is internal |

```
$ python kgqa/scripts/explain_answer.py --id M01
Question: Which view should be queried to investigate the alert addressed by knowledge article 1977207?
Because:
  Article_1977207 addresses Alert_55  [title-rule, confidence 1.0]
      https://userapps.support.sap.com/sap/support/knowledge/en/1977207
  Alert_55 investigatedWith View_M_CS_UNLOADS  [lexical-linker, confidence 0.457]
      https://github.com/SAP-docs/.../m-cs-unloads-system-view-20ad86c.md
      https://userapps.support.sap.com/sap/support/knowledge/en/1977207
```

An inferred link names its axiom: `View_M_PASSWORD_POLICY hasColumn ...
[inferred: hasColumn owl:inverseOf columnOf]`, from the `columnOf` triple
stated on the view's page. All 29 gold answers are fully sourced
(`explain_answer.py --all`), and `evaluate.py` reports
`explained_correct_rate`, the share of the agent's correct answers whose every
evidence triple has a source.

## Data rules

The documentation is CC-BY-4.0: reuse with attribution, which the graph
carries (`dct:license`, `dct:publisher "SAP SE"`, page URL per node). Article
pages are SAP's property: none is downloaded automatically; pages saved by
hand stay in `supportkg/data/`, which git ignores, and the graph keeps only
numbers, titles, codes and links. The test pages in `tests/fixtures/` are
synthetic (invented text, fake numbers).

## Status

- built: ontology, PPMS stand-in, SHACL shapes, documentation ingestion,
  article sources that do not scrape, spaCy extraction, lexical and embedding
  linkers, graph builder with provenance and OWL materialisation, link and
  extraction scoring
- built: support benchmark (32 questions, gold checked on both graphs),
  schema-first retrieval for the agents, the graph without inverses,
  explainable answers (evidence triples traced to pages or OWL axioms)
- next: run the agents on both graphs; post-hoc linker experiments (no
  template, hybrid score, LLM re-ranking), reported as post-hoc
