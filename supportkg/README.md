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
    --topology both --seed 0                                  # -> kgqa/results_support
python kgqa/src/evaluate.py --domain support --model claude-haiku-5-5 \
    --topology both --seed 0 --ontology supportkg/out_noinv/graph.ttl \
    --out kgqa/results_support_noinv
# the third arm: the orchestrator with the semantic check, on both graphs
python kgqa/src/evaluate.py --domain support --model claude-haiku-5-5 \
    --topology orchestrator_checked --seed 0
python kgqa/src/evaluate.py --domain support --model claude-haiku-5-5 \
    --topology orchestrator_checked --seed 0 --ontology supportkg/out_noinv/graph.ttl \
    --out kgqa/results_support_noinv
```

Two ways to the same goal, compared on the same questions: materialise the
inverses so either direction works, or keep the graph lean and let the
semantic check (`kgqa/src/semantic_check.py`) send a reversed triple back with
its fix. On this graph the domains and ranges differ, so the type check sees
every reversal.

One run of one arm is about 110 000 input tokens (the schema is in every
prompt), so about $0.03 with Claude Haiku 5.5 or gpt-oss-120b on a paid tier;
the free Groq tier's daily quota holds about one arm-run.

Results: pending.

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
