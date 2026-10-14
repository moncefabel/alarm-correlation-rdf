# supportkg: an SAP support knowledge graph from public sources

A small version of the full pipeline that a support knowledge graph needs: build a
multi-source knowledge graph for SAP service and support, then query it with
LLM agents in SPARQL, and measure both. The query side is
[`kgqa/`](../kgqa/); this module is the build side.

```
public KBA previews ──► collect ──► parse ──► extract (spaCy rules + HF linker)
PPMS-shaped stand-in ─────────────────────────────┐        │
tools gazetteer ──────────────────────────────────┤        ▼
support.ttl (OWL) ────────────────────────────────┴──► build: entity resolution,
                                                        OWL inverse/symmetric materialisation,
                                                        PROV-O provenance, SHACL validation
                                                              │
                                     graph.ttl ◄──────────────┘──► Fuseki ──► kgqa agents
```

## Sources

| Source | What it gives | How it enters |
|---|---|---|
| Public previews of SAP Notes and KBAs | number, title, symptom, environment, product, keywords | collected on your machine, parsed, extracted |
| The "How to handle HANA Alert N" KBA series | one article per SAP HANA alert check: alert numbers and names | extracted from titles and symptoms |
| Application component codes (HAN-DB-MON...) | the support routing hierarchy | extracted from keywords; hierarchy from the code |
| PPMS-shaped stand-in (`ontology/ppms_standin.ttl`) | ProductLine, Product, ProductVersion, SoftwareComponent, SupportPackage for SAP HANA | hand-written; **not PPMS data**, PPMS is internal |
| Tools gazetteer (`ontology/tools.ttl`) | HANA studio, cockpit, DBACOCKPIT, hdbcons... | hand-curated |

Seed list: `seeds/articles.txt`, 39 articles (32 alert how-tos plus overviews and
SAP HANA Cloud articles). `--follow 1` also collects the articles they cite.

## Ontology

`ontology/support.ttl` (OWL 2), aligned with the PPMS vocabulary instead of
redefining it (`ppms:Product rdfs:subClassOf sup:ProductItem`). It builds in
the two lessons of the kgqa error analysis:

- **every directed relation has a named inverse**: `addresses` / `addressedBy`,
  `assignedTo` / `hasArticle`, `appliesTo` / `hasApplicableArticle`,
  `references` / `referencedBy`, `subComponentOf` / `hasSubComponent`. In kgqa,
  most of the orchestrator's remaining errors were queries that read
  `causallyLinkedTo` against its direction;
- **an undirected relation is symmetric**: `cooccursWith` between alerts
  reported together (co-occurrence, explicitly not causation). In kgqa, K02
  failed because `correlatedWith` was stored one way.

Also: `subComponentOf` is transitive (and a sub-property of
`skos:broaderTransitive`), KBA and SAP Note are disjoint, keys
(`owl:hasKey`) on article number, alert number and component code.
`ontology/shapes.ttl` holds the SHACL shapes the built graph must pass.

## Extraction

| Relation | Method | Notes |
|---|---|---|
| `addresses` | title rule | "How to handle HANA Alert 54: ..." and its variants ("How-To: Handle", "Handle HANA Alert 69", "alert (ID=75)") |
| `mentionsAlert` | regex in title and symptom; embedding linker | "Alert ID 39", "alerts 1, 2 and 43"; the linker covers articles that never give a number |
| `cooccursWith` | same article | alerts reported together |
| `assignedTo` | keyword rule | component codes; the following keyword is the component name |
| `subComponentOf` | code prefix | HAN-DB-MON → HAN-DB → HAN |
| `appliesTo` | gazetteer; embedding linker | Product section strings to the PPMS stand-in |
| `references` | regex | "SAP Note 1758890", "KBAs 2000002 and 2000003" |
| `usesTool` | gazetteer (spaCy PhraseMatcher) | tools.ttl labels |

spaCy runs as a blank English pipeline (no model download): a custom tokenizer
rule keeps component codes in one token, an EntityRuler built from the
ontology's own labels finds tools and products, a custom component turns
numbered references into entity spans. The Hugging Face linker
(`src/link.py`, `sentence-transformers/all-MiniLM-L6-v2` on PyTorch) is
optional and switched on with `--linker hf`; comparing the two is the
extraction ablation.

Every extracted triple gets a statement node in `out/provenance.ttl`:
source page (`prov:wasDerivedFrom`), method, confidence. An answer from the
agents can be traced back to the page each triple came from.

## Run

On your machine (the collector needs internet access to SAP's public pages):

```bash
pip install -r supportkg/requirements.txt
python supportkg/src/collect.py --follow 1           # about 3 s per page, cached in supportkg/data/raw
python supportkg/src/build.py                        # rules only -> supportkg/out/graph.ttl
pip install sentence-transformers                    # optional: Hugging Face linker (PyTorch)
python supportkg/src/build.py --linker hf

# load into Fuseki next to the alarm graph, and point kgqa at it
docker compose -f kgqa/docker-compose.yml up -d
DATASET=support bash kgqa/scripts/load_fuseki.sh supportkg/out/graph.ttl
```

Extraction scores, after annotating a sample by hand:

```bash
python supportkg/src/evaluate_extraction.py --template 25    # writes supportkg/gold/annotations.yaml
# fill it from the pages in a browser, without looking at the extractor's output
python supportkg/src/evaluate_extraction.py --show-errors
python supportkg/src/evaluate_extraction.py --linker hf
```

Tests (offline, no SAP page, no model download):

```bash
pytest supportkg/tests -q
```

## Data rules

SAP's texts are SAP's property. The collector reads `robots.txt` and skips any
disallowed path, waits between requests, identifies itself and caches every
page so nothing is fetched twice. Raw pages and parsed text stay in
`supportkg/data/`, which git ignores. The graph keeps article numbers, titles,
component codes, alert numbers and names, and links to the public page each
fact came from. The test pages in `tests/fixtures/` are synthetic: invented
text and fake numbers (99000xx).

## Status

- built: ontology, PPMS stand-in, SHACL shapes, collector, parser, spaCy
  extraction, embedding linker, graph builder with provenance and OWL
  materialisation, extraction scoring, 32 offline tests
- next: first collection run, annotated sample and extraction scores; a
  support benchmark (lookup, multi-hop article → alert → component → product,
  lifecycle, negation, out of scope) run through the kgqa agents, with and
  without the materialised inverses
