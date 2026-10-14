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
was seen. They are scored against `gold/alert_views.yaml` (32 alerts),
**currently a draft** to be checked against the documentation before scores
are reported.

| Method | Precision | Recall | F1 | hit@2 |
|---|---|---|---|---|
| lexical (draft gold) | 0.469 | 0.375 | 0.417 | 0.438 |
| embedding | run `evaluate_links.py --method hf` | | | |

The lexical baseline finds links whose words match (`Disk usage` →
`M_DISK_USAGE`) and misses paraphrases (`Number of diagnosis files` →
`M_TRACEFILES`), which is what the embedding model is there to close.

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
python supportkg/src/build.py                       # -> supportkg/out/graph.ttl, provenance.ttl, report.json
python supportkg/src/evaluate_links.py --show       # lexical alert -> view links against the gold list

pip install sentence-transformers                   # optional: Hugging Face models (PyTorch)
python supportkg/src/evaluate_links.py --method both --show
python supportkg/src/build.py --view-linker hf      # graph with embedding links

# load into Fuseki next to the alarm graph
docker compose -f kgqa/docker-compose.yml up -d
DATASET=support bash kgqa/scripts/load_fuseki.sh supportkg/out/graph.ttl
```

With pages saved by hand in `supportkg/data/saved/`:
`python supportkg/src/collect.py --mode saved`, then `build.py` again, and
`evaluate_extraction.py --template 25` for the annotation form of the
article-level extraction.

Tests (offline, no SAP page, no model download; one test also checks the gold
views against the real documentation when the clone is present):

```bash
pytest supportkg/tests -q
```

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
- next: check the gold list, run the embedding linker, then a support
  benchmark (which view for which alert, which columns, which statement, which
  article) through the kgqa agents, with and without the materialised inverses
