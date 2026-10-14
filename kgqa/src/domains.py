"""Domain profiles: everything in the prompts that names what the graph is about.

The agents were written for the network-alarm graph, and three strings said so:
the scope paragraph of the generation prompt, the domain line given to the
question classifier, and the type-specific instructions. A profile holds those
strings, so the same agents run on another graph without editing their code.

The ALARMS profile reproduces the original strings byte for byte
(tests/test_domains.py checks a hash of every prompt the four arms send), so
the published alarm results stay valid.

`retrieval` selects how the schema context is chosen (schema_context.py):
  lexical       the original retrieval, kept for the alarm graph
  tbox_entities every class and property, plus the entities the question
                names. Needed on the support graph: 7 000 entities, 6 490 of
                them columns, swamp a top-25 lexical retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_GEN_HEAD = """You write SPARQL 1.1 SELECT queries against an RDF knowledge graph.
Rules:
- Use ONLY the classes and properties given in the schema. Never invent a URI.
- Always include the PREFIX declarations you use.
- Return ONLY the query, no prose, no markdown fences.
"""

_GEN_TAIL = """
- Return exactly OUT_OF_SCOPE only when the question is about something this
  graph does not describe at all (general knowledge, another domain)."""


@dataclass(frozen=True)
class Domain:
    name: str
    rules: str                     # domain lines of the generation prompt
    graph_line: str                # what the classifier is told the graph holds
    labels: tuple[str, ...]        # classifier labels, as listed to the model
    types: tuple[str, ...]         # the same labels, in matching order
    default_type: str
    hints: dict[str, str] = field(default_factory=dict)
    retrieval: str = "lexical"
    questions: str = ""            # default benchmark
    graph: str = ""                # default graph file

    @property
    def gen_system(self) -> str:
        return _GEN_HEAD + self.rules + _GEN_TAIL

    @property
    def classify_system(self) -> str:
        return ("Classify the question. Answer with exactly one word: "
                + ", ".join(self.labels[:-1]) + ", or " + self.labels[-1] + ".")


ALARMS = Domain(
    name="alarms",
    rules="""- For causal questions use the causal properties, not mere co-occurrence.
- Scope: the graph describes network devices, alarms, incidents, attack patterns
  and the causal links between alarms. Questions about root causes, where to
  intervene, what a prevented or blocked alarm would have avoided, or which
  alarm or device matters most ARE in scope: answer them from the causal links.""",
    graph_line="network alarms, incidents, causal dependencies.",
    labels=("factual", "causal", "counterfactual", "out_of_scope"),
    types=("counterfactual", "causal", "out_of_scope", "factual"),
    default_type="factual",
    hints={
        "causal": "This is a ROOT-CAUSE question. Traverse causal properties "
                  "transitively; do not rely on co-occurrence.",
        "counterfactual": "This is a COUNTERFACTUAL question. Identify the "
                          "downstream set that would not occur if the named "
                          "node were removed or blocked.",
    },
    questions="kgqa/benchmark/questions.yaml",
    graph="ontology/alarm_ontology.ttl",
)

SUPPORT = Domain(
    name="support",
    rules="""- Check each property's domain and range: they give the direction of the
  triple. When a property points the wrong way for the question, use its
  inverse property or swap subject and object.
- Entities are named by their identifiers: views and statements by their
  name (M_DISK_USAGE, ALTER SYSTEM SAVEPOINT), alerts by their number,
  articles by their number. Use the entity IRIs given in the schema.
- Scope: the graph describes SAP HANA support knowledge: knowledge articles
  (SAP Notes, KBAs), the alerts they address, the system and monitoring views
  and SQL statements of the documentation with their columns, and products,
  versions and support packages. Questions about which article or view to use
  for an alert, what a view contains, or how these relate ARE in scope.""",
    graph_line="SAP HANA support: knowledge articles, alerts, system views, "
               "SQL statements, columns, products.",
    labels=("lookup", "multihop", "aggregation", "negation", "out_of_scope"),
    types=("multihop", "aggregation", "negation", "out_of_scope", "lookup"),
    default_type="lookup",
    hints={
        "multihop": "This question CROSSES SOURCES. Chain the properties "
                    "(article, alert, view, column) and check each property's "
                    "domain and range for the direction of every hop.",
        "aggregation": "This is a COUNTING question. Use COUNT, with GROUP BY "
                       "and ORDER BY ... LIMIT for 'which has the most'.",
        "negation": "This question asks what is MISSING. Use FILTER NOT EXISTS "
                    "around the pattern that must be absent.",
    },
    retrieval="tbox_entities",
    questions="supportkg/benchmark/questions.yaml",
    graph="supportkg/out/graph.ttl",
)

DOMAINS = {d.name: d for d in (ALARMS, SUPPORT)}
