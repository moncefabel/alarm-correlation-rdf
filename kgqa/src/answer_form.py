"""Answer-form scoring: a name given instead of the entity it names.

The first scorer (evaluate.score) maps an rdfs:label to its entity, so an agent
that projects ?label instead of ?alarm is not penalised. On the SAP graph this
is not enough. "Which columns does M_PASSWORD_POLICY have?" is answered
correctly by the names PROPERTY and VALUE, but the label of the entity is
"PROPERTY column of M_PASSWORD_POLICY", and the bare name PROPERTY names many
columns, so no global label lookup can resolve it. The failure analysis of the
first Claude Haiku runs (2 of the 10 persistent errors) found this.

The rule: a predicted value counts as a gold entity if it equals one of that
gold entity's own names (rdfs:label, or a property whose name ends in "Name":
viewName, columnName). It is resolved against the gold answer only, never
against the whole graph, so a name that happens to match another entity does
not earn credit. It was written after seeing the results, so it is reported
next to the original scorer, never instead of it (scripts/rescore.py).
"""

from __future__ import annotations

from evaluate import score


def name_aliases(store, gold_rows: list[dict]) -> dict[str, str]:
    iris = {str(v) for row in gold_rows for v in row.values() if str(v).startswith("http")}
    aliases: dict[str, str] = {}
    for iri in iris:
        res = store.select(f"SELECT ?p ?v WHERE {{ <{iri}> ?p ?v . FILTER(isLiteral(?v)) }}")
        for r in res.rows:
            local = str(r["p"]).split("#")[-1].split("/")[-1]
            if local == "label" or local.endswith("Name"):
                aliases[str(r["v"]).strip().lower()] = iri
    return aliases


def score_by_name(pred_rows, gold_rows, labels, store):
    """evaluate.score, with the gold entities' own names also accepted."""
    return score(pred_rows, gold_rows, {**labels, **name_aliases(store, gold_rows)})
