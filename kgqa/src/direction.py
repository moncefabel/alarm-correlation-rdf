"""Direction audit: did a query traverse a causal property the wrong way round?

C02 and C03 fail in a way no symbolic check sees. The query is valid, every
IRI exists, it runs, and it returns an empty or wrong set, because the named
alarm sits on the wrong side of `causallyLinkedTo`: "what caused the latency
spike" written as `net:Alarm_Latency net:causallyLinkedTo ?c` asks for its
effects. This module finds those cases by comparing the position of each named
entity in the predicted query with its position in the gold query.

It is an analysis tool, not part of either agent: adding it to one topology
would change the arms being compared.
"""

from __future__ import annotations

import re

def _pairs(query: str, prop: str) -> list[tuple[str, str]]:
    # '#' starts a comment only outside an IRI: full IRIs <...#x> are kept
    body = re.sub(r"(?<![<\w])#[^\n]*", " ", query or "")
    pat = re.compile(r"([^\s{}()]+)\s+\(?" + re.escape(prop) + r"\)?[+*?]?\s+([^\s{}();.]+)")
    return [(s.strip(".;"), o.strip(".;")) for s, o in pat.findall(body)]


def _sides(pairs: list[tuple[str, str]]) -> tuple[set[str], set[str]]:
    subj = {s for s, _ in pairs if not s.startswith(("?", "$"))}
    obj = {o for _, o in pairs if not o.startswith(("?", "$"))}
    return subj, obj


def inverted_entities(pred: str, gold: str, prop: str = "net:causallyLinkedTo",
                      inverse: str | None = "net:causedBy") -> list[str]:
    """Named entities the gold query puts on one side of `prop` and the
    prediction puts only on the other side.

    An entity the gold uses on both sides (a UNION over both directions, as in
    K02) is ignored: missing one direction is incompleteness, not inversion.
    A triple of the prediction written with the inverse property (the OWL
    variant's `net:causedBy`) counts with subject and object swapped, so
    `X causedBy ?c` is read as `?c causallyLinkedTo X`, the right way round.
    """
    g_subj, g_obj = _sides(_pairs(gold, prop))
    pairs = _pairs(pred, prop)
    if inverse:
        pairs += [(o, s) for s, o in _pairs(pred, inverse)]
    p_subj, p_obj = _sides(pairs)
    out = []
    for e in sorted(g_obj - g_subj):
        if e in p_subj and e not in p_obj:
            out.append(e)
    for e in sorted(g_subj - g_obj):
        if e in p_obj and e not in p_subj:
            out.append(e)
    return out
