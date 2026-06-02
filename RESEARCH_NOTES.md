# Research Notes

## Session 1 — 2026-06-02

### Goal

Build a knowledge graph for network alarm correlation that demonstrates
the difference between statistical co-occurrence and causal reasoning.
The core question: given the same raw alarm data, what can a causal
knowledge graph answer that a correlation-based ML system cannot?

---

### Experiment 1 — SPARQL diagnostic queries

Built a 102-triple RDF/OWL graph modeling a DDoS incident (5 alarms,
3 devices). Five SPARQL queries cover alarm listing, causal chains,
correlations, device hotspots, and MITRE ATT&CK classification.

The most important finding is the **co-location trap**:

Q4 ranks Router R1 as the hotspot (3 alarms). A naive
co-occurrence system would investigate R1 first. Q2 shows the
causal root is Firewall FW1 (1 alarm). Intervening on R1 does
nothing — the attack traffic keeps flowing. The KG identifies
the correct intervention point with a single SPARQL query.

This is not a contrived example. In real network operations,
a single incident generates dozens of alarms spread across
multiple devices. Finding the root cause among correlated
symptoms is precisely the problem that kills MTTR (mean time
to resolution) in production NOCs.

---

### Experiment 2 — Explicit comparison: correlation vs causal

`causal_vs_correlation.py` makes the contrast programmatic and visual.

The correlation system groups alarms by device co-location and flags
R1 as the problem. The causal KG traverses the causal chain, finds the
node with no incoming edges (FW1), and identifies it as the intervention
point that stops all 4 downstream alarms.

Same 5 alarms, same raw data — fundamentally different conclusions.

---

### Experiment 3 — Counterfactual reasoning (Pearl Level 3)

For each alarm, the system answers: "If we had intervened on X,
how many downstream alarms would not have occurred?"

Results:
- FW traffic spike: prevents 4 alarms (optimal)
- CPU overload: prevents 3 alarms
- Packet drop: prevents 1 alarm
- Latency spike: prevents 0
- Service timeout: prevents 0

This is Pearl's Level 3 — counterfactual reasoning. The system
is not just describing what happened, it is answering a hypothetical
about what would have happened under a different action.

This level of reasoning is impossible with correlation models alone.
You cannot answer "what if" questions from co-occurrence statistics.
You need a causal model.

---

### Experiment 4 — OWL transitive closure

The ontology explicitly encodes 4 causal links:
FW→CPU, CPU→Latency, CPU→PacketDrop, PacketDrop→Timeout.

The transitive closure algorithm infers 4 additional implicit links:
- CPU overload → Service timeout (via Packet drop, not stated)
- FW spike → Latency spike (via CPU, not stated)
- FW spike → Packet drop (via CPU, not stated)
- FW spike → Service timeout (chain of 3 hops, not stated)

These 4 links were never written by anyone. The system derived them
from the transitivity of the causal relation. In production this means:
as diagnostic agents add new direct causal links, the full causal
picture grows automatically without any manual enrichment.

---

### Open questions for future work

1. Causal links in this prototype are encoded as hard facts. In practice
   they are uncertain — two agents might disagree on whether A causes B.
   Combining DS belief fusion (belief-fusion-diagnosis) with this KG is a
   natural next step: encode causal links as belief masses, not binary facts.

2. The KG currently has no notion of time. Alarms have timestamps but
   the causal reasoning ignores ordering. Adding temporal constraints
   (A can only cause B if A happened before B) would reduce false positives
   in the causal chain.

3. The SPARQL queries are hand-written. LLM-to-SPARQL translation would
   make the KG accessible to NOC operators without SPARQL knowledge.
   This connects to the agentic AI axis of the thesis.   