# Research Notes

## Session 1 — 2026-06-02

### Goal

Build a minimal but functional RDF knowledge graph for network alarm
correlation. The core scientific question: can a knowledge graph with
explicit causal structure identify the root cause of a network incident
where a correlation-only system would fail?

---

### Experiment 1 — SPARQL diagnostic queries

Built a 102-triple RDF graph modeling a DDoS incident with 5 alarms
across 3 network devices. Five SPARQL queries cover the full diagnostic
reasoning pipeline.

**Key finding — the co-location trap:**

Q4 shows Router R1 has 3 alarms, making it the obvious "problem device"
from a co-occurrence perspective. Q2 shows the actual root cause is on
Firewall FW1, which has only 1 alarm.

A system that ranks devices by alarm count would focus remediation on
R1 and leave the attack traffic flowing. The knowledge graph identifies
FW1 as the single intervention point that stops all 5 alarms.

**Why this matters practically:**

In production network monitoring, a single incident can generate
hundreds of correlated alarms (alarm storm). Without causal structure,
each alarm generates a separate ticket and a separate investigation.
The KG collapses 5 alarms into 1 root cause and 1 action.

**Pearl's causal hierarchy, illustrated concretely:**

- Association (what ML detects): Latency spike and Packet drop
  co-occur on R1. A co-occurrence model groups them.
- Intervention (what the KG enables): Remove the FW traffic spike,
  and CPU overload stops. Latency, packet drop, and service timeout
  all disappear as a consequence.
- Counterfactual: Would packet drop have occurred without CPU overload?
  The causal chain says no. This is the reasoning level required for
  automated remediation.

**NORIA-O alignment:**

The ontology classes (Alarm, Incident, NetworkDevice, AttackPattern)
and properties (causallyLinkedTo, correlatedWith, affectsDevice)
directly mirror the structure of NORIA-O, the reference ontology for
anomaly detection in ICT systems developed at Orange Innovation
(Tailhardat, Chabot, Troncy — ESWC 2023). This prototype can be seen
as a minimal NORIA-O instantiation for a concrete DDoS scenario.

---

### Experiment 2 — Explicit comparison: correlation vs causal

Added `causal_vs_correlation.py` to make the contrast quantifiable.

**Correlation system output:**
- Groups alarms by device co-location
- Flags Core Router R1 (3 alarms) as the intervention target
- Result: WRONG. Intervening on R1 does not stop the DDoS traffic.

**Causal KG output:**
- Traverses the causal chain to find the node with no incoming edges
- Identifies Firewall FW1 as the root cause
- Result: CORRECT. Blocking inbound traffic at FW1 stops 4 downstream alarms.

Same 5 alarms. Same raw data. Fundamentally different diagnostic conclusion.

---

### Open questions for future work

1. How to handle causal uncertainty? The current graph encodes causal
   links as facts. In practice, causal relationships between alarms are
   probabilistic and context-dependent. Combining DS belief fusion with
   causal KG reasoning is a natural next step.

2. How to maintain consistency when multiple agents write to the KG
   simultaneously? If two diagnostic agents update the graph concurrently,
   OWL open-world semantics may produce inconsistencies. This is the
   concurrency problem for the knowledge graph as a shared semantic space.

3. Can SPARQL queries be generated automatically from natural language
   incident descriptions? LLM-to-SPARQL translation is an active research
   area and would make the KG accessible to non-expert operators.