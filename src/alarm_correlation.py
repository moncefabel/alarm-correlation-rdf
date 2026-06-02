
from __future__ import annotations
import os
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

RESULTS = "../results"
os.makedirs(RESULTS, exist_ok=True)

NET = Namespace("http://example.org/network#")


def load_graph() -> Graph:
    g = Graph()
    g.parse("../ontology/alarm_ontology.ttl", format="turtle")
    print(f"Graph loaded: {len(g)} triples")
    return g


# --- SPARQL Queries ---

QUERIES = {

    "Q1 — All alarms with severity and device": """
        PREFIX net: <http://example.org/network#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?alarm ?label ?type ?severity ?device
        WHERE {
            ?alarm a net:Alarm ;
                   rdfs:label ?label ;
                   net:alarmType ?type ;
                   net:hasSeverity ?severity ;
                   net:affectsDevice ?dev .
            ?dev rdfs:label ?device .
        }
        ORDER BY ?alarm
    """,

    "Q2 — Causal chain: which alarm is the root cause?": """
        PREFIX net: <http://example.org/network#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?cause ?causeLabel ?effect ?effectLabel
        WHERE {
            ?cause net:causallyLinkedTo ?effect .
            ?cause rdfs:label ?causeLabel .
            ?effect rdfs:label ?effectLabel .
        }
        ORDER BY ?cause
    """,

    "Q3 — Correlated alarms (same cause, not causal)": """
        PREFIX net: <http://example.org/network#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?a1 ?label1 ?a2 ?label2
        WHERE {
            ?a1 net:correlatedWith ?a2 .
            ?a1 rdfs:label ?label1 .
            ?a2 rdfs:label ?label2 .
        }
    """,

    "Q4 — Alarms on the same device (co-location)": """
        PREFIX net: <http://example.org/network#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?device ?deviceLabel (COUNT(?alarm) AS ?alarmCount)
        WHERE {
            ?alarm a net:Alarm ;
                   net:affectsDevice ?device .
            ?device rdfs:label ?deviceLabel .
        }
        GROUP BY ?device ?deviceLabel
        ORDER BY DESC(?alarmCount)
    """,

    "Q5 — Incident classification and MITRE ATT&CK mapping": """
        PREFIX net: <http://example.org/network#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?incident ?incidentLabel ?pattern ?patternLabel ?mitreId
        WHERE {
            ?incident a net:Incident ;
                      rdfs:label ?incidentLabel ;
                      net:classifiedAs ?pattern .
            ?pattern  rdfs:label ?patternLabel ;
                      net:mitreId ?mitreId .
        }
    """,
}


def run_queries(g: Graph) -> dict:
    results = {}
    for name, query in QUERIES.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print('='*60)
        rows = list(g.query(query))
        if not rows:
            print("  (no results)")
        for row in rows:
            print("  " + " | ".join(str(v).split("#")[-1] for v in row))
        results[name] = rows
    return results


# --- Visualisation ---

def build_nx_graph(g: Graph) -> nx.DiGraph:
    G = nx.DiGraph()

    label_map = {}
    for s, _, o in g.triples((None, RDFS.label, None)):
        label_map[str(s)] = str(o)

    type_map = {}
    for s, _, o in g.triples((None, RDF.type, None)):
        type_map[str(s)] = str(o).split("#")[-1]

    causal = NET.causallyLinkedTo
    corr   = NET.correlatedWith

    for s, _, o in g.triples((None, causal, None)):
        sl = label_map.get(str(s), str(s).split("#")[-1])
        ol = label_map.get(str(o), str(o).split("#")[-1])
        G.add_node(sl, node_type=type_map.get(str(s), "Alarm"))
        G.add_node(ol, node_type=type_map.get(str(o), "Alarm"))
        G.add_edge(sl, ol, edge_type="causal")

    for s, _, o in g.triples((None, corr, None)):
        sl = label_map.get(str(s), str(s).split("#")[-1])
        ol = label_map.get(str(o), str(o).split("#")[-1])
        G.add_node(sl, node_type=type_map.get(str(s), "Alarm"))
        G.add_node(ol, node_type=type_map.get(str(o), "Alarm"))
        if not G.has_edge(sl, ol):
            G.add_edge(sl, ol, edge_type="correlated")

    return G


def plot_graph(g: Graph) -> None:
    G = build_nx_graph(g)

    severity_colors = {
        "CPU spike on R1":            "#DC2626",
        "Latency spike on R1":        "#F97316",
        "Packet drop on R1":          "#7C3AED",
        "Inbound traffic spike on FW1": "#2563EB",
        "Request timeout on WS1":     "#059669",
    }

    causal_edges = [(u, v) for u, v, d in G.edges(data=True)
                    if d.get("edge_type") == "causal"]
    corr_edges   = [(u, v) for u, v, d in G.edges(data=True)
                    if d.get("edge_type") == "correlated"]

    fig, ax = plt.subplots(figsize=(12, 7))
    pos = nx.spring_layout(G, seed=42, k=2.5)

    node_colors = [severity_colors.get(n, "#9CA3AF") for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=2200, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8,
                            font_weight="bold", ax=ax)

    nx.draw_networkx_edges(G, pos, edgelist=causal_edges,
                           edge_color="#1F2937", width=2.5,
                           arrows=True, arrowsize=25,
                           connectionstyle="arc3,rad=0.1", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=corr_edges,
                           edge_color="#9CA3AF", width=1.5,
                           style="dashed", arrows=True,
                           arrowsize=20,
                           connectionstyle="arc3,rad=0.2", ax=ax)

    legend = [
        mpatches.Patch(color="#2563EB", label="Firewall alarm (root)"),
        mpatches.Patch(color="#DC2626", label="CPU overload"),
        mpatches.Patch(color="#F97316", label="Latency spike"),
        mpatches.Patch(color="#7C3AED", label="Packet drop"),
        mpatches.Patch(color="#059669", label="Service timeout"),
        mpatches.Patch(color="#1F2937", label="Causal link →"),
        mpatches.Patch(color="#9CA3AF", label="Correlation (dashed)"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=9)
    ax.set_title(
        "Network Alarm Correlation Graph\n"
        "RDF Knowledge Graph — SPARQL-queryable\n"
        "Causal chain: FW spike → CPU → Latency / PacketDrop → Timeout",
        fontsize=12, fontweight="bold"
    )
    ax.axis("off")

    plt.tight_layout()
    out = f"{RESULTS}/alarm_correlation_graph.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nGraph saved: {out}")
    plt.close()


# --- Main ---

if __name__ == "__main__":
    print("Loading RDF graph from ontology/network_alarm.ttl...")
    g = load_graph()

    print("\nRunning SPARQL diagnostic queries...")
    run_queries(g)

    print("\nGenerating correlation graph visualization...")
    plot_graph(g)

    print("""
Thesis connection:

  Q2 (causal chain) vs Q3 (correlation) shows the core distinction
  of Verrou 1: ML agents detect co-occurrence (Q3-style), but diagnosis
  requires causal reasoning (Q2-style).

  The firewall spike is the ROOT CAUSE — if we had used correlation alone,
  we would have treated CPU spike, latency, and packet drop as independent
  anomalies requiring separate remediation. Causal reasoning identifies
  a single intervention point.

  Q5 (MITRE ATT&CK mapping) shows how the knowledge graph connects
  low-level alarms to standardised threat taxonomies — directly
  implementing the semantic layer described in NORIA-O.
""")