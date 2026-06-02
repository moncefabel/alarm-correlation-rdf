"""
Explicit comparison: what a correlation-based system does
vs what a causal knowledge graph does on the same incident.

The core problem in network incident diagnosis:
Most ML-based monitoring systems detect statistical co-occurrence
between alarms — they flag alarms that tend to appear together.
This is useful for grouping but insufficient for diagnosis, because
it cannot identify the ROOT CAUSE or the correct INTERVENTION POINT.

A knowledge graph with explicit causal structure can answer:
  "Which single action stops all the alarms simultaneously?"

This script makes that contrast concrete and measurable.
"""

from __future__ import annotations
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from rdflib import Graph, Namespace
from rdflib.namespace import RDFS

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)
NET = Namespace("http://example.org/network#")


def load_graph() -> Graph:
    g = Graph()
    g.parse("ontology/alarm_ontology.ttl", format="turtle")
    return g


#  Correlation-based approach (what ML co-occurrence detects) 

def correlation_approach(g: Graph) -> dict:
    """
    Simulate a correlation-based diagnostic system.
    Groups alarms by co-occurrence on the same device.
    Ranks devices by alarm count — assumes the device with most
    alarms is the problem.
    """
    q = """
        PREFIX net: <http://example.org/network#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?device ?deviceLabel (COUNT(?alarm) AS ?count)
        WHERE {
            ?alarm a net:Alarm ;
                   net:affectsDevice ?device .
            ?device rdfs:label ?deviceLabel .
        }
        GROUP BY ?device ?deviceLabel
        ORDER BY DESC(?count)
    """
    rows = list(g.query(q))
    top_device = str(rows[0][1]) if rows else "Unknown"
    alarm_count = int(rows[0][2]) if rows else 0

    return {
        "method": "Correlation (co-location)",
        "conclusion": f"Intervene on: {top_device} ({alarm_count} alarms)",
        "correct": False,
        "explanation": (
            f"{top_device} has the most alarms so it is flagged as "
            "the problem. But this is a symptom device, not the cause."
        ),
        "device_counts": {str(r[1]): int(r[2]) for r in rows},
    }


#  Causal approach (what the knowledge graph does) 

def causal_approach(g: Graph) -> dict:
    """
    Use the causal chain in the KG to find the root cause.
    Root cause = alarm with no incoming causal links (no parent).
    """
    # Find all alarms that ARE caused by something else
    has_parent = set()
    q_parents = """
        PREFIX net: <http://example.org/network#>
        SELECT ?effect WHERE {
            ?cause net:causallyLinkedTo ?effect .
        }
    """
    for row in g.query(q_parents):
        has_parent.add(str(row[0]))

    # Find all alarms
    all_alarms = {}
    q_alarms = """
        PREFIX net: <http://example.org/network#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?alarm ?label ?device ?deviceLabel WHERE {
            ?alarm a net:Alarm ;
                   rdfs:label ?label ;
                   net:affectsDevice ?dev .
            ?dev rdfs:label ?deviceLabel .
        }
    """
    for row in g.query(q_alarms):
        all_alarms[str(row[0])] = {
            "label": str(row[1]),
            "device": str(row[3]),
        }

    # Root cause = alarm with no parent
    root_causes = {k: v for k, v in all_alarms.items()
                   if k not in has_parent}

    # Count downstream effects of root cause
    def count_downstream(alarm_uri: str, visited=None) -> int:
        if visited is None:
            visited = set()
        if alarm_uri in visited:
            return 0
        visited.add(alarm_uri)
        children = []
        for row in g.query(f"""
            PREFIX net: <http://example.org/network#>
            SELECT ?child WHERE {{
                <{alarm_uri}> net:causallyLinkedTo ?child .
            }}
        """):
            children.append(str(row[0]))
        return len(children) + sum(count_downstream(c, visited)
                                   for c in children)

    root = list(root_causes.items())[0]
    downstream = count_downstream(root[0])

    return {
        "method": "Causal (knowledge graph)",
        "conclusion": f"Intervene on: {root[1]['device']} (root cause)",
        "correct": True,
        "explanation": (
            f"The KG identifies the alarm with no parent cause. "
            f"Removing it stops {downstream} downstream alarm(s)."
        ),
        "root_alarm": root[1]["label"],
        "root_device": root[1]["device"],
        "downstream_stopped": downstream,
    }


#  Visualisation 

def plot_comparison(corr: dict, causal: dict, g: Graph) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        "Correlation vs Causal Reasoning — Network Incident Diagnosis\n"
        "Same 5 alarms, same data — fundamentally different conclusions",
        fontsize=13, fontweight="bold", y=0.98
    )

    # --- Left: correlation bar chart ---
    ax = axes[0]
    devices = list(corr["device_counts"].keys())
    counts  = list(corr["device_counts"].values())
    device_colors = ["#DC2626" if i == 0 else "#9CA3AF"
                     for i in range(len(devices))]
    bars = ax.bar(devices, counts, color=device_colors, alpha=0.85, width=0.5)
    ax.bar_label(bars, fmt="%d alarms", fontsize=10, padding=5)
    ax.set_ylabel("Number of alarms", fontsize=11)
    ax.set_title(
        "Correlation approach\n(group by co-location)",
        fontweight="bold", fontsize=11, pad=10
    )
    ax.set_ylim(0, max(counts) + 2.5)
    ax.tick_params(axis="x", labelsize=9)

    wrong = mpatches.Patch(color="#DC2626", label="Flagged as root cause (WRONG)")
    ax.legend(handles=[wrong], fontsize=9, loc="upper right")

    # Conclusion box INSIDE the plot, top area
    ax.text(
        0.5, 0.97,
        f"Conclusion: {corr['conclusion']}\n"
        f"Intervenes on symptom — attack traffic continues.",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=9, color="#DC2626",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEE2E2",
                  edgecolor="#DC2626", alpha=0.9)
    )

    # --- Right: causal chain ---
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis("off")
    ax.set_title(
        "Causal approach\n(knowledge graph traversal)",
        fontweight="bold", fontsize=11, pad=10
    )

    nodes = [
        (5, 9.5, "FW spike\n(ROOT CAUSE)", "#2563EB", True),
        (5, 7.5, "CPU overload",           "#DC2626", False),
        (3, 5.5, "Latency spike",          "#F97316", False),
        (7, 5.5, "Packet drop",            "#7C3AED", False),
        (5, 3.5, "Service timeout",        "#059669", False),
    ]

    for x, y, label, color, is_root in nodes:
        ellipse = mpatches.Ellipse(
            (x, y), width=2.8, height=1.1,
            color=color, alpha=0.9, zorder=3
        )
        ax.add_patch(ellipse)
        if is_root:
            ring = mpatches.Ellipse(
                (x, y), width=3.1, height=1.3,
                color="gold", fill=False, lw=3, zorder=2
            )
            ax.add_patch(ring)
            ax.text(x + 1.8, y + 0.4, "← intervene here",
                    fontsize=9, color="#2563EB", fontweight="bold")
        ax.text(x, y, label, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=4)

    arrows = [
        (5, 8.95, 5, 8.05),
        (5, 6.95, 3.2, 6.05),
        (5, 6.95, 6.8, 6.05),
        (6.8, 5.05, 5.5, 4.05),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=2, color="#1F2937"),
            zorder=3
        )

    # Conclusion box at the very bottom, well below nodes
    ax.text(
        5, 1.8,
        f"Conclusion: {causal['conclusion']}\n"
        f"Stops {causal['downstream_stopped']} downstream alarms — single intervention point.",
        ha="center", va="center", fontsize=9, color="#059669",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#D1FAE5",
                  edgecolor="#059669", alpha=0.9)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = f"{RESULTS}/causal_vs_correlation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")
    plt.close()


#  Main 

if __name__ == "__main__":
    print("Loading knowledge graph...")
    g = load_graph()

    print("\n--- Correlation approach ---")
    corr = correlation_approach(g)
    print(f"  Conclusion : {corr['conclusion']}")
    print(f"  Correct    : {corr['correct']}")
    print(f"  Explanation: {corr['explanation']}")

    print("\n--- Causal approach ---")
    causal = causal_approach(g)
    print(f"  Root cause : {causal['root_alarm']}")
    print(f"  Device     : {causal['root_device']}")
    print(f"  Downstream : stops {causal['downstream_stopped']} alarms")
    print(f"  Conclusion : {causal['conclusion']}")

    print("\nGenerating comparison figure...")
    plot_comparison(corr, causal, g)

    print(f"""
Summary:
  Correlation sees: Router R1 has 3 alarms → act on R1
  Causal KG sees : Firewall FW1 is root → act on FW1

  Same 5 alarms. Same data. Different answer.
  The correct answer stops all 5 alarms with one action.
  The correlation answer misses the actual cause entirely.

  This is why network incident diagnosis needs causal reasoning,
  not just statistical pattern matching.
""")