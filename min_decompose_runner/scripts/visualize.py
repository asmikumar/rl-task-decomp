import json
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

out_dir = Path("decomp_runs/ifbench/single")
Gjson = json.loads((out_dir/"taskgraph.json").read_text())

G = nx.DiGraph()
for n in Gjson["nodes"]:
    G.add_node(n["id"], label=n["name"])
for e in Gjson["edges"]:
    G.add_edge(e["src"], e["dst"])

pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(6,4))
nx.draw_networkx_edges(G, pos, arrows=True)
nx.draw_networkx_nodes(G, pos, node_size=1200)
labels = {n: G.nodes[n]["label"] for n in G.nodes}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
plt.axis("off")
plt.tight_layout()
plt.savefig(out_dir/"graph.png", dpi=200)
print("Saved", out_dir/"graph.png")
