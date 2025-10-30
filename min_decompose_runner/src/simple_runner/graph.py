from typing import List, Dict
from collections import defaultdict, deque

def topo_order(subtasks: List[Dict]) -> List[Dict]:
    name_to_node = {s["subtask"]: s for s in subtasks}
    indeg = defaultdict(int)
    children = defaultdict(list)
    for s in subtasks:
        for p in s.get("depends_on", []):
            indeg[s["subtask"]] += 1
            children[p].append(s["subtask"])
    q = deque([s for s in name_to_node if indeg[s]==0])
    order_names = []
    while q:
        u = q.popleft()
        order_names.append(u)
        for v in children[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return [name_to_node[n] for n in order_names if n in name_to_node]

def plan_to_graph(decomp: Dict) -> Dict:
    nodes = []
    edges = []
    name_to_id = {}
    subs = decomp.get("subtasks", [])
    order = topo_order(subs) if subs else []
    for i, s in enumerate(order, 1):
        nid = f"n{i}"
        name_to_id[s["subtask"]] = nid
        nodes.append({
            "id": nid,
            "name": s["subtask"],
            "prompt_template": s.get("prompt_template",""),
            "input_vars_required": s.get("input_vars_required", []),
            "depends_on": s.get("depends_on", []),
        })
    for s in order:
        for p in s.get("depends_on", []):
            edges.append({"src": name_to_id[p], "dst": name_to_id[s["subtask"]]})
    return {"nodes": nodes, "edges": edges}
