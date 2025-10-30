import time, orjson
from typing import Dict, Any
from .render import render_template
from .llm_clients import call_llm

def run_graph(G: Dict, item: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    ctx = {"input": {"instruction": item["instruction"], "constraints": item.get("constraints", [])}}
    artifacts = {}
    last_name = None

    for n in G["nodes"]:
        name = n["name"]
        tmpl = n["prompt_template"] or ""
        rendered = render_template(tmpl, ctx)
        out = call_llm(rendered)
        artifacts[name] = {"prompt": rendered, "output": out}
        ctx[name] = out
        last_name = name

    return {
        "final": ctx.get(last_name, ""),
        "artifacts": artifacts,
        "latency": time.time() - t0
    }
