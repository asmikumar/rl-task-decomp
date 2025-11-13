from __future__ import annotations
import os, re, json, orjson, pathlib
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Iterator
from tqdm import tqdm
from datasets import load_dataset

# Use your repo's MELLEA-aware pipeline (planner → graph → executor)
from src.simple_runner.planner import plan_or_fallback
from src.simple_runner.graph import plan_to_graph
from src.simple_runner.run_exec import run_graph

@dataclass
class Args:
    out: str
    ifbench_repo: Optional[str] = None
    hf_dataset: str = "allenai/IFBench_test"
    hf_split: Optional[str] = None
    max_items: int = 0
    temperature: float = 0.0  # kept for parity; your pipeline should respect config

def _iter_ifbench_prompts(args: Args) -> Iterator[Dict[str, Any]]:
    if args.ifbench_repo:
        j = pathlib.Path(args.ifbench_repo) / "data" / "IFBench_test.jsonl"
        if j.exists():
            with j.open() as f:
                for i, line in enumerate(f):
                    ex = json.loads(line)
                    yield {"key": ex.get("key", f"ifb_{i:04d}"), "prompt": ex["prompt"]}
            return
    split = args.hf_split or "test"
    ds = load_dataset(args.hf_dataset, split=split)
    for i, ex in enumerate(ds):
        yield {"key": ex.get("key", f"ifb_{i:04d}"),
               "prompt": ex.get("prompt") or ex.get("instruction") or ex.get("input", "")}

def _strip_reasoning(text: str) -> str:
    lines = [l for l in text.splitlines() if not re.match(r"^\s*Thoughts?:", l, re.I)]
    return "\n".join(lines).strip()

def _prompt_to_item(prompt: str) -> Dict[str, Any]:
    # Your pipeline expects {"instruction": ..., "constraints": [...]}
    # IFBench encodes constraints in the text; we pass full prompt as the instruction.
    return {"instruction": prompt, "constraints": []}

def main(argv: Optional[List[str]] = None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--ifbench_repo")
    p.add_argument("--hf_dataset", default="allenai/IFBench_test")
    p.add_argument("--hf_split")
    p.add_argument("--max_items", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    a = p.parse_args(argv)
    args = Args(**vars(a))

    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "preds.jsonl"

    n = 0
    with pred_path.open("w") as w:
        for ex in tqdm(_iter_ifbench_prompts(args)):
            if args.max_items and n >= args.max_items:
                break
            key, prompt = ex["key"], ex["prompt"]
            try:
                item = _prompt_to_item(prompt)
                plan = plan_or_fallback(item["instruction"], item.get("constraints", []))
                G = plan_to_graph(plan)
                res = run_graph(G, item)
                raw = res.get("final","")
                clean = _strip_reasoning(raw)
            except Exception as e:
                raw = clean = f"<<ERROR: {e}>>"
            rec = {"key": key, "prompt": prompt, "completion": raw, "answer": clean}
            w.write(orjson.dumps(rec).decode() + "\n")
            n += 1
    print(f"Wrote {n} predictions to {pred_path}")

if __name__ == "__main__":
    main()
