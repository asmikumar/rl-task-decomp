from __future__ import annotations

import json
import orjson
import pathlib
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Iterator

from tqdm import tqdm
from datasets import load_dataset

# --- Robust imports for the MELLEA pipeline ---------------------------
# This lets you run:
#   python -m min_decompose_runner.scripts.run_ifbench_infer_mellea
# from the rl-task-decomp ROOT (matching the baseline behavior).
try:
    # Case 1: src is at repo root (rl-task-decomp/src/...)
    from src.simple_runner.planner import plan_or_fallback
    from src.simple_runner.graph import plan_to_graph
    from src.simple_runner.run_exec import run_graph
except ModuleNotFoundError:
    # Case 2: src is inside min_decompose_runner (rl-task-decomp/min_decompose_runner/src/...)
    from ..src.simple_runner.planner import plan_or_fallback
    from ..src.simple_runner.graph import plan_to_graph
    from ..src.simple_runner.run_exec import run_graph


# ----------------- Arguments & dataset iteration ----------------- #

@dataclass
class Args:
    out: str
    ifbench_repo: Optional[str] = None
    hf_dataset: str = "allenai/IFBench_test"
    hf_split: Optional[str] = None
    max_items: int = 0
    temperature: float = 0.0
    start_index: int = 0              # zero-based index to start from
    item_index: Optional[int] = None  # if set, run exactly this index


def _iter_ifbench_prompts(args: Args) -> Iterator[Dict[str, Any]]:
    """
    Yield {"key": ..., "prompt": ...} starting at args.start_index
    from either a local IFBench repo JSONL file or an HF dataset.

    This mirrors the baseline run_ifbench_infer.py so scoring stays identical.
    """
    start = args.start_index

    # Prefer local IFBench repo JSONL if provided
    if args.ifbench_repo:
        j = pathlib.Path(args.ifbench_repo) / "data" / "IFBench_test.jsonl"
        if j.exists():
            with j.open() as f:
                for i, line in enumerate(f):
                    if i < start:
                        continue
                    ex = json.loads(line)
                    yield {
                        "key": ex.get("key", f"ifb_{i:04d}"),
                        "prompt": ex["prompt"],
                    }
            return

    # Otherwise, fall back to HF dataset
    split = args.hf_split or "train"
    ds = load_dataset(args.hf_dataset, split=split)

    for i, ex in enumerate(ds):
        if i < start:
            continue
        yield {
            "key": ex.get("key", f"ifb_{i:04d}"),
            "prompt": ex.get("prompt") or ex.get("instruction") or ex.get("input", ""),
        }


# ----------------- Prompt → MELLEA item ----------------- #

def _strip_reasoning(text: str) -> str:
    """
    Remove lines starting with 'Thought:' or 'Thoughts:' (case-insensitive).
    Same heuristic as the baseline so eval stays comparable.
    """
    lines = [l for l in text.splitlines() if not re.match(r"^\s*Thoughts?:", l, re.I)]
    return "\n".join(lines).strip()


def _split_instruction_and_constraints(prompt: str) -> tuple[str, List[str]]:
    """
    Heuristic splitter for IFBench prompts into:
      - `instruction`: the main question / task
      - `constraints`: a list of textual constraints

    For many IFBench examples, the pattern is:
      "<question> Include keyword X once..., keyword Y twice..., ..."

    We:
      - Try to cut before 'Include keyword' (or close variants) as the base instruction.
      - Treat the rest as a blob of constraint text and split it into sentences.
    """
    if not prompt:
        return "", []

    m = re.search(r"(Include\s+keyword|Include\s+the\s+keyword)", prompt, re.IGNORECASE)
    if m:
        instruction = prompt[: m.start()].strip()
        constraints_blob = prompt[m.start():].strip()
    else:
        instruction = prompt.strip()
        constraints_blob = ""

    constraints: List[str] = []
    if constraints_blob:
        parts = re.split(r"[.;]\s+", constraints_blob)
        for p in parts:
            p = p.strip()
            if p:
                constraints.append(p)

    if not instruction:
        instruction = prompt.strip()

    return instruction, constraints


def _prompt_to_item(prompt: str) -> Dict[str, Any]:
    """
    Map an IFBench prompt into the MELLEA-friendly item structure.

    For now, we just pass the FULL IFBench prompt through as the instruction,
    so nothing gets lost and the planner definitely sees a question.

    We also set multiple aliases (question/task/query/user_input) since
    different parts of the pipeline might look at different names.
    """
    inst = (prompt or "").strip()

    item: Dict[str, Any] = {
        "instruction": inst,
        "question": inst,
        "task": inst,
        "query": inst,
        "user_input": inst,
        # keep constraints empty for now; MELLEA can still decompose the text
        "constraints": [],
        "raw_prompt": prompt,
    }

    return item



# ----------------- Main: run MELLEA pipeline over IFBench ----------------- #

def main(argv: Optional[List[str]] = None):
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--ifbench_repo")
    p.add_argument("--hf_dataset", default="allenai/IFBench_test")
    p.add_argument("--hf_split")
    p.add_argument("--max_items", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Zero-based index to start from (default: 0).",
    )
    p.add_argument(
        "--item_index",
        type=int,
        help="If set, run exactly this zero-based index (overrides start_index and max_items).",
    )

    a = p.parse_args(argv)
    args = Args(**vars(a))

    # If item_index is set, treat it as "run exactly one example"
    if args.item_index is not None:
        args.start_index = args.item_index
        args.max_items = 1

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "preds.jsonl"

    n = 0
    with pred_path.open("w", encoding="utf-8") as w:
        for ex in tqdm(_iter_ifbench_prompts(args)):
            if args.max_items and n >= args.max_items:
                break

            key, prompt = ex["key"], ex["prompt"]

            try:
                # 1) Turn raw prompt into MELLEA item
                item = _prompt_to_item(prompt)

                # 2) Plan → graph → execute
                # NOTE: we pass the *item* to the planner, not just a string,
                # so it can look at question/task/query/etc as needed.
                plan = plan_or_fallback(item["instruction"], item.get("constraints", []))
                G = plan_to_graph(plan)
                res = run_graph(G, item)

                # 3) Extract the final string
                if isinstance(res, dict):
                    raw = res.get("final") or res.get("answer") or res.get("output", "")
                else:
                    raw = str(res)

                clean = _strip_reasoning(raw)
            except Exception as e:
                raw = clean = f"<<ERROR: {e}>>"

            rec = {
                "key": key,
                "prompt": prompt,
                # "completion": raw,
                "answer": clean,
            }
            w.write(orjson.dumps(rec).decode() + "\n")
            n += 1

    print(f"Wrote {n} predictions to {pred_path}")


if __name__ == "__main__":
    main()
