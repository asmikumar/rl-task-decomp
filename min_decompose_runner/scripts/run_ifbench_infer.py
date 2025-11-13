# min_decompose_runner/scripts/run_ifbench_infer.py

from __future__ import annotations

import json
import orjson
import pathlib
import re
from typing import Optional, List, Dict, Any, Iterator
from dataclasses import dataclass

from tqdm import tqdm
from datasets import load_dataset

from .llm_clients import chat  # fixed llama3.1 via Ollama


@dataclass
class Args:
    out: str
    ifbench_repo: Optional[str] = None
    hf_dataset: str = "allenai/IFBench_test"
    hf_split: Optional[str] = None
    max_items: int = 0
    temperature: float = 0.0
    top_p: float = 1.0
    start_index: int = 0              # zero-based index to start from
    item_index: Optional[int] = None  # if set, run exactly this index


def _iter_prompts(args: Args) -> Iterator[Dict[str, Any]]:
    """
    Yield {"key": ..., "prompt": ...} starting at args.start_index
    from either a local IFBench repo JSONL file or an HF dataset.
    """
    start = args.start_index

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

    split = args.hf_split or "test"
    ds = load_dataset(args.hf_dataset, split=split)

    for i, ex in enumerate(ds):
        if i < start:
            continue
        yield {
            "key": ex.get("key", f"ifb_{i:04d}"),
            "prompt": ex.get("prompt") or ex.get("instruction") or ex.get("input", ""),
        }


def _strip_reasoning(text: str) -> str:
    """
    Remove lines starting with 'Thought:' or 'Thoughts:' (case-insensitive).
    """
    lines = [l for l in text.splitlines() if not re.match(r"^\s*Thoughts?:", l, re.I)]
    return "\n".join(lines).strip()


def main(argv: Optional[List[str]] = None):
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--ifbench_repo")
    p.add_argument("--hf_dataset", default="allenai/IFBench_test")
    p.add_argument("--hf_split")
    p.add_argument("--max_items", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
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
    with pred_path.open("w") as w:
        for ex in tqdm(_iter_prompts(args)):
            if args.max_items and n >= args.max_items:
                break

            key, prompt = ex["key"], ex["prompt"]

            try:
                raw = chat(
                    [{"role": "user", "content": prompt}],
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                clean = _strip_reasoning(raw)
            except Exception as e:
                raw = clean = f"<<ERROR: {e}>>"

            rec = {
                "key": key,
                "prompt": prompt,
                "completion": raw,
                "answer": clean,
            }
            w.write(orjson.dumps(rec).decode() + "\n")
            n += 1

    print(f"Wrote {n} predictions to {pred_path}")


if __name__ == "__main__":
    main()
