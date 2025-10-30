import argparse, orjson
from pathlib import Path
from src.simple_runner.fetch_ifbench import fetch_one_ifbench
from src.simple_runner.planner import plan_or_fallback
from src.simple_runner.graph import plan_to_graph
from src.simple_runner.run_exec import run_graph

ap = argparse.ArgumentParser()
ap.add_argument("--index", type=int, default=0)
ap.add_argument("--dataset", default="allenai/IFBench_test")
ap.add_argument("--split", default=None)  # auto-picks if None
ap.add_argument("--out_dir", default="decomp_runs/ifbench/single")
args = ap.parse_args()

out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
item = fetch_one_ifbench(out_dir, idx=args.index, dataset=args.dataset, split=args.split)
(out_dir / "item.json").write_bytes(orjson.dumps(item, option=orjson.OPT_INDENT_2))

decomp = plan_or_fallback(item["instruction"], item["constraints"])
(out_dir / "decompose.json").write_bytes(orjson.dumps(decomp, option=orjson.OPT_INDENT_2))

G = plan_to_graph(decomp)
(out_dir / "taskgraph.json").write_bytes(orjson.dumps(G, option=orjson.OPT_INDENT_2))

res = run_graph(G, item)
(out_dir / "run_result.json").write_bytes(orjson.dumps(res, option=orjson.OPT_INDENT_2))

print("Wrote:")
for f in ("item.json","decompose.json","taskgraph.json","run_result.json"):
    print(" ", out_dir / f)
