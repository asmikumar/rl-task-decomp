from datasets import load_dataset, get_dataset_split_names
from pathlib import Path
import orjson

def _pick_split(dataset: str) -> str:
    names = get_dataset_split_names(dataset)
    for cand in ("test","validation","dev","val","train"):
        if cand in names: return cand
    return "train"

def fetch_one_ifbench(out_dir: Path, idx: int = 0, dataset: str = "allenai/IFBench_test", split: str | None = None):
    if split is None:
        split = _pick_split(dataset)
    ds = load_dataset(dataset, split=split)
    ex = ds[int(idx)]
    item = {
        "id": ex.get("id", f"ifb_{idx:06d}"),
        "instruction": ex.get("instruction") or ex.get("prompt") or ex.get("input") or "",
        "constraints": ex.get("constraints") or ex.get("constraint_list") or ex.get("constraint") or [],
        "gold_check": ex.get("gold_check") or ex.get("judge") or None,
        "eval_meta": {"split": split, "source": dataset}
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "item.json").write_bytes(orjson.dumps(item, option=orjson.OPT_INDENT_2))
    return item
