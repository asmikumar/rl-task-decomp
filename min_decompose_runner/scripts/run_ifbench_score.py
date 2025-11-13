from __future__ import annotations
import argparse, json, pathlib, subprocess, sys, tempfile
from typing import Dict, Any, Iterable

def _load_jsonl(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ifbench_repo", required=True)          # ../IFBench
    p.add_argument("--data_jsonl", required=True)            # ../IFBench/data/IFBench_test.jsonl
    p.add_argument("--pred_jsonl", required=True)            # your preds.jsonl
    p.add_argument("--out_dir", required=True)               # output dir for IFBench results
    # Matching behavior
    p.add_argument("--prefer_key_match", action="store_true",
                   help="Prefer matching by key only; fall back to prompt match if key missing.")
    a = p.parse_args(argv)

    ifbench_repo = pathlib.Path(a.ifbench_repo).resolve()
    data_jsonl   = pathlib.Path(a.data_jsonl).resolve()
    pred_jsonl   = pathlib.Path(a.pred_jsonl).resolve()
    out_dir      = pathlib.Path(a.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load predictions; build {key->text} and {prompt->text} maps ----
    key_to_text: Dict[str, str] = {}
    prompt_to_text: Dict[str, str] = {}

    for ex in _load_jsonl(pred_jsonl):
        text = ex.get("answer") or ex.get("completion") or ""
        if not isinstance(text, str):
            text = str(text)

        k = ex.get("key")
        if isinstance(k, int):
            k = str(k)
        if isinstance(k, str) and k != "":
            key_to_text[k] = text

        pr = ex.get("prompt")
        if isinstance(pr, str) and pr != "":
            prompt_to_text[pr] = text

    if not key_to_text and not prompt_to_text:
        print("[ifbench_score] No usable predictions found in preds.jsonl.", file=sys.stderr)
        sys.exit(2)

    # ---- Filter official data to only examples we have a prediction for ----
    filtered_data = []
    used_prompts = set()
    matched_by_key = matched_by_prompt = 0

    for off in _load_jsonl(data_jsonl):
        off_key = off.get("key")
        if isinstance(off_key, int):
            off_key = str(off_key)
        off_prompt = off.get("prompt", "")

        resp = None
        # Match by key first (default & recommended)
        if a.prefer_key_match and isinstance(off_key, str) and off_key in key_to_text:
            resp = key_to_text[off_key]
            matched_by_key += 1
        else:
            # Try both paths with key first
            if isinstance(off_key, str) and off_key in key_to_text:
                resp = key_to_text[off_key]
                matched_by_key += 1
            elif isinstance(off_prompt, str) and off_prompt in prompt_to_text:
                resp = prompt_to_text[off_prompt]
                matched_by_prompt += 1

        if resp is not None:
            # Keep the official example as-is for input_data
            filtered_data.append(off)
            used_prompts.add(off_prompt)

    kept = len(filtered_data)
    if kept == 0:
        print("[ifbench_score] No overlap between preds.jsonl and IFBench data. "
              "Nothing to evaluate.", file=sys.stderr)
        sys.exit(2)

    print(f"[ifbench_score] Matched {kept} example(s): "
          f"{matched_by_key} by key, {matched_by_prompt} by prompt. "
          f"(Total preds: keys={len(key_to_text)}, prompts={len(prompt_to_text)})")

    # ---- Write temp files: filtered input_data + normalized responses ----
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".jsonl") as f_data, \
         tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".jsonl") as f_resp:

        filtered_data_path = pathlib.Path(f_data.name)
        norm_resp_path     = pathlib.Path(f_resp.name)

        # 1) filtered input_data
        for off in filtered_data:
            f_data.write(json.dumps(off, ensure_ascii=False) + "\n")

        # 2) normalized responses: {"prompt": <official prompt>, "response": <text>}
        for off in filtered_data:
            prompt = off.get("prompt", "")
            if prompt in prompt_to_text:
                text = prompt_to_text[prompt]
            else:
                # fall back to key map
                k = off.get("key")
                if isinstance(k, int):
                    k = str(k)
                text = key_to_text.get(k, "")
            f_resp.write(json.dumps({"prompt": prompt, "response": text}, ensure_ascii=False) + "\n")

    # ---- Call official IFBench runner on the subset ----
    cmd = [
        sys.executable,
        str(ifbench_repo / "run_eval.py"),
        f"--input_data={str(filtered_data_path)}",
        f"--input_response_data={str(norm_resp_path)}",
        f"--output_dir={str(out_dir)}",
    ]
    print(f"[ifbench_score] Running: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
        print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        print(f"[ifbench_score] Done. Results in: {out_dir}")
    except subprocess.CalledProcessError as e:
        print(e.stdout or "", file=sys.stdout)
        print(e.stderr or "", file=sys.stderr)
        print(f"[ifbench_score] IFBench eval failed (exit={e.returncode}). "
              f"Filtered data: {filtered_data_path} | "
              f"Normalized responses: {norm_resp_path}", file=sys.stderr)
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
