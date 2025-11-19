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
    # Root of the google-research checkout, or directly the instruction_following_eval dir
    # Example 1: /path/to/google-research
    # Example 2: /path/to/google-research/instruction_following_eval
    p.add_argument("--ifeval_repo", required=True,
                   help="Path to google-research or instruction_following_eval dir.")
    p.add_argument("--data_jsonl", required=True,
                   help="IFEval input_data.jsonl (or a subset in the same format).")
    p.add_argument("--pred_jsonl", required=True,
                   help="Your preds.jsonl with key/prompt/answer or completion fields.")
    p.add_argument("--out_dir", required=True,
                   help="Output dir for IFEval results.")
    p.add_argument("--prefer_key_match", action="store_true",
                   help="Prefer matching by key only; fall back to prompt match if key missing.")
    a = p.parse_args(argv)

    ifeval_repo = pathlib.Path(a.ifeval_repo).resolve()
    data_jsonl   = pathlib.Path(a.data_jsonl).resolve()
    pred_jsonl   = pathlib.Path(a.pred_jsonl).resolve()
    out_dir      = pathlib.Path(a.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # If the user pointed at the google-research root, descend into the subdir.
    if (ifeval_repo / "instruction_following_eval").is_dir():
        ifeval_repo = ifeval_repo / "instruction_following_eval"

    evaluation_main_py = ifeval_repo / "evaluation_main.py"
    if not evaluation_main_py.exists():
        print(f"[ifeval_score] Could not find evaluation_main.py at {evaluation_main_py}", file=sys.stderr)
        sys.exit(1)

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
        print("[ifeval_score] No usable predictions found in preds.jsonl.", file=sys.stderr)
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
            if isinstance(off_key, str) and off_key in key_to_text:
                resp = key_to_text[off_key]
                matched_by_key += 1
            elif isinstance(off_prompt, str) and off_prompt in prompt_to_text:
                resp = prompt_to_text[off_prompt]
                matched_by_prompt += 1

        if resp is not None:
            filtered_data.append(off)
            used_prompts.add(off_prompt)

    kept = len(filtered_data)
    if kept == 0:
        print("[ifeval_score] No overlap between preds.jsonl and IFEval data. "
              "Nothing to evaluate.", file=sys.stderr)
        sys.exit(2)

    print(f"[ifeval_score] Matched {kept} example(s): "
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
                k = off.get("key")
                if isinstance(k, int):
                    k = str(k)
                text = key_to_text.get(k, "")
            f_resp.write(json.dumps({"prompt": prompt, "response": text}, ensure_ascii=False) + "\n")

        # ---- Call official IFEval runner on the subset ----
    # We must run it as a module so `import instruction_following_eval` works.
    # If ifeval_repo is ".../instruction_following_eval", the package root is its parent.
    if ifeval_repo.name == "instruction_following_eval":
        pkg_root = ifeval_repo.parent
    else:
        # ifeval_repo points to the google-research root
        pkg_root = ifeval_repo

    cmd = [
        sys.executable,
        "-m",
        "instruction_following_eval.evaluation_main",
        f"--input_data={str(filtered_data_path)}",
        f"--input_response_data={str(norm_resp_path)}",
        f"--output_dir={str(out_dir)}",
    ]
    print(f"[ifeval_score] Running: (cwd={pkg_root}) {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
            cwd=str(pkg_root),   # <<< important: run from parent dir
        )
        print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        print(f"[ifeval_score] Done. Results in: {out_dir}")
    except subprocess.CalledProcessError as e:
        print(e.stdout or "", file=sys.stdout)
        print(e.stderr or "", file=sys.stderr)
        print(
            f"[ifeval_score] IFEval eval failed (exit={e.returncode}). "
            f"Filtered data: {filtered_data_path} | "
            f"Normalized responses: {norm_resp_path}",
            file=sys.stderr,
        )
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
