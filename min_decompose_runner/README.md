# Minimal Decomposition Runner (GPU-ready)

This repo runs **Mellea `decompose(...)`** (if installed) or a **fallback planner** to split a single IF‑Bench task into **subtasks**, then **executes those subtasks with one LLM call per subtask**. 
It’s intentionally *simple*: no VERIFY/FIX stages, no tag-specialization—just run the `prompt_template`s in dependency order.

## Backends (pick one)
- **OpenAI-compatible endpoint** (e.g., vLLM, TGI): set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `MODEL_ID`.
- **Ollama**: set `USE_OLLAMA=1` and `OLLAMA_MODEL`.

> GPU is used by your serving stack (vLLM/TGI/Ollama). This runner just calls the API.

## Setup

```bash
# in your existing venv
pip install -r requirements.txt

# Option A: OpenAI-compatible (recommended for GPU via vLLM)
export OPENAI_BASE_URL="http://localhost:8000/v1"   # example vLLM endpoint
export OPENAI_API_KEY="dummy"                        # vLLM usually accepts any string
export MODEL_ID="llama-3.1-8b-instruct"             # whatever your server exposes

# Option B: Ollama (ensure it’s serving your model)
export USE_OLLAMA=1
export OLLAMA_MODEL="llama3.1:8b-instruct"
```

## Run a single IF‑Bench example end-to-end

```bash
# 1) fetch → plan (mellea or fallback) → graph → run → visualize
python scripts/run_one.py --index 0
```

Artifacts will appear in `decomp_runs/ifbench/single/`:
- `item.json` – IF‑Bench record
- `decompose.json` – `DecompPipelineResult`-like plan (from Mellea or fallback)
- `taskgraph.json` – nodes/edges built from subtasks (`prompt_template`, `depends_on`)
- `run_result.json` – final text + per-node outputs + latency
- `graph.png` – simple visualization of the task graph

## Notes

- If `mellea` is *not* installed, we send a *strict JSON-only planning prompt* to your LLM that asks it to output the same schema (fields: `subtask`, `prompt_template`, `depends_on`, `input_vars_required`). 
- Execution is uniform: we render each `prompt_template` with a simple context and call the LLM. Context includes: 
  - `input.instruction`, `input.constraints`, and
  - outputs of prior subtasks under keys named by each subtask’s `subtask` field.
- Keep it simple: **no special-casing tags** and **no verify/fix loops**.
