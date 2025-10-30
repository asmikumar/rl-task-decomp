# src/simple_runner/planner.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import os

# Mellea (planner)
try:
    from mellea.decompose import decompose, DecompBackend
    _HAS_MELLEA = True
except Exception:
    _HAS_MELLEA = False

# Local helpers (already in your repo)
from .llm_clients import call_llm
from .fallback_planner import make_fallback_plan

STEER = """
You are a task decomposition engine. Decompose the task into 3–6 concrete, sequential, model-executable subtasks.

For each subtask, include:
- "subtask": a concise action name (free-form; e.g., parse, plan, draft, refine, check, answer).
- "prompt_template": a self-contained prompt the LLM can run when variables are filled.
- "input_vars_required": variables needed (e.g., input.instruction, input.constraints, and outputs of prior subtasks by name).
- "depends_on": names of earlier subtasks this one needs.

Guidelines:
- Prefer semantic separation (e.g., understand → draft → enforce constraints → finalize). Do NOT force a specific tag list.
- If constraints are present, include a first subtask named "CONSTRAINTS_JSON" that outputs ONLY strict JSON (no prose) that normalizes constraints (e.g., keyword_counts, length, format, schemas). Later subtasks must reference {{ CONSTRAINTS_JSON }} when enforcing constraints.
- The final subtask should produce ONLY the final task output (no analysis/markdown/headings).
- Ensure later prompts reference earlier outputs when needed (e.g., include {{ DRAFT }} if refining).
- Output MUST be valid JSON in the DecompPipelineResult schema.
"""

def _ask_mellea(task_text: str) -> Dict[str, Any]:
    """
    Single call to mellea.decompose with Ollama backend.
    Model and endpoint configured via env:
      - MELLEA_MODEL_ID       (default: "llama3.1")
      - MELLEA_ENDPOINT       (default: "http://127.0.0.1:11434")
      - MELLEA_TIMEOUT_SEC    (default: 300)
    """
    model_id = os.getenv("MELLEA_MODEL_ID", "llama3.1")
    endpoint = os.getenv("MELLEA_ENDPOINT", "http://127.0.0.1:11434")
    timeout = int(os.getenv("MELLEA_TIMEOUT_SEC", "300"))

    res = decompose(
        task_prompt=task_text,
        user_input_variable=None,
        model_id=model_id,
        backend=DecompBackend.ollama,
        backend_req_timeout=timeout,
        backend_endpoint=endpoint,
        backend_api_key=None,
    )
    if isinstance(res, dict):
        return res
    # Mellea returns a TypedDict-like object; be permissive:
    d = getattr(res, "__dict__", None)
    if isinstance(d, dict):
        return d
    return dict(res)

def _ensure_depth_or_retry(instruction: str, constraints: List[str]) -> Dict[str, Any]:
    """
    Ask Mellea once; if <3 subtasks, gently remind to return 3–6 and ask again.
    """
    base_prompt = STEER + "\n\nTASK:\n" + instruction
    plan = _ask_mellea(base_prompt)
    if not isinstance(plan, dict) or not plan.get("subtasks") or len(plan["subtasks"]) < 3:
        stronger = base_prompt + "\n\nReminder: Return 3–6 subtasks; avoid collapsing steps. Include the 'CONSTRAINTS_JSON' first if constraints exist."
        plan = _ask_mellea(stronger)
    return plan

def mellea_plan(task_prompt: str) -> Dict[str, Any]:
    """
    Main Mellea planner entry: wraps the depth/quality guard.
    """
    plan = _ensure_depth_or_retry(task_prompt, [])
    # absolute guard—if Mellea somehow returned nothing, caller will fallback
    if not isinstance(plan, dict) or not plan.get("subtasks"):
        raise RuntimeError("Mellea returned no subtasks")
    return plan

def plan_or_fallback(instruction: str, constraints: List[str]) -> Dict[str, Any]:
    """
    Public API used by the runner.
    - Prefer Mellea unless FORCE_FALLBACK=1 or Mellea is unavailable/broken.
    - Fall back to JSON planner (same schema) using the execution LLM.
    """
    if _HAS_MELLEA and os.getenv("FORCE_FALLBACK", "0") != "1":
        try:
            plan = mellea_plan(instruction)
        except Exception:
            plan = make_fallback_plan(instruction, constraints or [], model_call=call_llm)
    else:
        plan = make_fallback_plan(instruction, constraints or [], model_call=call_llm)

    # final safety net—synthesize a minimal plan if still empty
    if not isinstance(plan, dict) or not plan.get("subtasks"):
        from .fallback_planner import _synth_plan  # type: ignore
        plan = _synth_plan(instruction, constraints or [])
    return plan


# import os
# from typing import Dict, Any
# try:
#     from mellea.decompose import decompose, DecompBackend
#     _HAS_MELLEA = True
# except Exception:
#     _HAS_MELLEA = False

# from .llm_clients import call_llm
# from .fallback_planner import make_fallback_plan

# RETRY_MODELS = [
#     # (backend_model_id, human label)
#     (os.getenv("MELLEA_MODEL_ID", "llama3.1"), "llama3.1"),
#     ("mistral:7b-instruct", "mistral:7b-instruct"),
#     ("qwen2.5:7b-instruct", "qwen2.5:7b-instruct"),
# ]

# STEER = """
# You are a task decomposition engine. Decompose the task into 3–6 concrete, sequential, model-executable subtasks.

# For each subtask, include:
# - "subtask": a concise action name (free-form; e.g., parse, plan, draft, refine, check, answer).
# - "prompt_template": a self-contained prompt the LLM can run when variables are filled.
# - "input_vars_required": variables needed (e.g., input.instruction, input.constraints, outputs of prior subtasks by name).
# - "depends_on": names of earlier subtasks this one needs.

# Guidelines:
# - Prefer semantic separation (e.g., understand → draft → enforce constraints → finalize), but do not force any specific tags.
# - If constraints are present, each constriant should be handled in its own subtask.
# - The final subtask should produce ONLY the final task output (no analysis/markdown/headings).
# - Output MUST be valid JSON in the DecompPipelineResult schema.
# """

# def mellea_plan(task_prompt: str) -> Dict[str, Any]:
#     # Use Ollama as the backend; endpoint defaults to local Ollama
#     endpoint = os.getenv("MELLEA_ENDPOINT", "http://127.0.0.1:11434")
#     last_err = None
#     task_prompt = STEER + "\n\nTASK:\n" + instruction
#     for model_id, label in RETRY_MODELS:
#         try:
#             result = decompose(
#                 task_prompt=task_prompt,
#                 user_input_variable=None,
#                 model_id=model_id,
#                 backend=DecompBackend.ollama,
#                 backend_req_timeout=300,
#                 backend_endpoint=endpoint,
#                 backend_api_key=None,
#             )
#             plan = result if isinstance(result, dict) else getattr(result, "__dict__", dict(result))
#             # safety net: ensure subtasks present
#             if not isinstance(plan, dict) or not plan.get("subtasks"):
#                 raise ValueError("Mellea returned empty subtasks")
#             return plan
#         except Exception as e:
#             last_err = e
#             # try next model
#             continue
#     # If all retries fail, raise the last error so caller can fallback if allowed
#     raise last_err

# def plan_or_fallback(instruction: str, constraints: list[str]) -> dict:
#     task_prompt = instruction
#     if _HAS_MELLEA and os.getenv("FORCE_FALLBACK","0") != "1":
#         try:
#             plan = mellea_plan(task_prompt)
#         except Exception:
#             # final safety: use our JSON fallback
#             print("Using fallback plan...")
#             plan = make_fallback_plan(instruction, constraints or [], model_call=call_llm)
#     else:
#         plan = make_fallback_plan(instruction, constraints or [], model_call=call_llm)
#     # absolute guard
#     if not isinstance(plan, dict) or not plan.get("subtasks"):
#         from .fallback_planner import _synth_plan
#         plan = _synth_plan(instruction, constraints or [])
#     return plan
