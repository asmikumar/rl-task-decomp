from jinja2 import Template
import json, re
from typing import Optional

PLANNER_TMPL = Template(r'''
You will decompose the following task into a small sequence of concrete, model-executable subtasks.
Return **ONLY JSON** with this exact schema:
{
  "original_task_prompt": "<string>",
  "subtask_list": ["<short names>"],
  "identified_constraints": [{"constraint": "<str>", "validation_strategy": "<str>"}],
  "subtasks": [
    {
      "subtask": "<unique name>",
      "tag": "<optional label>",
      "constraints": [{"constraint":"<str>","validation_strategy":"<str>"}],
      "prompt_template": "<what to send to the model>",
      "input_vars_required": ["<vars to fill from context>"],
      "depends_on": ["<names of earlier subtasks>"]
    }
  ]
}

The task:
{{ instruction }}

Constraints (may be empty):
{% for c in constraints %}- {{ c }}
{% endfor %}

Guidelines:
- Produce 2–4 subtasks, then a final subtask named EXACTLY "FINAL".
- Each "prompt_template" must be runnable by an LLM when variables are filled.
- Use variables from context: input.instruction, input.constraints, and outputs of prior subtasks by name.
- Keep depends_on accurate.
- The "FINAL" prompt_template must say: output ONLY the final answer text, with NO explanations, NO headings, NO bullets, NO markdown.
- Output ONLY valid JSON, no prose, no markdown.
''')

def _extract_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    # Try fenced ```json blocks
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", raw, flags=re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Try any JSON-looking block (largest braces)
    starts = [i for i,ch in enumerate(raw) if ch == '{']
    ends   = [i for i,ch in enumerate(raw) if ch == '}']
    if starts and ends and max(ends) > min(starts):
        s, e = min(starts), max(ends)
        candidate = raw[s:e+1]
        try:
            return json.loads(candidate)
        except Exception:
            # try to trim trailing garbage
            for k in range(len(candidate)-1, 0, -1):
                if candidate[k] == '}':
                    try:
                        return json.loads(candidate[:k+1])
                    except Exception:
                        continue
    # Last attempt: first {...} occurrence non-greedy
    m2 = re.search(r"(\{[\s\S]*?\})", raw)
    if m2:
        try:
            return json.loads(m2.group(1))
        except Exception:
            return None
    return None

def _synth_plan(instruction: str, constraints: list[str]) -> dict:
    cons_text = "\\n".join(f"- {c}" for c in (constraints or []))
    return {
        "original_task_prompt": instruction,
        "subtask_list": ["DRAFT","FINAL"],
        "identified_constraints": [{"constraint": c, "validation_strategy": "self-check"} for c in (constraints or [])],
        "subtasks": [
            {
                "subtask": "DRAFT",
                "tag": "DRAFT",
                "constraints": [],
                "prompt_template": (
                    "You are assisting with an instruction-following task.\n"
                    "Instruction:\n{{ input.instruction }}\n\n"
                    "Constraints:\n" + cons_text + "\n\n"
                    "Write a best-effort draft that satisfies the instruction semantically. "
                    "Do not explain your reasoning."
                ),
                "input_vars_required": ["input.instruction","input.constraints"],
                "depends_on": []
            },
            {
                "subtask": "FINAL",
                "tag": "FINAL",
                "constraints": [],
                "prompt_template": (
                    "Given the task and constraints, produce ONLY the final answer text with no explanations.\n"
                    "Task:\n{{ input.instruction }}\n\n"
                    "Constraints (must be strictly satisfied):\n" + cons_text + "\n\n"
                    "Here is a draft to refine:\n{{ DRAFT }}\n\n"
                    "Output ONLY the final answer text, no markdown, no headings."
                ),
                "input_vars_required": ["input.instruction","input.constraints","DRAFT"],
                "depends_on": ["DRAFT"]
            }
        ]
    }

def make_fallback_plan(instruction: str, constraints: list[str], model_call) -> dict:
    prompt = PLANNER_TMPL.render(instruction=instruction, constraints=constraints or [])
    raw = model_call(prompt)
    data = _extract_json(raw)
    if not isinstance(data, dict) or not data.get("subtasks"):
        # robust safety net
        data = _synth_plan(instruction, constraints or [])
    return data
