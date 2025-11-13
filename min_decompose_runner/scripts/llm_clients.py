import os, httpx, json

# MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "llama3.1")
MODEL_NAME = "llama3.1"
BASE = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

def _ensure_server_ok():
    r = httpx.get(f"{BASE}/api/tags", timeout=10)
    r.raise_for_status()

def _parse_ndjson(text: str) -> str:
    objs = re.findall(r'\{.*?\}', text)
    out = []
    for o in objs:
        try:
            j = json.loads(o)
            if "response" in j and j["response"]:
                out.append(j["response"])
        except:
            continue
    return "".join(out)

def chat(messages, temperature: float = 0.0, top_p: float = 1.0) -> str:
    _ensure_server_ok()
    prompt = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
    try:
        r = httpx.post(
            f"{BASE}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "top_p": top_p},
            },
            timeout=(10, 60),  # 10s connect, 60s read
        )
    except httpx.ReadTimeout as e:
        raise RuntimeError(f"Ollama request timed out: {e}") from e

    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e

    # Primary: parse as JSON
    try:
        j = r.json()
        if isinstance(j, dict) and ("response" in j or "message" in j):
            return j.get("response") or j.get("message") or ""
    except json.JSONDecodeError:
        pass

    # Fallback: NDJSON or SSE style (streaming data concatenated)
    text = r.text
    parsed = _parse_ndjson(text)
    if parsed:
        return parsed

    # Last resort
    return text or ""
