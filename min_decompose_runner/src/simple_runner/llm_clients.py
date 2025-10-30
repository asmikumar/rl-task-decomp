import requests
from . import config

def call_openai_chat(prompt: str) -> str:
    url = f"{config.OPENAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}"}
    payload = {
        "model": config.MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "stream": False,
    }
    r = requests.post(url, json=payload, headers=headers, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def call_ollama_generate(prompt: str) -> str:
    url = f"{config.OLLAMA_URL}/api/generate"
    payload = {"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

def call_llm(prompt: str) -> str:
    if config.USE_OLLAMA:
        return call_ollama_generate(prompt)
    return call_openai_chat(prompt)
