import os

USE_OLLAMA = os.getenv("USE_OLLAMA", "0") == "1"

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
MODEL_ID        = os.getenv("MODEL_ID", "llama-3.1-8b-instruct")

OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct")
