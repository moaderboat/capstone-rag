"""Minimal Ollama client (stdlib only). Everything stays on localhost."""
import json
import os
import urllib.error
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.environ.get("RAG_CHAT_MODEL", "llama3.1:8b")


class OllamaUnavailable(RuntimeError):
    pass


def _post(path: str, payload: dict, timeout: float = 300):
    req = urllib.request.Request(
        HOST + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        raise OllamaUnavailable(f"Can't reach Ollama at {HOST}: {e}") from e


def available() -> bool:
    try:
        with urllib.request.urlopen(HOST + "/api/tags", timeout=2):
            return True
    except Exception:
        return False


def embed(texts: list[str], model: str = EMBED_MODEL, batch: int = 32) -> list[list[float]]:
    out = []
    for i in range(0, len(texts), batch):
        out.extend(_post("/api/embed", {"model": model, "input": texts[i:i + batch]})["embeddings"])
    return out


def chat(messages: list[dict], model: str = CHAT_MODEL, temperature: float = 0.1) -> str:
    r = _post("/api/chat", {"model": model, "messages": messages, "stream": False,
                            "options": {"temperature": temperature}})
    return r["message"]["content"]
