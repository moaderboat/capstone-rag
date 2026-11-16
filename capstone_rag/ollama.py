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


def _post(path: str, payload: dict, timeout: float = 90):
    req = urllib.request.Request(
        HOST + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        raise OllamaUnavailable(f"Can't reach Ollama at {HOST}: {e}") from e


_available = None


def available() -> bool:
    """Checked once per run, so a down server costs one 2-second timeout, not one per question."""
    global _available
    if _available is None:
        try:
            with urllib.request.urlopen(HOST + "/api/tags", timeout=2):
                _available = True
        except Exception:
            _available = False
    return _available


def embed(texts: list[str], model: str = EMBED_MODEL, batch: int = 32) -> list[list[float]]:
    out = []
    for i in range(0, len(texts), batch):
        out.extend(_post("/api/embed", {"model": model, "input": texts[i:i + batch]})["embeddings"])
    return out


def chat(messages: list[dict], model: str = CHAT_MODEL, temperature: float = 0.1,
         json_mode: bool = False) -> str:
    payload = {"model": model, "messages": messages, "stream": False,
               "options": {"temperature": temperature}}
    if json_mode:
        payload["format"] = "json"  # Ollama constrains output to valid JSON
        # small models in JSON mode can emit whitespace forever; cap the length
        payload["options"]["num_predict"] = 600
    return _post("/api/chat", payload)["message"]["content"]
