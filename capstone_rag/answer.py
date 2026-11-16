"""Answer a question from retrieved chunks, with citations and two guardrails:

1. Retrieval gate: if nothing relevant was found, refuse before calling the model.
2. Citation check: every [n] in the answer must point at a real source, and an
   answer with no citations is flagged as unsupported.
"""
import os
import re
from dataclasses import dataclass, field

from . import ollama
from .index import Index

NOT_FOUND = "NOT_IN_DOCS"
REFUSAL = "I couldn't find that in the capstone docs."
MIN_BM25 = float(os.environ.get("RAG_MIN_BM25", "6.0"))
MIN_COSINE = float(os.environ.get("RAG_MIN_COSINE", "0.55"))

SYSTEM = f"""You answer questions about David Moad's hospital IoT security capstone lab,
using ONLY the numbered sources provided.

Rules:
- Every factual sentence ends with the source number(s) it came from, like [2] or [1][3].
- If the sources don't contain the answer, reply with exactly: {NOT_FOUND}
- Don't guess IPs, ports, commands, or values. Copy them exactly from the sources.
- If sources disagree, say so and prefer the most recent one (the field guide wins
  over older notes).
- Some values appear as [REDACTED]. Never try to guess them; say they are redacted.
- Be brief: a few sentences or a short list of steps."""


@dataclass
class Answer:
    question: str
    text: str
    sources: list = field(default_factory=list)   # [(n, source, heading)]
    refused: bool = False
    reason: str = ""
    cited: list = field(default_factory=list)
    bad_citations: list = field(default_factory=list)

    @property
    def supported(self) -> bool:
        return self.refused or (bool(self.cited) and not self.bad_citations)


def relevant_enough(hits: list[dict]) -> tuple[bool, str]:
    if not hits:
        return False, "no keyword or vector matches"
    top_bm = max(h["bm25"] for h in hits)
    cos = [h["cosine"] for h in hits if h["cosine"] is not None]
    top_cos = max(cos) if cos else None
    if top_bm >= MIN_BM25 or (top_cos is not None and top_cos >= MIN_COSINE):
        return True, ""
    detail = f"best keyword score {top_bm:.1f} < {MIN_BM25}"
    if top_cos is not None:
        detail += f", best similarity {top_cos:.2f} < {MIN_COSINE}"
    return False, detail


def format_context(hits: list[dict]) -> str:
    blocks = []
    for n, h in enumerate(hits, 1):
        c = h["chunk"]
        blocks.append(f"[{n}] {c.source} — {c.heading or '(top)'}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def ask(index: Index, question: str, k: int = 6, model: str = None) -> Answer:
    hits = index.search(question, k=k)
    sources = [(n, h["chunk"].source, h["chunk"].heading) for n, h in enumerate(hits, 1)]
    ok, why = relevant_enough(hits)
    if not ok:
        return Answer(question, REFUSAL, sources, refused=True, reason=f"retrieval gate: {why}")

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Sources:\n\n{format_context(hits)}\n\nQuestion: {question}"},
    ]
    text = ollama.chat(messages, model=model or ollama.CHAT_MODEL).strip()
    if NOT_FOUND in text:
        return Answer(question, REFUSAL, sources, refused=True, reason="model found no answer in sources")

    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    bad = [n for n in cited if not 1 <= n <= len(hits)]
    return Answer(question, text, sources, cited=cited, bad_citations=bad)
