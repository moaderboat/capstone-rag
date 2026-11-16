"""Build, save, load and search the index.

Hybrid retrieval: BM25 (exact terms like IPs, ports, service names) fused with
embedding similarity (paraphrased questions) using reciprocal rank fusion.
If Ollama isn't running, search falls back to BM25 alone.
"""
import fnmatch
import hashlib
import json
import math
import re
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from . import ollama
from .chunk import Chunk, chunk_markdown
from .redact import find_secrets, redact

DEFAULT_EXCLUDE = ["*CREDENTIAL*", "*credential*", "*secret*", "*.git/*", "*/.claude/*",
                   "*/backups/*", "*_Rev_[0-8].md", "*_Rev_9.md", "*-DESKTOP-*",
                   "*conversation_lost*"]
TOKEN = re.compile(r"[a-z0-9][a-z0-9_.:\-/]*[a-z0-9]|[a-z0-9]")
STOP = set("a an and are as at be by for from has have how i in is it of on or that the this "
           "to was what when where which who why will with does do did can should my".split())
DOC_PREFIX, QUERY_PREFIX = "search_document: ", "search_query: "


def tokenize(text: str) -> list[str]:
    toks = []
    for t in TOKEN.findall(text.lower()):
        if t in STOP:
            continue
        toks.append(t)
        # also index the parts of compound tokens: "balloon-control-api.service"
        if any(c in t for c in "._-/:") and not re.fullmatch(r"[\d.:/]+", t):
            toks.extend(p for p in re.split(r"[._\-/:]+", t) if p and p not in STOP)
    return toks


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(len(docs), 1)
        df = Counter(t for d in self.tf for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, q: list[str]) -> list[float]:
        out = []
        for tf, dl in zip(self.tf, self.len):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            out.append(s)
        return out


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def build(docs_dir: str, out_path: str, exclude=None, use_embeddings: bool = True,
          log=print) -> dict:
    root = Path(docs_dir)
    exclude = DEFAULT_EXCLUDE + list(exclude or [])
    files = sorted(p for p in root.rglob("*.md")
                   if not any(fnmatch.fnmatch(p.as_posix(), pat) for pat in exclude))
    chunks: list[Chunk] = []
    redactions: Counter = Counter()
    texts = {p: p.read_text(encoding="utf-8", errors="replace") for p in files}
    # pass 1: learn every labeled secret across all files (including excluded
    # credential files, so a password kept there is scrubbed everywhere else)
    known: set[str] = set()
    for p in root.rglob("*.md"):
        try:
            known |= find_secrets(texts.get(p) or p.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            pass
    for p in files:
        raw = texts[p]
        clean, hits = redact(raw, known)
        redactions.update(hits)
        chunks.extend(chunk_markdown(clean, p.relative_to(root).as_posix()))
    for i, c in enumerate(chunks):
        c.id = i
    log(f"{len(files)} files -> {len(chunks)} chunks; redacted {sum(redactions.values())} "
        f"secrets {dict(redactions) or ''}")

    vectors = None
    if use_embeddings:
        if ollama.available():
            t = time.time()
            vectors = ollama.embed([DOC_PREFIX + f"{c.heading}\n{c.text}" for c in chunks])
            log(f"embedded {len(vectors)} chunks with {ollama.EMBED_MODEL} in {time.time() - t:.0f}s")
        else:
            log("Ollama not reachable: building a keyword-only index (run ingest again later for embeddings)")

    index = {
        "built": time.strftime("%Y-%m-%d %H:%M"),
        "docs_dir": str(root.resolve()),
        "files": [p.relative_to(root).as_posix() for p in files],
        "fingerprint": hashlib.sha256("".join(c.text for c in chunks).encode()).hexdigest()[:12],
        "embed_model": ollama.EMBED_MODEL if vectors else None,
        "redactions": dict(redactions),
        "chunks": [asdict(c) for c in chunks],
        "vectors": vectors,
    }
    Path(out_path).write_text(json.dumps(index), encoding="utf-8")
    return index


class Index:
    def __init__(self, path: str):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.meta = {k: v for k, v in data.items() if k not in ("chunks", "vectors")}
        self.chunks = [Chunk(**c) for c in data["chunks"]]
        self.vectors = data.get("vectors")
        self.bm25 = BM25([tokenize(f"{c.source} {c.heading} {c.text}") for c in self.chunks])

    def search(self, query: str, k: int = 6, rrf_k: int = 60) -> list[dict]:
        """Return top-k hits with their bm25 / cosine scores and fused rank."""
        bm = self.bm25.scores(tokenize(query))
        bm_rank = sorted(range(len(bm)), key=lambda i: -bm[i])
        cos = None
        if self.vectors and ollama.available():
            qv = ollama.embed([QUERY_PREFIX + query], model=self.meta["embed_model"])[0]
            cos = [_cos(qv, v) for v in self.vectors]
            vec_rank = sorted(range(len(cos)), key=lambda i: -cos[i])
        fused = Counter()
        for r, i in enumerate(bm_rank[:100]):
            if bm[i] > 0:
                fused[i] += 1 / (rrf_k + r)
        if cos:
            for r, i in enumerate(vec_rank[:100]):
                fused[i] += 1 / (rrf_k + r)
        top = [i for i, _ in fused.most_common(k)]
        best_bm = bm[bm_rank[0]] if bm_rank else 0
        return [{
            "chunk": self.chunks[i],
            "bm25": bm[i],
            "bm25_rel": bm[i] / best_bm if best_bm else 0.0,
            "cosine": cos[i] if cos else None,
        } for i in top]
