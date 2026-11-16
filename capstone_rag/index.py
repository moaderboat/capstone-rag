"""Build, save, load and search the index.

Hybrid retrieval: BM25 (exact terms like IPs, ports, service names) fused with
embedding similarity (paraphrased questions) using reciprocal rank fusion.
If Ollama isn't running, search falls back to BM25 alone.
"""
import fnmatch
import os
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
# How much the embedding ranking counts relative to BM25 in the fused score.
# 1.0 = equal; 0 = keyword-only. Default 0.25, chosen by eval --sweep (README). Tune with: python rag.py eval --sweep
VEC_WEIGHT = float(os.environ.get("RAG_VEC_WEIGHT", "0.25"))  # see README: recall vs. rank trade-off


# Small domain thesaurus: users describe symptoms in words the runbooks don't use.
SYNONYMS = {
    "readings": ["data", "values", "reads", "gauges"], "reading": ["data", "value", "reads", "gauge"],
    "data": ["readings"], "sensor": ["gauge", "reads"], "sensors": ["gauges", "reads"],
    "gauge": ["readings"], "gauges": ["readings"],
    "broken": ["not", "failing", "fix"], "down": ["not", "failing"], "dead": ["not", "down"],
    "missing": ["no", "not"], "nothing": ["no"], "empty": ["no"],
    "pressure": ["balloon", "bottle", "psi"], "bottle": ["balloon", "pressure"], "balloon": ["bottle"],
    "dashboard": ["grafana", "node-red", "ui"], "graph": ["dashboard", "grafana"],
    "grafana": ["dashboard"], "node-red": ["dashboard"],
    "heart": ["heartrate", "hr"], "heartrate": ["hr", "heart"], "vitals": ["heartrate", "hr"],
    "ids": ["suricata"], "siem": ["wazuh"], "sso": ["keycloak"], "login": ["keycloak", "sso"],
    "secrets": ["vault"], "sealed": ["unseal", "vault"], "firewall": ["udm", "rule"],
    "reboot": ["restart", "power"], "restart": ["reboot", "systemctl"],
}
SYMPTOM_Q = re.compile(r"(?i)\b(no|not|isn'?t|aren'?t|won'?t|can'?t|doesn'?t|don'?t|broken|missing|nothing|"
                       r"fail\w*|error\w*|stuck|down|dead|empty|wrong)\b")
FIX_SECTION = re.compile(r"(?i)troubleshoot|common issues|issue:|symptom|problem|fixed|resolved|fix:|"
                         r"root cause|not (working|receiving|showing|loading|publishing)|shows no")
SYMPTOM_BOOST = 1.6  # symptom-style questions favor troubleshooting/fix sections
HEADING_BOOST = 3  # heading words count 3x: runbook headings name the symptom


SYNONYM_WEIGHT = 0.35  # synonyms help, but must never outweigh the words the user typed


def expand(tokens: list[str]) -> list[tuple[str, float]]:
    """Query terms with weights: typed words 1.0, synonyms SYNONYM_WEIGHT."""
    out = [(t, 1.0) for t in tokens]
    typed = set(tokens)
    for t in tokens:
        out.extend((syn, SYNONYM_WEIGHT) for syn in SYNONYMS.get(t, []) if syn not in typed)
    return out


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

    PREFIX_WEIGHT = 0.7

    def _with_prefixes(self, q):
        """Partial words ("postgre", "keycl") match indexed words that start with them."""
        out = []
        for t, w in q:
            out.append((t, w))
            if t not in self.idf and len(t) >= 4 and not t[0].isdigit():
                matches = [v for v in self.idf if v.startswith(t)]
                for v in sorted(matches, key=lambda v: -self.idf[v])[:5]:
                    out.append((v, w * self.PREFIX_WEIGHT))
        return out

    def scores(self, q) -> list[float]:
        """q: list of terms, or of (term, weight) pairs."""
        q = [(t, 1.0) if isinstance(t, str) else t for t in q]
        q = self._with_prefixes(q)
        out = []
        for tf, dl in zip(self.tf, self.len):
            s = 0.0
            for t, w in q:
                f = tf.get(t)
                if f:
                    s += w * self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
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
        self._qcache = {}
        self._is_fix = [bool(FIX_SECTION.search(c.heading) or FIX_SECTION.search(c.text[:300]))
                        for c in self.chunks]
        self.bm25 = BM25([tokenize(f"{c.source} {c.text}") + tokenize(c.heading) * HEADING_BOOST
                          for c in self.chunks])

    FLOW = re.compile(r"→|->|──▶|=>")

    def flow_chunks(self, query: str, k: int = 4) -> list[dict]:
        """Best-matching chunks that contain data-flow arrows (pipeline / architecture notes)."""
        bm = self.bm25.scores(expand(tokenize(query + " pipeline data flow architecture")))
        idx = [i for i in sorted(range(len(bm)), key=lambda i: -bm[i])
               if bm[i] > 0 and len(self.FLOW.findall(self.chunks[i].text)) >= 2][:k]
        return [{"chunk": self.chunks[i], "bm25": bm[i], "bm25_rel": 0.0, "cosine": None} for i in idx]

    def query_vector(self, query: str):
        if not (self.vectors and ollama.available()):
            return None
        if query not in self._qcache:
            self._qcache[query] = ollama.embed([QUERY_PREFIX + query], model=self.meta["embed_model"])[0]
        return self._qcache[query]

    def search(self, query: str, k: int = 6, rrf_k: int = 60, use_vectors: bool = True,
               vec_weight: float = None) -> list[dict]:
        """Return top-k hits with their bm25 / cosine scores and fused rank."""
        w = VEC_WEIGHT if vec_weight is None else vec_weight
        bm = self.bm25.scores(expand(tokenize(query)))
        if SYMPTOM_Q.search(query):
            bm = [b * SYMPTOM_BOOST if self._is_fix[i] else b for i, b in enumerate(bm)]
        bm_rank = sorted(range(len(bm)), key=lambda i: -bm[i])
        cos = None
        qv = self.query_vector(query) if use_vectors and w > 0 else None
        if qv is not None:
            cos = [_cos(qv, v) for v in self.vectors]
            vec_rank = sorted(range(len(cos)), key=lambda i: -cos[i])
        fused = Counter()
        for r, i in enumerate(bm_rank[:100]):
            if bm[i] > 0:
                fused[i] += 1 / (rrf_k + r)
        if cos:
            for r, i in enumerate(vec_rank[:100]):
                fused[i] += w / (rrf_k + r)
        top = [i for i, _ in fused.most_common(k)]
        best_bm = bm[bm_rank[0]] if bm_rank else 0
        return [{
            "chunk": self.chunks[i],
            "bm25": bm[i],
            "bm25_rel": bm[i] / best_bm if best_bm else 0.0,
            "cosine": cos[i] if cos else None,
        } for i in top]
