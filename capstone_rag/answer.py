"""Answer a question from retrieved chunks, with citations and three guardrails:

1. Retrieval gate: if nothing relevant was found, refuse before calling the model.
2. Claim verification: the draft answer is split into claims; each claim is checked
   against only the source it cites, with a quote that must really appear there.
   Only verified claims are shown.
3. Premise check: a narrow JSON-only call asks whether a source explicitly disproves
   the question's assumption; the quoted evidence must really appear in that source.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from . import ollama
from .index import Index

NOT_FOUND = "NOT_IN_DOCS"
REFUSAL = "I couldn't find that in the capstone docs."
UNVERIFIED = ("I couldn't verify an answer from the capstone docs. The sources below are the "
              "closest matches; check them directly.")
# Coarse gate only: claim verification now catches irrelevant answers, so this just
# skips the model when nothing in the docs matches at all.
MIN_BM25 = float(os.environ.get("RAG_MIN_BM25", "2.5"))
MIN_COSINE = float(os.environ.get("RAG_MIN_COSINE", "0.55"))

SYSTEM = f"""You answer questions about David Moad's hospital IoT security capstone lab,
using ONLY the numbered sources provided.

Rules:
- Every factual sentence ends with the source number(s) it came from, like [2] or [1][3].
- If the sources don't contain the answer, reply with exactly: {NOT_FOUND}
  Exception: if the sources show the question's premise is false (the data, feature or
  connection it asks about doesn't exist), that IS an answer. Explain it; don't reply
  {NOT_FOUND}.
- Don't guess IPs, ports, commands, or values. Copy them exactly from the sources.
- If sources disagree, say so and prefer the most recent one (the field guide wins
  over older notes).
- Some values appear as [REDACTED]. Never try to guess them; say they are redacted.
- Match the exact symptom asked about. "No readings" is not "noisy readings"; "won't start"
  is not "runs slowly". If the sources only cover a related but different problem, say so
  plainly instead of presenting it as the answer.
- Check the question's assumptions against the sources before answering. Pipeline
  diagrams, architecture notes and "X → Y" data flows say where data does and doesn't go:
  if they show the thing the user describes doesn't exist (e.g. data sent to a tool it
  never reaches), say so plainly first, cite it, and point to where it actually is.
  But if the sources are merely silent about it, don't assume it's wrong; say the docs
  don't cover that part.
- For troubleshooting, list likely causes in the order the sources suggest checking them,
  each with its check or fix.
- Be brief: a few sentences or a short list of steps."""


PREMISE_CHECK = os.environ.get("RAG_PREMISE_CHECK", "1") != "0"

# Tools in the lab. A premise check only runs when the question names one of them.
TOOLS = re.compile(r"(?i)\b(grafana|node[- ]?red|wazuh|elastic\w*|kibana|postgres\w*|mirth|fhir|keycloak|"
                   r"vault|suricata|mqtt|mosquitto|truenas|influx\w*|splunk|opensearch)\b")

EXTRACT_SYSTEM = """Read the question. Does it assume some kind of data appears in, or comes from, a
specific tool? Reply JSON only:
{"data": "the kind of data, in a few words (e.g. 'pressure readings')",
 "tool": "the tool name exactly as written in the question"}
If the question doesn't assume that, reply {"data": null, "tool": null}."""

FLOW_SYSTEM = """You read data-flow diagrams. Answer ONE question: according to these sources, does
the given data go to the given tool?

Reply JSON only:
{"answer": "yes" | "no" | "unclear",
 "source": the source number that shows it, or null,
 "quote": "the line copied EXACTLY from that source showing where this data goes",
 "goes_to": "where that data actually goes, per the source (tool names only)"}

"no" only if a source shows where this data goes and the tool is not there.
If no source shows this data's path, answer "unclear"."""


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[`*_>|#]", " ", t)).strip().lower()


def quote_in_source(quote: str, source_text: str) -> bool:
    """Mechanical check that the model's quote really comes from the cited source."""
    q, src = _norm(quote), _norm(source_text)
    if len(q) < 8:
        return False
    if q in src:
        return True
    # tolerate small copy slips: 85%+ of the quote's words appear in order-insensitive form
    words = [w for w in re.findall(r"[\w.\-→]+", q) if len(w) > 1]
    src_words = set(re.findall(r"[\w.\-→]+", src))
    return len(words) >= 4 and sum(w in src_words for w in words) / len(words) >= 0.85


def _json_chat(system: str, user: str, model: str, debug, label: str) -> dict | None:
    """One narrow JSON call. A stalled or failed call counts as 'no result', never a crash."""
    try:
        raw = ollama.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                          model=model, temperature=0.0, json_mode=True)
    except ollama.OllamaUnavailable as e:
        debug and debug(f"{label}: call failed ({e}), treating as no result")
        return None
    debug and debug(f"{label}: {raw.strip()}")
    try:
        out = json.loads(raw)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        return None


def check_premise(index: Index, question: str, model: str, debug=None) -> dict | None:
    """Two narrow steps: (1) what data does the question expect in which tool?
    (2) do the data-flow docs say that data reaches that tool? Returns a verified
    false-premise finding, or None."""
    if not TOOLS.search(question):
        return None
    claim = _json_chat(EXTRACT_SYSTEM, f"Question: {question}", model, debug, "premise step 1 (claim)")
    if not claim or not claim.get("data") or not claim.get("tool"):
        return None
    data, tool = str(claim["data"]).strip(), str(claim["tool"]).strip()
    flows = index.flow_chunks(f"{data} {tool}")
    if not flows:
        debug and debug("premise: no data-flow sections found")
        return None
    if debug:
        for n, h in enumerate(flows, 1):
            debug(f"  flow source [{n}] {h['chunk'].source} — {h['chunk'].heading}")
    r = _json_chat(FLOW_SYSTEM, f"Sources:\n\n{format_context(flows)}\n\nDoes '{data}' go to '{tool}'?",
                   model, debug, "premise step 2 (flow check)")
    if not r or str(r.get("answer", "")).strip().strip(".").lower() != "no":
        return None
    n = _num(r.get("source"))
    if n is None or not 1 <= n <= len(flows):
        return None
    quote = str(r.get("quote", "")).strip()
    chunk = flows[n - 1]["chunk"]
    if not quote_in_source(quote, chunk.text):
        debug and debug("premise: quote NOT found in that source, ignoring")
        return None
    if tool.lower().replace(" ", "").replace("-", "") in _norm(quote).replace(" ", "").replace("-", ""):
        debug and debug("premise: the quoted line itself mentions the tool, so not a contradiction")
        return None
    debug and debug("premise: quote verified")
    goes_to = str(r.get("goes_to") or "").strip()
    text = (f"Your docs don't show {data} going to {tool} at all"
            + (f"; per {chunk.source}, it goes to {goes_to}" if goes_to else "")
            + f". Check there instead.\n\nFrom the source: \"{quote}\"")
    return {"source": chunk.source, "heading": chunk.heading, "quote": quote, "text": text}


# off: no claim checks | label (default): show all claims, mark unverified ones |
# strict: show only verified claims
VERIFY_MODE = {"0": "off", "1": "label"}.get(os.environ.get("RAG_VERIFY", "label"),
                                            os.environ.get("RAG_VERIFY", "label")).lower()
VERIFY = VERIFY_MODE != "off"
MAX_CLAIMS = 8

CLAIMS_SYSTEM = """Split an answer into its separate factual claims. Reply JSON only:
{"claims": [{"claim": "one self-contained factual statement", "source": the [n] number it cites, or null}]}
Keep commands, IPs, ports and file names exactly as written. Skip filler such as
"I hope this helps" or "further investigation may be needed", and skip statements about
the sources themselves ("source [8] contains a diagram", "the docs describe..."): only
claims about the lab. At most 8 claims."""

SUPPORT_SYSTEM = """You check one claim against one source, for one user question.

The claim is supported only if BOTH are true:
1. The source DIRECTLY states it. Related information, or something the claim merely could
   follow from, does not count. A claim the source contradicts is not supported.
2. What the source says applies to the SAME situation as the question: same device, service,
   connection or problem. A fix the source gives for a different problem (for example, a fix
   for an MQTT connection used to answer a question about Vault) is NOT supported.

Reply JSON only:
{"same_situation": true or false,
 "supported": true or false,
 "quote": "the exact text copied from the source that states it (empty if not supported)"}"""


def _yes(v) -> bool:
    """Small models write true, "true", "yes", "Yes." ... treat all as yes."""
    if isinstance(v, bool):
        return v
    return str(v).strip().strip(".").lower() in {"true", "yes", "y", "supported", "1"}


def _num(v):
    """Source numbers arrive as 3, "3", "[3]", [3] or "source 3" from small models."""
    if isinstance(v, list):
        v = v[0] if v else None
    if v is None or isinstance(v, bool):
        return None
    m = re.search(r"\d+", str(v))
    return int(m.group()) if m else None


# Claims that only describe the sources ("X is listed in [5]") carry no information.
_REFERS_TO_SOURCE = re.compile(r"(?i)\[\d+\]|\bsources?\b|\bdocs?\b|\bdocumentation\b")
_DESCRIBING = re.compile(r"(?i)\b(list|show|describe|mention|contain|provide|detail|document|include)\w*\b")


class _Meta:
    @staticmethod
    def search(text: str) -> bool:
        """A claim is 'meta' if it points at a source AND only says the source describes something."""
        return bool(_REFERS_TO_SOURCE.search(text) and _DESCRIBING.search(text))


META_CLAIM = _Meta()


def verify_claims(answer_text: str, hits: list[dict], model: str, debug=None,
                  question: str = "") -> tuple[list, list]:
    """Return (verified, rejected) lists of (claim, source_number, quote)."""
    r = _json_chat(CLAIMS_SYSTEM, f"Answer:\n{answer_text}", model, debug, "claims") or {}
    claims = r.get("claims") if isinstance(r.get("claims"), list) else []
    verified, rejected = [], []
    for c in claims[:MAX_CLAIMS]:
        if not isinstance(c, dict) or not str(c.get("claim", "")).strip():
            continue
        text = str(c["claim"]).strip()
        n = _num(c.get("source"))
        if META_CLAIM.search(text):
            rejected.append((text, n, "describes the source, not the lab"))
            continue
        candidates = [n] if n is not None and 1 <= n <= len(hits) else _best_sources(text, hits, 2)
        if not candidates:
            rejected.append((text, None, "no source mentions it"))
            continue
        result = None
        for cand in candidates:
            result = _check(text, cand, hits, model, debug, question)
            if result[0]:
                break
        ok, n, why_or_quote = result
        if ok:
            verified.append((text, n, why_or_quote))
        else:
            rejected.append((text, n, why_or_quote))
    return verified, rejected


def _best_sources(claim: str, hits: list[dict], k: int) -> list[int]:
    """For a claim the draft didn't cite, find which retrieved sections it most likely came from."""
    from .index import BM25, tokenize
    bm = BM25([tokenize(f"{h['chunk'].heading} {h['chunk'].text}") for h in hits])
    sc = bm.scores(tokenize(claim))
    ranked = sorted(range(len(hits)), key=lambda i: -sc[i])
    return [i + 1 for i in ranked[:k] if sc[i] > 0]


def _check(text, n, hits, model, debug, question):
    """Check one claim against source n. Returns (ok, n, quote_or_reason)."""
    src = hits[n - 1]["chunk"].text
    v = _json_chat(SUPPORT_SYSTEM, f"User question: {question}\n\nSource:\n{src}\n\nClaim: {text}",
                   model, debug, f"support [{n}]") or {}
    quote = str(v.get("quote", "")).strip()
    same = v.get("same_situation")
    if same is not None and not _yes(same):
        return False, n, "source is about a different situation"
    if not _yes(v.get("supported")):
        return False, n, "source doesn't state it"
    if not quote_in_source(quote, src):
        return False, n, "quote not found in source"
    return True, n, quote


@dataclass
class Answer:
    question: str
    text: str
    sources: list = field(default_factory=list)   # [(n, source, heading)]
    refused: bool = False
    reason: str = ""
    cited: list = field(default_factory=list)
    bad_citations: list = field(default_factory=list)
    premise: dict | None = None
    rejected: list = field(default_factory=list)   # claims removed by verification
    draft: str = ""                                # model's unverified answer

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


def ask(index: Index, question: str, k: int = 8, model: str = None, debug=None) -> Answer:
    hits = index.search(question, k=k)
    # Questions about how two tools relate are answered by data-flow diagrams, which keyword
    # search ranks poorly ("connected" rarely appears in them). Make sure the best ones are in.
    tools = {t.lower().replace(" ", "").replace("-", "") for t in TOOLS.findall(question)}
    if len(tools) >= 2:
        have = {id(h["chunk"]) for h in hits}
        extra = [f for f in index.flow_chunks(question, k=2) if id(f["chunk"]) not in have]
        if extra:
            hits = hits[: k - len(extra)] + extra
    if debug:
        debug("retrieved:")
        for n, h in enumerate(hits, 1):
            debug(f"  [{n}] bm25={h['bm25']:.1f}  {h['chunk'].source} — {h['chunk'].heading}")
    sources = [(n, h["chunk"].source, h["chunk"].heading) for n, h in enumerate(hits, 1)]
    ok, why = relevant_enough(hits)
    if not ok:
        return Answer(question, REFUSAL, sources, refused=True, reason=f"retrieval gate: {why}")

    model = model or ollama.CHAT_MODEL
    if PREMISE_CHECK:
        p = check_premise(index, question, model, debug)
        if p:
            return Answer(question, p["text"], [(1, p["source"], p["heading"])], cited=[1], premise=p)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Sources:\n\n{format_context(hits)}\n\nQuestion: {question}"},
    ]
    text = ollama.chat(messages, model=model).strip()
    debug and debug(f"answer model raw output: {text[:300]}")
    if NOT_FOUND in text:
        return Answer(question, REFUSAL, sources, refused=True, reason="model found no answer in sources")

    if not VERIFY:
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
        bad = [n for n in cited if not 1 <= n <= len(hits)]
        return Answer(question, text, sources, cited=cited, bad_citations=bad)

    verified, rejected = verify_claims(text, hits, model, debug, question)
    if VERIFY_MODE == "strict":
        if not verified:
            return Answer(question, UNVERIFIED, sources, refused=True, rejected=rejected, draft=text,
                          reason=f"none of the {len(rejected)} claims in the draft answer could be verified")
        body = "\n".join(f"- {c} [{n}]" for c, n, _ in verified)
    else:  # label: keep every claim, in the draft's order, marking the unverified ones
        order = {c: i for i, c in enumerate([c for c, _, _ in verified] + [c for c, _, _ in rejected])}
        rows = [(c, f"[{n}]") for c, n, _ in verified] + [(c, "⚠ unverified") for c, _, _ in rejected
                                                           if not META_CLAIM.search(c)]
        if not rows:
            return Answer(question, text, sources, draft=text, rejected=rejected)
        rows.sort(key=lambda r: order.get(r[0], 0))
        body = "\n".join(f"- {c} {tag}" for c, tag in rows)
    return Answer(question, body, sources, cited=sorted({n for _, n, _ in verified}),
                  rejected=rejected, draft=text)
