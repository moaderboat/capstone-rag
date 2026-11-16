#!/usr/bin/env python3
"""capstone-rag: ask questions about the capstone lab docs, answered by a local model.

  python rag.py ingest "C:/path/to/Senior Design I"
  python rag.py ask "How do I unseal Vault after a restart?"
  python rag.py search "suricata port mirror"        # retrieval only, no model
  python rag.py eval                                  # retrieval metrics
  python rag.py eval --answers                        # also grade model answers
"""
import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

from capstone_rag import ollama
from capstone_rag.answer import ask
from capstone_rag.index import Index, build

HERE = Path(__file__).parent
INDEX = HERE / "index.json"


def _load():
    if not INDEX.exists():
        sys.exit("No index yet. Run: python rag.py ingest <docs folder>")
    return Index(str(INDEX))


def cmd_ingest(a):
    build(a.docs, str(INDEX), exclude=a.exclude, use_embeddings=not a.keyword_only)
    print(f"Index written to {INDEX}")


def cmd_search(a):
    idx = _load()
    for n, h in enumerate(idx.search(a.query, k=a.k), 1):
        c = h["chunk"]
        cos = f" cos={h['cosine']:.2f}" if h["cosine"] is not None else ""
        print(f"[{n}] bm25={h['bm25']:.1f}{cos}  {c.source} — {c.heading}")
        print(textwrap.indent(textwrap.shorten(c.text, 220), "     "))


def cmd_ask(a):
    idx = _load()
    if not ollama.available():
        sys.exit(f"Ollama isn't running at {ollama.HOST}. Start it, or use `search` for retrieval only.")
    t = time.time()
    ans = ask(idx, a.question, k=a.k, model=a.model)
    print("\n" + ans.text + "\n")
    if ans.refused:
        print(f"(refused: {ans.reason})")
    else:
        print("Sources:")
        for n, src, head in ans.sources:
            if n in ans.cited:
                print(f"  [{n}] {src} — {head}")
        if not ans.cited:
            print("  ⚠ The model gave no citations. Treat this answer as unverified.")
        if ans.bad_citations:
            print(f"  ⚠ Cited sources that don't exist: {ans.bad_citations}")
    print(f"\n({time.time() - t:.1f}s, {a.model or ollama.CHAT_MODEL})")


def cmd_eval(a):
    idx = _load()
    qs = [json.loads(l) for l in (HERE / "evals" / "questions.jsonl").read_text().splitlines() if l.strip()]
    answerable = [q for q in qs if q.get("source")]
    hits1 = hitk = rr = 0
    misses = []
    for q in answerable:
        res = idx.search(q["q"], k=a.k)
        want = [q["source"]] if isinstance(q["source"], str) else q["source"]
        ranks = [i for i, h in enumerate(res, 1)
                 if any(w.lower() in h["chunk"].source.lower() for w in want)]
        if ranks:
            hitk += 1
            hits1 += ranks[0] == 1
            rr += 1 / ranks[0]
        else:
            misses.append(q["q"])
    n = len(answerable)
    mode = "hybrid" if idx.vectors and ollama.available() else "keyword-only"
    print(f"Retrieval ({mode}, {n} questions): hit@1 {hits1/n:.0%}  hit@{a.k} {hitk/n:.0%}  MRR {rr/n:.2f}")
    for m in misses:
        print(f"  miss: {m}")

    if not a.answers:
        return
    if not ollama.available():
        sys.exit("Ollama isn't running; can't grade answers.")
    correct = refused_ok = supported = 0
    for q in qs:
        ans = ask(idx, q["q"], k=a.k, model=a.model)
        if q.get("source"):
            need = [t.lower() for t in q.get("expect", [])]
            ok = not ans.refused and all(t in ans.text.lower() for t in need)
            correct += ok
            supported += ans.supported
            flag = "ok " if ok else "BAD"
        else:
            ok = ans.refused or (q.get("allow_redacted") and "redacted" in ans.text.lower())
            refused_ok += bool(ok)
            flag = "ok " if ok else "HALLUCINATED"
        print(f"  {flag} {q['q']}")
        if not ok and a.verbose:
            print(textwrap.indent(ans.text, "      "))
    un = len(qs) - n
    print(f"\nAnswers: {correct}/{n} contain the expected facts, {supported}/{n} fully cited; "
          f"{refused_ok}/{un} out-of-scope questions correctly refused.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest"); s.add_argument("docs")
    s.add_argument("--exclude", action="append", help="extra glob to skip (repeatable)")
    s.add_argument("--keyword-only", action="store_true"); s.set_defaults(fn=cmd_ingest)
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("-k", type=int, default=6)
    s.set_defaults(fn=cmd_search)
    s = sub.add_parser("ask"); s.add_argument("question"); s.add_argument("-k", type=int, default=6)
    s.add_argument("--model"); s.set_defaults(fn=cmd_ask)
    s = sub.add_parser("eval"); s.add_argument("-k", type=int, default=6)
    s.add_argument("--answers", action="store_true"); s.add_argument("--model")
    s.add_argument("-v", "--verbose", action="store_true"); s.set_defaults(fn=cmd_eval)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
