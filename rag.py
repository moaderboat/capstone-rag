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

from capstone_rag import index as index_mod
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
    dbg = (lambda m: print(f"  · {m}")) if a.debug else None
    ans = ask(idx, a.question, k=a.k, model=a.model, debug=dbg)
    if ans.premise:
        print("\n⚑ The question's assumption doesn't match the docs (quote verified in source):")
    print("\n" + ans.text + "\n")
    if ans.rejected:
        from capstone_rag.answer import VERIFY_MODE
        verb = "removed" if VERIFY_MODE == "strict" else "couldn't verify"
        print(f"({verb} {len(ans.rejected)} statement(s)"
              + ("; run with --debug for reasons)" if not a.debug else ":)"))
        if a.debug:
            for c, n, why in ans.rejected:
                print(f"   ✗ {c}  [{n}]  ({why})")
    if ans.refused:
        print(f"(refused: {ans.reason})")
        if ans.reason.startswith("none of"):
            print("Closest sources:")
            for n, src, head in ans.sources[:4]:
                print(f"  [{n}] {src} — {head}")
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


def _is_hit(q, chunk):
    """A retrieved chunk counts if it contains the answer (answer_re), or else if it comes
    from one of the expected source files."""
    import re as _re
    if q.get("answer_re"):
        return bool(_re.search(q["answer_re"], f"{chunk.heading}\n{chunk.text}", _re.I))
    want = [q["source"]] if isinstance(q["source"], str) else q["source"]
    return any(x.lower() in chunk.source.lower() for x in want)


def _retrieval_scores(idx, answerable, k, hybrid, w=None):
    hits1 = hitk = rr = 0
    misses = []
    for q in answerable:
        res = idx.search(q["q"], k=k, use_vectors=hybrid, vec_weight=w)
        ranks = [i for i, h in enumerate(res, 1) if _is_hit(q, h["chunk"])]
        if ranks:
            hitk += 1
            hits1 += ranks[0] == 1
            rr += 1 / ranks[0]
        else:
            misses.append((q["q"], [h["chunk"].source for h in res[:3]]))
    n = len(answerable)
    return hits1 / n, hitk / n, rr / n, misses


def cmd_holdout(a):
    """Manual review: questions written after tuning, never used to tune. Prints each answer
    next to the ideal behavior so a human grades it."""
    idx = _load()
    if not ollama.available():
        sys.exit("Ollama isn't running.")
    qs = [json.loads(l) for l in (HERE / "evals" / "holdout.jsonl").read_text().splitlines() if l.strip()]
    for i, q in enumerate(qs, 1):
        ans = ask(idx, q["q"], model=a.model)
        print(f"\n[{i}/{len(qs)}] {q['q']}")
        print(f"  ideal:  {q['ideal']}")
        print(textwrap.indent(ans.text, "  answer: ").replace("\n  answer: ", "\n          ", 0))
        if ans.rejected:
            from collections import Counter
            why = Counter(r[2] for r in ans.rejected)
            from capstone_rag.answer import VERIFY_MODE
            verb = "removed" if VERIFY_MODE == "strict" else "unverified"
            print(f"  ({verb} {len(ans.rejected)}: " + ", ".join(f"{n}× {w}" for w, n in why.items()) + ")")
    print("\nGrade each one yourself: right / safe refusal / wrong.")


def cmd_eval(a):
    idx = _load()
    qs = [json.loads(l) for l in (HERE / "evals" / "questions.jsonl").read_text().splitlines() if l.strip()]
    answerable = [q for q in qs if q.get("source") or q.get("answer_re")]
    if a.sweep:
        if not (idx.vectors and ollama.available()):
            sys.exit("Sweep needs embeddings in the index and Ollama running.")
        print(f"Embedding-weight sweep ({len(answerable)} questions, set the winner with RAG_VEC_WEIGHT):")
        print("  weight  hit@1  hit@6   MRR")
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            h1, hk, mrr, _ = _retrieval_scores(idx, answerable, a.k, True, w)
            print(f"  {w:5.2f}   {h1:4.0%}   {hk:4.0%}   {mrr:.2f}" + ("   (keyword-only)" if w == 0 else ""))
        return
    hybrid = bool(idx.vectors) and index_mod.VEC_WEIGHT > 0 and not a.keyword_only and ollama.available()
    n = len(answerable)
    print(f"Retrieval: {n} questions...", flush=True)
    h1, hk, mrr, misses = _retrieval_scores(idx, answerable, a.k, hybrid)
    mode = f"hybrid, weight {index_mod.VEC_WEIGHT}" if hybrid else "keyword-only"
    print(f"Retrieval ({mode}, {n} questions, {len(idx.meta['files'])} files): "
          f"hit@1 {h1:.0%}  hit@{a.k} {hk:.0%}  MRR {mrr:.2f}")
    for m, got in misses:
        print(f"  miss: {m}\n        got: {', '.join(got)}")

    if not a.answers:
        return
    if not ollama.available():
        sys.exit("Ollama isn't running; can't grade answers.")
    print(f"\nGrading answers with {a.model or ollama.CHAT_MODEL} (first one loads the model)...", flush=True)
    correct = refused_ok = supported = 0
    errors = 0
    for q in qs:
        try:
            ans = ask(idx, q["q"], k=a.k, model=a.model)
        except Exception as e:  # one bad question shouldn't kill the run
            errors += 1
            print(f"  ERR {q['q']}  ({type(e).__name__}: {e})", flush=True)
            continue
        if q.get("source") or q.get("answer_re"):
            # each expected fact may list alternatives: "air|internet"
            need = [t.lower().split("|") for t in q.get("expect", [])]
            ok = not ans.refused and all(any(alt in ans.text.lower() for alt in t) for t in need)
            correct += ok
            supported += ans.supported
            flag = "ok " if ok else "BAD"
        else:
            ok = ans.refused or (q.get("allow_redacted") and "redacted" in ans.text.lower())
            refused_ok += bool(ok)
            flag = "ok " if ok else "HALLUCINATED"
        if ok and q.get("source") and not ans.supported:
            flag = "ok*"  # right facts, but missing or invalid citations
        print(f"  {flag} {q['q']}", flush=True)
        if (not ok or flag == "ok*") and a.verbose:
            print(textwrap.indent(ans.text, "      "))
            for c, src_n, why in ans.rejected:
                print(f"      ✗ unverified: {c}  [{src_n}] ({why})")
    un = len(qs) - n
    print("\n(ok* = right facts but missing or invalid citations)" + (f"; {errors} errored" if errors else ""))
    print(f"Verification mode: {__import__('capstone_rag.answer', fromlist=['VERIFY_MODE']).VERIFY_MODE}")
    print(f"Answers: {correct}/{n} contain the expected facts, {supported}/{n} fully cited; "
          f"{refused_ok}/{un} out-of-scope questions correctly refused.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest"); s.add_argument("docs")
    s.add_argument("--exclude", action="append", help="extra glob to skip (repeatable)")
    s.add_argument("--keyword-only", action="store_true"); s.set_defaults(fn=cmd_ingest)
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("-k", type=int, default=6)
    s.set_defaults(fn=cmd_search)
    s = sub.add_parser("ask"); s.add_argument("question"); s.add_argument("-k", type=int, default=8)
    s.add_argument("--model")
    s.add_argument("--debug", action="store_true", help="show retrieval, premise check and raw model output")
    s.set_defaults(fn=cmd_ask)
    s = sub.add_parser("eval"); s.add_argument("-k", type=int, default=6)
    s.add_argument("--answers", action="store_true"); s.add_argument("--model")
    s.add_argument("--keyword-only", action="store_true", help="skip embeddings, BM25 only")
    s.add_argument("--sweep", action="store_true", help="compare embedding weights 0-1")
    s.add_argument("-v", "--verbose", action="store_true"); s.set_defaults(fn=cmd_eval)
    s = sub.add_parser("holdout", help="answers to held-out questions, for manual grading")
    s.add_argument("--model"); s.set_defaults(fn=cmd_holdout)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
