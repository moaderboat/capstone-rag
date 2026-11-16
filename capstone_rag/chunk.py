"""Split markdown into heading-scoped chunks.

Each chunk keeps its heading path ("Field Guide > 8. Troubleshooting > Vault sealed")
so retrieval and citations point somewhere a human can find. Lines starting
with '#' inside fenced code blocks are shell comments, not headings, and are
ignored as split points.
"""
import re
from dataclasses import dataclass, field

HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass
class Chunk:
    source: str          # path relative to the docs folder
    heading: str         # "H1 > H2 > H3"
    text: str
    id: int = 0
    meta: dict = field(default_factory=dict)


def _sections(md: str):
    """Yield (heading_path, body) pairs."""
    path: list[str] = []
    body: list[str] = []
    in_code = False
    for line in md.splitlines():
        if FENCE.match(line):
            in_code = not in_code
            body.append(line)
            continue
        m = None if in_code else HEADING.match(line)
        if m:
            if any(b.strip() for b in body):
                yield " > ".join(path), "\n".join(body).strip()
            level, title = len(m.group(1)), m.group(2).strip()
            path = path[: level - 1] + [title]
            body = []
        else:
            body.append(line)
    if any(b.strip() for b in body):
        yield " > ".join(path), "\n".join(body).strip()


def _split_long(text: str, max_chars: int, overlap: int):
    """Split on paragraph boundaries; hard-split paragraphs that are still too long."""
    if len(text) <= max_chars:
        return [text]
    paras = re.split(r"\n\s*\n", text)
    out, cur = [], ""
    for p in paras:
        while len(p) > max_chars:
            if cur:
                out.append(cur)
                cur = ""
            out.append(p[:max_chars])
            p = p[max_chars - overlap:]
        if len(cur) + len(p) + 2 > max_chars and cur:
            out.append(cur)
            cur = cur[-overlap:] + "\n\n" + p if overlap else p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur.strip():
        out.append(cur)
    return out


def chunk_markdown(md: str, source: str, max_chars: int = 1800, overlap: int = 200,
                   min_chars: int = 40) -> list[Chunk]:
    chunks = []
    for heading, body in _sections(md):
        for piece in _split_long(body, max_chars, overlap):
            if len(piece.strip()) >= min_chars:
                chunks.append(Chunk(source=source, heading=heading, text=piece.strip()))
    return chunks
