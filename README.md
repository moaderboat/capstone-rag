# capstone-rag

Ask plain-English questions about my hospital IoT security capstone lab and get
answers **with citations**, from a model running **entirely on my own machine**.

Illustrative example:

```
> python rag.py ask "The Vulnerable Pi can't reach MQTT after a gateway reboot. What fixed it last time?"

The UDM-SE firewall rules weren't reloaded after the reboot, so traffic from the
Vulnerable Pi (192.168.50.100) to the broker at 10.30.0.51:1883 was dropped [1]...

Sources:
  [1] TROUBLESHOOTING_HISTORY.md — UDM-SE Firewall Rules Not Loaded After Reboot
```

The lab has 50+ markdown runbooks, build logs and troubleshooting notes written
over a year. Finding "what did I do last time Mirth lost its connection" by hand
is slow, especially at a conference booth with the demo down.

## How it works

```
 docs/*.md ──► redact secrets ──► chunk by heading ──► BM25 index ─┐
                                                   └► embeddings ──┤  (nomic-embed-text via Ollama)
 question ─────────────────────────────────────────────────────────┴► hybrid search (RRF)
          ──► retrieval gate ──► local LLM with numbered sources ──► citation check ──► answer
```

| Design choice | Why |
|---|---|
| **Local model (Ollama)** | Lab docs describe a hospital network: IPs, firewall rules, credentials. Nothing leaves the machine. |
| **Secrets redacted at ingest** | Lab notes contained plaintext passwords and a Vault root token. Redaction runs before anything is indexed or sent to a model, and any password found next to a label is then scrubbed everywhere it appears. In testing on 24 of the lab docs it removed 259 secrets, and a check confirmed none survived in the index. |
| **Hybrid retrieval** | Keyword search (BM25) nails exact strings like `10.30.0.54` or `balloon-control-api.service`; embeddings catch paraphrased questions. Fused with reciprocal rank fusion. Falls back to keyword-only if Ollama is down. |
| **Heading-aware chunking** | Each chunk keeps its section path, so citations point somewhere a person can actually go read. `#` comments inside code blocks aren't mistaken for headings. |
| **Two-stage refusal** | A retrieval gate refuses before calling the model when nothing relevant was found; the model is told to answer `NOT_IN_DOCS` otherwise. Out-of-scope questions get "I couldn't find that", not a guess. |
| **Citation check** | Every `[n]` must point at a real source. Answers with no citations or bogus ones are flagged as unverified. |
| **Eval set** | `evals/questions.jsonl`: questions with known source docs and expected facts, plus out-of-scope questions that must be refused. |

## Setup (Windows)

1. Install [Ollama](https://ollama.com/download), then pull the models:
   ```
   ollama pull nomic-embed-text
   ollama pull llama3.1:8b
   ```
2. Python 3.10+ only; no packages to install.
3. Build the index (redacts, chunks and embeds; a few minutes the first time):
   ```
   python rag.py ingest "C:\Users\david\OneDrive\Documents\Fall 25 school\Senior Design I"
   ```

## Use

```
python rag.py ask "How do I unseal Vault after a power loss?"
python rag.py search "suricata port mirroring"     # retrieval only, no model
python rag.py ask "..." --model qwen2.5:7b          # compare models
```

Settings via environment variables: `RAG_CHAT_MODEL`, `RAG_EMBED_MODEL`, `OLLAMA_HOST`,
`RAG_MIN_BM25`, `RAG_MIN_COSINE`.

## Evaluate

```
python rag.py eval                 # retrieval: hit@1, hit@6, MRR
python rag.py eval --answers -v    # also grade answers, citations and refusals
python -m unittest discover tests  # redaction, chunking, ranking tests
```

Keyword-only baseline on the capstone docs (17 answerable questions):
**hit@1 76%, hit@6 100%, MRR 0.85**. Record hybrid and answer-level results here
after running with Ollama:

| Run | hit@1 | hit@6 | MRR | Answers correct | Fully cited | Out-of-scope refused |
|---|---|---|---|---|---|---|
| keyword-only | 76% | 100% | 0.85 | – | – | – |
| hybrid + llama3.1:8b | | | | | | |
| hybrid + qwen2.5:7b | | | | | | |

## Known limits

- The keyword gate alone can't tell an in-domain question with no answer ("what GPU is in
  the PowerEdge?") from a real one; the model's `NOT_IN_DOCS` step does that, and the
  eval measures how often it works.
- Redaction is pattern-based and errs toward over-masking (a username or two gets
  masked along with passwords). It reduces exposure; it isn't a substitute for moving
  credentials into a secrets manager.
- `index.json` holds redacted copies of the docs. It's git-ignored; don't publish it.
