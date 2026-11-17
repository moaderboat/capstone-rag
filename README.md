# capstone-rag

Ask plain-English questions about my hospital IoT security capstone lab and get
answers **with citations**, from a model running **entirely on my own machine**.

Example (from the held-out test set; the model's false claim is caught and marked):

```
> python rag.py ask "is node red connected to grafana"

- Node-RED provides real-time monitoring at http://10.30.0.52:1880/ui [2]
- Grafana complements Node-RED with long-term historical analysis and alerting [2]
- Node-RED is connected to Grafana ⚠ unverified
```

The lab has 50+ markdown runbooks, build logs and troubleshooting notes written
over a year. Finding "what did I do last time Mirth lost its connection" by hand
is slow, especially at a conference booth with the demo down.

## How it works

```
 docs/*.md ─► redact secrets ─► chunk by heading ─► BM25 index + embeddings (nomic-embed-text)

 question ─► hybrid search (BM25 + 0.25 × embeddings, RRF) ─► 8 sources
          ─► premise check: does a data-flow doc show the data never reaches the named tool?
          ─► local LLM drafts an answer from the numbered sources
          ─► claim verification: split into claims; each is checked against ONE source,
             with the same question, and must come with a quote that really appears there
          ─► verified claims cite their source; the rest are marked ⚠ unverified
```

| Design choice | Why |
|---|---|
| **Local model (Ollama)** | Lab docs describe a hospital network: IPs, firewall rules, credentials. Nothing leaves the machine. |
| **Secrets redacted at ingest** | Lab notes contained plaintext passwords and a Vault root token. Redaction runs before anything is indexed or sent to a model, and a password found next to a label is scrubbed everywhere it appears. 366 secrets removed from 51 docs; a check confirmed none survived in the index. |
| **Hybrid retrieval, tuned by measurement** | Keyword search nails exact strings (`10.30.0.54`); embeddings find paraphrased questions. The weight was chosen by sweeping it on two eval rounds (see below). Partial words match ("postgre" → postgresql); synonyms count at a third of a typed word. |
| **Heading-aware chunking** | Each chunk keeps its section path, so citations point somewhere a person can go read. Section titles count 3× in ranking, since runbook headings name the symptom. |
| **Per-claim verification** | A citation alone doesn't prove anything: early versions cited sources that didn't say what the answer claimed. Each claim is checked against only the source it came from (uncited claims are matched to their likeliest source), including whether that source is about the *same situation* as the question. The model must quote its evidence, and code confirms the quote exists. |
| **Label, don't delete** | Deleting unverified claims (strict mode) blocked correct answers. Default mode shows everything but marks what couldn't be verified. |
| **Robust to small-model JSON** | 8B models write `"3"`, `"[3]"`, `"yes"`, or stall in JSON mode. Parsing tolerates all of these, output length is capped, and a stalled check counts as "unverified", never a crash. Covered by unit tests. |
| **Refusal** | Out-of-scope questions ("what was the Bengals score?", "where's the oxygen chart?") get "I couldn't find that", not a guess. |

## Try it yourself

The docs it answers from are public in
[Hospital-IoT-HomeLab](https://github.com/moaderboat/Hospital-IoT-HomeLab)
(secrets were scrubbed before publishing, and ingest redacts again). Runs on Windows, macOS or Linux.

1. Install [Ollama](https://ollama.com/download), then pull the models
   (about 5 GB; a GPU with 8 GB+ VRAM is comfortable, CPU-only works but is slow):
   ```
   ollama pull nomic-embed-text
   ollama pull llama3.1:8b
   ```
2. Install Python 3.10+. There are no packages to install.
3. Clone both repos side by side and build the index (a few minutes the first time):
   ```
   git clone https://github.com/moaderboat/capstone-rag.git
   git clone https://github.com/moaderboat/Hospital-IoT-HomeLab.git
   cd capstone-rag
   python rag.py ingest ../Hospital-IoT-HomeLab
   ```
   No GPU? `python rag.py ingest ../Hospital-IoT-HomeLab --keyword-only`, then use
   `python rag.py search "..."` for retrieval without a model.

To point it at your own docs instead, ingest any folder of markdown files.

## Use

```
python rag.py ask "How do I unseal Vault after a power loss?"
python rag.py search "suricata port mirroring"     # retrieval only, no model
python rag.py ask "..." --model qwen2.5:7b          # compare models
```

Add `--debug` to see retrieval, the premise check, every claim and why it was or wasn't verified.

Settings (environment variables): `RAG_VERIFY` = `label` (default) / `strict` / `0`,
`RAG_VEC_WEIGHT` (default 0.25), `RAG_CHAT_MODEL`, `RAG_EMBED_MODEL`, `OLLAMA_HOST`.

## Evaluate

```
python rag.py eval                 # retrieval: hit@1, hit@6, MRR
python rag.py eval --sweep         # compare embedding weights
python rag.py eval --answers -v    # also grade answers, citations and refusals
python rag.py holdout              # questions written after tuning, graded by hand
python -m unittest discover tests  # redaction, chunking, ranking tests
```

Results on my full private capstone folder (51 docs, 1,548 chunks, 366 secrets redacted;
the public repo has 50 docs, 1,515 chunks),
llama3.1:8b on an RTX 5080:

**Retrieval.** Embedding weight swept from 0 (keyword only) to 1 (equal hybrid).

Round 1: 17 questions, mostly worded the way the docs are:

| Weight | hit@1 | hit@6 | MRR |
|---|---|---|---|
| 0.00 | **76%** | **100%** | **0.85** |
| 0.25 | 71% | 94% | 0.79 |
| 1.00 | 71% | 94% | 0.79 |

Round 2: 26 questions, adding natural, paraphrased questions written after round 1
("where do the VMs run", "what VMs are connected to postgre"), scored by whether the
retrieved text contains the answer:

| Weight | hit@1 | hit@6 | MRR |
|---|---|---|---|
| 0.00 | **73%** | 92% | **0.80** |
| **0.25 (chosen)** | 65% | **96%** | 0.76 |
| 0.50 | 62% | 96% | 0.75 |
| 1.00 | 58% | 96% | 0.75 |

Round 1 said embeddings hurt, but that eval was biased toward exact wording. With
real paraphrased questions, embeddings find sections keyword search misses (top-6
recall up), at the cost of ranking (hit@1 down). Since the model reads 8 sources and
every claim is verified, **recall matters more than rank 1**, so the default is 0.25.
With 26 questions, one question is about 4 points, so these are directional.

**Answers** (26 questions with known answers + 6 that must be refused), llama3.1:8b:

| Verification | Correct facts | Fully cited | Out-of-scope refused |
|---|---|---|---|
| off (citation check only) | 23/26 | 21/26 | 6/6 |
| **label (default)** | **24/26** | 23/26 | **6/6** |
| strict (verified claims only) | 22/26 | **26/26** | 6/6 |

**Held-out questions** (`evals/holdout.jsonl`, written by me after tuning, mostly false
premises such as "why can't I see the pressure readings in Grafana" when pressure never goes
to Grafana). Before claim verification: 3 of 8 answered confidently and wrongly. After: 0
wrong statements presented as fact. False claims the model still drafts appear marked
⚠ unverified, and the rest are correct or safe refusals.

What didn't work, and why it changed:
- **Prompt rules alone** ("check the question's assumptions") failed on both llama3.1:8b and
  qwen2.5:14b; a narrow, separate check works better than one long instruction list.
- **Strict verification v1** dropped correct answers from 23/26 to about 13/26 because the
  claim splitter lost citations; fixed by matching uncited claims to their likeliest source.
- **Round-1 retrieval eval** used the docs' own wording and favored keyword search; natural
  questions reversed that conclusion.

## Known limitations

- **Slow:** verification costs one model call per claim; long answers take 20–30 s on an RTX 5080.
- **Only as good as the docs:** if something was never written down, it can't be answered.
  Asking questions surfaces documentation gaps, which is useful in its own right.
- **Wrong assumptions are marked, not always explained:** for "pressure in Grafana" it marks
  the false claim but doesn't always say "that data never goes there"; the premise check
  catches only data-to-tool assumptions.
- **Small eval:** 26 + 8 questions; one question moves a score by about 4 points, and
  8B output varies between runs. Treat the numbers as directional.
- **Redaction is pattern-based** and errs toward over-masking. It reduces exposure; it
  isn't a substitute for moving credentials into a secrets manager.
- `index.json` holds redacted copies of the docs. It's git-ignored; don't publish it.

## License

MIT. See `LICENSE`.
