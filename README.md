# Mutual Fund Facts Assistant (facts-only RAG)

A small FAQ assistant that answers **factual** questions about five HDFC Mutual Fund
schemes — expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark,
NAV, and how to download statements. Every answer carries exactly one source link.
The assistant **refuses** advice and performance questions and **never** accepts or
stores personal data.

> **Facts-only. No investment advice.**

---

## Quick start

No third-party packages are required. Python 3.11+ is enough.

```bash
cd RAGproject

# 1. Build the index  (stages 1-4: fetch -> parse -> chunk -> index)
python run.py build

# 2. Ask from the terminal
python run.py ask "What is the expense ratio of HDFC Large Cap Fund?"
python run.py ask "Is there a lock-in period in HDFC ELSS Tax Saver Fund?"
python run.py ask "How do I download a capital gains statement?"

# 3. Run the tiny web UI  ->  http://127.0.0.1:8000
python run.py serve

# 4. Run the evaluation suite (28 checks) and the unit tests (204 tests)
python run.py eval
python -m unittest discover -s tests
```

Other commands:

| Command | Purpose |
|---|---|
| `python run.py build --refresh` | ignore the raw cache and refetch all sources |
| `python run.py sources` | print the source registry / citation allow-list |
| `python run.py sources --write` | regenerate `SOURCES.md` and `SOURCES.csv` |
| `python run.py samples` | regenerate `samples/sample_qa.{json,md}` from the live pipeline |
| `python run.py stage5 "<query>"` | show only the stage-5 intent-gate decision |
| `python run.py stage6 "<query>"` | show only the stage-6 hybrid retrieval ranking |
| `python --debug run.py ask "..."` | print the gate decision, retrieval ranking and guards |
| `python bench.py` | latency probe |

### Optional: LLM answer rewriting

The default answer path is **deterministic and offline** — no API key, no network at
query time. Set a key to add an LLM rewrite pass over the retrieved context:

```bash
set MF_FAQ_LLM=1
set OPENAI_API_KEY=...          # or ANTHROPIC_API_KEY
python run.py ask "..."
```

The LLM is **not** trusted. Its output goes through the same stage-8 guardrails as
the offline path, and any answer that fails a check is replaced by a refusal or
repaired. Without a key the system behaves identically, just with less fluent prose.

### Requirements

* Python 3.11+
* `numpy` — optional, used only to accelerate the vector index
* Network access for `python run.py build` (the raw cache in `data/raw/` makes
  rebuilds offline afterwards)

See `requirements.txt`.

---

## Scope

**AMC:** HDFC Mutual Fund (HDFC Asset Management Company). One AMC by design — the
assistant cannot compare across AMCs, and that limitation is a feature, not a gap.

**Schemes (direct-growth variants):**

| Scheme | Category | Source |
|---|---|---|
| HDFC Large Cap Fund (Direct - Growth) | Large cap | `groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth` |
| HDFC Flexi Cap Fund (Direct - Growth) | Flexi cap | `groww.in/mutual-funds/hdfc-equity-fund-direct-growth` |
| HDFC ELSS Tax Saver Fund (Direct Plan - Growth) | ELSS / tax-saver | `groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth` |
| HDFC Small Cap Fund (Direct - Growth) | Small cap | `groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth` |
| HDFC Balanced Advantage Fund (Direct - Growth) | Hybrid | `groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth` |

Full registry with roles and verification status: **`SOURCES.md`** / **`SOURCES.csv`**.

---

## Architecture — one module per RAG stage

Each stage is a separate, independently runnable module so it can be inspected and
evaluated on its own. The numbers below are the pipeline's own stage IDs.

```
                 ┌──────────── offline build (python run.py build) ────────────┐
 STAGE 0         │ sources.py     registry = hard citation allow-list          │
 source registry │                                                              │
                 ▼                                                              │
 STAGE 1         │ ingest.py      fetch -> data/raw/*.html (cached, hashed)   │
 data ingestion  │                + data/fetch_manifest.json (provenance)     │
                 ▼                                                              │
 STAGE 2         │ parsing.py     HTML -> visible text (stdlib HTMLParser)     │
 parsing         │                + __NEXT_DATA__ JSON -> normalised facts      │
                 ▼                                                              │
 STAGE 3         │ chunking.py    3 chunk families: fact | section | procedure │
 chunking        │                                                              │
                 ▼                                                              │
 STAGE 4         │ indexing.py    SQLite FTS5 BM25  +  hashed dense vectors    │
 indexing        │                                                              │
                 └──────────────────────────────────────────────────────────────┘
                                              │  data/index/mf_faq.sqlite
                                              ▼
                 ┌──────────────────── online (python run.py ask / serve) ────┐
 STAGE 5         │ intent.py       PII scrub -> scheme resolve -> topic route    │
 query           │                 -> ADVICE / PERFORMANCE / PII refusal gate     │
 understanding   ▼                                                                 │
 STAGE 6         │ retrieval.py    hybrid: BM25 + dense cosine + scheme boost    │
 data retrieval  │                 + topic boost + field boost + kind gate, MMR  │
                 ▼                                                                 │
 STAGE 7         │ synthesis.py    deterministic extractive composer (default)  │
 answer synth    │                 or LLM rewrite, same constraints               │
                 ▼                                                                 │
 STAGE 8         │ guardrails.py   G1 PII  G2 citation  G3 <=3 sentences         │
 post-processing │                 G4 advice  G5 performance  G6 freshness  G7    │
                 ▼                                                                 │
 STAGE 9         │ eval.py         13 golden + 15 policy cases, 28 checks        │
 evaluation      │ tests/          204 unit + end-to-end tests                    │
                 └──────────────────────────────────────────────────────────────┘
```

Deeper notes on the design decisions are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

### Design decisions worth flagging

1. **Chunk shape follows question shape.** "What is the exit load?" wants one number,
   not a paragraph. So 70 of 114 chunks are single-fact chunks whose text is a
   complete, self-contained sentence carrying the scheme name and the unit. The
   citation is therefore verifiable by reading the chunk.

2. **Two indexes, because financial jargon defeats either one alone.** BM25 is exact
   and self-explaining for "ELSS", "exit load", "riskometer"; the hashed dense index
   (IDF-weighted word unigrams + char 4-grams, cosine) absorbs morphology and
   paraphrase. Both are inspectable — no black box.

3. **A field hint is a hard constraint, not a preference.** "Minimum SIP" and
   "minimum lump-sum" are different numbers under the same topic, and "who manages X"
   does not lexically match a chunk saying "fund manager". A soft score boost was not
   enough — the wrong fact won on raw term count — so stage 5 emits `field_hints` that
   stage 7 treats as a filter.

4. **The scraper drops performance data on purpose.** The page payload contains
   return figures. `normalise_facts()` never copies them into the corpus, so they
   cannot reach an answer even if a prompt were injected.

5. **The default generator is not a language model.** A facts-only assistant whose
   numbers come from a source of record is safer when it cannot paraphrase a number.
   The LLM path is optional and sits behind the same guardrails.

6. **Refusals are templates, not free text.** That is what lets stage 9 assert
   *"the refusal was a template, not advice"* rather than guessing with a keyword
   blacklist (which fails, because the template itself contains "you should buy").

---

## Guarantees, and how each is enforced

| Promise | Enforced by | Verified by |
|---|---|---|
| One source link per answer | stage 8 `G2` — URL must be in `ALLOWED_URLS` | `test_every_golden_query_is_answered_with_an_allowlisted_citation` |
| ≤ 3 sentences | stage 8 `G3` — hard truncation | `test_sentence_cap_is_enforced` |
| "Last updated from sources: …" | stage 8 `G6` — always stamped from chunk provenance | `test_render_always_carries_source_and_freshness` |
| No advice | stage 5 gate (26 patterns) **and** stage 8 `G4` (output scan) | 7 policy cases + `test_performance_is_refused` |
| No performance claims | stage 5 gate **and** stage 8 `G5` (output scan) | 5 policy cases + `test_performance_claim_becomes_refusal` |
| No PII stored or echoed | stage 5 `scrub_pii()` before routing; `raw_query` is discarded; stage 8 `G1` re-scans output | `test_pii_never_appears_in_any_output` |
| Public sources only | stage 0 registry; test asserts no blog/news domain is present | `test_no_third_party_blog_or_news_in_registry` |

Stage 8 is applied to **every** response, including LLM output. A failed check is never
silently downgraded: `G1`/`G4`/`G5` convert the response into a refusal, `G2`/`G3`
repair it, `G6`/`G7` append it.

---

## Deviations from the brief

Four departures were required to build this offline. They are listed here with
the identifiers used in `implementation.md` so the two documents can be read
against each other.

| ID | Brief asked for | Built instead | Why | Cost if wrong |
|---|---|---|---|---|
| **D-1** | `MiniLM` sentence embeddings | IDF-weighted hashed vectors: word unigrams + char 4-grams, cosine | Zero third-party dependencies, so `python run.py build` works on a clean 3.11 install with no network for wheels and no model download | Weaker on semantic similarity. Matters little at 114 short factual chunks; would matter at 100k+ |
| **D-2** | `ChromaDB` vector store | SQLite + FTS5 BM25 alongside a vector table in the same file | Removes the only heavyweight dependency, and BM25 is what makes jargon like "ELSS" and "exit load" match exactly | None at this scale. A different store would need a rebuild, not a rewrite |
| **D-3** | `hdfcmf.com` as the source | The five `groww.in` pages supplied in the brief | `hdfcmf.com` is a bot-protected, client-rendered SPA; a plain HTTP client cannot read the figures | Exit load, the full fee schedule and the riskometer are authoritative only in the AMC's SID/KIM. Registered as a **pointer** source and quoted nowhere |
| **D-4** | Riskometer level per scheme | A pointer to the AMC monthly factsheet | The riskometer value is not present in any page collected for this build | **The clearest functional gap.** Requires a JS-capable fetcher or supplied factsheets to close |

D-1 and D-2 are implementation substitutions, not scope reductions. D-3 and D-4
reduce functional coverage and are the two items to close first if this goes
further.

---

## Known limits

These are real, deliberate, and worth stating plainly. Items 1, 2 and 5 above
are D-4, D-3 and D-1 respectively.

1. **Riskometer is answered by a pointer, not a value.** The five scheme pages do not
   publish a riskometer level, and `hdfcmf.com` is a bot-protected, client-rendered SPA
   that a plain HTTP client cannot read. Rather than guess, the assistant says the
   figure is not in its sources and points at the AMC's monthly factsheet, which is
   where SEBI requires it. **This is the clearest functional gap in the build.**

2. **The quotable corpus is Groww, not hdfcmf.com.** The brief supplied the five Groww
   URLs, and they are the only public pages in scope that are server-rendered and
   machine-verifiable. Groww is a SEBI-registered investment adviser platform, not a
   blog, and no third-party blog or news article is used. But the *authoritative*
   document for exit load, the full fee schedule and the riskometer is the AMC's own
   SID/KIM/factsheet, which is registered in `SOURCES.md` as a **pointer** source and
   quoted nowhere. **Production use should replace the Groww pages with the AMC
   documents; the code does not need to change, only the registry.**

3. **No cross-scheme or cross-AMC comparison is possible.** One AMC, five schemes,
   direct-growth only. Regular-plan / dividend-variant figures are not in scope and
   the assistant will not guess them.

4. **The topic and refusal rules are regex, not learned.** They are auditable and
   testable, which is the point for a safety-critical facts tool — but they are
   English-only and will miss novel phrasings ("would you sit in small cap for a
   decade?"). Mitigation: the stage-8 output scan is a second, independent net, and
   the failure mode is a wrongly *answered* question only in genuinely novel phrasing.

5. **Dense retrieval is feature-hashed, not a trained embedding model.** It gives
   morphology tolerance and runs with zero dependencies, but it is weaker than
   MiniLM on semantic similarity. This matters little for a 114-chunk corpus of short
   factual statements; it would matter at 100k+ chunks.

6. **Build-time only freshness.** `Last updated from sources:` reflects the date on the
   source page, not a live re-check. `python run.py build --refresh` re-fetches;
   there is no scheduler.

7. **The raw cache in `data/raw/` is committed-friendly but not secret.** It contains
   only public HTML. Nothing about a user is ever written to disk — queries are not
   logged by the pipeline at all.

8. **Not rate-limit hardened.** The bundled server is a localhost prototype
   (`http.server` + a single-writer lock). It binds `127.0.0.1` and is not intended for
   public exposure. It sends a `default-src 'none'` CSP, so the page loads nothing from
   the network and cannot be framed.

9. **An unanswerable question still gets a confident answer.** Ask for something the
   sources do not publish — a market-cap or sector split, top holdings — and the
   pipeline retrieves the right *page* but returns whichever fact chunk sits on it:
   a launch date, an ISIN, an AUM figure. The citation is real and the answer is
   irrelevant. Retrieval scores on term overlap and nothing checks that the winning
   chunk carries the attribute that was asked for. The UI's scope panel names the
   attributes the corpus does cover so this is visible up front, but the underlying
   gap is in the pipeline, not the UI.

---

## Repository map

```
RAGproject/
├── run.py                     CLI: build | ask | serve | eval | sources | stage5 | stage6 | samples
├── bench.py                   latency probe
├── samples_gen.py             regenerates the sample Q&A from the live pipeline
├── sources_gen.py             regenerates SOURCES.md / SOURCES.csv from the registry
├── README.md                  this file
├── SOURCES.md / SOURCES.csv   source list (generated from the registry)
├── DISCLAIMER.txt             the exact disclaimer strings used in the UI
├── requirements.txt
├── docs/ARCHITECTURE.md       stage-by-stage design notes and trade-offs
├── src/mf_faq/
│   ├── config.py              paths, policy constants, retrieval weights
│   ├── sources.py             STAGE 0  registry + ALLOWED_URLS
│   ├── ingest.py              STAGE 1  fetch + raw cache + provenance manifest
│   ├── parsing.py             STAGE 2  HTML→text, embedded JSON→facts, sentence split
│   ├── chunking.py            STAGE 3  fact / section / procedure chunks
│   ├── indexing.py            STAGE 4  FTS5 BM25 + hashed dense vectors
│   ├── intent.py              STAGE 5  PII scrub, scheme/topic routing, refusal gate
│   ├── retrieval.py           STAGE 6  hybrid scoring + MMR + explain()
│   ├── synthesis.py           STAGE 7  extractive composer / LLM rewriter
│   ├── guardrails.py          STAGE 8  G1–G7 enforcement
│   ├── pipeline.py            orchestration
│   ├── eval.py                STAGE 9  golden + policy suite
│   ├── llm.py                 optional LLM client + system prompt
│   └── ui.py                  tiny stdlib web UI
│   └── web/                   UI assets, inlined into one page at serve time
│       ├── index.html         markup
│       ├── app.css            styles (light + dark, no webfonts)
│       └── app.js             behaviour (no framework, no build step)
├── data/
│   ├── raw/                   cached source HTML (public content only)
│   ├── corpus.jsonl           stage 2 output: normalised documents
│   ├── chunks.jsonl           stage 3 output: chunks
│   ├── fetch_manifest.json    stage 1 provenance
│   ├── eval_report.json       stage 9 output
│   └── index/mf_faq.sqlite    stage 4 output: BM25 + vectors
├── samples/sample_qa.md       sample Q&A (generated, 10 queries)
├── samples/sample_qa.json     the same, machine-readable
└── tests/test_pipeline.py     204 tests (stdlib unittest)
```

---

## Deliverables map

| Deliverable | Where |
|---|---|
| Working prototype | this repo; `python run.py serve` for the UI, `python run.py ask` for the CLI |
| Source list (CSV/MD) | `SOURCES.csv`, `SOURCES.md` |
| README: setup, scope, known limits | this file |
| Sample Q&A | `samples/sample_qa.md`, `samples/sample_qa.json` |
| Disclaimer snippet | `DISCLAIMER.txt`, `src/mf_faq/config.py` |
| Evaluation evidence | `python run.py eval` → `data/eval_report.json` |
| Latency evidence | `python bench.py` → cold / warm / refusal reported separately |

### How the deliverables are kept honest

Each generated artifact has a test that fails if it drifts from the code:

| Artifact | Check | Test |
|---|---|---|
| `samples/sample_qa.{md,json}` | re-running `run.py samples` is byte-identical, and matches the committed bytes | `test_samples_regenerate_byte_identically`, `test_committed_samples_match_the_live_pipeline` |
| `samples/sample_qa.json` | carries no timing, so the file stays reproducible | `test_sample_artifact_carries_no_timing` |
| `DISCLAIMER.txt` | every string appears verbatim in `config.py` / `guardrails.py` | `test_disclaimer_file_matches_the_code_verbatim` |
| `README.md` | links every deliverable and labels D-1..D-4 | `test_readme_links_every_deliverable`, `test_readme_states_the_known_deviations_plainly` |
| `SOURCES.md` / `.csv` | regenerated from `src/mf_faq/sources.py` | `python run.py sources --write` |

The sample files are stamped with the **index build time**, not the render time,
and carry no latency field. That is deliberate: a wall-clock stamp would make
every run produce a different file, and a diff could no longer be trusted to
mean "an answer changed". Timings live in `bench.py` instead.
