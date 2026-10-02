# Architecture — Mutual Fund Facts Assistant

**Derived from:** [`PRD.md`](../PRD.md) v1.0
**Status:** as-built, verified against the code
**Supersedes:** the earlier stage-notes draft, preserved at `docs/ARCHITECTURE.notes.md`
**Code:** `src/mf_faq/` · **Verified:** 204/204 unit tests, 28/28 eval checks, 114 chunks

This document explains *how the system is built and why each structural choice was made*.
It is organised so that every section answers a requirement in the PRD, and so that a
reviewer can locate any behaviour by module without reading the pipeline end to end.

---

## 1. How to read this document

| If you want to… | Go to |
|---|---|
| See the whole system in one picture | §2 |
| Know the governing design rules | §3 |
| Find which file implements which PRD requirement | §4 |
| Understand the data model and on-disk artefacts | §5 |
| Understand one RAG stage | §6 (stages 0–9) |
| Understand how safety is actually guaranteed | §7 |
| Decide the two open stack deviations (D-1, D-2) | §8 |
| Audit requirement coverage | §9 |
| Know performance, failure modes, limits | §10 |
| Extend the system | §11 |

The single most important idea in this architecture is in §7: **the product's safety
properties are not implemented in one place.** They are enforced at three independent
admission points, and each one can refuse on its own.

---

## 2. Architecture at a glance

Two planes. The **build plane** runs offline, needs the network, and produces an immutable
artefact. The **query plane** runs online, needs neither network nor API key, and is
read-only against that artefact.

```
                     ┌─────────────── BUILD PLANE                            ┐
                     │  python run.py build          (offline, replayable)   │
                     │                                                       │
  STAGE 0            │  sources.py     ┌──────────────────────────────┐      │
  source registry    │                 │ SOURCES[9] → ALLOWED_URLS    │      │
  (PRD FR-8)         │                 │  5 quotable · 4 pointer-only│      │
                     │                 └──────────────┬───────────────┘      │
                     ▼                                │                      │
  STAGE 1            │  ingest.py      fetch 9 URLs, content-hash cache     │
  data ingestion     │                → data/raw/*.html + fetch_manifest    │
  (PRD NFR-3, FR-9)  │                failure ≠ crash                      │
                     ▼                                                       │
  STAGE 2            │  parsing.py     HTML → visible text                  │
  parsing            │                                + __NEXT_DATA__ → facts │
                     │                                ⚠ normalise_facts()    │
                     │                                  drops returns ←──────┼── FR-4
                     ▼                                                       │
  STAGE 3            │  chunking.py    fact 70 │ section 34 │ procedure 5  │
  chunking           │                114 chunks, each with provenance      │
                     ▼                                                       │
  STAGE 4            │  indexing.py    SQLite FTS5 (BM25) + 512-d hashed     │
  embedding + store  │                dense vectors, one file                │
                     └───────────────────────┬───────────────────────────────┘
                                             ▼   data/index/mf_faq.sqlite
                     ┌─────────────── QUERY PLANE ──────────────────────────┐
                     │  python run.py ask | serve        (offline, read-only)
  STAGE 5            │  intent.py      ① PII scrub  ② scheme  ③ topic       │
  query admission    │                 ④ field hints  ⑤ 3 refusal gates    │ ← FR-3,4,5
                     ▼                                                       │
  STAGE 6            │  retrieval.py   hybrid score + boosts + kind gate     │
  retrieval          │                 + MMR re-rank, fully explainable       │
                     ▼                                                       │
  STAGE 7            │  synthesis.py   extractive composer (default)         │
  synthesis          │                 or LLM rewrite, same contract          │
                     ▼                                                       │
  STAGE 8            │  guardrails.py  G1 PII  G2 citation  G3 ≤3 sentences  │
  output admission   │                 G4 advice  G5 perf  G6 fresh  G7 disc │ ← FR-2,6,7
                     ▼                                                       │
                     └──────────────►  Answer (text · 1 URL · date · notice) ┘
                     ┌─────────────── EVAL PLANE ───────────────────────────┐
  STAGE 9            │  eval.py  13 golden + 15 policy = 28 checks           │
                     │  tests/   46 unit + end-to-end                       │
                     └──────────────────────────────────────────────────────┘
```

The build/query split is a **security boundary**, not a convenience. Every mutation of the
corpus happens offline, behind the stage-2 field whitelist. Nothing a user types can reach
the corpus, and nothing in the corpus that stage 2 rejected can reach an answer.

---

## 3. Design principles

These are the rules that every later decision is justified against. Each traces to a PRD
goal.

| # | Principle | Consequence in the code | PRD |
|---|---|---|---|
| P1 | **Restraint over breadth.** A refusal is a correct answer. | 3 refusal gates before retrieval; comparison is structurally impossible with one AMC | G3, §3 non-goals |
| P2 | **Provenance, not generation.** A number is copied, never written. | Default synthesiser is extractive; LLM is optional and untrusted | G1, FR-1 |
| P3 | **Enforcement lives in code, not in documentation.** | `ALLOWED_URLS` is a hard gate on every citation | FR-2, FR-8 |
| P4 | **Independent nets.** Two implementations of the same rule fail independently. | Stage-5 input gates and stage-8 output scan use *different* regex sets | FR-3, FR-4 |
| P5 | **A source that cannot be read cannot be quoted.** | Pointer-only vs quotable is a first-class source flag | FR-8, FR-9 |
| P6 | **Bounded blast radius.** A dead source degrades coverage, never availability. | Fetch failures recorded, build continues | FR-9, NFR-9 |
| P7 | **Determinism.** Same input, same answer, offline, no key. | Content-hashed cache; extractive path; persisted IDF | FR-10, NFR-3, NFR-6 |
| P8 | **Auditable over clever.** Regex rules beat a classifier when the refusal *is* a deliverable. | 26 advice + 17 performance regexes, matched-rule reporting | FR-3 |
| P9 | **The cheapest useful thing, at this scale.** | A linear cosine scan, not an ANN index — see §8.3 | NFR-2 |

---

## 4. Component map

One module per RAG stage. Each has one job, one inspectable artefact, one test, and one CLI
inspector so a wrong answer can be localised to a stage instead of to a prompt.

| Stage | Module | PRD requirement | Artefact | Inspector |
|---|---|---|---|---|
| 0 | `sources.py` | FR-8, §10 source policy | `ALLOWED_URLS` | `run.py sources` |
| 1 | `ingest.py` | FR-9, NFR-3 | `data/raw/*.html`, `fetch_manifest.json` | `run.py build` |
| 2 | `parsing.py` | FR-4 (drop returns), FR-1 | `data/corpus.jsonl` | read file |
| 3 | `chunking.py` | FR-10 (chunking) | `data/chunks.jsonl` | read file |
| 4 | `indexing.py` | FR-10 (embed + store) | `data/index/mf_faq.sqlite` | any SQLite client |
| 5 | `intent.py` | FR-3, FR-4, FR-5, FR-10 (field hints) | `Decision` | `run.py stage5 "<q>"` |
| 6 | `retrieval.py` | FR-10 (hybrid, kind gate, MMR) | `RetrievalResult` | `run.py stage6 "<q>"` |
| 7 | `synthesis.py` | FR-1, FR-7 | `Draft` | `run.py ask --json "<q>"` |
| 8 | `guardrails.py` | FR-2, FR-3, FR-4, FR-5, FR-6, FR-7 | `Answer` | `run.py ask --debug "<q>"` |
| 9 | `eval.py`, `tests/` | §11 acceptance A1–A5 | `data/eval_report.json` | `run.py eval` |

Support modules: `pipeline.py` (orchestration), `config.py` (paths, policy constants,
retrieval weights), `llm.py` (optional provider client + system prompt), `ui.py`
(`http.server` frontend), `__init__.py` (welcome line, 3 example questions).

---

## 5. Data model and artefacts

### 5.1 The `Chunk` — the unit of everything

Every stage after 3 speaks in chunks. The schema is deliberately wider than a text blob,
because five separate requirements need fields that a naive chunker would not carry.

| Field | Type | Consumed by | Why it exists |
|---|---|---|---|
| `chunk_id` | str | index, MMR | stable primary key, `doc_id::kind::key` |
| `doc_id`, `source_id`, `url` | str | G2, stage 7 | **the citation is carried, not re-derived** (FR-2) |
| `kind` | `fact` \| `section` \| `procedure` | stage 6 kind gate, stage 7 | §6.3 |
| `topic` | str | stage 5 routing, stage 6 | 14 topics; drives exclusivity + gating |
| `scheme_key` | str \| None | stage 6 scheme boost | `None` = scheme-agnostic (procedures) |
| `text` | str | stage 6, stage 7 | fact chunks are quotable alone |
| `answer_hint` | str | stage 7 | canonical short answer, authored at chunk time |
| `last_updated` | str | G6 | **source-derived date, not the clock** (FR-6). Empty for authored procedures, which have no source currency to claim |
| `tokens` | list[str] | lexical scoring | precomputed at build |
| `meta.field` | str \| None | stage 5 hints → **stage 7 filter** | the hard constraint of §6.6 |
| `meta.raw_value` | any | debugging | the un-normalised source value |

The two fields that carry the most architectural weight are `url` and `meta.field`:

* `url` on the chunk means stage 7 never *chooses* a citation. It copies one. A model that
  invents a URL cannot do so, because there is no code path that lets synthesis type a
  citation that did not come from a retrieved chunk.
* `meta.field` is what makes a stage-5 field hint a filter rather than a preference
  (§6.6). **Dropping this field silently disables every field-level signal** — a bug that
  actually occurred, recorded in the comment at `retrieval.py:148`.

### 5.2 SQLite schema (stage 4)

```sql
chunks(chunk_id PK, doc_id, source_id, url, kind, topic, scheme_key,
       text, answer_hint, last_updated, meta_json, vector BLOB)
chunks_fts  -- FTS5 virtual table over (chunk_id UNINDEXED, text,
              answer_hint, topic, scheme_key UNINDEXED)
              tokenize = "unicode61 remove_diacritics 2"
index_meta(key PK, value)   -- built_at, stats, config, idf
doc_freq(term PK, df)
```

The fitted IDF table is **persisted into `index_meta`**, so `load_vectoriser()` rebuilds the
vectoriser at query time with no re-fit and no pickle. This is what lets the query plane be
read-only and start instantly.

### 5.3 FTS5 column weighting — a correction

`bm25()` is called as `bm25(chunks_fts, 1.0, 0.6, 0.0)`. Weights are **positional against
the declared column order**, so the effective assignment is:

| Column | Weight | Indexed | Effect |
|---|---|---|---|
| `chunk_id` | 1.0 | no | none |
| `text` | 0.6 | yes | the prose/answer body |
| `answer_hint` | **0.0** | yes | **excluded from scoring** |
| `topic` | 1.0 (default) | yes | **highest-weighted signal** |
| `scheme_key` | 1.0 (default) | no | none |

An earlier draft of this document claimed `answer_hint` was "weighted below `text`". That
is wrong: it is weighted to zero, and `topic` — not `text` — is the dominant BM25 signal.
This was verified against the built index. The design is sound, but the reasoning is
different from what was documented: the intended mechanism is that the **routing signal
(topic) is weighted above the body text**, and `answer_hint` is deliberately redundant with
`topic` so it is switched off to avoid double-counting the same tokens.

---

## 6. The stages

### 6.1 Stage 0 — Source registry (`sources.py`) · FR-8

A frozen list of `Source` records and the derived `ALLOWED_URLS` frozenset that stage 8
validates every citation against. **A URL that is not in this list can never appear in an
answer.** The allow-list is the enforcement mechanism, not a documentation convention.

Nine sources: 5 **quotable** (the Groww scheme pages named in the brief, all verified
retrievable at build time) and 4 **pointer-only**:

| Source | Why pointer-only |
|---|---|
| `hdfc-investor-portal` | DNS/host rejects non-browser clients |
| `cams-statements` | client-rendered SPA |
| `hdfc-official-scheme-docs` | client-rendered, bot-protected |
| `amfi-nav` | registered as a NAV cross-check source |

P5 in one sentence: *a page we cannot read is a place we can send you, never a page we can
quote.* This is what keeps D-3 (§8.4) from becoming a correctness problem — the authoritative
SID/KIM/factsheet is cited as the destination for anything we cannot verify ourselves.

`test_no_third_party_blog_or_news_in_registry` asserts no blog/news domain is present,
because the realistic failure mode for this product is citing a content farm that paraphrases
a factsheet.

### 6.2 Stage 1 — Data ingestion (`ingest.py`) · FR-9, NFR-3

Fetches each registered URL, writes bytes to `data/raw/<slug>.<hash>.html`, and records
provenance in `data/fetch_manifest.json` (status, bytes, SHA-256, timestamp, error).

Two properties:

* **Replayable.** The content-hashed cache makes the build deterministic and lets stages
  2–4 run with no network. `MF_FAQ_OFFLINE=1` enforces it.
* **Never hard-crashes.** A failed fetch becomes `FetchResult(ok=False, error=...)`. On a
  clean build 1 of 9 sources fails (`hdfc-investor-portal` — DNS does not resolve outside
  the corporate network) and 3 more fetch HTTP 200 but yield no quotable document, so the
  build continues with 5 quotable sources. The degradation is *visible in the build output
  and in the `quarantined` list*, not silent.
* **Offline means offline.** `MF_FAQ_OFFLINE=1` guarantees **no socket is opened at all** —
  a populated cache is still served from disk, and only an uncached source degrades to
  `error="offline and not cached"`. `force=True` ignores the cache but cannot bypass the
  flag. This is enforced by a test that swaps `urlopen` for a canary.

### 6.3 Stage 2 — Parsing (`parsing.py`) · FR-1, FR-4

Two extractions, because they fail in different ways:

* **`visible_text`** — `html.parser`-based render, `<script>/<style>` stripped, whitespace
  normalised, nav/footer chrome removed. Serves wording-level questions and is the lexical
  backstop.
* **`embedded_json`** — the `__NEXT_DATA__` payload these pages ship, carrying field-level
  facts: expense ratio, exit load, min SIP/lumpsum/additional, lock-in days, benchmark, NAV +
  NAV date, category, fund manager, launch date, ISIN, RTA. This is the higher-fidelity
  signal and answers anything a number can.

**`normalise_facts()` is the security-relevant function in this stage.** It *whitelists* the
fields it copies. The page payload also contains return figures, and they are deliberately
not copied — so no performance number exists in any **fact**, and answers are composed only
from whitelisted facts plus the fixed refusal templates, so none can reach a user even under
prompt injection. This is stronger than filtering at answer time, because the data is never
admitted. `test_facts_schema_drops_performance_fields` locks it in.

**One honest caveat about `text`.** The pages embed a peer-comparison widget listing *other*
HDFC funds with their 1-day/1-year returns, so `visible_text` legitimately carries 67–349
return-shaped figures per page (`+13.80%` style) in the `text` field. Those are in `text`
only, never in `facts`, and they are the reason two properties are load-bearing rather than
optional:

* stage 8 composes answers from `fact` chunks only, so page chrome cannot be quoted; and
* the `performance_refusal` gate must fire on the *question*, since the corpus itself
  cannot be scrubbed of the word.

Two phrases are deliberately **not** treated as performance: `Direct - Growth` /
`scheme_type="Growth"` (a plan designation, and the whole corpus is direct-growth) and
`benchmark_name="NIFTY 100 Total Return Index"` (a benchmark's published name). Tests assert
`growth` appears only inside identity keys, and that no **signed** percentage — the form
Groww renders returns in — ever appears in a fact.

7 documents survive to stage 3 out of 9 sources.

### 6.4 Stage 3 — Chunking (`chunking.py`) · FR-10

The brief asked for the chunking strategy to be *decided from the data*. The decision:

> **Chunk shape follows question shape.** "What is the exit load?" wants one number, not a
> paragraph. So emit three families and let stage 6/7 route between them.

| Kind | Count | Shape | Role |
|---|---|---|---|
| `fact` | 70 | one atomic sentence per (scheme, field), from a template | highest precision; carries `meta.field` + `answer_hint` |
| `section` | 34 | 4-sentence prose windows, 1-sentence overlap, ≥60 chars, ≤160 tokens, topic assigned only on an explicit phrase (`TOPIC_MARKS`) | lexical backstop; wording questions |
| `procedure` | 5 | authored, non-numeric, bound to an official destination | the questions that have no number |

A fact chunk, e.g.:

```
The expense ratio of HDFC Large Cap Fund (Direct - Growth) is 1.03%. The expense
ratio is the annual percentage of the scheme's assets charged to the scheme,
deducted from the scheme's returns rather than billed separately.
```

Sentence 1 is complete and quotable alone — scheme, value, unit. This is what makes the
FR-2 citation *verifiable by reading the chunk* rather than by trusting a paraphrase.

**Normalisation is formatting-only.** `_fmt_exit_load` fixes capitalisation, spacing, and one
badly-ordered source pattern:

```
raw:      "Exit Load for units in excess of 15% of the investment,1% will be
           charged for redemption within 1 year."
emitted:  "1% for units in excess of 15% of the investment, if redeemed within 1 year"
```

Rate, threshold and window are unchanged. Anything requiring judgement is not normalised —
values over 220 characters are skipped as free-form blobs and left to `section` chunks.

**Trade-off accepted:** `procedure` chunks are *authored* text, not scraped text. That is
what lets them point at a portal the scraper cannot read (§6.1), but it makes them a
maintenance obligation: a portal URL change is a code change, not a re-index. Recorded in
PRD §12.

**`last_updated` provenance (FR-6).** The date on a chunk is `nav_date`, else the fetch
timestamp — never the clock. `chunking.py` contains no clock read at all, not even a
fallback, and a test enforces that by walking the AST rather than grepping (the module
*documents* the absence in a docstring, which a text scan would misread).

Procedures therefore carry `last_updated=""`: they are authored prose, so a date on them
would assert that a source was refreshed when nothing was fetched. Because the UI prints this
field, an undifferentiated stamp would put a false provenance claim on the one answer that
must look most careful — a refusal, which quotes no source at all. So the flag is carried
end-to-end as `Answer.freshness_from_source` and the wording follows it:

| answer | date | rendered as |
|---|---|---|
| grounded in fetched facts | `nav_date` / `fetched_at` | `Last updated from sources: 2026-09-25` |
| authored procedure | *(none)* | `Answer prepared on: 2026-09-27` |
| any refusal | *(none)* | `Answer prepared on: 2026-09-27` |

Both paths always print a date (gate G6 and the eval's `freshness` check both require one);
what changes is whether it is a *claim about the source* or a statement about the answer.

### 6.5 Stage 4 — Indexing (`indexing.py`) · FR-10

Two indexes over the same chunks in one SQLite file.

**Lexical (BM25).** FTS5, `unicode61 remove_diacritics 2`, OR-joined quoted tokens. Exact
match, self-explaining, no training — the workhorse for `ELSS`, `exit load`, `riskometer`.
Effective weighting per §5.3.

**Dense (cosine).** IDF-weighted feature-hashed vectors, 512-dim, over word unigrams plus
character 4-grams, L2-normalised, stored as float32 BLOBs. Signed `blake2b` hashing reduces
collisional bias; sublinear TF and IDF weighting stop common words dominating. Pure Python,
with `numpy` used only to accelerate when present.

Measured on this corpus (114 chunks, 512-d, median of 12 runs): whole-corpus `transform` is
75.7 ms with numpy and 77.2 ms without - about 2%, and inside the run-to-run spread. The cost
is dominated by feature extraction and `blake2b` hashing, not the vector arithmetic, so numpy
is close to a no-op here. It is kept because it costs nothing and helps on larger corpora,
but it is not the speed story it appears to be, and the single-query cost (0.23 ms) was never
material.

The char-4-grams are the point: they give the morphology tolerance pure BM25 lacks
(`expense`/`expenses`, `riskometer`/`riskometre`). §8 covers why this is not
`sentence-transformers/all-MiniLM-L6-v2` and how to change it.

Neither index is a black box: `index_stats()` reports document frequencies, vocab size,
per-kind and per-topic counts, and whether numpy was active. The `doc_freq(term, df)` table
is materialised at build time so the vocabulary can be queried directly rather than inferred
from the persisted IDF.

**One file, two planes, identical arithmetic.** The document vectors and the query vectors
must be produced by the same code or the index quietly self-degrades. Two things enforce it:

- `index_meta` persists the whole fitted state - `idf` **and `default_idf`**, the weight for a
  term the corpus never saw. `default_idf` looks derivable (`log(n+1)+1`) but is not safe to
  re-derive: it was once reconstructed as `max(idf.values())`, which is 11.5% lower than the
  value the documents were built with, so every out-of-vocabulary query term was scored at
  the wrong weight. A query with one unseen term moved `1.6e-2` in vector space.
- The `numpy` fast path accumulates in **float64** and normalises through the *same* helper as
  the pure-Python path. A float32 accumulator drifts ~`5.8e-8` from float64, and
  `numpy.linalg.norm` sums in a different order again, so a machine without numpy would build
  a byte-different index from the same corpus. Verified: all 114 stored BLOBs are
  byte-identical to a pure-Python rebuild. For the same reason `cosine()` is pure Python even
  when numpy is installed - it is the ranking function, and two machines must break ties the
  same way.

The stored vector is float32 by design, so `_unpack` cannot equal the in-memory float64
vector. The reproducibility claim is about the BLOB bytes, which is the thing that gets
compared across machines.

### 6.6 Stage 5 — Query understanding (`intent.py`) · FR-3, FR-4, FR-5

Runs **before** retrieval. Five decisions, in order:

1. **PII scrub.** `scrub_pii()` removes PAN, Aadhaar, email, phone, OTP, folio/account and
   UPI/IFSC patterns, plus bare 10–12 digit runs. On a hit the decision is `pii_refusal`
   and **`raw_query` is set to `""`**.
   Coverage is by *value shape*, so the shapes are enumerated explicitly, and both of these
   were wrong in the as-built code:
   - A UPI handle is routinely **one to three characters** (`a@okaxis`). The pattern required
     `{4,}` before the `@`, so the most common handles in the country were not detected and
     came back as `kind="answer"` with the handle sitting in the decision.
   - An **IFSC has no `@` at all**, so a UPI-shaped pattern never matched one and
     `ifsc is HDFC0001234` leaked verbatim.

   The keyword gate (`upi`, `ifsc`, …) is a second line of defence, not a substitute: it
   refuses *"what is my upi id"* — an account-specific question the assistant has no data
   for — with or without a pasted value.
2. **Scheme resolution.** Longest-alias match over 5 schemes × 36 aliases, so
   `"HDFC Flexi Cap Fund"` beats the bare `"equity fund"` alias.
3. **Topic routing.** Ordered regex rules, then **exclusivity**: the three statement/tax
   topics are alternative answers, not additive facets. Without it a "tax statement" query
   drifts onto the capital-gains procedure because both mention statements.
4. **Field hints.** A more specific signal than topic — `min_sip_amount` vs
   `min_lumpsum_amount`, `fund_manager` vs `registrar_agent`, `isin`, `launch_date`.
5. **Refusal gates**, ordered: **performance → bare-PII-keyword → advice**.

**Why performance before advice.** "Compare the returns of A and B" trips both sets. The
no-returns refusal is the more specific and more useful response, and fixing the order makes
the behaviour deterministic instead of dependent on pattern list order.

**Why the gates are regex, not a classifier.** P8. For a tool whose refusal behaviour *is* a
deliverable, "show me the rule that fired" (`Decision.matched_rules`) beats a probability.
The cost — English-only, and vulnerable to genuinely novel phrasing — is stated in PRD §12,
and is mitigated by the independent stage-8 scan.

**Architectural invariant (FR-5).** `Decision.raw_query` is populated **only on paths where
`scrub_pii()` found nothing**. Every PII-bearing path returns `raw_query=""`. So the
unsanitised string never enters the decision record, the evidence trail, the debug output
or any log. Note that the advice/performance refusal paths pass `raw_query=raw` — this is
safe *only* because execution cannot reach them once PII has been found. If stage 5 is ever
reordered, that invariant breaks and FR-5 breaks with it. It is asserted in code comments
and covered by `test_pii_never_appears_in_any_output`.

**`raw_query` is the wrong place to stop.** Clearing it is necessary but not sufficient,
because `Decision.query` is the field the pipeline actually retrieves. On the keyword path no
*value* pattern fires by construction, so a credential whose shape the patterns do not
recognise survived there: `my pan is AB1` was refused correctly, yet `AB1` remained readable
in the decision. `_redact_keyword_values()` now strips the tail after a sensitive keyword on
that path. The rule to hold is that **no identifier value survives in any field of the
decision**, not merely in the one the type system happens to clear.

### 6.7 Stage 6 — Data retrieval (`retrieval.py`) · FR-10

```
score = 0.55 · bm25_norm          # FTS5 rank, max-normalised within the result set
      + 0.45 · cosine             # dense, clamped at 0
      + 0.22 · scheme_match       # full match | ×0.25 scheme-agnostic | ×−0.5 wrong scheme
      + 0.20 · field_match        # first hint | ×0.4 later hints
      + 0.14 · topic_match        # primary | ×0.6 other topics | ×0.5 procedure neighbour
      + kind_gate                 # positive prior, or a NEGATIVE penalty — see below
```

then **MMR re-rank** (λ = 0.72) over the candidates.

Four parts of this are doing work that a plain relevance score cannot:

* **The kind gate is a negative constraint.** `NON_FACT_TOPICS` and `PROCEDURAL_GROUP`
  encode what must *not* answer a topic: a per-scheme number can never answer a
  portfolio-wide procedural question (−0.25), and the three statement procedures are mutually
  exclusive destinations, so a capital-gains procedure is the wrong answer to a tax-statement
  question (−0.20). A pure similarity score is directionless; this is not.
* **The Phase-5 seam is enforced, not just intended.** `retrieval.py` imports only
  `load_vectoriser`, `all_vectors` and `cosine` from `indexing`; the vectoriser *class* stays
  private to that module. It did not hold as written — `HashedVectoriser` and the private
  `_pack` were both imported and neither was used, so the file carried a live coupling to the
  D-1 choice for no benefit. An AST test now pins the allowed import set, because an
  "unused import" lint would not have caught the *seam* half of the problem.
* **Field hints appear twice, asymmetrically.** Retrieval *boosts* (0.20) so the right chunk
  surfaces in the top-k, but correctness does not depend on the boost winning — stage 7
  **filters** on the same hint. This was found by evaluation, not design: "Who manages HDFC
  Balanced Advantage Fund?" was answered with the RTA, because the fact chunk says *fund
  manager*, a prose section literally contains *manages*, and no soft boost beat a longer
  text match. Raising the boost to 0.12 was not enough. The fix is structural.
* **MMR matters because of templating.** Each scheme produces ~13 near-identical template
  sentences, so the top-5 would otherwise be five paraphrases of the same fact.
* **`kind_gate` shares the `kind_prior` field** on `Scored`. It is a prior *or* a penalty
  depending on the topic. This is a naming wart inherited from the module docstring, noted
  here so the next reader is not misled.

**`final` is the pre-MMR relevance score.** The MMR redundancy penalty is applied during
*selection* (`val = λ·final − (1−λ)·redundancy`) and recorded on `Scored.penalty`; it is
never folded back into `final`. So `explain()` prints ranking order and reasons, not the
MMR-adjusted value.

**Explainability is a requirement, not a feature.** `explain()` prints the full additive
breakdown per hit — `bm25`, `dense`, `scheme`, `topic`, `field`, `kind`, plus `final` and the
MMR `mmr_penalty` — not just a compact list of winning reasons. Without the numbers you
cannot tell *which* boost carried a hit or by how much, which is the whole point when a
ranking is wrong.

**How a `field:` reason finds a bug, and how it misleads one.** A `field:` reason that never
appeared in any ranking was the visible symptom of `Retriever` not loading chunk `meta` from
the database — the documented failure mode. The converse also occurred: `field:` reasons
*were* appearing for some queries, so the diagnostic read as "meta is fine", when the actual
defect was upstream in `intent.FIELD_HINTS` — `expense_ratio_pct` matched only `ter` and
`total expense`, so the plain phrasing `"expense ratio"` produced no hint and the hint was
dead for its own topic. The lesson is that a `field:` reason proves the *pipeline* works only
for queries that produce a hint; the diagnostic is necessary, not sufficient.

**Performance.** The first MMR implementation re-projected candidate texts on every
comparison pair: 10.9 s warm. Reusing the vectors already stored in the index brought mean
latency to **49 ms** (258 ms cold) and the test suite from 20 s to 1.9 s.

### 6.8 Stage 7 — Answer synthesis (`synthesis.py`) · FR-1, FR-7

Two generators behind one interface.

**`ExtractiveSynthesiser` (default).** Not a language model. Routes on primary topic:

* `_fact` — sentence 1 is the winning fact chunk's own first sentence. Sentence 2 is added
  **only** if a companion field belonging to the asked topic exists. `COMPANION_FIELDS` is
  deliberately sparse: the two SIP minimums are the same family of number and useful
  together, but `"The fund manager is X. The ISIN is Y."` answers nothing asked, so
  `scheme_meta` gets no companion at all. For `lock_in` and `exit_load`, the chunk's own
  second sentence is kept as the qualifier rather than padding.
* `_procedural` — quotes the authored procedure chunk.
* `_pointer` — quotes the riskometer/fee-schedule pointer chunk, which is *already written*
  as a self-contained non-numeric answer.

All three generators emit **one** evidence schema, via the shared `_evidence()` helper:
`chunk_id`, `score`, `kind`, `topic`, `url`, `reasons`. They originally emitted three
different shapes (`_fact` carried `topic` + `reasons`, `_procedural` only `kind`, `_pointer`
neither), so any consumer reading `e["reasons"]` worked for a fact answer and raised
`KeyError` for a riskometer or statement answer — which is precisely the access pattern in
`implementation.md`'s own verification command. Divergent evidence shapes are a silent
contract break, so the shape is pinned by a test rather than by inspection.

`_draft_date()` is the single place the two provenance rules are distinguished, and the
distinction is observable in `Draft.freshness_from_source`:

| Answer kind | `last_updated` | `freshness_from_source` | Rendered as |
|---|---|---|---|
| fact, from a retrieved source chunk | the chunk's `last_updated`, copied verbatim | `True` | `Last updated from sources:` |
| procedure / pointer / refusal / empty | build date | `False` | `Answer prepared on:` |

The two are not interchangeable. Matching the date check on URL alone is also wrong: one
page yields many chunks with different dates, so the check must be against the specific
chunk that was quoted.

**`LLMSynthesiser` (optional).** Receives a compact numbered context and must return
`{"answer", "citation"}`. Its output is **untrusted** and goes through the identical stage-8
path. Any of: no provider, provider error, unparseable JSON, missing or non-string
`answer`, or a stage-8 guard failure, falls back to the extractive draft. So enabling the
LLM can make the assistant more fluent but never less safe.

The fallback guarantee is enforced by construction rather than by inspection: `generate`
catches only `LLMUnavailable`, so **nothing** may raise out of `_parse`. A truthy
non-string `answer` (e.g. `123`) used to pass the `if not obj.get("answer")` gate and then
raise `AttributeError` on `.strip()`, turning a malformed model response into a 500 rather
than the extractive draft. `_parse` now rejects non-dict payloads, non-strings and blank
strings up front. Both provider adapters pin `temperature=0` and cap `max_tokens`, so a
model cannot vary the answer between identical requests.

**The prompt is documentation, not enforcement.** `llm.SYSTEM` restates the hard rules,
including the PII the assistant actually detects. It names PAN, Aadhaar, folio/account
numbers, OTPs, email, phone, and UPI/IFSC details — the last two added in stage 6, so the
prompt no longer under-states the guard it describes. Stage 8 remains the only thing that
can reject an unsafe completion.

**Why default to no LLM (P2).** A facts-only assistant whose numbers come from a source of
record is safer when it *cannot* paraphrase a number. Fluency is worth less than the
guarantee that "1.03%" was copied.

### 6.9 Stage 8 — Post-processing and guardrails (`guardrails.py`) · FR-2,3,4,5,6,7

Applied to **every** response, including LLM output. Order matters: the three *blocking*
checks run before the body is cleaned, because a contaminated body should be replaced
wholesale rather than repaired.

| Check | Failure action | PRD |
|---|---|---|
| `G1` PII in body | → `pii_refusal` (never echo) | FR-5 |
| `G4` advice verbs in body | → `advice_refusal` | FR-3 |
| `G5` return/performance claim in body | → `performance_refusal` | FR-4 |
| `G2` citation not in `ALLOWED_URLS` | → drop citation, flag `G2_citation_missing` | FR-2 |
| `G3` > 3 sentences | → truncate to 3 | FR-7 |
| `G6` no freshness stamp | → stamp from chunk provenance | FR-6 |
| `G7` no disclaimer | → attach | FR-6 |

Also here: inline URLs in the body are promoted to the single citation and stripped from the
prose; duplicate sentences are collapsed; an empty body becomes the empty-query prompt.

**`final` guarantee for FR-2:** after `enforce()`, `Answer.citation` is either `""` or a
member of `ALLOWED_URLS`. There is no third case. This is the mechanical form of "every
answer carries one source link", and it holds for refusals too — all four canonical
refusal kinds carry an allow-listed education link, so the one response that most needs a
next step never renders without a source.

**Refusals are templates (P8).** `REFUSAL_BODIES` holds the only strings a refusal may emit.
That is what lets stage 9 assert *the refusal was a template, not advice* — a check that
replaced an earlier keyword grader which failed, correctly, on its own advice template
("whether you should buy" appears in order to decline).

Because that exception is load-bearing, `enforce()` short-circuits the three blocking nets
for an answer whose `kind` is already a canonical refusal. The pipeline routes stage-5
refusals around `enforce` entirely, but relying on that would make the safety property a
property of its callers: any future caller that passed a refusal through would have G4 read
"whether you should buy" as a leak and rebuild the refusal under a new kind.

**G1 is a second implementation, not a re-import.** `guardrails.py` must not import
`intent.PII_PATTERNS`, and an AST check enforces it. It originally did, which silently
collapsed the two layers into one: every gap in the input gate was then a gap in the output
gate. The nets now differ in *approach*, not merely in constants — stage 5 matches
"known label + known value shape", G1 matches "identifier shape on its own" plus
"sensitive label + any value containing a digit":

| Input | `intent.PII_PATTERNS` | G1 `_contains_pii` |
|---|---|---|
| `Your pan is abcde1234f.` | missed (PAN net is upper-case only) | caught (`pan` label + digit) |
| `Please confirm otp is 123.` | missed (OTP net needs 4–6 digits) | caught (label + 3-char value) |
| `IFSC SBIN0000123` | caught | caught (label-free IFSC shape) |
| `NIFTY 100 Total Return Index` | n/a | **not** caught — a benchmark name |

The second net closes the first's gaps by construction, and a differential test asserts the
two disagree on real inputs so they cannot quietly converge back into one object.

**`PERFORMANCE_LEAK` is anchored to return *phrasing*, never to a bare number.** That
anchoring is what keeps the stated pitfall safe: a legitimate NAV or expense-ratio answer
contains a percentage and must not be refused. Two corpus facts constrain the vocabulary
concretely, and both were found by sweeping all 75 fact chunks through all three nets:

- `growth` cannot be a return noun. Every scheme here is named "... (Direct - Growth)", so
  including it flagged the expense ratio, exit load and benchmark of all five funds.
- `Total Return` must not be read as a metric. `NIFTY 100 Total Return Index` is the
  benchmark's *name* and a legitimate quotable fact, so that alternative carries a negative
  lookahead for `index`.

The 70-chunk sweep is a test, not a spot check: a guardrail that refuses a legitimate fact
is worse than no guardrail, because it silently replaces a correct answer with a refusal.


### 6.10 Stage 9 — Evaluation (`eval.py`, `tests/`)

* **13 golden cases** — graded on correct citation, correct figure, ≤3 sentences,
  disclaimer, freshness stamp.
* **15 policy cases** — graded on correct refusal kind **and byte-identity** with the
  canonical template.

`Report.add(name, checks)` passes a case only when *every* check is true, and records the
names of the ones that failed, so a failing run says which check broke rather than only how
many. `run.py eval` maps `failed == 0` to the exit code and writes `data/eval_report.json`.

**Three graders were weaker than their names claimed**, and each is now tested directly
(`TestGradersCanFail`) rather than trusted because 28/28 was green:

| Check | Was | Now |
|---|---|---|
| `citation_allowlisted` | `citation.startswith("http")` — a *scheme* test, which waved through `http://evil.example.com/mutual-funds/fake` | membership of `ALLOWED_URLS` |
| `no_pii_echo` | two hardcoded literals (a PAN and an email), so the folio-number case had nothing to assert and a refusal that echoed the folio still graded clean | literals derived per case from the query by pattern (`_echoed_pii`), covering all three PII cases |
| `no_perf_claim` | `digits%` **then** `cagr\|return`, which caught "18.4% CAGR" and missed "3 year CAGR was 18.4%", "it returned 18.4%" and "CAGR of 18.4%" — three of five natural phrasings | any percent figure, or `cagr`/`return` followed by a digit. No canonical template contains a digit or a `%`, so nothing legitimate trips it |

The refusal grader also applied a looser sentence cap (`MAX_SENTENCES + 1`) than the answer
grader, on the theory that templates run long. They are two sentences against a cap of three,
so the slack bought nothing and only widened what could slip through; both use the same cap
now.

`evaluate()` also had a dead ownership guard. It ended with `if pipe is not None: pipe.close()`,
but the line above it is `pipe = pipe or FAQPipeline()`, so the condition is never false and a
caller that passed its own pipeline had it closed underneath them. It now tracks whether it
created the pipeline. (The same pattern exists in `ui.serve()`, where `pipe.close()` in the
`finally` still takes a caller-supplied pipeline — flagged, not changed.)

**The suite is verified non-vacuous, by mutation.** A green 28/28 only means something if a
bad answer would score badly, so each mutation is applied, observed to fail, and reverted:

| Mutation | Detected by |
|---|---|
| `MAX_SENTENCES = 9` in `config.py` | `test_configured_cap_is_three`, `test_sentence_cap_is_enforced`, `test_a_too_long_answer_fails_sentence_cap` |
| `MAX_SENTENCES` shadowed at the point of use | `test_no_module_shadows_the_cap` |
| advice refusal template reworded (`should` → `ought to`) | `test_only_the_canonical_template_passes_is_template`, `test_advice_template_is_not_its_own_leak` |
| golden figure changed with the URL left correct | eval `contains:1.03` → fails, which is the check's whole point |
| golden citation changed to a near-miss URL | eval `right_citation` |
| advice pattern widened to match "expense ratio" | eval `kind`, `right_citation`, `contains:1.03` |
| control: no mutation | still 28/28, as it must be |

The first row needed a test to exist before it could be caught. With the cap unpinned,
`MAX_SENTENCES = 9` produced **zero** failures: the golden answers are two or three sentences
because the corpus does not offer more, so raising the cap changed nothing observable. The cap
is now pinned both as a configured value and behaviourally, by trimming an input long enough
that a raised cap would return it untruncated.

**Named coverage is not coverage.** The four behaviours in spec item 6 that had no direct
behavioural test were mutation-tested, and two turned out to be named-but-vacuous:

| Area | Mutation | Was | Now |
|---|---|---|---|
| MMR diversity | `val = cand.final` — drop the redundancy term | **MISSED**, 0 failures | DETECTED, 2 tests |
| Alias precedence | `if best is None` — first hit, not longest | **MISSED**, 0 failures | DETECTED, 2 tests |
| Field-hint filter | `detect_field_hints` returns `[]` | DETECTED | — |
| Topic exclusivity | keep every matching topic | DETECTED | — |

The MMR test that existed checked *performance* — that stored vectors are reused instead of
re-projected — and said nothing about whether MMR selects anything. The alias test was named
`test_scheme_resolution_prefers_longest_alias` but every query in it matches exactly one
alias, so it passed just as well when resolution took the first hit. Both are now behavioural:
`_mmr` is driven with a pool of two near-identical high scorers and one orthogonal lower scorer
and must pick the distinct chunk second (relevance alone picks the duplicate — asserted as a
control), and alias precedence is decided by a query where a 9-character alias of the earlier
scheme and a 19-character alias of the later one both match, plus a systematic sweep over
every cross-scheme pair where the length rule and the first-seen rule disagree.

One trap worth recording: the grader and the product share the single `REFUSAL_BODIES`
constant, so *editing that dict in place* moves both sides of the `is_template` comparison and
proves nothing — the suite stays green while the template is reworded. Demonstrating the
failure requires rebinding `guardrails.REFUSAL_BODIES` so the pipeline emits new text while
the grader keeps the pristine copy.

### 6.11 Orchestration and CLI (`pipeline.py`, `run.py`, `ui.py`)

`FAQPipeline` owns one built index and answers queries. Three properties are load-bearing:

- **The retriever is lazy.** `__init__` opens nothing; the index is read on first use. This is
  what lets `build()` run in the same process that later answers, and it is why a missing
  index produces an actionable `FileNotFoundError` naming `python run.py build` instead of an
  import-time crash.
- **A stage-5 refusal returns before retrieval.** A refused query therefore answers even when
  no index exists on disk, which is the mechanical proof that refusals never touch the corpus.
- **`build()` is offline and repeatable.** Nine sources: one cannot be fetched, three more
  fetch successfully but are pointer-only. The build does not abort on either; it reports
  `fetch.quarantined` and still writes the full 114-chunk index.

`run.py` exposes eight subcommands behind a `--debug` global. Two constraints are Windows
requirements rather than preferences: `sys.stdout.reconfigure(encoding="utf-8",
errors="replace")` runs before any printing, because the console defaults to cp1252 and the
answers contain `₹`; and `sys.path.insert(0, .../src)` makes the package importable without
installation. `cmd_eval` maps the report to an exit code, so the suite is usable in CI.

**The web UI is threaded, and that is its one sharp edge.** `serve` runs on
`ThreadingHTTPServer`, which handles every connection on a new thread, so the SQLite
connection is opened with `check_same_thread=False`. Without it the connection was created in
whichever thread answered the *first* question and every later request raised
`ProgrammingError`; `socketserver` logs the traceback and closes the socket **with no HTTP
response**, so the UI answered exactly one question and then failed silently. The failure mode
is unusually well camouflaged — a single smoke test passes — so the regression test sends
repeated sequential *and* concurrent requests and asserts no traceback reached the log.
Sharing is safe because `sqlite3.threadsafety` is SERIALIZED on CPython's bundled SQLite, and
`do_POST` additionally holds a lock for the whole `ask()`.

**Routing is on the path, not the request target.** `self.path` is the whole request target
including any query string, so comparing it directly 404s `/healthz?x=1` — an ordinary thing for
a health probe to send. Handlers compare `urlsplit(self.path).path`, which keeps `/healthz/`
(strict) a 404 while query strings stop mattering.

**`serve()` refuses a port that already has a listener, and the reason is Windows.** `HTTPServer`
sets `allow_reuse_address = True`. On Unix `SO_REUSEADDR` means "ignore `TIME_WAIT`" and a
duplicate bind still fails with `EADDRINUSE`; on Windows it additionally lets a *second* socket
bind a port with a live listener. So without a guard a second `run.py serve` does not crash — it
binds, prints its banner, and connections are delivered to whichever socket the kernel picks. The
user restarts the server and silently keeps talking to the stale process, which is considerably
worse than a refusal. `serve()` therefore probes the port with a short `connect_ex` before
binding and raises an `OSError` naming `--port`. Probing rather than clearing `allow_reuse_address`
is deliberate: turning that flag off would also break rebinding a port left in `TIME_WAIT` by a
quick restart.

Binding a non-loopback host prints a warning to stderr. The spec's own pitfall is that this is a
prototype with no auth and no rate limiting, so `--host 0.0.0.0` publishes an unauthenticated
query endpoint. There is deliberately no CORS and no auth here (PRD §15 Q5).

Any unexpected exception in `do_POST` is returned as a 500 with the error type and message
rather than being allowed to escape to `socketserver`, which would close the socket with no
response and leave the cause visible only in a console the user may not have open.

**Escaping has two contexts, and only one of them is `html.escape`.** The welcome line, example
buttons and disclaimer are interpolated into markup and get `html.escape`. The example list is
interpolated into a `<script>` element, where `html.escape` is the wrong tool and `json.dumps`
alone is *not enough*: JSON does not escape `<`, `>` or `&`, so an example containing
`</script>` closes the element and injects markup. `_json_for_script()` rewrites those three
characters — plus U+2028/U+2029, which are line terminators to a JS parser but legal inside a
JSON string — as `\uXXXX`. That is safe post-encoding precisely because none of them are JSON
structural characters, so each one is necessarily inside a string literal where the JS parser
decodes it back. Backslashes are deliberately left alone; `json.dumps` already emitted them, and
doubling them again would round-trip to a literal double backslash. The examples are constants
today, so this is a latent hole rather than a live one — but the examples are exactly the thing
that gets edited when a scheme is added.

On the client, `esc()` covers text and attribute values; the citation additionally goes through
`safeUrl()`, because escaping quotes stops a value breaking out of `href="…"` but does not stop
a `javascript:` URL from being navigated to. Citations are allow-listed server-side, so that is
defence in depth.

**Two logging requirements that look like one.** `log_message` is suppressed so the console
stays readable, and no query body is ever logged. They are separate claims and are tested
separately: `http.server` logs the *request line* but never the body, so a test that only greps
the log for a query marker passes whether or not `log_message` is suppressed. The suppression
test instead asserts the access log is absent and that the console holds exactly the two banner
lines. Because the server is killed rather than shut down cleanly, the test harness launches it
with `-u`; without unbuffered output a block-buffered banner is lost and the assertion would
silently pass on an empty file.

---

## 7. Safety architecture — defence in depth

This is the core of the design. Each product guarantee is enforced at **three independent
admission points**, and the layers use different mechanisms so that a gap in one is unlikely
to be a gap in all three.

```
   SOURCE LAYER          QUERY LAYER           OUTPUT LAYER
   ┌──────────────┐      ┌──────────────┐       ┌──────────────┐
   │ Stage 0      │      │ Stage 5      │       │ Stage 8      │
   │ registry     │      │ scrub_pii()  │       │ G1 PII       │
   │ allow-list   │      │ raw_query="" │       │ G4 advice    │
   │ no blog/news │      │ 3 gates,     │       │ G5 perf      │
   │ pointer vs   │      │ ordered      │       │ G2 cite      │
   │ quotable     │      │              │       │ G3 length    │
   ├──────────────┤      ├──────────────┤       │ G6 fresh     │
   │ Stage 2      │      │ Stage 6      │       │ G7 notice    │
   │ field WHITELIST│    │ kind GATE    │       └──────────────┘
   │ → returns    │      │ (negative    │
   │   never      │      │  constraints)│
   │   admitted    │     │ field FILTER │       Different regex set,
   └──────────────┘      └──────────────┘       opposite direction
      Structural:         Structural:          (output, not input)
      data never          wrong kind
      enters              cannot win
```

| Guarantee | Structural mechanism | Mechanical mechanism | Verified by |
|---|---|---|---|
| No performance claim (FR-4) | returns never enter the corpus (stage 2 whitelist) | stage-5 gate + stage-8 G5 | 5 policy cases; `test_facts_schema_drops_performance_fields` |
| No advice (FR-3) | comparison impossible with one AMC; procedures are pre-written non-advice | stage-5 gate + stage-8 G4 | 7 policy cases + template identity |
| No PII (FR-5) | `raw_query` invariant; nothing user-specific written to disk | stage-5 scrub + stage-8 G1 | `test_pii_never_appears_in_any_output` |
| One allow-listed citation (FR-2) | citation copied from chunk `url`, never generated | stage-8 G2 membership test | allow-list test over every golden query |
| Public sources only (FR-8) | registry is closed; pointer-only cannot be quoted | G2 | `test_no_third_party_blog_or_news_in_registry` |
| Right *kind* of answer (FR-1) | topic exclusivity + kind gate | stage-7 field filter | 13 golden cases |

The important property: **stage 5 and stage 8 are independent implementations.** Different
regex sets, operating on opposite sides of the model. A phrasing that evades an input gate
still has to survive an output scan to reach a user.

"Independent" has to mean *implemented* independence, not merely two call sites. Stage 8
once imported `intent.PII_PATTERNS` directly, which left the two layers sharing one set of
objects: any gap in the input gate was then automatically a gap in the output gate, and the
property above was false while still reading true in this table. `guardrails.py` now builds
its own nets, an AST test forbids the import, and a differential test asserts the two
disagree on inputs the stage-5 net misses.

---

## 8. Open architectural decisions — D-1 and D-2

PRD §15 open question 1 is the one blocking sign-off: the brief mandates
`sentence-transformers/all-MiniLM-L6-v2` and **ChromaDB**; the build uses feature-hashed
vectors and SQLite FTS5. This section states the seams precisely so the decision can be made
on cost rather than on principle.

### 8.1 D-1 — Embedding model seam

**The contract.** `indexing.HashedVectoriser` exposes exactly what the rest of the system
depends on:

```python
dim: int
fit(corpus: list[str]) -> Self
transform_one(text: str) -> list[float]
transform(texts: list[str]) -> list[list[float]]
idf: dict[str, float]          # persisted to index_meta
ngram: int
```

**Who actually depends on it.** Only `indexing.py` itself. `retrieval.py` imports
`load_vectoriser`, `all_vectors` and `cosine` — all model-agnostic — and its
`HashedVectoriser` / `_pack` imports are vestigial. **Retrieval does not need to change.**

**The seam is implemented, not hypothetical (since PRD-D1 resolution, see below).** The
swap is wired today:

| File | What is in place |
|---|---|
| `indexing.py` | `SentenceTransformerVectoriser` + a `get_vectoriser(cfg)` factory keyed on `RetrievalConfig.embed_model`; `load_vectoriser()` reconstructs whichever model the index was built with (`index_meta['config']`), defaulting to hashed for pre-existing indexes |
| `indexing.py` | `build_index()` computes the vectors **before** touching the on-disk index, so a build that requests an unavailable HF model fails loudly and leaves the previous index intact |
| `retrieval.py` | **No change** — still uses `load_vectoriser`, `all_vectors`, `cosine` |
| `config.py` | `RetrievalConfig.embed_model` (`hashed` default, overridable via `MF_FAQ_EMBED_MODEL`); the value is read at instantiation, so `.env`/environment drives every `RetrievalConfig()` |
| `.env` / `.env.example` | `MF_FAQ_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2` to select the mandated 384-dim model |
| `requirements.txt` | `sentence-transformers` (+ `torch`) documented as the optional install — **this is the real cost** |
| Tests | `TestEmbeddingFactory` covers the factory, the index-meta round-trip, the deliberate ImportError on a missing package, and the no-clobber guarantee; hashed-specific tests are gated on the index's recorded model |
| CLI | `python run.py build` on an unrequestable HF model raises *with a fix hint* instead of silently degrading to hashed |

**Switching the live index:** set the env var, `pip install sentence-transformers`, re-run
`python run.py build`. Golden + policy suites should pass unchanged; expect BM25/dense ratio
to need retuning.

**What is gained:** genuine semantic matching; paraphrase and cross-vocabulary robustness the
hashed index cannot provide; the char-4-gram hack becomes unnecessary.

**What is lost:** the zero-dependency property (NFR-1/NFR-7) — `torch` is a ~2 GB install and
a model download on first run. A reviewer can no longer clone-and-run in one command offline.

**Recommendation:** the current index is adequate for 114 short factual chunks. If the corpus
grows past ~10⁴ chunks, or if evaluation shows paraphrase failures, take the swap — it is
contained to `indexing.py` by design. Record the deviation in the PRD either way, so the
submission is not read as an oversight.

### 8.2 D-2 — Vector store seam

D-2 is really **two** decisions, and conflating them is the trap:

**(a) The vector store.** ChromaDB can take this half. Seam: `build_index()` writes and
`Retriever.__init__` / `DenseIndex` read. Cost: a new dependency, a second persistence
format, and loss of single-file inspectability.

**(b) The lexical store.** ChromaDB **cannot** take this half. FR-10 requires hybrid
retrieval, and Chroma has no native BM25/exact-match operator. FTS5 must stay regardless —
either as the primary store with vectors beside it (today's design), or as a sidecar
alongside Chroma.

**Therefore adopting Chroma does not remove SQLite; it adds a second store.** The question is
whether that complexity buys anything at this scale.

**Minimal change set:** `indexing.build_index` / `load_vectoriser` / `all_vectors` become
Chroma-backed; `bm25_search` and the `chunks` table stay in SQLite; `retrieval.DenseIndex`
swaps its linear scan for a Chroma query. `retrieval.py` changes; stages 0–3, 5, 7, 8 do not.

### 8.3 The scale argument (P9)

At 114 chunks, a dense query is a linear cosine scan over 114 × 512 float32 values — roughly
55 KB, sub-millisecond, no index structure required. A vector database is a **scale**
optimisation, and the corpus is three orders of magnitude below the threshold where one pays
for itself. The same is true of ANN: exact search is both faster *and* more accurate at this
size. Adopting Chroma now would add a dependency, a second persistence format, and a
migration surface to optimise a problem that does not yet exist.

### 8.4 D-3 — Corpus provenance (not a decision, a debt)

The quotable corpus is Groww, not `hdfcmf.com`. The brief supplied the Groww URLs, they are
the only in-scope pages that are server-rendered and machine-verifiable, and Groww is a
SEBI-registered investment adviser platform rather than a blog — so FR-8 holds as written.

But the *authoritative* document for exit load, the full fee schedule and the riskometer is
the AMC's own SID/KIM/factsheet, registered as **pointer-only** and quoted nowhere. This is
the direct cause of D-4: **the riskometer is answered by a pointer, not a value**, because
the pages that publish it are SPAs a plain HTTP client cannot read.

Closing it requires either a JavaScript-rendering fetcher or a human-supplied export of the
AMC PDFs — a scoping decision (PRD §15 Q2), not a code change. The registry is the only thing
that would need to change to swap corpora.

---

## 9. Requirement → component traceability

| PRD req | Primary component | Supporting | Verified by |
|---|---|---|---|
| FR-1 factual answering | `synthesis.py` | `parsing.normalise_facts`, `chunking.FACT_SPECS` | 13 golden cases |
| FR-2 one source link | `guardrails.G2` | `sources.ALLOWED_URLS`, `Chunk.url` | allow-list test over all golden queries |
| FR-3 advice refusal | `intent.ADVICE_PATTERNS` | `guardrails.G4`, `REFUSAL_BODIES` | 7 policy cases + template identity |
| FR-4 performance refusal | `intent.PERFORMANCE_PATTERNS` | `parsing.normalise_facts` (drop), `guardrails.G5` | 5 policy cases; `test_facts_schema_drops_performance_fields` |
| FR-5 no PII | `intent.scrub_pii` | `raw_query` invariant, `guardrails.G1` | `test_pii_never_appears_in_any_output` |
| FR-6 freshness | `guardrails.G6` | `Chunk.last_updated` provenance | `test_render_always_carries_source_and_freshness` |
| FR-7 ≤3 sentences | `guardrails.G3` | `synthesis._trim_to_sentences` | `test_sentence_cap_is_enforced` |
| FR-8 public sources only | `sources` registry | `pointer`/`quotable` flags, G2 | `test_no_third_party_blog_or_news_in_registry` |
| FR-9 graceful failure | `ingest.FetchResult` | `pipeline.build` continues, `quarantined` | build output on 2 failing sources |
| FR-10 determinism | `ingest` content-hash cache | `index_meta` persisted IDF, extractive path | `MF_FAQ_OFFLINE=1` rebuild |
| FR-10 stages 0–9 | one module per stage | per-stage CLI inspectors | 204 tests |
| FR-10 field hint as filter | `intent.FIELD_HINTS` | `synthesis._fact` filter, `retrieval._field_boost` | golden case "who manages" |
| FR-10 kind gate | `retrieval._kind_gate` | `NON_FACT_TOPICS`, `PROCEDURAL_GROUP` | procedure + riskometer golden cases |
| FR-10 MMR | `retrieval._mmr` | stored vectors reused | `bench.py` (49 ms warm) |
| FR-10 explainability | `retrieval.explain` | `Scored.reasons` | `run.py stage6` |
| FR-11 guardrails G1–G7 | `guardrails.enforce` | applied to LLM path too | 15 policy cases |
| §9 UI | `ui.py` | `__init__.WELCOME` / `EXAMPLE_QUESTIONS` | manual + `DISCLAIMER.txt` |
| NFR-1/7 stdlib-only | `requirements.txt` | `numpy` optional only | quick start |
| NFR-2 latency <150 ms | `retrieval` | vector reuse in MMR | `bench.py` |
| NFR-4 correctness | `eval.py` | — | 28/28 |
| NFR-5 regression | `tests/` | — | 204/204 |

---

## 10. Operational characteristics

**Measured** (this machine, warm index): mean query latency **49 ms**; first cold call
**258 ms**; full build with network **9-13 s**; eval **28/28** in **1 s**; test suite
**204 tests in ~78 s**.

The suite is dominated by three full index builds (~30 s) plus a booted web server, both of
which are the point rather than overhead: a rebuild over an existing index, and a threaded
serve that has to be exercised more than once to be worth anything. The query-side tests
themselves run in about 2 s.

**Degradation matrix.**

| Failure | Behaviour | User-visible? |
|---|---|---|
| A source page changes layout | Fewer facts extracted; sections still index | Coverage drops, no crash |
| A source 404s or blocks | `FetchResult(ok=False)`; build continues | Listed in `quarantined` |
| All sources unreachable | Build produces a thin index | Build output states chunk count |
| Index missing | `FAQPipeline.retriever` raises with the fix command | Actionable error |
| Index held open by a running `serve` | `build_index` refuses with the process to stop | Actionable; no half-applied build |
| A second `serve` request after the first | Shared connection, `check_same_thread=False` | None — 8 sequential + 12 concurrent requests pass |
| A second `serve` on a live port | Refused before binding, names `--port` | Actionable; no silent double-bind |
| A health probe sends `/healthz?x=1` | Routed on the path component | None |
| LLM provider down / rate-limited | Falls back to extractive draft | Fluent prose lost, answer intact |
| LLM returns junk | `_parse` fails → extractive fallback | Invisible |
| Golden figure drifts, URL unchanged | Eval `contains:<sub>` fails | Caught before release |
| LLM emits advice | Stage-8 G4 converts to refusal | Visible, correctly |
| Index goes stale | Freshness stamp still shows the true source date | Stamp is honest by construction |

**Scaling limits.** The dense stage is a linear scan, O(n·d) per query. At 114 chunks that is
sub-millisecond; at 100k chunks it would need ANN. BM25 is handled by SQLite. Both are noted
rather than pre-optimised, per P9.

**Security posture.** Binds `127.0.0.1`. A `threading.Lock` serialises whole `ask()` calls,
so the shared SQLite connection has a single writer; statement-level safety is SQLite's own
SERIALIZED threading mode. Not rate-limit hardened and not intended for public exposure —
stated in PRD §12 and §13. User queries are **not logged** by the pipeline; the HTTP handler
suppresses `log_message`.

---

## 11. Extension points

Ordered by value per unit of risk.

| Change | Where | Risk | Notes |
|---|---|---|---|
| Add a 6th scheme | `sources.SCHEMES` + registry entry | Low | Aliases matter for stage-5 resolution |
| Swap the embedding model | `indexing.get_vectoriser` | **Low** (§8.1) | `retrieval.py` unchanged |
| Add a fact field | `parsing.normalise_facts` + `chunking.FACT_SPECS` | Low | Keep the whitelist discipline |
| Add a topic | `intent.TOPIC_RULES` + `FIELD_HINTS` + `TOPIC_LABELS` | Medium | Check exclusivity groups |
| Make a procedure quotable | registry `quotable` flag | Medium | Only if the page is actually machine-readable |
| Add a refusal kind | `intent` patterns + `REFUSAL_BODIES` + `REFUSAL_LINKS` | Medium | Keep it a **template** |
| Port the vector store to Chroma | `indexing` + `retrieval.DenseIndex` | **High** (§8.2) | SQLite FTS5 must stay |
| Multilingual routing | `intent` regex sets | High | Every rule set is English-only |

The rule when extending: **a new source must be classified pointer-vs-quotable before it is
fetched, and a new refusal must be a template.** Those two invariants are what keep §7's
guarantees true as the system grows.

---

## 12. Open architectural questions

1. **D-1 / D-2 (blocking sign-off).** Accept the embedding and vector-store substitutions, or
   reconcile to the brief's `all-MiniLM-L6-v2` + ChromaDB? §8 gives the change set and the
   cost for each. *Recommend: accept and record*, with the `indexing.py` seam left clean.
   Variant A is now built and audited, so the decision is concrete rather than theoretical:
   114 chunks, 512-d, and the `default_idf` and float32-accumulator defects found during that
   audit (§6.5) are fixed and pinned by tests. Neither is a reason to prefer the brief's stack
   — they are bugs any implementation of the same design would have.
2. **D-3 closure.** Is a JS-rendering fetcher or a human-supplied AMC PDF export acceptable,
   so the riskometer can be answered with a value rather than a pointer? (PRD §15 Q2)
3. **Is the Groww-hosted corpus acceptable for the submitted prototype?** FR-8 permits it
   and no blog/news source is used. (PRD §15 Q3)
4. **English-only routing** — acceptable, or is multilingual scope expected? §11 rates this
   the highest-risk extension.
5. **Hosting intent**, given the prototype binds `127.0.0.1` and is not rate-limit hardened.
   Any real deployment needs a reverse proxy, request limits, and an auth story for an
   assistant that refuses PII — a control that implies a service that may be targeted.
6. **Scheduled rebuilds.** Freshness is build-time only; there is no scheduler. What is the
   intended re-index cadence, and who is paged when a source starts failing?
