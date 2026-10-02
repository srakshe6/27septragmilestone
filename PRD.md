# PRD — Mutual Fund Facts Assistant (facts-only RAG chatbot)

| Field | Value |
|---|---|
| Document | Product Requirements Document, v1.0 |
| Date | 2026-09-27 |
| Source | `problemstatement.txt` |
| Status | Implemented — see §12 for as-built verification |
| Owner | RAG project team |
| Companion docs | `README.md` (setup/scope/limits), `docs/ARCHITECTURE.md` (stage design), `SOURCES.md` (citation allow-list), `DISCLAIMER.txt` (UI strings) |

---

## 1. Background and problem

Retail investors and support/content teams repeatedly ask the same small set of factual
questions about mutual fund schemes: expense ratio, exit load, minimum SIP, ELSS lock-in,
riskometer level, benchmark, and how to download statements. Today these are answered by
reading PDFs, or by a human agent. The answers are slow to produce, inconsistent between
agents, and — critically — sit right next to the boundary of regulated investment advice.

The product is a **retrieval-grounded FAQ assistant that answers published facts about a
narrow, fixed set of schemes, and nothing else.** Its value is not breadth of financial
knowledge; it is that every number it emits is traceable to one public source, that it
declines opinion questions deterministically, and that it cannot leak personal data.

The defining constraint is **risk, not accuracy**. A wrong expense ratio is a bad answer.
A confident "you should buy this" is a compliance problem. Therefore the requirements
below are weighted toward *provable restraint* over helpful breadth.

## 2. Goals

| # | Goal | Measure of success |
|---|---|---|
| G1 | Answer factual scheme questions from official public pages | Correct figure + correct source in ≥ 90% of golden factual queries |
| G2 | Every answer carries exactly one source link | 100% — no exceptions, enforced in code |
| G3 | Never give advice, never make performance claims | 100% refusal on all policy queries, verified against canonical templates |
| G4 | Never accept, store or echo PII | 100% — PII scrubbed before routing; nothing user-specific written to disk |
| G5 | Answers are short and scannable | ≤ 3 sentences, hard-truncated |
| G6 | Transparency about staleness | Every answer stamped `Last updated from sources: <date>` |
| G7 | Demonstrable RAG architecture | Each RAG stage separately runnable and inspectable |

## 3. Non-goals

These are explicitly **out of scope**. Saying so is a requirement, because scope creep here
is the primary risk to G3.

- No investment advice, recommendations, ratings, rankings or scheme comparison.
- No computation, restatement or comparison of returns or any performance figure.
- No cross-scheme or cross-AMC comparison. One AMC, five schemes.
- No portfolio advice, allocation, timing or tax-planning judgement.
- No personal/account-specific servicing. The assistant is not a login surface.
- No chat history, accounts, user profiles or analytics on users.
- No live NAV streaming or market data beyond what a cited page already publishes.
- Regular-plan and dividend-variant figures (only direct-growth is in scope).

## 4. Users and primary use cases

| User | Need | Representative question |
|---|---|---|
| Retail investor comparing schemes | "Before I read the factsheet, what are the headline terms?" | "What is the exit load on HDFC Small Cap Fund?" |
| Retail investor | "Is this an ELSS and what does that lock in?" | "Is there a lock-in period in HDFC ELSS Tax Saver Fund?" |
| Support / content team | "Answer the same 20 questions a day, identically, with a citable link." | "Minimum SIP for HDFC Flexi Cap Fund?" |
| Support / content team | "Where does the customer actually download this?" | "How do I download a capital gains statement?" |
| Any user | "I want a recommendation." | "Should I buy HDFC Small Cap Fund?" → **refused** |

## 5. Scope

**AMC in scope:** HDFC Mutual Fund (HDFC Asset Management Company). One AMC by design.

**Schemes in scope** (direct-growth variants, as named in the brief):

| # | Scheme | Category | Brief-supplied source |
|---|---|---|---|
| 1 | HDFC Large Cap Fund (Direct - Growth) | Large cap | `groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth` |
| 2 | HDFC Flexi Cap Fund (Direct - Growth) | Flexi cap | `groww.in/mutual-funds/hdfc-equity-fund-direct-growth` |
| 3 | HDFC ELSS Tax Saver Fund (Direct Plan - Growth) | ELSS / tax-saver | `groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-growth` |
| 4 | HDFC Small Cap Fund (Direct - Growth) | Small cap | `groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth` |
| 5 | HDFC Balanced Advantage Fund (Direct - Growth) | Hybrid / Balanced Advantage | `groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth` |

**Answerable topics (the "supported intent" set):** expense ratio, exit load, minimum SIP,
minimum lump-sum, ELSS lock-in, benchmark, riskometer, NAV, fund manager, RTA, ISIN, launch
date, AUM, and the procedural topics — account statement, capital gains statement, tax
documents, fee/charges schedule.

**Everything else is out of scope** and must be declined or answered with a pointer, never
guessed.

## 6. Functional requirements

Each requirement is traceable to `problemstatement.txt`. "AC" = acceptance criterion, and
each AC must be backed by an automated check.

### FR-1 — Factual answering
The assistant answers factual questions about the five in-scope schemes.
*AC:* a golden set of factual queries returns the correct figure, the correct source URL,
and no fabricated content. Verified by the stage-9 golden suite.

### FR-2 — Mandatory single citation
Every answer includes exactly one source link.
*AC:* the URL is a member of the registry allow-list; a citation outside the allow-list is
dropped and the response is flagged. Verified by an allow-list membership test over every
golden query.

### FR-3 — Advice refusal
Opinionated and portfolio questions ("Should I buy/sell?", "which is better?", "is now a
good time?") are refused with a polite, facts-only message plus a relevant educational link.
*AC:* refusal text is byte-identical to the canonical advice template; the correct
refusal kind is returned. Verified by 7 policy cases.
*Design note:* the refusal is a **template, not free text**, and is verified by template
identity rather than by absence of advice keywords — the advice template legitimately
contains the phrase "whether you should buy" in order to decline.

### FR-4 — Performance-claim refusal
The assistant does not compute, compare, restate or claim returns.
*AC:* performance questions return the canonical performance-refusal template. Verified by
5 policy cases.
*Design note:* performance is checked **before** advice, because "compare the returns of A
and B" trips both rule sets and the no-returns refusal is the more useful and more
deterministic answer.

### FR-5 — PII handling
The assistant does not accept or store PAN, Aadhaar, account/folio numbers, OTPs, email
addresses or phone numbers.
*AC:* PII patterns are scrubbed **before** routing, the raw query is discarded (not carried
into the decision, evidence trail or any log), and the emitted response never echoes the
input PII. Verified by a test asserting PII never appears in any output.

### FR-6 — Freshness transparency
Every answer carries `Last updated from sources: <date>`, taken from the chunk's source
provenance, not from the current clock.
*AC:* the string is present on 100% of responses, including refusals and LLM output.
Verified by a render-level test.

### FR-7 — Answer length
Answers are ≤ 3 sentences.
*AC:* over-length output is hard-truncated to 3 sentences. Verified by a sentence-cap test.

### FR-8 — Public sources only
Only official public pages are used. No login, no paywall, no app back-end, no screenshots
of internal systems, and **no third-party blog or news article** as a source of fact.
*AC:* the registry is asserted to contain no blog/news domain. The allow-list is the
enforcement mechanism, not a documentation convention.

### FR-9 — Graceful source failure
A source that cannot be fetched degrades coverage; it must not take the assistant down.
*AC:* a failed fetch is recorded with status/bytes/SHA-256/error and the build continues.

### FR-10 — Deterministic, offline operation
The default answer path requires no API key and no network at query time, and produces the
same answer for the same query.
*AC:* identical input yields identical output; the build is replayable from the raw cache
with networking disabled.

## 7. RAG architecture requirements

The brief mandates explicit, separately inspectable stages. One module per stage, each with
one job, one inspectable artefact, and one test.

| Stage | Name | Artefact | Inspect via |
|---|---|---|---|
| 0 | Source registry / citation allow-list | `src/mf_faq/sources.py` | `python run.py sources` |
| 1 | **Data ingestion** (loading) | `data/raw/*.html` + `data/fetch_manifest.json` | `python run.py build` |
| 2 | Parsing (HTML → text + structured facts) | `data/corpus.jsonl` | read directly |
| 3 | **Chunking** | `data/chunks.jsonl` | read directly |
| 4 | **Embedding + vector store** | `data/index/mf_faq.sqlite` | read directly |
| 5 | Query understanding (PII scrub, routing, refusal gate) | decision record | `python run.py stage5 "<q>"` |
| 6 | **Data retrieval** (hybrid rank) | ranked chunks + signal breakdown | `python run.py stage6 "<q>"` |
| 7 | Answer synthesis | answer + citation | `python run.py ask --json "<q>"` |
| 8 | Post-processing / guardrails (G1–G7) | final response + guard flags | `python run.py ask --debug "<q>"` |
| 9 | Evaluation | `data/eval_report.json` | `python run.py eval` |

### FR-10 — Pipeline shape requirements
* **Ingestion** must be replayable (content-hashed cache) and fault-tolerant.
* **Parsing** must extract both rendered visible text *and* embedded structured JSON, because
  they fail in different ways; the structured path answers anything a number can.
* **Chunking strategy is decided from the data, not fixed a priori.** The decision and its
  rationale must be documented. The implemented strategy: **three chunk families whose shape
  follows question shape** — `fact` (one atomic self-contained sentence per scheme-field,
  carrying scheme name + value + unit so the citation is verifiable by reading the chunk),
  `section` (4-sentence prose windows as a lexical backstop), and `procedure` (authored
  non-numeric text for the questions that have no number, each bound to an official
  destination page).
* **Performance data must be dropped at ingestion.** The source payload contains return
  figures; the normaliser must whitelist fields and never copy them, so no performance number
  exists in the corpus and none can reach an answer even under prompt injection.
* **Retrieval must be hybrid**, because financial jargon defeats either signal alone: an
  exact-match lexical signal for "ELSS", "exit load", "riskometer", plus a dense signal for
  morphology and paraphrase.
* **Field hints must be a hard constraint, not a score boost.** "Minimum SIP" vs "minimum
  lump-sum" and "who manages" vs "RTA" are different facts under the same topic; a soft
  boost is not sufficient, so a field hint filters candidates at synthesis.
* **A kind gate** must encode negative constraints a relevance score cannot express (e.g. a
  per-scheme number must not answer a portfolio-wide procedural question; the three
  statement procedures are mutually exclusive destinations).
* **MMR re-ranking** must remove near-duplicates, because each scheme yields many
  near-identical template sentences.
* **Retrieval must be explainable**: every intermediate signal per hit must be inspectable.
* **Synthesis** must default to deterministic extractive composition from a source of record
  rather than free generation, so a number cannot be paraphrased. An LLM rewrite path is
  permitted only if its output passes the identical stage-8 checks.
* **Guardrails are applied to every response**, including LLM output, and a failed check is
  never silently downgraded: PII/advice/performance convert to a refusal; citation/length
  are repaired; freshness/disclaimer are appended. Input-side and output-side checks must be
  independent implementations.

### FR-11 — Guardrail set
| ID | Check | Failure action |
|---|---|---|
| G1 | PII in output body | → PII refusal, never echo |
| G2 | Citation not in allow-list | → drop citation, flag |
| G3 | > 3 sentences | → truncate to 3 |
| G4 | Advice verbs in body | → advice refusal |
| G5 | Performance claim in body | → performance refusal |
| G6 | Missing freshness stamp | → stamp from provenance |
| G7 | Missing disclaimer | → attach |

## 8. Non-functional requirements

| ID | Requirement | Target | Verified |
|---|---|---|---|
| NFR-1 | Setup friction | One command, no API key; `numpy` optional only | `README.md` quick start |
| NFR-2 | Warm query latency | < 150 ms mean | `python bench.py` (measured 49 ms warm / 258 ms cold) |
| NFR-3 | Build determinism | Identical index from identical inputs | content-hashed cache + `MF_FAQ_OFFLINE=1` |
| NFR-4 | Correctness | 100% of eval checks pass | `python run.py eval` → 28/28 |
| NFR-5 | Regression safety | Full suite green | `python -m unittest discover -s tests` → 184 tests |
| NFR-6 | Offline query time | No network, no key | deterministic extractive path |
| NFR-7 | Dependency footprint | Stdlib-only core | `requirements.txt` |
| NFR-8 | Data at rest | Public HTML only; nothing user-specific ever written | pipeline logs no queries |
| NFR-9 | Exposure | Localhost prototype, not for public exposure | binds `127.0.0.1` |

## 9. UI requirements

- A **tiny** web UI: welcome line, 3 example questions, and the note
  **"Facts-only. No investment advice."**
- Each answer renders the answer text, the single source link, the freshness stamp, and the
  disclaimer.
- Refusals render with the same chrome as answers — a refusal is a first-class response, not
  an error.
- A long-form footer disclaimer stating: facts-only, no advice; sources listed and dated; no
  recommendation/comparison/rating; no return computation; and that PAN, Aadhaar,
  account/folio numbers, OTPs, emails and phone numbers are neither requested nor stored.
- A terminal path (`ask`) with the same guarantees, for reviewers who will not start a server.

## 10. Data and source policy

1. **Public sources only.** No login, paywall, app back-end or screenshot of an internal
   system.
2. **Pointer vs. quotable.** A source is either *quotable* (text may be quoted) or *pointer
   only* (cited as the official place to read or download something, quoted nowhere). Sources
   that are client-rendered or bot-protected are pointer-only. This is what makes it
   structurally impossible for an unverifiable page to put text into an answer.
3. **One AMC.** No cross-AMC comparison.
4. **No performance data ingested.**
5. **Privacy.** The AMC portal and RTA statement pages are cited as navigation destinations
   only. Account-specific requests are routed to the official portal, never serviced here.

## 11. Success metrics and acceptance

The build is accepted when all of the following hold:

| # | Criterion | Threshold | Status |
|---|---|---|---|
| A1 | Golden factual queries: correct figure + correct source + ≤3 sentences + disclaimer + freshness | 13/13 | **met** |
| A2 | Policy queries: correct refusal kind + byte-identical canonical template | 15/15 | **met** |
| A3 | Total eval checks | 28/28 | **met** |
| A4 | Unit + end-to-end tests | all green (46) | **met** |
| A5 | Every response carries a citation, a freshness stamp and the disclaimer | 100% | **met** |
| A6 | PII never appears in any output | 100% | **met** |
| A7 | No blog/news domain in the registry | 0 | **met** |
| A8 | Sample Q&A of 5–10 queries generated from the live pipeline, not hand-written | 10 queries | **met** |
| A9 | Source list deliverable (CSV + MD) | present | **met** |
| A10 | Working prototype | runnable | **met** |
| A11 | README with setup, scope, known limits | present | **met** |
| A12 | Riskometer answered with a value | — | **not met — see §12** |

## 12. As-built status, deviations and open gaps

The requirements above are implemented. Four things a reviewer must know.

### D-1 — Mandated embedding model was substituted *(deviation, decision needed)*
The brief specifies `sentence-transformers/all-MiniLM-L6-v2`. As built, the dense index uses
**IDF-weighted feature-hashed vectors** (512-dim, word unigrams + character 4-grams, signed
`blake2b` hashing, cosine), implemented in pure Python with `numpy` used only to accelerate.

*Rationale:* zero dependencies, so a reviewer can run the whole deliverable in one command
with no model download, and the vectors remain inspectable rather than a black box.
*Cost:* weaker than MiniLM on semantic similarity. This matters little for a 149-chunk
corpus of short factual statements; it would matter at 100k+ chunks.
**Status: RESOLVED — seam provided (see below).** The two vectorisers now coexist behind
`indexing.get_vectoriser` / `load_vectoriser`; retrieval is model-agnostic and unchanged.
Set `MF_FAQ_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2` in `.env` and re-run
`python run.py build` (requires `sentence-transformers` / `torch`) to use the mandated
384-dim model. A build that requests an HF model without the package installed fails with an
actionable message, and — since the vectoriser completes before the old index is replaced —
leaves the previous index intact.

### D-2 — Mandated vector database was substituted *(deviation, decision needed)*
The brief specifies ChromaDB. As built, the index is **SQLite FTS5 (BM25) plus a hashed
dense-vector table**, in one file (`data/index/mf_faq.sqlite`), using stdlib `sqlite3`.

*Rationale:* the hybrid requirement needs an exact-match lexical index, which FTS5 provides
natively and ChromaDB does not; a single stdlib file keeps the deliverable dependency-free
and inspectable with any SQLite client.
**Status: open decision.** Accept, or port the two index halves to ChromaDB + a BM25 sidecar.

### D-3 — Corpus source is Groww, not the AMC's own documents *(limitation)*
The five URLs named in the brief are Groww scheme pages. Groww is a SEBI-registered
investment adviser platform, not a blog, and no third-party blog or news article is used, so
FR-8 holds as written. But the *authoritative* document for exit load, the full fee schedule
and the riskometer is the AMC's own SID/KIM/factsheet. `hdfcmf.com`,
`investor.hdfcfund.com` and `camsonline.com` are bot-protected, client-rendered SPAs that a
plain HTTP client cannot read, so they are registered as **pointer-only** sources and quoted
nowhere. **Production use should replace the Groww pages with the AMC documents; only the
registry needs to change, not the code.**

### D-4 — Riskometer is answered by a pointer, not a value *(functional gap)*
The five scheme pages do not publish a riskometer level, and the AMC pages that do are
unreadable by a plain HTTP client. Rather than guess, the assistant states that the level is
not in its sources and points at the AMC's monthly factsheet, which is where SEBI requires it
to be published. This is the clearest functional gap in the build, and it is the direct
consequence of D-3.

### Other stated limits
- **No comparison** across schemes or AMCs; one AMC, direct-growth only.
- **Rules are regex, not learned.** Auditable and testable, which is the point for a
  safety-critical facts tool, but English-only and vulnerable to genuinely novel phrasings.
  Mitigation is the independent stage-8 output scan; the residual failure mode is a wrongly
  *answered* question in novel phrasing.
- **Freshness is build-time only.** The stamp reflects the source page date, not a live
  re-check. There is no scheduler; `build --refresh` re-fetches.
- **The raw cache is committed-friendly but public.** It contains only public HTML; nothing
  about a user is ever written to disk.
- **Not rate-limit hardened.** Localhost prototype; not intended for public exposure.
- **The three statement procedures need ongoing upkeep** — they are authored text, so a
  change to a portal's URL is a code change, not a re-index.

## 13. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| A confident wrong figure erodes trust in every answer | Medium | High | Single-sourced figures, mandatory citation, atomic fact chunks that are verifiable by reading the citation |
| Refusal gate misses a novel advice phrasing | Medium | **High** | Two independent nets (input gate + output scan); refusals are templates so behaviour is deterministic; stated as a known limit, not hidden |
| A client-rendered page silently yields empty content | High | Medium | Pointer-vs-quotable flag; fetch status visible in build output; graceful degradation (FR-9) |
| Scheme figures change and the index goes stale | High | Medium | Freshness stamp on every answer; documented `--refresh`; the sample Q&A is regenerated from the live pipeline so it cannot drift |
| Scope creep toward advice | Medium | **High** | Explicit non-goals (§3); advice/performance templates; comparison is structurally impossible with one AMC |
| PII reaches logs or disk | Low | **High** | Scrub before routing, raw query discarded, output re-scanned, test asserts absence from all output |
| A reviewer cannot run the build | Low | Medium | Stdlib-only, content-hashed raw cache, offline rebuild mode |

## 14. Deliverables

| Deliverable (from the brief) | Artefact | Status |
|---|---|---|
| Working prototype link (app/notebook) or ≤3-min demo video | This repo: `python run.py serve` (UI) and `python run.py ask` (CLI) | delivered as a runnable app |
| Source list of the 5 URLs (CSV/MD) | `SOURCES.csv`, `SOURCES.md` | delivered |
| README with setup steps, scope (AMC + schemes), known limits | `README.md` | delivered |
| Sample Q&A (5–10 queries with answers + links) | `samples/sample_qa.md`, `samples/sample_qa.json` (10 queries, generated from the live pipeline) | delivered |
| Disclaimer snippet used in the UI | `DISCLAIMER.txt`, `src/mf_faq/config.py` | delivered |
| Demonstrable RAG stage architecture | `docs/ARCHITECTURE.md` + per-stage modules and CLI inspectors | delivered |
| Evaluation evidence | `python run.py eval` → `data/eval_report.json` (28/28) | delivered |

## 15. Open questions

1. **Accept the embedding and vector-store substitutions (D-1, D-2), or reconcile to the
   brief's `all-MiniLM-L6-v2` + ChromaDB?** The substitution is a deliberate
   dependency-free trade-off; reconciling is a contained change behind the existing
   vectoriser seam. **This is the one decision that blocks sign-off.**
2. Should D-3 be closed for production — i.e. is a JavaScript-rendering fetch (or a
   human-supplied export of the AMC SID/KIM/factsheet PDFs) acceptable, so the riskometer can
   be answered with a value?
3. Is the Groww-hosted corpus acceptable for the submitted prototype, given FR-8 permits it
   and no blog/news source is used?
4. Is English-only regex routing acceptable, or is a multilingual scope expected?
5. What hosting is intended, given the prototype binds `127.0.0.1` and is not rate-limit
   hardened?

---

## Appendix A — Traceability: requirement → source line → verification

| Req | `problemstatement.txt` | Verified by |
|---|---|---|
| FR-1 factual answering | L1, L13 | golden suite (`eval.py`) |
| FR-2 one source link | L1, L14 | `test_every_golden_query_is_answered_with_an_allowlisted_citation` |
| FR-3 advice refusal | L15 | 7 policy cases + template identity |
| FR-4 performance refusal | L20 | 5 policy cases + `test_performance_claim_becomes_refusal` |
| FR-5 no PII | L19 | `test_pii_never_appears_in_any_output` |
| FR-6 freshness stamp | L21 | `test_render_always_carries_source_and_freshness` |
| FR-7 ≤ 3 sentences | L21 | `test_sentence_cap_is_enforced` |
| FR-8 public sources only | L6, L18 | `test_no_third_party_blog_or_news_in_registry` |
| FR-10a–j RAG stages | L29–L31 | `docs/ARCHITECTURE.md`, per-stage CLI inspectors |
| FR-11 guardrails G1–G7 | L18–L21 | `guardrails.py` + eval policy suite |
| §9 UI: welcome + 3 examples + note | L16 | `ui.py`, `DISCLAIMER.txt` |
| §5 scope: 1 AMC, 3–5 schemes | L5, L7–L11 | `SOURCES.md`, `config.py` |
| Deliverables | L22–L27 | `README.md` §Deliverables map |

## Appendix B — Glossary

| Term | Meaning in this project |
|---|---|
| Fact chunk | One atomic sentence carrying scheme + field + value + unit, quotable alone |
| Pointer-only source | An official page cited as a destination; nothing quoted from it |
| Field hint | A stage-5 signal that filters synthesis candidates, rather than boosting a score |
| Kind gate | A negative constraint: what must **not** answer this topic |
| MMR | Maximal Marginal Relevance — re-ranking that drops near-duplicate chunks |
| Quotable | A source whose text may be reproduced in an answer |
| Stage 5 / 6 / 8 | Query understanding / retrieval / post-processing — the fault-localising inspectors |
