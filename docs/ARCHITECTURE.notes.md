> **SUPERSEDED SNAPSHOT - DO NOT USE AS REFERENCE.**
> Kept only for history. Replaced by `docs/ARCHITECTURE.md`.
> Its figures are known to be wrong: it states 15 golden / 13 policy
> (actually 13 / 15), 21 advice patterns (26) and 15 performance
> patterns (14), and mis-describes the FTS5 column weighting.

# Architecture — stage by stage

One module per stage, each independently runnable and inspectable:

| Stage | Module | CLI to inspect it |
|---|---|---|
| 0 source registry | `src/mf_faq/sources.py` | `python run.py sources` |
| 1 data ingestion | `src/mf_faq/ingest.py` | `python run.py build` |
| 2 parsing | `src/mf_faq/parsing.py` | `data/corpus.jsonl` |
| 3 chunking | `src/mf_faq/chunking.py` | `data/chunks.jsonl` |
| 4 indexing | `src/mf_faq/indexing.py` | `data/index/mf_faq.sqlite` |
| 5 query understanding | `src/mf_faq/intent.py` | `python run.py stage5 "<q>"` |
| 6 data retrieval | `src/mf_faq/retrieval.py` | `python run.py stage6 "<q>"` |
| 7 answer synthesis | `src/mf_faq/synthesis.py` | `python run.py ask --json "<q>"` |
| 8 post-processing | `src/mf_faq/guardrails.py` | `python run.py ask --debug "<q>"` |
| 9 evaluation | `src/mf_faq/eval.py` | `python run.py eval` |

The split is not decoration. Each stage has one job, an inspectable artefact, and a
test. When an answer is wrong, `stage5`/`stage6`/`--debug` localises the fault to a
stage instead of leaving you reading a prompt.

---

## STAGE 0 — Source registry (`sources.py`)

A frozen list of `Source` records, and the derived `ALLOWED_URLS` frozenset that the
guardrail stage validates every citation against. A URL that is not in this file can
never appear in an answer — the allow-list is the enforcement mechanism, not a
documentation convention.

Each source carries two flags that drive behaviour:

* `quotable` — may text be quoted from this source in an answer?
* `verified_at_build` — could a plain HTTP client actually retrieve it?

`quotable=False, verified_at_build=False` is the honest description of
`hdfcmf.com`, `investor.hdfcfund.com` and `camsonline.com`: they are official
destinations whose content is client-rendered, so they are cited as *where to go* and
quoted nowhere. This distinction is why the assistant can never put unverifiable text
in an answer, and it is stated in `SOURCES.md` rather than hidden.

`test_no_third_party_blog_or_news_in_registry` asserts no blog/news domain is present,
because the failure mode for this product is citing a content farm that rewrites a
factsheet.

---

## STAGE 1 — Data ingestion (`ingest.py`)

Fetches each registered URL, writes the bytes to `data/raw/<slug>.<hash>.html`, and
records provenance in `data/fetch_manifest.json` (status, byte count, SHA-256, fetch
timestamp, error).

Two properties matter:

* **Replayable.** The cache makes the whole build deterministic and lets stages 2–4
  run with no network. `MF_FAQ_OFFLINE=1` enforces that.
* **Never hard-crashes.** A failed fetch becomes a `FetchResult(ok=False, error=...)`
  and the build continues. A dead source degrades coverage; it does not take the
  assistant down. Two of the nine registered sources fail on a clean build for exactly
  this reason, and that is visible in the build output rather than silent.

---

## STAGE 2 — Parsing (`parsing.py`)

Two extractions, because they fail in different ways:

* **`visible_text`** — `html.parser`-based render with `<script>/<style>` stripped,
  whitespace normalised, and nav/footer chrome removed. Used for wording-level
  questions and as the lexical backstop.
* **`embedded_json`** — the `__NEXT_DATA__` payload these pages ship, which carries
  field-level scheme facts (expense ratio, exit load, minimum SIP, lock-in, benchmark,
  ISIN, NAV and NAV date). This is the higher-fidelity signal and answers anything a
  number can.

`normalise_facts()` is the security-relevant function in this stage: it whitelists the
fields it copies. The page payload also contains return figures, and they are
deliberately **not** copied, so no performance number exists anywhere in the corpus and
none can reach an answer. `test_facts_schema_drops_performance_fields` locks this in.

---

## STAGE 3 — Chunking (`chunking.py`)

**Chunk shape follows question shape.** "What is the exit load of HDFC Small Cap?"
wants one number, not a paragraph, so the chunker emits three families:

| Kind | Count | Purpose |
|---|---|---|
| `fact` | 66 | One atomic self-contained sentence per (scheme, field). Always carries an `answer_hint` and `meta.field`. |
| `section` | 34 | 4-sentence prose windows over the rendered page, with a topic label assigned only when an explicit phrase names the topic. Windows over `SECTION_MAX_TOKENS` (160) are dropped as bulk/tabular content. Lexical backstop. |
| `procedure` | 5 | Authored, non-numeric answer text for the questions with no number: account statement, capital gains, tax documents, riskometer, fee schedule. Each bound to an official destination page. |

Fact chunks are templated at build time, e.g.

```
The expense ratio of HDFC Large Cap Fund (Direct - Growth) is 1.03%. The expense
ratio is the annual percentage of the scheme's assets charged to the scheme,
deducted from the scheme's returns rather than billed separately.
```

Sentence 1 is complete and quotable on its own — scheme name, value, unit — so the
stage-8 citation can be verified by reading the chunk rather than trusting a
paraphrase.

Source wording is normalised but never reinterpreted. `_fmt_exit_load` fixes
capitalisation, spacing and one badly-ordered pattern:

```
raw:      "Exit Load for units in excess of 15% of the investment,1% will be
           charged for redemption within 1 year."
emitted:  "1% for units in excess of 15% of the investment, if redeemed within 1 year"
```

The rate, the threshold and the window are unchanged. Anything that would require
judgement is not normalised.

---

## STAGE 4 — Indexing (`indexing.py`)

Two indexes over the same chunks, stored in one SQLite file.

**Lexical (BM25).** SQLite FTS5, `unicode61` tokenizer, with `answer_hint` weighted
below `text`. Exact-match robust and self-explaining, which is what you want for
`ELSS`, `exit load`, `riskometer`.

**Dense (cosine).** IDF-weighted **feature-hashed** vectors, 512 dimensions, over word
unigrams plus character 4-grams, L2-normalised, stored as BLOBs. Implemented in pure
Python with `numpy` used only to accelerate. Signed hashing (`blake2b`) reduces
collisional bias; sublinear TF and IDF weighting stop common words dominating.

Why hashed rather than a trained embedding model: it gives the morphology tolerance
that pure BM25 lacks (`expense`/`expenses`, `riskometer`/`riskometre`) with **zero
dependencies**, which keeps the deliverable runnable by a reviewer in one command. The
cost is real and is recorded in the README's known limits — at 100k+ chunks a real
embedding model would win.

`build_index()` also persists the fitted IDF table, so `load_vectoriser()` reconstructs
the vectoriser at query time with no re-fit and no pickle.

---

## STAGE 5 — Query understanding (`intent.py`)

Runs before retrieval and decides, in order:

1. **PII scrub.** `scrub_pii()` removes PAN, Aadhaar, email, phone, OTP, folio/account
   and UPI/IFSC patterns, plus bare 10–12 digit numbers. On a hit the decision is
   `pii_refusal` and **`raw_query` is set to `""`** — the original string is not
   carried into the decision, the evidence trail or any log.
2. **Scheme resolution.** Longest-alias match over 5 schemes × ~9 aliases. `"HDFC
   Flexi Cap Fund"` wins over the bare `"equity fund"` alias.
3. **Topic routing.** Ordered regex rules, then **exclusivity**: the three
   statement/tax topics are alternative answers, not additive facets. Without this, a
   "tax statement" query drifts onto the capital-gains procedure because both mention
   statements. Only the first match in rule order survives.
4. **Field hints.** A more specific signal than topic: `minimum SIP` vs `minimum
   lump-sum`, `who manages` vs `RTA`, `ISIN`, `launch date`. See the note below.
5. **Refusal gates**, in this order: performance → bare-PII-keyword → advice.

**Why performance is checked before advice.** "Compare the returns of A and B" trips
both sets of patterns. The no-returns policy is the more specific and more useful
refusal to hand back, and the ordering makes the behaviour deterministic rather than
dependent on which pattern happens to be listed first.

**Why the gates are regex, not a classifier.** They are auditable, unit-testable, and
free. For a facts-only tool whose refusal behaviour is a deliverable, "show me the
rule that fired" beats a probability. The independent stage-8 output scan is the
second net. The known limit is stated in the README: novel phrasings can slip through.

**Field hints are a hard constraint, not a preference.** This was found by
evaluation, not by design. "Who manages HDFC Balanced Advantage Fund?" was answered
with the RTA, because the fact chunk says *fund manager*, the page-prose section
literally contains the word *manages*, and no soft score boost could overcome a
longer text match. Raising the boost to 0.12 was not enough. The fix is structural:
stage 5 emits `field_hints`, stage 7 filters candidates to them. Retrieval still
boosts (0.20) so the right chunk surfaces in the top-k, but correctness no longer
depends on the boost winning.

---

## STAGE 6 — Data retrieval (`retrieval.py`)

```
score = 0.55 · bm25_norm
      + 0.45 · cosine
      + 0.22 · scheme_match        (½ if scheme-agnostic, −½ if wrong scheme)
      + 0.14 · topic_match         (full for the primary topic, 60% for others,
                                    50% if a procedure covers a neighbouring topic)
      + 0.20 · field_match         (full for the first hint, 40% for later ones)
      + 0.06 · kind_prior          (fact > procedure > prose)
      − kind_gate                 (what must NOT answer this topic)
      − MMR redundancy penalty
```

The **kind gate** encodes negative constraints that a pure relevance score cannot:

* a per-scheme number can never answer a portfolio-wide procedural question;
* the three statement procedures are mutually exclusive destinations, so a
  capital-gains procedure is the wrong answer to a tax-statement question.

Then **MMR** (λ = 0.72) drops near-duplicate chunks, which matters because each scheme
produces 13 near-identical template sentences.

`explain()` prints every intermediate signal for every hit, with the winning reasons:

```
  1. +0.932  [section/expense_ratio/balanced_advantage]  bm25=1.00, cos=0.36, scheme+
  2. +0.839  [fact/scheme_meta/balanced_advantage]  bm25=0.41, cos=0.42, scheme+, topic:scheme_meta, kind:fact
```

This is what made the field-hint bug findable — a `field:` reason that never appeared
in any ranking was the visible symptom of `Retriever` not loading chunk `meta` from
the database.

**Performance note.** The first implementation of MMR re-projected the candidate
texts on every comparison pair, costing 10.9 s on a warm index. Reusing the vectors
already stored in the index brought mean latency to **49 ms** (258 ms on the first cold
call), and the test suite from 20 s to 1.9 s.

---

## STAGE 7 — Answer synthesis (`synthesis.py`)

Two generators behind one interface.

**`ExtractiveSynthesiser` (default, offline).** Not a language model. It composes
answers deterministically: sentence 1 is the winning fact chunk's own first sentence;
sentence 2 is added only if a **companion field belonging to the asked topic** exists.
`COMPANION_FIELDS` is deliberately sparse — the two SIP minimums are the same family of
number and useful together, but `"The fund manager is X. The ISIN is Y."` answers
nothing the user asked, so scheme-metadata topics get no companion at all.

**`LLMSynthesiser` (optional).** Receives a compact numbered context and must return
`{"answer": ..., "citation": ...}`. Its output is treated as untrusted and goes through
the identical stage-8 path. Any parse failure, provider error or guard failure falls
back to the extractive answer — so enabling the LLM cannot make the assistant *less*
safe, only more fluent.

**Why default to no LLM.** A facts-only assistant whose numbers come from a source of
record is safer when it cannot paraphrase a number. Fluency is worth less than the
guarantee that "1.03%" was copied, not generated.

---

## STAGE 8 — Post-processing & guardrails (`guardrails.py`)

Applied to **every** response, including LLM output. A failed check never silently
degrades.

| Check | Action on failure |
|---|---|
| `G1` PII in the body | → `pii_refusal` (never echo it back) |
| `G2` citation not in `ALLOWED_URLS` | → drop the citation, flag `G2_citation_missing` |
| `G3` more than 3 sentences | → truncate to 3 |
| `G4` advice verbs in the body | → `advice_refusal` |
| `G5` return/performance claim in the body | → `performance_refusal` |
| `G6` no freshness stamp | → stamp from chunk provenance |
| `G7` no disclaimer | → attach it |

Also handled here: inline URLs in the body are promoted to the single citation and
stripped from the prose; duplicate sentences are collapsed; a missing answer becomes
the empty-query prompt.

`G1`/`G4`/`G5` are the second net behind the stage-5 gates. They are deliberately
independent implementations — a different regex set, operating on output rather than
input — so a gap in one is unlikely to be a gap in both.

---

## STAGE 9 — Evaluation (`eval.py`, `tests/`)

**15 golden cases** — graded on right citation, correct figure, ≤3 sentences,
disclaimer, freshness stamp. **13 policy cases** — graded on correct refusal kind, and
that the emitted text is byte-identical to the canonical refusal template.

That last check replaced an earlier keyword grader that asserted the refusal contains
no advice words. It failed, correctly, on its own template — the advice refusal
contains "whether you should buy" in order to decline. **A refusal is verified by being
the template, not by keyword absence.**

```bash
python run.py eval                        # 28/28 checks
python -m unittest discover -s tests      # 46 tests
```

`samples/sample_qa.md` is generated by running the live pipeline, not written by hand,
so the deliverable cannot drift from real behaviour.
