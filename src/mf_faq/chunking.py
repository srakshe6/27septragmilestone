"""STAGE 3 - Chunking.

Design: chunk *shape follows question shape*.  A user asking "what is the
exit load of HDFC Small Cap?" wants one number, not a paragraph.  So the
chunker emits three complementary chunk families:

  1. `fact`      - one atomic, self-contained statement per (scheme, field).
                   Highest precision, always carries an explicit answer string.
  2. `section`   - contiguous prose windows from the rendered page, used for
                   wording-level questions and as lexical backstop.
  3. `procedure` - authored step lists for the non-scheme questions
                   (downloading statements / tax documents), each bound to the
                   official destination page.

Every chunk carries `scheme_key`, `topic`, `source_id`, `answer_hint` and
`last_updated`, so the synthesis stage can emit a citation and a freshness
stamp without guessing, and the guardrail stage can prove the citation is
allow-listed.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .parsing import Document, normalise_date, split_sentences
from .sources import SOURCES_BY_ID, SCHEMES

WORD_RE = re.compile(r"[A-Za-z0-9₹%./]+")

#: topic -> human label, used in answer hints and the UI
TOPIC_LABELS: dict[str, str] = {
    "expense_ratio": "Expense ratio",
    "expense_ratio_note": "Expense ratio",
    "exit_load": "Exit load",
    "min_sip": "Minimum SIP",
    "min_lumpsum": "Minimum lump-sum",
    "lock_in": "Lock-in period",
    "riskometer": "Riskometer",
    "benchmark": "Benchmark",
    "nav": "NAV",
    "statement_download": "Downloading statements",
    "capital_gains": "Capital gains statement",
    "tax_doc": "Tax documents",
    "scheme_meta": "Scheme details",
}


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_id: str
    url: str
    kind: str                     # fact | section | procedure
    topic: str
    scheme_key: str | None
    text: str
    answer_hint: str = ""         # canonical short answer (facts/procedures)
    last_updated: str = ""
    tokens: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def embed_text(self) -> str:
        """The exact string this chunk is embedded and IDF-fitted on.

        Single source of truth for the text handed to the vectoriser. It used to
        be an f-string literal inside `indexing.build_index` while `tokens` was
        built separately in `build_fact_chunks`, so the two disagreed: the
        report printed tokens that were not the ones that produced the vector,
        and nothing but a diff could have caught it. Anything that changes what
        is embedded must change this and nowhere else.
        """
        return f"{self.topic} {self.scheme_key or ''} {self.text} {self.answer_hint}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tokens"] = list(self.tokens)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in WORD_RE.findall(text or "")]


#: Sentinel for "this chunk has no source currency to claim". Procedures are
#: authored text, not fetched data, so a date on them would be a fabricated
#: freshness claim (see `build_procedure_chunks`).
NO_SOURCE_DATE = ""


def _source_date(doc: Document, facts: dict) -> str:
    """The date to stamp on a chunk, derived *only* from the source.

    Order is `nav_date` then `fetched_at`, exactly as the architecture
    specifies. There is deliberately no `date.today()` fallback: I1 requires
    that this never be a clock read at build or query time, because the UI
    labels this field "Last updated from sources:".
    """
    return normalise_date(facts.get("nav_date")) or (doc.fetched_at or "")[:10]


# ------------------------------------------------------------------ formatters

def _fmt_expense(v) -> str:
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_lockin(days) -> str:
    if not days:
        return "Nil - there is no lock-in period for this scheme"
    d = int(days)
    if d % 365 == 0:
        return f"{d // 365} year{'s' if d // 365 != 1 else ''}"
    months = round(d / 30)
    if months % 12 == 0:
        return f"{months // 12} year{'s' if months // 12 != 1 else ''}"
    return f"{months} months"


def _fmt_exit_load(v) -> str:
    """Tidy a page-supplied exit-load string into a readable fact.

    Source pages phrase this inconsistently ("Exit load of 1% if redeemed
    within 1 year", "Exit Load for units in excess of 15% of the
    investment,1% will be charged for redemption within 1 year").  This
    normalises capitalisation, spacing and the redundant leading label, and
    re-orders the one badly-worded pattern into word order.  The rate, the
    excess threshold and the redemption window are all preserved verbatim.
    """
    s = re.sub(r"\s+", " ", str(v)).strip()
    s = s.replace("investment,1%", "investment, 1%")
    s = re.sub(r",\s*(?=\d)", ", ", s)

    m = re.search(
        r"for\s+units\s+in\s+excess\s+of\s+([\d.]+%)\s+of\s+the\s+investment,?\s*"
        r"([\d.]+%)\s+will\s+be\s+charged\s+for\s+redemption\s+within\s+(\d+\s+\w+)",
        s, re.I)
    if m:
        return (f"{m.group(2)} for units in excess of {m.group(1)} of the "
                f"investment, if redeemed within {m.group(3)}")

    s = re.sub(r"^exit\s+load\s+(?:of|for)\s+", "", s, flags=re.I)
    if s and not s[:2].isupper():
        s = s[0].lower() + s[1:]
    return s


def _fmt_rupees(v) -> str:
    return f"{int(v):,}" if str(v).isdigit() else str(v)


def _fmt_managers(v) -> str:
    """Join manager names into one readable phrase, primary first."""
    names = ([str(n).strip() for n in v if str(n).strip()]
             if isinstance(v, (list, tuple)) else [str(v).strip()])
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f" and {names[-1]}"


FORMATTERS = {
    "expense": _fmt_expense,
    "lockin": _fmt_lockin,
    "exit": _fmt_exit_load,
    "min_sip": _fmt_rupees,
    "min_lump": _fmt_rupees,
    "min_add": _fmt_rupees,
    "nav": lambda v: f"{float(v):,.3f}",
    # Indian digit grouping, matching how the source page renders the figure
    # (Rs 1,13,606.47 Cr rather than Rs 113,606.47 Cr). Two decimals, because
    # the payload value carries fractions of a crore.
    "aum": lambda v: f"{float(v):,.2f}",
    "manager": _fmt_managers,
}


#: Fields whose sentence has to agree in number with its value.
#: "The fund manager of HDFC Flexi Cap Fund is Dhruv Muchhal and Amit Ganatra"
#: is not English, and the schemes here really are co-managed -- all five carry
#: at least two managers, so the singular form was wrong every time. Keyed by
#: fact key -> (singular template, plural template); the plural is used whenever
#: the value is a list that is not a single name.
PLURAL_TEMPLATES: dict[str, tuple[str, str]] = {
    "fund_manager": (
        "The fund manager of {scheme} is {v}.",
        "The fund managers of {scheme} are {v}.",
    ),
}


#: Word-count ceiling for one `section` chunk.
#:
#: `build_section_chunks` windows by *sentence* count, which is only safe while
#: the input is prose. It is not. `portal.amfiindia.com/spages/NAVAll.txt` is a
#: 1.5 MB semicolon-delimited CSV of every scheme's NAV for every plan, and it
#: contains almost no sentence terminators -- `split_sentences` returned 21
#: "sentences" for it, five of them 128k-498k characters long. Windows over
#: those produced four chunks of 73,890 / 70,282 / 54,643 / 26,631 tokens that
#: between them held 91% of the corpus token mass, and saturated their own
#: embeddings (512 of 512 dimensions non-zero, against 80 of 512 for a fact
#: chunk). A 512-dim hashed bag-of-features cannot represent half a million
#: tokens: it hashes noise, and the score it returns is meaningless.
#:
#: 160 tokens is a little under the 4-sentence window's typical length, so
#: ordinary prose is unaffected; only the unbounded windows are dropped.
#: Over-budget windows are *discarded, not truncated*. Fragmenting a CSV at
#: whitespace would turn it into ~1,200 well-sized chunks of table noise and
#: swamp a 153-chunk index with content that answers nothing. Discarding keeps
#: the index honest about what a section chunk is for.
#:
#: Nothing is lost that matters here: the per-scheme NAV answers come from the
#: structured `nav` fact chunks the scrape extracts off each scheme page, not
#: from this CSV.
SECTION_MAX_TOKENS = 160


# ------------------------------------------------------------------ fact templates
# (fact_key, topic, template, formatter_key)

FACT_SPECS: list[tuple[str, str, str, str]] = [
    ("expense_ratio_pct", "expense_ratio",
     "The expense ratio of {scheme} is {v}%. The expense ratio is the annual "
     "percentage of the scheme's assets charged to the scheme, deducted from "
     "the scheme's returns rather than billed separately.",
     "expense"),
    ("exit_load", "exit_load",
     "The exit load of {scheme} is {v}.",
     "exit"),
    ("min_sip_amount", "min_sip",
     "The minimum SIP amount for {scheme} is ₹{v} per month.",
     "min_sip"),
    ("min_lumpsum_amount", "min_lumpsum",
     "The minimum one-time (lumpsum) investment for {scheme} is ₹{v}.",
     "min_lump"),
    ("min_additional_amount", "min_sip",
     "The minimum additional purchase amount for {scheme} is ₹{v}.",
     "min_add"),
    ("lock_in_days", "lock_in",
     "The lock-in period for {scheme} is {v}. Units cannot be redeemed before "
     "the lock-in period ends.",
     "lockin"),
    ("benchmark", "benchmark",
     "The benchmark of {scheme} is {v}.",
     "benchmark"),
    ("nav", "nav",
     "The NAV of {scheme} was ₹{v} as on {nav_date}.",
     "nav"),
    ("category", "scheme_meta",
     "{scheme} is categorised as {v}.",
     "category"),
    ("fund_manager", "scheme_meta",
     "The fund manager of {scheme} is {v}.",
     "manager"),
    ("launch_date", "scheme_meta",
     "{scheme} was launched on {v}.",
     "launch"),
    ("isin", "scheme_meta",
     "The ISIN of {scheme} is {v}.",
     "isin"),
    ("registrar_agent", "scheme_meta",
     "The Registrar and Transfer Agent (RTA) for {scheme} is {v}.",
     "rta"),
    (    "aum", "scheme_meta",
     "The fund size (assets under management) of {scheme} is ₹{v} crore, "
     "as on {nav_date}.",
     "aum"),
]


#: Fields whose *absent* value is itself the published answer, not a gap in the
#: scrape. `lock_in_days` is `None` for every scheme that has no lock-in, and
#: `_fmt_lockin` already knows how to say so -- but the generic skip below threw
#: the `None` away, which left ELSS as the only lock-in fact in the corpus. A
#: lock-in question about any other scheme then matched ELSS's chunk on lexical
#: and topic score and answered with the wrong scheme. Retrieval now refuses to
#: let one scheme's fact speak for another (`Retriever._same_scheme`), but
#: without these records the four non-ELSS schemes would simply have no lock-in
#: chunk and the question would degrade to "could not find this in the sources".
NIL_BODIES: dict[str, str] = {
    "lock_in_days": (
        "There is no lock-in period for {scheme}. Units can be redeemed at any "
        "time, subject to the applicable exit load."
    ),
}


def build_fact_chunks(doc: Document) -> list[Chunk]:
    """One chunk per (field, scheme) with a pre-written answer sentence."""
    if not doc.scheme_key or not doc.facts:
        return []
    scheme = SCHEMES[doc.scheme_key]["label"]
    facts = doc.facts
    out: list[Chunk] = []

    for key, topic, template, fmt_key in FACT_SPECS:
        raw = facts.get(key)
        # The source stating a field is nil is a published fact ("no lock-in
        # period for this scheme"). The scrape simply not capturing the field is
        # not, and must not be upgraded into one. `key in facts` is what
        # separates them, so a document that never mentioned lock-in does not
        # gain a claim about it.
        nil_fact = key in facts and raw is None and key in NIL_BODIES
        if raw in (None, "", [], {}) and not nil_fact:
            continue
        if isinstance(raw, str) and len(raw) > 220:
            continue  # free-form blob: better served by a section chunk
        if isinstance(raw, (list, tuple)) and len(raw) > 8:
            # A roster this long is a page listing, not a management team the
            # assistant can render as one sentence. Leave it to a section chunk
            # rather than publish a name dump as a fact.
            continue
        value = FORMATTERS[fmt_key](raw) if fmt_key in FORMATTERS else raw
        if nil_fact:
            # Written per field rather than derived from `template`: the lock_in
            # template's second sentence asserts units cannot be redeemed before
            # the period ends, which is the opposite of the truth here.
            body = NIL_BODIES[key].format(scheme=scheme)
        elif key in ("nav", "aum"):
            # Both templates date-stamp themselves. AUM is a point-in-time figure
            # exactly like NAV, and "as of nothing" would be a bare number the
            # reader cannot age.
            nav_date = normalise_date(facts.get("nav_date")) or "the date shown on the page"
            body = template.format(scheme=scheme, v=value, nav_date=nav_date)
        elif (key in PLURAL_TEMPLATES
              and isinstance(raw, (list, tuple)) and len(raw) != 1):
            body = PLURAL_TEMPLATES[key][1].format(scheme=scheme, v=value)
        else:
            body = template.format(scheme=scheme, v=value)

        chunk = Chunk(
            chunk_id=f"{doc.doc_id}::fact::{key}",
            doc_id=doc.doc_id, source_id=doc.source_id, url=doc.url,
            kind="fact", topic=topic, scheme_key=doc.scheme_key,
            text=body,
            answer_hint=f"{TOPIC_LABELS.get(topic, topic)}: {value}",
            last_updated=_source_date(doc, facts),
            meta={"field": key, "raw_value": raw},
        )
        chunk.tokens = tokenize(chunk.embed_text())
        out.append(chunk)

    if facts.get("sip_allowed"):
        chunk = Chunk(
            chunk_id=f"{doc.doc_id}::fact::sip_allowed",
            doc_id=doc.doc_id, source_id=doc.source_id, url=doc.url,
            kind="fact", topic="min_sip", scheme_key=doc.scheme_key,
            text=f"SIPs and one-time investments are available for {scheme}.",
            answer_hint=f"SIP is available for {scheme}",
            last_updated=_source_date(doc, facts),
            meta={"field": "sip_allowed", "raw_value": True},
        )
        chunk.tokens = tokenize(chunk.text)
        out.append(chunk)
    return out


# ------------------------------------------------------------------ section chunks

#: Topic inference for `section` chunks. A topic is assigned only when the
#: window contains an *explicit* phrase that names the thing being asked about.
#:
#: This used to be a first-match-wins scan over patterns that included page
#: boilerplate -- `charges`, `direct plan`, `index`, `nifty`, `elss`. Every HDFC
#: and Groww scheme page says "Direct Plan" and links to a "Charges" heading, so
#: the scan labelled half the corpus `expense_ratio`: 28 of the 39 chunks tagged
#: that way did not contain the phrase "expense ratio" at all, they were fund
#: manager bios, annualised-returns tables, exit-load change histories and tax
#: paragraphs. That is not a cosmetic label, because `retrieval._topic_boost`
#: hands +0.14 to any chunk whose topic matches the classified intent -- so an
#: expense-ratio question was paying a relevance bonus to a holdings table.
#:
#: Each pattern below is deliberately multi-word or a distinctive single token
#: that only appears when the topic is genuinely under discussion. Anything
#: without an explicit phrase falls through to `scheme_meta`, which is the
#: honest "no topic claimed" label and earns no topic boost.
TOPIC_MARKS: list[tuple[str, str]] = [
    ("lock_in", r"lock.?in|\b80c\b|\b3\s+years?\b|\bthree\s+years\b"),
    ("exit_load", r"exit\s*load|redemption\s+charge|premature\s+redemption"),
    ("expense_ratio", r"expense\s+ratio"),
    ("min_sip", r"minimum\s+sip|minimum\s+(?:lumpsum|investment)|"
                r"lump.?sum\s+investment|installment\s+amount"),
    ("benchmark", r"\bbenchmark\b|\bbenchmark\s+index\b"),
    ("nav", r"net\s+asset\s+value|\bnavs?\b"),
]


def build_section_chunks(doc: Document, max_sentences: int = 4, overlap: int = 1,
                         max_tokens: int = 160) -> tuple[list[Chunk], int]:
    """Prose windows over the rendered page text (lexical backstop).

    Returns `(chunks, dropped)` where `dropped` counts windows rejected for
    exceeding `max_tokens`. A section chunk exists to serve wording-level
    questions, so anything that does not fit a token budget is not a usable
    prose window and is discarded rather than truncated -- see
    `SECTION_MAX_TOKENS`.
    """
    if not doc.text:
        return [], 0
    sentences = split_sentences(doc.text)
    if not sentences:
        return [], 0

    out: list[Chunk] = []
    dropped = 0
    step = max(1, max_sentences - overlap)
    for i in range(0, len(sentences), step):
        window = sentences[i : i + max_sentences]
        if not window:
            continue
        body = " ".join(window).strip()
        if len(body) < 60:
            continue
        if len(tokenize(body)) > max_tokens:
            dropped += 1
            continue
        low = body.lower()
        topic = "scheme_meta"
        for t, pattern in TOPIC_MARKS:
            if re.search(pattern, low):
                topic = t
                break
        chunk = Chunk(
            chunk_id=f"{doc.doc_id}::sec::{i:04d}",
            doc_id=doc.doc_id, source_id=doc.source_id, url=doc.url,
            kind="section", topic=topic, scheme_key=doc.scheme_key,
            text=body, last_updated=(doc.fetched_at or "")[:10],
            meta={"sentence_offset": i},
        )
        chunk.tokens = tokenize(chunk.embed_text())
        out.append(chunk)
    return out, dropped


# ------------------------------------------------------------------ procedures
# Authored, sourced, non-numeric. Each binds to an official destination page.

PROCEDURES: list[dict] = [
    {
        "topic": "statement_download",
        "source_id": "hdfc-investor-portal",
        "scheme_key": None,
        "title": "Download a mutual fund account or transaction statement",
        "answer_hint": "Use the AMC investor portal's statement request service",
        "body": (
            "Download a mutual fund account or transaction statement from the AMC's "
            "official investor portal, investor.hdfcfund.com, using its statement "
            "request service to choose your date range. You sign in to your own "
            "account on the portal itself, so nothing personal needs to be shared "
            "here."
        ),
    },
    {
        "topic": "capital_gains",
        "source_id": "cams-statements",
        "scheme_key": None,
        "title": "Download a capital gains / loss statement",
        "answer_hint": "CAMS Online serves capital-gains statements on request",
        "body": (
            "Download a capital gains / loss statement for your mutual fund holdings "
            "from the registrar's self-service request page, the Statements section "
            "of camsonline.com, where you choose the period, delivery option and "
            "fund houses to include. The statement is sent to the email address "
            "already registered with your folio, so no personal details need to be "
            "shared here."
        ),
    },
    {
        "topic": "tax_doc",
        "source_id": "hdfc-investor-portal",
        "scheme_key": None,
        "title": "Download tax documents and the annual tax statement",
        "answer_hint": "Tax statements are available from the AMC investor portal",
        "body": (
            "Download your annual tax documents, including the tax statement and "
            "capital gains report, from the AMC's official investor portal, "
            "investor.hdfcfund.com, under its statement and tax reports section. "
            "This assistant cannot access, request or store any investor document "
            "on your behalf."
        ),
    },
    {
        "topic": "riskometer",
        "source_id": "hdfc-official-scheme-docs",
        "scheme_key": None,
        "title": "Where the riskometer level is published",
        "answer_hint": "The monthly factsheet published by the AMC states the riskometer",
        "body": (
            "That level is not in the source pages collected for this build. The "
            "SEBI riskometer level and the month-wise riskometer trend are published "
            "by the AMC in the scheme's monthly factsheet, available on the AMC's "
            "official scheme pages alongside the SID and KIM."
        ),
    },
    {
        "topic": "expense_ratio_note",
        "source_id": "hdfc-official-scheme-docs",
        "scheme_key": None,
        "title": "Where the full fee and charge schedule is published",
        "answer_hint": "The AMC's official fee page and the SID are authoritative",
        "body": (
            "The complete fee and charge schedule - management fee, the ratio of "
            "expense ratio to average net assets, and the exit load table - is "
            "published on the AMC's official scheme pages and in the scheme's SID. "
            "The figure quoted on a scheme page is the ongoing annual charge; the SID "
            "is the authoritative fee document."
        ),
    },
]


def build_procedure_chunks() -> list[Chunk]:
    out: list[Chunk] = []
    for i, spec in enumerate(PROCEDURES):
        src = SOURCES_BY_ID[spec["source_id"]]
        chunk = Chunk(
            chunk_id=f"proc::{spec['source_id']}::{i:02d}",
            doc_id=spec["source_id"], source_id=spec["source_id"], url=src.url,
            kind="procedure", topic=spec["topic"], scheme_key=spec["scheme_key"],
            text=spec["body"],
            answer_hint=spec["answer_hint"], last_updated=NO_SOURCE_DATE,
            meta={"title": spec["title"]},
        )
        chunk.tokens = tokenize(chunk.embed_text())
        out.append(chunk)
    return out


# ------------------------------------------------------------------ entry point

def build_chunks(docs: list[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(build_fact_chunks(doc))
        sections, dropped = build_section_chunks(doc, max_tokens=SECTION_MAX_TOKENS)
        if dropped:
            # Surfaced rather than silently swallowed: a source that suddenly
            # starts dropping windows is a source whose content shape changed,
            # and that is worth seeing in the build log.
            print(f"[chunk] {doc.doc_id}: dropped {dropped} section window(s) "
                  f"over {SECTION_MAX_TOKENS} tokens (bulk/tabular content)")
        chunks.extend(sections)
    chunks.extend(build_procedure_chunks())
    return [c for c in chunks if c.text.strip()]
