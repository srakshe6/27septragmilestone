"""STAGE 5 - Query understanding, routing and policy gate.

This stage runs *before* retrieval and decides, for every incoming query:

  1. Is the query admissible at all?        -> `Decision.kind`
  2. Does it contain personal data?         -> redacted, then judged again
  3. Which scheme is it about?              -> `Decision.scheme_key`
  4. Which fact topic is it asking about?   -> `Decision.topics`

Three refusals exist and all of them are non-negotiable product rules:

  * `advice`     - "should I buy / which is better / is it good for me"
  * `performance`- "which gave better returns / 3-year CAGR"  (we never
                   compute or compare returns; we point at the factsheet)
  * `pii`        - the query carried PAN / Aadhaar / folio / OTP / email /
                   phone.  The value is dropped, never logged, never stored.

Classification is rule-based and inspectable on purpose: this is a
facts-only assistant whose safety properties are part of the deliverable, so
the decision must be auditable rather than learned.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .sources import SCHEMES

# ------------------------------------------------------------------ PII

PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("aadhaar", re.compile(r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b")),
    ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("phone", re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")),
    ("otp", re.compile(r"\b(?:otp|one[\s-]?time\s+password)\b\s*(?:is|:|=)?\s*\d{4,6}\b", re.I)),
    ("otp", re.compile(r"\b(?:otp|verification code|auth code)\s*(?:is|:|=)\s*\d{4,6}\b", re.I)),
    ("folio", re.compile(r"\b(?:folio|acc(?:oun)?t|account)\s*(?:no|number|#|is|:|=)\s*[A-Z0-9][A-Z0-9\/-]{5,}\b", re.I)),
    # Bank credentials, both shapes. The handle quantifier is {1,} because real
    # UPI handles are routinely one to three characters ("a@okaxis"); it used to
    # be {4,}, which quietly exempted the most common handles in the country and
    # let them through as an ordinary answer. An IFSC has no "@" at all, so it
    # needs its own alternative -- the UPI shape alone never matched one.
    ("bank", re.compile(r"\b(?:upi|ifsc)\b\s*(?:id|code)?\s*(?:is|:|=)?\s*"
                        r"(?:[a-z0-9._-]{1,}@[a-z]{2,}|[A-Z]{4}0[A-Z0-9]{6})\b", re.I)),
    # The same handle shape with the `upi|ifsc` keyword gate removed.
    # `PII_KEYWORDS` catches "what is my upi id", but a query can carry the
    # handle with the word "upi" nowhere in it -- "transfer to bob@okaxis" --
    # and the email pattern above cannot help, because that one requires a
    # dotted domain and a UPI handle has none. The handle then survives into
    # `Decision.query`, which the design forbids: no identifier value may sit in
    # any field of the decision, only in the refusal.
    #
    # Ordering matters and is deliberate. `email` is earlier in this list, so a
    # real address is labelled `email` and this never sees it; a dotless handle
    # falls through to here. Both redact, so the label is the only difference.
    ("bank", re.compile(r"\b[a-z0-9._-]{1,}@[a-z]{2,}\b", re.I)),
]

#: Query-level terms that merely *mention* a sensitive field (no value given).
#: `upi` and `ifsc` belong here for the same reason `pan` does: asking about your
#: own bank credentials is an account-specific question this assistant cannot
#: answer, whether or not the value was pasted.
PII_KEYWORDS = re.compile(
    r"\b(?:pan|aadhaar|folio\s*(?:no|number)|account\s*(?:no|number)|otp|demat|"
    r"passcode|password|bank\s*account|upi|ifsc)\b",
    re.I,
)


def scrub_pii(query: str) -> tuple[str, list[str]]:
    """Remove personal data. Returns (clean_query, kinds_found)."""
    cleaned = query
    found: list[str] = []
    for kind, pattern in PII_PATTERNS:
        cleaned, n = pattern.subn("[redacted]", cleaned)
        if n:
            found.append(kind)
    # a bare 10-digit number with no context is still dropped
    cleaned, n = re.subn(r"(?<!\d)\d{10,12}(?!\d)", "[redacted]", cleaned)
    if n and "long_number" not in found:
        found.append("long_number")
    return re.sub(r"\s+", " ", cleaned).strip(), found


#: Redacts the value that follows a sensitive keyword. On the `PII_KEYWORDS` path
#: no value *pattern* fires, so anything the user pasted after the keyword would
#: otherwise survive in `Decision.query` -- `raw_query` is cleared, but `query` is
#: the field the pipeline retrieves and logs. A credential whose shape the value
#: patterns do not recognise ("upi is 12345", a malformed IFSC) is exactly the
#: case this exists for.
_KEYWORD_VALUE = re.compile(
    r"\b(pan|aadhaar|folio\s*(?:no|number)|account\s*(?:no|number)|otp|demat|"
    r"passcode|password|bank\s*account|upi|ifsc)\b"
    r"\s*(?:id|code|number|no)?\s*(?:is|are|:|=)?\s*[^\s,;?.]{3,}",
    re.I,
)


def _redact_keyword_values(text: str) -> str:
    return _KEYWORD_VALUE.sub(lambda m: f"{m.group(1)} [redacted]", text)


# ------------------------------------------------------------------ refusal rules

ADVICE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bshould\s+(?:i|we)\b", re.I),
    re.compile(r"\bshould\s+(?:i|we)\s+(?:buy|sell|invest|redeem|switch|hold|start)\b", re.I),
    re.compile(r"\b(?:can|could|would)\s+(?:i|we)\s+(?:buy|sell|invest|redeem)\b", re.I),
    re.compile(r"\b(?:is|are)\s+(?:it|this|that)\s+(?:a\s+)?(?:good|bad|worth|safe|risky)\b", re.I),
    # "Is HDFC Small Cap Fund risky?" / "is HDFC Balanced Advantage safe?" -
    # a copula whose predicate is a subjective adjective and nothing more.  The
    # end anchor is what makes it safe: it cannot fire on a factual question
    # that merely happens to contain an adjective ("what is the exit load").
    re.compile(r"\b(?:is|are|was|were)\b[^.?!]{0,60}?\b"
               r"(?:risky|riskier|safer|safe|volatile|aggressive|conservative|"
               r"worthwhile|good|bad|poor|great|profitable)\b\s*\??\s*$", re.I),
    # "Is HDFC ELSS a good buy?" - the subject is a scheme name, not a pronoun.
    re.compile(r"\b(?:is|are|was|were)\s+[\w\s.'-]{0,40}?\s+(?:a\s+)?"
               r"(?:good|bad|great|poor|worthwhile|safe|risky)\s+"
               r"(?:buy|investment|choice|option|pick)\b", re.I),
    re.compile(r"\b(?:good|bad|great)\s+(?:buy|entry\s*point)\b", re.I),
    re.compile(r"\bworth\s+(?:investing|buying|it)\b", re.I),
    re.compile(r"\bworth\s+it\b", re.I),
    re.compile(r"\b(?:which|what)\s+(?:is|one)?\s*(?:is\s+)?(?:the\s+)?best\b", re.I),
    re.compile(r"\bbest\s+(?:fund|scheme|option|choice|mutual\s*fund)\b", re.I),
    re.compile(r"\bwhich\s+(?:is|one|scheme|fund)\s+(?:should|do)\b", re.I),
    # "Which is better, A or B?" / "which fund is safer?" - an optional
    # intervening noun phrase keeps "which fund" from slipping past.
    re.compile(r"\bwhich\s+(?:\w+\s+){0,2}?(?:is|are|was|were)?\s*"
               r"(?:better|worse|safer|riskier|higher|lower|more\s+profitable)\b", re.I),
    re.compile(r"\bwhich\s+(?:better|worse|safer|riskier)\b", re.I),
    re.compile(r"\bis\s+(?:it\s+)?better\s+to\b", re.I),
    re.compile(r"\bcompare\s+(?:hdfc|them|the)\b", re.I),
    re.compile(r"\b(?:better|worse)\s+(?:than|for)\b", re.I),
    re.compile(r"\brecommend\w*\b", re.I),
    re.compile(r"\bsuggest\b", re.I),
    re.compile(r"\bsuitable\s+for\s+me\b", re.I),
    re.compile(r"\b(?:my|our)\s+portfolio\b", re.I),
    re.compile(r"\bhow\s+(?:do|should)\s+i\s+allocate\b", re.I),
    re.compile(r"\bhow\s+much\s+(?:should|do)\s+i\b", re.I),
    re.compile(r"\b(?:good|bad)\s+(?:time|entry\s*point)\b", re.I),
    re.compile(r"\b(?:target|goal)\s+of\s+\d+\s*%", re.I),
    re.compile(r"\b(?:buy|sell|park)\s+my\s+money\b", re.I),
]

PERFORMANCE_PATTERNS: list[re.Pattern] = [
    # "best performing fund" is a returns question; plain "best fund" is a
    # recommendation question and must fall through to the advice gate.
    re.compile(r"\b(?:best|highest|lowest|top)\s+"
               r"(?:performer|performers|performing|performing\s+funds?)\b", re.I),
    re.compile(r"\b(?:returns?|performance)\s+(?:of|for|comparison)\b", re.I),
    re.compile(r"\bcompare\s+(?:returns?|performance|nav|growth)\b", re.I),
    re.compile(r"\bhow\s+(?:much|did|has)\b.*\b(?:grown|grow|grew|return(?:ed|s)?|gained|growth)\b", re.I),
    re.compile(r"\b(?:cagr|xirr|irr|absolute\s+return|annual\s+return)\b", re.I),
    re.compile(r"\b(?:1|2|3|5|10)[\s-]*(?:yr|year)[\s-]*(?:returns?|performance|growth)\b", re.I),
    # "tell me about HDFC ELSS returns" - the plural noun on its own is a
    # returns question in this domain.  Deliberately plural-only: the singular
    # "return the money" is a redemption question, not a performance one.
    re.compile(r"\breturns\b", re.I),
    re.compile(r"\b(?:tell|show|give)\s+me\s+(?:about\s+)?(?:the\s+)?"
               r"(?:returns?|performance|growth|yield)\b", re.I),
    # Quant risk/return metrics we cannot compute or source.  Refused with the
    # factsheet pointer, which is more useful than guessing.
    re.compile(r"\b(?:volatility|drawdown|sharpe|sortino|alpha|beta|sip\s*calculator)\b", re.I),
    # "which fund gave the highest return" - needs the *return* word, not the
    # bare "which fund", so that "which fund is better" stays an advice query.
    re.compile(r"\bwhich\s+(?:fund|scheme|one|plan|of\s+them)\b"
               r"[^.?]{0,60}?\b(?:returns?|cagr|performance|growth|outperform)", re.I),
    re.compile(r"\bworth\s+in\s+the\s+long\s+term\b", re.I),
    re.compile(r"\bhow\s+did\b.*\b(?:do|perform\w*)\b", re.I),
    re.compile(r"\bperform(?:ed|ance|ing)?\b", re.I),
    re.compile(r"\bprojected?\s+(?:returns?|growth)\b", re.I),
    # ---- singular "return" used as a NOUN (not the redemption verb) --------
    # The plural-only rule above is deliberate: "return the money" is a
    # redemption question, not a performance one.  But that left a real FR-4
    # hole -- "what did X return last year?" and "what is the return on X?"
    # were *answered* (with the benchmark, or worse with lock-in) instead of
    # refused.  A singular "return" is a performance noun when it is
    #   (a) introduced by a determiner or adjective: "the return", "a good return"
    #   (b) followed by a time qualifier:            "return last year", "return in 2024"
    #   (c) used in the fixed phrase "return on":   "return on my investment"
    # and is a redemption *verb* when a direct object follows it
    # ("return the money", "return my investment", "return the units") -- in
    # that case the determiner comes *after* "return", so none of these match.
    re.compile(r"\b(?:the|a|an|my|good|bad|best|worst|average|expected|total|"
               r"annual|monthly|overall|average|pre[\s-]?tax|post[\s-]?tax|"
               r"latest|historical)\s+returns?\b", re.I),
    re.compile(r"\breturns?\s+(?:last|in|for|of|over)\s+"
               r"(?:\d{4}|\d+(?:\.\d+)?\s*%|\d{1,3}\s*(?:year|month|quarter|week|period)s?"
               r"|year|month|quarter|week|period|since|this)\b", re.I),
    re.compile(r"\breturn\s+on\s+(?:my|the|a|an|invest|investing|equity|"
               r"mutual|this)\b", re.I),
]

#: "the official factsheet" is the sanctioned escape hatch for return talk.
FACTSHEET_REQUEST = re.compile(
    r"\b(factsheet|fact\s*sheet|monthly\s+sheet|scheme\s+document|sid|kim)\b", re.I
)

# ------------------------------------------------------------------ topic routing

TOPIC_RULES: list[tuple[str, re.Pattern]] = [
    ("expense_ratio", re.compile(r"\bexpense\s*ratio\b|\bter\b|\bexpense\b|\bcharges?\s*ratio\b", re.I)),
    ("exit_load", re.compile(r"\bexit\s*load\b|\bredemption\s+charge\b|\bload\b|\bpremature\s+(?:redemption|exit)\b", re.I)),
    ("lock_in", re.compile(r"\block[\s-]?in\b|\blocked\b|\belss\b|\b3\s*years?\b|\bthree\s+years?\b|\b80c\b|\btax[\s-]sav(er|ing)\b", re.I)),
    # SIP and lump-sum are separate fields with different numbers, so they get
    # separate topics; a bare "minimum" defaults to the SIP topic below.
    ("min_sip", re.compile(r"\bmin(?:imum)?\s*sip\b|\bsip\b|\binstal?l?ments?\b|\bmonthly\b", re.I)),
    ("min_lumpsum", re.compile(r"\blump\s*sum\b|\blumpsum\b|\bone[\s-]time\b|\bminimum\s+(?:investment|amount)\b", re.I)),
    ("riskometer", re.compile(r"\brisk[\s-]?o?[\s-]?meter\b|\brisk\s+level\b|\brisk\s+category\b|\bhow\s+risky\b", re.I)),
    ("benchmark", re.compile(r"\bbenchmark\w*\b|\bindex\b|\bnifty\b|\bbse\b", re.I)),
    ("capital_gains", re.compile(r"\bcapital\s+gains?\b|\bltcg\b|\bstcg\b", re.I)),
    ("tax_doc", re.compile(r"\btax\s*(?:doc|document|report)s?\b|\btax\s*statement\b|\b26aq\b|\b16a\b|\btax\s*certificate\b", re.I)),
    ("statement_download", re.compile(r"\bstatement\b|\bdownload\b|\bcas\b|\btransaction\s+report\b", re.I)),
    ("nav", re.compile(r"\bnav\b|\bnet\s+asset\s+value\b", re.I)),
    ("scheme_meta", re.compile(r"\babout\b|\bwhat\s+is\b|\bscheme\s+name\b|\bfund\s+manager\b|\bwho\s+manages\b|\bcategory\b", re.I)),
]

#: Field-level preference when two facts share a topic but hold different
#: numbers (minimum SIP vs minimum lump-sum).
FIELD_HINTS: list[tuple[str, re.Pattern]] = [
    ("min_sip_amount", re.compile(r"\bsip\b|\bmonthly\b|\binstal?l?ment\b", re.I)),
    ("min_lumpsum_amount", re.compile(r"\blump\s?sum\b|\bone[\s-]time\b", re.I)),
    ("min_additional_amount", re.compile(r"\badditional\b|\btop[\s-]?up\b", re.I)),
    # `expense ratio` itself belongs here, not just "TER"/"total expense". The
    # hint exists to make the expense-ratio *field* win, and without the plain
    # phrasing it never fired on the way users actually ask -- so the
    # `expense_ratio_pct` hint was dead for its own topic.
    ("expense_ratio_pct", re.compile(r"\bter\b|\btotal\s+expense\b|\bexpense\s*ratio\b", re.I)),
    # "who manages X" does not lexically match the fact chunk's "fund manager",
    # so without this hint the RTA fact wins on raw term overlap.
    ("fund_manager", re.compile(r"\bwho\s+manage|\bmanages?\b|\bmanaged\s+by\b|\bfund\s+manager\b", re.I)),
    ("registrar_agent", re.compile(r"\bregistrar\b|\brta\b|\btransfer\s+agent\b|\bkfin\b|\bcams\b", re.I)),
    ("launch_date", re.compile(r"\blaunch\w*\b|\bwhen\s+was\b|\bstarted?\b|\binception\b", re.I)),
    ("isin", re.compile(r"\bisin\b", re.I)),
    ("category", re.compile(r"\bcategor\w*\b", re.I)),
    # Fund size has no lexical overlap with its chunk text -- the fact says
    # "assets under management" and the question says "fund size" -- so without
    # this hint the ISIN fact wins on raw term overlap and the answer is a
    # 12-character identifier nobody asked for. "corpus" is included because it
    # is the standard finance synonym for AUM, but it is last and narrow.
    # "how big is X" is the commonest phrasing and carries no AUM vocabulary at
    # all, so it has to be matched on its own. `\bbig\b` alone would be too broad,
    # hence the required "is" and the trailing boundary.
    ("aum", re.compile(r"\baum\b|\bfund\s+size\b|\bsize\s+of\b|\bassets?\s+under\s+management\b|\bcorpus\b|\bhow\s+big\s+is\b|\ba\s+big(?:ger)?\s+fund\b", re.I)),
]

#: Question-shape cues for the "one number please" style.
FACTUAL_VERBS = re.compile(
    r"^\s*(?:what|whats|what's|how\s+much|how\s+many|which|is|are|does|do|"
    r"can\s+i|minimum|max|maxim(um)?|tell|show|list|explain)\b",
    re.I,
)


# ------------------------------------------------------------------ decision

@dataclass
class Decision:
    kind: str                       # answer | advice_refusal | performance_refusal | pii_refusal | empty
    query: str                      # PII-scrubbed
    raw_query: str
    scheme_key: str | None = None
    topics: list[str] = field(default_factory=list)
    primary_topic: str | None = None
    field_hints: list[str] = field(default_factory=list)
    pii_kinds: list[str] = field(default_factory=list)
    reason: str = ""
    matched_rules: list[str] = field(default_factory=list)

    @property
    def is_refusal(self) -> bool:
        return self.kind.endswith("_refusal")

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "query": self.query, "scheme_key": self.scheme_key,
            "topics": self.topics, "primary_topic": self.primary_topic,
            "field_hints": self.field_hints, "pii_kinds": self.pii_kinds,
            "reason": self.reason, "matched_rules": self.matched_rules,
        }


def detect_field_hints(query: str) -> list[str]:
    return [f for f, pat in FIELD_HINTS if pat.search(query)]


def detect_scheme(query: str) -> tuple[str | None, list[str]]:
    """Resolve the scheme by longest alias match. Returns (key, aliases hit)."""
    low = f" {re.sub(r'[^a-z0-9 ]', ' ', query.lower())} "
    best: tuple[str, str] | None = None
    hits: list[str] = []
    for key, meta in SCHEMES.items():
        for alias in meta["aliases"]:
            a = f" {alias} "
            if a in low:
                hits.append(alias)
                if best is None or len(alias) > len(best[1]):
                    best = (key, alias)
    return (best[0] if best else None), hits


def detect_topics(query: str) -> list[str]:
    topics: list[str] = []
    for topic, pattern in TOPIC_RULES:
        if pattern.search(query):
            topics.append(topic)
    return _apply_topic_exclusivity(topics)


#: The three statement/tax topics are alternative answers to alternative
#: questions, not additive facets. Keeping all three lets a "tax statement"
#: query drift onto the capital-gains procedure, so only the first match in
#: rule order (capital gains > tax docs > account statement) is retained.
_PROCEDURAL_GROUP = ("capital_gains", "tax_doc", "statement_download")


def _apply_topic_exclusivity(topics: list[str]) -> list[str]:
    picked = next((t for t in _PROCEDURAL_GROUP if t in topics), None)
    if picked is None:
        return topics
    return [t for t in topics if t == picked]


def classify(query: str) -> Decision:
    """Full stage-5 pass over one user query."""
    raw = query or ""
    if not raw.strip():
        return Decision(kind="empty", query="", raw_query=raw,
                        reason="Empty query received.")

    clean, pii_kinds = scrub_pii(raw)
    if pii_kinds:
        return Decision(
            kind="pii_refusal", query=clean, raw_query="", pii_kinds=sorted(set(pii_kinds)),
            reason="Query contained personal identifiers, which were removed and not stored.",
        )

    scheme_key, alias_hits = detect_scheme(clean)
    topics = detect_topics(clean)
    field_hints = detect_field_hints(clean)
    rules: list[str] = ([f"alias:{a}" for a in alias_hits]
                        + [f"topic:{t}" for t in topics]
                        + [f"field:{f}" for f in field_hints])

    def _refuse(kind: str, pat: re.Pattern, label: str) -> Decision:
        return Decision(
            kind=kind, query=clean, raw_query=raw, scheme_key=scheme_key,
            topics=topics, field_hints=field_hints,
            primary_topic=topics[0] if topics else None,
            reason=f"{label} pattern matched: {pat.pattern}",
            matched_rules=rules + [f"{label}:{pat.pattern}"],
        )

    # Order of the three gates matters and is deliberate.

    # 1. Performance before advice. "Compare the returns of A and B" trips
    #    both sets, and the no-returns policy is the more specific and more
    #    useful refusal to hand back than "I can't recommend anything".
    for pat in PERFORMANCE_PATTERNS:
        if pat.search(clean):
            return _refuse("performance_refusal", pat, "perf")

    # 2. A bare PII *keyword* with no value ("what's my PAN status?") is also
    #    refused: the assistant has no access to investor data and must never
    #    take account-specific questions.
    if PII_KEYWORDS.search(clean):
        return Decision(
            kind="pii_refusal", query=_redact_keyword_values(clean), raw_query="",
            pii_kinds=["identifier_reference"],
            scheme_key=scheme_key, topics=topics, field_hints=field_hints,
            reason="Account-specific identifier question; this assistant has no access to investor data.",
            matched_rules=rules + ["pii_keyword"],
        )

    # 3. Advice last, so a question that is both comparative and advice-shaped
    #    still lands on a refusal either way.
    for pat in ADVICE_PATTERNS:
        if pat.search(clean):
            return _refuse("advice_refusal", pat, "advice")

    if not topics:
        topics = ["scheme_meta"]

    return Decision(
        kind="answer", query=clean, raw_query=raw, scheme_key=scheme_key,
        topics=topics, primary_topic=topics[0], field_hints=field_hints,
        reason="Factual query.", matched_rules=rules,
    )


def is_factual_shaped(query: str) -> bool:
    return bool(FACTUAL_VERBS.search(query.strip()))
