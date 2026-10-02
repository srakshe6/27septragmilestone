"""STAGE 8 - Post-processing & guardrails.

This is the stage that makes the product promises mechanically true. It runs
on *every* response, including LLM output, and it is allowed to rewrite or
block. Checks, in order:

  G1  PII            - answer must not contain personal identifiers
  G2  citation       - exactly one citation, and it must be in the allow-list
  G3  sentence cap   - body at most N sentences (default 3)
  G4  advice         - no recommendation verbs may survive
  G5  performance    - no return/performance claim may survive
  G6  freshness      - the "Last updated from sources:" stamp is present
  G7  disclaimer     - the facts-only notice is attached

A failed check never silently degrades: G1/G4/G5 convert the response into a
refusal, G2/G3 repair it, G6 appends it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime

from .config import ANSWER_PREPARED_PREFIX, DISCLAIMER, LAST_UPDATED_PREFIX, MAX_SENTENCES
from .parsing import split_sentences
from .sources import ALLOWED_URLS, EDUCATION_LINKS, SOURCES_BY_ID


def prepared_stamp() -> str:
    """Local wall-clock date and time at which an answer was produced.

    Deliberately *not* the source's date. The two are different facts: a NAV of
    1447.383 is the NAV as of 25 Sep, and stamping it "2 Oct" would assert a
    value the source never published. What the current timestamp buys is
    something the source date cannot: the ability to see that an answer was
    produced today out of a corpus that is five days stale.
    """
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z").strip()

# ------------------------------------------------------------------ G1 PII
#
# Stage 8 runs a net written independently of `intent.PII_PATTERNS`, on
# purpose. Reusing those objects would collapse the two layers into one: every
# gap in the input gate would then be a gap in the output gate as well, which is
# exactly the failure the two-net design exists to prevent. The *approaches*
# differ, not merely the constants -- stage 5 matches "known label + a known
# value shape", this matches "an identifier shape on its own" plus "a sensitive
# label + any value that looks like a credential".

#: Bare identifier shapes, no label required. An echo that lost its surrounding
#: words (a model summarising "your PAN is ABCDE1234F" as the bare token) is
#: still caught. Deliberately case-insensitive: the stage-5 PAN pattern is
#: upper-case only, so a lower-cased identifier passed both nets.
_PII_SHAPES: list[tuple[str, re.Pattern]] = [
    ("pan", re.compile(r"\b[A-Za-z]{5}\d{4}[A-Za-z]\b")),
    ("aadhaar", re.compile(r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b")),
    # A 16-digit folio and a 12-digit account number are both just long digit
    # runs once the label is gone.
    ("long_number", re.compile(r"(?<!\d)\d{9,}(?!\d)")),
    ("card", re.compile(r"(?<!\d)\d{13,19}(?!\d)")),
    # An IFSC has a fixed shape and is recognisable without the word "IFSC",
    # which is exactly how it tends to be echoed once the label is dropped.
    ("bank", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", re.I)),
    # One shape for both: a UPI handle ("a@okaxis") has no TLD dot, and an
    # email address is PII either way, so they need no separate rules.
    ("email_or_upi", re.compile(r"\b[\w.+-]{1,}@[\w-]{2,}\b")),
    ("phone", re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")),
]

#: A sensitive label followed by *any* value that contains a digit. The digit
#: requirement is what keeps prose safe: "account statements", "ifsc code is
#: published on the factsheet" and "upi id is optional" carry a label and a
#: word, but no credential, so they are not flagged. The short minimum length
#: is deliberate too -- a 3-digit OTP is a credential, and stage 5 required 4.
_PII_LABELLED = re.compile(
    r"\b(?:pan|aadhaar|aadhar|folio|account|customer\s*id|otp|"
    r"one[\s-]?time\s+password|verification\s+code|auth\s+code|"
    r"passcode|password|demat|upi|ifsc|bank\s+account)\b"
    r"\s*(?:id|code|number|no\.?|is|are|:|=)?[\s:=-]*"
    r"([A-Za-z0-9][A-Za-z0-9@._/-]{2,})",
    re.I,
)

# ------------------------------------------------------------------ G4 advice

ADVICE_LEAK = re.compile(
    r"\b(?:you\s+should|we\s+recommend|recommend(?:ed|ation)?\s+(?:buying|selling|investing)|"
    r"i\s+suggest|is\s+(?:a\s+)?good\s+(?:buy|investment|choice)|better\s+than|"
    r"outperform|underperform|is\s+the\s+best|prefer(?:red)?\s+over|"
    r"ideal\s+for\s+you|suitable\s+for\s+you|worth\s+buying)\b",
    re.I,
)

# ------------------------------------------------------------------ G5 performance

#: Words that only appear in a *return* claim in this domain. Kept separate from
#: the pattern so the two shapes below can share one vocabulary. "growth" is
#: deliberately absent: every scheme in this corpus is named "... (Direct -
#: Growth)", so treating it as a return noun flagged the expense ratio, exit load
#: and benchmark of all five funds -- a false positive on the entire corpus.
_PERF_NOUN = (r"(?:cagr|returns?|returned|returning|yields?|yielded|"
              r"appreciation|gains?|gained)")

PERFORMANCE_LEAK = re.compile(
    # Shape 1: a percentage followed by a return frame -- "18% p.a.", "12% last
    # year", "14% since inception", "13.4% CAGR".
    r"\d+(?:\.\d+)?\s*%\s*(?:cagr|p\.?\s?a\.?|per\s+annum|annually|annual|"
    r"since\s+inception|over\s+(?:the\s+)?(?:past|last)|\d+\s*(?:year|month)s?|"
    r"last\s+year|year[\s-]to[\s-]date|ytd)"
    # Shape 2: a return noun with a percentage inside the same sentence --
    # "the 1-year return is 12.5%", "Annualised return since inception is
    # 14.1%", "it yielded 15%". The percentage is *required*, which is what
    # keeps "you can download returns for 3 years" from reading as a return
    # claim, and what keeps a NAV or expense-ratio answer safe.
    r"|\b" + _PERF_NOUN + r"\b[^.!?]{0,32}?\d+(?:\.\d+)?\s*%"
    # Shape 3: a completed return verb, with or without a number.
    r"|\b(?:has|have|had)\s+(?:returned|gained|grown|yielded|appreciated|generated)\b"
    # "Total Return" as a *metric* leaks; "NIFTY 100 Total Return Index" is the
    # name of the benchmark, which is a legitimate quotable fact -- it is the
    # index name for four of the five funds in this corpus.
    r"|\b(?:absolute|annualised|annualized|cumulative|total|overall|average)\s+returns?\b(?!\s+index)"
    r"|\b(?:out|under)perform(?:ed|ance|s|ing)?\b"
    r"|\b(?:best|top|highest)[\s-]performing\b",
    re.I,
)

_URL_RE = re.compile(r"https?://[^\s\)\]\"'>]+")


@dataclass
class Answer:
    kind: str                       # answer | advice_refusal | performance_refusal | pii_refusal | empty
    text: str                       # the ≤3-sentence body
    citation: str = ""              # exactly one URL
    citation_title: str = ""
    last_updated: str = ""
    #: True only when `last_updated` is a date carried by a fetched source
    #: (nav_date or fetched_at). False for refusals and authored procedures,
    #: where any date is a preparation date and must not be presented as
    #: source freshness.
    freshness_from_source: bool = True
    disclaimer: str = DISCLAIMER
    stage: str = ""                 # which generator produced the body
    confidence: float = 0.0
    evidence: list[dict] = field(default_factory=list)
    guards: list[str] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)
    retrieval_debug: dict = field(default_factory=dict)

    @property
    def is_refusal(self) -> bool:
        # `empty` is one of the four canonical refusal kinds, so testing the
        # `_refusal` suffix alone reported False for the "I didn't catch a
        # question" answer -- the one response where refusing matters most.
        return self.kind in REFUSAL_BODIES

    def render(self) -> str:
        parts = [self.text]
        if self.citation:
            parts.append(f"Source: {self.citation}")
        # Every answer carries the wall-clock moment it was produced. Without it
        # a reader cannot tell a freshly-built answer from one served out of a
        # week-old cache, because the only date shown is the source's.
        parts.append(f"{ANSWER_PREPARED_PREFIX}{prepared_stamp()}")
        if self.last_updated and self.freshness_from_source:
            parts.append(f"{LAST_UPDATED_PREFIX}{self.last_updated}")
        if self.disclaimer:
            parts.append(self.disclaimer)
        return "\n".join(p for p in parts if p)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "text": self.text, "citation": self.citation,
            "citation_title": self.citation_title, "last_updated": self.last_updated,
            "freshness_from_source": self.freshness_from_source,
            "disclaimer": self.disclaimer, "stage": self.stage,
            "confidence": round(self.confidence, 4), "evidence": self.evidence,
            "guards": self.guards, "timings_ms": self.timings_ms,
            "retrieval_debug": self.retrieval_debug,
            # Wall-clock moment of production, kept separate from `last_updated`
            # so a JSON client can show both without re-deriving either.
            "prepared_on": prepared_stamp(),
        }


# ------------------------------------------------------------------ refusals

#: Canonical refusal bodies. These are the *only* strings a refusal may emit,
#: which is what lets stage 9 assert "the refusal was a template, not advice".
#: Derived from the registry rather than restated, so the link a refusal prints
#: cannot drift away from the link the allow-list was built from.
EDUCATIVE_REFUSAL = EDUCATION_LINKS["default"]

REFUSAL_BODIES: dict[str, str] = {
    "advice_refusal": (
        "I can only share published facts about these schemes, so I can't say "
        "whether you should buy, sell or hold anything. The official scheme "
        "pages carry the factsheet, SID and KIM if you want to read the "
        "disclosures yourself."
    ),
    "performance_refusal": (
        "I don't compute or compare returns, and this assistant makes no "
        "performance claims. Returns and riskometer history are published in "
        "each scheme's monthly factsheet on the AMC's official pages."
    ),
    "pii_refusal": (
        "I don't accept or store PAN, Aadhaar, folio or account numbers, OTPs, "
        "email addresses, phone numbers, or UPI/IFSC bank details, so please "
        "don't send them. For anything account-specific, use the AMC investor "
        "portal or your RTA's own request page."
    ),
    "empty": (
        "I didn't catch a question. Ask me about expense ratio, exit load, "
        "minimum SIP, ELSS lock-in, benchmark or how to download statements."
    ),
}

def _edu_url(topic: str | None) -> str:
    return EDUCATION_LINKS.get(topic or "default", EDUCATION_LINKS["default"])


#: Every canonical refusal carries an allow-listed educational link, so FR-2
#: ("every answer carries one source link") holds for refusals too. `empty`
#: used to map to "", which meant the one response that most needed a next step
#: rendered with no citation at all.
REFUSAL_LINKS: dict[str, str] = {
    "advice_refusal": _edu_url(None),
    "performance_refusal": _edu_url(None),
    "pii_refusal": _edu_url(None),
    "empty": _edu_url(None),
}


def make_refusal(kind: str, topic: str | None = None, extra_guards: list[str] | None = None) -> Answer:
    body = REFUSAL_BODIES.get(kind, REFUSAL_BODIES["empty"])
    # Prefer a topic-specific educational page where one exists (the riskometer
    # factsheet, the AMFI NAV feed) and fall back to the default. `sources.py`
    # asserts at import that every education link is allow-listed, so a refusal
    # can never be handed a URL that G2 would have to drop.
    url = _edu_url(topic) if topic in EDUCATION_LINKS else REFUSAL_LINKS.get(kind, "")
    return Answer(
        kind=kind, text=body, citation=url, last_updated=date.today().isoformat(),
        # A refusal quotes no fetched fact, so its date is when the refusal was
        # written, not when a source was updated. Saying otherwise would put a
        # false provenance claim on the one answer that must look most careful.
        freshness_from_source=False,
        stage="refusal", confidence=1.0,
        guards=(extra_guards or []) + ["refusal_template"],
    )


# ------------------------------------------------------------------ checks

def _strip_urls(text: str) -> tuple[str, list[str]]:
    urls = _URL_RE.findall(text)
    return _URL_RE.sub("", text).strip(), urls


def _contains_pii(text: str) -> list[str]:
    """Output-side PII scan, written independently of `intent.PII_PATTERNS`.

    Two nets in two directions is the whole point of the design, so this must
    not be a re-import of the input patterns. It is also deliberately able to
    catch shapes the input net misses: a lower-cased PAN, a 3-digit OTP, and a
    bare IFSC whose "IFSC" label was dropped somewhere in between.
    """
    # An "@" inside a URL is not an email address, and the body has not been
    # cleaned yet (G1 runs before URL stripping, by design), so mask URL tokens
    # before shape matching rather than exempting the whole pattern.
    scan = _URL_RE.sub(" ", text)
    hits: list[str] = []
    for kind, pattern in _PII_SHAPES:
        if pattern.search(scan):
            hits.append(kind)
    for m in _PII_LABELLED.finditer(scan):
        # A label followed by a word is not a credential. Requiring a digit is
        # what separates "my folio number is 12345678901" from "your folio
        # number is on the statement".
        if any(c.isdigit() for c in m.group(1)):
            hits.append("labelled_value")
    return sorted(set(hits))


def _cap_sentences(text: str, n: int) -> str:
    sents = split_sentences(text)
    if len(sents) <= n:
        return " ".join(sents).strip()
    return " ".join(sents[:n]).strip().rstrip(",;:") + "."


def _dedupe_sentences(text: str) -> str:
    seen, out = set(), []
    for s in split_sentences(text):
        k = re.sub(r"[^a-z0-9]", "", s.lower())
        if k and k not in seen:
            seen.add(k)
            out.append(s)
    return " ".join(out).strip()


def enforce(answer: Answer, kind: str, topic: str | None = None) -> Answer:
    """Run G1-G7 over a drafted answer and return the final, compliant object."""
    guards = list(answer.guards)

    # A canonical refusal is already the safest text this system can emit, and
    # running the nets over it would be actively wrong: the advice template says
    # "whether you should buy" *in order to decline*, so G4 would read our own
    # refusal as a leak. The pipeline routes stage-5 refusals around `enforce`,
    # but this makes the guarantee structural instead of caller-dependent.
    already_refusal = answer.kind in REFUSAL_BODIES

    # G1 PII in the generated body -> refuse rather than echo it back
    if not already_refusal:
        pii = _contains_pii(answer.text)
        if pii:
            guards.append(f"G1_pii_blocked:{','.join(pii)}")
            return make_refusal("pii_refusal", topic, guards)

        # G4 advice leak
        if ADVICE_LEAK.search(answer.text):
            guards.append("G4_advice_blocked")
            return make_refusal("advice_refusal", topic, guards)

        # G5 performance claim
        if PERFORMANCE_LEAK.search(answer.text):
            guards.append("G5_performance_blocked")
            return make_refusal("performance_refusal", topic, guards)

    body, inline_urls = _strip_urls(answer.text)
    body = _dedupe_sentences(body)

    # G2 citation: inline URLs first, then the declared citation; must be
    # allow-listed. After this there is no third case -- "" or a member of
    # ALLOWED_URLS, which is the mechanical form of FR-2.
    citation = ""
    for cand in [*inline_urls, answer.citation]:
        cand = cand.rstrip(".,);")
        if cand in ALLOWED_URLS:
            citation = cand
            break
    if citation:
        guards.append("G2_citation_ok")
    else:
        guards.append("G2_citation_missing")

    # G3 sentence cap
    if len(split_sentences(body)) > MAX_SENTENCES:
        body = _cap_sentences(body, MAX_SENTENCES)
        guards.append("G3_sentence_capped")

    # G6 freshness stamp. If the chunk carried no source date (an authored
    # procedure), the stamp we add is a *preparation* date, so the flag must go
    # off or the UI would label it "Last updated from sources".
    if not answer.last_updated:
        answer.last_updated = date.today().isoformat()
        answer.freshness_from_source = False
        guards.append("G6_stamp_added")

    answer.text = body.strip()
    answer.citation = citation
    answer.guards = guards

    # G7 disclaimer
    answer.disclaimer = answer.disclaimer or DISCLAIMER
    if DISCLAIMER not in answer.disclaimer:
        answer.disclaimer = f"{answer.disclaimer} {DISCLAIMER}".strip()

    if citation:
        src = next((s for s in SOURCES_BY_ID.values() if s.url == citation), None)
        answer.citation_title = src.title if src else ""

    if not answer.text:
        return make_refusal("empty", topic, guards)
    return answer
