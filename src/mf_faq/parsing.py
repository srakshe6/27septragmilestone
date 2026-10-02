"""STAGE 2 - Document parsing & cleaning.

Two complementary extractions are performed on each fetched page:

  A. `visible_text`  - boilerplate-stripped, whitespace-normalised render of
     the page (stdlib HTMLParser; no third-party dependency).
  B. `embedded_json` - the structured `__NEXT_DATA__` payload these pages ship,
     which carries the authoritative, field-level scheme facts (expense ratio,
     exit load, minimum SIP, lock-in, benchmark, ISIN, NAV date...).

B is the higher-fidelity signal and is preferred for anything a number can
answer.  A exists so that free-text questions (procedure walkthroughs, note
wording) still have something to retrieve over, and so the corpus is not
purely a JSON dump.
"""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

# ------------------------------------------------------------------ text

_SKIP_TAGS = {"script", "style", "noscript", "svg", "head", "template"}
_BLOCK_TAGS = {
    "p", "div", "br", "li", "tr", "td", "th", "section", "article", "h1",
    "h2", "h3", "h4", "h5", "h6", "ul", "ol", "table", "footer", "header",
}
_BOILERPLATE = (
    "log in", "sign in", "sign up", "download app", "cookie", "privacy policy",
    "terms and conditions", "disclaimer", "all rights reserved", "follow us",
    "newsletter", "subscribe", "careers", "contact us", "google play",
    "app store", "skip to main content",
)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def html_to_text(raw_html: str) -> str:
    """Render HTML to clean plain text."""
    parser = _TextExtractor()
    try:
        parser.feed(raw_html)
        parser.close()
    except Exception:  # noqa: BLE001 - malformed markup must not kill the build
        pass
    out = html.unescape(parser.text())
    out = out.replace("\u00a0", " ").replace("\u200b", "")
    out = re.sub(r"[ \t\r\f\v]+", " ", out)
    out = re.sub(r"\n\s*\n+", "\n", out)
    lines = [ln.strip() for ln in out.split("\n")]
    kept = [
        ln for ln in lines
        if ln and not any(b in ln.lower() for b in _BOILERPLATE) or len(ln) > 90
    ]
    return "\n".join(kept).strip()


def visible_text(raw_html: str) -> str:
    """Text as a browser would render it, with the nav/footer chrome removed."""
    text = html_to_text(raw_html)
    if "__NEXT_DATA__" in raw_html:  # strip Next.js SSR payload leftovers
        text = re.sub(r"\s+", " ", text)
    return text


# ------------------------------------------------------------------ json

_NEXT_DATA = re.compile(
    r'<script[^>]+id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S | re.I
)
_APOLLO = re.compile(
    r'<script[^>]+type="application/json"[^>]*>(.*?)</script>', re.S | re.I
)


def _try_json(blob: str) -> dict | None:
    try:
        data = json.loads(blob)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def embedded_json(raw_html: str) -> dict | None:
    """Return the first parseable embedded JSON payload on the page."""
    for pattern in (_NEXT_DATA, _APOLLO):
        m = pattern.search(raw_html)
        if m:
            data = _try_json(m.group(1))
            if data:
                return data
    return None


# ------------------------------------------------------------------ doc model

@dataclass
class Document:
    doc_id: str
    source_id: str
    url: str
    title: str
    publisher: str
    doc_type: str
    scheme_key: str | None
    topics: tuple[str, ...]
    quotable: bool
    verified_at_build: bool
    text: str = ""
    facts: dict = field(default_factory=dict)
    fetched_at: str = ""
    content_sha256: str = ""

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id, "source_id": self.source_id, "url": self.url,
            "title": self.title, "publisher": self.publisher, "doc_type": self.doc_type,
            "scheme_key": self.scheme_key, "topics": list(self.topics),
            "quotable": self.quotable, "verified_at_build": self.verified_at_build,
            "text": self.text, "facts": self.facts, "fetched_at": self.fetched_at,
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        return cls(
            doc_id=d["doc_id"], source_id=d["source_id"], url=d["url"],
            title=d["title"], publisher=d["publisher"], doc_type=d["doc_type"],
            scheme_key=d.get("scheme_key"), topics=tuple(d.get("topics", ())),
            quotable=bool(d.get("quotable", True)),
            verified_at_build=bool(d.get("verified_at_build", True)),
            text=d.get("text", ""), facts=d.get("facts", {}),
            fetched_at=d.get("fetched_at", ""),
            content_sha256=d.get("content_sha256", ""),
        )


def parse_document(source, fetch_result, raw_html: str) -> Document:
    """Turn a fetched page into a normalised Document."""
    text = visible_text(raw_html) if raw_html else ""
    facts: dict = {}
    payload = embedded_json(raw_html) if raw_html else None
    if payload:
        mf = (
            payload.get("props", {})
                 .get("pageProps", {})
                 .get("mfServerSideData")
        )
        if isinstance(mf, dict):
            facts = normalise_facts(mf)
    return Document(
        doc_id=source.source_id,
        source_id=source.source_id,
        url=source.url,
        title=source.title,
        publisher=source.publisher,
        doc_type=source.doc_type,
        scheme_key=source.scheme_key,
        topics=source.topics,
        quotable=source.quotable,
        verified_at_build=source.verified_at_build,
        text=text,
        facts=facts,
        fetched_at=fetch_result.fetched_at,
        content_sha256=fetch_result.sha256,
    )


# ------------------------------------------------------------------ facts

#: Values the source uses to mean "no exit load". Anything in this set becomes
#: the canonical string ``"Nil"`` so phase 4 can render one consistent phrase.
_NIL_TOKENS = {"nil", "none", "na", "n/a", "-", "--", "not applicable", "0%"}


def _as_text(raw) -> str:
    """Coerce any source value to a stripped string. Never raises."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, (dict, list, tuple)):
        return ""  # structured payload where a scalar was expected: no value
    return str(raw).strip()


def _as_bool(raw) -> bool:
    """Coerce a source flag to bool without the `bool("false") is True` trap.

    The live payload sends real JSON booleans, so `bool()` happens to work
    today; it silently inverts the fact the moment the field arrives as the
    string "false" (a common API variation), turning `sip_allowed=False` into
    a published `sip_allowed=True`.
    """
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return raw != 0
    if isinstance(raw, str):
        return raw.strip().lower() in {"true", "yes", "y", "1"}
    return False


def _lock_in_days(raw) -> int | None:
    if not isinstance(raw, dict):
        return None
    y, m, d = raw.get("years"), raw.get("months"), raw.get("days")
    if y is None and m is None and d is None:
        return None
    try:
        y, m, d = int(y or 0), int(m or 0), int(d or 0)
    except (TypeError, ValueError):
        return None  # a non-numeric component means "unknown", not "crash"
    return y * 365 + m * 30 + d


def _fund_manager(mf: dict) -> list[str]:
    """The scheme's current fund managers, primary first.

    The payload carries **two** fields that disagree, on all five collected
    pages:

    ==========================  =====================  =========================
    scheme                      ``fund_manager``        ``fund_manager_details``
    ==========================  =====================  =========================
    HDFC Large Cap              Prashant Jain           Rahul Baijal, Dhruv Muchhal
    HDFC Flexi Cap              Prashant Jain           Dhruv Muchhal, Amit Ganatra
    HDFC ELSS Tax Saver         Vinay Kulkarni          Amar Kalkundrikar, Dhruv Muchhal
    HDFC Small Cap              Chirag Setalvad         Dhruv Muchhal, Chirag Setalvad
    HDFC Balanced Advantage     Srinivas Rao Ravuri     Anil Bamboli + 5 more
    ==========================  =====================  =========================

    ``fund_manager`` is a legacy single-name field the platform left populated
    with whoever managed the scheme years ago. ``fund_manager_details`` is the
    array the page actually renders, and its first entry is the manager the
    page's own visible prose calls "the Current Fund Manager" -- so the two
    halves of the same page corroborate each other and the legacy field
    contradicts both.

    Reading the legacy field is what put a stale name into a fact chunk behind a
    *correct* citation URL, which is the one failure the guardrails cannot
    catch: the URL is in the allow-list and the sentence is a verbatim template,
    so G2 passed and the assistant stated a false claim with a real citation.

    **The whole array is this scheme's team, not the AMC's roster.** That was
    worth proving rather than assuming: every entry carries a ``scheme_code``
    equal to the page's own ``scheme_code``, and every entry lists this scheme
    inside its own ``funds_managed``. Taking only ``details[0]`` -- on the
    theory that the rest was shared carousel chrome, the way the AUM figure is
    -- silently dropped real co-managers, and for HDFC Small Cap it dropped
    Chirag Setalvad, who has managed it since 2014.

    Names are de-duplicated and order-preserving, because a manager listed twice
    is a data artefact, not a second co-manager. Falls back to the legacy field
    only when the array yields no name at all.
    """
    names: list[str] = []
    details = mf.get("fund_manager_details")
    if isinstance(details, list):
        for entry in details:
            name = ""
            if isinstance(entry, dict):
                name = _as_text(entry.get("person_name"))
            elif isinstance(entry, str):
                name = entry.strip()
            if name and name not in names:
                names.append(name)
    if not names:
        legacy = _as_text(mf.get("fund_manager"))
        if legacy:
            names.append(legacy)
    return names


def normalise_facts(mf: dict) -> dict:
    """Flatten the page payload into the canonical fact schema.

    Only *facts* are extracted here.  Nothing in this module derives, ranks or
    compares performance - return figures present in the payload are dropped
    on purpose so they can never leak into an answer.
    """
    lock = _lock_in_days(mf.get("lock_in"))
    exit_load = _as_text(mf.get("exit_load"))
    if exit_load.lower() in _NIL_TOKENS:
        exit_load = "Nil"
    return {
        # identity
        "scheme_name": mf.get("scheme_name"),
        "fund_name": mf.get("fund_name"),
        "scheme_code": mf.get("scheme_code"),
        "isin": mf.get("isin"),
        "amc": mf.get("amc") or mf.get("fund_house"),
        "category": mf.get("category"),
        "sub_category": mf.get("sub_category"),
        "plan_type": mf.get("plan_type"),
        "scheme_type": mf.get("scheme_type"),
        "launch_date": mf.get("launch_date"),
        "fund_manager": _fund_manager(mf),
        "registrar_agent": mf.get("registrar_agent"),
        # costs
        "expense_ratio_pct": mf.get("expense_ratio"),
        "base_expense_ratio_pct": mf.get("base_expense_ratio"),
        "exit_load": exit_load if exit_load and exit_load.lower() not in {"nil", "none"} else "Nil",
        "stamp_duty": mf.get("stamp_duty"),
        "direct_scheme_code": mf.get("direct_scheme_code"),
        # minimums
        "min_sip_amount": mf.get("min_sip_investment"),
        "min_lumpsum_amount": mf.get("min_investment_amount"),
        "min_additional_amount": mf.get("mini_additional_investment"),
        "max_sip_amount": mf.get("max_sip_investment"),
        "min_withdrawal_amount": mf.get("min_withdrawal"),
        "sip_multiplier": mf.get("sip_multiplier"),
        "sip_allowed": _as_bool(mf.get("sip_allowed")),
        "lumpsum_allowed": _as_bool(mf.get("lumpsum_allowed")),
        # lock-in
        "lock_in_days": lock,
        # index / risk
        "benchmark": mf.get("benchmark_name") or mf.get("benchmark"),
        "nav": mf.get("nav"),
        "nav_date": mf.get("nav_date"),
        "closed_scheme": _as_bool(mf.get("closed_scheme")),
        # Fund size, in crores. A published scale fact, not a return, so it is
        # admissible under FR-4 -- but it MUST come from this payload field and
        # never from the page prose. Each scheme page carries a sentence reading
        # "The fund currently has an Asset Under Management(AUM) of Rs 9,86,237
        # Cr", and that identical figure appears on all five pages: it is the
        # AMC/platform-level total scraped out of shared chrome, not the scheme's
        # own size. The per-scheme `aum` field is the only trustworthy source.
        "aum": mf.get("aum"),
    }


# ------------------------------------------------------------------ sentences

_ABBREV = {"e.g", "i.e", "vs", "mr", "mrs", "ms", "dr", "no", "p.a", "amc", "sid", "kim", "nav", "sebi", "amfi"}


def normalise_date(raw: str) -> str:
    """Normalise any source date format to ISO-8601 (YYYY-MM-DD)."""
    from datetime import date as _date
    raw = (raw or "").strip()
    if not raw:
        return ""
    for fmt in ("%d-%b-%Y", "%d %b %Y", "%Y-%m-%d", "%d-%m-%Y", "%d-%B-%Y",
                "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return _date(*_strptime(raw, fmt)[:3]).isoformat()
        except (ValueError, TypeError):
            continue
    return raw


def _strptime(value: str, fmt: str):
    import time
    return time.strptime(value, fmt)


def split_sentences(text: str) -> list[str]:
    """Dependency-free sentence splitter tuned for financial page text."""
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9₹\"'(])", text)
    out: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # merge obvious abbreviation splits back together
        if out:
            tail = out[-1].split()[-1].lower().rstrip(".") if out[-1].split() else ""
            if tail in _ABBREV:
                out[-1] = f"{out[-1]} {part}"
                continue
        out.append(part)
    return out
