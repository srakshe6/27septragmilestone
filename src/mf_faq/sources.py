"""STAGE 0 - Source registry.

A single, auditable allow-list of every public page the assistant may cite.
Nothing outside this registry can ever be returned as a citation: the
guardrail stage validates every emitted URL against it.

Scope decision (documented in README):
  AMC ......... HDFC Mutual Fund (HDFC Asset Management Company)
  Schemes ..... Large Cap, Flexi Cap, ELSS Tax Saver, Small Cap,
                Balanced Advantage  (5 schemes, direct-growth variants)

Citation policy: the five scheme pages below are the *quotable corpus* -
they are public, server-rendered and machine-verifiable at build time, and
they are the exact URLs specified in the brief.  The two portal URLs are
*destination* citations for procedural questions ("where do I download a
statement") where the answer is a pointer to an official system rather than
a number.  Official AMC/SEBI/AMFI document deep-links are registered too so
that "link me the official factsheet" resolves, but they are tagged
`verified_at_build=False` because hdfcmf.com is a bot-protected,
client-rendered SPA whose content is not retrievable by a plain HTTP client.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    source_id: str
    url: str
    title: str
    publisher: str
    doc_type: str
    scheme_key: str | None = None
    quotable: bool = True          # may be quoted for facts in an answer
    verified_at_build: bool = True  # fetchable + parsed by the ingest stage
    note: str = ""
    topics: tuple[str, ...] = field(default_factory=tuple)


# ------------------------------------------------------------------ schemes
SCHEMES: dict[str, dict] = {
    "large_cap": {
        "label": "HDFC Large Cap Fund (Direct - Growth)",
        "category": "Large cap",
        "search_id": "hdfc-large-cap-fund-direct-growth",
        "aliases": [
            "large cap", "largecap", "hdfc large cap", "large cap fund",
            "hdfc large cap fund", "hdfclargecap",
        ],
    },
    "flexi_cap": {
        "label": "HDFC Flexi Cap Fund (Direct - Growth)",
        "category": "Flexi cap",
        "search_id": "hdfc-equity-fund-direct-growth",
        "aliases": [
            "flexi cap", "flexicap", "hdfc flexi cap", "flexi cap fund",
            "hdfc equity fund", "hdfc equity", "equity fund", "hdfc flexi cap fund",
        ],
    },
    "elss": {
        "label": "HDFC ELSS Tax Saver Fund (Direct Plan - Growth)",
        "category": "ELSS (tax-saver)",
        "search_id": "hdfc-elss-tax-saver-fund-direct-plan-growth",
        "aliases": [
            "elss", "elss tax saver", "tax saver", "elss fund",
            "hdfc elss", "hdfc elss tax saver fund", "elss tax saver fund",
            "80c", "tax saving", "tax-saver",
        ],
    },
    "small_cap": {
        "label": "HDFC Small Cap Fund (Direct - Growth)",
        "category": "Small cap",
        "search_id": "hdfc-small-cap-fund-direct-growth",
        "aliases": [
            "small cap", "smallcap", "hdfc small cap", "small cap fund",
            "hdfc small cap fund",
        ],
    },
    "balanced_advantage": {
        "label": "HDFC Balanced Advantage Fund (Direct - Growth)",
        "category": "Hybrid / balanced advantage",
        "search_id": "hdfc-balanced-advantage-fund-direct-growth",
        "aliases": [
            "balanced advantage", "balancedadvantage", "hybrid",
            "hdfc balanced advantage fund", "balanced advantage fund",
            "hdfc balanced advantage", "balanced fund",
        ],
    },
}

AMC = "HDFC Mutual Fund (HDFC Asset Management Company)"


def _scheme_url(search_id: str) -> str:
    return f"https://groww.in/mutual-funds/{search_id}"


# ------------------------------------------------------------------ registry
SOURCES: list[Source] = [
    Source(
        source_id="groww-large-cap",
        url=_scheme_url(SCHEMES["large_cap"]["search_id"]),
        title="HDFC Large Cap Fund Direct Growth - scheme overview",
        publisher="Groww (SEBI-registered investment adviser platform)",
        doc_type="Scheme overview page",
        scheme_key="large_cap",
        topics=("expense_ratio", "exit_load", "min_sip", "benchmark", "scheme_meta"),
    ),
    Source(
        source_id="groww-flexi-cap",
        url=_scheme_url(SCHEMES["flexi_cap"]["search_id"]),
        title="HDFC Flexi Cap Direct Plan Growth - scheme overview",
        publisher="Groww (SEBI-registered investment adviser platform)",
        doc_type="Scheme overview page",
        scheme_key="flexi_cap",
        topics=("expense_ratio", "exit_load", "min_sip", "benchmark", "scheme_meta"),
    ),
    Source(
        source_id="groww-elss",
        url=_scheme_url(SCHEMES["elss"]["search_id"]),
        title="HDFC ELSS Tax Saver Fund Direct Plan Growth - scheme overview",
        publisher="Groww (SEBI-registered investment adviser platform)",
        doc_type="Scheme overview page",
        scheme_key="elss",
        topics=("lock_in", "min_sip", "expense_ratio", "exit_load", "benchmark", "scheme_meta"),
    ),
    Source(
        source_id="groww-small-cap",
        url=_scheme_url(SCHEMES["small_cap"]["search_id"]),
        title="HDFC Small Cap Fund Direct Growth - scheme overview",
        publisher="Groww (SEBI-registered investment adviser platform)",
        doc_type="Scheme overview page",
        scheme_key="small_cap",
        topics=("expense_ratio", "exit_load", "min_sip", "benchmark", "scheme_meta"),
    ),
    Source(
        source_id="groww-balanced-advantage",
        url=_scheme_url(SCHEMES["balanced_advantage"]["search_id"]),
        title="HDFC Balanced Advantage Fund Direct Growth - scheme overview",
        publisher="Groww (SEBI-registered investment adviser platform)",
        doc_type="Scheme overview page",
        scheme_key="balanced_advantage",
        topics=("expense_ratio", "exit_load", "min_sip", "benchmark", "scheme_meta"),
    ),
    # ------------------------------------------------ procedural / pointer sources
    Source(
        source_id="hdfc-investor-portal",
        url="https://investor.hdfcfund.com/",
        title="HDFC Mutual Fund - investor portal (transaction & statement requests)",
        publisher="HDFC Mutual Fund (official AMC)",
        doc_type="Investor portal",
        quotable=False,
        verified_at_build=False,
        note=(
            "Official AMC portal used as the *destination* for statement/tax-document "
            "downloads. The host rejects non-browser clients, so no content is "
            "extracted; the citation tells the user where to go."
        ),
        topics=("statement_download", "tax_doc"),
    ),
    Source(
        source_id="cams-statements",
        url="https://www.camsonline.com/Service/Request",
        title="CAMS Online - Request a statement / capital-gains (loss) statement",
        publisher="CAMS (official Registrar and Transfer Agent for HDFC MF schemes)",
        doc_type="RTA statement request page",
        quotable=False,
        verified_at_build=False,
        note=(
            "Official RTA self-service page. Verified reachable at build time but "
            "client-rendered, so cited as a navigation destination rather than "
            "quoted. See README 'Known limits' for the PAN/email privacy note."
        ),
        topics=("statement_download", "tax_doc"),
    ),
    # ------------------------------------------------ official document deep-links
    Source(
        source_id="hdfc-official-scheme-docs",
        url="https://www.hdfcmf.com/mutual-funds/equity/hdfc-large-cap-fund",
        title="HDFC Mutual Fund - official scheme pages (SID / KIM / factsheet / exit load)",
        publisher="HDFC Mutual Fund (official AMC)",
        doc_type="Official scheme document index",
        quotable=False,
        verified_at_build=False,
        note=(
            "Canonical AMC page carrying the Scheme Information Document (SID), Key "
            "Information Memorandum (KIM), monthly factsheet (which states the "
            "SEBI riskometer) and the exit-load schedule. hdfcmf.com is a "
            "bot-protected client-rendered SPA, so it is cited as the authoritative "
            "place to read a scheme's own documents rather than scraped."
        ),
        topics=("riskometer", "exit_load", "factsheet", "benchmark", "scheme_meta"),
    ),
    Source(
        source_id="amfi-nav",
        url="https://portal.amfiindia.com/spages/NAVAll.txt",
        title="AMFI - NAV of all schemes (official daily NAV feed)",
        publisher="Association of Mutual Funds of India (AMFI)",
        doc_type="Official daily NAV feed",
        quotable=False,
        verified_at_build=False,
        note=(
            "Pointer-only, but not because it failed to fetch: this feed does download "
            "(a bulk ISIN/NAV table, ~1.5 MB).  It is excluded because it carries no "
            "prose about fees, minimums or exit load, so it can corroborate a NAV value "
            "but can never be quoted in answer to a question.  Used at build time to "
            "cross-check the NAV and NAV date shown on the scheme pages."
        ),
        topics=("nav", "scheme_meta"),
    ),
]

SOURCES_BY_ID: dict[str, Source] = {s.source_id: s for s in SOURCES}

#: The hard citation allow-list used by the guardrail stage.
ALLOWED_URLS: frozenset[str] = frozenset(s.url for s in SOURCES)

#: The five URLs named in the brief (used for the SOURCES deliverable).
BRIEF_URLS: list[str] = [
    "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
    "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
    "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
    "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
]

#: Educational landing page offered when a question is refused.
#:
#: Hard rule (PRD FR-2 + the stage-8 G2 gate): every URL the assistant is capable
#: of emitting must appear in :data:`ALLOWED_URLS`.  A URL that is not allow-listed
#: is silently dropped by the G2 gate, which would render a refusal with no source
#: link at all -- i.e. the refusal would break the one-citation contract it exists
#: to uphold.  The assertion below makes that structural instead of conventional.
EDUCATION_LINKS: dict[str, str] = {
    # The official AMC page, identical to the link attached to every refusal.
    "default": "https://www.hdfcmf.com/mutual-funds/equity/hdfc-large-cap-fund",
    "riskometer": "https://www.hdfcmf.com/mutual-funds/equity/hdfc-large-cap-fund",
    # The registered AMFI feed: the only AMFI host on the allow-list.
    "amfi": "https://portal.amfiindia.com/spages/NAVAll.txt",
}

assert set(EDUCATION_LINKS.values()) <= ALLOWED_URLS, (
    "every education link must be allow-listed (PRD FR-2 / stage-8 gate G2)"
)


def source_for_scheme(scheme_key: str) -> Source | None:
    for s in SOURCES:
        if s.scheme_key == scheme_key:
            return s
    return None
