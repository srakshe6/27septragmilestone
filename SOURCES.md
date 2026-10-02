# Source list

**AMC in scope:** HDFC Mutual Fund (HDFC Asset Management Company)  
**Schemes in scope (direct-growth variants):** Large Cap, Flexi Cap, ELSS Tax Saver, Small Cap, Balanced Advantage

This file is generated from `src/mf_faq/sources.py`, which is the allow-list the assistant's guardrail stage checks every citation against. A URL that is not in this file can never appear in an answer.

---

## 1. The five URLs named in the brief

These are the quotable fact sources: public, machine-retrievable, and verified at build time. Every figure in an answer comes from one of them.

| # | Scheme | URL |
|---|--------|-----|
| 1 | HDFC Large Cap Fund (Direct - Growth) | <https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth> |
| 2 | HDFC Flexi Cap Fund (Direct - Growth) | <https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth> |
| 3 | HDFC ELSS Tax Saver Fund (Direct Plan - Growth) | <https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth> |
| 4 | HDFC Small Cap Fund (Direct - Growth) | <https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth> |
| 5 | HDFC Balanced Advantage Fund (Direct - Growth) | <https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth> |

## 2. Full registry

| Source id | URL | Publisher | Doc type | Role | Verified at build |
|-----------|-----|-----------|----------|------|-------------------|
| `groww-large-cap` | <https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth> | Groww (SEBI-registered investment adviser platform) | Scheme overview page | quotable | yes |
| `groww-flexi-cap` | <https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth> | Groww (SEBI-registered investment adviser platform) | Scheme overview page | quotable | yes |
| `groww-elss` | <https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth> | Groww (SEBI-registered investment adviser platform) | Scheme overview page | quotable | yes |
| `groww-small-cap` | <https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth> | Groww (SEBI-registered investment adviser platform) | Scheme overview page | quotable | yes |
| `groww-balanced-advantage` | <https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth> | Groww (SEBI-registered investment adviser platform) | Scheme overview page | quotable | yes |
| `hdfc-investor-portal` | <https://investor.hdfcfund.com/> | HDFC Mutual Fund (official AMC) | Investor portal | pointer only | no &mdash; URLError: <urlopen error [Errno 11001] getaddrinfo failed> |
| `cams-statements` | <https://www.camsonline.com/Service/Request> | CAMS (official Registrar and Transfer Agent for HDFC MF schemes) | RTA statement request page | pointer only | no &mdash; fetched OK, but yields no quotable document (see note) |
| `hdfc-official-scheme-docs` | <https://www.hdfcmf.com/mutual-funds/equity/hdfc-large-cap-fund> | HDFC Mutual Fund (official AMC) | Official scheme document index | pointer only | no &mdash; fetched OK, but yields no quotable document (see note) |
| `amfi-nav` | <https://portal.amfiindia.com/spages/NAVAll.txt> | Association of Mutual Funds of India (AMFI) | Official daily NAV feed | pointer only | no &mdash; fetched OK, but yields no quotable document (see note) |

### Notes per source

- **`hdfc-investor-portal`** &mdash; Official AMC portal used as the *destination* for statement/tax-document downloads. The host rejects non-browser clients, so no content is extracted; the citation tells the user where to go.
- **`cams-statements`** &mdash; Official RTA self-service page. Verified reachable at build time but client-rendered, so cited as a navigation destination rather than quoted. See README 'Known limits' for the PAN/email privacy note.
- **`hdfc-official-scheme-docs`** &mdash; Canonical AMC page carrying the Scheme Information Document (SID), Key Information Memorandum (KIM), monthly factsheet (which states the SEBI riskometer) and the exit-load schedule. hdfcmf.com is a bot-protected client-rendered SPA, so it is cited as the authoritative place to read a scheme's own documents rather than scraped.
- **`amfi-nav`** &mdash; Pointer-only, but not because it failed to fetch: this feed does download (a bulk ISIN/NAV table, ~1.5 MB).  It is excluded because it carries no prose about fees, minimums or exit load, so it can corroborate a NAV value but can never be quoted in answer to a question.  Used at build time to cross-check the NAV and NAV date shown on the scheme pages.

## 3. Sourcing policy

- **Public sources only.** No login, no paywall, no app back-end, no screenshots, no third-party blog or news article is used as a source of fact.
- **One AMC.** All scheme facts come from HDFC Mutual Fund schemes. No cross-AMC comparison is possible or attempted.
- **Pointer vs quotable.** A `pointer only` source is cited as the place to read or download something on an official system. Nothing is quoted from it, so a page that renders client-side cannot put unverifiable text into an answer.
- **No performance data is ingested.** The scraper deliberately drops the return/performance fields present in the page payload (`src/mf_faq/parsing.py::normalise_facts`) so they cannot reach an answer.
- **Privacy.** The CAMS and AMC-portal pages are cited as navigation destinations. The assistant neither asks for nor stores PAN, Aadhaar, folio/account numbers, OTPs, email addresses or phone numbers; requests for those go to the official portal, not to this system.
