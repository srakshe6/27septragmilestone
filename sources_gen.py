"""Regenerate SOURCES.md and SOURCES.csv from the source registry.

Run:  python run.py sources --write

The registry in `src/mf_faq/sources.py` is the single source of truth; these
files are a rendering of it, so they cannot fall out of sync with what the
assistant is actually allowed to cite.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mf_faq.ingest import FetchResult  # noqa: E402,F401
from mf_faq.sources import BRIEF_URLS, SOURCES, SCHEMES  # noqa: E402

CSV_COLS = [
    "source_id", "url", "publisher", "doc_type", "scheme", "role",
    "quotable", "verified_at_build", "in_brief_five", "note",
]


def _row(s) -> dict:
    scheme = SCHEMES[s.scheme_key]["label"] if s.scheme_key else "-"
    if not s.quotable:
        role = "pointer/destination citation"
    elif s.scheme_key:
        role = "quotable fact source"
    else:
        role = "quotable prose source"
    return {
        "source_id": s.source_id, "url": s.url, "publisher": s.publisher,
        "doc_type": s.doc_type, "scheme": scheme, "role": role,
        "quotable": "yes" if s.quotable else "no",
        "verified_at_build": "yes" if s.verified_at_build else "no",
        "in_brief_five": "yes" if s.url in BRIEF_URLS else "no",
        "note": s.note,
    }


def write_csv(path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        for s in SOURCES:
            w.writerow(_row(s))
    return path


def write_md(path: Path, manifest: dict | None = None) -> Path:
    fetched: dict[str, dict] = {}
    if manifest:
        fetched = {r["source_id"]: r for r in manifest.get("results", [])}

    L: list[str] = [
        "# Source list",
        "",
        "**AMC in scope:** HDFC Mutual Fund (HDFC Asset Management Company)  ",
        "**Schemes in scope (direct-growth variants):** Large Cap, Flexi Cap, "
        "ELSS Tax Saver, Small Cap, Balanced Advantage",
        "",
        "This file is generated from `src/mf_faq/sources.py`, which is the "
        "allow-list the assistant's guardrail stage checks every citation against. "
        "A URL that is not in this file can never appear in an answer.",
        "",
        "---",
        "",
        "## 1. The five URLs named in the brief",
        "",
        "These are the quotable fact sources: public, machine-retrievable, and "
        "verified at build time. Every figure in an answer comes from one of them.",
        "",
        "| # | Scheme | URL |",
        "|---|--------|-----|",
    ]
    for i, (key, u) in enumerate(zip(
            ["large_cap", "flexi_cap", "elss", "small_cap", "balanced_advantage"],
            BRIEF_URLS), 1):
        L.append(f"| {i} | {SCHEMES[key]['label']} | <{u}> |")

    L += [
        "",
        "## 2. Full registry",
        "",
        "| Source id | URL | Publisher | Doc type | Role | Verified at build |",
        "|-----------|-----|-----------|----------|------|-------------------|",
    ]
    for s in SOURCES:
        fr = fetched.get(s.source_id, {})
        if s.verified_at_build:
            ok = "yes"
        elif fr.get("ok"):
            # Fetched successfully, but the ingest stage could not turn it into a
            # quotable document.  Say exactly that -- do not invent a fetch failure.
            ok = ("no &mdash; fetched OK, but yields no quotable document "
                  "(see note)")
        else:
            ok = f"no &mdash; {fr.get('error') or 'not fetched'}"
        L.append(f"| `{s.source_id}` | <{s.url}> | {s.publisher} | {s.doc_type} | "
                 f"{'quotable' if s.quotable else 'pointer only'} | {ok} |")

    L += [
        "",
        "### Notes per source",
        "",
    ]
    for s in SOURCES:
        if s.note:
            L.append(f"- **`{s.source_id}`** &mdash; {s.note}")

    L += [
        "",
        "## 3. Sourcing policy",
        "",
        "- **Public sources only.** No login, no paywall, no app back-end, no "
        "screenshots, no third-party blog or news article is used as a source of fact.",
        "- **One AMC.** All scheme facts come from HDFC Mutual Fund schemes. "
        "No cross-AMC comparison is possible or attempted.",
        "- **Pointer vs quotable.** A `pointer only` source is cited as the place "
        "to read or download something on an official system. Nothing is quoted "
        "from it, so a page that renders client-side cannot put unverifiable text "
        "into an answer.",
        "- **No performance data is ingested.** The scraper deliberately drops the "
        "return/performance fields present in the page payload "
        "(`src/mf_faq/parsing.py::normalise_facts`) so they cannot reach an answer.",
        "- **Privacy.** The CAMS and AMC-portal pages are cited as navigation "
        "destinations. The assistant neither asks for nor stores PAN, Aadhaar, "
        "folio/account numbers, OTPs, email addresses or phone numbers; requests "
        "for those go to the official portal, not to this system.",
        "",
    ]
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def generate() -> tuple[Path, Path]:
    mpath = ROOT / "data" / "fetch_manifest.json"
    manifest = None
    if mpath.exists():
        import json
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    return write_csv(ROOT / "SOURCES.csv"), write_md(ROOT / "SOURCES.md", manifest)


if __name__ == "__main__":
    c, m = generate()
    print(f"wrote {c}\nwrote {m}")
