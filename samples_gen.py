"""Generate the sample Q&A deliverable by actually running the assistant.

Run:  python run.py samples

Nothing here is hand-written: every answer in samples/sample_qa.md and
samples/sample_qa.json is produced by the live pipeline, so the deliverable
cannot drift from the system's real behaviour.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mf_faq.config import CONFIG, DISCLAIMER, LAST_UPDATED_PREFIX  # noqa: E402
from mf_faq.pipeline import FAQPipeline                           # noqa: E402

#: (query, why this query is in the sample).  Ten cases, chosen to cover every
#: field family, both procedure destinations, the missing-fact case and all three
#: refusals - not ten variations on "expense ratio".
SAMPLE_QUERIES: list[tuple[str, str]] = [
    ("What is the expense ratio of HDFC Large Cap Fund?",
     "Cost figure, large cap"),
    ("minimum SIP for HDFC ELSS Tax Saver Fund",
     "Minimum investment, terse phrasing with no question mark"),
    ("Is there a lock-in period in HDFC ELSS Tax Saver Fund?",
     "Statutory lock-in"),
    ("What is the exit load on HDFC Balanced Advantage Fund?",
     "Non-standard exit-load wording (excess-units rule)"),
    ("benchmark of HDFC Flexi Cap Fund",
     "Index benchmark, terse phrasing"),
    ("What is the riskometer level of HDFC Small Cap Fund?",
     "Fact not present in the quotable corpus - must point to the official factsheet"),
    ("How do I download a capital gains statement?",
     "Procedural, routes to the RTA"),
    ("Where can I download my account statement?",
     "Procedural, routes to the AMC investor portal (a different destination)"),
    ("Should I buy HDFC Small Cap Fund?",
     "Advice question - must refuse"),
    ("My PAN is ABCDE1234F, what is the exit load?",
     "PII in the query - must refuse and not echo the PAN"),
]

def build(pipe: FAQPipeline | None = None) -> tuple[str, str]:
    """Run the ten queries and return (json_text, markdown_text) without writing.

    Kept separate from :func:`generate` so a test can compare what the pipeline
    produces *now* against what is committed on disk. A generate-then-compare
    test is vacuous: it overwrites the file it is meant to be checking.
    """
    own = pipe is None
    pipe = pipe or FAQPipeline()
    # The LLM rewrite pass is forced off for this run. The contract of this
    # artifact is that it is a pure function of the built index, and a
    # rewriter is not deterministic: with `MF_FAQ_LLM=1` and a live key, two
    # runs of the same ten queries produce differently-worded answers and
    # `test_samples_regenerate_byte_identically` fails. The committed samples
    # are therefore always the deterministic extractive answers, whatever the
    # environment says. `CONFIG` is global, so the caller's setting is restored
    # on the way out rather than left silently flipped.
    was_enabled = CONFIG.llm_enabled
    pipe.set_llm(False)
    try:
        rows = []
        for q, note in SAMPLE_QUERIES:
            ans = pipe.ask(q)
            rows.append({
                "query": q,
                "intent_note": note,
                "kind": ans.kind,
                "answer": ans.text,
                "citation": ans.citation,
                "citation_title": ans.citation_title,
                "last_updated_from_sources": ans.last_updated,
                "disclaimer": ans.disclaimer,
                "sentence_count": len(_sents(ans.text)),
                "guards_passed": ans.guards,
            })

        # The stamp is the index build time, not the wall clock at render time.
        # `datetime.now()` here made `run.py samples` produce a different file on
        # every run, which defeats the entire point of the deliverable: the spec
        # requires the regenerated file to be byte-identical so that a diff means
        # the *answers* changed. Stamping with built_at keeps a meaningful time
        # while making the artifact a pure function of the built index.
        generated_at = pipe.stats().get("built_at") or "unknown"
        json_text = json.dumps(
            {"index_built_at": generated_at, "disclaimer": DISCLAIMER, "samples": rows},
            indent=2, ensure_ascii=False)
        md_text = _render_markdown(rows, generated_at)
        return json_text, md_text
    finally:
        pipe.set_llm(was_enabled)
        if own:
            pipe.close()


def generate(pipe: FAQPipeline | None = None) -> tuple[Path, Path]:
    """Run the assistant and write both sample artifacts."""
    json_text, md_text = build(pipe)
    jpath = ROOT / "samples" / "sample_qa.json"
    mpath = ROOT / "samples" / "sample_qa.md"
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json_text, encoding="utf-8")
    mpath.write_text(md_text, encoding="utf-8")
    return jpath, mpath


def _render_markdown(rows: list[dict], generated_at: str) -> str:
    lines = [
        "# Sample Q&A - MF facts assistant",
        "",
        f"Index built at: {generated_at}  ",
        f"Disclaimer shown in the UI: **{DISCLAIMER}**  ",
        "Scope: HDFC Mutual Fund - Large Cap, Flexi Cap, ELSS Tax Saver, Small Cap, "
        "Balanced Advantage (direct-growth variants).",
        "",
        "Every answer below was produced by `python run.py samples` running the live "
        "pipeline; none of it is hand-written. Re-running the generator is byte-identical: "
        "the only time recorded is the index build, so a diff here means an answer, a "
        "citation or a guard changed. Latency is measured separately by `python bench.py`.",
        "",
        "| # | Query | Outcome | Source |",
        "|---|-------|---------|--------|",
    ]
    for i, r in enumerate(rows, 1):
        outcome = {"answer": "answered", "advice_refusal": "refused (advice)",
                   "performance_refusal": "refused (performance)",
                   "pii_refusal": "refused (PII)"}.get(r["kind"], r["kind"])
        cite = r["citation"].replace("https://", "") if r["citation"] else "-"
        q = r["query"].replace("|", "\\|")
        lines.append(f"| {i} | {q} | {outcome} | [{cite}]({r['citation']}) |"
                     if r["citation"] else f"| {i} | {q} | {outcome} | - |")

    lines += ["", "---", ""]
    for i, r in enumerate(rows, 1):
        lines += [
            f"### {i}. {r['query']}",
            "",
            f"*{r['intent_note']}*",
            "",
            "```",
            r["answer"],
            f"Source: {r['citation'] or '-'}",
            f"{LAST_UPDATED_PREFIX}{r['last_updated_from_sources']}",
            f"{r['disclaimer']}",
            "```",
            "",
            f"Sentences: {r['sentence_count']} &middot; Outcome: `{r['kind']}` &middot; "
            f"Guards: `{', '.join(r['guards_passed']) or 'none'}`",
            "",
        ]

    lines += [
        "---",
        "",
        "## Notes on these answers",
        "",
        "- No answer recommends, ranks or compares schemes, and none states or computes a return.",
        "- Every answer carries exactly one source link, drawn from the allow-list in `src/mf_faq/sources.py`.",
        "- The riskometer query is answered by pointing at the AMC's monthly factsheet: the "
        "riskometer value is not present in the pages collected for this build, and the "
        "assistant says so rather than guessing.",
        "- Query 10 is the PII case, and it is quoted verbatim here on purpose: the sample "
        "exists to document the refusal, which is not demonstrable without the triggering "
        "query. The assistant's reply never echoes the PAN, and the runtime redacts it before "
        "the query is logged or routed. So the PAN occurs in this file only in the query text "
        "above, and in no answer, citation or guard record.",
        "",
    ]
    return "\n".join(lines)


def _sents(text: str) -> list[str]:
    from mf_faq.parsing import split_sentences
    return split_sentences(text or "")


if __name__ == "__main__":
    j, m = generate()
    print(f"wrote {j}\nwrote {m}")
