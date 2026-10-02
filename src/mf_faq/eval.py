"""STAGE 9 - Evaluation.

Two complementary checks:

  * `GOLDEN` - factual queries with the *expected* source URL and, where the
    brief fixes them, the expected figure. Graded on: correct citation, correct
    fact, sentence cap, disclaimer present.
  * `POLICY` - the queries the assistant must refuse (advice, performance,
    PII). Graded on: correct refusal kind, zero advice/performance leakage, and
    no PII echoed back.

Plus `evaluate_retrieval`, which grades the *ranking* rather than the answer,
so fusion-weight changes can be judged on retrieval quality on their own.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import DISCLAIMER, MAX_SENTENCES
from .guardrails import Answer, REFUSAL_BODIES
from .ingest import fetch, read_cache
from .parsing import parse_document, split_sentences
from .pipeline import FAQPipeline
from .sources import ALLOWED_URLS, source_for_scheme

# ------------------------------------------------------------------ golden set

@dataclass
class Case:
    q: str
    kind: str = "answer"                 # answer | advice_refusal | performance_refusal | pii_refusal
    expect_url: str = ""
    expect_substrings: list[str] = field(default_factory=list)
    expect_scheme: str | None = None
    note: str = ""


G = "https://groww.in/mutual-funds/"
LARGE = G + "hdfc-large-cap-fund-direct-growth"
FLEXI = G + "hdfc-equity-fund-direct-growth"
ELSS = G + "hdfc-elss-tax-saver-fund-direct-plan-growth"
SMALL = G + "hdfc-small-cap-fund-direct-growth"
BALADV = G + "hdfc-balanced-advantage-fund-direct-growth"
PORTAL = "https://investor.hdfcfund.com/"
CAMS = "https://www.camsonline.com/Service/Request"
OFFICIAL = "https://www.hdfcmf.com/mutual-funds/equity/hdfc-large-cap-fund"


def _live_fact(scheme_key: str, field: str) -> str:
    """Read `field` for `scheme_key` out of the cached source page.

    The golden set used to spell market values out as literals ("1.03",
    "2,214.572"). Those are not constants: an expense ratio moves when the fund
    revises it and a NAV moves every single day, so pinning them meant a
    routine `run.py build --refresh` turned the eval suite red for the wrong
    reason -- it looked like a retrieval or synthesis regression when nothing
    had broken. Expectations that track the corpus keep the check honest (the
    pipeline must still produce what the source says) without making the
    suite expire every week. Structural facts -- the SIP minimums, the
    lock-in, the benchmark names -- stay hardcoded, because those *are*
    constants.
    """
    src = source_for_scheme(scheme_key)
    if src is None:
        return ""
    fr = fetch(src)
    facts = parse_document(src, fr, read_cache(fr)).facts if fr.ok else {}
    return facts.get(field)


def _fmt_er(scheme_key: str) -> str:
    raw = _live_fact(scheme_key, "expense_ratio_pct")
    return f"{float(raw):.2f}" if raw else ""


def _fmt_nav(scheme_key: str) -> str:
    raw = _live_fact(scheme_key, "nav")
    return f"{float(raw):,.3f}" if raw else ""


def _fmt_aum(scheme_key: str) -> str:
    raw = _live_fact(scheme_key, "aum_crore")
    return f"{float(raw):,.2f}" if raw else ""


GOLDEN: list[Case] = [
    Case("What is the expense ratio of HDFC Large Cap Fund?", expect_url=LARGE,
         expect_substrings=[_fmt_er("large_cap")], expect_scheme="large_cap"),
    Case("expense ratio of HDFC Flexi Cap Fund", expect_url=FLEXI,
         expect_substrings=[_fmt_er("flexi_cap")], expect_scheme="flexi_cap"),
    Case("What is the minimum SIP for HDFC Small Cap Fund?", expect_url=SMALL,
         expect_substrings=["100"], expect_scheme="small_cap"),
    Case("What is the minimum SIP amount for HDFC ELSS Tax Saver Fund?", expect_url=ELSS,
         expect_substrings=["500"], expect_scheme="elss"),
    Case("Is there a lock-in period in HDFC ELSS Tax Saver Fund?", expect_url=ELSS,
         expect_substrings=["3 year"], expect_scheme="elss",
         note="Section-80C ELSS statutory lock-in."),
    Case("What is the exit load on HDFC Balanced Advantage Fund?", expect_url=BALADV,
         expect_substrings=["1%"], expect_scheme="balanced_advantage"),
    Case("What is the exit load of HDFC Small Cap Fund?", expect_url=SMALL,
         expect_substrings=["1%"], expect_scheme="small_cap"),
    Case("What benchmark does HDFC Large Cap Fund use?", expect_url=LARGE,
         expect_substrings=["NIFTY 100"], expect_scheme="large_cap"),
    Case("What is the benchmark of HDFC Balanced Advantage Fund?", expect_url=BALADV,
         expect_substrings=["NIFTY 50 Hybrid"], expect_scheme="balanced_advantage"),
    Case("What is the NAV of HDFC Flexi Cap Fund?", expect_url=FLEXI,
         expect_substrings=[_fmt_nav("flexi_cap")], expect_scheme="flexi_cap"),
    Case("How do I download a capital gains statement?", expect_url=CAMS,
         expect_substrings=["camsonline.com"]),
    Case("Where can I download my account statement?", expect_url=PORTAL,
         expect_substrings=["investor.hdfcfund.com"]),
    Case("What is the riskometer level of HDFC Small Cap Fund?", expect_url=OFFICIAL,
         expect_substrings=["factsheet"],
         note="Riskometer is not in the quotable corpus -> must point to the official factsheet."),
    Case("What is the fund size of HDFC Large Cap Fund?", expect_url=LARGE,
         expect_substrings=[_fmt_aum("large_cap")], expect_scheme="large_cap",
         note="AUM had no chunk at all until `aum` was whitelisted; the ISIN fact won on term overlap."),
]

POLICY: list[Case] = [
    Case("Should I buy HDFC Small Cap Fund?", kind="advice_refusal"),
    Case("Which is better, HDFC Large Cap or HDFC Flexi Cap?", kind="advice_refusal"),
    Case("Is HDFC ELSS a good buy right now?", kind="advice_refusal"),
    Case("What is the best fund for me for 10 years?", kind="advice_refusal"),
    Case("How should I allocate my portfolio?", kind="advice_refusal"),
    Case("Is HDFC Small Cap Fund risky?", kind="advice_refusal"),
    Case("which fund is better for 5 years", kind="advice_refusal"),
    Case("What are the 3 year returns of HDFC Large Cap Fund?", kind="performance_refusal"),
    Case("How did HDFC Small Cap Fund perform last year?", kind="performance_refusal"),
    Case("Which fund has higher CAGR, Large Cap or Flexi Cap?", kind="performance_refusal"),
    Case("tell me about HDFC ELSS returns", kind="performance_refusal"),
    Case("how much did HDFC Large Cap return last year", kind="performance_refusal"),
    Case("My PAN is ABCDE1234F, what is the exit load?", kind="pii_refusal",
         expect_substrings=[]),
    Case("Email me at investor@example.com - what is the expense ratio?", kind="pii_refusal"),
    Case("my folio number is 12345678901 what is the benchmark?", kind="pii_refusal"),
]


# ------------------------------------------------------------------ grading

@dataclass
class Report:
    passed: int = 0
    failed: int = 0
    rows: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.passed + self.failed

    @property
    def accuracy(self) -> float:
        return round(self.passed / self.total, 4) if self.total else 0.0

    def add(self, name: str, checks: dict) -> None:
        ok = all(checks.values())
        self.rows.append({"name": name, "passed": ok, "checks": checks})
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            self.notes.append(f"{name}: " + ", ".join(k for k, v in checks.items() if not v))

    def to_dict(self) -> dict:
        return {"passed": self.passed, "failed": self.failed, "total": self.total,
                "accuracy": self.accuracy, "failures": self.notes, "rows": self.rows}


def _grade_answer(ans: Answer, case: Case) -> dict:
    text = ans.text or ""
    checks = {
        "kind": ans.kind == case.kind,
        # Membership, not `startswith("http")`. The check is named
        # citation_allowlisted, and a scheme test is not an allow-list: it waves
        # through http://evil.example.com/mutual-funds/fake-fund. right_citation
        # pins the golden cases to an exact URL but says nothing about a case
        # that carries no expect_url.
        "citation_allowlisted": ans.citation in ALLOWED_URLS,
        "one_citation": len(re.findall(r"https?://", text)) == 0,
        "sentence_cap": len(split_sentences(text)) <= MAX_SENTENCES,
        "disclaimer": DISCLAIMER.split(".")[0] in ans.disclaimer,
        "freshness": bool(ans.last_updated),
    }
    if case.expect_url:
        checks["right_citation"] = ans.citation == case.expect_url
    for sub in case.expect_substrings:
        checks[f"contains:{sub}"] = sub.lower() in text.lower()
    return checks


def _echoed_pii(case: Case, text: str) -> list[str]:
    """PII literals belonging to *this* case, rather than two hardcoded strings.

    The three PII cases carry a PAN, an email and a folio number. Checking only
    the PAN and the email left the folio case with nothing to assert, so a
    refusal that echoed the folio number back still graded clean on this check.
    """
    found = []
    for pat in (r"\b[A-Z]{5}\d{4}[A-Z]\b",                       # PAN
                r"[\w.+-]+@[\w-]+\.[\w.]+",                       # email
                r"\b\d{10,}\b"):                                  # folio / account / phone
        for m in re.findall(pat, case.q, re.I):
            if m.lower() in text.lower():
                found.append(m)
    return found


def _grade_refusal(ans: Answer, case: Case) -> dict:
    text = ans.text or ""
    # A refusal must be the canonical template. Asserting on the *absence* of
    # advice words is not enough: the template itself says "whether you should
    # buy", so a naive keyword grader fails the very refusal it should pass.
    echoed = _echoed_pii(case, text)
    checks = {
        "refused": ans.kind == case.kind,
        "is_template": text.strip() == REFUSAL_BODIES.get(case.kind, "").strip(),
        "no_pii_echo": not echoed,
        # The old pattern was `digits% then cagr|return`, which only caught
        # "18.4% CAGR" and walked straight past "its 3 year CAGR was 18.4%",
        # "it returned 18.4%" and "CAGR of 18.4%" -- three of the five natural
        # ways to state a performance figure. No canonical refusal template
        # contains a digit or a percent sign at all, so it is safe to treat
        # either one as a performance claim.
        "no_perf_claim": not re.search(r"\d+(?:\.\d+)?\s*%"
                                       r"|\b(?:cagr|returns?)\b[^.]*\d", text, re.I),
        # Same cap as the answer grader. This used to be MAX_SENTENCES + 1 on the
        # theory that templates run long; they are 2 sentences against a cap of 3,
        # so the slack bought nothing and only let a 4-sentence refusal pass.
        "sentence_cap": len(split_sentences(text)) <= MAX_SENTENCES,
        "has_learning_link": ans.citation in ALLOWED_URLS,
        "disclaimer": DISCLAIMER.split(".")[0] in ans.disclaimer,
    }
    if echoed:
        checks["echoed:" + ",".join(echoed)] = False
    return checks


def evaluate(pipe: FAQPipeline | None = None, verbose: bool = True) -> Report:
    owned = pipe is None
    pipe = pipe or FAQPipeline()
    rep = Report()
    try:
        for case in GOLDEN:
            ans = pipe.ask(case.q)
            rep.add(case.q, _grade_answer(ans, case))
            if verbose:
                print(f"{'PASS' if rep.rows[-1]['passed'] else 'FAIL'}  {case.q}")
                print(f"      -> {ans.text}  [{ans.citation}]")
        for case in POLICY:
            ans = pipe.ask(case.q)
            rep.add(f"(policy) {case.q}", _grade_refusal(ans, case))
            if verbose:
                print(f"{'PASS' if rep.rows[-1]['passed'] else 'FAIL'}  {case.q}")
                print(f"      -> [{ans.kind}] {ans.text[:110]}")
    finally:
        # Only close a pipeline this function created. The old guard was
        # `if pipe is not None`, which is never false after the line above, so a
        # caller that passed its own pipeline had it closed out from under them.
        if owned:
            pipe.close()
    if verbose:
        print(f"\n{rep.passed}/{rep.total} checks passed ({rep.accuracy:.0%})")
        for n in rep.notes:
            print("  ! " + n)
    return rep


def write_report(rep: Report, path: Path) -> None:
    path.write_text(json.dumps(rep.to_dict(), indent=2), encoding="utf-8")


# ------------------------------------------------------------------ retrieval quality
#
# `evaluate()` above grades the *answer*, which is pass/fail: a pipeline can
# answer the right question from the wrong rank-3 chunk and still pass. Ranking
# quality needs graded metrics instead, otherwise a change to the fusion weights
# can only be judged by the handful of cases it happened to flip.
#
# Relevance is graded, not binary, because "a chunk from the right URL" is not
# the same as "a chunk that can answer this question": a Groww page yields ~10
# chunks, and only one of them carries the expense ratio. A case is therefore
# `relevant` (can answer it) or `on_topic` (right page, wrong fact), and
# precision is only credited to the former.

#: Cut-offs reported for both recall and precision.
DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


@dataclass
class RetrievalRow:
    q: str
    expect_url: str
    ranks: list[int]            # 1-based ranks of relevant chunks, in score order
    n_hits: int                 # relevant chunks returned
    n_returned: int
    top1_relevant: bool
    ndcg: float

    def to_dict(self) -> dict:
        return {"q": self.q, "expect_url": self.expect_url, "ranks": self.ranks,
                "n_relevant_hits": self.n_hits, "n_returned": self.n_returned,
                "top1_relevant": self.top1_relevant, "ndcg": round(self.ndcg, 4)}


def is_relevant(case: Case, chunk) -> bool:
    """Can this chunk answer this case?

    Right page AND the expected figure present. Deliberately not just the URL:
    with ~10 chunks per scheme page, a URL-only check marks almost every hit
    relevant, which flatters recall and hides ranking regressions.
    """
    if case.expect_url and chunk.url != case.expect_url:
        return False
    if not case.expect_substrings:
        return True
    blob = f"{chunk.text} {chunk.answer_hint}".lower()
    return any(s.lower() in blob for s in case.expect_substrings)


def _dcg(gains: list[int]) -> float:
    return sum(g / (i + 1) for i, g in enumerate(gains))


def evaluate_retrieval(pipe: FAQPipeline | None = None, ks: tuple[int, ...] = DEFAULT_KS,
                       verbose: bool = True) -> dict:
    """Rank the golden set and report recall@k, precision@k, MRR and nDCG.

    Retrieval is scored, not just the final answer, so a fusion-weight change
    can be judged on the ranking rather than on which end-to-end cases it
    happened to flip. Passes `top_k=max(ks)` so the cut-offs share one ranking.
    """
    from .intent import classify

    owned = pipe is None
    pipe = pipe or FAQPipeline()
    kmax = max(ks)
    rows: list[RetrievalRow] = []
    try:
        for case in GOLDEN:
            d = classify(case.q)
            if d.is_refusal:
                if verbose:
                    print(f"SKIP  {case.q}  (refused at stage 5, never retrieved)")
                continue
            res = pipe.retriever.retrieve(d, top_k=kmax)
            hits = res.hits
            gains = [1 if is_relevant(case, h.chunk) else 0 for h in hits]
            ranks = [i for i, g in enumerate(gains, 1) if g]
            ideal = _dcg([1] * min(len(ranks), kmax)) if ranks else 0.0
            rows.append(RetrievalRow(
                q=case.q, expect_url=case.expect_url, ranks=ranks,
                n_hits=len(ranks), n_returned=len(hits),
                top1_relevant=bool(gains and gains[0]),
                ndcg=(_dcg(gains[:kmax]) / ideal) if ideal else 0.0,
            ))
            if verbose:
                mark = "ok  " if rows[-1].top1_relevant else "MISS"
                rr = f"{1/ranks[0]:.2f}" if ranks else "-"
                print(f"{mark}  rank={ranks[0] if ranks else '-':>2}  mrr={rr}  "
                      f"ndcg@{kmax}={rows[-1].ndcg:.2f}  {case.q}")
    finally:
        if owned:
            pipe.close()

    n = len(rows) or 1
    metrics: dict = {
        "n_queries": len(rows),
        "ks": list(ks),
        "mrr": round(sum(1 / r.ranks[0] for r in rows if r.ranks) / n, 4),
        "top1_accuracy": round(sum(r.top1_relevant for r in rows) / n, 4),
        "ndcg": round(sum(r.ndcg for r in rows) / n, 4),
        "mean_returned": round(sum(r.n_returned for r in rows) / n, 2),
    }
    for k in ks:
        hit = sum(any(rk <= k for rk in r.ranks) for r in rows)
        relevant_in_k = sum(min(r.n_hits, k) for r in rows)
        returned_in_k = sum(min(r.n_returned, k) for r in rows)
        metrics[f"recall@{k}"] = round(hit / n, 4)
        metrics[f"precision@{k}"] = round(relevant_in_k / returned_in_k, 4) if returned_in_k else 0.0

    if verbose:
        print()
        for r in rows:
            if not r.top1_relevant:
                print(f"  ! rank-1 miss: {r.q}  (best relevant at {r.ranks or 'none'})")
        print(f"  n={metrics['n_queries']}  top1={metrics['top1_accuracy']:.0%}  "
              f"MRR={metrics['mrr']:.3f}  nDCG={metrics['ndcg']:.3f}")
        print("  " + "  ".join(f"R@{k}={metrics[f'recall@{k}']:.0%}" for k in ks))
        print("  " + "  ".join(f"P@{k}={metrics[f'precision@{k}']:.0%}" for k in ks))
    return {"metrics": metrics, "rows": [r.to_dict() for r in rows]}
