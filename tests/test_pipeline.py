"""Standard-library test suite (no pytest required).

    python -m unittest discover -s tests -v

The suite covers the properties the brief makes promises about, not just
happy-path plumbing: PII is never echoed, advice and performance questions are
refused, every citation is allow-listed, and every answer is within the
sentence cap. It also runs the retrieval/guardrail invariants over every
golden query, so a regression in scoring cannot silently ship.
"""
from __future__ import annotations

import ast
import hashlib
import html
import json
import os
import re
import socket
import sys
import time
import unittest
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

# The suite must be hermetic against the local `.env`, which this developer's
# machine points at an HF model that needs `sentence-transformers` (torch).
# Pin the guaranteed-available hashed vectors BEFORE mf_faq is imported: the
# `.env` loader never overrides a real environment value. The HF path is
# exercised explicitly in TestEmbeddingFactory instead of implicitly here.
os.environ.setdefault("MF_FAQ_EMBED_MODEL", "hashed")

import samples_gen                                                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))          # for the top-level samples_gen deliverable

from mf_faq.chunking import (PROCEDURES, SECTION_MAX_TOKENS, build_fact_chunks,  # noqa: E402
                             build_procedure_chunks, _fmt_exit_load, _fmt_lockin,
                             tokenize)
from mf_faq import ingest as ingest_mod                              # noqa: E402
from mf_faq import EXAMPLE_QUESTIONS, WELCOME                        # noqa: E402
from mf_faq.config import (ANSWER_PREPARED_PREFIX, DISCLAIMER, DISCLAIMER_LONG,  # noqa: E402
                           LAST_UPDATED_PREFIX, MAX_SENTENCES, PROJECT_ROOT)
from mf_faq.eval import GOLDEN, POLICY                               # noqa: E402
from mf_faq.guardrails import REFUSAL_BODIES, Answer, enforce        # noqa: E402
from mf_faq.indexing import (HashedVectoriser, all_chunks, cosine,  # noqa: E402
                             all_vectors, index_stats, load_vectoriser)
from mf_faq.intent import classify, scrub_pii                        # noqa: E402
from mf_faq.parsing import (html_to_text, normalise_facts, normalise_date,  # noqa: E402
                            split_sentences, visible_text)
from mf_faq.pipeline import FAQPipeline                              # noqa: E402
from mf_faq.synthesis import get_synthesiser                        # noqa: E402
from mf_faq.sources import ALLOWED_URLS, BRIEF_URLS, EDUCATION_LINKS, SOURCES, SCHEMES  # noqa: E402

INDEX = ROOT / "data" / "index" / "mf_faq.sqlite"
HAVE_INDEX = INDEX.exists()


def _doc(facts=None, text="", scheme_key="large_cap", fetched_at="2026-09-26T10:00:00+00:00"):
    """A minimal Document for chunker unit tests."""
    from mf_faq.parsing import Document
    return Document(doc_id="test-doc", source_id="test-src", url="https://example.com/x",
                    title="t", publisher="p", doc_type="dt", scheme_key=scheme_key,
                    topics=(), quotable=True, verified_at_build=True,
                    text=text, facts=facts or {}, fetched_at=fetched_at,
                    content_sha256="")


# ------------------------------------------------------------------ stage 1

class TestIngestion(unittest.TestCase):
    """The fetch layer must degrade, never raise -- and must honour offline."""

    def test_cache_hit_reports_when_the_bytes_were_fetched_not_when_they_were_read(self):
        """Regression: `now = datetime.now()` was computed *before* the cache
        branch, so every cache hit rewrote the provenance record with the
        current time. A rebuild that did not refetch -- and an offline build in
        particular -- reported a week-old cache as fetched seconds ago. The
        manifest is the record of what the corpus is based on, so restating the
        age of the data is the one thing it must never do."""
        src = self._src(url="https://example.invalid/probe-stamp")
        path = ingest_mod._cache_path(src.url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"<html>cached</html>")
        self.addCleanup(path.unlink)

        # Backdate the file so "now" and "actually fetched" cannot coincide.
        old = time.time() - 5 * 86400
        os.utime(path, (old, old))

        r = ingest_mod.fetch(src)
        self.assertTrue(r.from_cache)
        stamp = datetime.fromisoformat(r.fetched_at)
        age_days = (datetime.now(timezone.utc) - stamp).days
        self.assertGreaterEqual(age_days, 4,
                                f"cache hit restated freshness: {r.fetched_at}")
        self.assertAlmostEqual(stamp.timestamp(), old, delta=60)

    def test_network_fetch_still_stamps_the_real_time(self):
        """The counterpart: a genuine fetch must keep the wall clock."""
        src = self._src(url="https://example.invalid/probe-live")
        real = urllib.request.urlopen

        class _Resp:
            status = 200

            def read(self):
                return b"<html>live</html>"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        urllib.request.urlopen = lambda *a, **k: _Resp()
        try:
            r = ingest_mod.fetch(src)
        finally:
            urllib.request.urlopen = real
            ingest_mod._cache_path(src.url).unlink(missing_ok=True)
        self.assertFalse(r.from_cache)
        stamp = datetime.fromisoformat(r.fetched_at)
        self.assertLess((datetime.now(timezone.utc) - stamp).total_seconds(), 300)

    def _src(self, **over):
        base = {**SOURCES[0].__dict__, "source_id": "probe"}
        return type(SOURCES[0])(**{**base, **over})

    def test_offline_never_opens_a_socket(self):
        """Regression: `and not CONFIG.offline` on the cache branch made
        offline+cached fall through to urlopen, so MF_FAQ_OFFLINE=1 still
        fetched.  The flag has to mean 'no network' unconditionally."""
        src = self._src(url="https://example.invalid/probe-offline")
        path = ingest_mod._cache_path(src.url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"<html>cached</html>")
        self.addCleanup(path.unlink)

        opened = []
        real = urllib.request.urlopen
        urllib.request.urlopen = lambda *a, **k: opened.append(1)
        prev = ingest_mod.CONFIG.offline
        ingest_mod.CONFIG.offline = True
        try:
            r = ingest_mod.fetch(src)
        finally:
            urllib.request.urlopen = real
            ingest_mod.CONFIG.offline = prev

        self.assertTrue(r.ok)
        self.assertTrue(r.from_cache)
        self.assertEqual(opened, [], "offline mode opened a socket")
        self.assertEqual(r.sha256,
                         hashlib.sha256(b"<html>cached</html>").hexdigest())

    def test_offline_uncached_reports_rather_than_raises(self):
        src = self._src(url="https://example.invalid/probe-uncached")
        prev = ingest_mod.CONFIG.offline
        ingest_mod.CONFIG.offline = True
        try:
            r = ingest_mod.fetch(src)
        finally:
            ingest_mod.CONFIG.offline = prev
        self.assertFalse(r.ok)
        self.assertEqual(r.error, "offline and not cached")
        self.assertEqual(r.status, 0)

    def test_force_still_cannot_bypass_offline(self):
        """`force` means 'ignore the cache', not 'ignore offline'."""
        src = self._src(url="https://example.invalid/probe-force")
        opened = []
        real = urllib.request.urlopen
        urllib.request.urlopen = lambda *a, **k: opened.append(1)
        prev = ingest_mod.CONFIG.offline
        ingest_mod.CONFIG.offline = True
        try:
            ingest_mod.fetch(src, force=True)
        finally:
            urllib.request.urlopen = real
            ingest_mod.CONFIG.offline = prev
        self.assertEqual(opened, [], "force=True escaped offline mode")

    def test_malformed_url_is_a_result_not_an_exception(self):
        """Regression: `Request(...)` used to be built *outside* the try, so a
        URL urllib rejects in Request.__init__ (bad bracketed host) escaped
        fetch() and aborted the whole build."""
        for url in ("https://[bad-ipv6]/x", "htp://typo", "not a url at all"):
            with self.subTest(url=url):
                r = ingest_mod.fetch(self._src(url=url), timeout=5)
                self.assertFalse(r.ok, url)
                self.assertTrue(r.error, url)
                self.assertEqual(r.status, 0, url)

    def test_cache_is_keyed_on_url_not_source_id(self):
        a = ingest_mod._cache_path("https://example.com/one")
        b = ingest_mod._cache_path("https://example.com/two")
        self.assertNotEqual(a, b)
        self.assertEqual(a.suffix, ".html")
        self.assertEqual(len(a.stem.rsplit(".", 1)[-1]), 16)

    def test_read_cache_missing_file_is_empty_string(self):
        r = ingest_mod.FetchResult(
            source_id="x", url="u", status=200, ok=True, bytes=1, sha256="s",
            fetched_at="t", cache_path=str(ROOT / "data" / "raw" / "nope.html"),
            from_cache=True)
        self.assertEqual(ingest_mod.read_cache(r), "")


# ------------------------------------------------------------------ stage 2

class TestParsing(unittest.TestCase):
    #: The exact contract from implementation.md Phase 3 item 5.
    SPEC_KEYS = {
        "scheme_name", "fund_name", "scheme_code", "isin", "amc", "category",
        "sub_category", "plan_type", "scheme_type", "launch_date", "fund_manager",
        "registrar_agent", "expense_ratio_pct", "base_expense_ratio_pct",
        "exit_load", "stamp_duty", "direct_scheme_code", "min_sip_amount",
        "min_lumpsum_amount", "min_additional_amount", "max_sip_amount",
        "min_withdrawal_amount", "sip_multiplier", "sip_allowed",
        "lumpsum_allowed", "lock_in_days", "benchmark", "nav", "nav_date",
        "closed_scheme", "aum",
    }

    def test_whitelist_holds_on_every_payload_shape(self):
        """FR-4 is a whitelist, so the key set must be identical no matter what
        arrives -- empty, adversarial, or all-None."""
        for name, payload in {
            "empty": {},
            "adversarial": {"returns_1y": 18.2, "return_stats": {"a": 1},
                            "simple_return": 2, "sip_return": 3, "cagr": 4,
                            "growth": 5, "ytd_return": 6, "one_year_return": 7,
                            "nav_absolute_return": 8, "performance": 9,
                            "expense_ratio": 1.03},
            "all-none": {k: None for k in self.SPEC_KEYS},
        }.items():
            with self.subTest(payload=name):
                self.assertEqual(set(normalise_facts(payload)), self.SPEC_KEYS)

    def test_no_performance_value_can_escape_the_whitelist(self):
        f = normalise_facts({"returns_1y": 18.2, "cagr": 22.1, "growth": 9,
                             "return_stats": {"x": 1}, "expense_ratio": 1.03})
        blob = json.dumps(f).lower()
        for token in ("cagr", "growth", "ytd", "return_stats", "18.2", "22.1"):
            self.assertNotIn(token, blob, token)

    def test_normalise_facts_is_total(self):
        """It must never raise: a crash here aborts the build at the one place
        FR-4 is enforced.  Regression: `.strip()` on a non-string exit_load
        raised AttributeError."""
        for payload in ({"exit_load": 12345}, {"exit_load": {"a": 1}},
                        {"lock_in": "junk"}, {"lock_in": {"years": "abc"}},
                        {"benchmark": {"n": 1}}, {"fund_manager": ["a", "b"]},
                        {"sip_allowed": object()}):
            with self.subTest(payload=payload):
                f = normalise_facts(payload)  # must not raise
                self.assertEqual(set(f), self.SPEC_KEYS)

    def test_fund_manager_prefers_the_rendered_array_over_the_legacy_field(self):
        """The payload carries two fund-manager fields and they disagree on all
        five collected pages. `fund_manager` is a legacy single-name field still
        populated with a manager who no longer runs the scheme (Prashant Jain for
        Large Cap *and* Flexi Cap, Vinay Kulkarni for ELSS, Srinivas Rao Ravuri
        for Balanced Advantage). `fund_manager_details` is what the page renders,
        and its first entry matches the page's own "Current Fund Manager" prose.

        Reading the legacy field published a stale name in a verbatim fact chunk
        behind a *correct* citation URL -- the one wrong answer the guardrails
        cannot catch, because G2 passes and nothing downstream can tell a true
        template from a false one.
        """
        # both fields present: the rendered array wins
        self.assertEqual(
            normalise_facts({"fund_manager": "Prashant Jain",
                             "fund_manager_details": [
                                 {"person_name": "Dhruv Muchhal"},
                                 {"person_name": "Amit Ganatra"}]})["fund_manager"],
            ["Dhruv Muchhal", "Amit Ganatra"])
        # only the array, in both shapes the payload has used
        self.assertEqual(
            normalise_facts({"fund_manager_details": [{"person_name": "Rahul Baijal"}]})
            ["fund_manager"], ["Rahul Baijal"])
        self.assertEqual(
            normalise_facts({"fund_manager_details": ["Rahul Baijal"]})["fund_manager"],
            ["Rahul Baijal"])
        # a manager listed twice is a data artefact, not a second co-manager
        self.assertEqual(
            normalise_facts({"fund_manager_details": [
                {"person_name": "Dhruv Muchhal"}, {"person_name": "Dhruv Muchhal"}]})
            ["fund_manager"], ["Dhruv Muchhal"])
        # array present but nameless -> fall back, do not publish an empty manager
        self.assertEqual(
            normalise_facts({"fund_manager": "Prashant Jain",
                             "fund_manager_details": [{"education": "B.Com"}]})
            ["fund_manager"], ["Prashant Jain"])
        # array absent entirely -> the legacy field is all there is
        self.assertEqual(normalise_facts({"fund_manager": "Prashant Jain"})
                         ["fund_manager"], ["Prashant Jain"])
        # and the whole corpus must carry no retired manager any more
        import json
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / "data" / "chunks.jsonl"
        retired = {"Prashant Jain", "Vinay Kulkarni", "Srinivas Rao Ravuri"}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            chunk = json.loads(line)
            if chunk.get("topic") == "scheme_meta" and "fund manager" in (chunk.get("text") or "").lower():
                for name in retired:
                    self.assertNotIn(name, chunk["text"],
                                     f"{chunk['chunk_id']} still names {name}")

    def test_every_co_manager_is_kept_and_the_sentence_agrees_in_number(self):
        """All five schemes are co-managed -- two managers each except HDFC
        Balanced Advantage, which carries six. Taking only the first entry (on
        the theory that the rest was the AMC's manager carousel) silently
        dropped real co-managers, and for HDFC Small Cap it dropped
        Chirag Setalvad, who has managed it since 2014.

        The whole array is this scheme's team: every entry carries a
        `scheme_code` equal to the page's own and lists this scheme inside its
        `funds_managed`. So the fact keeps every name, and the sentence has to
        switch to the plural or it reads as broken English.
        """
        import json
        from pathlib import Path
        from mf_faq.chunking import _fmt_managers
        self.assertEqual(_fmt_managers(["A"]), "A")
        self.assertEqual(_fmt_managers(["A", "B"]), "A and B")
        self.assertEqual(_fmt_managers(["A", "B", "C"]), "A, B and C")

        path = Path(__file__).resolve().parents[1] / "data" / "chunks.jsonl"
        manager_chunks = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            chunk = json.loads(line)
            text = chunk.get("text") or ""
            if chunk.get("kind") == "fact" and text.startswith(("The fund manager of",
                                                                "The fund managers of")):
                manager_chunks.append(text)
        self.assertEqual(len(manager_chunks), 5, manager_chunks)
        for text in manager_chunks:
            body = text.rstrip(".")
            names = body.split(" are ")[-1] if " are " in body else body.split(" is ")[-1]
            n = len(names.split(" and ")) + names.count(",")
            if n == 1:
                self.assertTrue(body.endswith((" is",)) or " is " in body,
                                f"singular list must use 'is': {text}")
                self.assertTrue(text.startswith("The fund manager of"), text)
            else:
                self.assertTrue(text.startswith("The fund managers of"), text)
                self.assertIn(" are ", text)
        # the two the question was actually about, and the longest roster
        self.assertTrue(any("Dhruv Muchhal and Amit Ganatra" in t for t in manager_chunks),
                        manager_chunks)
        self.assertTrue(any(t.startswith("The fund managers of") and "Ihab Dalwai" in t
                            for t in manager_chunks), manager_chunks)
        # Small Cap's long-tenured manager must survive alongside the primary
        self.assertTrue(any("Dhruv Muchhal and Chirag Setalvad" in t for t in manager_chunks),
                        manager_chunks)

    def test_aum_comes_from_the_payload_not_the_page_prose(self):
        """Every scheme page carries the sentence "The fund currently has an
        Asset Under Management(AUM) of Rs 9,86,237 Cr" -- the identical figure
        on all five pages, because it is platform-level chrome rather than the
        scheme's own size. A fact chunk built from page text would therefore
        publish the same wrong AUM five times. Pin that the payload field wins
        and that the corpus figures actually differ per scheme."""
        f = normalise_facts({"aum": 39933.3663})
        self.assertEqual(f["aum"], 39933.3663)
        self.assertNotIn("aum", {k for k in f if k == "returns_1y"})

        docs = [json.loads(l) for l in
                (PROJECT_ROOT / "data" / "corpus.jsonl").read_text(
                    encoding="utf-8").splitlines() if l.strip()]
        aums = {d["scheme_key"]: d["facts"].get("aum")
                for d in docs if d.get("facts", {}).get("aum") is not None}
        self.assertEqual(len(aums), 5, f"expected an AUM per scheme, got {aums}")
        # Distinct per scheme: a shared value means the chrome number leaked in.
        self.assertEqual(len(set(aums.values())), 5, f"AUMs are not distinct: {aums}")

    def test_aum_is_not_treated_as_performance_data(self):
        """FR-4 drops returns/CAGR/growth. AUM is a published scale fact and is
        admissible -- but it must not drag the performance nets in with it, so
        pin both directions: it is whitelisted, and no return vocabulary appears."""
        f = normalise_facts({"aum": 39933.3663, "cagr": 22.1, "returns_1y": 18.2})
        self.assertIn("aum", f)
        self.assertNotIn("cagr", f)
        self.assertNotIn("returns_1y", f)

    def test_bool_coercion_does_not_invert_a_string_false(self):
        """Regression: `bool("false")` is True, which would have published
        sip_allowed=True for a scheme that does not allow SIP."""
        self.assertIs(normalise_facts({"sip_allowed": "false"})["sip_allowed"], False)
        self.assertIs(normalise_facts({"closed_scheme": "false"})["closed_scheme"], False)
        self.assertIs(normalise_facts({"sip_allowed": "true"})["sip_allowed"], True)
        self.assertIs(normalise_facts({"sip_allowed": True})["sip_allowed"], True)
        self.assertIs(normalise_facts({"sip_allowed": 0})["sip_allowed"], False)
        self.assertIs(normalise_facts({"sip_allowed": None})["sip_allowed"], False)

    def test_exit_load_nil_normalisation(self):
        for raw in ("nil", "NONE", "", None, "  Nil  ", "n/a", "-"):
            self.assertEqual(normalise_facts({"exit_load": raw})["exit_load"], "Nil",
                             raw)
        long_form = "Exit load of 1% if redeemed within 1 year"
        self.assertEqual(normalise_facts({"exit_load": long_form})["exit_load"],
                         long_form)

    def test_growth_is_only_ever_a_plan_designation(self):
        """"Direct - Growth" / scheme_type "Growth" are identity facts we must
        keep.  What must never appear is a growth *figure*."""
        f = normalise_facts({"scheme_name": "HDFC Large Cap Fund Direct Growth",
                             "scheme_type": "Growth", "benchmark_name":
                             "NIFTY 100 Total Return Index"})
        self.assertEqual(f["scheme_type"], "Growth")
        self.assertEqual(f["benchmark"], "NIFTY 100 Total Return Index")
        self.assertNotRegex(json.dumps(f), r"[+-]\d+\.\d+%")

    def test_lock_in_arithmetic_and_garbage(self):
        for raw, want in [({"years": 3}, 1095), ({"months": 6}, 180),
                          ({"years": 1, "months": 2, "days": 3}, 365 + 60 + 3),
                          ({"years": None, "months": None, "days": None}, None),
                          ({}, None), (None, None), ("junk", None)]:
            self.assertEqual(normalise_facts({"lock_in": raw})["lock_in_days"],
                             want, raw)

    def test_field_mapping_is_explicit(self):
        m = normalise_facts({
            "min_sip_investment": 100, "min_investment_amount": 5000,
            "mini_additional_investment": 500, "max_sip_investment": 50000,
            "min_withdrawal": 1000, "benchmark_name": "B", "benchmark": "B2"})
        self.assertEqual(m["min_sip_amount"], 100)
        self.assertEqual(m["min_lumpsum_amount"], 5000)
        self.assertEqual(m["min_additional_amount"], 500)
        self.assertEqual(m["max_sip_amount"], 50000)
        self.assertEqual(m["min_withdrawal_amount"], 1000)
        self.assertEqual(m["benchmark"], "B", "benchmark_name must win")

    def test_split_sentences_does_not_split_on_decimals_or_abbreviations(self):
        self.assertEqual(len(split_sentences(
            "The ratio is 1.03%. This is e.g. the ongoing charge. Next one.")), 3)
        self.assertEqual(len(split_sentences("NAV was ₹1,189.08. Then it fell.")), 2)
        self.assertEqual(len(split_sentences("Dr. Rao manages it. He joined.")), 2)
        self.assertIn("e.g.", " ".join(split_sentences("A is e.g. B. C.")))

    def test_html_to_text_never_raises_on_malformed_markup(self):
        for bad in ("<div><p>unclosed", "<<<>>>", "", "<p>a</p><div>b",
                    "&#xZZ; &#;", "\x00\x01", "<script>x</script>"):
            with self.subTest(html=bad[:12]):
                self.assertIsInstance(html_to_text(bad), str)

    def test_visible_text_strips_scripts_and_nav(self):
        html = ("<html><head><script>var x='SECRET';</script></head><body>"
                "<h1>Expense Ratio</h1><p>The expense ratio is 1.03%.</p>"
                "<script>tracker()</script></body></html>")
        text = visible_text(html)
        self.assertIn("expense ratio is 1.03%", text)
        self.assertNotIn("SECRET", text)
        self.assertNotIn("tracker", text)

    def test_split_sentences_keeps_abbreviations(self):
        s = split_sentences("The exit load is 1% e.g. within 1 year. NAV was 10.")
        self.assertEqual(len(s), 2)

    def test_normalise_date_formats(self):
        self.assertEqual(normalise_date("25-Sep-2026"), "2026-09-25")
        self.assertEqual(normalise_date("2026-09-25"), "2026-09-25")
        self.assertEqual(normalise_date(""), "")

    def test_facts_schema_drops_performance_fields(self):
        facts = normalise_facts({
            "scheme_name": "X", "expense_ratio": 1.03, "nav": 10.5,
            "nav_date": "25-Sep-2026", "exit_load": "Exit load of 1% if redeemed within 1 year",
            "lock_in": {"years": 3, "months": 0, "days": 0},
            "simple_return": [1, 2, 3], "return_stats": {"a": 1}, "sip_return": {},
            # the page carries return figures; none may survive into the corpus
        })
        self.assertEqual(facts["expense_ratio_pct"], 1.03)
        self.assertEqual(facts["lock_in_days"], 1095)
        for banned in ("simple_return", "return_stats", "sip_return", "1_year_return"):
            self.assertNotIn(banned, facts)
        self.assertNotIn("return", " ".join(facts.keys()))

    def test_singular_return_noun_is_refused_as_performance(self):
        """FR-4 regression.

        The performance rules were plural-only, to keep "return the money" a
        redemption question.  That left singular *noun* uses unanswered:
        "what did X return last year?" was answered with the benchmark
        ("NIFTY 100 Total Return Index") and "did X give a good return?" was
        answered with lock-in.  Both are performance questions and must refuse.
        """
        for q in ("What did HDFC Large Cap Fund return last year?",
                  "What did HDFC Large Cap Fund return in 2024?",
                  "What is the return on HDFC Flexi Cap?",
                  "Did HDFC ELSS give a good return?",
                  "What is my return on investment?",
                  "what is the average return?",
                  "What did the fund return over 3 years?"):
            with self.subTest(q=q):
                self.assertEqual(classify(q).kind, "performance_refusal", q)

    def test_return_as_a_redemption_verb_is_still_answered(self):
        """The other half of the distinction: 'return' as a verb with a direct
        object is how you redeem, and must not be refused as a performance
        question or the assistant would refuse a legitimate procedure."""
        for q in ("How do I return the money?",
                  "how do I return my investment",
                  "can I return the units",
                  "What is the exit load?"):
            with self.subTest(q=q):
                self.assertEqual(classify(q).kind, "answer", q)


# ------------------------------------------------------------------ stage 3

class TestChunkingHelpers(unittest.TestCase):
    def test_lockin_formatting(self):
        self.assertEqual(_fmt_lockin(1095), "3 years")
        self.assertEqual(_fmt_lockin(365), "1 year")
        self.assertIn("no lock-in", _fmt_lockin(None))

    def test_exit_load_normalisation(self):
        self.assertEqual(
            _fmt_exit_load("Exit load of 1% if redeemed within 1 year"),
            "1% if redeemed within 1 year")
        self.assertEqual(
            _fmt_exit_load("Exit Load for units in excess of 15% of the investment,"
                           "1% will be charged for redemption within 1 year."),
            "1% for units in excess of 15% of the investment, if redeemed within 1 year")

    def test_tokenize_is_lowercased(self):
        self.assertEqual(tokenize("Expense Ratio 1.03%"), ["expense", "ratio", "1.03%"])
        self.assertEqual(tokenize("SIP ₹500"), ["sip", "₹500"])

    def test_exit_load_preserves_rate_threshold_and_window_verbatim(self):
        """Re-ordering the badly-worded source pattern must not alter a number."""
        out = _fmt_exit_load("Exit Load for units in excess of 15% of the investment,"
                             "1% will be charged for redemption within 1 year.")
        for figure in ("1%", "15%", "1 year"):
            self.assertIn(figure, out)

    def test_fact_chunks_carry_meta_field(self):
        """Spec pitfall 1: `meta['field']` is load-bearing for phase 8. Omitting
        it made "who manages X?" silently return the RTA."""
        doc = _doc(facts={"expense_ratio_pct": 1.03, "sip_allowed": True})
        chunks = build_fact_chunks(doc)
        for c in chunks:
            self.assertIn("field", c.meta, c.chunk_id)
            self.assertIn("raw_value", c.meta, c.chunk_id)
            self.assertTrue(c.chunk_id.endswith("::fact::" + c.meta["field"]), c.chunk_id)

    def test_procedures_carry_no_source_date(self):
        """Procedures are authored text, not fetched data. A date on them would
        be a fabricated freshness claim -- this is the `date.today()` bug."""
        for c in build_procedure_chunks():
            self.assertEqual(c.last_updated, "",
                             f"{c.topic} must not claim a source date")
            self.assertTrue(c.url.startswith("https://"), c.chunk_id)

    def test_last_updated_is_source_derived_not_a_clock_read(self):
        """nav_date wins, then fetched_at, and there is no `date.today()` escape
        hatch anywhere in chunking (checked via AST, so a docstring mentioning
        it does not count)."""
        doc = _doc(facts={"expense_ratio_pct": 1.03, "nav": 1189.079,
                          "nav_date": "25-Sep-2026"},
                   fetched_at="2026-09-26T10:00:00+00:00")
        got = {c.meta["field"]: c.last_updated for c in build_fact_chunks(doc)}
        self.assertEqual(got["nav"], "2026-09-25", "nav_date must win")
        # nav_date is per-document, so it applies to every fact chunk of that doc
        self.assertEqual(got["expense_ratio_pct"], "2026-09-25")
        # with no nav_date, fall back to the fetch timestamp - never the clock
        plain = _doc(facts={"expense_ratio_pct": 1.03},
                     fetched_at="2026-09-26T10:00:00+00:00")
        self.assertEqual(
            [c.last_updated for c in build_fact_chunks(plain)], ["2026-09-26"])

    def test_chunking_module_has_no_clock_read(self):
        src = (ROOT / "src" / "mf_faq" / "chunking.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                self.assertNotIn(name, {"today", "now", "utcnow"},
                                 "chunking must never read the clock")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = getattr(node, "module", "") or ""
                self.assertNotIn("datetime", f"{mod}", "no clock module needed")


class TestProvenanceWording(unittest.TestCase):
    """A refusal or an authored procedure must not claim source freshness."""

    def _ask(self, q):
        if not HAVE_INDEX:
            self.skipTest("index not built")
        pipe = FAQPipeline()
        try:
            return pipe.ask(q)
        finally:
            pipe.close()

    def test_refusal_is_labelled_prepared_not_source_freshness(self):
        a = self._ask("Should I invest in HDFC Large Cap Fund?")
        self.assertTrue(a.is_refusal)
        self.assertFalse(a.freshness_from_source)
        self.assertIn("Answer prepared on:", a.render())
        self.assertNotIn("Last updated from sources", a.render())

    def test_source_grounded_answer_keeps_the_source_freshness_claim(self):
        a = self._ask("What is the NAV of HDFC Flexi Cap Fund?")
        self.assertEqual(a.kind, "answer")
        self.assertTrue(a.freshness_from_source)
        self.assertIn("Last updated from sources:", a.render())

    def test_procedure_answer_is_labelled_prepared_not_source_freshness(self):
        a = self._ask("How do I download my capital gains statement?")
        self.assertEqual(a.kind, "answer")
        self.assertFalse(a.freshness_from_source,
                         "a procedure has no fetched data to be fresh about")
        self.assertIn("Answer prepared on:", a.render())


# ------------------------------------------------------------------ stage 4

class TestIndexing(unittest.TestCase):
    def test_cosine_self_similarity_is_one(self):
        v = HashedVectoriser(dim=64).fit(["expense ratio of a large cap fund",
                                          "minimum sip amount"])
        docs = v.transform(["expense ratio of a large cap fund"])
        self.assertAlmostEqual(cosine(docs[0], docs[0]), 1.0, places=5)

    def test_similar_text_scores_above_unrelated(self):
        v = HashedVectoriser(dim=256).fit(
            ["minimum sip amount is 500 rupees", "benchmark is nifty 100 tri index"])
        a = v.transform_one("minimum sip amount 500")
        b = v.transform_one("benchmark nifty 100 index")
        self.assertGreater(cosine(a, b), 0.0)

    def test_signed_hashing_keeps_norm_unit(self):
        v = HashedVectoriser(dim=128).fit(["hello world fund"])
        x = v.transform_one("hello world fund")
        norm = sum(i * i for i in x) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=5)


# ------------------------------------------------------------------ embeddings (D-1)

class TestEmbeddingFactory(unittest.TestCase):
    """`get_vectoriser` is the only place that knows which dense index is live.

    Build must be explicit: an HF model that is requested but not installed
    FAILS with an actionable message rather than silently building a hashed
    index, because the two indexes are not interchangeable and a quiet swap
    would ship a different retrieval behaviour than the operator asked for.
    """

    def test_hashed_is_the_fallback(self):
        from mf_faq.config import RetrievalConfig
        from mf_faq.indexing import HashedVectoriser, get_vectoriser
        # The default is environment-driven, so "hashed" is enforced per value,
        # not assumed as the machine default.
        for value in ("hashed", "default", "none", ""):
            cfg = RetrievalConfig()
            cfg.embed_model = value
            self.assertIsInstance(get_vectoriser(cfg), HashedVectoriser, value)
        self.assertIsInstance(get_vectoriser(None), HashedVectoriser)

    def test_hf_without_the_package_raises_actionable_import_error(self):
        import sys
        import unittest.mock as mock
        from mf_faq.config import RetrievalConfig
        from mf_faq.indexing import get_vectoriser
        cfg = RetrievalConfig()
        cfg.embed_model = "sentence-transformers/all-MiniLM-L6-v2"
        with mock.patch.dict(sys.modules, {"sentence_transformers": None}):
            with self.assertRaises(ImportError) as ctx:
                get_vectoriser(cfg)
        msg = str(ctx.exception)
        self.assertIn("all-MiniLM-L6-v2", msg)
        self.assertIn("hashed", msg)
        self.assertIn("pip install sentence-transformers", msg)

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("sentence_transformers"),
        "sentence-transformers not installed")
    def test_hf_model_produces_normalised_384_dim_vectors(self):
        from mf_faq.config import RetrievalConfig
        from mf_faq.indexing import SentenceTransformerVectoriser, cosine
        cfg = RetrievalConfig()
        cfg.embed_model = "sentence-transformers/all-MiniLM-L6-v2"
        v = SentenceTransformerVectoriser(cfg.embed_model)
        self.assertEqual(v.dim, 384)
        x = v.transform_one("minimum sip amount for HDFC Small Cap Fund")
        self.assertEqual(len(x), 384)
        norm = sum(i * i for i in x) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=5)
        self.assertGreater(cosine(x, v.transform_one("minimum sip of a fund")), 0.3)

    def test_embed_model_is_recorded_in_index_meta(self):
        from mf_faq.indexing import index_stats, load_vectoriser
        stats = index_stats()
        self.assertIn("embed_model", stats, "build must record which model was used")
        self.assertEqual(load_vectoriser().dim, stats["dim"],
                         "the query plane must agree with the recorded model")

    def test_failed_build_leaves_the_existing_index_intact(self):
        # Regression: a build that requests an uninstalled HF model used to
        # unlink the old index before constructing the vectoriser, leaving an
        # empty schema behind and a working system silently broken by a failed
        # rebuild. The vectoriser (the only step that can fail) must complete
        # first.
        import sys
        import tempfile
        import unittest.mock as mock
        from pathlib import Path
        from mf_faq.config import RetrievalConfig
        from mf_faq.indexing import all_chunks, build_index
        chunks = all_chunks()
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "idx.sqlite"
            good = RetrievalConfig()
            good.embed_model = "hashed"
            build_index(chunks, db, good, verbose=False)
            before = db.read_bytes()
            bad = RetrievalConfig()
            bad.embed_model = "sentence-transformers/all-MiniLM-L6-v2"
            with mock.patch.dict(sys.modules, {"sentence_transformers": None}):
                with self.assertRaises(ImportError):
                    build_index(chunks, db, bad, verbose=False)
            self.assertEqual(db.read_bytes(), before,
                             "a failed build must not touch the existing index")
            # and the surviving index is still loadable
            from mf_faq.indexing import load_vectoriser
            self.assertIsInstance(load_vectoriser(db), HashedVectoriser)


class TestGroqProvider(unittest.TestCase):
    """Groq is the project's primary LLM provider (OpenAI-compatible REST).

    Tests pin routing and the request shape without touching the network: the
    `urlopen` socket is swapped for a stub that returns a canned completion.
    """

    def _stub_urlopen(self, resp_body: bytes) -> list[dict]:
        import urllib.request
        seen: list[dict] = []
        class _Resp:
            def read(self):
                return resp_body
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
        def fake(req, timeout=45):
            seen.append({"url": req.full_url, "data": req.data,
                         "headers": {k: v for k, v in req.header_items()}})
            return _Resp()
        real = urllib.request.urlopen
        urllib.request.urlopen = fake
        self.addCleanup(setattr, urllib.request, "urlopen", real)
        return seen

    def test_auto_prefers_groq(self):
        import os
        from mf_faq import llm as llm_mod
        saved = {k: os.environ.get(k) for k in ("GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
        os.environ["GROQ_API_KEY"] = "gsk-x"
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            self.assertEqual(llm_mod.provider(), "groq")
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_groq_completes_against_the_groq_endpoint(self):
        import os
        import json
        from mf_faq import llm as llm_mod
        from mf_faq.config import CONFIG
        seen = self._stub_urlopen(
            json.dumps({"choices": [{"message": {"content": "1.03%."}}]}).encode())
        saved = {k: os.environ.get(k) for k in ("GROQ_API_KEY", "GROQ_BASE_URL")}
        os.environ["GROQ_API_KEY"] = "gsk-test"
        os.environ["GROQ_BASE_URL"] = "https://api.groq.com/openai/v1/chat/completions"
        try:
            out = llm_mod.complete("prompt", "system")
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.assertEqual(out, "1.03%.")
        self.assertEqual(len(seen), 1)
        req = seen[0]
        self.assertIn("api.groq.com", req["url"])
        self.assertEqual(req["headers"].get("Authorization"), "Bearer gsk-test")
        body = json.loads(req["data"])
        self.assertEqual(body["model"], CONFIG.groq_model)
        self.assertEqual(body["temperature"], 0)


# ------------------------------------------------------------------ stage 5

class TestPII(unittest.TestCase):
    def test_pan_is_stripped(self):
        clean, kinds = scrub_pii("My PAN is ABCDE1234F, what is the exit load?")
        self.assertNotIn("ABCDE1234F", clean)
        self.assertIn("pan", kinds)

    def test_aadhaar_email_phone_folio_stripped(self):
        for q, needle in [
            ("aadhaar 2345 6789 0123", "2345 6789 0123"),
            ("mail me at investor@example.com", "investor@example.com"),
            ("call 9876543210 now", "9876543210"),
            ("folio number is 12345678901", "12345678901"),
            ("my otp is 482913", "482913"),
        ]:
            clean, kinds = scrub_pii(q)
            self.assertNotIn(needle, clean, f"leaked in: {q}")
            self.assertTrue(kinds, f"no PII detected in: {q}")

    def test_normal_questions_are_untouched(self):
        q = "What is the expense ratio of HDFC Large Cap Fund?"
        clean, kinds = scrub_pii(q)
        self.assertEqual(clean, q)
        self.assertEqual(kinds, [])


class TestIntentGate(unittest.TestCase):
    def test_factual_queries_pass(self):
        for q in ["What is the expense ratio of HDFC Large Cap Fund?",
                  "minimum SIP for HDFC ELSS",
                  "What is the exit load of HDFC Small Cap Fund?",
                  "How do I download a capital gains statement?"]:
            self.assertEqual(classify(q).kind, "answer", q)

    def test_advice_is_refused(self):
        for q in ["Should I buy HDFC Small Cap Fund?",
                  "Which is better, HDFC Large Cap or HDFC Flexi Cap?",
                  "Is HDFC ELSS a good buy right now?",
                  "What is the best fund for me?",
                  "How should I allocate my portfolio?",
                  "Recommend a scheme for my child",
                  "Is now a good time to enter?"]:
            self.assertEqual(classify(q).kind, "advice_refusal", q)

    def test_performance_is_refused(self):
        for q in ["What are the 3 year returns of HDFC Large Cap Fund?",
                  "How did HDFC Small Cap Fund perform last year?",
                  "Which fund has higher CAGR?",
                  "Compare the returns of Large Cap and Flexi Cap",
                  "What is the 1 year performance?",
                  "How much did it grow in 5 years?"]:
            self.assertEqual(classify(q).kind, "performance_refusal", q)

    def test_pii_is_refused_before_routing(self):
        d = classify("My PAN is ABCDE1234F, what is the exit load?")
        self.assertEqual(d.kind, "pii_refusal")
        self.assertNotIn("ABCDE1234F", d.query)
        self.assertEqual(d.raw_query, "", "raw query must not be retained")

    def test_scheme_resolution_prefers_longest_alias(self):
        self.assertEqual(classify("expense ratio of HDFC Flexi Cap Fund").scheme_key,
                         "flexi_cap")
        self.assertEqual(classify("HDFC ELSS Tax Saver Fund lock in").scheme_key, "elss")
        self.assertIsNone(classify("What is a mutual fund?").scheme_key)

    def test_procedural_topics_are_exclusive(self):
        d = classify("Where do I download my tax statement?")
        self.assertEqual(d.primary_topic, "tax_doc")
        self.assertEqual(d.topics, ["tax_doc"])
        d2 = classify("How do I download a capital gains statement?")
        self.assertEqual(d2.primary_topic, "capital_gains")

    def test_sip_and_lumpsum_route_separately(self):
        self.assertEqual(classify("minimum SIP for HDFC Small Cap").primary_topic, "min_sip")
        self.assertEqual(classify("minimum lump sum for HDFC Small Cap").primary_topic,
                         "min_lumpsum")

    def test_field_hint_for_who_manages(self):
        self.assertIn("fund_manager", classify("Who manages HDFC Flexi Cap Fund?").field_hints)

    # --- regressions -------------------------------------------------------
    # Each of these leaked at some point during development: the gate let the
    # query through and retrieval answered it with an unrelated fact chunk
    # (a launch date, a benchmark, an expense ratio).  An answered-and-wrong
    # response is worse than a refusal, so each phrasing is pinned here.

    def test_subjective_risk_copula_is_refused(self):
        # Answered with "was launched on 01-Jan-2013" before the end-anchored
        # copula rule existed.
        for q in ["Is HDFC Small Cap Fund risky?",
                  "is HDFC Balanced Advantage safe?",
                  "is the HDFC ELSS fund volatile?"]:
            self.assertEqual(classify(q).kind, "advice_refusal", q)
        # "are the returns risky?" mentions returns, so the (earlier, by design)
        # performance gate wins. Either refusal is correct; what matters is that
        # it is not answered.
        self.assertTrue(classify("Are HDFC Flexi Cap Fund's returns risky?").is_refusal)

    def test_bare_plural_returns_is_a_performance_question(self):
        # "tell me about HDFC ELSS returns" was answered with the expense ratio.
        for q in ["tell me about HDFC ELSS returns",
                  "HDFC ELSS returns",
                  "show me the returns of HDFC Small Cap",
                  "how much did HDFC Large Cap return last year",
                  "what is the volatility of HDFC Large Cap"]:
            self.assertEqual(classify(q).kind, "performance_refusal", q)

    def test_which_fund_better_is_advice_not_performance(self):
        # "which fund" must not trip the performance gate on its own: the
        # useful refusal here is the no-recommendation one.
        d = classify("which fund is better for 5 years")
        self.assertEqual(d.kind, "advice_refusal")
        self.assertEqual(classify("which fund has higher CAGR").kind,
                         "performance_refusal")

    def test_plural_returns_rule_does_not_break_factual_queries(self):
        # The singular/plural split is load-bearing: "return the money" is a
        # redemption question, and no factual query in scope contains "returns".
        for q in ["What is the exit load of HDFC Large Cap Fund?",
                  "What is the expense ratio of HDFC Large Cap Fund?",
                  "minimum SIP for HDFC ELSS",
                  "What is the ISIN of HDFC Small Cap Fund?",
                  "How do I download a capital gains statement?"]:
            self.assertEqual(classify(q).kind, "answer", q)


# ------------------------------------------------------------------ stage 8

class TestGuardrails(unittest.TestCase):
    def _ans(self, text: str, citation: str = "", **kw) -> Answer:
        kw.setdefault("last_updated", "2026-09-25")
        return Answer(kind="answer", text=text, citation=citation, **kw)

    def test_non_allowlisted_citation_is_dropped(self):
        a = enforce(self._ans("The exit load is 1%.", "https://random-blog.example/x"), "answer")
        self.assertEqual(a.citation, "")
        self.assertIn("G2_citation_missing", a.guards)

    def test_allowlisted_citation_survives(self):
        a = enforce(self._ans("The exit load is 1%.", BRIEF_URLS[0]), "answer")
        self.assertEqual(a.citation, BRIEF_URLS[0])

    def test_sentence_cap_is_enforced(self):
        long = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
        a = enforce(self._ans(long, BRIEF_URLS[0]), "answer")
        self.assertLessEqual(len(split_sentences(a.text)), MAX_SENTENCES)
        self.assertIn("G3_sentence_capped", a.guards)

    def test_pii_in_output_becomes_refusal(self):
        a = enforce(self._ans("Your PAN ABCDE1234F was verified.", BRIEF_URLS[0]), "answer")
        self.assertEqual(a.kind, "pii_refusal")
        self.assertNotIn("ABCDE1234F", a.text)

    def test_advice_in_output_becomes_refusal(self):
        a = enforce(self._ans("You should buy this fund.", BRIEF_URLS[0]), "answer")
        self.assertEqual(a.kind, "advice_refusal")
        self.assertEqual(a.text, REFUSAL_BODIES["advice_refusal"])

    def test_performance_claim_becomes_refusal(self):
        a = enforce(self._ans("The fund delivered 14.2% CAGR over 3 years.", BRIEF_URLS[0]),
                    "answer")
        self.assertEqual(a.kind, "performance_refusal")

    def test_freshness_and_disclaimer_always_present(self):
        a = enforce(self._ans("The NAV was 10.", BRIEF_URLS[0], last_updated=""), "answer")
        self.assertTrue(a.last_updated)
        self.assertIn("Facts-only", a.disclaimer)

    def test_inline_url_is_promoted_to_the_single_citation(self):
        a = enforce(self._ans(f"See {BRIEF_URLS[1]} for details.", ""), "answer")
        self.assertEqual(a.citation, BRIEF_URLS[1])
        self.assertNotIn("http", a.text)

    def test_duplicate_sentences_are_collapsed(self):
        a = enforce(self._ans("The exit load is 1%. The exit load is 1%.", BRIEF_URLS[0]),
                    "answer")
        self.assertEqual(len(split_sentences(a.text)), 1)


# ------------------------------------------------------------------ registry

class TestSourceRegistry(unittest.TestCase):
    def test_brief_urls_are_allowlisted(self):
        for u in BRIEF_URLS:
            self.assertIn(u, ALLOWED_URLS)

    def test_no_third_party_blog_or_news_in_registry(self):
        banned = ("blog.", "medium.com", "news18", "scripbox", "5paisa", "blinkx",
                  "basunivesh", "wikipedia", "reddit", "youtube", "facebook")
        for s in SOURCES:
            for b in banned:
                self.assertNotIn(b, s.url.lower(), f"non-official source: {s.url}")

    def test_every_scheme_has_a_source(self):
        for key in SCHEMES:
            self.assertTrue(any(s.scheme_key == key for s in SOURCES), key)

    def test_unquotable_sources_are_flagged(self):
        for s in SOURCES:
            if not s.quotable:
                self.assertTrue(s.note, f"{s.source_id} must explain why it is a pointer")
                self.assertFalse(s.verified_at_build, s.source_id)

    def test_pointer_notes_name_an_actual_reason(self):
        """A pointer-only source must record *why*, not just what it is.

        A note that only describes the source's purpose would let a genuinely
        broken source (typo in the URL, dead host) masquerade as a deliberate
        design decision.
        """
        causes = ("spa", "bot", "client-rendered", "no prose", "dns",
                  "non-browser", "does download")
        for s in SOURCES:
            if not s.quotable:
                note = s.note.lower()
                self.assertTrue(any(c in note for c in causes),
                                f"{s.source_id} note does not name a reason: {s.note}")

    def test_every_emittable_url_is_allowlisted(self):
        """Nothing the assistant can print may fall outside the allow-list.

        The stage-8 G2 gate silently drops a non-allow-listed URL, so an education
        link outside ALLOWED_URLS would make a refusal render with no citation at
        all -- breaking FR-2 in exactly the case that must never improvise.
        """
        for topic, url in EDUCATION_LINKS.items():
            self.assertIn(url, ALLOWED_URLS, f"education link {topic!r} not allow-listed")
        for u in BRIEF_URLS:
            self.assertIn(u, ALLOWED_URLS, u)

    def test_aliases_are_unique_within_each_scheme(self):
        for key, v in SCHEMES.items():
            aliases = v["aliases"]
            self.assertEqual(len(aliases), len(set(aliases)),
                             f"{key} has a duplicate alias, which makes "
                             f"longest-match resolution ambiguous")


# ------------------------------------------------------------------ end to end

@unittest.skipUnless(HAVE_INDEX, "index not built; run `python run.py build` first")
class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipe = FAQPipeline()

    @classmethod
    def tearDownClass(cls):
        cls.pipe.close()

    def test_every_golden_query_is_answered_with_an_allowlisted_citation(self):
        for case in GOLDEN:
            a = self.pipe.ask(case.q)
            self.assertEqual(a.kind, "answer", case.q)
            self.assertIn(a.citation, ALLOWED_URLS, case.q)
            self.assertLessEqual(len(split_sentences(a.text)), MAX_SENTENCES, case.q)

    def test_every_policy_query_is_refused(self):
        for case in POLICY:
            a = self.pipe.ask(case.q)
            self.assertEqual(a.kind, case.kind, case.q)
            self.assertEqual(a.text, REFUSAL_BODIES[case.kind], case.q)

    def test_pii_never_appears_in_any_output(self):
        for case in GOLDEN + POLICY:
            a = self.pipe.ask(case.q)
            blob = f"{a.text} {a.citation} {a.guards}"
            self.assertNotIn("ABCDE1234F", blob, case.q)
            self.assertNotIn("investor@example.com", blob, case.q)

    def test_answers_never_contain_a_second_link(self):
        for case in GOLDEN:
            a = self.pipe.ask(case.q)
            self.assertEqual(len(re.findall(r"https?://", a.text)), 0, case.q)

    def test_render_always_carries_source_and_freshness(self):
        a = self.pipe.ask("What is the expense ratio of HDFC Large Cap Fund?")
        out = a.render()
        self.assertIn("Source: https://", out)
        self.assertIn("Last updated from sources: ", out)
        self.assertIn("Facts-only. No investment advice.", out)

    def test_retrieval_is_deterministic(self):
        q = "What is the minimum SIP for HDFC Small Cap Fund?"
        self.assertEqual(self.pipe.ask(q).text, self.pipe.ask(q).text)

    def test_cross_scheme_queries_do_not_leak(self):
        a = self.pipe.ask("What is the expense ratio of HDFC ELSS Tax Saver Fund?")
        self.assertIn("1.21", a.text)
        self.assertNotIn("1.03", a.text)

    def test_ui_assets_render(self):
        from mf_faq import EXAMPLE_QUESTIONS
        from mf_faq.ui import render_page
        page = render_page()
        self.assertIn("Facts-only", page)
        for q in EXAMPLE_QUESTIONS:
            self.assertIn(q, page, f"example question missing from UI: {q}")
        self.assertIn("/ask", page)


class TestIndexReproducibility(unittest.TestCase):
    """Phase 5: the build plane and the query plane must not disagree.

    Each test here pins a bug that actually shipped in this repo, not a
    hypothetical one.

    These are hashed-vector invariants (IDf, floating-point planes). They only
    apply to a hashed index: an HF-embedded index reconstructs a model instead
    of an IDf table, which these tests would read as two planes disagreeing.
    """

    @classmethod
    def setUpClass(cls):
        st = index_stats()
        if st.get("embed_model", "hashed") != "hashed":
            raise unittest.SkipTest(
                f"index uses {st.get('embed_model')}; hashed-plan tests not applicable")

    def test_query_plane_restores_the_fitted_default_idf(self):
        # The weight given to a term the corpus never saw used to be rebuilt as
        # max(idf.values()) instead of the log((n+1)+1) the documents were built
        # with -- an 11.5% error applied to every out-of-vocabulary query term.
        stats = index_stats()
        loaded = load_vectoriser()
        corpus = [c.embed_text() for c in all_chunks()]
        refit = HashedVectoriser(dim=loaded.dim, ngram=loaded.ngram).fit(corpus)
        self.assertAlmostEqual(loaded.default_idf, refit.default_idf, places=12)
        self.assertAlmostEqual(loaded.default_idf,
                               __import__("math").log(stats["n_chunks"] + 1) + 1.0,
                               places=12)

    def test_unseen_query_vector_is_identical_in_both_planes(self):
        loaded = load_vectoriser()
        corpus = [c.embed_text() for c in all_chunks()]
        refit = HashedVectoriser(dim=loaded.dim, ngram=loaded.ngram).fit(corpus)
        for query in ("quantum entanglement zebra", "exit load ELSS taxsaver xyzzy"):
            a, b = refit.transform_one(query), loaded.transform_one(query)
            for x, y in zip(a, b):
                self.assertEqual(x, y, f"planes disagree on {query!r}")

    def test_numpy_and_pure_python_agree_bit_for_bit(self):
        # A float32 numpy accumulator drifted ~6e-8 from the float64 python one,
        # so a machine without numpy produced a different index from the same
        # corpus. Guards against reintroducing float32 or numpy's own norm.
        v = load_vectoriser()
        corpus = [c.embed_text() for c in all_chunks()]
        batched = v.transform(corpus)
        one_by_one = [v.transform_one(t) for t in corpus]
        for i, (row, single) in enumerate(zip(batched, one_by_one)):
            for x, y in zip(row, single):
                self.assertEqual(x, y, f"row {i} differs between planes")

    def test_stored_vectors_are_normalised_and_complete(self):
        stats = index_stats()
        vecs = all_vectors()
        chunks = all_chunks()
        self.assertEqual(len(vecs), stats["n_chunks"])
        # Fact and procedure counts are pinned because they are a function of
        # the scrape and the authored PROCEDURES list. The section count is not
        # pinned: it depends on how much tabular bulk sits on the source pages,
        # which is not a number worth hard-coding and used to be 78 -- four
        # chunks of which held 91% of the corpus token mass.
        # 75 = 15 whitelisted facts present on all five scheme pages. `aum` is
        # the 15th; without it "fund size" had no chunk at all and the ISIN fact
        # won on raw term overlap.
        self.assertEqual(stats["by_kind"]["fact"], 75)
        self.assertEqual(stats["by_kind"]["procedure"], len(PROCEDURES))
        for vec in vecs.values():
            self.assertEqual(len(vec), stats["dim"])
            self.assertAlmostEqual(sum(x * x for x in vec), 1.0, places=5)
        # Every chunk is embeddable: nothing may exceed the token budget that
        # keeps a 512-dim hashed vector from saturating into noise.
        for c in chunks:
            self.assertLessEqual(len(tokenize(c.text)), SECTION_MAX_TOKENS,
                                 f"{c.chunk_id} is {len(tokenize(c.text))} tokens")

    def test_doc_freq_is_populated(self):
        # doc_freq is declared in SCHEMA; it used to stay empty, so the
        # "inspect document frequencies" promise was unbacked.
        import sqlite3
        from mf_faq.config import DB_PATH
        con = sqlite3.connect(str(DB_PATH))
        try:
            n = con.execute("SELECT count(*) FROM doc_freq").fetchone()[0]
        finally:
            con.close()
        self.assertGreater(n, 0)
        self.assertEqual(n, index_stats()["vocab"])

    def test_cosine_is_the_pure_python_path(self):
        # Guards the deliberate removal of the numpy branch: a float32 BLAS dot
        # and a float64 python sum rank ties differently.
        import inspect
        from mf_faq import indexing as indexing_mod
        src = inspect.getsource(indexing_mod.cosine)
        self.assertNotIn("_np", src)
        self.assertAlmostEqual(cosine([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]), 1.0, places=12)
        self.assertAlmostEqual(cosine([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), 0.0, places=12)


class TestPIIValueCoverage(unittest.TestCase):
    """Phase 6: FR-3. Each test pins a leak that actually shipped here."""

    def test_short_upi_handles_are_detected(self):
        # The bank pattern required {4,} chars before the "@", so the most common
        # UPI handles in the country -- one to three characters -- were not
        # detected, and the query came back as an ordinary answer.
        for handle in ("a@okaxis", "rk@okicici", "ab@okhdfcbank", "rohan@okaxis"):
            q = f"my upi is {handle}"
            d = classify(q)
            self.assertEqual(d.kind, "pii_refusal", f"{handle!r} was not refused")
            self.assertEqual(d.raw_query, "")
            self.assertNotIn(handle, d.query, f"{handle!r} survived in query")

    def test_ifsc_values_are_detected(self):
        # An IFSC has no "@" at all, so the UPI-shaped pattern never matched one
        # and the code leaked verbatim.
        for code in ("HDFC0001234", "SBIN0000999", "ICIC0000032"):
            q = f"ifsc is {code}"
            d = classify(q)
            self.assertEqual(d.kind, "pii_refusal", f"{code!r} was not refused")
            self.assertNotIn(code, d.query, f"{code!r} survived in query")

    def test_bank_credential_questions_refuse_without_a_value(self):
        # "upi"/"ifsc" were missing from PII_KEYWORDS while "bank account" was
        # present, so asking about your own UPI id was *answered* rather than
        # refused.
        for q in ("what is my upi id", "what is my ifsc",
                  "how do I find my upi", "is my ifsc correct",
                  "what is my bank account balance"):
            self.assertEqual(classify(q).kind, "pii_refusal", q)

    def test_unrecognised_value_after_a_keyword_is_still_redacted(self):
        # On the keyword path no value pattern fires, so a credential whose
        # shape the patterns do not know used to survive in Decision.query --
        # raw_query was cleared but query is the field the pipeline uses.
        for q, secret in (("my pan is AB1", "AB1"),
                          ("my upi is 12345", "12345"),
                          ("my ifsc is ABCD0123456", "ABCD0123456")):
            d = classify(q)
            self.assertEqual(d.kind, "pii_refusal", q)
            self.assertNotIn(secret, d.query, f"{secret!r} survived in query: {d.query!r}")

    def test_keyword_redaction_does_not_eat_ordinary_words(self):
        # "account statement" must not become "account [redacted]" -- that query
        # is a golden one and must stay an answer.
        for q in ("Where can I download my account statement?",
                  "How do I download a capital gains statement?"):
            d = classify(q)
            self.assertEqual(d.kind, "answer", q)
            self.assertNotIn("redacted", d.query, f"over-redacted: {d.query!r}")

    def test_no_identifier_value_survives_any_refusal_path(self):
        secrets = ["ABCDE1234F", "2345 6789 0123", "a@okaxis",
                   "HDFC0001234", "rohan.sharma@example.com", "9876543210"]
        queries = [
            f"My PAN is {secrets[0]}, what is the exit load?",
            f"aadhaar {secrets[1]} and should I buy HDFC Large Cap?",
            f"email {secrets[4]} - what are the returns of HDFC Large Cap?",
            f"should I buy HDFC Small Cap with folio no {secrets[5]}?",
            f"my upi is {secrets[2]} and is HDFC Small Cap risky?",
            f"ifsc {secrets[3]} which is better, large or small cap?",
        ]
        for q in queries:
            d = classify(q)
            self.assertEqual(d.kind, "pii_refusal", q)
            self.assertEqual(d.raw_query, "", f"raw_query survived: {q}")
            blob = json.dumps(d.to_dict())
            for s in secrets:
                if s in q:
                    self.assertNotIn(s, blob, f"{s!r} leaked into {blob}")


class TestRetrievalSeamAndExplainability(unittest.TestCase):
    """Phase 7: the Phase-5 swap must stay inside indexing.py, and every
    intermediate ranking signal must be visible."""

    @classmethod
    def setUpClass(cls):
        from mf_faq.retrieval import Retriever
        cls.r = Retriever()

    @classmethod
    def tearDownClass(cls):
        cls.r.close()

    def test_a_fact_from_another_scheme_never_answers(self):
        # "What is the lock-in period in HDFC Small Cap Fund?" was answered with
        # ELSS Tax Saver's 3-year lock-in. Only ELSS carried a lock-in fact, and
        # the wrong-scheme penalty (-0.11) could not survive a saturated BM25
        # score plus the topic boost. Scheme identity has to be a hard gate.
        for q, scheme in (
            ("What is the lock-in period in HDFC Small Cap Fund?", "small_cap"),
            ("What is the expense ratio of HDFC Large Cap Fund?", "large_cap"),
            ("Who manages HDFC Balanced Advantage Fund?", "balanced_advantage"),
        ):
            res = self.r.retrieve(classify(q), top_k=5)
            self.assertEqual(res.decision.scheme_key, scheme)
            for h in res.hits:
                if h.chunk.kind == "fact":
                    self.assertIn(h.chunk.scheme_key, (None, scheme),
                                  f"{scheme} question returned a fact for "
                                  f"{h.chunk.scheme_key}: {h.chunk.chunk_id}")

    def test_every_scheme_has_a_lock_in_fact(self):
        # The half of the fix that keeps the gate from degrading into "could not
        # find". A scheme with no lock-in must still be able to say so.
        from mf_faq.indexing import all_chunks
        from mf_faq.sources import SCHEMES
        have = {c.scheme_key for c in all_chunks()
                if c.kind == "fact" and c.meta.get("field") == "lock_in_days"}
        self.assertEqual(have, set(SCHEMES))

    def test_scheme_agnostic_questions_still_reach_procedures(self):
        # The gate keys on `kind == "fact"`, so it must not touch procedures.
        for q in ("How do I download a capital gains statement?",
                  "How do I get my tax statement?"):
            res = self.r.retrieve(classify(q), top_k=3)
            self.assertTrue(res.hits, q)
            self.assertTrue(any(h.chunk.kind == "procedure" for h in res.hits), q)

    def test_retrieval_does_not_import_the_vectoriser_class(self):
        # The seam constraint exists so a Phase-5 Variant A/B swap stays in one
        # module. Importing the class or a private helper puts that choice back
        # on retrieval.py's doorstep. Both were imported and unused.
        import ast
        src = (ROOT / "src" / "mf_faq" / "retrieval.py").read_text(encoding="utf-8")
        imported = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ImportFrom) and node.module == "indexing":
                imported = {a.name for a in node.names}
        self.assertLessEqual(imported, {"load_vectoriser", "all_vectors", "cosine"},
                             f"seam leak: {sorted(imported)}")

    def test_chunk_meta_is_loaded(self):
        # The documented failure mode: everything runs, only the field signals
        # vanish, and the wrong fact wins on raw term count.
        withfield = [c for c in self.r._chunks.values() if c.meta.get("field")]
        self.assertGreater(len(withfield), 0)
        self.assertTrue(all(c.meta for c in self.r._chunks.values()))

    def test_field_reason_appears_for_a_hinted_query(self):
        # `expense_ratio_pct` only matched "TER"/"total expense", so the plain
        # phrasing never produced a hint and no `field:` reason ever showed up
        # in the phase's own verification output.
        d = classify("What is the expense ratio of HDFC Large Cap Fund?")
        self.assertIn("expense_ratio_pct", d.field_hints)
        res = self.r.retrieve(d, top_k=3)
        flat = [f for h in res.hits for f in h.reasons if f.startswith("field:")]
        self.assertIn("field:expense_ratio_pct", flat)
        self.assertEqual(res.hits[0].chunk.meta.get("field"), "expense_ratio_pct")

    def test_every_hinted_golden_query_gets_a_field_reason(self):
        from mf_faq.eval import GOLDEN
        for case in GOLDEN:
            d = classify(case.q)
            if not d.field_hints:
                continue
            res = self.r.retrieve(d, top_k=3)
            flat = [f for h in res.hits for f in h.reasons if f.startswith("field:")]
            self.assertTrue(flat, f"no field: reason for {case.q!r} (hints={d.field_hints})")

    def test_explain_prints_every_intermediate_signal(self):
        from mf_faq.retrieval import explain
        res = self.r.retrieve(classify("What is the minimum SIP for HDFC Small Cap Fund?"),
                              top_k=3)
        out = explain(res)
        for token in ("final=", "bm25=", "dense=", "scheme=", "topic=",
                      "field=", "kind=", "mmr_penalty="):
            self.assertIn(token, out, f"explain() omits {token}")
        self.assertEqual(out.count("final="), len(res.hits))

    def test_procedural_topics_are_not_answered_by_facts(self):
        for q, kind, topic in [
                ("How do I download a capital gains statement?", "procedure", "capital_gains"),
                ("Where can I download my account statement?", "procedure", "statement_download"),
                ("What is the riskometer level of HDFC Small Cap Fund?", "procedure", "riskometer")]:
            res = self.r.retrieve(classify(q), top_k=3)
            self.assertTrue(res.hits, q)
            self.assertEqual((res.hits[0].chunk.kind, res.hits[0].chunk.topic),
                             (kind, topic), q)

    def test_mmr_reuses_stored_vectors(self):
        import ast
        src = (ROOT / "src" / "mf_faq" / "retrieval.py").read_text(encoding="utf-8")
        fn = [n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_mmr"][0]
        body = ast.unparse(fn)
        self.assertNotIn("transform_one", body)
        self.assertIn("self.dense.matrix", body)

    def test_every_candidate_has_a_vector(self):
        from mf_faq.indexing import all_vectors
        vecs = all_vectors()
        self.assertEqual([c for c in self.r._chunks if c not in vecs], [])


@unittest.skipUnless(HAVE_INDEX, "index not built; run `python run.py build` first")
class TestRetrievalQualityMetrics(unittest.TestCase):
    """Ranking quality, not just answer correctness.

    `evaluate()` grades the answer, which is pass/fail: a pipeline can answer
    from the wrong rank-3 chunk and still pass. These pin the graded metrics so
    a change to the fusion weights can be judged on the ranking itself.
    """

    @classmethod
    def setUpClass(cls):
        from mf_faq.eval import evaluate_retrieval
        cls.rep = evaluate_retrieval(verbose=False)
        cls.m = cls.rep["metrics"]

    def test_every_golden_answer_query_puts_a_relevant_chunk_at_rank_1(self):
        self.assertEqual(self.m["n_queries"], len(GOLDEN))
        self.assertEqual(self.m["top1_accuracy"], 1.0)
        self.assertEqual(self.m["recall@1"], 1.0)
        self.assertEqual(self.m["mrr"], 1.0)

    def test_reported_cutoffs_are_honest(self):
        # recall@k is monotone non-increasing in k, and precision@k cannot
        # exceed recall@k by enough to look like a bug. Cheap sanity on the
        # aggregation, which is easy to get subtly wrong.
        ks = self.m["ks"]
        recalls = [self.m[f"recall@{k}"] for k in ks]
        self.assertEqual(recalls, sorted(recalls, reverse=True))
        for k in ks:
            self.assertLessEqual(self.m[f"precision@{k}"], self.m[f"recall@{k}"] + 1e-9)

    def test_relevance_requires_the_figure_not_just_the_url(self):
        # A Groww page contributes ~10 chunks, so URL-only relevance would mark
        # nearly every hit relevant and make recall meaningless.
        from mf_faq.eval import Case, _fmt_er, is_relevant
        from mf_faq.indexing import all_chunks
        chunks = all_chunks()
        large = [c for c in chunks if c.url.endswith("hdfc-large-cap-fund-direct-growth")]
        self.assertGreater(len(large), 5, "expected several chunks on one page")
        # Derived from the cached source, not a literal: the TER was revised
        # upstream mid-suite-development and the hardcoded "1.03" silently
        # stopped matching its own chunk.
        case = Case("expense ratio of HDFC Large Cap Fund",
                    expect_url=large[0].url, expect_substrings=[_fmt_er("large_cap")])
        hit = [c for c in large if is_relevant(case, c)]
        self.assertTrue(hit)
        self.assertLess(len(hit), len(large), "relevance must discriminate within a page")

    def test_ndcg_penalises_a_relevant_chunk_below_a_relevant_one(self):
        from mf_faq.eval import _dcg
        # Discounted gain must fall when the relevant hit is pushed down, and a
        # perfect ordering must score highest.
        good = _dcg([1, 0, 0, 0])
        bad = _dcg([0, 0, 0, 1])
        self.assertGreater(good, bad)
        self.assertEqual(good, 1.0)

    def test_ks_flag_limits_the_reported_cutoffs(self):
        from mf_faq.eval import evaluate_retrieval
        from mf_faq.pipeline import FAQPipeline
        pipe = FAQPipeline()
        try:
            rep = evaluate_retrieval(pipe, ks=(1, 2), verbose=False)
        finally:
            pipe.close()
        self.assertEqual(rep["metrics"]["ks"], [1, 2])
        self.assertIn("recall@2", rep["metrics"])
        self.assertNotIn("recall@10", rep["metrics"])


class TestNearestNeighbours(unittest.TestCase):
    """`dump -n` is the only way to see relative position in vector space."""

    def setUp(self):
        self.vecs = {
            "a": [1.0, 0.0, 0.0],
            "b": [0.9, 0.1, 0.0],     # close to a
            "c": [0.0, 1.0, 0.0],
        }

    def test_ranks_by_cosine_descending(self):
        from mf_faq.indexing import nearest_neighbours
        nn = nearest_neighbours(self.vecs, "a", k=2)
        self.assertEqual([cid for cid, _ in nn], ["b", "c"])
        self.assertGreater(nn[0][1], nn[1][1])

    def test_excludes_self_and_respects_k(self):
        from mf_faq.indexing import nearest_neighbours
        nn = nearest_neighbours(self.vecs, "a", k=5)
        self.assertNotIn("a", [cid for cid, _ in nn])
        self.assertEqual(len(nn), 2)
        self.assertEqual(nearest_neighbours(self.vecs, "a", k=0), [])

    def test_missing_id_is_empty_not_an_error(self):
        from mf_faq.indexing import nearest_neighbours
        self.assertEqual(nearest_neighbours(self.vecs, "nope", k=3), [])

    def test_identical_normalised_vectors_score_one(self):
        # `cosine` is a plain dot product and relies on the caller having
        # L2-normalised both sides -- which the index guarantees, and these
        # fixtures must reproduce, or this asserts 1.0 for the wrong reason.
        from mf_faq.indexing import nearest_neighbours
        u = [0.5 ** 0.5, 0.5 ** 0.5, 0.0]
        v = {"a": u, "b": list(u)}
        self.assertAlmostEqual(nearest_neighbours(v, "a", 1)[0][1], 1.0, places=6)


class TestSynthesisComposition(unittest.TestCase):
    """Phase 8: the field hint must FILTER, and the evidence schema must be
    uniform across all three generators."""

    @classmethod
    def setUpClass(cls):
        from mf_faq.retrieval import Retriever
        cls.r = Retriever()
        cls.syn = get_synthesiser()

    @classmethod
    def tearDownClass(cls):
        cls.r.close()

    def _fake(self, hits, hints, topic="scheme_meta", scheme="balanced_advantage"):
        class _O:
            def __init__(self, **kw):
                self.__dict__.update(kw)
        hs = []
        for cid, fld, txt in hits:
            hs.append(_O(chunk=_O(chunk_id=cid, kind="fact", topic=topic,
                                  scheme_key=scheme, url="https://groww.in/x",
                                  text=txt, answer_hint=txt, last_updated="2026-09-25",
                                  meta={"field": fld}), final=1.0, reasons=[]))
        return _O(decision=_O(primary_topic=topic, field_hints=hints, scheme_key=scheme,
                              query="q"), hits=hs, n_candidates=len(hs),
                  index_stats={}, timings_ms={})

    def test_field_hint_filters_rather_than_boosts(self):
        # Retrieval already boosts the hinted field; correctness must not depend
        # on that boost winning, so synthesis drops every non-hinted fact.
        from mf_faq.synthesis import ExtractiveSynthesiser
        res = self._fake([("a", "fund_manager", "The fund manager of X is A Person."),
                          ("b", "registrar_agent", "The registrar and transfer agent is CAMS."),
                          ("c", "isin", "The ISIN of X is INE040H01021.")],
                         ["fund_manager"])
        d = ExtractiveSynthesiser().generate(res)
        self.assertIn("A Person", d.answer)
        self.assertNotIn("CAMS", d.answer)
        self.assertNotIn("INE040H01021", d.answer)

    def test_field_hint_with_no_match_still_answers(self):
        from mf_faq.synthesis import ExtractiveSynthesiser
        res = self._fake([("a", "isin", "The ISIN of X is INE040H01021.")],
                         ["no_such_field"])
        d = ExtractiveSynthesiser().generate(res)
        self.assertTrue(d.answer.strip())

    def test_evidence_schema_is_uniform_across_generators(self):
        # The three generators used to emit three different shapes, so the
        # documented access pattern e["reasons"] raised KeyError for a riskometer
        # or statement answer while working for a fact answer.
        from mf_faq.synthesis import ExtractiveSynthesiser
        syn = ExtractiveSynthesiser()
        cases = {
            "fact": "What is the expense ratio of HDFC Large Cap Fund?",
            "procedural": "How do I download a capital gains statement?",
            "pointer": "What is the riskometer level of HDFC Small Cap Fund?",
        }
        shapes = set()
        for label, q in cases.items():
            res = self.r.retrieve(classify(q), top_k=3)
            ev = syn.generate(res).evidence
            self.assertTrue(ev, label)
            for e in ev:
                shapes.add(tuple(sorted(e)))
                self.assertIn("reasons", e, f"{label} evidence missing reasons")
                _ = e["reasons"][:1]
        self.assertEqual(len(shapes), 1,
                         f"generators disagree on evidence keys: {shapes}")

    def test_date_is_coupled_to_the_cited_chunk(self):
        # A fact answer copies the chunk's source date; an authored procedure has
        # no source date, so the draft must not claim source freshness.
        from mf_faq.synthesis import ExtractiveSynthesiser
        syn = ExtractiveSynthesiser()
        fact = syn.generate(self.r.retrieve(
            classify("What is the expense ratio of HDFC Large Cap Fund?"), top_k=3))
        self.assertTrue(fact.freshness_from_source)
        self.assertTrue(fact.last_updated)
        for q in ("How do I download a capital gains statement?",
                  "What is the riskometer level of HDFC Small Cap Fund?"):
            d = syn.generate(self.r.retrieve(classify(q), top_k=3))
            self.assertFalse(d.freshness_from_source, q)
            self.assertTrue(d.last_updated, q)

    def test_companion_sentence_is_same_scheme_only(self):
        from mf_faq.synthesis import ExtractiveSynthesiser
        class _O:
            def __init__(self, **kw):
                self.__dict__.update(kw)
        def h(cid, fld, scheme, txt):
            return _O(chunk=_O(chunk_id=cid, kind="fact", topic="min_sip",
                              scheme_key=scheme, url="https://groww.in/x", text=txt,
                              answer_hint=txt, last_updated="2026-09-25",
                              meta={"field": fld}), final=1.0, reasons=[])
        hits = [h("a", "min_sip_amount", "small_cap", "The minimum SIP is 500."),
                h("b", "min_lumpsum_amount", "large_cap", "The minimum lump sum is 1000.")]
        res = _O(decision=_O(primary_topic="min_sip", field_hints=["min_sip_amount"],
                             scheme_key="small_cap", query="q"), hits=hits,
                 n_candidates=2, index_stats={}, timings_ms={})
        self.assertNotIn("1000", ExtractiveSynthesiser().generate(res).answer)

    def test_empty_retrieval_message_on_every_path(self):
        from mf_faq.synthesis import ExtractiveSynthesiser
        class _O:
            def __init__(self, **kw):
                self.__dict__.update(kw)
        res = _O(decision=_O(primary_topic="expense_ratio", field_hints=[],
                             scheme_key=None, query="q"), hits=[], n_candidates=0,
                 index_stats={}, timings_ms={})
        syn = ExtractiveSynthesiser()
        for got in (syn.generate(res), syn._procedural(res), syn._pointer(res)):
            self.assertEqual(got.answer, "I could not find this in the collected sources.")
            self.assertEqual(got.notes, ["empty_retrieval"])

    def test_dead_date_formatter_is_gone(self):
        # _format_date was unreachable dead code that truncated its input to 10
        # chars, making its own ISO-datetime format permanently unmatchable.
        from mf_faq import synthesis
        self.assertFalse(hasattr(synthesis, "_format_date"))

    def test_llm_prompts_pin_temperature_and_cover_bank_details(self):
        from mf_faq import llm as llm_mod
        import ast
        tree = ast.parse((ROOT / "src" / "mf_faq" / "llm.py").read_text(encoding="utf-8"))
        # groq/openai/anthropic all build the body through the shared
        # `_post_openai_compatible` / `_post_anthropic`, so the temperature is
        # pinned in one place for the compatible family.
        shared = [n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name in ("_post_openai_compatible", "_post_anthropic")]
        temp_vals = []
        for fdef in shared:
            temp_vals += [v.value for node in ast.walk(fdef) if isinstance(node, ast.Dict)
                          for k, v in zip(node.keys, node.values)
                          if isinstance(k, ast.Constant) and k.value == "temperature"]
        self.assertTrue(temp_vals, "no temperature=0 found in the request bodies")
        self.assertEqual(set(temp_vals), {0})
        # must name what stage 5 actually detects
        for token in ("PAN", "Aadhaar", "OTP", "UPI", "IFSC"):
            self.assertIn(token, llm_mod.SYSTEM)

    def test_llm_falls_back_with_a_note(self):
        import unittest.mock as mock
        from mf_faq.synthesis import LLMSynthesiser
        from mf_faq.synthesis import ExtractiveSynthesiser
        res = self.r.retrieve(
            classify("What is the expense ratio of HDFC Large Cap Fund?"), top_k=3)
        # The note under test is the *no provider* path, so the provider has to be
        # off. Left to the environment this test can never pass on a configured
        # machine: `available()` is True whenever .env holds a live key, so it made
        # a real API call, and either succeeded (used_llm True) or failed with a
        # quota error, giving `llm_error:...` rather than the note asserted here.
        with mock.patch("mf_faq.llm.available", return_value=False):
            d = LLMSynthesiser().generate(res)
        self.assertFalse(d.used_llm)
        self.assertIn("llm_unavailable_fallback", d.notes)
        self.assertEqual(d.answer, ExtractiveSynthesiser().generate(res).answer)

    def test_llm_parse_rejects_unusable_payloads(self):
        from mf_faq.synthesis import LLMSynthesiser
        p = LLMSynthesiser._parse
        for raw in (None, "not json", '{"citation": "u"}', '{"answer": ""}',
                    '{"answer": 123}', "[1,2,3]", "{}"):
            self.assertIsNone(p(raw), f"should reject {raw!r}")
        self.assertIsNotNone(p('{"answer": "ok", "citation": "u"}'))
        self.assertIsNotNone(p('prose {"answer": "ok"} prose'))


class TestGuardrailIndependence(unittest.TestCase):
    """Stage 5 and stage 8 must be two nets, not one net used twice.

    `guardrails.py` used to `from .intent import PII_PATTERNS`, which made G1
    literally the same objects as the input gate. Every gap in one was then a gap
    in the other, which is the exact failure the two-net design exists to stop.
    """

    def test_guardrails_imports_nothing_from_intent(self):
        import ast
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src" / "mf_faq" / "guardrails.py")
        tree = ast.parse(src.read_text(encoding="utf-8"))
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mods.add((node.module or "").lstrip("."))
                mods.update(a.name for a in node.names)
            elif isinstance(node, ast.Import):
                mods.update(a.name for a in node.names)
        for banned in ("intent", "PII_PATTERNS", "ADVICE_PATTERNS", "PERFORMANCE_PATTERNS"):
            self.assertNotIn(banned, mods, f"guardrails must not import {banned}")

    def test_output_net_catches_what_the_value_net_misses(self):
        # Each of these is missed by intent.PII_PATTERNS: the PAN net is
        # upper-case only and the OTP net requires 4-6 digits. If G1 ever goes
        # back to importing those patterns, these stop being caught.
        from mf_faq.guardrails import _contains_pii
        for s in ("Your pan is abcde1234f.", "Please confirm otp is 123."):
            self.assertTrue(_contains_pii(s), s)

    def test_stage5_value_net_really_does_miss_those(self):
        # Guards the guard: if stage 5 is ever hardened to catch them, this test
        # tells us the two nets stopped being independent evidence.
        import re
        from mf_faq.intent import PII_PATTERNS
        for s in ("Your pan is abcde1234f.", "Please confirm otp is 123."):
            hit = any(p.search(s) for _, p in PII_PATTERNS) or \
                re.search(r"(?<!\d)\d{10,12}(?!\d)", s)
            self.assertFalse(hit, f"stage 5 now catches {s!r}; nets are no longer distinct")


class TestOutputNetRecall(unittest.TestCase):
    """The output nets must catch the phrasings the stage-5 nets are aimed at,
    and must not fire on any quotable fact."""

    def test_performance_net_catches_canonical_return_phrasings(self):
        from mf_faq.guardrails import PERFORMANCE_LEAK
        for s in ("The 1-year return is 12.5%.",
                  "HDFC Small Cap returned 18% last year.",
                  "The fund has given a return of 12% in 2024.",
                  "Annualised return since inception is 14.1%.",
                  "It yielded 15% over 5 years.",
                  "CAGR is 13.4%.",
                  "The scheme underperformed last year."):
            self.assertTrue(PERFORMANCE_LEAK.search(s), s)

    def test_performance_net_spares_legitimate_answers(self):
        from mf_faq.guardrails import PERFORMANCE_LEAK
        for s in ("The expense ratio of HDFC Large Cap Fund (Direct - Growth) is 1.03%.",
                  "The exit load is 1% for holding periods under 12 months.",
                  "The NAV of HDFC Flexi Cap Fund (Direct - Growth) is 2,214.572.",
                  "The ISIN of HDFC Large Cap Fund is INE040H01021.",
                  "The fund manager is Srinivas Rao Ravuri.",
                  # "Total Return Index" is the benchmark's name, not a claim.
                  "The benchmark of HDFC Large Cap Fund (Direct - Growth) is "
                  "NIFTY 100 Total Return Index."):
            self.assertFalse(PERFORMANCE_LEAK.search(s), s)

    def test_no_fact_chunk_in_the_corpus_trips_any_net(self):
        """A guardrail that refuses a legitimate fact is worse than no guardrail,
        so the whole quotable corpus is swept rather than a sample of it."""
        import json
        from pathlib import Path
        from mf_faq.guardrails import (ADVICE_LEAK, PERFORMANCE_LEAK, _contains_pii,
                                       enforce)
        path = Path(__file__).resolve().parents[1] / "data" / "chunks.jsonl"
        n = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            if c["kind"] != "fact":
                continue
            n += 1
            self.assertEqual(_contains_pii(c["text"]), [], c["chunk_id"])
            self.assertFalse(ADVICE_LEAK.search(c["text"]), c["chunk_id"])
            self.assertFalse(PERFORMANCE_LEAK.search(c["text"]), c["chunk_id"])
            got = enforce(Answer(kind="answer", text=c["text"], citation=c["url"],
                                 last_updated=c["last_updated"]), "answer")
            self.assertEqual(got.kind, "answer", c["chunk_id"])
        self.assertGreater(n, 50, "the sweep covered the whole fact corpus")


class TestRefusalContract(unittest.TestCase):
    def test_all_four_canonical_kinds_report_as_refusals(self):
        # `empty` is a refusal kind, but it does not end in "_refusal", so the
        # suffix test reported False for "I didn't catch a question".
        from mf_faq.guardrails import REFUSAL_BODIES
        for kind in REFUSAL_BODIES:
            self.assertTrue(Answer(kind=kind, text="x").is_refusal, kind)
            self.assertFalse(Answer(kind="answer", text="x").is_refusal)

    def test_every_refusal_carries_an_allowlisted_learning_link(self):
        from mf_faq.guardrails import REFUSAL_LINKS, make_refusal
        self.assertEqual(set(REFUSAL_LINKS), set(REFUSAL_BODIES))
        for kind in REFUSAL_LINKS:
            self.assertTrue(REFUSAL_LINKS[kind], f"{kind} has no link")
            self.assertIn(REFUSAL_LINKS[kind], ALLOWED_URLS, kind)
            self.assertIn(make_refusal(kind).citation, ALLOWED_URLS, kind)

    def test_enforce_never_reclassifies_its_own_refusal_templates(self):
        """The advice template says "whether you should buy" in order to
        decline. Running the nets over it would make the refusal look like a
        leak, so enforce() leaves a canonical refusal alone."""
        from mf_faq.guardrails import REFUSAL_BODIES, enforce
        for kind, body in REFUSAL_BODIES.items():
            got = enforce(Answer(kind=kind, text=body), kind)
            self.assertEqual(got.kind, kind)
            self.assertEqual(got.text, body)

    def test_advice_template_is_not_its_own_leak(self):
        # Documented as a deliberate exception: asserting "the refusal contains
        # no advice words" fails on the very refusal it should pass.
        from mf_faq.guardrails import ADVICE_LEAK, REFUSAL_BODIES
        self.assertIn("you should", REFUSAL_BODIES["advice_refusal"])
        self.assertTrue(ADVICE_LEAK.search(REFUSAL_BODIES["advice_refusal"]),
                        "if this ever stops matching, the short-circuit is untested")


class TestAnswerProvenanceAndGuards(unittest.TestCase):
    def test_g6_stamp_is_a_preparation_date_not_source_freshness(self):
        from mf_faq.guardrails import enforce
        a = enforce(Answer(kind="answer", text="The NAV was 10.",
                           citation=BRIEF_URLS[0], last_updated=""), "answer")
        self.assertTrue(a.last_updated)
        self.assertIn("G6_stamp_added", a.guards)
        self.assertFalse(a.freshness_from_source)
        self.assertIn("Answer prepared on:", a.render())
        self.assertNotIn("Last updated from sources:", a.render())

    def test_every_answer_carries_todays_date_and_time(self):
        """The complaint that motivated this: an answer served out of a week-old
        cache showed only the *source's* date, so it read as though the system
        were answering about the present. The fix is not to overwrite the
        source date -- a NAV of 1447.383 is the NAV as of 25 Sep and stamping it
        "today" would assert a value the source never published -- but to always
        add the wall-clock moment the answer was produced, alongside it."""
        import re as _re
        from mf_faq.guardrails import enforce, prepared_stamp
        today = date.today().isoformat()
        for last_updated, from_source in (("", False), ("2026-09-25", True),
                                          ("01-Oct-2026", True)):
            a = enforce(Answer(kind="answer", text="The NAV was 10.",
                               citation=BRIEF_URLS[0], last_updated=last_updated),
                        "answer")
            if last_updated:
                a.last_updated = last_updated
                a.freshness_from_source = from_source
            out = a.render()
            self.assertIn("Answer prepared on:", out)
            stamp = out.split(ANSWER_PREPARED_PREFIX)[1].splitlines()[0]
            self.assertTrue(stamp.startswith(today),
                            f"no current date on the stamp: {stamp!r}")
            self.assertTrue(_re.search(r"\d{2}:\d{2}", stamp),
                            f"stamp carries no time of day: {stamp!r}")
            self.assertIn(prepared_stamp()[:16], out)

    def test_source_date_is_kept_next_to_the_prepared_stamp(self):
        """Both dates, never one substituted for the other."""
        from mf_faq.guardrails import enforce
        a = Answer(kind="answer", text="The NAV was 10.", citation=BRIEF_URLS[0],
                   last_updated="2026-09-25", freshness_from_source=True)
        out = enforce(a, "answer").render()
        self.assertIn(LAST_UPDATED_PREFIX + "2026-09-25", out)
        self.assertIn(ANSWER_PREPARED_PREFIX, out)
        self.assertLess(out.index(ANSWER_PREPARED_PREFIX), out.index(LAST_UPDATED_PREFIX))

    def test_prepared_stamp_does_not_leak_the_body_into_the_source_line(self):
        from mf_faq.guardrails import enforce
        a = enforce(Answer(kind="answer", text="The NAV was 10.",
                           citation=BRIEF_URLS[0], last_updated=""), "answer")
        lines = [ln for ln in a.render().splitlines() if ln.strip()]
        self.assertEqual(len([ln for ln in lines if ln.startswith(ANSWER_PREPARED_PREFIX)]), 1)

    def test_citation_is_empty_or_allowlisted_and_nothing_else(self):
        from mf_faq.guardrails import enforce
        for text, cite in (("The exit load is 1%.", "https://random-blog.example/x"),
                           ("See https://evil.example.com here.", ""),
                           ("The exit load is 1%.", BRIEF_URLS[0]),
                           ("The exit load is 1%.", "")):
            got = enforce(Answer(kind="answer", text=text, citation=cite,
                                 last_updated="2026-09-25"), "answer")
            self.assertTrue(got.citation == "" or got.citation in ALLOWED_URLS,
                            f"third case: {got.citation!r}")
            self.assertEqual("G2_citation_ok" in got.guards, bool(got.citation))
            self.assertNotIn("http", got.text)

    def test_blocking_checks_run_before_the_body_is_cleaned(self):
        """A contaminated body is replaced wholesale, so an allow-listed URL
        inside it must not survive as the citation."""
        from mf_faq.guardrails import enforce
        got = enforce(Answer(kind="answer",
                             text=f"Your PAN ABCDE1234F. See {BRIEF_URLS[0]}.",
                             last_updated="2026-09-25"), "answer")
        self.assertEqual(got.kind, "pii_refusal")
        self.assertNotIn("ABCDE1234F", got.text)
        self.assertNotIn(BRIEF_URLS[0], got.text)
        self.assertTrue(any(g.startswith("G1_pii_blocked") for g in got.guards), got.guards)

    def test_empty_body_becomes_the_empty_refusal_with_its_guards(self):
        from mf_faq.guardrails import REFUSAL_BODIES, enforce
        got = enforce(Answer(kind="answer", text="   ", citation=BRIEF_URLS[0],
                             last_updated="2026-09-25"), "answer")
        self.assertEqual(got.kind, "empty")
        self.assertEqual(got.text, REFUSAL_BODIES["empty"])
        self.assertIn("refusal_template", got.guards)
        self.assertIn("G2_citation_ok", got.guards)


class TestCommandLine(unittest.TestCase):
    """The spec's own verify commands must run, in the order it writes them."""

    def _run(self, *argv):
        import subprocess
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        return subprocess.run([sys.executable, str(root / "run.py"), *argv],
                              capture_output=True, text=True, cwd=str(root))

    def test_debug_flag_is_accepted_on_both_sides_of_the_subcommand(self):
        # `ask --debug "q"` used to die with "unrecognized arguments", and the
        # naive fix (a plain subparser flag) would then clobber the global value
        # and silently turn `run.py --debug ask "q"` into a non-debug run.
        for argv in (["ask", "--debug", "What is the expense ratio of HDFC Large Cap Fund?"],
                     ["--debug", "ask", "What is the expense ratio of HDFC Large Cap Fund?"]):
            p = self._run(*argv)
            self.assertEqual(p.returncode, 0, f"{argv}: {p.stderr[-300:]}")
            self.assertIn("[gate]", p.stdout, f"{argv} did not enable debug output")

    def test_ask_does_not_redisprint_an_identifier_the_user_pasted(self):
        p = self._run("ask", "My PAN is ABCDE1234F, what is the exit load?")
        self.assertEqual(p.returncode, 0, p.stderr[-300:])
        self.assertNotIn("ABCDE1234F", p.stdout)
        self.assertIn("[redacted]", p.stdout)
        self.assertIn("pii_refusal", p.stdout)


class TestOrchestration(unittest.TestCase):
    """Phase 10: the pipeline surface and the properties the CLI depends on."""

    @classmethod
    def setUpClass(cls):
        # One build, run twice: a build costs ~10 s, and doing it per-test turned
        # a 2 s suite into a 70 s one. The second run is not redundancy -- it is
        # the only way to show a rebuild over an existing index succeeds.
        #
        # The index goes to a temp path. `FAQPipeline()` defaults to the real
        # `data/index/mf_faq.sqlite`, so this used to overwrite the shipped index
        # twice per suite run, which rewrote its `built_at` and made
        # `test_committed_samples_match_the_live_pipeline` fail on a stamp that
        # had nothing to do with any answer -- the samples carry the index build
        # time, so the suite invalidated the deliverable it was checking.
        import shutil
        import tempfile
        from mf_faq.pipeline import FAQPipeline
        from mf_faq.sources import SOURCES
        cls.SOURCES = SOURCES
        tmp = tempfile.mkdtemp(prefix="mf_faq_orchestration_")
        cls.p = FAQPipeline(db_path=Path(tmp) / "index.sqlite")
        try:
            cls.first = cls.p.build()
            cls.second = cls.p.build()
        finally:
            cls.p.close()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_retriever_is_lazy(self):
        from mf_faq.pipeline import FAQPipeline
        p = FAQPipeline()
        try:
            self.assertIsNone(p._retriever, "constructing the pipeline opened the index")
        finally:
            p.close()

    def test_missing_index_names_the_fix(self):
        from pathlib import Path
        from mf_faq.pipeline import FAQPipeline
        missing = Path(__file__).resolve().parents[1] / "data" / "index" / "_no_such.sqlite"
        p = FAQPipeline(db_path=missing)
        try:
            with self.assertRaises(FileNotFoundError) as cm:
                p.ask("What is the expense ratio of HDFC Large Cap Fund?")
            self.assertIn("run.py build", str(cm.exception))
        finally:
            p.close()

    def test_refused_query_never_touches_the_index(self):
        """The spec's stated pitfall: the stage-5 refusal short-circuits *before*
        retrieval, so a refused query answers even with no index on disk."""
        from pathlib import Path
        from mf_faq.pipeline import FAQPipeline
        missing = Path(__file__).resolve().parents[1] / "data" / "index" / "_no_such.sqlite"
        p = FAQPipeline(db_path=missing)
        try:
            for q, kind in (("Should I buy HDFC Small Cap Fund?", "advice_refusal"),
                            ("My PAN is ABCDE1234F, what is the exit load?", "pii_refusal"),
                            ("What are the 3 year returns of HDFC Large Cap Fund?",
                             "performance_refusal")):
                a = p.ask(q)
                self.assertEqual(a.kind, kind, q)
                self.assertIsNone(p._retriever, f"{q} opened the index")
                self.assertIn("total_ms", a.timings_ms)
        finally:
            p.close()

    def test_build_is_repeatable_and_does_not_abort_on_a_failed_fetch(self):
        """One of the nine sources cannot be fetched and three more fetch fine but
        are not quotable; the build must still produce the full corpus, and a
        rebuild over an existing index must succeed."""
        for st in (self.first, self.second):
            # Pinned on the fact/procedure counts, which the scrape and the
            # authored PROCEDURES list determine. The section count moves with
            # the token budget in build_section_chunks, so it is asserted
            # against that budget rather than a literal.
            self.assertEqual(st["by_kind"]["fact"], 75)
            self.assertEqual(st["by_kind"]["procedure"], len(PROCEDURES))
            self.assertGreater(st["by_kind"]["section"], 0)
            self.assertLess(st["fetch"]["ok"], st["fetch"]["total"])
            self.assertTrue(st["fetch"]["quarantined"])
            self.assertIsInstance(st["build_seconds"], (int, float))
        self.assertEqual(self.first["n_chunks"], self.second["n_chunks"])
        self.assertEqual(len(self.SOURCES), 9)
        self.assertEqual(sum(1 for s in self.SOURCES if not s.quotable), 4)
        # one fetch failure, three further sources that fetch but are not quotable
        self.assertEqual(len(set(self.second["fetch"]["quarantined"])), 1)
        self.assertEqual(sum(1 for s in self.SOURCES if not s.quotable)
                         - len(self.second["fetch"]["quarantined"]), 3)


class TestCommandLineSurface(unittest.TestCase):
    """The eight subcommands, their flags, and the exit code the spec names."""

    @classmethod
    def setUpClass(cls):
        import re
        from pathlib import Path
        cls.src = (Path(__file__).resolve().parents[1] / "run.py").read_text(encoding="utf-8")
        cls.subs = re.findall(r'sub\.add_parser\("(\w+)"', cls.src)

    def test_the_eight_specified_subcommands_are_present_and_unchanged(self):
        # The brief names eight subcommands and their order; they must still be
        # there. Two diagnostics (`dump`, `retrieval`) were added after it, so
        # this pins the spec'd prefix rather than the whole list -- an exact
        # match would fail on a purely additive change, which is the wrong
        # signal to give a reviewer.
        self.assertEqual(self.subs[:8], ["build", "ask", "serve", "eval", "sources",
                                          "stage5", "stage6", "samples"])
        self.assertEqual(self.subs[8:], ["dump", "retrieval"])

    def test_windows_console_is_reconfigured_before_any_printing(self):
        # cp1252 cannot encode the rupee sign the answers contain.
        self.assertIn("reconfigure", self.src)
        self.assertIn('errors="replace"', self.src)
        self.assertLess(self.src.index("reconfigure"), self.src.index("print("))

    def test_src_is_on_the_path_before_the_package_is_imported(self):
        self.assertLess(self.src.index("sys.path.insert"), self.src.index("from mf_faq"))

    def test_cmd_eval_returns_one_when_a_check_fails(self):
        import run as R
        from mf_faq.eval import Report
        saved = R.evaluate
        try:
            class _A:
                debug = False
            R.evaluate = lambda pipe=None, verbose=True: Report(passed=3, failed=2)
            self.assertEqual(R.cmd_eval(_A()), 1)
            R.evaluate = lambda pipe=None, verbose=True: Report(passed=5, failed=0)
            self.assertEqual(R.cmd_eval(_A()), 0)
        finally:
            R.evaluate = saved

    def test_every_subcommand_exits_zero(self):
        import os
        import subprocess
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        cases = {
            "build": ["build"],
            "ask": ["ask", "What is the expense ratio of HDFC Large Cap Fund?"],
            "eval": ["eval"],
            "sources": ["sources"],
            "stage5": ["stage5", "Should I buy HDFC Small Cap Fund?"],
            "stage6": ["stage6", "What is the minimum SIP for HDFC Small Cap Fund?"],
            "samples": ["samples"],
            "retrieval": ["retrieval"],
        }
        # `build` and `samples` both write into the repository: build overwrites
        # data/corpus.jsonl, data/chunks.jsonl, data/index/mf_faq.sqlite and the
        # fetch manifest, samples overwrites samples/*. That changed `built_at`,
        # which made `test_committed_samples_match_the_live_pipeline` fail on a
        # stamp that had nothing to do with any answer -- the suite was
        # invalidating the deliverable it was checking. `data/raw` is read-only
        # input, so it is left alone. Snapshot the writable artifacts and put
        # them back, so the subcommands still run end to end but leave no trace.
        writable = [p for p in sorted((root / "data").rglob("*"))
                    if p.is_file() and "raw" not in p.relative_to(root / "data").parts]
        writable += [p for p in (root / "samples").glob("*") if p.is_file()]
        saved = {p: p.read_bytes() for p in writable}
        try:
            for sub, argv in cases.items():
                env = dict(os.environ)
                # `.env` may ask for sentence-transformers, which this test host
                # does not install (torch). `build` then exercises the zero-dep
                # hashed path, which is the guaranteed-available default; the HF
                # path is a separate, opt-in contract tested in
                # TestEmbeddingFactory.
                if sub == "build":
                    env["MF_FAQ_EMBED_MODEL"] = "hashed"
                r = subprocess.run([sys.executable, str(root / "run.py"), *argv],
                                   capture_output=True, text=True, cwd=str(root), timeout=600,
                                   env=env)
                self.assertEqual(r.returncode, 0,
                                 f"{sub}: rc={r.returncode}\n"
                                 f"--- stderr ---\n{r.stderr[-800:]}\n"
                                 f"--- stdout ---\n{r.stdout[-800:]}")
                self.assertTrue(r.stdout.strip(), f"{sub} printed nothing")
        finally:
            for p, blob in saved.items():
                p.write_bytes(blob)

    def test_dump_writes_chunks_with_vectors(self):
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "dump.txt"
            r = subprocess.run([sys.executable, str(root / "run.py"), "dump",
                                "-o", str(out), "--kind", "fact", "--limit", "3",
                                "-n", "2", "--no-tokens"],
                               capture_output=True, text=True, cwd=str(root), timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr[-300:])
            body = out.read_text(encoding="utf-8")
        self.assertIn("CHUNKS AND EMBEDDINGS", body)
        self.assertIn("shown        : 3 of", body)
        self.assertEqual(body.count("CHUNK "), 3)
        self.assertIn("l2_norm=1.000000", body)
        self.assertIn("nearest    : cos=", body)

    def test_ask_renders_answer_source_freshness_and_disclaimer(self):
        import subprocess
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        r = subprocess.run([sys.executable, str(root / "run.py"), "ask",
                            "What is the expense ratio of HDFC Large Cap Fund?"],
                           capture_output=True, text=True, cwd=str(root), timeout=300)
        from mf_faq.eval import _fmt_er
        for needle in (f"The expense ratio of HDFC Large Cap Fund "
                       f"(Direct - Growth) is {_fmt_er('large_cap')}%.",
                       "Source: https://groww.in/mutual-funds/"
                       "hdfc-large-cap-fund-direct-growth",
                       "Answer prepared on:", "Last updated from sources:",
                       "Facts-only. No investment advice."):
            self.assertIn(needle, r.stdout, needle)

    def test_ask_json_is_valid_json_with_the_answer_shape(self):
        import json
        import subprocess
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        r = subprocess.run([sys.executable, str(root / "run.py"), "ask", "--json",
                            "benchmark of HDFC Flexi Cap Fund"],
                           capture_output=True, text=True, cwd=str(root), timeout=300)
        d = json.loads(r.stdout)
        self.assertEqual(d["kind"], "answer")
        self.assertIn(d["citation"], ALLOWED_URLS)
        self.assertIn("total_ms", d["timings_ms"])
        self.assertIn("NIFTY 500", d["text"])


class TestWebUIConcurrency(unittest.TestCase):
    """`run.py serve` runs on ThreadingHTTPServer, one thread per connection.

    It used to answer the *first* request and drop every later one: the sqlite
    connection was created in whichever thread handled that first request, and
    every subsequent thread raised ProgrammingError, which socketserver turns
    into a closed socket with no HTTP response. A single smoke test passes, so
    the regression has to send more than one request.
    """

    @classmethod
    def setUpClass(cls):
        import socket
        import subprocess
        import sys
        import tempfile
        import time
        from pathlib import Path
        cls.root = Path(__file__).resolve().parents[1]
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            cls.port = s.getsockname()[1]
        # Log to a file, not a pipe: the assertions need to read the server's
        # output while it is still running, and communicate() would block.
        # -u matters: redirected stdout is block-buffered, and the server is
        # killed rather than shut down cleanly, so a buffered banner is lost and
        # the suppression assertions below would pass on an empty file.
        cls._log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False,
                                               encoding="utf-8", errors="replace")
        cls.proc = subprocess.Popen(
            [sys.executable, "-u", str(cls.root / "run.py"), "serve",
             "--host", "127.0.0.1", "--port", str(cls.port)],
            stdout=cls._log, stderr=subprocess.STDOUT, cwd=str(cls.root))
        cls.base = f"http://127.0.0.1:{cls.port}"
        for _ in range(80):
            time.sleep(0.25)
            if cls.proc.poll() is not None:
                break
            try:
                cls._get("/healthz")
                return
            except Exception:                                  # noqa: BLE001
                continue

    @classmethod
    def tearDownClass(cls):
        if cls.proc.poll() is None:
            cls.proc.terminate()
            try:
                cls.proc.wait(timeout=15)
            except Exception:                                  # noqa: BLE001
                cls.proc.kill()
        try:
            cls._log.close()
        except Exception:                                      # noqa: BLE001
            pass

    @classmethod
    def server_log(cls) -> str:
        import os
        try:
            with open(cls._log.name, encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError:
            return ""

    @classmethod
    def _get(cls, path):
        import urllib.request
        with urllib.request.urlopen(f"{cls.base}{path}", timeout=10) as r:
            return r.status, r.read()

    def _post(self, payload, timeout=30):
        import json
        import urllib.request
        rq = urllib.request.Request(f"{self.base}/ask", data=json.dumps(payload).encode(),
                                    headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(rq, timeout=timeout) as r:
            return json.loads(r.read())

    def test_server_came_up(self):
        st, body = self._get("/healthz")
        self.assertEqual(st, 200)
        self.assertTrue(json.loads(body)["ok"])

    def test_every_sequential_request_is_answered(self):
        # The regression: only the first of these used to come back.
        for i in range(6):
            d = self._post({"q": "What is the expense ratio of HDFC Large Cap Fund?"})
            self.assertEqual(d["kind"], "answer", f"request {i}")
            from mf_faq.eval import _fmt_er
            self.assertIn(_fmt_er("large_cap"), d["text"])

    def test_concurrent_requests_are_all_answered(self):
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            futs = [ex.submit(self._post, {"q": f"What is the exit load on HDFC Small Cap Fund? #{i}"})
                    for i in range(10)]
            kinds = []
            for f in futs:
                kinds.append(f.result(timeout=60)["kind"])
        self.assertEqual(kinds, ["answer"] * 10)

    def test_pii_is_refused_over_http_and_not_echoed(self):
        import json as _json
        d = self._post({"q": "My PAN is ABCDE1234F, what is the exit load?"})
        self.assertEqual(d["kind"], "pii_refusal")
        self.assertNotIn("ABCDE1234F", _json.dumps(d))

    def test_malformed_bodies_do_not_break_the_server(self):
        for payload in ({}, {"q": ""}, {"nope": 1}):
            d = self._post(payload)
            self.assertIn(d["kind"], ("empty", "answer"), payload)
        # and the server is still alive afterwards
        self.assertEqual(self._post({"q": "What is the minimum SIP for HDFC Small Cap Fund?"})
                         ["kind"], "answer")

    def test_no_traceback_reached_the_server_log(self):
        # socketserver logs handler exceptions and closes the socket silently, so
        # an exception here is the signature of the failure mode this class exists
        # to catch.
        log = self.server_log()
        self.assertNotIn("Traceback", log, f"server logged an exception:\n{log[-1500:]}")
        self.assertNotIn("ProgrammingError", log, log[-1500:])

    def test_the_index_is_reachable_over_http(self):
        """The spec's `GET /` and the answer/refusal pair, over real HTTP.

        /healthz and /examples are covered by test_server_came_up and
        test_examples_endpoint; the page itself and the two /ask outcomes are the
        "done when" conditions, so they get their own assertions here.
        """
        with urllib.request.urlopen(f"{self.base}/", timeout=15) as r:
            self.assertEqual(r.status, 200)
            self.assertEqual(r.headers.get_content_type(), "text/html")
            page = r.read().decode("utf-8")
        self.assertIn("Hi! I answer factual questions", page)
        self.assertEqual(page.count('class="ex"'), 3)

        a = self._post({"q": "What is the exit load on HDFC Balanced Advantage Fund?"})
        self.assertEqual(a["kind"], "answer")
        self.assertIn(a["citation"], ALLOWED_URLS)
        self.assertTrue(a["last_updated"])
        self.assertIn("No investment advice", a["disclaimer"])

        r = self._post({"q": "Should I buy HDFC Small Cap Fund?"})
        self.assertEqual(r["kind"], "advice_refusal")
        self.assertIn(r["citation"], ALLOWED_URLS,
                      "a refusal must still carry an allow-listed link")

    def test_query_strings_do_not_break_routing(self):
        """Routing on the raw request target 404s '/healthz?x=1', which is an
        ordinary thing for a health probe to send."""
        for path, expect in (("/healthz?x=1", 200), ("/examples?cachebust=9", 200),
                             ("/?a=b", 200), ("/healthz/", 404), ("/nope?x=1", 404)):
            try:
                with urllib.request.urlopen(f"{self.base}{path}", timeout=10) as r:
                    self.assertEqual(r.status, expect, path)
            except urllib.error.HTTPError as e:
                self.assertEqual(e.code, expect, path)

    def test_json_responses_declare_their_charset(self):
        def headers_of(path):
            try:
                with urllib.request.urlopen(f"{self.base}{path}", timeout=10) as r:
                    return r.status, r.headers
            except urllib.error.HTTPError as e:
                return e.code, e.headers

        for path, expect in (("/healthz", 200), ("/examples", 200), ("/nope", 404)):
            status, hdrs = headers_of(path)
            self.assertEqual(status, expect, path)
            self.assertEqual(hdrs.get_content_type(), "application/json", path)
            self.assertEqual(hdrs.get_content_charset(), "utf-8", path)

    def test_no_cors_or_auth_headers(self):
        """The spec's pitfall: CORS and auth are a separate decision (PRD 15 Q5)."""
        for path in ("/", "/healthz", "/examples"):
            with urllib.request.urlopen(f"{self.base}{path}", timeout=10) as r:
                self.assertEqual(r.headers.get("Access-Control-Allow-Origin"), None, path)
                self.assertEqual(r.headers.get("WWW-Authenticate"), None, path)
        try:
            urllib.request.urlopen(urllib.request.Request(f"{self.base}/healthz",
                                                          method="OPTIONS"), timeout=10)
            self.fail("OPTIONS should not be implemented")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 501)

    def test_user_queries_are_never_logged(self):
        """I1. The marker is distinctive, so a single grep is conclusive."""
        marker = "ZEBRAFISH-QUERY-9d2f4a"
        self._post({"q": f"What is the exit load? {marker}"})
        self._post({"q": f"My PAN is ABCDE1234F and I ask about {marker}"})
        log = self.server_log()
        self.assertNotIn(marker, log, f"query body reached the log:\n{log[-1500:]}")
        self.assertNotIn("ABCDE1234F", log, f"PII reached the log:\n{log[-1500:]}")

    def test_request_logging_is_suppressed(self):
        """Kept separate from the body check on purpose.

        http.server logs the *request line* but never the body, so a test that
        only greps for the query passes whether or not `log_message` is
        suppressed. This asserts the suppression itself: the access log must be
        absent, leaving just the two banner lines.
        """
        log = self.server_log()
        self.assertTrue(log.strip(), "server produced no output; suppression test would be vacuous")
        for line in ("GET /healthz", "POST /ask", "GET /examples", "HTTP/1.1"):
            self.assertNotIn(line, log, f"access log not suppressed:\n{log[-1500:]}")
        self.assertEqual(
            [ln for ln in log.splitlines() if ln.strip()],
            ["MF facts assistant -> http://127.0.0.1:%d   (facts-only. No investment advice.)"
             % self.port, "Ctrl+C to stop."],
            f"unexpected console output:\n{log}")


class TestPageRendering(unittest.TestCase):
    """Phase 11: the page itself, and the escaping the spec insists on.

    `render_page` is pure and cheap, so these run without booting a server;
    the HTTP contract is covered by TestWebUIConcurrency.
    """

    def setUp(self):
        from mf_faq.ui import render_page
        self.page = render_page()

    def test_welcome_line_and_exactly_three_clickable_examples(self):
        from mf_faq import EXAMPLE_QUESTIONS, WELCOME
        self.assertIn(html.escape(WELCOME), self.page)
        self.assertEqual(len(EXAMPLE_QUESTIONS), 3)
        self.assertEqual(self.page.count('class="ex"'), 3)
        # clickable: every example is wired to a submit handler
        self.assertIn(".ex').forEach(b=>b.onclick", self.page)

    def test_facts_only_notice_is_on_the_page(self):
        from mf_faq.config import DISCLAIMER_LONG
        self.assertIn("No investment advice", self.page)
        self.assertIn(html.escape(DISCLAIMER_LONG), self.page)

    def test_client_escape_helper_exists_and_is_used_for_text_and_urls(self):
        self.assertIn("function esc(", self.page)
        self.assertIn("esc(d.text)", self.page)
        self.assertIn("esc(d.citation)", self.page)
        # quotes are escaped, so a value cannot break out of href="..."
        self.assertIn("'\"':'&quot;'", self.page)
        self.assertIn('rel="noopener"', self.page)
        self.assertIn("safeUrl(", self.page)

    def test_refusals_get_their_own_left_border(self):
        self.assertIn(".refusal{border-left", self.page)
        self.assertIn("d.kind==='answer'?'':'refusal'", self.page)

    def test_no_unsubstituted_placeholder_survives(self):
        for ph in ("{welcome}", "{examples}", "{disclaimer}", "{examples_json}"):
            self.assertNotIn(ph, self.page, ph)

    def test_json_for_script_cannot_close_the_script_element(self):
        from mf_faq.ui import _json_for_script
        hostile = ["</script><img src=x onerror=alert(1)>", "a & b < c > d",
                   "sep\u2028par", 'quote " and backslash \\']
        dumped = _json_for_script(hostile)
        for bad in ("</script", "<", ">", "&", "\u2028", "\u2029"):
            self.assertNotIn(bad, dumped, f"{bad!r} survived into the script block: {dumped}")
        # ...and it still parses back to exactly what went in
        self.assertEqual(json.loads(dumped), hostile)

    def test_a_hostile_example_cannot_inject_markup(self):
        from mf_faq import ui
        saved = ui.EXAMPLE_QUESTIONS
        try:
            ui.EXAMPLE_QUESTIONS = ["</script><img src=x onerror=alert(1)>"]
            page = ui.render_page()
        finally:
            ui.EXAMPLE_QUESTIONS = saved
        self.assertNotIn("<img src=x", page)
        # the visible button is html-escaped, the script payload is \u-escaped
        self.assertIn("&lt;/script&gt;", page)
        self.assertIn("\\u003c/script\\u003e", page)
        # the script element is still terminated exactly once
        self.assertEqual(page.lower().count("</script>"), 1)

    def test_page_inlines_its_own_css_and_js_from_the_web_assets(self):
        """The markup, styling and behaviour are separate files under
        mf_faq/web/, inlined into one response. If an asset is missing the page
        must fail loudly at render time rather than serve a half-styled shell
        that still answers questions."""
        from mf_faq.ui import ASSET_DIR, render_page
        for name in ("index.html", "app.css", "app.js"):
            self.assertTrue((ASSET_DIR / name).is_file(), f"missing UI asset {name}")
        page = render_page()
        self.assertIn("--acc", page)          # from app.css
        self.assertIn("form.requestSubmit", page)  # from app.js

    def test_a_literal_script_close_in_the_js_cannot_truncate_the_page(self):
        """The HTML parser ends a <script> at the first `</script` it sees,
        whatever the JS thinks. A string or comment containing one would
        otherwise truncate the script silently -- no error, just a dead UI."""
        from mf_faq import ui
        saved = ui._read
        try:
            ui._read = lambda name: ('var s="</script>";' if name == "app.js"
                                     else saved(name))
            page = ui.render_page()
        finally:
            ui._read = saved
        self.assertEqual(page.lower().count("</script>"), 1)
        self.assertIn("<\\/script>", page)

    def test_page_shows_both_dates_and_never_one_in_place_of_the_other(self):
        """The regression behind the current UI: an answer served out of a stale
        cache showed only the source's date and so read as though it were
        current. Both stamps are rendered, side by side."""
        from mf_faq.ui import render_page
        page = render_page()
        self.assertIn("Answer prepared on:", page)
        self.assertIn("Last updated from sources:", page)
        # the source line is conditional on the answer actually having one
        self.assertIn("d.freshness_from_source && d.last_updated", page)

    def test_page_states_what_the_corpus_cannot_answer(self):
        """Sector and market-cap splits are absent from every source, and the
        pipeline will still return some other fact for them. The UI says so up
        front instead of letting a user discover it by being misled."""
        from mf_faq.ui import render_page
        page = render_page()
        self.assertIn("sector or market-cap split", page)

    def test_page_loads_nothing_from_the_network(self):
        """No CDN, no webfont, no analytics: the tool has to work on a machine
        that cannot reach the internet, which is the corpus's own constraint."""
        import re
        from mf_faq.ui import render_page
        page = render_page()
        self.assertEqual(re.findall(r'(?:src|href)="https?://', page), [])
        self.assertNotIn("//fonts.", page)

    def test_csp_forbids_remote_content_and_framing(self):
        from mf_faq.ui import CSP
        self.assertIn("default-src 'none'", CSP)
        self.assertIn("connect-src 'self'", CSP)
        self.assertIn("frame-ancestors 'none'", CSP)
        self.assertNotIn("*", CSP.replace("'self'", "").replace("'none'", "")
                         .replace("'unsafe-inline'", ""))

    def test_non_string_query_degrades_instead_of_erroring(self):
        """`{"q": 123}` is valid JSON, so it reached the pipeline as an int and
        raised downstream, answering a bad request with a 500. Anything that is
        not a string is not a question."""
        import json as _json
        import socket
        import threading
        import time
        import urllib.request
        from mf_faq.pipeline import FAQPipeline
        from mf_faq.ui import serve

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        pipe = FAQPipeline()
        thread = threading.Thread(target=serve,
                                  kwargs={"host": "127.0.0.1", "port": port, "pipe": pipe},
                                  daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{port}"
        try:
            for _ in range(80):
                time.sleep(0.25)
                try:
                    urllib.request.urlopen(base + "/healthz", timeout=5).read()
                    break
                except OSError:
                    continue
            for payload in ({"q": 123}, {"q": None}, {"q": ["a"]}, {"q": {"a": 1}}, {}):
                rq = urllib.request.Request(f"{base}/ask",
                                            data=_json.dumps(payload).encode(),
                                            headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(rq, timeout=60) as r:
                    self.assertEqual(r.status, 200, payload)
                    d = _json.loads(r.read())
                self.assertEqual(d["kind"], "empty", payload)
        finally:
            pipe.close()


class TestSentenceCapIsPinned(unittest.TestCase):
    """The spec's non-vacuity check for the sentence cap.

    Phase 12 asks for `MAX_SENTENCES = 9` to make the suite fail. It does not,
    and the reason is worth recording: the golden answers are two or three
    sentences because the corpus does not offer more, so raising the cap changes
    nothing observable. Without a test that exercises the cap directly, the
    constant is free to drift and the mutation the spec names stays silent.

    So the cap is pinned twice: once as the configured value, and once
    behaviourally, by trimming an input long enough that a raised cap would
    return it untruncated.
    """

    def test_configured_cap_is_three(self):
        from mf_faq import config
        self.assertEqual(config.MAX_SENTENCES, 3)

    def test_no_module_shadows_the_cap(self):
        """Both modules did `from .config import MAX_SENTENCES`, so a local
        override in one of them would silently disagree with the other. Editing
        config.py is caught by the test above; shadowing it at the point of use
        is caught here."""
        from mf_faq import config, synthesis
        self.assertEqual(synthesis.MAX_SENTENCES, config.MAX_SENTENCES)
        from mf_faq import eval as ev
        self.assertEqual(ev.MAX_SENTENCES, config.MAX_SENTENCES)

    def test_trim_actually_truncates_to_the_cap(self):
        from mf_faq.synthesis import _trim_to_sentences
        long_text = " ".join(f"Sentence number {i} carries a fact." for i in range(1, 7))
        self.assertEqual(_trim_to_sentences(long_text, 3).count("Sentence number"), 3)
        # with the cap raised to 9 the same call returns all six
        self.assertEqual(_trim_to_sentences(long_text, 9).count("Sentence number"), 6)

    def test_a_trimmed_answer_ends_as_a_sentence(self):
        from mf_faq.synthesis import _trim_to_sentences
        trimmed = _trim_to_sentences("One, two, three, four, five, six.", 3)
        self.assertTrue(trimmed.endswith("."), trimmed)
        self.assertFalse(trimmed.endswith(","), trimmed)

    def test_golden_answers_respect_the_cap(self):
        from mf_faq.config import MAX_SENTENCES
        from mf_faq.eval import GOLDEN
        from mf_faq.parsing import split_sentences
        from mf_faq.pipeline import FAQPipeline
        p = FAQPipeline()
        try:
            for case in GOLDEN:
                n = len(split_sentences(p.ask(case.q).text))
                self.assertLessEqual(n, MAX_SENTENCES, f"{case.q} -> {n} sentences")
        finally:
            p.close()


class TestGradersCanFail(unittest.TestCase):
    """Phase 12: the graders themselves, fed answers that must not pass.

    `evaluate()` reports 28/28, which is only meaningful if a bad answer would
    score badly. Each check below is exercised directly against a hand-built
    Answer, so a future edit that quietly turns a check into a no-op is caught
    here rather than in production.
    """

    def _answer(self, **kw):
        from mf_faq.guardrails import Answer
        from mf_faq.eval import LARGE as allowlisted
        base = dict(kind="answer", text="The expense ratio is 1.03%.",
                    citation=allowlisted,
                    disclaimer="Facts-only. No investment advice.",
                    last_updated="2026-09-25")
        base.update(kw)
        return Answer(**base)

    def test_a_non_allowlisted_domain_fails_citation_allowlisted(self):
        from mf_faq.eval import Case, _grade_answer
        ans = self._answer(citation="http://evil.example.com/mutual-funds/fake")
        checks = _grade_answer(ans, Case("q", expect_substrings=["1.03"]))
        self.assertFalse(checks["citation_allowlisted"])

    def test_a_url_in_the_answer_body_fails_one_citation(self):
        from mf_faq.eval import Case, _grade_answer
        ans = self._answer(text="The ratio is 1.03%. See https://example.com/x")
        self.assertFalse(_grade_answer(ans, Case("q"))["one_citation"])

    def test_a_too_long_answer_fails_sentence_cap(self):
        from mf_faq.eval import Case, _grade_answer
        ans = self._answer(text=" ".join(f"Fact {i} here." for i in range(6)))
        self.assertFalse(_grade_answer(ans, Case("q"))["sentence_cap"])

    def test_a_missing_disclaimer_or_stamp_fails(self):
        from mf_faq.eval import Case, _grade_answer
        self.assertFalse(_grade_answer(self._answer(disclaimer=""), Case("q"))["disclaimer"])
        self.assertFalse(_grade_answer(self._answer(last_updated=""), Case("q"))["freshness"])

    def test_a_refusal_graded_as_an_answer_fails_kind(self):
        from mf_faq.eval import Case, _grade_refusal
        ans = self._answer(kind="answer")
        self.assertFalse(_grade_refusal(ans, Case("q", kind="advice_refusal"))["refused"])

    def test_only_the_canonical_template_passes_is_template(self):
        from mf_faq.eval import Case, _grade_refusal
        from mf_faq.guardrails import REFUSAL_BODIES
        canonical = REFUSAL_BODIES["advice_refusal"]
        ok = _grade_refusal(self._answer(kind="advice_refusal", text=canonical),
                            Case("q", kind="advice_refusal"))
        self.assertTrue(ok["is_template"])
        for mutated in (canonical + " In my opinion you should buy it.",
                        canonical.replace("should", "ought to"),
                        canonical.strip().lower(),
                        "You should consult an adviser."):
            bad = _grade_refusal(self._answer(kind="advice_refusal", text=mutated),
                                 Case("q", kind="advice_refusal"))
            self.assertFalse(bad["is_template"], mutated[:50])

    def test_echoed_pii_is_caught_for_every_pii_case(self):
        """Derived from the case query, not two hardcoded literals -- the folio
        case used to have nothing to assert."""
        from mf_faq.eval import POLICY, _echoed_pii
        pii_cases = [c for c in POLICY if c.kind == "pii_refusal"]
        self.assertEqual(len(pii_cases), 3)
        for case in pii_cases:
            self.assertTrue(_echoed_pii(case, f"echoing {case.q}"),
                            f"nothing detected to echo for {case.q!r}")
        # and a refusal that echoes nothing is clean
        self.assertEqual(_echoed_pii(pii_cases[0], "I cannot help with that."), [])

    def test_a_performance_claim_in_a_refusal_fails(self):
        """Every natural phrasing, not just the one the old regex happened to
        match: a grader that only catches "18.4% CAGR" misses "3 year CAGR was
        18.4%", which is the more natural way to say it."""
        from mf_faq.eval import Case, _grade_refusal
        from mf_faq.guardrails import REFUSAL_BODIES
        canonical = REFUSAL_BODIES["performance_refusal"]
        for leak in ("Its 18.4% CAGR was strong.",
                     "Its 3 year CAGR was 18.4%.",
                     "It returned 18.4% last year.",
                     "CAGR of 18.4%.",
                     "The exit load is 1%."):
            checks = _grade_refusal(
                self._answer(kind="performance_refusal", text=canonical + " " + leak),
                Case("q", kind="performance_refusal"))
            self.assertFalse(checks["no_perf_claim"], leak)
        # the untouched template is still clean
        self.assertTrue(_grade_refusal(
            self._answer(kind="performance_refusal", text=canonical),
            Case("q", kind="performance_refusal"))["no_perf_claim"])

    def test_a_case_passes_only_when_every_check_passes(self):
        from mf_faq.eval import Report
        rep = Report()
        rep.add("all good", {"a": True, "b": True})
        self.assertEqual((rep.passed, rep.failed), (1, 0))
        rep.add("one bad", {"a": True, "b": False})
        self.assertEqual((rep.passed, rep.failed), (1, 1))
        self.assertEqual(rep.accuracy, 0.5)
        self.assertIn("b", rep.notes[0])


class TestAliasPrecedenceIsReal(unittest.TestCase):
    """`test_scheme_resolution_prefers_longest_alias` exists, but every query in
    it matches exactly one alias, so it passes just as well when resolution takes
    the *first* hit instead of the longest. Mutating the comparison to
    `if best is None` left the suite green, which is how the gap was found.

    The discriminating case needs a query where two aliases of *different* length
    and *different* scheme both match, and the longer one is not the one reached
    first. SCHEMES iterates large_cap, flexi_cap, elss, small_cap,
    balanced_advantage, and aliases within a scheme in list order, so a query
    mentioning "large cap" (9 chars, reached first) and "hdfc flexi cap fund"
    (19 chars, reached later) is decided by the length rule alone.
    """

    def test_longest_alias_beats_an_earlier_shorter_one(self):
        from mf_faq.intent import detect_scheme
        key, hits = detect_scheme("HDFC Large Cap or HDFC Flexi Cap Fund?")
        self.assertIn("hdfc flexi cap fund", hits)
        self.assertIn("large cap", hits)
        self.assertEqual(key, "flexi_cap",
                         "the longer alias must win, not the first one reached")

    def test_a_single_alias_still_resolves(self):
        from mf_faq.intent import detect_scheme
        self.assertEqual(detect_scheme("expense ratio of HDFC Flexi Cap Fund")[0], "flexi_cap")
        self.assertEqual(detect_scheme("HDFC ELSS Tax Saver Fund lock in")[0], "elss")
        self.assertEqual(detect_scheme("What is a mutual fund?")[0], None)

    def test_every_cross_scheme_alias_pair_is_decided_by_length(self):
        """Systematic version: for any query matching two schemes, the resolved
        key must be the one owning the longest alias, whatever the iteration
        order."""
        from mf_faq.intent import detect_scheme
        from mf_faq.sources import SCHEMES
        checked = 0
        for key_a, meta_a in SCHEMES.items():
            for key_b, meta_b in SCHEMES.items():
                if key_a >= key_b:
                    continue
                # pair the longest alias of A with the shortest of B, so a
                # length rule and a first-seen rule disagree
                longest = max(meta_a["aliases"], key=len)
                shortest = min(meta_b["aliases"], key=len)
                if len(longest) <= len(shortest):
                    continue
                q = f"Tell me about {longest} and {shortest}."
                got, _ = detect_scheme(q)
                self.assertEqual(got, key_a,
                                 f"{q!r}: expected {key_a} (owns {longest!r}), got {got}")
                checked += 1
        self.assertGreater(checked, 0, "no cross-scheme pair could discriminate")


class TestMMRDiversityIsReal(unittest.TestCase):
    """The existing MMR test checks performance -- that stored vectors are reused
    rather than re-projected -- not that MMR *does* anything. Replacing the
    selection value with plain relevance left the suite green.

    So: two near-identical high-scoring chunks and one orthogonal chunk that
    scores slightly lower. Relevance alone takes both duplicates; MMR must take
    the first and then trade the second for the distinct one.
    """

    def _stub(self, matrix, lam=0.72):
        from types import SimpleNamespace
        from mf_faq.config import CONFIG
        return SimpleNamespace(dense=SimpleNamespace(matrix=matrix),
                               cfg=SimpleNamespace(mmr_lambda=lam))

    def _scored(self, cid, final):
        from mf_faq.chunking import Chunk
        from mf_faq.retrieval import Scored
        return Scored(chunk=Chunk(chunk_id=cid, doc_id=cid, source_id="src", text=cid,
                                  kind="fact", topic="t", scheme_key="s", url="https://example.com"),
                      final=final)

    def test_a_near_duplicate_is_traded_for_a_distinct_chunk(self):
        from mf_faq.retrieval import Retriever
        # a and b are near-identical; c is orthogonal but scores a little lower
        matrix = {"a": [1.0, 0.0], "b": [0.99, 0.01], "c": [0.0, 1.0]}
        stub = self._stub(matrix)
        pool = [self._scored("a", 1.00), self._scored("b", 0.99), self._scored("c", 0.98)]
        picked = [s.chunk.chunk_id for s in Retriever._mmr(stub, pool, 2, None)]
        self.assertEqual(picked, ["a", "c"],
                         "the second pick must be the distinct chunk, not the duplicate")

    def test_relevance_alone_would_take_the_duplicate(self):
        """The control: with the redundancy term removed the same pool yields the
        duplicate. If this ever agrees with the test above, MMR is not running."""
        from types import SimpleNamespace
        matrix = {"a": [1.0, 0.0], "b": [0.99, 0.01], "c": [0.0, 1.0]}
        pool = [self._scored("a", 1.00), self._scored("b", 0.99), self._scored("c", 0.98)]
        lam = 0.72
        relevance_only = sorted(pool, key=lambda s: -s.final)[:2]
        self.assertEqual([s.chunk.chunk_id for s in relevance_only], ["a", "b"])
        # and MMR with lam -> 1.0 degrades to exactly that, which is why the
        # mutation had to be expressed in the selection value, not the constant
        stub = self._stub(matrix, lam=1.0)
        from mf_faq.retrieval import Retriever
        self.assertEqual([s.chunk.chunk_id for s in Retriever._mmr(stub, pool, 2, None)],
                         ["a", "b"])

    def test_a_recorded_penalty_reflects_real_redundancy(self):
        from mf_faq.retrieval import Retriever
        matrix = {"a": [1.0, 0.0], "b": [0.99, 0.01], "c": [0.0, 1.0]}
        stub = self._stub(matrix)
        # Only two near-identical candidates and k=2, so the second pick is
        # forced to be the duplicate -- that is where a penalty should appear.
        pool = [self._scored("a", 1.00), self._scored("b", 0.99)]
        picked = Retriever._mmr(stub, pool, 2, None)
        self.assertEqual([s.chunk.chunk_id for s in picked], ["a", "b"])
        self.assertEqual(picked[0].penalty, 0.0,
                         "the first pick has nothing to compare against yet")
        self.assertGreater(picked[1].penalty, 0.0,
                           "a pick that is a near-duplicate must carry a penalty")
        # and the orthogonal chunk carries none
        pool3 = [self._scored("a", 1.00), self._scored("b", 0.99), self._scored("c", 0.98)]
        self.assertEqual(Retriever._mmr(stub, pool3, 2, None)[1].penalty, 0.0)

    def test_mmr_still_keeps_the_most_relevant_first(self):
        from mf_faq.retrieval import Retriever
        matrix = {"a": [1.0, 0.0], "b": [0.99, 0.01], "c": [0.0, 1.0]}
        stub = self._stub(matrix)
        pool = [self._scored("a", 1.00), self._scored("b", 0.99), self._scored("c", 0.98)]
        self.assertEqual(Retriever._mmr(stub, pool, 1, None)[0].chunk.chunk_id, "a")

    def test_mmr_never_selects_the_same_chunk_twice(self):
        from mf_faq.retrieval import Retriever
        matrix = {"a": [1.0, 0.0], "b": [0.99, 0.01], "c": [0.0, 1.0]}
        stub = self._stub(matrix)
        pool = [self._scored("a", 1.00), self._scored("b", 0.99), self._scored("c", 0.98)]
        for k in (1, 2, 3, 4):
            picked = [s.chunk.chunk_id for s in Retriever._mmr(stub, pool, k, None)]
            self.assertEqual(len(picked), len(set(picked)), f"k={k}")
            self.assertLessEqual(len(picked), len(pool))


class TestServeRefusesToHijackALivePort(unittest.TestCase):
    """Windows lets two sockets bind one port.

    `HTTPServer.allow_reuse_address` is True, and SO_REUSEADDR on Windows means
    "ignore TIME_WAIT" *and* "let a second socket bind a port that already has a
    live listener" -- unlike Unix, where the duplicate bind fails with
    EADDRINUSE. So without a guard a second `run.py serve` starts, prints its
    banner, and connections land on whichever socket the kernel picks: the user
    restarts the server and silently keeps talking to the stale one. The failure
    is a hang, not a crash, so the test has to assert the refusal explicitly.
    """

    def test_second_server_on_the_same_port_refuses_with_an_actionable_error(self):
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        first = subprocess.Popen(
            [sys.executable, "-u", str(root / "run.py"), "serve", "--port", str(port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(root))
        log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False,
                                          encoding="utf-8", errors="replace")
        try:
            for _ in range(80):
                time.sleep(0.25)
                if first.poll() is not None:
                    self.fail("first server exited early")
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2).read()
                    break
                except Exception:                                  # noqa: BLE001
                    continue
            else:
                self.fail("first server never came up")

            # must terminate promptly with a non-zero code, not block forever
            try:
                second = subprocess.run(
                    [sys.executable, str(root / "run.py"), "serve", "--port", str(port)],
                    text=True, cwd=str(root),
                    timeout=60, stdout=log, stderr=subprocess.STDOUT)
            except subprocess.TimeoutExpired:
                self.fail("a second serve on a live port hung instead of refusing")
            self.assertEqual(second.returncode, 1)
            with open(log.name, encoding="utf-8", errors="replace") as fh:
                out = fh.read()
            self.assertIn("already in use", out)
            self.assertIn("--port", out)
            # it must not pretend to have started
            self.assertNotIn("MF facts assistant ->", out)
        finally:
            first.terminate()
            try:
                first.wait(timeout=15)
            except Exception:                                      # noqa: BLE001
                first.kill()
            log.close()

    def test_a_free_port_still_starts_normally(self):
        """The guard must not refuse a port nothing is listening on."""
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False,
                                          encoding="utf-8", errors="replace")
        p = subprocess.Popen([sys.executable, "-u", str(root / "run.py"), "serve",
                              "--port", str(port)],
                             stdout=log, stderr=subprocess.STDOUT, cwd=str(root))
        try:
            ok = False
            for _ in range(80):
                time.sleep(0.25)
                if p.poll() is not None:
                    break
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2).read()
                    ok = True
                    break
                except Exception:                                  # noqa: BLE001
                    continue
            self.assertTrue(ok, "serve refused a free port")
        finally:
            p.terminate()
            try:
                p.wait(timeout=15)
            except Exception:                                      # noqa: BLE001
                p.kill()
            log.close()


class TestDeliverableRegeneration(unittest.TestCase):
    """Phase 13: the shipped artifacts must be provably generated, not typed.

    Each test here pins a defect that was actually present in these files when
    this class was written, not a hypothetical one.
    """

    @classmethod
    def setUpClass(cls):
        cls.pipe = FAQPipeline()

    @classmethod
    def tearDownClass(cls):
        cls.pipe.close()

    # -- samples ----------------------------------------------------------
    def test_samples_regenerate_byte_identically(self):
        # `run.py samples` used to stamp the files with datetime.now() and to
        # record per-answer latency, so every run produced a different file.
        # That defeats the point of a generated deliverable: a diff can no
        # longer mean "the answers changed". The stamp is the index build time
        # and timings live in bench.py, which makes the artifact a pure
        # function of the built index.
        first = samples_gen.build(self.pipe)
        second = samples_gen.build(self.pipe)
        self.assertEqual(first[1], second[1], "sample_qa.md is not reproducible")
        self.assertEqual(first[0], second[0], "sample_qa.json is not reproducible")

    def test_committed_samples_match_the_live_pipeline(self):
        # Catches the failure mode the generator exists to prevent: someone
        # edits the answer text, or an answer changes, and the committed file
        # silently no longer describes the system.
        #
        # This compares in-memory render output against the bytes on disk. An
        # earlier version called generate() and then compared the file to
        # itself, which regenerated the very file it was meant to check and so
        # could never fail -- mutation testing confirmed it missed a hand-edit.
        json_text, md_text = samples_gen.build(self.pipe)
        committed_md = (ROOT / "samples" / "sample_qa.md").read_text(encoding="utf-8")
        committed_json = (ROOT / "samples" / "sample_qa.json").read_text(encoding="utf-8")
        if committed_md != md_text:
            self.fail("samples/sample_qa.md has drifted from the live pipeline; "
                      "re-run `python run.py samples`")
        if committed_json != json_text:
            self.fail("samples/sample_qa.json has drifted from the live pipeline; "
                      "re-run `python run.py samples`")

    def test_sample_artifact_carries_no_timing(self):
        data = json.loads((ROOT / "samples" / "sample_qa.json").read_text(encoding="utf-8"))
        for row in data["samples"]:
            self.assertNotIn("latency_ms", row)
        self.assertIn("index_built_at", data)

    def test_ten_samples_cover_every_required_case(self):
        data = json.loads((ROOT / "samples" / "sample_qa.json").read_text(encoding="utf-8"))
        rows = data["samples"]
        self.assertEqual(len(rows), 10)
        kinds = [r["kind"] for r in rows]
        self.assertEqual(kinds.count("answer"), 8, "spec: 8 factual samples")
        self.assertEqual(kinds.count("advice_refusal"), 1, "spec: 1 advice sample")
        self.assertEqual(kinds.count("pii_refusal"), 1, "spec: 1 PII sample")
        for r in rows:
            self.assertTrue(r["citation"].startswith("http"), r["query"])
            self.assertIn(r["citation"], ALLOWED_URLS, r["query"])
            self.assertLessEqual(r["sentence_count"], MAX_SENTENCES)
            self.assertTrue(r["guards_passed"])

    def test_pii_sample_documents_the_refusal_without_leaking_into_answers(self):
        # The sample quotes its triggering query verbatim on purpose. What must
        # never happen is the PAN reaching an answer, citation or guard record.
        data = json.loads((ROOT / "samples" / "sample_qa.json").read_text(encoding="utf-8"))
        pii_rows = [r for r in data["samples"] if r["kind"] == "pii_refusal"]
        self.assertEqual(len(pii_rows), 1)
        row = pii_rows[0]
        self.assertIn("ABCDE1234F", row["query"], "the case must be recognisable")
        for field in ("answer", "citation", "citation_title", "disclaimer"):
            self.assertNotIn("ABCDE1234F", str(row.get(field) or ""), field)

    # -- DISCLAIMER.txt ---------------------------------------------------
    def test_disclaimer_file_matches_the_code_verbatim(self):
        # The Phase 8 PII refusal was widened to cover UPI/IFSC bank details
        # and DISCLAIMER.txt was never regenerated, so the file a reviewer
        # reads under-stated what the assistant actually refuses. Prose is
        # hard-wrapped, so compare whitespace-normalised.
        text = re.sub(r"\s+", " ", (ROOT / "DISCLAIMER.txt").read_text(encoding="utf-8"))
        expected = {
            "short form": DISCLAIMER,
            "long form": DISCLAIMER_LONG,
            "welcome line": WELCOME,
            "freshness line": LAST_UPDATED_PREFIX + "<date on the source page>",
        }
        for kind, body in REFUSAL_BODIES.items():
            expected[f"refusal {kind}"] = body
        for i, q in enumerate(EXAMPLE_QUESTIONS, 1):
            expected[f"example {i}"] = q
        for name, s in expected.items():
            with self.subTest(string=name):
                # assertTrue, not assertIn: assertIn dumps the whole file into
                # the failure message, which is unreadable for a 15 KB README.
                self.assertTrue(re.sub(r"\s+", " ", s).strip() in text,
                                f"DISCLAIMER.txt does not carry {name} verbatim")

    def test_disclaimer_file_documents_where_each_string_is_used(self):
        text = (ROOT / "DISCLAIMER.txt").read_text(encoding="utf-8")
        for token in ("config.py", "ui.py", "guardrails.py"):
            self.assertIn(token, text, f"DISCLAIMER.txt must say where {token} renders it")

    # -- README -----------------------------------------------------------
    def test_readme_links_every_deliverable(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for artefact in ("DISCLAIMER.txt", "samples/sample_qa.md",
                         "samples/sample_qa.json", "SOURCES.md", "SOURCES.csv",
                         "bench.py"):
            with self.subTest(artefact=artefact):
                self.assertIn(artefact, readme)

    def test_readme_states_the_known_deviations_plainly(self):
        # D-1..D-4 are the accepted departures from the brief; the README is
        # where a reviewer will look, so they cannot be left in a design note.
        # Scoped to the deviations table: a bare substring search passed even
        # when the table row was renamed, because the ids also appear in prose.
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        m = re.search(r"(?ms)^## Deviations from the brief\s*\n(.*?)(?=^## )", readme)
        self.assertIsNotNone(m, "README.md has no '## Deviations from the brief' section")
        table = m.group(1)
        for token in ("D-1", "D-2", "D-3", "D-4"):
            with self.subTest(deviation=token):
                self.assertRegex(table, rf"\*\*{token}\*\*",
                                 f"{token} is not a labelled row in the deviations table")


if __name__ == "__main__":
    unittest.main(verbosity=2)
