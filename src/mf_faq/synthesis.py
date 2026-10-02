"""STAGE 7 - Answer synthesis.

Two generators behind one interface:

  * `LLMSynthesiser`     - used when an API key is configured. Sees a compact,
                           numbered CONTEXT and must return JSON with an
                           answer plus a citation copied from the context.
  * `ExtractiveSynthesiser` - the default, zero-dependency, fully offline path.
    It is *not* a language model: it composes answers deterministically from
    the `answer_hint` written at chunking time plus verified retrieved text.
    This is a deliberate design choice - a facts-only assistant whose numbers
    come from a source of record is safer when it cannot paraphrase a number.

Both feed the same post-processing contract, enforced in stage 8.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date

from . import llm
from .config import CONFIG, MAX_SENTENCES
from .intent import Decision
from .parsing import split_sentences
from .retrieval import RetrievalResult, Scored
from .sources import SCHEMES

#: Topics answered with a composed, navigational answer rather than a number.
PROCEDURAL = {"statement_download", "capital_gains", "tax_doc"}

#: Which companion fact may be added as sentence 2, per asked topic. Padding an
#: answer with an on-topic-but-unrelated number ("... Scheme details: Equity")
#: reads as filler, so the default is deliberately empty.
COMPANION_FIELDS: dict[str, set[str]] = {
    # The two SIP minimums are the same family of number, so the second one is
    # useful. Scheme-metadata fields get no companion: "the fund manager is X.
    # The ISIN is Y." answers nothing the user asked.
    "min_sip": {"min_lumpsum_amount", "min_additional_amount", "sip_allowed"},
    "min_lumpsum": {"min_sip_amount"},
}

SCHEME_DISPLAY = {k: v["label"] for k, v in SCHEMES.items()}


@dataclass
class Draft:
    answer: str
    citation: str
    last_updated: str
    used_llm: bool = False
    notes: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    #: True only when `last_updated` came from the chunk's own source date.
    freshness_from_source: bool = True


def _draft_date(chunk) -> tuple[str, bool]:
    """Return (date, came_from_source) for a chunk.

    A chunk with no source date is authored text (a procedure). Inventing a date
    for it here would silently turn "no source currency" into a freshness claim.
    """
    if chunk.last_updated:
        return chunk.last_updated, True
    return date.today().isoformat(), False


# ------------------------------------------------------------------ helpers

def _trim_to_sentences(text: str, n: int) -> str:
    sents = split_sentences(text)
    if len(sents) <= n:
        return " ".join(sents).strip()
    out = " ".join(sents[:n]).strip()
    return out.rstrip(",;") + "."


def _schema_label(decision: Decision) -> str | None:
    if not decision.scheme_key:
        return None
    return SCHEME_DISPLAY[decision.scheme_key]


def _evidence(result: RetrievalResult) -> list[dict]:
    """One evidence schema for all three generators.

    The three paths used to emit three different shapes -- `_fact` carried
    `topic` and `reasons`, `_procedural` only `kind`, `_pointer` neither. Any
    consumer that reads `e["reasons"]` therefore worked for a fact answer and
    raised `KeyError` for a riskometer or statement answer, which is exactly the
    documented verification command's access pattern.
    """
    return [{"chunk_id": s.chunk.chunk_id, "score": round(s.final, 4),
             "kind": s.chunk.kind, "topic": s.chunk.topic, "url": s.chunk.url,
             "reasons": s.reasons} for s in result.hits]


# ------------------------------------------------------------------ extractive

class ExtractiveSynthesiser:
    """Deterministic composer. No model, no hallucination surface."""

    name = "extractive"

    def build_context(self, result: RetrievalResult) -> str:
        lines = []
        for i, s in enumerate(result.hits, 1):
            c = s.chunk
            lines.append(
                f"[{i}] ({c.kind}/{c.topic}) {c.text}\n"
                f"    HINT: {c.answer_hint}\n"
                f"    URL: {c.url}\n    UPDATED: {c.last_updated}"
            )
        return "\n".join(lines)

    def generate(self, result: RetrievalResult) -> Draft:
        d = result.decision
        notes: list[str] = []
        if d.primary_topic in PROCEDURAL:
            return self._procedural(result)
        if d.primary_topic in {"riskometer", "expense_ratio_note"}:
            return self._pointer(result)
        return self._fact(result)

    # -- numeric fact -------------------------------------------------------
    def _fact(self, result: RetrievalResult) -> Draft:
        d = result.decision
        facts = [s for s in result.hits if s.chunk.kind == "fact"]
        if not facts:
            facts = result.hits
        if not facts:
            return Draft(answer="I could not find this in the collected sources.",
                         citation="", last_updated="", used_llm=False,
                         notes=["empty_retrieval"])

        # A field hint from stage 5 is a hard constraint, not a preference.
        # "Who manages X" and "the fund manager of X" do not overlap
        # lexically, and several scheme_meta facts share the topic, so a soft
        # score boost alone lets the wrong fact (RTA, ISIN) win on term count.
        hints = set(d.field_hints)
        if hints:
            hinted = [s for s in facts if s.chunk.meta.get("field") in hints]
            if hinted:
                facts = hinted
            else:
                facts = sorted(
                    facts,
                    key=lambda s: s.chunk.meta.get("field") in hints,
                    reverse=True)

        best = facts[0]
        chunk = best.chunk

        # Sentence 1: the chunk's own first sentence. It is authored to be
        # self-contained (scheme label + unit), which is what makes a cited
        # fact answer trustworthy without a language model in the loop.
        sents = split_sentences(chunk.text)
        sent1 = sents[0].rstrip(".") + "." if sents else chunk.answer_hint.rstrip(".") + "."

        # Sentence 2: only a companion field that belongs to the asked topic.
        allowed = COMPANION_FIELDS.get(d.primary_topic or "", set())
        sent2 = ""
        for s in result.hits[1:]:
            c = s.chunk
            field_name = c.meta.get("field")
            if (c.kind == "fact" and c.scheme_key == chunk.scheme_key
                    and field_name in allowed):
                extra = split_sentences(c.text)[0].rstrip(".") + "."
                if extra.lower() != sent1.lower():
                    sent2 = extra
                    break

        # The lock-in and exit-load fact texts already carry their qualifier as
        # sentence 2; keep it rather than padding with an unrelated fact.
        if not sent2 and len(sents) > 1 and d.primary_topic in {"lock_in", "exit_load"}:
            sent2 = sents[1].rstrip(".") + "."

        answer = " ".join(x for x in (sent1, sent2) if x)
        lu, lu_from_source = _draft_date(chunk)
        return Draft(
            answer=_trim_to_sentences(answer, MAX_SENTENCES),
            citation=chunk.url,
            last_updated=lu,
            freshness_from_source=lu_from_source,
            used_llm=False,
            evidence=_evidence(result),
        )

    # -- procedure ----------------------------------------------------------
    def _procedural(self, result: RetrievalResult) -> Draft:
        procs = [s for s in result.hits if s.chunk.kind == "procedure"]
        pick = procs[0] if procs else (result.hits[0] if result.hits else None)
        if pick is None:
            return Draft(answer="I could not find this in the collected sources.",
                         citation="", last_updated="", notes=["empty_retrieval"])
        c = pick.chunk
        lu, lu_from_source = _draft_date(c)
        return Draft(
            answer=_trim_to_sentences(c.text, MAX_SENTENCES),
            citation=c.url, last_updated=lu,
            freshness_from_source=lu_from_source,
            used_llm=False,
            evidence=_evidence(result),
        )

    # -- pointer (riskometer / authoritative-doc) ---------------------------
    def _pointer(self, result: RetrievalResult) -> Draft:
        procs = [s for s in result.hits if s.chunk.kind == "procedure"]
        if not procs:
            procs = result.hits
        if not procs:
            return Draft(answer="I could not find this in the collected sources.",
                         citation="", last_updated="", notes=["empty_retrieval"])
        c = procs[0].chunk
        lu, lu_from_source = _draft_date(c)
        # The procedure chunk is already written as a self-contained,
        # non-numeric pointer ("That level is not in the source pages
        # collected for this build. The SEBI riskometer level ... factsheet"),
        # so it is quoted rather than re-templated.
        return Draft(
            answer=_trim_to_sentences(c.text, MAX_SENTENCES),
            citation=c.url, last_updated=lu,
            freshness_from_source=lu_from_source,
            used_llm=False,
            evidence=_evidence(result),
        )


# ------------------------------------------------------------------ LLM

class LLMSynthesiser(ExtractiveSynthesiser):
    name = "llm"

    def generate(self, result: RetrievalResult) -> Draft:
        if not llm.available():
            d = super().generate(result)
            d.notes.append("llm_unavailable_fallback")
            return d
        prompt = (
            f"QUESTION: {result.decision.query}\n\n"
            f"SCHEME: {_schema_label(result.decision) or 'any / general'}\n\n"
            f"CONTEXT:\n{self.build_context(result)[:CONFIG.max_context_chars]}\n"
        )
        try:
            raw = llm.complete(prompt, llm.SYSTEM)
        except llm.LLMUnavailable as exc:
            d = super().generate(result)
            d.notes.append(f"llm_error:{exc}")
            return d

        obj = self._parse(raw)
        if not obj:
            d = super().generate(result)
            d.notes.append("llm_unparseable_fallback")
            return d
        fallback = super().generate(result)
        return Draft(
            answer=obj["answer"].strip(),
            citation=obj.get("citation", "").strip() or fallback.citation,
            last_updated=fallback.last_updated,
            freshness_from_source=fallback.freshness_from_source,
            used_llm=True,
            evidence=fallback.evidence,
        )

    @staticmethod
    def _parse(raw: str) -> dict | None:
        # `generate` only catches LLMUnavailable, so anything raised here would
        # escape as a 500 instead of falling back. A provider that returns a
        # non-string (or a stubbed/mocked one) must degrade, not crash.
        if not isinstance(raw, str) or not raw.strip():
            return None
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            return None
        if not isinstance(obj, dict):
            return None
        # `answer` must be a non-empty *string*. A truthy non-string (123, a
        # list, a dict) used to pass this gate and then blow up on `.strip()`
        # inside `generate`, which only catches LLMUnavailable -- so a bad model
        # response became a 500 rather than the extractive fallback.
        answer = obj.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            return None
        return obj


# ------------------------------------------------------------------ factory

def get_synthesiser() -> ExtractiveSynthesiser:
    if llm.available():
        return LLMSynthesiser()
    return ExtractiveSynthesiser()
