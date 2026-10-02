"""STAGE 6 - Data retrieval.

Hybrid, explainable retrieval:

  score(chunk) = w_bm25 * bm25_norm
               + w_dense * cosine
               + scheme_match_boost
               + topic_match_boost
               + kind_prior                       (facts beat prose)
               - redundancy_penalty               (MMR)

`retrieve()` returns `RetrievalResult` objects that keep every intermediate
signal, so `explain()` can print why a chunk was chosen.  Evaluation in
stage 9 depends on being able to audit that.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import Chunk, tokenize
from .config import DB_PATH, RetrievalConfig
# The vectoriser class stays an implementation detail of `indexing`. Importing
# it here would put the Phase-5 Variant A/B choice back on this file's doorstep
# and stop the swap being contained to one module.
from .indexing import all_vectors, cosine, load_vectoriser
from .intent import Decision

FTS_TABLE = "chunks_fts"
CHUNK_COLS = "chunk_id,doc_id,source_id,url,kind,topic,scheme_key,text,answer_hint,last_updated,meta_json"

#: Facts are the highest-precision answer source; procedures are reliable but
#: only relevant to their own topic. Prose sections are the fallback.
KIND_PRIOR = {"fact": 0.06, "procedure": 0.02, "section": 0.0}

#: Topics that must not be answered with a per-scheme fact chunk.
NON_FACT_TOPICS = {"statement_download", "capital_gains", "tax_doc", "riskometer",
                   "expense_ratio_note"}

#: The three statement/tax procedures point at three different systems and are
#: mutually exclusive answers.
PROCEDURAL_GROUP = ("statement_download", "capital_gains", "tax_doc")

#: Where a procedure chunk may legitimately serve a neighbouring topic.
NEIGHBOUR_TOPICS = {
    "statement_download": {"capital_gains", "tax_doc", "scheme_meta"},
    "capital_gains": {"statement_download", "tax_doc", "scheme_meta"},
    "tax_doc": {"statement_download", "capital_gains", "scheme_meta"},
    "riskometer": {"scheme_meta", "benchmark"},
    "expense_ratio_note": {"expense_ratio", "scheme_meta"},
}


@dataclass
class Scored:
    chunk: Chunk
    bm25: float = 0.0
    dense: float = 0.0
    scheme_boost: float = 0.0
    topic_boost: float = 0.0
    field_boost: float = 0.0
    kind_prior: float = 0.0
    penalty: float = 0.0
    final: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    query: str
    decision: Decision
    hits: list[Scored]
    n_candidates: int
    index_stats: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)


# ------------------------------------------------------------------ lexical

def _fts_query(q: str) -> str:
    toks = [t for t in tokenize(q) if len(t) > 1]
    if not toks:
        return ""
    return " OR ".join(f'"{t}"' for t in toks)


def bm25_search(con: sqlite3.Connection, query: str, limit: int) -> list[tuple[str, float]]:
    match = _fts_query(query)
    if not match:
        return []
    sql = (
        f"SELECT chunk_id, bm25({FTS_TABLE}, 1.0, 0.6, 0.0) AS rank "
        f"FROM {FTS_TABLE} WHERE {FTS_TABLE} MATCH ? ORDER BY rank LIMIT ?"
    )
    try:
        rows = con.execute(sql, (match, limit)).fetchall()
    except sqlite3.OperationalError:
        return []
    # FTS5 bm25() returns a *negative* score where more negative == better.
    return [(r[0], -float(r[1])) for r in rows]


# ------------------------------------------------------------------ vector

class DenseIndex:
    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path or DB_PATH)
        self.vec = load_vectoriser(self.db_path)
        self.matrix = all_vectors(self.db_path)
        self._q_cache: dict[str, list[float]] = {}

    def query_vector(self, query: str) -> list[float]:
        v = self._q_cache.get(query)
        if v is None:
            v = self.vec.transform_one(query)
            # Bounded so a long-running server cannot grow this without limit.
            if len(self._q_cache) > 256:
                self._q_cache.clear()
            self._q_cache[query] = v
        return v

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        if not self.matrix:
            return []
        qv = self.query_vector(query)
        scored = [(cid, cosine(qv, vec)) for cid, vec in self.matrix.items()]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:limit]


# ------------------------------------------------------------------ main

class Retriever:
    def __init__(self, db_path: Path | None = None, cfg: RetrievalConfig | None = None):
        self.db_path = Path(db_path or DB_PATH)
        self.cfg = cfg or RetrievalConfig()
        # check_same_thread=False, deliberately. `ui.serve` runs on
        # ThreadingHTTPServer, which handles each connection on a new thread, so
        # a connection created while answering the first request was unusable
        # from every later one: sqlite3 raised ProgrammingError, socketserver
        # printed a traceback and closed the socket with no HTTP response, and
        # the UI silently answered exactly one question. Sharing is safe because
        # sqlite3.threadsafety is SERIALIZED (module- or connection-level) on
        # CPython's bundled SQLite, and the UI additionally serialises whole
        # `ask()` calls behind a lock; without that lock the safety would rest
        # on statement-level serialisation alone, which is enough for reads.
        self.con = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.dense = DenseIndex(self.db_path)
        self._chunks: dict[str, Chunk] = {}
        for r in self.con.execute(f"SELECT {CHUNK_COLS} FROM chunks"):
            c = Chunk(
                chunk_id=r["chunk_id"], doc_id=r["doc_id"], source_id=r["source_id"],
                url=r["url"], kind=r["kind"], topic=r["topic"], scheme_key=r["scheme_key"],
                text=r["text"], answer_hint=r["answer_hint"] or "",
                last_updated=r["last_updated"] or "",
                # `meta.field` is what the field-hint filter keys on; dropping
                # it here silently disables every field-level signal.
                meta=json.loads(r["meta_json"] or "{}"),
            )
            c.tokens = tokenize(c.embed_text())
            self._chunks[c.chunk_id] = c
        self.stats = dict(
            n_chunks=len(self._chunks),
            n_facts=sum(1 for c in self._chunks.values() if c.kind == "fact"),
            n_sections=sum(1 for c in self._chunks.values() if c.kind == "section"),
            n_procedures=sum(1 for c in self._chunks.values() if c.kind == "procedure"),
        )

    # -- scoring ------------------------------------------------------------
    def _scheme_boost(self, chunk: Chunk, scheme_key: str | None, query: str) -> float:
        if not scheme_key:
            return 0.0
        if chunk.scheme_key == scheme_key:
            return self.cfg.scheme_match_boost
        if chunk.scheme_key is None:
            return self.cfg.scheme_match_boost * 0.25  # scheme-agnostic docs stay viable
        return -self.cfg.scheme_match_boost * 0.5        # penalise the wrong scheme

    def _topic_boost(self, chunk: Chunk, topics: list[str]) -> float:
        if not topics:
            return 0.0
        rank = {t: i for i, t in enumerate(topics)}
        if chunk.topic in rank:
            return self.cfg.topic_match_boost * (1.0 if rank[chunk.topic] == 0 else 0.6)
        if chunk.kind == "procedure":
            for t in topics:
                if chunk.topic in NEIGHBOUR_TOPICS.get(t, set()):
                    return self.cfg.topic_match_boost * 0.5
        return 0.0

    def _field_boost(self, chunk: Chunk, hints: list[str]) -> float:
        """Prefer the specific fact field the wording points at.

        "minimum SIP" and "minimum lump-sum" are different numbers under the
        same topic, and "who manages X" does not lexically match a chunk that
        says "fund manager".  Without this hint the wrong fact wins on raw
        term overlap.  The first hint is the strongest signal; later ones are
        secondary matches.
        """
        if not hints:
            return 0.0
        field_name = chunk.meta.get("field")
        if not field_name:
            return 0.0
        if field_name == hints[0]:
            return self.cfg.field_match_boost
        if field_name in hints:
            return self.cfg.field_match_boost * 0.4
        return 0.0

    def _kind_gate(self, chunk: Chunk, topics: list[str]) -> float:
        """Encode what must *not* answer a topic, not just what may.

        Two asymmetries matter:
          * a per-scheme number can never answer a portfolio-wide procedural
            question ("how do I download a statement");
          * the three statement procedures are mutually exclusive
            destinations, so a "capital gains" procedure is the wrong answer to
            a "tax statement" question even though both mention statements.
        """
        t = topics[0] if topics else None
        if t in PROCEDURAL_GROUP:
            if chunk.kind == "fact":
                return -0.25
            if chunk.kind == "procedure" and chunk.topic != t:
                return -0.20
        if t in {"riskometer", "expense_ratio_note"} and chunk.kind == "fact":
            return -0.25
        return KIND_PRIOR.get(chunk.kind, 0.0)

    # -- retrieve -----------------------------------------------------------
    def _same_scheme(self, chunk: Chunk, scheme_key: str | None) -> bool:
        """Whether `chunk` is allowed to answer a question about `scheme_key`.

        A `fact` chunk is per-scheme by construction -- `build_fact_chunks`
        refuses a document with no `scheme_key` -- so a fact belonging to a
        *different* scheme is not a weaker answer, it is an answer about the
        wrong product. A soft penalty cannot enforce that: the wrong-scheme
        penalty is only `-0.5 * 0.22 = -0.11`, which a saturated BM25 score plus
        the primary-topic boost overruns easily. "What is the lock-in period in
        HDFC Small Cap Fund?" was answered with ELSS Tax Saver's 3-year lock-in,
        because ELSS was the only scheme carrying a lock-in fact and therefore
        won every lexical comparison against the right scheme's weaker hits.

        This is a filter rather than a boost for the same reason the Phase 8
        field hint is: correctness must not depend on a soft signal winning.

        Scheme-agnostic chunks (procedures, and the shared NAV-all page) still
        pass, so "how do I download a statement" is unaffected.
        """
        if not scheme_key or chunk.kind != "fact":
            return True
        return chunk.scheme_key is None or chunk.scheme_key == scheme_key

    def retrieve(self, decision: Decision, top_k: int | None = None) -> RetrievalResult:
        import time
        t0 = time.perf_counter()
        cfg = self.cfg
        top_k = top_k or cfg.top_k
        q = decision.query

        lex = bm25_search(self.con, q, cfg.candidate_k)
        t_lex = time.perf_counter()
        dense = self.dense.search(q, cfg.candidate_k)
        t_dense = time.perf_counter()

        bm_max = max((s for _, s in lex), default=1.0) or 1.0
        bm_map = {cid: s / bm_max for cid, s in lex}
        dn_map = dict(dense)

        cand_ids: list[str] = []
        for cid, _ in lex:
            cand_ids.append(cid)
        for cid, _ in dense:
            if cid not in bm_map:
                cand_ids.append(cid)

        pool = [c for c in (self._chunks.get(cid) for cid in cand_ids)
                if c is not None]
        allowed = [c for c in pool if self._same_scheme(c, decision.scheme_key)]
        # The gate must never empty the result set. A mis-detected scheme key
        # should degrade to a weaker answer, not to no answer at all.
        if allowed:
            pool = allowed

        scored: list[Scored] = []
        for chunk in pool:
            b = bm_map.get(chunk.chunk_id, 0.0)
            d = max(0.0, dn_map.get(chunk.chunk_id, 0.0))
            s = Scored(chunk=chunk, bm25=b, dense=d)
            s.scheme_boost = self._scheme_boost(chunk, decision.scheme_key, q)
            s.topic_boost = self._topic_boost(chunk, decision.topics)
            s.kind_prior = self._kind_gate(chunk, decision.topics)
            s.field_boost = self._field_boost(chunk, decision.field_hints)
            s.final = (
                cfg.bm25_weight * b
                + cfg.dense_weight * d
                + s.scheme_boost + s.topic_boost + s.kind_prior + s.field_boost
            )
            scored.append(s)

        scored.sort(key=lambda x: x.final, reverse=True)
        scored = [s for s in scored if s.final >= cfg.min_score][: cfg.candidate_k]

        picked = self._mmr(scored, top_k, decision)
        for s in picked:
            s.reasons = self._explain(s, decision)

        return RetrievalResult(
            query=q, decision=decision, hits=picked, n_candidates=len(scored),
            index_stats=self.stats,
            timings_ms={
                "lexical_ms": round((t_lex - t0) * 1000, 2),
                "dense_ms": round((t_dense - t_lex) * 1000, 2),
                "rerank_ms": round((time.perf_counter() - t_dense) * 1000, 2),
            },
        )

    def _mmr(self, scored: list[Scored], k: int, decision: Decision) -> list[Scored]:
        """Maximal Marginal Relevance - drops near-duplicate fact chunks.

        Vectors are projected once for the whole candidate pool; recomputing
        them per comparison pair is what made the first version of this method
        two orders of magnitude slower than the rest of retrieval.
        """
        if not scored:
            return []
        # Reuse the vectors already stored in the index rather than
        # re-projecting the candidate texts: the first version of this method
        # recomputed them per comparison pair and dominated end-to-end latency.
        vecs = self.dense.matrix
        selected: list[Scored] = []
        pool = list(scored)
        lam = self.cfg.mmr_lambda
        while pool and len(selected) < k:
            best, best_val, best_red = None, -1e9, 0.0
            for cand in pool:
                cv = vecs[cand.chunk.chunk_id]
                red = 0.0
                for sel in selected:
                    red = max(red, cosine(cv, vecs[sel.chunk.chunk_id]))
                val = lam * cand.final - (1 - lam) * red
                if val > best_val:
                    best, best_val, best_red = cand, val, red
            if best is None:
                break
            best.penalty = round((1 - lam) * best_red, 4)
            selected.append(best)
            pool.remove(best)
        return selected

    def _explain(self, s: Scored, decision: Decision) -> list[str]:
        out = []
        if s.bm25 > 0.05:
            out.append(f"bm25={s.bm25:.2f}")
        if s.dense > 0.05:
            out.append(f"cos={s.dense:.2f}")
        if s.scheme_boost:
            out.append(f"scheme{'+' if s.scheme_boost > 0 else '-'}")
        if s.topic_boost:
            out.append(f"topic:{s.chunk.topic}")
        if s.field_boost:
            out.append(f"field:{s.chunk.meta.get('field')}")
        if s.kind_prior:
            out.append(f"kind:{s.chunk.kind}")
        return out

    def close(self) -> None:
        self.con.close()


# ------------------------------------------------------------------ diagnostics

def explain(result: RetrievalResult) -> str:
    """Print every intermediate signal, then the winning reasons.

    FR-10 asks for explainability, and a compact reason list is not enough to
    debug a ranking: you cannot tell *which* boost carried a hit, or by how
    much. So each hit gets the full additive breakdown, including the MMR
    redundancy penalty, which otherwise is only visible in `Scored.penalty`.
    """
    lines = [f"query      : {result.query!r}",
             f"decision   : {result.decision.kind} | scheme={result.decision.scheme_key} | topics={result.decision.topics}",
             f"hints      : {result.decision.field_hints}",
             f"candidates : {result.n_candidates} | timings={result.timings_ms}"]
    for i, s in enumerate(result.hits, 1):
        c = s.chunk
        lines.append(
            f"  {i}. final={s.final:+.3f}  [{c.kind}/{c.topic}/{c.scheme_key or 'any'}]"
            f"  mmr_penalty={s.penalty:+.3f}"
        )
        lines.append(
            f"     bm25={s.bm25:+.3f} dense={s.dense:+.3f} "
            f"scheme={s.scheme_boost:+.3f} topic={s.topic_boost:+.3f} "
            f"field={s.field_boost:+.3f} kind={s.kind_prior:+.3f}"
        )
        lines.append(f"     field={c.meta.get('field') or '-'} | reasons: {', '.join(s.reasons)}")
        lines.append(f"     {c.text[:150]}")
        lines.append(f"     src: {c.url}")
    return "\n".join(lines)


def qvec_dump(query: str, db_path: Path | None = None) -> list[tuple[str, float]]:
    """Diagnostics: dense-only ranking (used in evaluation)."""
    return DenseIndex(db_path).search(query, 10)
