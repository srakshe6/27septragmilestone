"""Pipeline orchestrator: wires stages 1-8 together.

    build()  : ingest -> parse -> chunk -> index      (offline, repeatable)
    ask()    : stage 5 gate -> stage 6 retrieve -> stage 7 synthesise
               -> stage 8 guardrails -> Answer
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .chunking import Chunk, build_chunks
from .config import (CHUNKS_PATH, CONFIG, CORPUS_PATH, DB_PATH, RetrievalConfig,
                     ensure_dirs)
from .guardrails import Answer, enforce, make_refusal
from .indexing import build_index, index_stats
from .ingest import fetch_all, read_cache, write_manifest
from .intent import Decision, classify
from .parsing import Document, parse_document
from .retrieval import Retriever, RetrievalResult, explain
from .sources import SOURCES
from .synthesis import Draft, get_synthesiser

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "data" / "fetch_manifest.json"


class FAQPipeline:
    """Holds the built index and answers queries. One instance per process."""

    def __init__(self, db_path: Path | None = None, cfg: RetrievalConfig | None = None,
                 verbose: bool = False):
        ensure_dirs()
        self.db_path = Path(db_path or DB_PATH)
        self.cfg = cfg or RetrievalConfig()
        self.verbose = verbose
        self._retriever: Retriever | None = None
        self._synth = get_synthesiser()

    # -------------------------------------------------------------- build
    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            if not self.db_path.exists():
                raise FileNotFoundError(
                    f"Index not found at {self.db_path}. Run `python run.py build` first.")
            self._retriever = Retriever(self.db_path, self.cfg)
        return self._retriever

    def build(self, force_fetch: bool = False, write_corpus: bool = True) -> dict:
        """Stages 1-4."""
        t0 = time.perf_counter()
        ensure_dirs()

        # -- stage 1: ingestion
        results = fetch_all(SOURCES, force=force_fetch)
        write_manifest(results, MANIFEST_PATH)
        if self.verbose:
            ok = sum(1 for r in results if r.ok)
            print(f"[ingest] {ok}/{len(results)} sources fetched ok")

        by_id = {r.source_id: r for r in results}

        # -- stage 2: parsing
        docs: list[Document] = []
        for src in SOURCES:
            fr = by_id[src.source_id]
            raw = read_cache(fr) if fr.ok else ""
            doc = parse_document(src, fr, raw)
            if not doc.facts and not doc.text:
                if self.verbose:
                    print(f"[parse]  {src.source_id}: no content extracted (expected for SPAs)")
                continue
            docs.append(doc)
        if self.verbose:
            print(f"[parse]   {len(docs)} documents parsed; "
                  f"{sum(1 for d in docs if d.facts)} with structured facts")

        if write_corpus:
            with CORPUS_PATH.open("w", encoding="utf-8") as fh:
                for d in docs:
                    fh.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")

        # -- stage 3: chunking
        chunks: list[Chunk] = build_chunks(docs)
        with CHUNKS_PATH.open("w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        if self.verbose:
            print(f"[chunk]   {len(chunks)} chunks written to {CHUNKS_PATH.name}")

        # -- stage 4: indexing
        stats = build_index(chunks, self.db_path, self.cfg, verbose=self.verbose)
        stats["build_seconds"] = round(time.perf_counter() - t0, 2)
        stats["fetch"] = {
            "ok": sum(1 for r in results if r.ok),
            "total": len(results),
            "quarantined": [r.source_id for r in results if not r.ok],
        }
        return stats

    # -------------------------------------------------------------- query
    def ask(self, query: str, top_k: int | None = None, debug: bool = False) -> Answer:
        t0 = time.perf_counter()

        # -- stage 5: gate
        decision: Decision = classify(query)
        if debug and self.verbose:
            print("[gate]", json.dumps(decision.to_dict(), indent=2))
        if decision.kind != "answer":
            ref = make_refusal(decision.kind, decision.primary_topic,
                               [f"stage5:{decision.kind}"])
            ref.timings_ms = {"total_ms": round((time.perf_counter() - t0) * 1000, 2)}
            return ref

        # -- stage 6: retrieval
        result: RetrievalResult = self.retriever.retrieve(decision, top_k=top_k)
        if debug and self.verbose:
            print(explain(result))

        # -- stage 7: synthesis
        draft: Draft = self._synth.generate(result)

        # -- stage 8: guardrails
        conf = result.hits[0].final if result.hits else 0.0
        answer = Answer(
            kind="answer", text=draft.answer, citation=draft.citation,
            last_updated=draft.last_updated, stage=self._synth.name,
            freshness_from_source=draft.freshness_from_source,
            confidence=conf, evidence=draft.evidence,
        )
        answer = enforce(answer, decision.kind, decision.primary_topic)
        answer.timings_ms = {
            **result.timings_ms,
            "total_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        if debug:
            answer.retrieval_debug = {
                "decision": decision.to_dict(),
                "n_candidates": result.n_candidates,
                "hits": [
                    {"chunk_id": s.chunk.chunk_id, "kind": s.chunk.kind,
                     "topic": s.chunk.topic, "scheme": s.chunk.scheme_key,
                     "score": round(s.final, 4), "reasons": s.reasons,
                     "text": s.chunk.text[:220]}
                    for s in result.hits
                ],
            }
        return answer

    # -------------------------------------------------------------- meta
    def stats(self) -> dict:
        return index_stats(self.db_path)

    def set_llm(self, enabled: bool, provider: str | None = None) -> bool:
        """Toggle the LLM rewrite pass at runtime and return the resolved mode.

        `get_synthesiser` reads the config at construction time, so switching the
        LLM on after the pipeline exists has to re-pick the generator too.
        """
        CONFIG.llm_enabled = enabled
        if provider:
            CONFIG.llm_provider = provider
        self._synth = get_synthesiser()
        return self._synth.name

    def close(self) -> None:
        if self._retriever is not None:
            self._retriever.close()
            self._retriever = None
