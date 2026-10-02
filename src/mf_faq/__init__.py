"""MF FAQ RAG assistant - a staged, auditable mutual-fund facts pipeline.

Stages
  0 sources      source registry / citation allow-list
  1 ingest       fetch + raw cache
  2 parsing      HTML -> text, embedded JSON -> facts
  3 chunking     fact / section / procedure chunks
  4 indexing     FTS5 BM25 + hashed dense vectors
  5 intent       PII scrub, refusal gate, scheme + topic routing
  6 retrieval    hybrid BM25 + dense + boosts + MMR
  7 synthesis    LLM (optional) or deterministic extractive composer
  8 guardrails   citation / sentence cap / advice / performance / PII
  9 evaluation   golden set + policy assertions
"""
from .config import DISCLAIMER, DISCLAIMER_LONG, MAX_SENTENCES
from .guardrails import Answer
from .intent import Decision, classify
from .pipeline import FAQPipeline
from .retrieval import Retriever, explain
from .sources import SOURCES, SCHEMES

__all__ = [
    "FAQPipeline", "Answer", "Decision", "Retriever", "classify", "explain",
    "SOURCES", "SCHEMES", "DISCLAIMER", "DISCLAIMER_LONG", "MAX_SENTENCES",
    "WELCOME", "EXAMPLE_QUESTIONS", "ask",
]

WELCOME = (
    "Hi! I answer factual questions about five HDFC Mutual Fund schemes - "
    "Large Cap, Flexi Cap, ELSS Tax Saver, Small Cap and Balanced Advantage. "
    "Every answer carries one source link. Facts-only. No investment advice."
)

EXAMPLE_QUESTIONS = [
    "What is the expense ratio of HDFC Large Cap Fund?",
    "Is there a lock-in period in HDFC ELSS Tax Saver Fund?",
    "How do I download a capital gains statement?",
]


def ask(query: str, **kwargs) -> Answer:
    """One-shot convenience helper (builds the index on first use)."""
    pipe = FAQPipeline(**kwargs)
    try:
        return pipe.ask(query, debug=kwargs.get("debug", False))
    finally:
        pipe.close()
