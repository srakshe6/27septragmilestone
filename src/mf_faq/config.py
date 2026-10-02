"""Global configuration for the MF FAQ RAG assistant.

Everything tunable lives here so the pipeline stages stay declarative.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- .env loading
#
# python-dotenv is deliberately not a dependency: this project runs on the
# standard library alone, and a 15-line parser is cheaper than a dependency for
# a file that is read once. It runs before the CONFIG block below, because that
# block reads `os.environ` at import time.

ENV_PATH = PROJECT_ROOT / ".env"


def _unquote(value: str) -> str:
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        inner = v[1:-1]
        # Only double quotes get escape handling, so a Windows path such as
        # BASE_URL="C:\keys" keeps its backslashes.
        return inner.encode().decode("unicode_escape") if v[0] == '"' else inner
    return v


def load_env(path: Path | None = None, override: bool = False) -> dict[str, str]:
    """Populate `os.environ` from a .env file. Returns what was applied.

    A real process environment always wins over the file, so an exported
    variable in a shell or CI secret is never silently replaced by a stale
    value on disk. Pass `override=True` to flip that, which is what a test
    wanting a known value needs.
    """
    path = Path(path or ENV_PATH)
    applied: dict[str, str] = {}
    if not path.exists():
        return applied
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or (key in os.environ and not override):
            continue
        os.environ[key] = _unquote(value)
        applied[key] = os.environ[key]
    return applied
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
DB_PATH = INDEX_DIR / "mf_faq.sqlite"
RAW_CACHE_DIR = RAW_DIR

# ---------------------------------------------------------------- facts-only policy
DISCLAIMER = "Facts-only. No investment advice."

DISCLAIMER_LONG = (
    "Facts-only. No investment advice. Answers are sourced from the public pages "
    "listed below and are reproduced as of the stated date. This assistant does not "
    "recommend, compare or rate schemes, and does not compute returns. It also never "
    "asks for or stores PAN, Aadhaar, account/folio numbers, OTPs, email addresses or "
    "phone numbers."
)

MAX_SENTENCES = 3

#: Provenance wording. An answer grounded in a *fetched source* may claim
#: source freshness. A refusal or an authored procedure contains no fetched
#: data, so claiming "Last updated from sources" for it is a false provenance
#: statement - the very thing this system's provenance is meant to guarantee.
#: Those answers get a preparation date instead, under a label that says so.
LAST_UPDATED_PREFIX = "Last updated from sources: "
ANSWER_PREPARED_PREFIX = "Answer prepared on: "

# ---------------------------------------------------------------- retrieval knobs
@dataclass
class RetrievalConfig:
    top_k: int = 5
    candidate_k: int = 24
    bm25_weight: float = 0.55
    dense_weight: float = 0.45
    scheme_match_boost: float = 0.22
    topic_match_boost: float = 0.14
    field_match_boost: float = 0.20
    min_score: float = 0.06
    mmr_lambda: float = 0.72
    dim: int = 512
    snippet_sentences: int = 2
    #: Which embedding model the dense index uses. "hashed" is the zero-
    #: dependency IDF-weighted feature-hash; anything else is a HuggingFace
    #: sentence-transformers model id (e.g. "sentence-transformers/all-MiniLM-L6-v2",
    #: the PRD's mandated 384-dim model). Falling back is explicit, never silent:
    #: `get_vectoriser` refuses to build an HF index without sentence-transformers.
    #: A default_factory (not a literal) so the value is read at instantiation
    #: time: the `.env`/environment can drive it and the generated `__init__`
    #: does not bake a stale literal into its parameters.
    embed_model: str = field(default_factory=lambda: os.getenv("MF_FAQ_EMBED_MODEL") or "hashed")


@dataclass
class PipelineConfig:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    http_timeout: int = 30
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    offline: bool = False
    llm_enabled: bool = False
    llm_provider: str = "auto"          # auto | groq | openai | anthropic | none
    llm_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-5-haiku-latest"
    #: Groq retired the llama-3.3-70b-versatile id: it now answers
    #: `404 The model ... does not exist`. gpt-oss-120b is the current
    #: flagship free-tier id. Override with MF_FAQ_GROQ_MODEL.
    groq_model: str = "openai/gpt-oss-120b"
    max_context_chars: int = 6000


load_env()

CONFIG = PipelineConfig()

# Optional LLM credentials (never required; the assistant runs fully offline).
if os.getenv("MF_FAQ_LLM") == "1":
    CONFIG.llm_enabled = True
if os.getenv("MF_FAQ_LLM_PROVIDER"):
    CONFIG.llm_provider = os.getenv("MF_FAQ_LLM_PROVIDER")
if os.getenv("MF_FAQ_LLM_MODEL"):
    CONFIG.llm_model = os.getenv("MF_FAQ_LLM_MODEL")
if os.getenv("MF_FAQ_GROQ_MODEL"):
    CONFIG.groq_model = os.getenv("MF_FAQ_GROQ_MODEL")
if os.getenv("ANTHROPIC_API_KEY"):
    CONFIG.llm_enabled = True
if os.getenv("MF_FAQ_OFFLINE") == "1":
    CONFIG.offline = True


def ensure_dirs() -> None:
    for d in (DATA_DIR, RAW_DIR, INDEX_DIR):
        d.mkdir(parents=True, exist_ok=True)
