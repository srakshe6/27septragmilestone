"""STAGE 4 - Indexing (lexical + vector).

Two indexes are built over the same chunk collection so that retrieval
(stage 6) can be hybrid:

  * **BM25 lexical index** - SQLite FTS5. Exact-match robust, needs no
    training, explains itself, and is the workhorse for jargon-heavy
    financial queries ("ELSS", "exit load", "riskometer").

  * **Dense vector index** - L2-normalised, IDF-weighted feature-hashed
    vectors stored as BLOBs. Implemented with the standard library (numpy is
    used only to accelerate when present) so the project installs with zero
    third-party dependencies. Hashed word unigrams + character 4-grams give
    partial-term tolerance ("expense" ~ "expenses", "riskometre" ~ "riskometer")
    that a pure BM25 index would miss.

Neither index is a black box: `index_stats()` reports document frequencies so
retrieval behaviour can be inspected during evaluation.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import struct
import textwrap
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .chunking import Chunk, tokenize
from .config import DB_PATH, RetrievalConfig, ensure_dirs

try:  # optional acceleration only
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

FTS_TABLE = "chunks_fts"
CHUNK_TABLE = "chunks"
META_TABLE = "index_meta"
SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL,
    source_id     TEXT NOT NULL,
    url           TEXT NOT NULL,
    kind          TEXT NOT NULL,
    topic         TEXT NOT NULL,
    scheme_key    TEXT,
    text          TEXT NOT NULL,
    answer_hint   TEXT,
    last_updated  TEXT,
    meta_json     TEXT,
    vector        BLOB
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    text,
    answer_hint,
    topic,
    scheme_key UNINDEXED,
    tokenize = "unicode61 remove_diacritics 2"
);
CREATE TABLE IF NOT EXISTS index_meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS doc_freq (term TEXT PRIMARY KEY, df INTEGER);
"""

#: The zero-dependency vectoriser. Any other `embed_model` is a HuggingFace
#: sentence-transformers model id. Persisted in `index_meta['config']` so the
#: query plane rebuilds the *right* vectoriser, never the wrong one.
EMBED_HASHED = "hashed"


# ------------------------------------------------------------------ vectoriser

class HashedVectoriser:
    """IDF-weighted feature-hashed bag-of-features with L2 normalisation.

    Pure-python reference implementation; `transform` is vectorised with
    numpy when it is available.
    """

    def __init__(self, dim: int = 512, ngram: int = 4, use_chargrams: bool = True):
        self.dim = dim
        self.ngram = ngram
        self.use_chargrams = use_chargrams
        self.idf: dict[str, float] = {}
        self.df: Counter[str] = Counter()
        self.default_idf = 1.0

    # -- feature extraction -------------------------------------------------
    @staticmethod
    def _bucket(feature: str) -> tuple[int, float]:
        h = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        idx = struct.unpack("<Q", h)[0]
        sign = 1.0 if (h[0] & 1) == 0 else -1.0  # signed hashing reduces bias
        return idx, sign

    def features(self, text: str) -> list[str]:
        toks = tokenize(text)
        feats = [f"w:{t}" for t in toks]
        if self.use_chargrams:
            joined = " " + " ".join(toks) + " "
            n = self.ngram
            for i in range(len(joined) - n + 1):
                gram = joined[i : i + n].strip()
                if len(gram) == n and " " not in gram:
                    feats.append(f"c:{gram}")
        return feats

    # -- corpus fit ---------------------------------------------------------
    def fit(self, corpus: list[str]) -> "HashedVectoriser":
        n = max(1, len(corpus))
        df: Counter[str] = Counter()
        for text in corpus:
            df.update(set(self.features(text)))
        self.df = df
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self.default_idf = math.log(n + 1) + 1.0
        return self

    # -- transform ----------------------------------------------------------
    @staticmethod
    def _normalise(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def _accumulate(self, counts: Counter, vec: list[float]) -> None:
        """Shared inner loop, so both planes do identical float64 arithmetic.

        Order of `counts.items()` is identical on both paths, and every value is
        a plain IEEE-754 double, so numpy and pure-python agree to the last bit.
        """
        for feat, tf in counts.items():
            w = (1.0 + math.log(tf)) * self.idf.get(feat, self.default_idf)
            idx, sign = self._bucket(feat)
            vec[idx % self.dim] += sign * w

    def transform_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        self._accumulate(Counter(self.features(text)), vec)
        return self._normalise(vec)

    def transform(self, texts: list[str]) -> "list[list[float]]":
        if _np is None:
            return [self.transform_one(t) for t in texts]
        # float64, not float32: the pure-python path accumulates in double, and
        # a float32 accumulator drifts (~6e-8 here), which would make an index
        # built with numpy differ from one built without it.  That breaks the
        # reproducibility guarantee in the phase notes.
        mat = _np.zeros((len(texts), self.dim), dtype=_np.float64)
        for r, t in enumerate(texts):
            row = [0.0] * self.dim
            self._accumulate(Counter(self.features(t)), row)
            mat[r] = row
        # Normalise through the same helper as transform_one. numpy's
        # `linalg.norm` sums in a different order, which is enough to change the
        # last bit of the result.
        return [self._normalise(row.tolist()) for row in mat]


def _pack(vec: list[float]) -> bytes:
    if _np is not None:
        return _np.asarray(vec, dtype=_np.float32).tobytes()
    return struct.pack(f"<{len(vec)}f", *vec)


class SentenceTransformerVectoriser:
    """HuggingFace `sentence-transformers` vectors behind the same interface
    as `HashedVectoriser`, so `build_index`/`load_vectoriser` are the only
    places that know which one is live (the D-1 seam).

    Transform is L2-normalised sentence embeddings; `fit` is a no-op (there is
    no corpus-wide vocabulary to estimate). The model is loaded lazily, so
    importing `indexing` never pulls `torch` into a process that only wants the
    hashed fallback.
    """

    def __init__(self, model_id: str):
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - depends on the install
            raise ImportError(
                f"embed_model={model_id!r} needs 'sentence-transformers', which is "
                f"not importable here: {type(exc).__name__}: {exc}. "
                f"Install it (`python -m pip install sentence-transformers`, pulls "
                f"torch) or set MF_FAQ_EMBED_MODEL=hashed for the zero-dependency "
                f"vectors."
            ) from exc
        self.model_id = model_id
        self.model = SentenceTransformer(model_id)
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.idf: dict[str, float] = {}
        self.ngram = 0

    def fit(self, corpus: list[str]) -> "SentenceTransformerVectoriser":
        return self

    def transform_one(self, text: str) -> list[float]:
        return self.transform([text])[0]

    def transform(self, texts: list[str]) -> "list[list[float]]":
        emb = self.model.encode(list(texts), normalize_embeddings=True)
        return [row.tolist() for row in emb]


def _unpack(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def cosine(a: list[float], b: list[float]) -> float:
    """Dot product of two L2-normalised vectors, i.e. their cosine similarity.

    Deliberately pure-python even when numpy is available. A float32 BLAS dot
    and a float64 Python sum disagree in the last bits, and this is the ranking
    function: two machines must order the same candidates the same way. 512
    multiply-adds per candidate is not worth the divergence.
    """
    return sum(x * y for x, y in zip(a, b))


# ------------------------------------------------------------------ build

def get_vectoriser(cfg: RetrievalConfig | None = None) -> "HashedVectoriser | SentenceTransformerVectoriser":
    """Factory for the dense embedding vectoriser (the D-1 seam).

    `embed_model == "hashed"` (or unset) gives the zero-dependency vectors.
    Anything else is treated as a HuggingFace sentence-transformers model id.
    An HF model that cannot be loaded raises rather than silently degrading:
    a user who asked for MiniLM should not receive hashed vectors without
    noticing, because the two indexes are not comparable.
    """
    cfg = cfg or RetrievalConfig()
    model = (cfg.embed_model or EMBED_HASHED).strip().lower()
    if model in (EMBED_HASHED, "default", "none", ""):
        return HashedVectoriser(dim=cfg.dim)
    return SentenceTransformerVectoriser(cfg.embed_model)


def build_index(chunks: list[Chunk], db_path: Path | None = None,
                cfg: RetrievalConfig | None = None, verbose: bool = True) -> dict:
    cfg = cfg or RetrievalConfig()
    path = Path(db_path or DB_PATH)
    ensure_dirs()

    # Compute the vectors BEFORE touching the on-disk index. Building the
    # embeddings is the only step that can fail (an HF model download, an
    # ImportError for an uninstalled model); a failure here must leave the
    # previous good index intact rather than an empty schema in its place.
    corpus_text = [c.embed_text() for c in chunks]
    vec = get_vectoriser(cfg)
    if isinstance(vec, HashedVectoriser):
        vec.fit(corpus_text)
    vectors = vec.transform(corpus_text)

    if path.exists():
        try:
            path.unlink()
        except PermissionError as e:
            # Windows refuses to unlink a file another process still holds open,
            # and the obvious holder is a `run.py serve` left running. Unhandled,
            # this surfaced as a bare PermissionError traceback *after* the
            # corpus and chunk files had already been overwritten, leaving a
            # half-applied build: new chunks.jsonl beside a stale index.
            raise PermissionError(
                f"Cannot replace the index at {path}: it is open in another "
                f"process. Stop any running `python run.py serve` and retry. "
                f"(underlying error: {e})"
            ) from e
    con = sqlite3.connect(str(path))
    con.executescript(SCHEMA)

    rows_fts, rows_chunks = [], []
    for c, v in zip(chunks, vectors):
        rows_chunks.append((
            c.chunk_id, c.doc_id, c.source_id, c.url, c.kind, c.topic,
            c.scheme_key, c.text, c.answer_hint, c.last_updated,
            json.dumps(c.meta, default=str), _pack(v),
        ))
        rows_fts.append((c.chunk_id, c.text, c.answer_hint, c.topic, c.scheme_key))

    con.executemany(
        f"INSERT INTO chunks ({','.join(['chunk_id','doc_id','source_id','url','kind','topic','scheme_key','text','answer_hint','last_updated','meta_json','vector'])}) "
        f"VALUES ({','.join(['?']*12)})", rows_chunks)
    con.executemany(
        "INSERT INTO chunks_fts (chunk_id, text, answer_hint, topic, scheme_key) VALUES (?,?,?,?,?)",
        rows_fts)

    vec_cfg: dict = {"dim": vec.dim}
    is_hashed = isinstance(vec, HashedVectoriser)
    meta_rows = []
    if is_hashed:
        # Materialise document frequencies. `fit` already computed them; without
        # this the table is an empty shell and the "inspect the vocabulary" promise
        # in the module docstring is not kept.
        con.executemany("INSERT OR REPLACE INTO doc_freq (term, df) VALUES (?,?)",
                        sorted(vec.df.items()))
        vec_cfg.update({"embed_model": EMBED_HASHED, "ngram": vec.ngram,
                        "default_idf": vec.default_idf})
        meta_rows.append(("idf", json.dumps(vec.idf)))
    else:
        vec_cfg["embed_model"] = vec.model_id

    built = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stats = {
        "built_at": built, "n_chunks": len(chunks),
        "n_sources": len({c.source_id for c in chunks}),
        "n_schemes": len({c.scheme_key for c in chunks if c.scheme_key}),
        "by_kind": dict(Counter(c.kind for c in chunks)),
        "by_topic": dict(Counter(c.topic for c in chunks)),
        "embed_model": vec_cfg["embed_model"], "dim": vec.dim,
        "vocab": len(vec.idf) if is_hashed else 0, "numpy": _np is not None,
    }
    con.executemany(
        "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?,?)",
        [("built_at", built), ("stats", json.dumps(stats)),
         # `default_idf` is part of the fitted state, not a derivable detail:
         # the weight given to a term the corpus never saw. It used to be
         # reconstructed as max(idf.values()), which is ~11% lower than the
         # value the documents were built with, so every query containing an
         # unseen term was scored against a different weighting.
         ("config", json.dumps(vec_cfg)), *meta_rows])
    con.commit(); con.close()
    if verbose:
        print(f"[index] {stats['n_chunks']} chunks -> {path}")
        print(f"[index] embed_model={stats['embed_model']} dim={stats['dim']} "
              f"kinds={stats['by_kind']}")
    return stats


def load_vectoriser(db_path: Path | None = None) -> "HashedVectoriser | SentenceTransformerVectoriser":
    """Rebuild the query-plane vectoriser from `index_meta` alone.

    No pickle, no re-fit, no corpus access -- this must keep working offline
    against a read-only index (I6). The stored `embed_model` decides which
    vectoriser to reconstruct: hashed vectors rebuild their IDf table, an HF
    model re-loads the sentence-transformers model. An index written before
    this field existed defaults to hashed.
    """
    path = Path(db_path or DB_PATH)
    con = sqlite3.connect(str(path))
    cfg = json.loads(con.execute("SELECT value FROM index_meta WHERE key='config'").fetchone()[0])
    model = cfg.get("embed_model", EMBED_HASHED)
    if model != EMBED_HASHED:
        con.close()
        return SentenceTransformerVectoriser(cfg["embed_model"])
    idf = json.loads(con.execute("SELECT value FROM index_meta WHERE key='idf'").fetchone()[0])
    con.close()
    vec = HashedVectoriser(dim=cfg["dim"], ngram=cfg.get("ngram", 4))
    vec.idf = idf
    # The stored weight for out-of-vocabulary terms. Fall back to the old
    # `max(idf)` guess only for an index written before this was persisted;
    # new indexes always carry the value the documents were built with.
    vec.default_idf = float(cfg.get("default_idf", max(idf.values()) if idf else 1.0))
    return vec


def index_stats(db_path: Path | None = None) -> dict:
    path = Path(db_path or DB_PATH)
    if not path.exists():
        return {}
    con = sqlite3.connect(str(path))
    row = con.execute("SELECT value FROM index_meta WHERE key='stats'").fetchone()
    con.close()
    return json.loads(row[0]) if row else {}


def row_to_chunk(row) -> Chunk:
    c = Chunk(
        chunk_id=row[0], doc_id=row[1], source_id=row[2], url=row[3], kind=row[4],
        topic=row[5], scheme_key=row[6], text=row[7], answer_hint=row[8] or "",
        last_updated=row[9] or "", meta=json.loads(row[10] or "{}"),
    )
    # Same derivation as build time, so the token list printed by the report is
    # the one that actually produced the stored vector.
    c.tokens = tokenize(c.embed_text())
    return c


def fetch_chunk(chunk_id: str, db_path: Path | None = None) -> Chunk | None:
    con = sqlite3.connect(str(Path(db_path or DB_PATH)))
    row = con.execute("SELECT chunk_id,doc_id,source_id,url,kind,topic,scheme_key,text,answer_hint,last_updated,meta_json FROM chunks WHERE chunk_id=?",
                      (chunk_id,)).fetchone()
    con.close()
    return row_to_chunk(row) if row else None


def all_chunks(db_path: Path | None = None) -> list[Chunk]:
    con = sqlite3.connect(str(Path(db_path or DB_PATH)))
    rows = con.execute(
        "SELECT chunk_id,doc_id,source_id,url,kind,topic,scheme_key,text,answer_hint,last_updated,meta_json FROM chunks"
    ).fetchall()
    con.close()
    return [row_to_chunk(r) for r in rows]


def all_vectors(db_path: Path | None = None) -> dict[str, list[float]]:
    con = sqlite3.connect(str(Path(db_path or DB_PATH)))
    rows = con.execute("SELECT chunk_id, vector FROM chunks").fetchall()
    con.close()
    return {cid: _unpack(blob) for cid, blob in rows if blob}


# ------------------------------------------------------------------ inspection

REPORT_WIDTH = 88
REPORT_RULE = "=" * REPORT_WIDTH
REPORT_SUBRULE = "-" * REPORT_WIDTH


def _wrap(label: str, body: str, width: int = REPORT_WIDTH) -> list[str]:
    lines = textwrap.wrap(body or "", width=width - 4,
                          initial_indent=f"  {label} ", subsequent_indent="      ",
                          break_long_words=False, break_on_hyphens=False)
    return lines or [f"  {label}"]


def _vector_stats(vec: list[float]) -> tuple[list[tuple[int, float]], float]:
    """Non-zero (dim, value) pairs and the L2 norm of one stored vector."""
    nz = [(i, v) for i, v in enumerate(vec) if v != 0.0]
    norm = math.sqrt(sum(v * v for v in vec))
    return nz, norm


def _format_vector(vec: list[float], full: bool, top_n: int) -> list[str]:
    """Render one embedding.

    Full 512-d rows per chunk is ~76k lines of noise, and almost every value is
    0.0: the hashed buckets are sparse. So by default only the non-zero
    dimensions are listed, plus the strongest few by magnitude, which is what
    actually distinguishes two chunks. `full=True` prints every dimension.
    """
    nz, norm = _vector_stats(vec)
    lines = [f"  embedding  : dim={len(vec)} non_zero={len(nz)} "
             f"l2_norm={norm:.6f} sum={sum(vec):+.6f}"]
    top = sorted(nz, key=lambda iv: abs(iv[1]), reverse=True)[:top_n]
    if top:
        lines.append(f"  top_dims   : " + "  ".join(f"[{i}]={v:+.4f}" for i, v in top))
    rows = [(i, v) for i, v in enumerate(vec)] if full else nz
    if full:
        lines.append("  dims (all):")
    else:
        lines.append("  dims (non-zero):")
    for lo in range(0, len(rows), 4):
        cell = "".join(f"[{i:>3}]={v:+.5f}  " for i, v in rows[lo:lo + 4])
        lines.append("      " + cell.rstrip())
    return lines


def nearest_neighbours(vectors: dict[str, list[float]], chunk_id: str,
                       k: int = 3) -> list[tuple[str, float]]:
    """The `k` closest other chunks to `chunk_id` by cosine.

    Sorted descending. The dot product of two L2-normalised vectors is already
    cosine, so this reuses `cosine()` rather than recomputing a norm.
    """
    base = vectors.get(chunk_id)
    if base is None or k <= 0:
        return []
    out: list[tuple[str, float]] = []
    for cid, vec in vectors.items():
        if cid == chunk_id:
            continue
        out.append((cid, cosine(base, vec)))
    out.sort(key=lambda cv: cv[1], reverse=True)
    return out[:k]


def write_txt_report(path: Path | str, db_path: Path | None = None,
                     full: bool = False, top_n: int = 24,
                     kind: str | None = None, topic: str | None = None,
                     contains: str | None = None, limit: int | None = None,
                     show_tokens: bool = True, neighbors: int = 0,
                     chunks_by_id: dict | None = None) -> dict:
    """Write every chunk and its stored embedding to a readable .txt file.

    `chunks.jsonl` already holds the chunk text, but not the vector: the blob
    lives in the SQLite index and there is no other way to eyeball what the
    vectoriser actually produced. This pairs the two in one artifact per chunk -
    the sentence that was indexed, the tokens that fed the hashed features, and
    the resulting 512 numbers - so a retrieval result can be checked by hand.

    `neighbors=N` adds the N nearest chunks by cosine to each entry, which is
    the only way to see whether the dense plane is grouping the right things:
    a 512-number row on its own says nothing about relative position.

    Returns a small summary dict (counts written, path, filters applied).
    """
    path = Path(path)
    stats = index_stats(db_path)
    chunks = all_chunks(db_path)
    vectors = all_vectors(db_path)
    # Neighbour lookups are by chunk_id, so index the full set before any
    # filtering: a filter narrows what is *written*, not what a neighbour can be.
    all_by_id = {c.chunk_id: c for c in chunks}
    chunks_by_id = all_by_id if chunks_by_id is None else chunks_by_id

    if kind:
        chunks = [c for c in chunks if c.kind == kind]
    if topic:
        chunks = [c for c in chunks if c.topic == topic]
    if contains:
        needle = contains.lower()
        chunks = [c for c in chunks
                  if needle in c.text.lower() or needle in (c.answer_hint or "").lower()
                  or needle in c.chunk_id.lower()]
    total_matched = len(chunks)
    if limit:
        chunks = chunks[:limit]

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out: list[str] = [
        REPORT_RULE,
        "MF FAQ - CHUNKS AND EMBEDDINGS",
        REPORT_RULE,
        f"generated_at : {now}",
        f"index        : {Path(db_path or DB_PATH)}",
        f"index_built  : {stats.get('built_at', 'unknown')}",
        f"n_chunks     : {stats.get('n_chunks', len(chunks))}",
        f"dim          : {stats.get('dim', '?')}  (hashed, IDF-weighted, L2-normalised)",
        f"numpy        : {stats.get('numpy', False)}",
        f"filters      : kind={kind or '-'} topic={topic or '-'} "
        f"contains={contains or '-'} limit={limit or '-'}",
        f"shown        : {len(chunks)} of {total_matched} matching chunk(s)",
        f"vector_mode  : {'all dims' if full else 'non-zero dims only'} (top_n={top_n})",
        "",
    ]
    by_kind = Counter(c.kind for c in chunks)
    by_topic = Counter(c.topic for c in chunks)
    out.append("COUNTS")
    out.append(f"  by kind  : {dict(by_kind) or '-'}")
    out.append(f"  by topic : {dict(by_topic) or '-'}")
    out.append("")

    for n, c in enumerate(chunks, 1):
        out.append(REPORT_SUBRULE)
        out.append(f"CHUNK {n} / {len(chunks)}")
        out.append(REPORT_SUBRULE)
        out.append(f"  chunk_id     : {c.chunk_id}")
        out.append(f"  doc_id       : {c.doc_id}")
        out.append(f"  source_id    : {c.source_id}")
        out.append(f"  url          : {c.url}")
        out.append(f"  kind         : {c.kind}")
        out.append(f"  topic        : {c.topic}")
        out.append(f"  scheme_key   : {c.scheme_key or '-'}")
        out.append(f"  last_updated : {c.last_updated or '-'}")
        out.append(f"  answer_hint  : {c.answer_hint or '-'}")
        out.append(f"  meta         : {json.dumps(c.meta, default=str, ensure_ascii=False)}")
        out.append("")
        out.extend(_wrap("text    :", c.text))
        if show_tokens:
            toks = c.tokens or tokenize(c.text)
            out.append(f"  tokens  ({len(toks)}):")
            for lo in range(0, len(toks), 10):
                out.append("      " + " ".join(toks[lo:lo + 10]))
        vec = vectors.get(c.chunk_id)
        out.append("")
        if vec is None:
            out.append("  embedding  : <none stored in the index>")
        else:
            out.extend(_format_vector(vec, full, top_n))
        if neighbors:
            nn = nearest_neighbours(vectors, c.chunk_id, neighbors)
            if not nn:
                out.append("  nearest    : <no other vectors>")
            for cid, score in nn:
                other = chunks_by_id.get(cid)
                snippet = f"  {other.text[:64]}" if other is not None else ""
                kind = f"[{other.kind}] " if other is not None else ""
                out.append(f"  nearest    : cos={score:+.3f}  {kind}{cid}{snippet}")
        out.append("")

    missing = [c.chunk_id for c in chunks if c.chunk_id not in vectors]
    out.append(REPORT_RULE)
    out.append("END - " + ("all chunks have a stored vector"
                          if not missing else f"{len(missing)} chunk(s) have no vector"))
    out.append(REPORT_RULE)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return {"path": str(path), "chunks_written": len(chunks),
            "chunks_matched": total_matched, "missing_vectors": len(missing),
            "lines": len(out)}
