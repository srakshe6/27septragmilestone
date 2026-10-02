"""STAGE 1 - Data ingestion (fetch layer).

Responsibilities, kept deliberately dumb:
  * fetch every registered public page over plain HTTP
  * cache raw bytes on disk so the rest of the pipeline is replayable and the
    build is deterministic/offline-testable
  * record fetch provenance (status, bytes, sha256, fetched_at)

No parsing, no chunking happens here - that is stages 2 and 3.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import CONFIG, RAW_CACHE_DIR, ensure_dirs
from .sources import Source


@dataclass
class FetchResult:
    source_id: str
    url: str
    status: int
    ok: bool
    bytes: int
    sha256: str
    fetched_at: str
    cache_path: str
    from_cache: bool
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _cache_path(url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    slug = url.rstrip("/").split("/")[-1][:48] or "index"
    return RAW_CACHE_DIR / f"{slug}.{digest}.html"


def _cache_fetched_at(path: Path) -> str:
    """UTC ISO timestamp of when a cache file was last written.

    Falls back to the epoch rather than "now" if the mtime is unreadable: a
    missing timestamp is a visible gap in the record, whereas inventing a fresh
    one is a silent lie.
    """
    try:
        ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        ts = datetime.fromtimestamp(0, tz=timezone.utc)
    return ts.isoformat(timespec="seconds")


def fetch(source: Source, force: bool = False, timeout: int | None = None) -> FetchResult:
    """Fetch `source`, using the on-disk cache unless `force` is set."""
    ensure_dirs()
    path = _cache_path(source.url)
    # Computed after the cache branch below so it only ever describes a fetch
    # that actually touches the network.
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if path.exists() and not force:
        # A cache hit is served from disk whenever we are not being forced to
        # refetch -- *including* in offline mode.  Offline must never open a
        # socket, so the offline test has to come after this branch, not before
        # it; guarding it with `and not CONFIG.offline` made offline+cached fall
        # through to the network, which defeated the whole point of the flag.
        data = path.read_bytes()
        return FetchResult(
            source_id=source.source_id, url=source.url, status=200, ok=True,
            bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
            # When the bytes were *fetched*, not when they were read. This used to
            # be `datetime.now()`, computed before the cache check, so every cache
            # hit -- including an offline build -- rewrote the provenance record
            # with the current time. A 5-day-old cache then claimed to have been
            # fetched seconds ago, which is the one thing this manifest exists to
            # rule out. A rebuild that does not refetch must not restate the age
            # of the data.
            fetched_at=_cache_fetched_at(path),
            cache_path=str(path), from_cache=True,
        )

    if CONFIG.offline:
        return FetchResult(
            source_id=source.source_id, url=source.url, status=0, ok=False,
            bytes=0, sha256="", fetched_at=now, cache_path=str(path),
            from_cache=False, error="offline and not cached",
        )

    try:
        # Built inside the try: urllib rejects some malformed URLs (e.g. a bad
        # bracketed host) in Request.__init__, and ingestion must never raise.
        req = urllib.request.Request(
            source.url,
            headers={
                "User-Agent": CONFIG.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout or CONFIG.http_timeout) as resp:
            data = resp.read()
            status = int(resp.status)
        path.write_bytes(data)
        return FetchResult(
            source_id=source.source_id, url=source.url, status=status,
            ok=200 <= status < 400, bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(), fetched_at=now,
            cache_path=str(path), from_cache=False,
        )
    except urllib.error.HTTPError as exc:
        return FetchResult(
            source_id=source.source_id, url=source.url, status=int(exc.code),
            ok=False, bytes=0, sha256="", fetched_at=now, cache_path=str(path),
            from_cache=False, error=f"HTTP {exc.code} {exc.reason}",
        )
    except Exception as exc:  # noqa: BLE001 - ingestion must never hard-crash
        return FetchResult(
            source_id=source.source_id, url=source.url, status=0, ok=False,
            bytes=0, sha256="", fetched_at=now, cache_path=str(path),
            from_cache=False, error=f"{type(exc).__name__}: {exc}",
        )


def fetch_all(sources, force: bool = False, sleep: float = 0.4) -> list[FetchResult]:
    """Sequential fetch with polite spacing between requests."""
    results: list[FetchResult] = []
    for i, src in enumerate(sources):
        results.append(fetch(src, force=force))
        if i < len(sources) - 1:
            time.sleep(sleep)
    return results


def read_cache(result: FetchResult) -> str:
    p = Path(result.cache_path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def write_manifest(results: list[FetchResult], path: Path) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
