"""Local web UI (standard-library http.server, no dependencies).

Markup, styling and behaviour live in `mf_faq/web/{index.html,app.css,app.js}`
and are inlined into a single response by `render_page()`. One request, no build
step, no CDN -- the tool has to work on a machine that cannot reach the
internet, which is the same constraint the corpus itself is built under.

The UI is a thin, honest view over `pipeline.ask()`. It shows the answer, the
one allow-listed citation, the retrieved chunks, and the two separate dates:

  * "Answer prepared on"      - when this response was produced (wall clock)
  * "Last updated from sources" - what the fetched source itself said

They are displayed side by side and never substituted for one another. Printing
only the source date made an answer served out of a week-old cache read as if it
were current, which is the impression this system's provenance exists to avoid.
"""
from __future__ import annotations

import html
import json
import socket
import sys
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .config import DISCLAIMER_LONG
from .pipeline import FAQPipeline
from . import EXAMPLE_QUESTIONS, WELCOME

ASSET_DIR = Path(__file__).resolve().parent / "web"


@lru_cache(maxsize=8)
def _asset(name: str, mtime: float) -> str:
    """Read a web asset, cached per mtime so an edit shows up without a restart."""
    return (ASSET_DIR / name).read_text(encoding="utf-8")


def _read(name: str) -> str:
    path = ASSET_DIR / name
    try:
        return _asset(name, path.stat().st_mtime)
    except OSError as exc:
        raise FileNotFoundError(
            f"UI asset {name} is missing from {ASSET_DIR}. The web UI cannot render "
            f"without it ({exc})."
        ) from exc


def _json_for_script(obj) -> str:
    """json.dumps() is not safe inside a <script> element.

    The three examples are constants today, so nothing user-supplied reaches
    here, but the examples are exactly the kind of thing that gets edited to
    match a new scheme later -- and a question containing `</script>` would
    close the element and inject markup. Escaping `<`, `>` and `&` as \\uXXXX
    is safe to do on the dumped JSON: none of those are JSON structural
    characters, so every one of them sits inside a string literal, where the
    \\u escape is decoded back to the original character by the JS parser.
    \\u2028/\\u2029 are line terminators in JS but legal in JSON strings.

    Backslashes are deliberately *not* re-escaped: json.dumps already emitted
    them, and doubling them again would decode to a literal double backslash.
    """
    raw = json.dumps(obj)
    for ch, esc in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"),
                    ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        raw = raw.replace(ch, esc)
    return raw


def _for_script_block(js: str) -> str:
    """Defuse a literal `</script` inside inlined JavaScript.

    The HTML parser ends the element at the first `</script` it sees, whatever
    the JS thinks. A string like that can arrive from a regex or a comment
    without anyone intending to break out, and the result is a silently
    truncated script rather than an error.
    """
    return js.replace("</script", "<\\/script")


def render_page() -> str:
    ex_buttons = "\n".join(
        f'<button type="button" class="ex">{html.escape(q)}</button>'
        for q in EXAMPLE_QUESTIONS
    )
    page = _read("index.html")
    for placeholder, value in (
        ("{welcome}", html.escape(WELCOME)),
        ("{examples}", ex_buttons),
        ("{examples_json}", _json_for_script(EXAMPLE_QUESTIONS)),
        ("{disclaimer}", html.escape(DISCLAIMER_LONG)),
        ("{app_css}", _read("app.css")),
        ("{app_js}", _for_script_block(_read("app.js"))),
    ):
        page = page.replace(placeholder, value)
    return page


#: The UI loads nothing from the network: no CDN, no webfont, no analytics.
#: Inline style/script are permitted because that is how the single-file page is
#: assembled; `connect-src 'self'` still confines every fetch to this server, and
#: `frame-ancestors 'none'` stops another site framing the tool.
CSP = ("default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
       "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; "
       "frame-ancestors 'none'")


def serve(host: str = "127.0.0.1", port: int = 8000, pipe: FAQPipeline | None = None):
    pipe = pipe or FAQPipeline()
    lock = threading.Lock()
    JSON_CT = "application/json; charset=utf-8"
    NOT_FOUND = json.dumps({"error": "not found"}).encode()

    class Handler(BaseHTTPRequestHandler):
        server_version = "MFFacts/2.0"

        def log_message(self, fmt, *args):  # keep the console readable; never the body
            pass

        def route(self) -> str:
            """Path only. Comparing self.path directly 404s '/healthz?x=1',
            which is a perfectly ordinary thing for a health probe to send."""
            return urlsplit(self.path).path

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", CSP)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            route = self.route()
            if route in ("/", "/index.html"):
                try:
                    body = render_page().encode()
                except FileNotFoundError as exc:
                    self._send(500, json.dumps({"error": str(exc)}).encode(), JSON_CT)
                    return
                self._send(200, body, "text/html; charset=utf-8")
            elif route == "/healthz":
                self._send(200, json.dumps({"ok": True, "stats": pipe.stats()}).encode(), JSON_CT)
            elif route == "/examples":
                self._send(200, json.dumps(
                    {"welcome": WELCOME, "examples": EXAMPLE_QUESTIONS}).encode(), JSON_CT)
            else:
                self._send(404, NOT_FOUND, JSON_CT)

        def do_POST(self) -> None:
            if self.route() != "/ask":
                self._send(404, NOT_FOUND, JSON_CT)
                return
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            try:
                query = (json.loads(raw or b"{}") or {}).get("q", "")
            except ValueError:
                query = ""
            # `{"q": 123}` is valid JSON, so it got through the parse above and
            # then reached the pipeline as an int, which raised somewhere
            # downstream and answered a bad request with a 500. Anything that is
            # not a string is not a question; degrade to the same empty-input
            # refusal a blank box gets.
            if not isinstance(query, str):
                query = ""
            # Any unhandled exception here used to escape to socketserver, which
            # logs it and closes the socket with no response at all -- the
            # browser shows an opaque network error and the cause is only visible
            # in the server's stderr. A local tool should say what went wrong.
            try:
                with lock:                  # one writer at a time: one shared sqlite conn
                    ans = pipe.ask(query)
                payload = json.dumps(ans.to_dict(), ensure_ascii=False)
            except Exception as exc:        # noqa: BLE001 - reported, not swallowed
                self._send(500, json.dumps(
                    {"error": f"{type(exc).__name__}: {exc}"}).encode(),
                    "application/json; charset=utf-8")
                return
            self._send(200, payload.encode(), JSON_CT)

    # On Windows SO_REUSEADDR does not mean "ignore TIME_WAIT" as it does on
    # Unix; it lets a *second* socket bind a port that already has a live
    # listener. So a duplicate `serve` does not fail with EADDRINUSE -- it
    # starts, prints the banner, and connections are delivered to whichever
    # socket the kernel picks. That is worse than a crash: the user believes
    # they restarted the server and silently talks to a stale process. Probe the
    # port first and refuse, rather than relying on bind() to fail.
    with socket.socket() as probe:
        probe.settimeout(0.5)
        if probe.connect_ex((host, port)) == 0:
            raise OSError(
                f"port {port} on {host} is already in use, so this server would "
                f"compete with whatever is listening on it. Stop it, or pass a "
                f"different --port.")

    if host not in ("127.0.0.1", "localhost", "::1"):
        # The spec's own pitfall: this is a prototype, not a deployment. It has
        # no auth and no rate limiting, so binding it to a routable interface
        # publishes an unauthenticated query endpoint to the network.
        print(f"WARNING: binding {host}, not loopback. There is no auth and no rate "
              f"limiting; anything that can reach this port can query the index.",
              file=sys.stderr)

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"MF facts assistant -> http://{host}:{port}   (facts-only. No investment advice.)")
    print("Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
        pipe.close()
