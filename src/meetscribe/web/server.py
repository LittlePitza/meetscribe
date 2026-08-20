"""The local web panel: a threaded HTTP server on loopback, streaming over SSE.

**Why SSE and not WebSocket.** The traffic is one-directional -- the pipeline
publishes, the browser renders -- and SSE gives three things for free that would
otherwise be hand-written: automatic reconnection, a replay cursor
(``Last-Event-ID``), and plain HTTP semantics that need no framing library. The two
control actions the panel does send (rename a speaker, stop the session) are
ordinary POSTs, and keeping them as POSTs is what lets the CSRF defence below be
simple and airtight. A WebSocket would add a second protocol, a second auth path,
and an origin check that is easy to get wrong.

**Why the standard library's HTTP server.** With no asyncio anywhere else in the
process there is no event loop to integrate with, and a threaded server matches the
pipeline's threading model exactly. It also removes an entire dependency from a
tool that records private conversations, which is worth more here than the
throughput a production framework would add for a single local user.

**The security posture, control by control.**

* *Bound to 127.0.0.1 only.* Configurable in principle, refused in practice: the
  config validator rejects any other host unless ``allow_non_local`` is set
  explicitly, because a live transcript on a LAN interface is a leak, not a
  preference.
* *A 256-bit session token* on every data endpoint. Loopback binding stops the
  network; the token stops *another user account on the same machine* from simply
  opening the port. Compared in constant time.
* *Host header validation* -- defeats DNS rebinding, where a malicious page
  resolves its own hostname to 127.0.0.1 and talks to this server from a foreign
  origin.
* *Origin validation plus a token required in a custom header for POSTs.* A
  cross-origin page cannot set ``X-MeetScribe-Token`` without a CORS preflight, and
  this server never emits CORS headers, so the browser blocks the request before it
  is made. Simple form-based CSRF is impossible for the same reason.
* *No CORS, ever.* There is no legitimate cross-origin consumer.
* *Static assets are ungated; every data endpoint is gated.* The HTML and JS contain
  no meeting content -- they ship in the repository -- so gating them would add
  friction without adding secrecy.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from meetscribe.errors import PanelError
from meetscribe.security.secrets import new_session_token, token_matches

log = logging.getLogger(__name__)

__all__ = ["PanelServer", "STATIC_DIR"]

STATIC_DIR = Path(__file__).parent / "static"

_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "base-uri 'none'; "
    "form-action 'none'; "
    "frame-ancestors 'none'"
)

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}

TOKEN_HEADER = "X-MeetScribe-Token"  # noqa: S105 - a header name, not a secret

MAX_BODY_BYTES = 64_000
"""Ceiling on a control request. The two real ones carry a speaker id and a name;
anything larger is a mistake or an attempt, and is refused without being read."""


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    """A threaded server that does not shout about normal disconnections.

    Closing a browser tab aborts an in-flight SSE read. The stdlib's default
    ``handle_error`` prints a full traceback to stderr for that, which would land
    in the middle of the user's CLI output during a meeting -- an alarming display
    for an entirely routine event. Genuine failures still reach the log.
    """

    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        exc = sys.exception()
        if isinstance(
            exc,
            (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, TimeoutError),
        ):
            log.debug("panel client %s disconnected: %s", client_address, exc)
            return
        # socketserver calls this from inside its own `except`, so the active
        # exception is still set and the traceback is real.
        log.exception("panel request from %s failed", client_address)  # noqa: LOG004


class PanelServer:
    """Owns the HTTP server thread and the session token."""

    def __init__(
        self,
        bus,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        token: str | None = None,
        heartbeat_s: float = 15.0,
        state_provider: Callable[[], dict] | None = None,
        on_rename: Callable[[str, str], int] | None = None,
        on_stop: Callable[[], None] | None = None,
    ) -> None:
        self.bus = bus
        self.host = host
        self.requested_port = port
        self.token = token or new_session_token()
        self.heartbeat_s = heartbeat_s
        self.state_provider = state_provider
        self.on_rename = on_rename
        self.on_stop = on_stop
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ control

    def start(self) -> str:
        if self._server is not None:
            raise PanelError("panel already started")
        handler = _make_handler(self)
        try:
            self._server = _QuietThreadingHTTPServer(
                (self.host, self.requested_port), handler
            )
        except OSError as exc:
            raise PanelError(
                f"cannot bind {self.host}:{self.requested_port}: {exc}",
                user_message=(
                    f"No se pudo abrir el panel en {self.host}:{self.requested_port} "
                    f"({exc.strerror or exc}). Cambia 'panel.port' en la "
                    "configuración o usa --panel-port."
                ),
            ) from exc
        self._server.daemon_threads = True
        # Long-lived SSE connections must not keep the process alive at shutdown.
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.2},
            name="panel-http",
            daemon=True,
        )
        self._thread.start()
        log.info("panel listening on %s", self.url)
        return self.url

    def stop(self, timeout: float = 5.0) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout)
        self._server = None
        self._thread = None

    @property
    def port(self) -> int:
        if self._server is None:
            return self.requested_port
        return int(self._server.server_address[1])

    @property
    def origin(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def url(self) -> str:
        """The full URL including the token. Never log this -- the redaction filter
        strips the token if it ever reaches a log record anyway."""
        return f"{self.origin}/?t={self.token}"

    # ------------------------------------------------------------------ checking

    def valid_host(self, header: str | None) -> bool:
        if not header:
            return False
        name = header.rsplit(":", 1)[0].strip("[]")
        return name in ("127.0.0.1", "localhost", "::1") or (
            self.host not in ("127.0.0.1", "localhost") and name == self.host
        )

    def valid_origin(self, header: str | None) -> bool:
        """Absent is fine (same-origin GETs omit it); present must match exactly."""
        if header is None or header == "null":
            return header is None
        parsed = urlparse(header)
        return (
            parsed.scheme == "http"
            and parsed.hostname in ("127.0.0.1", "localhost")
            and (parsed.port or 80) == self.port
        )


def _make_handler(panel: PanelServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "MeetScribe"
        sys_version = ""
        protocol_version = "HTTP/1.1"

        # ------------------------------------------------------------- plumbing

        def log_message(self, fmt: str, *args: object) -> None:
            # Default handler writes to stderr, which would put the token-bearing
            # request line straight onto the user's terminal.
            log.debug("panel %s - %s", self.address_string(), fmt % args)

        def _headers(self, status: HTTPStatus, content_type: str, length: int) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", _CSP)
            self.end_headers()

        def _send(
            self, status: HTTPStatus, body: bytes, content_type: str = "text/plain; charset=utf-8"
        ) -> None:
            self._headers(status, content_type, len(body))
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, status: HTTPStatus, payload: dict) -> None:
            self._send(
                status,
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def _deny(self, status: HTTPStatus, message: str) -> None:
            self._json(status, {"error": message})

        # ----------------------------------------------------------------- auth

        def _preflight_ok(self) -> bool:
            if not panel.valid_host(self.headers.get("Host")):
                self._deny(HTTPStatus.BAD_REQUEST, "host no permitido")
                return False
            if not panel.valid_origin(self.headers.get("Origin")):
                self._deny(HTTPStatus.FORBIDDEN, "origen no permitido")
                return False
            return True

        def _authorised(self, query: dict[str, list[str]], require_header: bool) -> bool:
            header_token = self.headers.get(TOKEN_HEADER)
            if require_header:
                # A cross-origin page cannot set a custom header without a CORS
                # preflight, and this server never answers one.
                if not token_matches(panel.token, header_token):
                    self._deny(HTTPStatus.UNAUTHORIZED, "token inválido")
                    return False
                return True
            supplied = header_token or (query.get("t") or [""])[0]
            if not token_matches(panel.token, supplied):
                self._deny(HTTPStatus.UNAUTHORIZED, "token inválido")
                return False
            return True

        # ------------------------------------------------------------------ GET

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            route = parsed.path

            if route in ("/", "/index.html"):
                self._static("index.html")
                return
            if route.startswith("/static/"):
                self._static(route[len("/static/") :])
                return

            if not self._preflight_ok():
                return

            if route == "/events":
                if not self._authorised(query, require_header=False):
                    return
                self._stream(query)
                return
            if route == "/api/state":
                if not self._authorised(query, require_header=False):
                    return
                state = panel.state_provider() if panel.state_provider else {}
                self._json(HTTPStatus.OK, state)
                return

            self._deny(HTTPStatus.NOT_FOUND, "no encontrado")

        def do_HEAD(self) -> None:
            self.do_GET()

        # ----------------------------------------------------------------- POST

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)

            # The body is consumed *before* any check that can reject the request.
            # On a keep-alive connection an unread body is not discarded: it stays
            # in the socket, and the server then reads it as the start of the next
            # request line -- so one rejected POST breaks the following legitimate
            # one. Draining first makes every rejection path leave the connection
            # in a consistent state.
            raw = self._read_body()
            if raw is None:
                return
            if not self._preflight_ok():
                return
            if not self._authorised(query, require_header=True):
                return

            try:
                body = json.loads(raw or b"{}")
            except (json.JSONDecodeError, ValueError):
                self._deny(HTTPStatus.BAD_REQUEST, "json inválido")
                return
            if not isinstance(body, dict):
                self._deny(HTTPStatus.BAD_REQUEST, "se esperaba un objeto json")
                return

            if parsed.path == "/api/speaker":
                self._rename(body)
                return
            if parsed.path == "/api/stop":
                if panel.on_stop is None:
                    self._deny(HTTPStatus.NOT_IMPLEMENTED, "no hay sesión activa")
                    return
                threading.Thread(target=panel.on_stop, name="panel-stop", daemon=True).start()
                self._json(HTTPStatus.ACCEPTED, {"ok": True})
                return

            self._deny(HTTPStatus.NOT_FOUND, "no encontrado")

        def _read_body(self) -> bytes | None:
            """The request body, or ``None`` having already answered with an error.

            A body larger than the cap is not drained -- that would let anyone who
            can reach the port make the server read arbitrary volumes. The
            connection is closed instead, which is the only other way to leave no
            unread bytes behind.
            """
            header = self.headers.get("Content-Length")
            try:
                length = int(header) if header else 0
            except ValueError:
                self.close_connection = True
                self._deny(HTTPStatus.BAD_REQUEST, "'Content-Length' inválido")
                return None
            if length < 0 or length > MAX_BODY_BYTES:
                self.close_connection = True
                self._deny(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "cuerpo demasiado grande")
                return None
            if length == 0:
                return b""
            raw = self.rfile.read(length)
            if len(raw) < length:
                # The client went away mid-body; nothing left to keep alive for.
                self.close_connection = True
                self._deny(HTTPStatus.BAD_REQUEST, "cuerpo incompleto")
                return None
            return raw

        def _rename(self, body: dict) -> None:
            speaker_id = str(body.get("speaker_id") or "").strip()
            name = str(body.get("name") or "").strip()
            if not speaker_id or not name:
                self._deny(HTTPStatus.BAD_REQUEST, "faltan 'speaker_id' o 'name'")
                return
            if len(name) > 80:
                self._deny(HTTPStatus.BAD_REQUEST, "el nombre es demasiado largo")
                return
            if panel.on_rename is None:
                self._deny(HTTPStatus.NOT_IMPLEMENTED, "no hay sesión activa")
                return
            try:
                updated = panel.on_rename(speaker_id, name)
            except Exception as exc:
                log.warning("rename failed: %s", exc)
                self._deny(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._json(HTTPStatus.OK, {"ok": True, "segments": updated})

        # ------------------------------------------------------------------ SSE

        def _stream(self, query: dict[str, list[str]]) -> None:
            cursor = _cursor(self.headers.get("Last-Event-ID"), query)
            subscription = panel.bus.subscribe(after_seq=cursor)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "keep-alive")
            # Without this an intermediary may buffer the stream into uselessness.
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                while True:
                    event = subscription.get(timeout=panel.heartbeat_s)
                    if event is None:
                        if subscription.closed:
                            break
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    frame = (
                        f"id: {event.seq}\n"
                        f"event: {event.type}\n"
                        f"data: {event.to_json()}\n\n"
                    ).encode()
                    self.wfile.write(frame)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                log.debug("panel client disconnected")
            finally:
                subscription.close()

        # --------------------------------------------------------------- static

        def _static(self, relative: str) -> None:
            candidate = (STATIC_DIR / relative).resolve()
            try:
                candidate.relative_to(STATIC_DIR.resolve())
            except ValueError:
                # Path traversal attempt; treat as not found rather than confirming
                # anything about the filesystem layout.
                self._deny(HTTPStatus.NOT_FOUND, "no encontrado")
                return
            if not candidate.is_file():
                self._deny(HTTPStatus.NOT_FOUND, "no encontrado")
                return
            content_type = _CONTENT_TYPES.get(candidate.suffix, "application/octet-stream")
            self._send(HTTPStatus.OK, candidate.read_bytes(), content_type)

    return Handler


def _cursor(last_event_id: str | None, query: dict[str, list[str]]) -> int:
    """Resume point: the SSE header wins, the query parameter is the fallback."""
    for candidate in (last_event_id, (query.get("cursor") or ["0"])[0]):
        if candidate:
            try:
                return max(0, int(candidate))
            except (TypeError, ValueError):
                continue
    return 0
