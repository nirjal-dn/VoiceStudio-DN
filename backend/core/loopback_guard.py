"""Browser-origin guard for loopback callers (CSRF + DNS rebinding).

A loopback caller is trusted by socket address alone (``core.auth``). A web
page open in the user's browser is also a loopback caller, so without this
guard any site could send state-changing requests to the local API (a form
POST or ``fetch(..., {mode: "no-cors"})`` needs no CORS preflight), and a
DNS-rebound site could read responses too.

Two checks, applied only to loopback clients (remote/LAN/Docker callers go
through the PIN and API-key gates instead):

* **Host allow-list** (every request): the ``Host`` header must name this
  machine — ``localhost``, ``127.0.0.1``, ``[::1]``, ``tauri.localhost``, a
  host from ``OMNIVOICE_ALLOWED_ORIGINS``, any ``*.ts.net`` Tailscale name, or
  an entry of ``OMNIVOICE_ALLOWED_HOSTS``. A rebound attacker domain fails.
* **Origin check** (non-GET/HEAD/OPTIONS requests and WebSocket handshakes): a
  request carrying an ``Origin`` must come from an allowed origin or from the
  backend's own (allow-listed) origin; with no ``Origin``, a ``Sec-Fetch-Site``
  of ``cross-site``/``same-site`` is rejected. CLI, MCP and SDK clients send
  neither header and are unaffected.
"""
from __future__ import annotations

import os
from urllib.parse import urlsplit

from core.auth import is_loopback
from core.csrf import (
    SAFE_HTTP_METHODS,
    _destination_origin,
    _origin_tuple,
    configured_allowed_origins,
)

_DEFAULT_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "tauri.localhost"})
# Tailscale MagicDNS names are issued by Tailscale, so an attacker cannot point
# one at this machine's loopback; `tailscale serve` proxies them from loopback.
_DEFAULT_SUFFIXES = (".ts.net",)


def _header(scope, name: bytes) -> str:
    for key, value in scope.get("headers") or ():
        if key == name:
            return value.decode("latin-1")
    return ""


def _hostname(host_header: str) -> str | None:
    try:
        return urlsplit(f"http://{host_header}").hostname
    except ValueError:
        return None


def host_allowed(host_header: str) -> bool:
    """Whether a loopback request's Host header names this machine."""
    if not host_header:
        return True  # HTTP/1.0-style clients; browsers always send Host
    hostname = _hostname(host_header)
    if not hostname:
        return False
    exact = set(_DEFAULT_HOSTS)
    suffixes = list(_DEFAULT_SUFFIXES)
    exact.update(origin[1] for origin in configured_allowed_origins())
    for entry in os.environ.get("OMNIVOICE_ALLOWED_HOSTS", "").split(","):
        entry = entry.strip().lower()
        if entry.startswith("*."):
            suffixes.append(entry[1:])
        elif entry.startswith("."):
            suffixes.append(entry)
        elif entry:
            exact.add(entry)
    return hostname in exact or any(hostname.endswith(s) for s in suffixes)


def origin_allowed_for_loopback(connection) -> bool:
    """Origin/Sec-Fetch check for a state-changing loopback request."""
    headers = connection.headers
    origin_value = headers.get("origin")
    if origin_value is not None:
        presented = _origin_tuple(origin_value)
        if presented is None:  # "null" or malformed
            return False
        # The backend-served UI posts to its own origin; the Host behind that
        # origin already passed host_allowed(), so rebinding cannot reach here.
        return (
            presented in configured_allowed_origins()
            or presented == _destination_origin(connection)
        )
    return headers.get("sec-fetch-site", "").lower() not in {"cross-site", "same-site"}


class LoopbackBrowserGuardMiddleware:
    """Pure ASGI: rejects rebound hosts and cross-site browser writes from loopback."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        client = scope.get("client")
        if not client or not is_loopback(client[0]):
            return await self.app(scope, receive, send)

        from starlette.requests import HTTPConnection
        from starlette.responses import JSONResponse

        rejected = None
        if not host_allowed(_header(scope, b"host")):
            rejected = "host not allowed"
        elif scope["type"] == "websocket" or (
            str(scope.get("method", "GET")).upper() not in SAFE_HTTP_METHODS
        ):
            if not origin_allowed_for_loopback(HTTPConnection(scope)):
                rejected = "browser origin rejected"

        if rejected is None:
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            await receive()  # websocket.connect
            await send({"type": "websocket.close", "code": 1008})
            return
        resp = JSONResponse({"detail": rejected}, status_code=403)
        return await resp(scope, receive, send)
