"""Loopback browser guard: CSRF from any website + DNS rebinding.

A page in the user's browser reaches the local API as a loopback client, which
core.auth trusts with every capability. These pin the guard that stops a
cross-site page (simple requests need no preflight) and a rebound Host.
"""
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from core.loopback_guard import LoopbackBrowserGuardMiddleware, host_allowed


async def _ok(request):
    return JSONResponse({"ok": True})


async def _ws(websocket):
    await websocket.accept()
    await websocket.send_text("hi")
    await websocket.close()


def _client(host="127.0.0.1"):
    app = Starlette(routes=[
        Route("/x", _ok, methods=["GET", "POST", "DELETE"]),
        WebSocketRoute("/ws", _ws),
    ])
    app.add_middleware(LoopbackBrowserGuardMiddleware)
    return TestClient(app, base_url="http://127.0.0.1:3900", client=(host, 5555))


@pytest.fixture(autouse=True)
def _default_env(monkeypatch):
    monkeypatch.delenv("OMNIVOICE_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("OMNIVOICE_ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("OMNIVOICE_UI_PORT", raising=False)


# ── CSRF: cross-site simple requests from a web page ─────────────────────────


@pytest.mark.parametrize("kwargs", [
    {},
    {"content": b"a=1", "headers": {"content-type": "text/plain"}},
    {"data": {"a": "1"}},
    {"files": {"f": ("x.txt", b"x")}},
])
def test_cross_site_post_is_rejected(kwargs):
    headers = {"origin": "https://evil.example", "sec-fetch-site": "cross-site"}
    headers.update(kwargs.pop("headers", {}))
    res = _client().post("/x", headers=headers, **kwargs)
    assert res.status_code == 403


def test_null_origin_and_cross_site_without_origin_are_rejected():
    assert _client().post("/x", headers={"origin": "null"}).status_code == 403
    assert _client().delete("/x", headers={"sec-fetch-site": "cross-site"}).status_code == 403


@pytest.mark.parametrize("origin", [
    "tauri://localhost",
    "http://tauri.localhost",
    "http://localhost:3901",
    "http://127.0.0.1:3901",
    "http://127.0.0.1:3900",  # the backend-served UI posting to itself
])
def test_first_party_origins_pass(origin):
    assert _client().post("/x", headers={"origin": origin}).status_code == 200


def test_native_clients_without_browser_headers_pass():
    assert _client().post("/x").status_code == 200


def test_safe_methods_are_not_origin_checked():
    res = _client().get("/x", headers={"origin": "https://evil.example"})
    assert res.status_code == 200


def test_non_loopback_clients_are_left_to_the_auth_gates():
    res = _client(host="192.168.1.20").post("/x", headers={"origin": "https://evil.example"})
    assert res.status_code == 200


# ── DNS rebinding: attacker hostname resolving to 127.0.0.1 ─────────────────


def test_rebound_host_is_rejected_even_for_same_origin_reads():
    client = _client()
    assert client.get("/x", headers={"host": "evil.example:3900"}).status_code == 403
    res = client.post(
        "/x",
        headers={"host": "evil.example:3900", "origin": "http://evil.example:3900"},
    )
    assert res.status_code == 403


@pytest.mark.parametrize("host", [
    "127.0.0.1:3900", "localhost:3900", "[::1]:3900", "tauri.localhost",
    "gpu-box.tailnet-name.ts.net",
])
def test_local_and_tailscale_hosts_are_allowed(host):
    assert host_allowed(host)


def test_extra_hosts_come_from_env(monkeypatch):
    assert not host_allowed("studio.lan:3900")
    monkeypatch.setenv("OMNIVOICE_ALLOWED_HOSTS", "studio.lan, *.home.arpa")
    assert host_allowed("studio.lan:3900")
    assert host_allowed("box.home.arpa")
    assert not host_allowed("home.arpa.evil.example")


# ── WebSockets have no CORS at all ───────────────────────────────────────────


def test_cross_site_websocket_handshake_is_rejected():
    with pytest.raises(WebSocketDisconnect) as exc:
        with _client().websocket_connect(
            "/ws", headers={"host": "127.0.0.1:3900", "origin": "https://evil.example"},
        ) as ws:
            ws.receive_text()
    assert exc.value.code == 1008


def test_first_party_websocket_handshake_passes():
    with _client().websocket_connect(
        "/ws", headers={"host": "127.0.0.1:3900", "origin": "tauri://localhost"},
    ) as ws:
        assert ws.receive_text() == "hi"
