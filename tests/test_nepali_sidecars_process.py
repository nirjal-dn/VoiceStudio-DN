"""The xtts-nepali and indic-parler-tts sidecars as real child processes.

Both are stdlib-only until the first synthesize, so the protocol can be driven
end to end without their venvs or weights: ready handshake, ping, bad requests
answered with error frames (the process survives), stdout pollution kept off
the frame channel, and clean shutdown."""
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ENGINES = Path(__file__).resolve().parents[1] / "backend" / "engines"
SIDECARS = {
    "xtts-nepali": ENGINES / "xtts_nepali" / "main.py",
    "indic-parler-tts": ENGINES / "indic_parler" / "main.py",
}


def _send(proc, obj):
    body = json.dumps(obj).encode("utf-8")
    proc.stdin.write(struct.pack("!I", len(body)) + body)
    proc.stdin.flush()


def _recv(proc):
    header = proc.stdout.read(4)
    assert len(header) == 4, "sidecar closed the frame channel"
    (n,) = struct.unpack("!I", header)
    return json.loads(proc.stdout.read(n).decode("utf-8"))


@pytest.fixture(params=sorted(SIDECARS))
def sidecar(request, tmp_path):
    engine = request.param
    proc = subprocess.Popen(
        [sys.executable, str(SIDECARS[engine])],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=tmp_path,
    )
    try:
        yield engine, proc
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            stream.close()


def test_handshake_ping_errors_and_shutdown(sidecar):
    engine, proc = sidecar
    ready = _recv(proc)
    assert ready["op"] == "ready" and ready["engine"] == engine and ready["sample_rate"] > 0

    _send(proc, {"op": "ping"})
    assert _recv(proc)["op"] == "pong"

    # Invalid requests answer with an error frame and never load a model.
    for bad in (
        {"op": "synthesize"},
        {"op": "synthesize", "text": ""},
        {"op": "synthesize", "text": 123},
        {"op": "teleport"},
    ):
        _send(proc, bad)
        reply = _recv(proc)
        assert reply["op"] == "error", reply
        assert "Traceback" not in reply["message"]

    # The same process still serves after the errors.
    _send(proc, {"op": "ping"})
    assert _recv(proc)["op"] == "pong"

    _send(proc, {"op": "shutdown"})
    assert proc.wait(timeout=10) == 0


def test_eof_on_stdin_exits_cleanly(sidecar):
    _, proc = sidecar
    assert _recv(proc)["op"] == "ready"
    proc.stdin.close()
    assert proc.wait(timeout=10) == 0


def test_garbage_frame_is_reported_and_exits_nonzero(sidecar):
    _, proc = sidecar
    assert _recv(proc)["op"] == "ready"
    proc.stdin.write(struct.pack("!I", 0x7FFFFFFF))  # absurd length: over the frame cap
    proc.stdin.flush()
    reply = _recv(proc)
    assert reply["op"] == "error" and reply["stage"] == "recv"
    assert proc.wait(timeout=10) == 1
