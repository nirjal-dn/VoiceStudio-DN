"""indic-parler-tts sidecar: same length-prefixed JSON protocol as
engines/xtts_nepali/main.py (ready / ping / synthesize / shutdown).

Stdlib-only at import time; torch + parler_tts load on the first synthesize.
The request's ``instruct`` is the voice description; without one, the Nepali
speaker "Amrita" is used (``OMNIVOICE_INDIC_PARLER_DESCRIPTION`` overrides).
Text is split into sentences (danda / ? ! .) and rendered one at a time, since
Parler drifts on long inputs.
"""
from __future__ import annotations

import base64
import json
import os
import re
import struct
import sys
import threading
import traceback

MAX_FRAME_BYTES = 64 * 1024 * 1024
REPO_ID = "ai4bharat/indic-parler-tts"
DEFAULT_DESCRIPTION = os.environ.get(
    "OMNIVOICE_INDIC_PARLER_DESCRIPTION",
    "Amrita speaks with a clear voice at a moderate pace. "
    "The recording is of very high quality, with no background noise.",
)
_HEARTBEAT_S = 5.0
_GAP_S = 0.2
_SENTENCE_END = re.compile(r"(?<=[।॥?!])\s*|(?<=\.)\s+")

_send_lock = threading.Lock()
_model = None


def _send(stream, obj: dict) -> None:
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    with _send_lock:
        stream.write(struct.pack("!I", len(body)))
        stream.write(body)
        stream.flush()


def _recv(stream):
    header = stream.read(4)
    if len(header) < 4:
        return None
    (n,) = struct.unpack("!I", header)
    if n > MAX_FRAME_BYTES:
        raise IOError(f"frame too large: {n}")
    body = bytearray()
    while len(body) < n:
        chunk = stream.read(n - len(body))
        if not chunk:
            raise IOError("short read")
        body.extend(chunk)
    return json.loads(bytes(body).decode("utf-8"))


def sentences(text: str) -> list[str]:
    parts = (p.strip() for p in _SENTENCE_END.split(" ".join(text.split())))
    return [p for p in parts if any(ch.isalnum() for ch in p)]


class _Heartbeat:
    def __init__(self, stdout, stage: str) -> None:
        self.stdout, self.stage, self.percent = stdout, stage, 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(_HEARTBEAT_S):
            _send(self.stdout, {"op": "progress", "stage": self.stage, "percent": self.percent})

    def __enter__(self):
        _send(self.stdout, {"op": "progress", "stage": self.stage, "percent": 0})
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=_HEARTBEAT_S + 1)


def _load(stdout):
    global _model
    if _model is None:
        with _Heartbeat(stdout, "loading_model"):
            import torch
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer

            device = os.environ.get("OMNIVOICE_INDIC_PARLER_DEVICE") or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            model = ParlerTTSForConditionalGeneration.from_pretrained(REPO_ID).to(device)
            tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
            description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
            _model = (model, tokenizer, description_tokenizer, device)
    return _model


def _handle_synthesize(msg: dict, stdout) -> None:
    text = msg.get("text")
    if not text or not isinstance(text, str):
        raise ValueError("synthesize: missing or non-string 'text'")
    parts = sentences(text)
    if not parts:
        raise ValueError("synthesize: text has nothing speakable")
    model, tokenizer, description_tokenizer, device = _load(stdout)

    import numpy as np
    import torch

    seed = msg.get("seed")
    if isinstance(seed, int) and not isinstance(seed, bool):
        torch.manual_seed(seed)
    description = (msg.get("instruct") or "").strip() or DEFAULT_DESCRIPTION
    desc = description_tokenizer(description, return_tensors="pt").to(device)
    sr = int(model.config.sampling_rate)
    gap = np.zeros(int(_GAP_S * sr), dtype=np.float32)
    pieces = []
    with _Heartbeat(stdout, "synthesizing") as hb, torch.inference_mode():
        for i, part in enumerate(parts):
            hb.percent = int(100 * i / len(parts))
            prompt = tokenizer(part, return_tensors="pt").to(device)
            audio = model.generate(
                input_ids=desc.input_ids,
                attention_mask=desc.attention_mask,
                prompt_input_ids=prompt.input_ids,
                prompt_attention_mask=prompt.attention_mask,
            )
            if pieces:
                pieces.append(gap)
            pieces.append(audio.cpu().numpy().astype(np.float32).reshape(-1))

    arr = np.clip(np.concatenate(pieces), -1.0, 1.0)
    pcm = (arr * 32767.0).astype(np.int16).tobytes()
    _send(stdout, {
        "op": "audio",
        "audio_pcm_b64": base64.b64encode(pcm).decode("ascii"),
        "sample_rate": sr,
        "n_samples": int(arr.shape[0]),
    })


def main() -> int:
    stdin = sys.stdin.buffer
    # Frames go down a private fd; library prints on fd 1 go to stderr (#1428).
    frame_fd = os.dup(1)
    os.dup2(2, 1)
    stdout = os.fdopen(frame_fd, "wb")
    _send(stdout, {"op": "ready", "engine": "indic-parler-tts", "sample_rate": 44100})
    while True:
        try:
            msg = _recv(stdin)
        except Exception as exc:
            _send(stdout, {"op": "error", "stage": "recv", "message": f"{type(exc).__name__}: {exc}"})
            return 1
        if msg is None:
            return 0
        op = msg.get("op") if isinstance(msg, dict) else None
        try:
            if op == "ping":
                _send(stdout, {"op": "pong", "vram_mb": 0})
            elif op == "synthesize":
                _handle_synthesize(msg, stdout)
            elif op == "shutdown":
                return 0
            else:
                _send(stdout, {"op": "error", "stage": "dispatch", "message": f"unknown op: {op!r}"})
        except Exception as exc:
            _send(stdout, {
                "op": "error",
                "stage": op or "unknown",
                "message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })


if __name__ == "__main__":
    sys.exit(main())
