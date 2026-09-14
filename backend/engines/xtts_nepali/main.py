"""xtts-nepali sidecar entry point.

Runs Coqui XTTS v2 with the Oshara Nepali fine-tune under the engine's own venv
(coqui-tts + transformers 4.57). Wire protocol, fd handling and heartbeats
mirror engines/pockettts/main.py::

    [ 4-byte big-endian uint32 length ][ N bytes UTF-8 JSON ]

    sidecar -> parent: {"op":"ready","engine":"xtts-nepali","sample_rate":24000}
    parent -> sidecar: {"op":"ping"} -> {"op":"pong","vram_mb":...}
    parent -> sidecar: {"op":"synthesize","text":"...","language":"ne",
                        "ref_audio":"/path/ref.wav","speed":1.0,"seed":42}
      -> {"op":"progress",...}* then {"op":"audio","audio_pcm_b64":"...",...}
    parent -> sidecar: {"op":"shutdown"} -> exit 0

Stdlib-only at import time; torch + TTS load on the first synthesize.

Nepali route. Stock coqui-tts has no Nepali text cleaners (``preprocess_text``
raises for ``ne``) and the fine-tune's vocab has no ``[ne]`` token, although its
config lists ``ne``. ``OMNIVOICE_XTTS_NEPALI_ROUTE`` picks how Nepali reaches the
GPT: ``hi`` (default) sends it as plain Hindi; ``ne`` runs the Hindi cleaners and
keeps the ``[ne]`` prefix, as the model card's ``language="ne"`` implies.

Defaults reproduce a plain ``Xtts.inference(text, language="hi",
temperature=0.7)`` call (so repetition_penalty 10.0, top_k 50, top_p 0.85, one
pass). The model card's settings are ``ne`` / 0.65 / 5.0.

Text within the tokenizer's per-language character limit goes to the GPT in one
pass. Longer text is split on the danda / sentence punctuation under that limit,
with a short gap between pieces, per the model card's advice against the GPT
drifting between sentences.

Other knobs: ``OMNIVOICE_XTTS_NEPALI_CHECKPOINT`` (``epoch-10`` recommended,
``epoch-20``), ``OMNIVOICE_XTTS_NEPALI_MODEL_DIR`` (local checkpoint folder,
skips the download), ``OMNIVOICE_XTTS_NEPALI_DEVICE`` (``cpu``/``cuda``),
``OMNIVOICE_XTTS_NEPALI_SPEAKER`` (built-in speaker used when no reference clip
is given), ``OMNIVOICE_XTTS_NEPALI_TEMPERATURE`` (default 0.7),
``OMNIVOICE_XTTS_NEPALI_REPETITION_PENALTY`` (default 10.0).
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
from collections import OrderedDict

# Mirrors services/subprocess_backend.py::MAX_FRAME_BYTES.
MAX_FRAME_BYTES = 64 * 1024 * 1024

XTTS_SAMPLE_RATE = 24_000

_REPO_ID = "Oshara/xtts-v2-nepali"
_CHECKPOINT = os.environ.get("OMNIVOICE_XTTS_NEPALI_CHECKPOINT", "epoch-10").strip()
_NE_ROUTE = os.environ.get("OMNIVOICE_XTTS_NEPALI_ROUTE", "hi").strip().lower()

#: Sampling defaults: a plain inference(language="hi", temperature=0.7) call.
_TEMPERATURE = float(os.environ.get("OMNIVOICE_XTTS_NEPALI_TEMPERATURE", "0.7"))
_REPETITION_PENALTY = float(os.environ.get("OMNIVOICE_XTTS_NEPALI_REPETITION_PENALTY", "10.0"))
_TOP_K = 50
_TOP_P = 0.85

#: Silence between split pieces, in seconds.
_GAP_S = 0.2

#: Emit a progress frame at least this often so the parent's recv watchdog and
#: the GPU pool's generate budget see a slow CPU synth as alive.
_HEARTBEAT_S = 5.0

#: ref_audio must be a local file path, not a URL (local-first; no SSRF).
_URL_RE = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.IGNORECASE)

_VOICE_CACHE_MAX = 8

#: Language names / ISO-639-3 codes the app may send -> XTTS language codes.
_LANG_ALIASES = {
    "nepali": "ne", "nep": "ne", "npi": "ne",
    "hindi": "hi", "hin": "hi",
    "english": "en", "eng": "en",
    "spanish": "es", "french": "fr", "german": "de", "italian": "it",
    "portuguese": "pt", "polish": "pl", "turkish": "tr", "russian": "ru",
    "dutch": "nl", "czech": "cs", "arabic": "ar", "hungarian": "hu",
    "korean": "ko", "japanese": "ja", "chinese": "zh-cn", "zh": "zh-cn",
}

_ZERO_WIDTH_RE = re.compile("[​‌‍⁠﻿]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[।॥?!])\s*|(?<=\.)\s+")

_MODEL = None
_voice_cache: OrderedDict[str, tuple] = OrderedDict()


# -- wire protocol -----------------------------------------------------------

_send_lock = threading.Lock()


def _send(stream, obj: dict) -> None:
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    with _send_lock:
        stream.write(struct.pack("!I", len(body)))
        stream.write(body)
        stream.flush()


def _recv(stream):
    header = stream.read(4)
    if len(header) < 4:
        return None  # EOF
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


def _measure_vram_mb() -> float:
    if "torch" not in sys.modules:
        return 0.0
    import torch

    if torch.cuda.is_available():
        return float(torch.cuda.memory_allocated()) / (1024 * 1024)
    return 0.0


class _Heartbeat:
    """Send progress frames every _HEARTBEAT_S while the block runs."""

    def __init__(self, stdout, stage: str) -> None:
        self._stdout = stdout
        self._stage = stage
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.percent = 0

    def _frame(self) -> dict:
        return {"op": "progress", "stage": self._stage, "percent": self.percent}

    def _run(self) -> None:
        while not self._stop.wait(_HEARTBEAT_S):
            _send(self._stdout, self._frame())

    def __enter__(self) -> "_Heartbeat":
        _send(self._stdout, self._frame())
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=_HEARTBEAT_S + 1)


# -- model -------------------------------------------------------------------


def _device() -> str:
    import torch

    wanted = os.environ.get("OMNIVOICE_XTTS_NEPALI_DEVICE", "").strip().lower()
    if wanted:
        return wanted
    return "cuda" if torch.cuda.is_available() else "cpu"


def _patch_nepali_tokenizer(tokenizer) -> None:
    """Let ``ne`` through coqui-tts's tokenizer using the Hindi cleaners."""
    original = tokenizer.preprocess_text

    def preprocess_text(txt, lang):
        return original(txt, "hi" if lang == "ne" else lang)

    tokenizer.preprocess_text = preprocess_text
    tokenizer.char_limits.setdefault("ne", tokenizer.char_limits.get("hi", 150))


def _load_model(stdout):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _Heartbeat(stdout, "loading_model"):
        from TTS.tts.configs.xtts_config import XttsConfig  # noqa: PLC0415
        from TTS.tts.models.xtts import Xtts  # noqa: PLC0415

        model_dir = os.environ.get("OMNIVOICE_XTTS_NEPALI_MODEL_DIR", "").strip()
        if not model_dir:
            from huggingface_hub import snapshot_download  # noqa: PLC0415

            root = snapshot_download(_REPO_ID, allow_patterns=[f"{_CHECKPOINT}/*"])
            model_dir = os.path.join(root, _CHECKPOINT)

        config = XttsConfig()
        config.load_json(os.path.join(model_dir, "config.json"))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_path=os.path.join(model_dir, "model.pth"),
            vocab_path=os.path.join(model_dir, "vocab.json"),
            speaker_file_path=os.path.join(model_dir, "speakers_xtts.pth"),
            eval=True,
            use_deepspeed=False,
        )
        model.to(_device())
        _patch_nepali_tokenizer(model.tokenizer)
        _MODEL = model
    return model


def _xtts_language(raw, supported) -> str:
    """Map the app's language value to an XTTS code. Empty/auto means Nepali;
    a specific unsupported language raises instead of mispronouncing."""
    if not raw:
        return "ne"
    s = str(raw).strip().lower()
    if s in ("", "auto", "multi", "na"):
        return "ne"
    s = _LANG_ALIASES.get(s, s)
    if s in supported:
        return s
    base = s.split("-")[0]
    base = _LANG_ALIASES.get(base, base)
    if base in supported:
        return base
    raise ValueError(
        f"XTTS Nepali does not support language {raw!r}; supported: {', '.join(supported)}."
    )


def _voice_latents(model, ref_audio):
    """(gpt_cond_latent, speaker_embedding) for a local reference clip, or for a
    built-in speaker when no clip is given. LRU-cached per file contents."""
    if ref_audio:
        if _URL_RE.match(ref_audio):
            raise ValueError(
                "ref_audio must be a local file path; URLs are not accepted (local-first)."
            )
        st = os.stat(ref_audio)
        key = f"{ref_audio}|m{st.st_mtime_ns}s{st.st_size}"
        cached = _voice_cache.get(key)
        if cached is not None:
            _voice_cache.move_to_end(key)
            return cached
        latents = model.get_conditioning_latents(audio_path=[ref_audio])
        _voice_cache[key] = latents
        if len(_voice_cache) > _VOICE_CACHE_MAX:
            _voice_cache.popitem(last=False)
        return latents

    speakers = model.speaker_manager.speakers if model.speaker_manager else {}
    name = os.environ.get("OMNIVOICE_XTTS_NEPALI_SPEAKER", "").strip() or next(iter(speakers), None)
    if not name or name not in speakers:
        raise ValueError(
            "No reference clip given and no built-in speaker found; pick a voice to clone."
        )
    entry = speakers[name]
    return entry["gpt_cond_latent"], entry["speaker_embedding"]


def _chunks(text: str, limit: int) -> list[str]:
    """Sentence pieces no longer than ``limit`` characters, speakable ones only."""
    text = re.sub(r"\s+", " ", _ZERO_WIDTH_RE.sub("", text)).strip()
    if len(text) <= limit:
        return [text] if any(ch.isalnum() for ch in text) else []
    out: list[str] = []
    for sent in _SENTENCE_SPLIT_RE.split(text):
        sent = sent.strip()
        while len(sent) > limit:
            cut = max(sent.rfind(",", 0, limit), sent.rfind(" ", 0, limit))
            if cut <= 0:
                cut = limit - 1
            head, sent = sent[: cut + 1].strip(), sent[cut + 1 :].strip()
            if head:
                out.append(head)
        if sent:
            out.append(sent)
    return [c for c in out if any(ch.isalnum() for ch in c)]


def _pcm_b64(audio) -> tuple[str, int]:
    import numpy as np

    arr = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (arr * 32767.0).astype(np.int16).tobytes()
    return base64.b64encode(pcm).decode("ascii"), int(arr.shape[0])


def _handle_synthesize(msg: dict, stdout) -> None:
    text = msg.get("text")
    if not text or not isinstance(text, str):
        raise ValueError("synthesize: missing or non-string 'text'")

    model = _load_model(stdout)

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    language = _xtts_language(msg.get("language"), model.config.languages)
    route = "hi" if language == "ne" and _NE_ROUTE == "hi" else language
    chunks = _chunks(text, model.tokenizer.char_limits.get(route, 250))
    if not chunks:
        raise ValueError("synthesize: text has nothing speakable")

    seed = msg.get("seed")
    if isinstance(seed, int) and not isinstance(seed, bool):
        torch.manual_seed(seed)
    try:
        speed = min(2.0, max(0.5, float(msg.get("speed") or 1.0)))
    except (TypeError, ValueError):
        speed = 1.0

    gpt_cond_latent, speaker_embedding = _voice_latents(model, msg.get("ref_audio") or None)
    gap = np.zeros(int(_GAP_S * XTTS_SAMPLE_RATE), dtype=np.float32)
    pieces = []
    with _Heartbeat(stdout, "synthesizing") as hb, torch.inference_mode():
        for i, chunk in enumerate(chunks):
            hb.percent = int(100 * i / len(chunks))
            out = model.inference(
                chunk,
                route,
                gpt_cond_latent,
                speaker_embedding,
                temperature=_TEMPERATURE,
                length_penalty=1.0,
                repetition_penalty=_REPETITION_PENALTY,
                top_k=_TOP_K,
                top_p=_TOP_P,
                speed=speed,
                enable_text_splitting=False,
            )
            if pieces:
                pieces.append(gap)
            pieces.append(np.asarray(out["wav"], dtype=np.float32).reshape(-1))

    pcm_b64, n_samples = _pcm_b64(np.concatenate(pieces))
    _send(stdout, {
        "op": "audio",
        "audio_pcm_b64": pcm_b64,
        "sample_rate": XTTS_SAMPLE_RATE,
        "n_samples": n_samples,
    })


# -- main loop ---------------------------------------------------------------


def main() -> int:
    # Coqui's model manager asks for CPML consent on stdin, which is the frame
    # channel here; weights load from explicit paths, but never risk a prompt.
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    stdin = sys.stdin.buffer
    # Frames go down a private fd; fd 1 points at stderr so library prints
    # cannot corrupt the framing (#1428, see engines/pockettts/main.py).
    _frame_fd = os.dup(1)
    os.dup2(2, 1)
    stdout = os.fdopen(_frame_fd, "wb")

    _send(stdout, {"op": "ready", "engine": "xtts-nepali", "sample_rate": XTTS_SAMPLE_RATE})

    while True:
        try:
            msg = _recv(stdin)
        except Exception as exc:
            _send(stdout, {
                "op": "error",
                "stage": "recv",
                "message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
            return 1
        if msg is None:
            return 0

        op = msg.get("op") if isinstance(msg, dict) else None
        try:
            if op == "ping":
                _send(stdout, {"op": "pong", "vram_mb": _measure_vram_mb()})
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
