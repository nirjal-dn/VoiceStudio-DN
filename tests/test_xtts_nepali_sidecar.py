"""xtts-nepali sidecar: wire protocol, chunking, language routing, reference
handling and synthesis orchestration — with a fake XTTS model (no coqui-tts,
no weights, CPU only). The sidecar is loaded by path because it runs in its
own venv."""
import base64
import importlib.util
import io
import json
import os
import struct
import unicodedata
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MAIN = _ROOT / "backend" / "engines" / "xtts_nepali" / "main.py"


@pytest.fixture
def xtts(monkeypatch):
    saved = os.environ.get("TORCHAUDIO_USE_TORCHCODEC")
    spec = importlib.util.spec_from_file_location("xtts_nepali_sidecar_t", _MAIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    if saved is None:
        os.environ.pop("TORCHAUDIO_USE_TORCHCODEC", None)
    else:
        os.environ["TORCHAUDIO_USE_TORCHCODEC"] = saved


class _CharTokenizer:
    """One token per code point: Devanagari expands like the real BPE does."""

    def __init__(self):
        self.langs = []

    def encode(self, text, lang=None):
        self.langs.append(lang)
        return list(text)


class _FakeXtts:
    def __init__(self, max_tokens=40, speakers=None):
        self.tokenizer = _CharTokenizer()
        self.config = type("C", (), {"languages": ["en", "hi", "ne", "es"]})()
        self.args = type("A", (), {"gpt_max_text_tokens": max_tokens})()
        self.speaker_manager = type("S", (), {"speakers": speakers or {}})()
        self.inference_calls = []
        self.latent_calls = []

    def get_conditioning_latents(self, audio_path):
        self.latent_calls.append(list(audio_path))
        return ("gpt-latent", "speaker-embedding")

    def inference(self, text, language, gpt_cond_latent, speaker_embedding, **kw):
        self.inference_calls.append({"text": text, "language": language, **kw})
        return {"wav": np.full(2400, 0.5, dtype=np.float32)}  # 0.1 s at 24 kHz


def _frames(stream: io.BytesIO, xtts):
    stream.seek(0)
    out = []
    while True:
        frame = xtts._recv(stream)
        if frame is None:
            return out
        out.append(frame)


def _ref_wav(tmp_path, seconds=3.0, sr=22050):
    import soundfile as sf

    path = tmp_path / "ref.wav"
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(path, (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), sr)
    return str(path)


# ── wire protocol ───────────────────────────────────────────────────────────


def test_frames_round_trip_unicode(xtts):
    buf = io.BytesIO()
    msg = {"op": "synthesize", "text": "नमस्ते, क्षत्रिय ज्ञान।", "language": "ne"}
    xtts._send(buf, msg)
    buf.seek(0)
    assert xtts._recv(buf) == msg
    assert xtts._recv(buf) is None  # clean EOF


def test_oversized_and_truncated_frames_are_rejected(xtts):
    too_big = io.BytesIO(struct.pack("!I", xtts.MAX_FRAME_BYTES + 1))
    with pytest.raises(IOError, match="too large"):
        xtts._recv(too_big)
    body = json.dumps({"op": "ping"}).encode()
    truncated = io.BytesIO(struct.pack("!I", len(body) + 10) + body)
    with pytest.raises(IOError, match="short read"):
        xtts._recv(truncated)


# ── language routing ────────────────────────────────────────────────────────


@pytest.mark.parametrize(("raw", "code"), [
    (None, "ne"), ("", "ne"), ("Auto", "ne"), ("multi", "ne"),
    ("Nepali", "ne"), ("npi", "ne"), ("nep", "ne"), ("ne-NP", "ne"),
    ("Hindi", "hi"), ("English", "en"), ("es-MX", "es"),
])
def test_language_values_map_to_xtts_codes(xtts, raw, code):
    assert xtts._xtts_language(raw, ["en", "hi", "ne", "es"]) == code


def test_unsupported_language_raises_instead_of_mispronouncing(xtts):
    with pytest.raises(ValueError, match="does not support"):
        xtts._xtts_language("Swahili", ["en", "hi", "ne"])


# ── chunking ────────────────────────────────────────────────────────────────


def test_long_nepali_text_respects_token_limit_and_loses_nothing(xtts):
    text = (
        "काठमाडौं उपत्यकामा आज बिहानदेखि हल्का वर्षा भइरहेको छ। "
        "मौसमविद्का अनुसार भोलि पनि बादल लाग्ने सम्भावना रहेको छ। "
    ) * 12
    chunks = xtts._chunks(text, _CharTokenizer(), "hi", 60)
    assert len(chunks) > 1
    assert all(len(c) < 60 for c in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_single_overlong_word_is_split_on_syllables(xtts):
    word = "संयुक्ताक्षरहरूक्षत्रज्ञश्रीद्वाराप्रतिष्ठान" * 6
    for limit in range(6, 60, 5):
        chunks = xtts._chunks(word, _CharTokenizer(), "hi", limit)
        assert "".join(chunks) == word
        for prev, nxt in zip(chunks, chunks[1:]):
            assert unicodedata.category(nxt[0]) not in ("Mn", "Mc"), (limit, prev[-3:], nxt[:3])
            assert not prev.endswith("्"), (limit, prev[-3:], nxt[:3])


def test_zero_width_chars_and_whitespace_are_normalized(xtts):
    chunks = xtts._chunks("नमस्ते​   साथी﻿।\n\nकस्तो छ?", _CharTokenizer(), "hi", 400)
    assert chunks == ["नमस्ते साथी।", "कस्तो छ?"]


@pytest.mark.parametrize("text", ["", "   ", "।।", "... ? !", "​​"])
def test_nothing_speakable_yields_no_chunks(xtts, text):
    assert xtts._chunks(text, _CharTokenizer(), "hi", 400) == []


# ── reference audio ─────────────────────────────────────────────────────────


def test_reference_urls_are_refused(xtts):
    with pytest.raises(ValueError, match="local file path"):
        xtts._voice_latents(_FakeXtts(), "https://example.com/voice.wav")


def test_missing_reference_file_raises(xtts, tmp_path):
    with pytest.raises(FileNotFoundError):
        xtts._voice_latents(_FakeXtts(), str(tmp_path / "gone.wav"))


def test_reference_latents_are_cached_by_content(xtts, tmp_path):
    import shutil

    import soundfile as sf

    model = _FakeXtts()
    ref = _ref_wav(tmp_path)
    xtts._voice_latents(model, ref)
    xtts._voice_latents(model, ref)
    assert len(model.latent_calls) == 1
    # The same clip re-uploaded to a new temp path (ad-hoc clone, every generate).
    copy = tmp_path / "upload-2.wav"
    shutil.copy(ref, copy)
    os.utime(copy, ns=(1, 1))
    xtts._voice_latents(model, str(copy))
    assert len(model.latent_calls) == 1
    # Different audio is a different voice.
    other = tmp_path / "other.wav"
    sf.write(other, np.zeros(22050, dtype=np.float32), 22050)
    xtts._voice_latents(model, str(other))
    assert len(model.latent_calls) == 2


def test_no_reference_uses_a_built_in_speaker_or_explains(xtts, monkeypatch):
    monkeypatch.delenv("OMNIVOICE_XTTS_NEPALI_SPEAKER", raising=False)
    with_speaker = _FakeXtts(speakers={"Ana Florence": {"gpt_cond_latent": "g", "speaker_embedding": "s"}})
    assert xtts._voice_latents(with_speaker, None) == ("g", "s")
    with pytest.raises(ValueError, match="pick a voice"):
        xtts._voice_latents(_FakeXtts(), None)


# ── synthesis orchestration ─────────────────────────────────────────────────


def _synth(xtts, model, msg):
    xtts._MODEL = model
    out = io.BytesIO()
    xtts._handle_synthesize(msg, out)
    frames = _frames(out, xtts)
    audio = frames[-1]
    assert audio["op"] == "audio"
    return frames, audio


def test_nepali_is_normalized_and_sent_on_the_hindi_route(xtts, tmp_path, monkeypatch):
    monkeypatch.setattr(xtts, "_NE_ROUTE", "hi")
    model = _FakeXtts(max_tokens=400)
    frames, audio = _synth(xtts, model, {
        "op": "synthesize", "text": "मूल्य रु. 1,500 मात्र।",
        "language": "Nepali", "ref_audio": _ref_wav(tmp_path),
    })
    assert [c["language"] for c in model.inference_calls] == ["hi"]
    assert model.inference_calls[0]["text"] == "मूल्य रु. एक हजार पाँच सय मात्र।"
    assert audio["sample_rate"] == 24000
    pcm = np.frombuffer(base64.b64decode(audio["audio_pcm_b64"]), dtype=np.int16)
    assert pcm.size == audio["n_samples"] == 2400
    assert frames[0]["op"] == "progress"


def test_multiple_chunks_are_joined_with_silence_gaps(xtts, tmp_path):
    model = _FakeXtts(max_tokens=40)
    text = "पहिलो वाक्य यहाँ छ। दोस्रो वाक्य यहाँ छ। तेस्रो वाक्य यहाँ छ।"
    _, audio = _synth(xtts, model, {"op": "synthesize", "text": text, "ref_audio": _ref_wav(tmp_path)})
    n = len(model.inference_calls)
    assert n == 3
    gap = int(xtts._GAP_S * xtts.XTTS_SAMPLE_RATE)
    assert audio["n_samples"] == n * 2400 + (n - 1) * gap


def test_english_text_is_not_nepali_normalized(xtts, tmp_path):
    model = _FakeXtts(max_tokens=400)
    _synth(xtts, model, {"op": "synthesize", "text": "Call 911 now.", "language": "English",
                         "ref_audio": _ref_wav(tmp_path)})
    assert model.inference_calls[0] == {**model.inference_calls[0], "text": "Call 911 now.", "language": "en"}


@pytest.mark.parametrize(("speed", "expected"), [(None, 1.0), ("fast", 1.0), (5, 2.0), (0.1, 0.5), (1.25, 1.25)])
def test_speed_is_clamped(xtts, tmp_path, speed, expected):
    model = _FakeXtts(max_tokens=400)
    _synth(xtts, model, {"op": "synthesize", "text": "नमस्ते।", "speed": speed, "ref_audio": _ref_wav(tmp_path)})
    assert model.inference_calls[0]["speed"] == expected


@pytest.mark.parametrize("msg", [{"op": "synthesize"}, {"op": "synthesize", "text": ""}, {"op": "synthesize", "text": 42}])
def test_missing_text_fails_before_loading_the_model(xtts, monkeypatch, msg):
    monkeypatch.setattr(xtts, "_load_model", lambda stdout: pytest.fail("model loaded"))
    with pytest.raises(ValueError, match="text"):
        xtts._handle_synthesize(msg, io.BytesIO())


def test_punctuation_only_text_is_reported_as_nothing_speakable(xtts, tmp_path):
    xtts._MODEL = _FakeXtts()
    with pytest.raises(ValueError, match="nothing speakable"):
        xtts._handle_synthesize({"op": "synthesize", "text": "। । ?", "ref_audio": _ref_wav(tmp_path)}, io.BytesIO())


def test_inference_failure_propagates_and_heartbeat_stops(xtts, tmp_path):
    import threading

    model = _FakeXtts(max_tokens=400)

    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory")

    model.inference = boom
    xtts._MODEL = model
    before = threading.active_count()
    with pytest.raises(RuntimeError, match="out of memory"):
        xtts._handle_synthesize({"op": "synthesize", "text": "नमस्ते।", "ref_audio": _ref_wav(tmp_path)}, io.BytesIO())
    assert threading.active_count() <= before


# ── device selection (CPU / GPU) ────────────────────────────────────────────


def test_device_prefers_cuda_when_available_and_honors_override(xtts, monkeypatch):
    import torch

    monkeypatch.delenv("OMNIVOICE_XTTS_NEPALI_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert xtts._device() == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert xtts._device() == "cpu"
    monkeypatch.setenv("OMNIVOICE_XTTS_NEPALI_DEVICE", "CPU")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert xtts._device() == "cpu"


# ── parent-side adapter ─────────────────────────────────────────────────────


def test_backend_is_unavailable_without_its_venv(tmp_path, monkeypatch):
    from engines.xtts_nepali import XttsNepaliBackend

    monkeypatch.setenv("OMNIVOICE_XTTS_NEPALI_DIR", str(tmp_path))
    ok, reason = XttsNepaliBackend.is_available()
    assert not ok and str(tmp_path) in reason and "coqui-tts" in reason


def test_backend_is_available_with_a_venv_interpreter(tmp_path, monkeypatch):
    from engines.xtts_nepali import XttsNepaliBackend

    monkeypatch.setenv("OMNIVOICE_XTTS_NEPALI_DIR", str(tmp_path))
    py = XttsNepaliBackend.venv_python()
    py.parent.mkdir(parents=True)
    py.touch()
    assert XttsNepaliBackend.is_available()[0]


@pytest.mark.parametrize(("raw", "expected"), [(None, 600.0), ("120", 120.0), ("5", 30.0), ("nan", 600.0), ("abc", 600.0)])
def test_recv_timeout_is_parsed_defensively(monkeypatch, raw, expected):
    from engines.xtts_nepali import XttsNepaliBackend

    if raw is None:
        monkeypatch.delenv("OMNIVOICE_XTTS_NEPALI_RECV_TIMEOUT_S", raising=False)
    else:
        monkeypatch.setenv("OMNIVOICE_XTTS_NEPALI_RECV_TIMEOUT_S", raw)
    backend = XttsNepaliBackend.__new__(XttsNepaliBackend)
    assert backend.recv_timeout_s == expected


def test_legacy_engine_alias_resolves_to_xtts_nepali():
    from services.tts_backend import canonical_engine_id, get_backend_class
    from engines.xtts_nepali import XttsNepaliBackend

    assert canonical_engine_id(" Oshara-XTTS-v2 ") == "xtts-nepali"
    assert get_backend_class("oshara-xtts-v2") is XttsNepaliBackend


@pytest.mark.parametrize("text", [
    "मूल्य रु. पाँच सय मात्र।",
    "डा. राम शर्मा आउनुभयो।",
    "घर नं. १५ मा बस्छु।",
    "वि.सं. २०८१ मा सुरु भयो।",
])
def test_nepali_abbreviations_do_not_split_a_sentence(xtts, text):
    assert xtts._chunks(text, _CharTokenizer(), "hi", 400) == [text]


def test_period_after_a_normal_word_still_ends_a_sentence(xtts):
    chunks = xtts._chunks("उहाँ घरमा हुनुहुन्छ. भोलि आउनुहुन्छ.", _CharTokenizer(), "hi", 400)
    assert chunks == ["उहाँ घरमा हुनुहुन्छ.", "भोलि आउनुहुन्छ."]
