"""The ``indic-conformer-qwen`` engine: IndicConformer + automatic Qwen norm.

Pipeline contract, pinned with fakes (no weights, no llama.cpp, no GPU):
IndicConformer transcribes → the raw Devanagari goes through the Qwen
normalizer → the normalized text is the final ``text``, the raw is kept in
``raw_text``, and any normalizer miss falls back to the raw transcript.
"""
from __future__ import annotations

import pytest

from services import asr_backend
from services import codeswitch_restore as cs


class _FakeIndic:
    """Stand-in IndicConformerBackend: returns a fixed Devanagari result."""

    model_repo_id = "ai4bharat/indic-conformer-600m-multilingual"

    def __init__(self, text="मेरो अकाउन्ट", available=True):
        self._text = text
        self._available = available
        self.unloaded = False

    @classmethod
    def is_available(cls):
        return (True, "ready (CPU)")

    def transcribe(self, audio_path, *, word_timestamps=True, language=None):
        return {
            "text": self._text,
            "segments": [{"text": self._text, "start": 0.0, "end": 1.0}],
            "chunks": [{"text": self._text, "timestamp": (0.0, 1.0)}],
            "language": "ne",
        }

    def unload(self):
        self.unloaded = True


def _engine(monkeypatch, fake_indic):
    """An IndicConformerQwenBackend wired to ``fake_indic`` as its sub-backend."""
    monkeypatch.setattr(asr_backend, "_indic_conformer", lambda: type(fake_indic))
    eng = asr_backend.IndicConformerQwenBackend()
    eng._indic = fake_indic  # skip the real lazy construction
    return eng


def test_registered_in_registry():
    assert "indic-conformer-qwen" in asr_backend._REGISTRY
    assert asr_backend._REGISTRY["indic-conformer-qwen"] is asr_backend.IndicConformerQwenBackend


def test_normalized_text_becomes_final_raw_kept(monkeypatch):
    fake = _FakeIndic(text="मेरो अकाउन्टको ब्यालेन्स")
    eng = _engine(monkeypatch, fake)
    monkeypatch.setattr(
        cs, "maybe_normalize_transcript",
        lambda text, **kw: "मेरो account को balance।",
    )
    out = eng.transcribe("x.wav")
    assert out["text"] == "मेरो account को balance।"
    assert out["raw_text"] == "मेरो अकाउन्टको ब्यालेन्स"
    # Segments stay raw (IndicConformer timings), untouched by normalization.
    assert out["segments"][0]["text"] == "मेरो अकाउन्टको ब्यालेन्स"


def test_falls_back_to_raw_when_normalizer_returns_none(monkeypatch):
    fake = _FakeIndic(text="मेरो अकाउन्ट")
    eng = _engine(monkeypatch, fake)
    monkeypatch.setattr(cs, "maybe_normalize_transcript", lambda text, **kw: None)
    out = eng.transcribe("x.wav")
    assert out["text"] == "मेरो अकाउन्ट"
    assert out["raw_text"] == "मेरो अकाउन्ट"


def test_empty_transcript_skips_normalizer(monkeypatch):
    fake = _FakeIndic(text="")
    eng = _engine(monkeypatch, fake)

    def _boom(*a, **k):
        raise AssertionError("normalizer must not run on empty transcript")
    monkeypatch.setattr(cs, "maybe_normalize_transcript", _boom)
    out = eng.transcribe("x.wav")
    assert out["text"] == ""
    assert out["raw_text"] == ""


def test_is_available_gates_on_indic(monkeypatch):
    class _UnavailableIndic:
        @staticmethod
        def is_available():
            return (False, "IndicConformer weights missing")
    monkeypatch.setattr(asr_backend, "_indic_conformer", lambda: _UnavailableIndic)
    ok, msg = asr_backend.IndicConformerQwenBackend.is_available()
    assert ok is False
    assert "missing" in msg


def test_is_available_gates_on_llama_cpp(monkeypatch):
    monkeypatch.setattr(asr_backend, "_indic_conformer", lambda: _FakeIndic)
    # Simulate the codeswitch extra not being installed.
    import builtins
    real_import = builtins.__import__

    def _no_llama(name, *a, **k):
        if name == "llama_cpp":
            raise ImportError("no llama_cpp")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", _no_llama)
    ok, msg = asr_backend.IndicConformerQwenBackend.is_available()
    assert ok is False
    assert "codeswitch" in msg


def test_unload_releases_sub_backend(monkeypatch):
    fake = _FakeIndic()
    eng = _engine(monkeypatch, fake)
    eng.unload()
    assert fake.unloaded is True
    assert eng._indic is None
