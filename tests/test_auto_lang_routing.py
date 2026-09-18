"""Automatic language-based model routing (auto-lang engines).

Two delegating backends let a user dictate/synthesize Nepali and English in
one conversation without switching engines by hand:

* ASR ``auto-lang`` — Whisper transcribes once (detecting the language for
  free); a Nepali detection re-runs the audio through IndicConformer, every
  other language keeps Whisper's single-pass result.
* TTS ``auto-lang`` — Devanagari text routes to the Nepali engine
  (Oshara/XTTS), everything else to the default (OmniVoice).

These pin the routing contract with fakes, so they need neither model weights
nor a GPU.
"""
from __future__ import annotations

import pytest


# ── ASR: Whisper detects → route Nepali to IndicConformer ───────────────────

class _FakeASR:
    def __init__(self, text, language):
        self._text, self._lang = text, language
        self.unloaded = False

    def transcribe(self, audio_path, *, word_timestamps=True):
        return {"text": self._text, "segments": [], "language": self._lang}

    def unload(self):
        self.unloaded = True


def _auto_asr(monkeypatch, whisper, indic, *, indic_available=True):
    from services import asr_backend

    class _IndicCls:
        @staticmethod
        def is_available():
            return (indic_available, "ready" if indic_available else "no")

    monkeypatch.setattr(asr_backend, "_indic_conformer", lambda: _IndicCls)
    bk = asr_backend.AutoLangASRBackend()
    bk._whisper = whisper
    bk._indic = indic
    return bk


def test_asr_english_stays_on_whisper_single_pass(monkeypatch):
    whisper = _FakeASR("hello world", "en")
    indic = _FakeASR("should not run", "ne")
    bk = _auto_asr(monkeypatch, whisper, indic)
    assert bk.transcribe("x.wav")["text"] == "hello world"


def test_asr_nepali_reroutes_to_indic_conformer(monkeypatch):
    whisper = _FakeASR("whisper nepali guess", "ne")
    indic = _FakeASR("नेपाली पाठ", "ne")
    bk = _auto_asr(monkeypatch, whisper, indic)
    assert bk.transcribe("x.wav")["text"] == "नेपाली पाठ"


def test_asr_indic_failure_falls_back_to_whisper(monkeypatch):
    whisper = _FakeASR("whisper fallback", "ne")

    class _BoomIndic(_FakeASR):
        def transcribe(self, *a, **k):
            raise RuntimeError("indic boom")

    bk = _auto_asr(monkeypatch, whisper, _BoomIndic("", "ne"))
    # A crashing IndicConformer must never lose the recording.
    assert bk.transcribe("x.wav")["text"] == "whisper fallback"


def test_asr_indic_unavailable_keeps_whisper(monkeypatch):
    whisper = _FakeASR("whisper only", "ne")
    bk = _auto_asr(monkeypatch, whisper, _FakeASR("unused", "ne"),
                   indic_available=False)
    assert bk.transcribe("x.wav")["text"] == "whisper only"


def test_asr_indic_lang_set_is_configurable(monkeypatch):
    monkeypatch.setenv("OMNIVOICE_AUTO_ASR_INDIC_LANGS", "hi,mr")
    # Re-evaluate the class-level frozenset the env feeds.
    from services import asr_backend
    import importlib
    importlib.reload(asr_backend)
    assert "hi" in asr_backend.AutoLangASRBackend._INDIC_LANGS
    assert "ne" not in asr_backend.AutoLangASRBackend._INDIC_LANGS
    importlib.reload(asr_backend)  # restore default for later tests


# ── TTS: Devanagari → Nepali engine, else default ───────────────────────────
# tts_backend imports torch at module load; skip cleanly where it's absent.
torch = pytest.importorskip("torch")


def test_is_devanagari():
    from services.tts_backend import _is_devanagari
    assert _is_devanagari("नमस्ते")
    assert _is_devanagari("hello नमस्ते")   # mixed → Nepali engine (handles it)
    assert not _is_devanagari("hello world")
    assert not _is_devanagari("")


class _FakeTTS:
    def __init__(self, sr):
        self._sr = sr
        self.calls = []

    @property
    def sample_rate(self):
        return self._sr

    def generate(self, text, **kw):
        self.calls.append(text)
        return text

    def unload(self):
        pass


def test_tts_routes_by_script(monkeypatch):
    from services import tts_backend

    nep, eng = _FakeTTS(24000), _FakeTTS(24000)
    bk = tts_backend.AutoLangTTSBackend()
    bk._subs = {bk._NEPALI_ID: nep, bk._DEFAULT_ID: eng}

    bk.generate("नमस्ते")
    bk.generate("good morning")
    assert nep.calls == ["नमस्ते"]
    assert eng.calls == ["good morning"]
    # sample_rate reflects the last delegate.
    assert bk.sample_rate == 24000
