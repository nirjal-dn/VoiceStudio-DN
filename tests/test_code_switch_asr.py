"""Nepali ↔ English code-switch ASR (Whisper large-v3, ``whisper-ne-en``).

Implements the validated notebook recipe (``STT_openai_Whisper``): faster-whisper
with a forced ``language="ne"`` decode and ``task="transcribe"`` — no language
detection.  Forcing ``ne`` keeps Nepali in Devanagari and embedded English in
Latin without the decoder drifting to Hindi, for all three input types (pure
Nepali, pure English, and mixed).

Real acoustic accuracy needs audio + the 3 GB model, so it can't be a unit test
(see the opt-in integration test at the bottom).  These pin the decode contract
with a fake faster-whisper model, so they need neither weights nor a GPU.
"""
from __future__ import annotations

import os
import re

import pytest

from services import asr_backend as ab


# ── Fake faster-whisper model ───────────────────────────────────────────────

class _FakeInfo:
    def __init__(self, language):
        self.language = language
        self.language_probability = 0.9
        self.duration = 1.0


class _FakeSeg:
    def __init__(self, text):
        self.text = text
        self.start = 0.0
        self.end = 1.0
        self.words = []


class _FakeModel:
    """Records transcribe kwargs; echoes canned segment text back.

    ``seg_texts`` may be a single string (one segment) or a list (multiple VAD
    segments) to exercise the whole-recording collapse.
    """

    def __init__(self, seg_texts):
        self._seg_texts = [seg_texts] if isinstance(seg_texts, str) else list(seg_texts)
        self.transcribe_kwargs: dict | None = None

    def transcribe(self, audio, **kw):
        self.transcribe_kwargs = kw
        segs = []
        for i, txt in enumerate(self._seg_texts):
            s = _FakeSeg(txt)
            s.start = float(i)
            s.end = float(i) + 1.0
            segs.append(s)
        return iter(segs), _FakeInfo(kw.get("language"))


def _backend_with(seg_texts):
    pytest.importorskip("faster_whisper")
    bk = ab.CodeSwitchWhisperBackend()
    bk._model = _FakeModel(seg_texts)  # skip real _ensure_model()
    return bk


# ── Decoding configuration (the notebook recipe) ─────────────────────────────

def test_default_language_is_pinned_nepali(monkeypatch):
    monkeypatch.delenv("OMNIVOICE_CS_LANGUAGE", raising=False)
    bk = _backend_with("नमस्ते, मेरो account को balance कति छ?")
    out = bk.transcribe("x.wav")
    kw = bk._model.transcribe_kwargs
    assert kw["language"] == "ne"
    assert out["language"] == "ne"


def test_decoding_uses_notebook_params(monkeypatch):
    monkeypatch.delenv("OMNIVOICE_CS_BEAM_SIZE", raising=False)
    monkeypatch.delenv("OMNIVOICE_CS_TEMPERATURE", raising=False)
    bk = _backend_with("नमस्ते")
    bk.transcribe("x.wav")
    kw = bk._model.transcribe_kwargs
    assert kw["task"] == "transcribe"            # never translate
    # Temperature-fallback ladder — 0.0 first (deterministic for clean audio),
    # higher rungs recover failed Nepali windows.
    assert kw["temperature"] == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    assert kw["temperature"][0] == 0.0
    # Quality gates that actually trigger the fallback.
    assert kw["compression_ratio_threshold"] == 2.4
    assert kw["log_prob_threshold"] == -1.0
    assert kw["no_speech_threshold"] == 0.6
    assert kw["beam_size"] == 5
    assert kw["best_of"] == 5
    assert kw["vad_filter"] is True
    assert kw["vad_parameters"] == {"min_silence_duration_ms": 500}
    assert kw["condition_on_previous_text"] is False


def test_temperature_env_forces_greedy(monkeypatch):
    monkeypatch.setenv("OMNIVOICE_CS_TEMPERATURE", "0.0")
    bk = _backend_with("नमस्ते")
    bk.transcribe("x.wav")
    assert bk._model.transcribe_kwargs["temperature"] == (0.0,)


def test_temperature_env_invalid_falls_back_to_ladder(monkeypatch):
    monkeypatch.setenv("OMNIVOICE_CS_TEMPERATURE", "not-a-number")
    bk = _backend_with("नमस्ते")
    bk.transcribe("x.wav")
    assert bk._model.transcribe_kwargs["temperature"] == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def test_language_env_override_pins_english(monkeypatch):
    monkeypatch.setenv("OMNIVOICE_CS_LANGUAGE", "en")
    bk = _backend_with("hello world")
    bk.transcribe("x.wav")
    assert bk._model.transcribe_kwargs["language"] == "en"


def test_beam_size_env_override(monkeypatch):
    monkeypatch.setenv("OMNIVOICE_CS_BEAM_SIZE", "1")
    bk = _backend_with("नमस्ते")
    bk.transcribe("x.wav")
    assert bk._model.transcribe_kwargs["beam_size"] == 1
    assert bk._model.transcribe_kwargs["best_of"] == 1


def test_initial_prompt_primes_mixed_script_by_default(monkeypatch):
    monkeypatch.delenv("OMNIVOICE_CS_PROMPT", raising=False)
    bk = _backend_with("नमस्ते")
    bk.transcribe("x.wav")
    prompt = bk._model.transcribe_kwargs["initial_prompt"]
    # Nudges English → Latin: the prompt carries Latin words inside Devanagari.
    assert prompt and "email" in prompt
    assert any("ऀ" <= ch <= "ॿ" for ch in prompt)  # also has Devanagari


def test_initial_prompt_can_be_disabled(monkeypatch):
    monkeypatch.setenv("OMNIVOICE_CS_PROMPT", "")
    bk = _backend_with("नमस्ते")
    bk.transcribe("x.wav")
    assert bk._model.transcribe_kwargs["initial_prompt"] is None


def test_beam_size_env_invalid_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("OMNIVOICE_CS_BEAM_SIZE", "not-a-number")
    bk = _backend_with("नमस्ते")
    bk.transcribe("x.wav")
    assert bk._model.transcribe_kwargs["beam_size"] == 5


@pytest.mark.parametrize("text", [
    "नमस्ते",                                             # pure Nepali
    "hello world",                                        # pure English
    "नमस्ते, मेरो account को balance कति छ?",              # mixed Nepali+English
    "मेरो account मा Rs. 5000 छ",                          # price (Latin + Devanagari)
    "आज 2081-05-15 मा meeting छ at 3 PM",                 # date + number + English
    "मैले 25 percent discount पाएँ",                       # number + English word
])
def test_one_complete_transcript_scripts_and_numbers_verbatim(monkeypatch, text):
    # The backend must not transliterate, translate, split, or strip either
    # script / digits — the whole mixed string round-trips as ONE transcript.
    bk = _backend_with(text)
    out = bk.transcribe("x.wav")
    assert out["text"] == text
    assert out["segments"][0]["text"] == text
    assert out["chunks"][0]["text"] == text


def test_whole_recording_collapses_to_one_segment(monkeypatch):
    # Whisper's VAD splits a recording into several utterance segments; the
    # dictation engine must return ONE continuous transcript spanning start→stop.
    bk = _backend_with([
        "नमस्ते, मेरो account को balance कति छ?",   # 0.0–1.0
        "Please transfer Rs. 5000 today.",           # 1.0–2.0
        "धन्यवाद।",                                    # 2.0–3.0
    ])
    out = bk.transcribe("x.wav")
    expected = (
        "नमस्ते, मेरो account को balance कति छ? "
        "Please transfer Rs. 5000 today. धन्यवाद।"
    )
    assert out["text"] == expected            # single continuous transcript
    assert len(out["segments"]) == 1          # not broken into segments
    assert len(out["chunks"]) == 1
    seg = out["segments"][0]
    assert seg["text"] == expected
    assert seg["start"] == 0.0                # start of the recording
    assert seg["end"] == 3.0                  # stop of the recording


def test_single_segment_recording_is_unchanged(monkeypatch):
    bk = _backend_with("नमस्ते")
    out = bk.transcribe("x.wav")
    assert out["text"] == "नमस्ते"
    assert len(out["segments"]) == 1


def test_registered_and_is_faster_whisper_subclass():
    assert ab._REGISTRY["whisper-ne-en"] is ab.CodeSwitchWhisperBackend
    assert issubclass(ab.CodeSwitchWhisperBackend, ab.FasterWhisperBackend)
    assert ab.CodeSwitchWhisperBackend.serves_capture is True


# ── Optional: real-audio integration (opt-in; needs the model + clips) ──────
# Set OMNIVOICE_CS_ASR_AUDIO_DIR to a folder of {ne,en,mixed}.wav to run a real
# large-v3 pass. Skipped by default so CI stays weightless.
@pytest.mark.skipif(
    not os.environ.get("OMNIVOICE_CS_ASR_AUDIO_DIR"),
    reason="set OMNIVOICE_CS_ASR_AUDIO_DIR to run the real-audio code-switch pass",
)
def test_real_audio_code_switch():  # pragma: no cover - manual/eval only
    d = os.environ["OMNIVOICE_CS_ASR_AUDIO_DIR"]
    devanagari = re.compile(r"[ऀ-ॿ]")
    bk = ab.CodeSwitchWhisperBackend()
    for name in ("ne", "mixed"):
        path = os.path.join(d, f"{name}.wav")
        if not os.path.exists(path):
            continue
        out = bk.transcribe(path, word_timestamps=False)
        assert out["language"] == "ne"
        assert devanagari.search(out["text"]), f"{name}.wav produced no Devanagari"
