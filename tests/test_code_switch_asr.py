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


def test_anti_repetition_defaults(monkeypatch):
    # A soft repetition penalty (>1.0) is on by default so the greedy pass does
    # not loop — the loop is what both duplicates the tail and (via its
    # overshooting timestamp) truncates long files. The hard n-gram block stays
    # off so legitimate reduplication is not clipped.
    for e in ("OMNIVOICE_CS_REPETITION_PENALTY", "OMNIVOICE_CS_NO_REPEAT_NGRAM"):
        monkeypatch.delenv(e, raising=False)
    bk = _backend_with("नमस्ते")
    bk.transcribe("x.wav")
    kw = bk._model.transcribe_kwargs
    assert kw["repetition_penalty"] == 1.1
    assert kw["no_repeat_ngram_size"] == 0
    # condition_on_previous_text stays False so a loop can't seed the next window.
    assert kw["condition_on_previous_text"] is False


def test_anti_repetition_env_overrides(monkeypatch):
    monkeypatch.setenv("OMNIVOICE_CS_REPETITION_PENALTY", "1.3")
    monkeypatch.setenv("OMNIVOICE_CS_NO_REPEAT_NGRAM", "3")
    bk = _backend_with("नमस्ते")
    bk.transcribe("x.wav")
    kw = bk._model.transcribe_kwargs
    assert kw["repetition_penalty"] == 1.3
    assert kw["no_repeat_ngram_size"] == 3


def test_repetition_penalty_env_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("OMNIVOICE_CS_REPETITION_PENALTY", "not-a-number")
    bk = _backend_with("नमस्ते")
    bk.transcribe("x.wav")
    assert bk._model.transcribe_kwargs["repetition_penalty"] == 1.1


def test_hallucination_threshold_only_with_word_timestamps(monkeypatch):
    # faster-whisper can only locate silent gaps when it has word timestamps, so
    # the hallucination-skip is passed then and omitted otherwise (passing it
    # without word timestamps is a no-op at best, an error at worst).
    monkeypatch.delenv("OMNIVOICE_CS_HALLUCINATION_SILENCE_S", raising=False)
    bk = _backend_with("नमस्ते")
    bk.transcribe("x.wav", word_timestamps=True)
    assert bk._model.transcribe_kwargs["hallucination_silence_threshold"] == 2.0

    bk2 = _backend_with("नमस्ते")
    bk2.transcribe("x.wav", word_timestamps=False)
    assert "hallucination_silence_threshold" not in bk2._model.transcribe_kwargs


def test_hallucination_threshold_can_be_disabled(monkeypatch):
    monkeypatch.setenv("OMNIVOICE_CS_HALLUCINATION_SILENCE_S", "0")
    bk = _backend_with("नमस्ते")
    bk.transcribe("x.wav", word_timestamps=True)
    assert "hallucination_silence_threshold" not in bk._model.transcribe_kwargs


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


# ── Bare digits are PRESERVED — never guessed from surrounding script ─────────
# The spoken language of a number Whisper already wrote as digits is not
# recoverable from the transcript, so the digits are left exactly as-is. Only
# number *words* are converted (see the section below), each in its own script.

@pytest.mark.parametrize("text", [
    "नमस्ते",                                          # pure Nepali, no numbers
    "hello world",                                     # pure English, no numbers
    "नमस्ते, मेरो account को balance कति छ?",            # mixed, no digits
    "मलाई 100 रुपैयाँ चाहियो",                          # bare digits in a Nepali sentence
    "मेरो balance 9845 छ",                             # English amount in a Nepali sentence
    "आज 2081-05-15 मा meeting छ",                       # date
    "at 3 PM",
    "version 2.0 release भयो",
])
def test_bare_digits_are_preserved(text):
    # No number words → nothing changes. Digits are NOT flipped to Devanagari
    # just because the sentence is Nepali — that would corrupt an English amount
    # ("9845" must never become "९८४५"). The transcript round-trips verbatim.
    out = _backend_with(text).transcribe("x.wav")
    assert out["text"] == text
    assert out["segments"][0]["text"] == text
    assert out["chunks"][0]["text"] == text


def test_whole_recording_collapses_to_one_segment():
    # Whisper's VAD splits a recording into several utterance segments; the
    # dictation engine must return ONE continuous transcript spanning start→stop.
    bk = _backend_with([
        "नमस्ते, मेरो account को balance कति छ?",   # 0.0–1.0
        "Please transfer Rs. 5000 today.",           # 1.0–2.0 (bare digits kept)
        "मैले ५ मा 10 पैसा दिएँ।",                     # 2.0–3.0 (bare digits kept)
    ])
    out = bk.transcribe("x.wav")
    expected = (
        "नमस्ते, मेरो account को balance कति छ? "
        "Please transfer Rs. 5000 today. मैले ५ मा 10 पैसा दिएँ।"
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


# ── Spoken number WORDS → digits, in the spoken language's script ────────────
# Whisper emits numbers as words as often as digits; the engine converts them to
# digits, Devanagari for Nepali-spoken and Western for English-spoken (the digit
# glyphs then follow context exactly as the digit-script tests above check).

@pytest.mark.parametrize("text,expected", [
    # Nepali number words → Devanagari digits (the spec example; the comma keeps
    # the two numbers apart).
    ("सन्तानब्बे, एकचालिस", "९७, ४१"),
    ("मलाई एक सय रुपैयाँ चाहियो", "मलाई १०० रुपैयाँ चाहियो"),
    ("दुई हजार एकासी साल", "२०८१ साल"),
    # English number words → Western digits (the spec example).
    ("nine thousand eight hundred forty-five dollars", "9845 dollars"),
    ("twenty five percent discount", "25 percent discount"),
    ("एक सय पच्चीस", "१२५"),
    # Code-switched: each number renders in the script of the language it was
    # spoken in, within one sentence.
    ("मलाई एक सय रुपैयाँ र five dollars चाहियो",
     "मलाई १०० रुपैयाँ र 5 dollars चाहियो"),
    # The exact spec sentences: same surrounding Nepali, different spoken number
    # language → English words stay ASCII, Nepali words become Devanagari.
    ("मेरो account को balance nine thousand eight hundred forty five छ।",
     "मेरो account को balance 9845 छ।"),
    ("मेरो account को balance सन्तानब्बे छ।",
     "मेरो account को balance ९७ छ।"),
])
def test_spoken_number_words_become_digits(monkeypatch, text, expected):
    monkeypatch.delenv("OMNIVOICE_CS_SPOKEN_NUMBERS", raising=False)
    out = _backend_with(text).transcribe("x.wav")
    assert out["text"] == expected
    assert out["segments"][0]["text"] == expected


def test_copula_chha_is_not_turned_into_a_number(monkeypatch):
    # "छ" is both "six" and the copula "is"; a standalone/trailing छ must survive
    # so ordinary Nepali is not corrupted ("उमेर पच्चीस छ" = "age is twenty-five").
    monkeypatch.delenv("OMNIVOICE_CS_SPOKEN_NUMBERS", raising=False)
    out = _backend_with("मेरो उमेर पच्चीस छ").transcribe("x.wav")
    assert out["text"] == "मेरो उमेर २५ छ"


def test_spoken_numbers_can_be_disabled(monkeypatch):
    monkeypatch.setenv("OMNIVOICE_CS_SPOKEN_NUMBERS", "0")
    text = "सन्तानब्बे, एकचालिस"
    out = _backend_with(text).transcribe("x.wav")
    assert out["text"] == text  # number words kept verbatim


def test_default_prompt_protects_common_english_words():
    # English words must stay Latin, not be transliterated to Devanagari
    # ("office" → "अफिस"). The mixed-script priming prompt is the model-side
    # nudge; assert the common code-switch English it names is present so the
    # protection isn't silently dropped by a prompt edit.
    prompt = ab.CodeSwitchWhisperBackend._DEFAULT_PROMPT
    for word in ("office", "email", "meeting"):
        assert word in prompt
    # And it genuinely mixes scripts (Latin words inside Devanagari).
    assert any("ऀ" <= ch <= "ॿ" for ch in prompt)


# ── Turbo variant ────────────────────────────────────────────────────────────


def test_turbo_registered_and_reuses_code_switch_recipe():
    assert ab._REGISTRY["whisper-ne-en-turbo"] is ab.CodeSwitchWhisperTurboBackend
    assert issubclass(ab.CodeSwitchWhisperTurboBackend, ab.CodeSwitchWhisperBackend)
    assert ab.CodeSwitchWhisperTurboBackend.serves_capture is True


def test_turbo_pins_the_turbo_model(monkeypatch):
    pytest.importorskip("faster_whisper")
    monkeypatch.delenv("OMNIVOICE_CS_ASR_MODEL", raising=False)
    bk = ab.CodeSwitchWhisperTurboBackend()
    assert bk._model_name == "deepdml/faster-whisper-large-v3-turbo-ct2"
    # Base variant is unaffected — still large-v3.
    assert ab.CodeSwitchWhisperBackend()._model_name != bk._model_name


def test_turbo_model_env_override_wins(monkeypatch):
    pytest.importorskip("faster_whisper")
    monkeypatch.setenv("OMNIVOICE_CS_ASR_MODEL", "custom/ct2-turbo")
    assert ab.CodeSwitchWhisperTurboBackend()._model_name == "custom/ct2-turbo"


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
