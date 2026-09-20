"""CodeSwitchTTSService routing, with both child engines mocked.

What these pin down is everything a real render would only reveal after two
multi-GB model loads: which engine gets which span, that both get the SAME
reference voice, that the cached engine-instance API is what resolves them,
that the machine's active backend is never read or written, and that a missing
English engine fails with something a user can act on.
"""
from __future__ import annotations

import math

import pytest
import torch

from services import code_switch_tts as cs
from services.code_switch_tts import (
    CodeSwitchConfig, CodeSwitchError, CodeSwitchTTSService,
)
from services.language_segmenter import ENGLISH, NEPALI

NE_TEXT = "म आज कार्यालय जान्छु।"
MIXED = "म आज office गएर meeting गर्छु।"


class FakeEngine:
    """A child backend with the surface the service actually touches."""

    supports_cloning = True
    applies_own_mastering = False
    supported_languages = ["multi"]
    available = (True, "ready")

    def __init__(self, engine_id: str, sample_rate: int = 24_000):
        self.id = engine_id
        self.sample_rate = sample_rate
        self.calls: list[dict] = []
        self.loads = 0

    def generate(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        n = max(1, len(text) * 100)
        t = torch.arange(n, dtype=torch.float32) / self.sample_rate
        return (0.4 * torch.sin(2 * math.pi * 200 * t)).unsqueeze(0)


@pytest.fixture
def engines(monkeypatch):
    """Wire the fake engines into the cached engine-instance API only.

    Nothing here stubs ``get_active_tts_backend`` / ``active_backend_id`` — a
    service that reached for them would raise, which is the point.
    """
    made = {
        "xtts-nepali": FakeEngine("xtts-nepali"),
        "xtts-en": FakeEngine("xtts-en"),
    }
    resolved: list[str] = []

    class FakeClass:
        def __init__(self, engine_id):
            self.engine_id = engine_id

        supports_cloning = True

        def is_available(self):
            return made[self.engine_id].available

    def fake_get_backend_class(engine_id):
        if engine_id not in made:
            raise ValueError(f"Unknown TTS backend: {engine_id!r}")
        return FakeClass(engine_id)

    def fake_instance_for(engine_id, **_kw):
        resolved.append(engine_id)
        return made[engine_id]

    monkeypatch.setattr("services.tts_backend.get_backend_class", fake_get_backend_class)
    monkeypatch.setattr("services.tts_backend.get_engine_instance_for", fake_instance_for)

    def _boom(*_a, **_kw):  # pragma: no cover - only runs if the service regresses
        raise AssertionError("the code-switch service must not touch the active backend")

    monkeypatch.setattr("services.tts_backend.get_active_tts_backend", _boom)
    monkeypatch.setattr("services.tts_backend.active_backend_id", _boom)
    monkeypatch.setattr("services.tts_backend.reset_active_backend", _boom)

    made["resolved"] = resolved
    return made


def _service(**overrides):
    config = CodeSwitchConfig(
        nepali_engine="xtts-nepali", english_engine="xtts-en", **overrides,
    )
    return CodeSwitchTTSService(config)


# ── routing ───────────────────────────────────────────────────────────────

def test_each_span_goes_to_its_own_engine_in_textual_order(engines):
    service = _service()
    service.generate(MIXED, ref_audio="/tmp/ref.wav")

    ne_texts = [c["text"] for c in engines["xtts-nepali"].calls]
    en_texts = [c["text"] for c in engines["xtts-en"].calls]
    assert len(ne_texts) == 3 and len(en_texts) == 2
    assert "office" in en_texts[0] and "meeting" in en_texts[1]
    assert all("ऀ" <= ch <= "ॿ" for text in ne_texts for ch in text.strip()[:1])
    assert [c["language"] for c in engines["xtts-nepali"].calls] == [NEPALI] * 3
    assert [c["language"] for c in engines["xtts-en"].calls] == [ENGLISH] * 2


def test_pure_nepali_never_reaches_the_english_engine(engines):
    _service().generate(NE_TEXT, ref_audio="/tmp/ref.wav")
    assert engines["xtts-en"].calls == []
    assert len(engines["xtts-nepali"].calls) == 1


def test_an_explicit_language_mode_bypasses_segmentation(engines):
    _service().generate(MIXED, ref_audio="/tmp/ref.wav", language_mode="ne")
    assert len(engines["xtts-nepali"].calls) == 1
    assert engines["xtts-en"].calls == []


# ── the voice ─────────────────────────────────────────────────────────────

def test_both_engines_get_the_same_reference_audio_text_and_speed(engines):
    _service().generate(MIXED, ref_audio="/tmp/ref.wav", ref_text="hello there",
                        speed=1.25)
    every_call = engines["xtts-nepali"].calls + engines["xtts-en"].calls
    assert every_call
    assert {c["ref_audio"] for c in every_call} == {"/tmp/ref.wav"}
    assert {c["ref_text"] for c in every_call} == {"hello there"}
    assert {c["speed"] for c in every_call} == {1.25}


# ── engine resolution ─────────────────────────────────────────────────────

def test_children_come_from_the_cached_instance_api_and_are_resolved_once(engines):
    service = _service()
    service.generate(MIXED, ref_audio="/tmp/ref.wav")
    # Five spans, two engines: the instance cache is consulted per engine, not
    # per span, and the active-backend helpers are never called (they raise).
    assert sorted(set(engines["resolved"])) == ["xtts-en", "xtts-nepali"]
    assert engines["resolved"].count("xtts-nepali") == 1
    assert engines["resolved"].count("xtts-en") == 1


def test_engines_are_held_against_the_idle_sweep_while_they_render(engines, monkeypatch):
    from services import tts_backend

    held: list = []
    real = tts_backend.engine_in_use

    def spy(instance, **kw):
        held.append(getattr(instance, "id", instance))
        return real(instance, **kw)

    monkeypatch.setattr("services.tts_backend.engine_in_use", spy)
    _service().generate(MIXED, ref_audio="/tmp/ref.wav")
    assert "xtts-nepali" in held and "xtts-en" in held


# ── validation ────────────────────────────────────────────────────────────

def test_a_missing_english_engine_is_an_actionable_error_not_a_nepali_render(engines):
    engines["xtts-en"].available = (False, "XTTS venv not found at /x/.venv")
    service = _service()
    with pytest.raises(CodeSwitchError) as excinfo:
        service.generate(MIXED, ref_audio="/tmp/ref.wav")
    message = str(excinfo.value)
    assert "English" in message and "xtts-en" in message
    assert "XTTS venv not found" in message
    # The English words were NOT quietly handed to the Nepali model.
    assert engines["xtts-nepali"].calls == []


def test_an_unknown_engine_id_names_itself(engines):
    service = CodeSwitchTTSService(CodeSwitchConfig(
        nepali_engine="xtts-nepali", english_engine="not-an-engine"))
    with pytest.raises(CodeSwitchError) as excinfo:
        service.validate()
    assert "not-an-engine" in str(excinfo.value)


def test_a_non_cloning_engine_is_refused_because_the_speaker_would_change(engines, monkeypatch):
    from services import tts_backend

    original = tts_backend.get_backend_class

    def patched(engine_id):
        cls = original(engine_id)
        if engine_id == "xtts-en":
            cls.supports_cloning = False
        return cls

    monkeypatch.setattr("services.tts_backend.get_backend_class", patched)
    with pytest.raises(CodeSwitchError, match="clone"):
        _service().validate()


def test_an_engine_that_cannot_speak_the_language_is_refused(engines):
    engines["xtts-en"].supported_languages = ["zh", "ja"]
    with pytest.raises(CodeSwitchError, match="does not list"):
        _service().validate()


def test_an_unusable_sample_rate_is_refused(engines):
    engines["xtts-en"].sample_rate = 0
    with pytest.raises(CodeSwitchError, match="sample rate"):
        _service().validate()


def test_text_with_nothing_speakable_is_a_typed_input_error(engines):
    from services.tts_backend import TTSInputError

    with pytest.raises(TTSInputError):
        _service().generate("   ")


# ── output ────────────────────────────────────────────────────────────────

def test_the_take_comes_back_at_one_rate_and_one_channel_shape(engines):
    audio, sample_rate = _service().generate(MIXED, ref_audio="/tmp/ref.wav")
    assert sample_rate == 24_000
    assert audio.ndim == 2 and audio.shape[0] == 1
    assert audio.dtype == torch.float32


def test_a_slower_child_engine_is_resampled_up_not_the_other_way_round(engines):
    engines["xtts-en"].sample_rate = 22_050
    audio, sample_rate = _service().generate(MIXED, ref_audio="/tmp/ref.wav")
    assert sample_rate == 24_000
    assert audio.shape[-1] > 0


def test_mastering_is_skipped_only_when_every_child_masters_its_own_output(engines):
    service = _service()
    service.generate(MIXED, ref_audio="/tmp/ref.wav")
    assert service.applies_own_mastering is False
    for engine in ("xtts-nepali", "xtts-en"):
        engines[engine].applies_own_mastering = True
    assert service.applies_own_mastering is True


def test_parallel_rendering_preserves_span_order(engines):
    audio_seq, _ = _service().generate(MIXED, ref_audio="/tmp/ref.wav")
    engines["xtts-nepali"].calls.clear()
    engines["xtts-en"].calls.clear()
    audio_par, _ = _service(allow_parallel=True).generate(MIXED, ref_audio="/tmp/ref.wav")
    assert [c["text"] for c in engines["xtts-en"].calls] == ["office", "meeting"]
    assert audio_par.shape == audio_seq.shape


# ── configuration ─────────────────────────────────────────────────────────

def test_config_reads_env_first_then_prefs_then_defaults(monkeypatch):
    monkeypatch.delenv("OMNIVOICE_CODE_SWITCH_EN_BACKEND", raising=False)
    assert CodeSwitchConfig.from_prefs().english_engine == cs.DEFAULT_ENGLISH_ENGINE
    monkeypatch.setenv("OMNIVOICE_CODE_SWITCH_EN_BACKEND", "omnivoice")
    monkeypatch.setenv("OMNIVOICE_CODE_SWITCH_CROSSFADE_MS", "12")
    monkeypatch.setenv("OMNIVOICE_CODE_SWITCH_PARALLEL", "1")
    monkeypatch.setenv("OMNIVOICE_CODE_SWITCH_TRIM_SILENCE", "0")
    config = CodeSwitchConfig.from_prefs()
    assert config.english_engine == "omnivoice"
    assert config.crossfade_ms == 12
    assert config.allow_parallel is True
    assert config.trim_boundary_silence is False


def test_a_junk_crossfade_value_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("OMNIVOICE_CODE_SWITCH_CROSSFADE_MS", "not-a-number")
    assert CodeSwitchConfig.from_prefs().crossfade_ms == 30


def test_auto_mode_only_engages_on_genuinely_mixed_text(monkeypatch):
    monkeypatch.delenv("OMNIVOICE_CODE_SWITCH_ENABLED", raising=False)
    assert cs.should_code_switch(MIXED) is True
    assert cs.should_code_switch(NE_TEXT) is False
    assert cs.should_code_switch("I am going to the office today.") is False
    monkeypatch.setenv("OMNIVOICE_CODE_SWITCH_ENABLED", "0")
    assert cs.should_code_switch(MIXED) is False
