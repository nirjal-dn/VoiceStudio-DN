"""The request's language reaches speech-recognition engines that take one
(IndicConformer) through every entry point; auto-detecting engines are called
exactly as before."""
import sys

import numpy as np
import pytest
import soundfile as sf

import services.asr_backend as ab
from engines import indic_conformer as ic

pytestmark = pytest.mark.usefixtures("asr_model_installed")


class _LanguageAware(ab.ASRBackend):
    id = "fake-language-aware"
    display_name = "Fake language-aware ASR"
    accepts_language = True

    def __init__(self):
        self.languages = []

    @classmethod
    def is_available(cls):
        return True, "ready"

    def transcribe(self, audio_path, *, word_timestamps=True, language=None):
        self.languages.append(language)
        return {"text": "नमस्ते साथी", "segments": [], "language": "ne"}


class _AutoDetect(ab.ASRBackend):
    id = "fake-auto-detect"
    display_name = "Fake auto-detect ASR"

    @classmethod
    def is_available(cls):
        return True, "ready"

    def transcribe(self, audio_path, *, word_timestamps=True):  # no language kwarg
        return {"text": "hello", "segments": [], "language": "en"}


def _wav(tmp_path, name="clip.wav", seconds=1.0, freq=220.0):
    path = tmp_path / name
    t = np.arange(int(16000 * seconds)) / 16000
    sf.write(path, (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32), 16000)
    return str(path)


# ── helper ──────────────────────────────────────────────────────────────────


def test_language_is_passed_only_to_engines_that_accept_it(tmp_path):
    aware = _LanguageAware()
    ab.transcribe_in_language(aware, _wav(tmp_path), "Nepali", word_timestamps=False)
    assert aware.languages == ["Nepali"]
    # An auto-detecting engine's signature has no language: must not TypeError.
    assert ab.transcribe_in_language(_AutoDetect(), _wav(tmp_path), "Nepali")["text"] == "hello"


def test_no_language_leaves_the_engine_default(tmp_path):
    aware = _LanguageAware()
    ab.transcribe_in_language(aware, _wav(tmp_path), None)
    assert aware.languages == [None]


# ── IndicConformer picks the request's language ─────────────────────────────


@pytest.mark.parametrize(("requested", "used"), [
    ("Nepali", "ne"), ("ne-NP", "ne"), ("npi", "ne"), ("hi", "hi"), ("mr", "mr"),
    (None, "ne"), ("Auto", "ne"),
])
def test_indic_conformer_uses_the_requested_language(tmp_path, monkeypatch, requested, used):
    monkeypatch.delenv("OMNIVOICE_INDIC_CONFORMER_LANG", raising=False)
    seen = []

    class _Runtime:
        def words(self, audio, lang):
            seen.append(lang)
            return [("नमस्ते", 0.0, 0.4, 0.9)]

    monkeypatch.setattr(ic, "_load", lambda: _Runtime())
    result = ic.IndicConformerBackend().transcribe(_wav(tmp_path), language=requested)
    assert seen == [used] and result["language"] == used


@pytest.mark.parametrize("requested", ["English", "en", "fr-FR"])
def test_indic_conformer_refuses_languages_it_cannot_transcribe(tmp_path, monkeypatch, requested):
    monkeypatch.setattr(ic, "_load", lambda: pytest.fail("model loaded for an unsupported language"))
    with pytest.raises(ValueError, match="does not transcribe"):
        ic.IndicConformerBackend().transcribe(_wav(tmp_path), language=requested)


def test_indic_conformer_declares_language_support():
    assert ic.IndicConformerBackend.accepts_language is True
    assert ab.ASRBackend.accepts_language is False


# ── entry points ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("form", "expected"), [({"language": "Nepali"}, ["Nepali"]), ({}, [None])])
def test_transcribe_route_forwards_the_language_form_field(tmp_path, monkeypatch, form, expected):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routers import capture

    module = sys.modules["services.asr_backend"]
    backend = _LanguageAware()
    monkeypatch.setattr(module, "asr_model_missing_error", lambda **_kw: None)
    monkeypatch.setattr(module, "get_capture_asr_backend", lambda *a, **k: backend)
    app = FastAPI()
    app.include_router(capture.router)
    with open(_wav(tmp_path), "rb") as fh:
        res = TestClient(app).post("/transcribe", data=form, files={"audio": ("clip.wav", fh, "audio/wav")})
    assert res.status_code == 200, res.text
    assert backend.languages == expected
    assert res.json()["text"] == "नमस्ते साथी।"


def test_reference_transcript_is_requested_and_cached_per_language(tmp_path, monkeypatch):
    module = sys.modules["services.asr_backend"]
    backend = _LanguageAware()
    monkeypatch.setattr(module, "load_active_asr_backend", lambda *a, **k: backend)
    monkeypatch.setattr(module, "_ref_transcript_cache", type(module._ref_transcript_cache)())
    ref = _wav(tmp_path, "ref.wav", freq=333.0)

    assert module.transcribe_reference(ref, "Nepali") == "नमस्ते साथी"
    assert module.transcribe_reference(ref, "ne") == "नमस्ते साथी"  # same language: cached
    module.transcribe_reference(ref, "Hindi")                       # other language: new call
    assert backend.languages == ["Nepali", "Hindi"]


class _RefusesLanguage(_LanguageAware):
    def transcribe(self, audio_path, *, word_timestamps=True, language=None):
        # The class the routes catch: a full-suite run can re-import
        # services.asr_backend, leaving this file's `ab` alias stale.
        error = sys.modules["services.asr_backend"].ASRLanguageNotSupportedError
        raise error(f"cannot transcribe {language!r}")


def test_transcribe_route_answers_400_for_an_unsupported_language(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routers import capture

    module = sys.modules["services.asr_backend"]
    monkeypatch.setattr(module, "asr_model_missing_error", lambda **_kw: None)
    monkeypatch.setattr(module, "get_capture_asr_backend", lambda *a, **k: _RefusesLanguage())
    app = FastAPI()
    app.include_router(capture.router)
    with open(_wav(tmp_path), "rb") as fh:
        res = TestClient(app).post(
            "/transcribe", data={"language": "English"}, files={"audio": ("clip.wav", fh, "audio/wav")},
        )
    assert res.status_code == 400
    assert "cannot transcribe" in res.json()["detail"]


def test_openai_transcriptions_answer_400_for_an_unsupported_language(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routers import openai_compat

    module = sys.modules["services.asr_backend"]
    monkeypatch.setattr(module, "asr_model_missing_error", lambda **_kw: None)
    monkeypatch.setattr(module, "load_active_asr_backend", lambda *a, **k: _RefusesLanguage())
    app = FastAPI()
    app.include_router(openai_compat.router)
    with open(_wav(tmp_path), "rb") as fh:
        res = TestClient(app).post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1", "language": "en"},
            files={"file": ("clip.wav", fh, "audio/wav")},
        )
    assert res.status_code == 400, res.text


def test_indic_conformer_raises_the_typed_error():
    with pytest.raises(ab.ASRLanguageNotSupportedError):
        ic._language("English")
