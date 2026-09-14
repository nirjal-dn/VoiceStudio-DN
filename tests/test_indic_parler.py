"""Indic Parler-TTS engine: registry, availability, sentence splitting (no model)."""
from __future__ import annotations

from engines import indic_parler
from engines.indic_parler import main as sidecar


def test_registered_and_unavailable_without_its_venv(tmp_path, monkeypatch):
    from services.tts_backend import _REGISTRY

    monkeypatch.setenv("OMNIVOICE_INDIC_PARLER_DIR", str(tmp_path))
    assert _REGISTRY["indic-parler-tts"] is indic_parler.IndicParlerBackend
    ok, reason = indic_parler.IndicParlerBackend.is_available()
    assert not ok and str(tmp_path) in reason


def test_text_is_split_into_speakable_sentences():
    text = "नमस्ते, म अमृता हुँ। तपाईंलाई कस्तो छ?  ।  Thanks. Bye!"
    assert sidecar.sentences(text) == ["नमस्ते, म अमृता हुँ।", "तपाईंलाई कस्तो छ?", "Thanks.", "Bye!"]
