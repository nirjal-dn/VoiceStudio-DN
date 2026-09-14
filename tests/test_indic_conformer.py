"""IndicConformer ASR engine: adapter contract, with the model runtime stubbed."""
from __future__ import annotations

import numpy as np
import soundfile as sf

from engines import indic_conformer as ic


class _StubRuntime:
    def __init__(self, words):
        self._words = words

    def words(self, audio, lang):
        return list(self._words)


def _wav(tmp_path, seconds=5.0):
    path = tmp_path / "clip.wav"
    sf.write(path, np.zeros(int(16000 * seconds), dtype=np.float32), 16000)
    return str(path)


def test_transcribe_returns_whisper_shape_in_native_script(tmp_path, monkeypatch):
    monkeypatch.delenv("OMNIVOICE_INDIC_CONFORMER_LANG", raising=False)
    stub = _StubRuntime([
        ("मेरो", 0.0, 0.3, 0.9),
        ("अकाउन्टमा", 0.3, 1.2, 0.8),
        ("अपडेट", 1.2, 1.8, 0.9),
        ("भएको", 1.8, 2.2, 0.9),
        ("छैन", 3.5, 3.9, 0.9),  # after a 1.3 s pause: new segment
    ])
    monkeypatch.setattr(ic, "_load", lambda: stub)

    result = ic.IndicConformerBackend().transcribe(_wav(tmp_path))

    assert result["language"] == "ne"
    assert [s["text"] for s in result["segments"]] == ["मेरो अकाउन्टमा अपडेट भएको", "छैन"]
    assert result["text"] == "मेरो अकाउन्टमा अपडेट भएको छैन"
    words = result["segments"][0]["words"]
    assert [w["word"].strip() for w in words] == ["मेरो", "अकाउन्टमा", "अपडेट", "भएको"]
    assert words[1]["start"] == 0.3 and words[1]["end"] == 1.2
    assert result["chunks"][1] == {"text": "छैन", "timestamp": (3.5, 3.9)}


def test_word_timestamps_can_be_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(ic, "_load", lambda: _StubRuntime([("पोलिसी", 0.0, 0.5, 0.9)]))
    result = ic.IndicConformerBackend().transcribe(_wav(tmp_path), word_timestamps=False)
    assert result["text"] == "पोलिसी"
    assert result["segments"][0]["words"] == []


def test_unsupported_language_is_reported_unavailable(monkeypatch):
    monkeypatch.setenv("OMNIVOICE_INDIC_CONFORMER_LANG", "fr")
    ok, reason = ic.IndicConformerBackend.is_available()
    assert not ok and "OMNIVOICE_INDIC_CONFORMER_LANG" in reason


def test_whole_recording_is_transcribed_in_one_pass(tmp_path, monkeypatch):
    calls = []

    class _Recorder(_StubRuntime):
        def words(self, audio, lang):
            calls.append(len(audio))
            return []

    monkeypatch.setattr(ic, "_load", lambda: _Recorder([]))
    ic.IndicConformerBackend().transcribe(_wav(tmp_path, seconds=70.0))
    assert calls == [16000 * 70]


def test_nepali_output_is_strictly_devanagari():
    assert ic.clean_word("<unk>नमस्ते", "ne") == "नमस्ते"
    assert ic.clean_word("छ|", "ne") == "छ।"
    assert ic.clean_word("hello안녕سلامमेरो", "ne") == "मेरो"
    assert ic.clean_word("<unk>سلام", "ur") == "سلام"  # other scripts keep their own


def test_capture_uses_pinned_engine_that_serves_capture(monkeypatch):
    from core import prefs as _prefs
    from services import asr_backend

    monkeypatch.delenv("OMNIVOICE_ASR_BACKEND", raising=False)
    monkeypatch.delenv("OMNIVOICE_SHERPA_ASR_MODEL", raising=False)
    # A sherpa dictation model is configured too: the pinned engine must still win.
    store = {
        "asr_backend": "indic-conformer",
        "dictation.enabled": True,
        "dictation.model_id": "sherpa-whisper-tiny",
    }
    monkeypatch.setattr(_prefs, "get", lambda k, d=None: store.get(k, d))
    monkeypatch.setattr(asr_backend, "_capture_backend", None)
    monkeypatch.setattr(asr_backend, "_capture_backend_key", None)
    monkeypatch.setattr(
        ic.IndicConformerBackend, "is_available", classmethod(lambda cls: (True, "ready"))
    )

    backend = asr_backend.get_capture_asr_backend()
    assert isinstance(backend, ic.IndicConformerBackend)
    assert asr_backend.get_capture_asr_backend() is backend  # kept warm
    assert asr_backend.pinned_whole_recording_engine() == "indic-conformer"


def test_registered_as_lazy_asr_backend():
    from services import asr_backend

    assert asr_backend._REGISTRY["indic-conformer"] is ic.IndicConformerBackend
    assert "indic-conformer" in asr_backend._INSTALL_HINTS
