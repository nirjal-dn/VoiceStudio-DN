"""Per-segment Nepali↔English router (``ne-en-router``).

Pins the routing contract with fakes — no model weights, no GPU, no ffmpeg:

* hysteresis turns raw per-segment predictions into a stable language sequence;
* language ID is restricted to {ne, en} (other languages are dropped);
* a code-switched recording routes ne spans → IndicConformer, en spans →
  Whisper, and merges the pieces back in chronological order.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("soundfile")  # _indic_span writes a temp wav slice

from services import asr_backend  # noqa: E402

Router = asr_backend.NeEnRouterASRBackend


# ── stage 3: hysteresis (pure) ──────────────────────────────────────────────

def test_hysteresis_holds_language_until_margin_cleared():
    raw = [("ne", 0.9), ("en", 0.3), ("en", 0.9), ("ne", 0.4), ("ne", 0.9)]
    # en 0.3 and ne 0.4 are below the 0.6 margin → inherit the running language;
    # only confident predictions flip it. No flapping.
    assert Router._apply_hysteresis(raw, 0.6) == ["ne", "ne", "en", "en", "ne"]


def test_hysteresis_first_segment_sets_language_regardless_of_confidence():
    assert Router._apply_hysteresis([("en", 0.01)], 0.6) == ["en"]


# ── stage 2: LID restricted to ne / en ──────────────────────────────────────

def test_detect_lang_ignores_other_languages():
    # Hindi outscores both, but the router only ever chooses ne or en.
    probs = {"hi": 0.5, "ne": 0.3, "en": 0.2}
    wm = SimpleNamespace(transcribe=lambda audio, language=None, **kw: (
        iter(()),
        SimpleNamespace(all_language_probs=list(probs.items()),
                        language="hi", language_probability=0.5),
    ))
    bk = Router()
    bk._whisper = SimpleNamespace(_model=wm)
    lang, conf = bk._detect_lang(np.zeros(16000, dtype=np.float32))
    assert lang == "ne"
    assert conf == pytest.approx(0.3 / 0.5)  # renormalised over ne+en only


# ── full pipeline: route + merge (fakes) ────────────────────────────────────

class _FakeWord:
    def __init__(self, word, start, end):
        self.word, self.start, self.end = word, start, end


class _FakeSeg:
    def __init__(self, start, end, text, words=()):
        self.start, self.end, self.text = start, end, text
        self.words = list(words)


class _FakeWM:
    """language=None → detection (counter-driven); language="en" → main pass."""

    def __init__(self, main_segs, det_probs):
        self._main, self._det, self._n = main_segs, det_probs, 0

    def transcribe(self, audio, language=None, **kw):
        if language is None:
            probs = self._det[self._n]
            self._n += 1
            return iter(()), SimpleNamespace(
                all_language_probs=list(probs.items()),
                language=max(probs, key=probs.get),
                language_probability=max(probs.values()),
            )
        return list(self._main), SimpleNamespace(
            all_language_probs=None, language="en", language_probability=1.0)


def _fake_indic_cls():
    counter = {"n": 0}

    class _FakeIndic:
        id = "indic-conformer"

        @staticmethod
        def is_available():
            return True, "ready"

        def transcribe(self, path, *, word_timestamps=True):
            counter["n"] += 1
            text = f"NE{counter['n']}"
            return {"text": text,
                    "segments": [{"text": text, "start": 0.0, "end": 1.0, "words": []}],
                    "language": "ne"}

        def unload(self):
            pass

    return _FakeIndic


def test_code_switch_routes_and_merges_in_order(monkeypatch):
    # नमस्ते मेरो | account | को balance कति छ  →  ne, en, ne
    segs = [
        _FakeSeg(0.0, 2.0, "romanized nepali", [_FakeWord("x", 0.0, 2.0)]),
        _FakeSeg(2.0, 3.0, "account", [_FakeWord("account", 2.0, 3.0)]),
        _FakeSeg(3.0, 5.0, "more romanized", [_FakeWord("y", 3.0, 5.0)]),
    ]
    det = [{"ne": 0.9, "en": 0.1}, {"en": 0.9, "ne": 0.1}, {"ne": 0.9, "en": 0.1}]

    monkeypatch.setattr(asr_backend, "_decode_audio_16k_mono",
                        lambda p: np.zeros(5 * 16000, dtype=np.float32))
    monkeypatch.setattr(asr_backend, "_indic_conformer", _fake_indic_cls)

    bk = Router()
    bk._whisper = SimpleNamespace(_model=_FakeWM(segs, det), unload=lambda: None)
    out = bk.transcribe("x.wav")

    # English span keeps Whisper's Latin text; Nepali spans come from IndicConformer;
    # merged in chronological order.
    assert out["text"] == "NE1 account NE2"
    assert [s["language"] for s in out["segments"]] == ["ne", "en", "ne"]
    assert out["segments"][1]["text"] == "account"
    # timestamps preserved + non-decreasing.
    starts = [s["start"] for s in out["segments"]]
    assert starts == sorted(starts)


def test_indic_unavailable_keeps_whisper_text(monkeypatch):
    segs = [_FakeSeg(0.0, 2.0, "hello world", [_FakeWord("hello", 0.0, 2.0)])]
    det = [{"ne": 0.9, "en": 0.1}]  # detected Nepali...

    class _Unavailable:
        @staticmethod
        def is_available():
            return False, "not installed"

    monkeypatch.setattr(asr_backend, "_decode_audio_16k_mono",
                        lambda p: np.zeros(2 * 16000, dtype=np.float32))
    monkeypatch.setattr(asr_backend, "_indic_conformer", lambda: _Unavailable)

    bk = Router()
    bk._whisper = SimpleNamespace(_model=_FakeWM(segs, det), unload=lambda: None)
    out = bk.transcribe("x.wav")
    # ...but with IndicConformer missing, the span must not be lost — keep Whisper.
    assert out["text"] == "hello world"


def test_registered_and_labelled():
    assert asr_backend._REGISTRY["ne-en-router"] is Router
    assert Router.display_name == "Nepali + English (Auto)"
