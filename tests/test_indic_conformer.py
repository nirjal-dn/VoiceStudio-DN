"""IndicConformer ASR engine: adapter contract, with the model runtime stubbed."""
from __future__ import annotations

import math

import numpy as np
import pytest
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
    # Each pause-delimited sentence ends with the danda (।, पूर्णविराम).
    assert [s["text"] for s in result["segments"]] == ["मेरो अकाउन्टमा अपडेट भएको।", "छैन।"]
    assert result["text"] == "मेरो अकाउन्टमा अपडेट भएको। छैन।"
    words = result["segments"][0]["words"]
    assert [w["word"].strip() for w in words] == ["मेरो", "अकाउन्टमा", "अपडेट", "भएको"]
    assert words[1]["start"] == 0.3 and words[1]["end"] == 1.2
    assert result["chunks"][1] == {"text": "छैन।", "timestamp": (3.5, 3.9)}


def test_word_timestamps_can_be_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(ic, "_load", lambda: _StubRuntime([("पोलिसी", 0.0, 0.5, 0.9)]))
    result = ic.IndicConformerBackend().transcribe(_wav(tmp_path), word_timestamps=False)
    assert result["text"] == "पोलिसी।"
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


def test_sentence_final_danda_is_added_per_segment():
    # Devanagari: one danda per pause-delimited sentence, none doubled when the
    # model already emitted a trailing danda (via clean_word's `|` -> `।`).
    assert ic._terminate("मेरो नाम", "ne") == "मेरो नाम।"
    assert ic._terminate("मेरो नाम।", "ne") == "मेरो नाम।"
    assert ic._terminate("", "ne") == ""
    # Non-Devanagari scripts keep their own punctuation conventions — no danda.
    assert ic._terminate("سلام", "ur") == "سلام"


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


def test_model_snapshot_requires_runtime_assets(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    for name in ("encoder.onnx", "vocab.json"):
        (assets / name).touch()

    missing = ic._missing_assets(str(tmp_path))

    assert missing == ["preprocessor.ts", "ctc_decoder.onnx", "language_masks.json"]


# ── CTC decoding (pure; no ONNX) ─────────────────────────────────────────────

_VOCAB = ["▁नमस्", "ते", "▁साथी", "|", "<unk>", "▁hello", "▁छ"]
_B = ic._BLANK_ID


def test_ctc_collapses_repeats_joins_pieces_and_times_words():
    path = [0, 0, _B, 1, _B, _B, 2, 2, 2, 3, _B, 6]
    logp = [-0.1] * len(path)
    words = ic.ctc_words(path, logp, _VOCAB, "ne")
    assert [w[0] for w in words] == ["नमस्ते", "साथी।", "छ"]
    (_, s0, e0, p0), (_, s1, e1, _), (_, s2, e2, _) = words
    assert (s0, e0) == (0.0, pytest.approx(4 * ic._FRAME_S))
    assert s1 == pytest.approx(6 * ic._FRAME_S) and e1 == pytest.approx(10 * ic._FRAME_S)
    assert e2 == pytest.approx(12 * ic._FRAME_S)
    assert p0 == pytest.approx(math.exp(-0.1))


def test_ctc_blank_between_identical_tokens_emits_both():
    words = ic.ctc_words([6, _B, 6], [-0.2, -0.2, -0.2], _VOCAB, "ne")
    assert [w[0] for w in words] == ["छ", "छ"]


def test_ctc_drops_unknown_and_non_devanagari_words_for_nepali():
    words = ic.ctc_words([4, _B, 5, _B, 2], [-0.1] * 5, _VOCAB, "ne")
    assert [w[0] for w in words] == ["साथी"]


def test_ctc_all_blank_or_empty_is_no_words():
    assert ic.ctc_words([_B] * 50, [-0.01] * 50, _VOCAB, "ne") == []
    assert ic.ctc_words([], [], _VOCAB, "ne") == []


# ── audio edge cases through transcribe() ───────────────────────────────────


class _CountingRuntime(_StubRuntime):
    def __init__(self, words=()):
        super().__init__(words)
        self.audio = []

    def words(self, audio, lang):
        self.audio.append(audio)
        return list(self._words)


def test_very_short_audio_skips_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr(ic, "_load", lambda: pytest.fail("model loaded for 50 ms of audio"))
    result = ic.IndicConformerBackend().transcribe(_wav(tmp_path, seconds=0.05))
    assert result == {"text": "", "segments": [], "chunks": [], "language": "ne"}


def test_empty_audio_file_transcribes_to_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(ic, "_load", lambda: pytest.fail("model loaded for empty audio"))
    path = tmp_path / "empty.wav"
    sf.write(path, np.zeros(0, dtype=np.float32), 16000)
    assert ic.IndicConformerBackend().transcribe(str(path))["text"] == ""


def test_stereo_48k_upload_reaches_the_model_as_16k_mono(tmp_path, monkeypatch):
    runtime = _CountingRuntime([("नमस्ते", 0.0, 0.5, 0.9)])
    monkeypatch.setattr(ic, "_load", lambda: runtime)
    path = tmp_path / "stereo.wav"
    sf.write(path, np.zeros((48000 * 2, 2), dtype=np.float32), 48000)
    ic.IndicConformerBackend().transcribe(str(path))
    (audio,) = runtime.audio
    assert audio.ndim == 1 and audio.dtype == np.float32 and len(audio) == 32000


def test_invalid_language_setting_fails_the_transcription(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIVOICE_INDIC_CONFORMER_LANG", "xx")
    with pytest.raises(ValueError, match="not supported"):
        ic.IndicConformerBackend().transcribe(_wav(tmp_path))


def test_hindi_setting_keeps_devanagari_and_reports_hindi(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIVOICE_INDIC_CONFORMER_LANG", "hi")
    monkeypatch.setattr(ic, "_load", lambda: _StubRuntime([("नमस्ते", 0.0, 0.4, 0.9)]))
    result = ic.IndicConformerBackend().transcribe(_wav(tmp_path))
    assert result["language"] == "hi" and result["text"] == "नमस्ते।"


# ── model loading, missing model, lifecycle ─────────────────────────────────


@pytest.fixture
def fresh_runtime(monkeypatch):
    monkeypatch.setattr(ic, "_runtime", None)
    built = []

    class _FakeRuntime:
        def __init__(self, root):
            built.append(root)

    monkeypatch.setattr(ic, "_Runtime", _FakeRuntime)
    return built


def _complete_snapshot(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name in ic._REQUIRED_ASSETS:
        (assets / name).touch()
    return str(tmp_path)


def test_cached_snapshot_loads_once_and_is_reused(tmp_path, monkeypatch, fresh_runtime):
    import huggingface_hub

    root = _complete_snapshot(tmp_path)
    calls = []
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download",
        lambda *a, **k: calls.append(k) or root,
    )
    first = ic._load()
    assert ic._load() is first
    assert fresh_runtime == [root] and len(calls) == 1
    assert calls[0]["local_files_only"] is True and calls[0]["revision"] == ic.REVISION

    ic.IndicConformerBackend().unload()
    assert ic._runtime is None
    ic._load()
    assert len(fresh_runtime) == 2  # reloaded after unload


def test_concurrent_first_use_builds_one_runtime(tmp_path, monkeypatch, fresh_runtime):
    import threading
    import time

    import huggingface_hub

    root = _complete_snapshot(tmp_path)

    def slow_snapshot(*a, **k):
        time.sleep(0.05)
        return root

    monkeypatch.setattr(huggingface_hub, "snapshot_download", slow_snapshot)
    threads = [threading.Thread(target=ic._load) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(fresh_runtime) == 1


def test_gated_model_without_token_explains_how_to_get_access(monkeypatch, fresh_runtime):
    import huggingface_hub

    class GatedRepoError(Exception):
        pass

    def snapshot(*a, local_files_only=False, **k):
        if local_files_only:
            raise FileNotFoundError("not cached")
        raise GatedRepoError("401 Client Error")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    with pytest.raises(RuntimeError, match="gated.*Hugging Face"):
        ic._load()
    assert fresh_runtime == [] and ic._runtime is None


def test_offline_without_cache_surfaces_the_network_error(monkeypatch, fresh_runtime):
    import huggingface_hub

    def snapshot(*a, local_files_only=False, **k):
        raise (FileNotFoundError if local_files_only else ConnectionError)("offline")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    with pytest.raises(ConnectionError):
        ic._load()


def test_incomplete_snapshot_is_repaired_by_forced_download(tmp_path, monkeypatch, fresh_runtime):
    import huggingface_hub

    broken = tmp_path / "broken"
    (broken / "assets").mkdir(parents=True)
    repaired = _complete_snapshot(tmp_path / "repaired")
    calls = []

    def snapshot(*a, **k):
        calls.append(k)
        return repaired if k.get("force_download") else str(broken)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    ic._load()
    assert fresh_runtime == [repaired]
    assert calls[-1]["force_download"] is True


def test_snapshot_still_incomplete_after_repair_is_an_error(tmp_path, monkeypatch, fresh_runtime):
    import huggingface_hub

    broken = tmp_path / "broken"
    (broken / "assets").mkdir(parents=True)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(broken))
    with pytest.raises(RuntimeError, match="still missing"):
        ic._load()
    assert fresh_runtime == []


def test_runtime_is_cpu_only():
    assert ic.IndicConformerBackend.gpu_compat == ("cpu",)


# ── long audio: bounded passes, correct timeline ────────────────────────────


def _speechlike(seconds, silences=()):
    """Noise 'speech' with exact-silence gaps at the given (start_s, end_s)."""
    rng = np.random.default_rng(0)
    audio = (0.1 * rng.standard_normal(int(16000 * seconds))).astype(np.float32)
    for a, b in silences:
        audio[int(a * 16000):int(b * 16000)] = 0.0
    return audio


def test_audio_up_to_the_limit_is_a_single_pass():
    audio = _speechlike(ic._MAX_PASS_S)
    assert ic._windows(audio) == [(0, len(audio))]


def test_long_audio_is_cut_into_bounded_contiguous_windows_at_silence():
    audio = _speechlike(300.0, silences=[(85.0, 85.5), (170.0, 170.4), (255.0, 255.6)])
    windows = ic._windows(audio)
    assert len(windows) >= 4
    assert windows[0][0] == 0 and windows[-1][1] == len(audio)
    assert all(a[1] == b[0] for a, b in zip(windows, windows[1:]))  # no gaps, no overlap
    assert all(end - start <= ic._MAX_PASS_S * 16000 for start, end in windows)
    first_cut = windows[0][1] / 16000
    assert 85.0 <= first_cut <= 85.5  # inside the pause, not mid-word


def test_long_recording_words_keep_their_absolute_times(tmp_path, monkeypatch):
    calls = []

    class _Runtime:
        def words(self, audio, lang):
            calls.append(len(audio) / 16000)
            return [("सुरु", 0.0, 0.5, 0.9), ("अन्त्य", len(audio) / 16000 - 1.0, len(audio) / 16000 - 0.5, 0.9)]

    monkeypatch.setattr(ic, "_load", lambda: _Runtime())
    path = tmp_path / "long.wav"
    sf.write(path, _speechlike(200.0, silences=[(80.0, 81.0), (165.0, 166.0)]), 16000)
    result = ic.IndicConformerBackend().transcribe(str(path))

    assert len(calls) >= 3 and max(calls) <= ic._MAX_PASS_S
    starts = [w["start"] for s in result["segments"] for w in s["words"]]
    assert starts == sorted(starts)
    assert starts[-1] == pytest.approx(200.0 - 1.0, abs=0.05)
