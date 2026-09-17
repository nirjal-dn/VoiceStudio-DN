"""services.asr_backend._load_audio_16k_mono_f32 — the decoder in front of the
IndicConformer and sherpa ASR engines: formats, sample rates, channel layouts,
empty/short/long audio, and corrupt input (ffmpeg fallback)."""
import numpy as np
import pytest
import soundfile as sf

from services import asr_backend as ab


def _tone(seconds, sr, channels=1, freq=440.0):
    t = np.arange(int(seconds * sr)) / sr
    mono = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return mono if channels == 1 else np.stack([mono] * channels, axis=1)


@pytest.fixture
def no_ffmpeg(monkeypatch):
    calls = []

    def fake_decode(path):
        calls.append(path)
        raise RuntimeError("ffmpeg not found — install ffmpeg")

    monkeypatch.setattr(ab, "_decode_audio_16k_mono", fake_decode)
    return calls


@pytest.mark.parametrize(("ext", "subtype"), [
    ("wav", "PCM_16"), ("wav", "FLOAT"), ("flac", "PCM_24"), ("ogg", "VORBIS"),
])
def test_soundfile_formats_decode_without_ffmpeg(tmp_path, no_ffmpeg, ext, subtype):
    path = tmp_path / f"clip.{ext}"
    sf.write(path, _tone(1.0, 16000), 16000, subtype=subtype)
    audio, sr = ab._load_audio_16k_mono_f32(str(path))
    assert sr == 16000 and audio.dtype == np.float32 and audio.ndim == 1
    assert abs(len(audio) - 16000) <= 1
    assert no_ffmpeg == []
    assert np.max(np.abs(audio)) == pytest.approx(0.3, abs=0.02)  # level preserved


@pytest.mark.parametrize("source_sr", [8000, 22050, 44100, 48000])
def test_any_sample_rate_is_resampled_to_16k(tmp_path, source_sr):
    path = tmp_path / "clip.wav"
    sf.write(path, _tone(2.0, source_sr), source_sr)
    audio, sr = ab._load_audio_16k_mono_f32(str(path))
    assert sr == 16000
    assert len(audio) == round(2.0 * 16000)


def test_resampled_tone_keeps_its_pitch(tmp_path):
    path = tmp_path / "clip.wav"
    sf.write(path, _tone(1.0, 44100, freq=300.0), 44100)
    audio, _ = ab._load_audio_16k_mono_f32(str(path))
    spectrum = np.abs(np.fft.rfft(audio))
    peak_hz = np.argmax(spectrum) * 16000 / len(audio)
    assert peak_hz == pytest.approx(300.0, abs=5.0)


@pytest.mark.parametrize("channels", [2, 6])
def test_multichannel_is_downmixed_to_mono(tmp_path, channels):
    path = tmp_path / "multi.wav"
    sf.write(path, _tone(1.0, 16000, channels=channels), 16000)
    audio, _ = ab._load_audio_16k_mono_f32(str(path))
    assert audio.ndim == 1 and len(audio) == 16000


def test_opposite_phase_stereo_downmix_is_an_average(tmp_path):
    left = _tone(0.5, 16000)
    path = tmp_path / "phase.wav"
    sf.write(path, np.stack([left, -left], axis=1), 16000, subtype="FLOAT")
    audio, _ = ab._load_audio_16k_mono_f32(str(path))
    assert np.max(np.abs(audio)) < 1e-6


def test_empty_file_decodes_to_zero_samples(tmp_path):
    path = tmp_path / "empty.wav"
    sf.write(path, np.zeros(0, dtype=np.float32), 44100)
    audio, sr = ab._load_audio_16k_mono_f32(str(path))
    assert sr == 16000 and len(audio) == 0


def test_very_short_audio_is_not_padded(tmp_path):
    path = tmp_path / "blip.wav"
    sf.write(path, _tone(0.01, 48000), 48000)
    audio, _ = ab._load_audio_16k_mono_f32(str(path))
    assert len(audio) == 160


def test_long_audio_keeps_every_sample(tmp_path):
    path = tmp_path / "long.flac"
    sf.write(path, _tone(600.0, 16000), 16000)  # 10 minutes
    audio, _ = ab._load_audio_16k_mono_f32(str(path))
    assert len(audio) == 600 * 16000


def test_unreadable_container_falls_back_to_ffmpeg(tmp_path, monkeypatch):
    path = tmp_path / "voice.webm"
    path.write_bytes(b"\x1aE\xdf\xa3" + b"\x00" * 64)  # WebM magic, not soundfile-readable
    decoded = np.zeros(8000, dtype=np.float32)
    seen = []
    monkeypatch.setattr(ab, "_decode_audio_16k_mono", lambda p: seen.append(p) or decoded)
    audio, sr = ab._load_audio_16k_mono_f32(str(path))
    assert seen == [str(path)] and sr == 16000 and audio is decoded


@pytest.mark.parametrize("payload", [b"", b"not audio at all", b"RIFF\x10\x00\x00\x00WAVEfmt "])
def test_corrupt_audio_without_ffmpeg_raises_an_actionable_error(tmp_path, no_ffmpeg, payload):
    path = tmp_path / "corrupt.wav"
    path.write_bytes(payload)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        ab._load_audio_16k_mono_f32(str(path))
    assert no_ffmpeg == [str(path)]


def test_missing_file_raises(tmp_path, no_ffmpeg):
    with pytest.raises(RuntimeError):
        ab._load_audio_16k_mono_f32(str(tmp_path / "does-not-exist.wav"))


def test_downsampling_removes_content_above_8k_instead_of_aliasing(tmp_path):
    """A 10 kHz tone in a 44.1 kHz file cannot exist at 16 kHz; linear
    interpolation folds it to 6 kHz (in the speech band), a proper resampler
    filters it out."""
    path = tmp_path / "hiss.wav"
    sf.write(path, _tone(1.0, 44100, freq=10000.0), 44100, subtype="FLOAT")
    audio, _ = ab._load_audio_16k_mono_f32(str(path))
    rms = float(np.sqrt(np.mean(np.square(audio[800:-800]))))
    assert rms < 0.3 / np.sqrt(2) * 0.1  # at least 20 dB below the input tone


def test_resampler_falls_back_when_torchaudio_is_unavailable(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_torchaudio(name, *args, **kwargs):
        if name.startswith("torchaudio"):
            raise ImportError("torchaudio missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torchaudio)
    path = tmp_path / "clip.wav"
    sf.write(path, _tone(1.0, 48000), 48000)
    audio, sr = ab._load_audio_16k_mono_f32(str(path))
    assert sr == 16000 and len(audio) == 16000
