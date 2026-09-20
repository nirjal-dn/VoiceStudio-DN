"""Audio normalization + stitching for code-switched (multi-engine) renders.

Two TTS engines rendering one sentence do not agree on sample rate, channel
shape, dtype or loudness, and each pads its own leading/trailing silence. Left
alone that is audible at every language boundary as a rate glitch, a volume
step or a hole in the middle of a sentence.

These helpers put every child engine's output into one shape, then hand the
pieces to the existing chunk joiner
(:func:`services.chunked_tts.concatenate_audio_chunks`) — deliberately, so
there is exactly one crossfade implementation in the codebase.

MVP scope on purpose: one target sample rate, one channel shape, a clamped
gain alignment and a small boundary trim. No speech-rate matching, no F0
matching, no per-segment loudness normalization (which would flatten the
natural dynamics between a phrase and the clause around it).
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

logger = logging.getLogger("omnivoice.code_switch")

#: Crossfade between language spans. Short — a language boundary inside one
#: sentence is not a paragraph break, and anything longer smears the consonant
#: the next span starts on.
DEFAULT_CROSSFADE_MS = 30

#: Most a boundary trim may remove from either end of a span. A span really can
#: start with a quiet consonant, so the trim is capped rather than greedy.
DEFAULT_MAX_TRIM_MS = 120

#: Below this absolute amplitude a sample counts as silence for trimming.
SILENCE_THRESHOLD = 0.005

#: Bounds on the per-segment gain alignment. Conservative: correct a step
#: between engines, never rebuild the take's dynamics.
MIN_GAIN = 0.5
MAX_GAIN = 2.0

#: Peak the stitched result is scaled back to if gain alignment pushes it hot.
PEAK_CEILING = 0.99


def ensure_float_audio(audio):
    """Return *audio* as float32 in roughly [-1, 1].

    Integer PCM tensors are scaled by their dtype's range; float tensors are
    only cast. Never rescales a float tensor — an engine that legitimately
    returns a quiet take must stay quiet.
    """
    import torch

    if not torch.is_tensor(audio):
        audio = torch.as_tensor(audio)
    if audio.dtype == torch.float32:
        return audio
    if audio.dtype.is_floating_point:
        return audio.to(torch.float32)
    # ``abs(min)``, not ``max``: the same 32768 divisor SubprocessBackend uses
    # to decode a sidecar's int16 PCM, so a full-scale negative sample does not
    # come back at -1.00003 and clip on the way out.
    scale = float(abs(torch.iinfo(audio.dtype).min))
    return audio.to(torch.float32) / scale


def ensure_mono(audio):
    """Return *audio* as a 2-D ``(1, n_samples)`` tensor.

    Accepts 1-D ``(n,)``, ``(1, n)``, ``(channels, n)`` and anything with the
    samples on the last axis; extra channels are averaged, because the spans
    have to agree on shape before they can be concatenated.
    """
    import torch

    if not torch.is_tensor(audio):
        audio = torch.as_tensor(audio)
    if audio.ndim == 0:
        return audio.reshape(1, 1)
    if audio.ndim == 1:
        return audio.unsqueeze(0)
    if audio.numel() == 0:
        # An engine that returned nothing: keep the (1, 0) shape the joiner
        # recognises as an empty chunk instead of failing the reshape.
        return audio.reshape(1, 0)
    flat = audio.reshape(-1, audio.shape[-1])
    if flat.shape[0] == 1:
        return flat
    return flat.mean(dim=0, keepdim=True)


def resample_audio(audio, src_sr: int, dst_sr: int):
    """Resample ``(1, n)`` *audio* from *src_sr* to *dst_sr*.

    Uses torchaudio's polyphase resampler when it is importable and falls back
    to linear interpolation otherwise — the fallback is audibly worse but a
    missing optional import must not fail a render.
    """
    import torch

    audio = ensure_mono(ensure_float_audio(audio))
    if src_sr == dst_sr or audio.shape[-1] == 0:
        return audio
    if src_sr <= 0 or dst_sr <= 0:
        raise ValueError(f"sample rates must be positive, got {src_sr} -> {dst_sr}")
    try:
        import torchaudio

        return torchaudio.functional.resample(audio, int(src_sr), int(dst_sr))
    except Exception:  # noqa: BLE001 — quality fallback, never a hard failure
        logger.debug("torchaudio resample unavailable; using linear interpolation")
        target = max(1, round(audio.shape[-1] * dst_sr / src_sr))
        return torch.nn.functional.interpolate(
            audio.unsqueeze(0), size=target, mode="linear", align_corners=False,
        ).squeeze(0)


def trim_boundary_silence(audio, sample_rate: int,
                          max_trim_ms: int = DEFAULT_MAX_TRIM_MS,
                          threshold: float = SILENCE_THRESHOLD):
    """Trim up to *max_trim_ms* of near-silence from each end of *audio*.

    Bounded rather than greedy: the point is to close the gap an engine's own
    padding opens at a language boundary, not to clip a span that genuinely
    begins softly. An all-silent span is returned untouched — there is nothing
    to keep and dropping it is the joiner's call, not this function's.
    """
    audio = ensure_mono(ensure_float_audio(audio))
    n = audio.shape[-1]
    if n == 0 or max_trim_ms <= 0:
        return audio
    loud = (audio.abs() > threshold).any(dim=0).nonzero()
    if loud.numel() == 0:
        return audio
    budget = int(sample_rate * max_trim_ms / 1000)
    first, last = int(loud[0].item()), int(loud[-1].item())
    start = min(first, budget)
    end = max(last + 1, n - budget)
    return audio[..., start:end]


def normalize_segment_gain(audio, target_rms: float,
                           min_gain: float = MIN_GAIN, max_gain: float = MAX_GAIN):
    """Scale *audio* towards *target_rms*, with the gain clamped.

    The clamp is what keeps this an alignment rather than a compressor: a span
    that is twice as loud as its neighbour gets pulled back, a span that is
    silent stays silent.
    """
    import torch

    audio = ensure_mono(ensure_float_audio(audio))
    if audio.shape[-1] == 0 or target_rms <= 0:
        return audio
    rms = float(torch.sqrt(torch.mean(audio.to(torch.float32) ** 2)))
    if rms <= 0:
        return audio
    gain = max(min_gain, min(max_gain, target_rms / rms))
    return audio * gain


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if not ordered:
        return 0.0
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def stitch_segments(segments, *, target_sample_rate: Optional[int] = None,
                    crossfade_ms: int = DEFAULT_CROSSFADE_MS,
                    normalize_gain: bool = True,
                    normalize_sample_rate: bool = True,
                    trim_silence: bool = True,
                    texts=None, sink=None):
    """Join per-language spans into one waveform.

    ``segments`` is a sequence of ``(audio, sample_rate)`` pairs in playback
    order. Returns ``(audio, sample_rate)`` — a ``(1, n)`` float32 tensor at
    one rate. The target rate defaults to the highest rate present, so a
    24 kHz engine is never downsampled to meet a 22.05 kHz one.

    Empty spans are dropped by the shared joiner, which reports them through
    ``texts``/``sink`` exactly as the long-text chunk path does.
    """
    import torch

    from services.chunked_tts import concatenate_audio_chunks

    pairs = [(audio, int(sr)) for audio, sr in segments if audio is not None]
    if not pairs:
        return torch.zeros(1, dtype=torch.float32), int(target_sample_rate or 24_000)

    rates = [sr for _, sr in pairs if sr > 0]
    if target_sample_rate:
        target_sr = int(target_sample_rate)
    elif normalize_sample_rate and rates:
        target_sr = max(rates)
    else:
        target_sr = rates[0] if rates else 24_000

    prepared = []
    for audio, sr in pairs:
        wave = ensure_mono(ensure_float_audio(audio))
        if normalize_sample_rate and sr > 0 and sr != target_sr:
            wave = resample_audio(wave, sr, target_sr)
        if trim_silence:
            wave = trim_boundary_silence(wave, target_sr)
        prepared.append(wave)

    if normalize_gain:
        levels = [
            float(torch.sqrt(torch.mean(w.to(torch.float32) ** 2)))
            for w in prepared if w.shape[-1] > 0
        ]
        reference = _median([level for level in levels if level > 0])
        if reference > 0:
            prepared = [normalize_segment_gain(w, reference) for w in prepared]

    joined = concatenate_audio_chunks(
        prepared, target_sr, max(0, int(crossfade_ms)), texts=texts, sink=sink,
    )
    joined = ensure_mono(ensure_float_audio(joined))

    peak = float(joined.abs().max()) if joined.numel() else 0.0
    if peak > PEAK_CEILING:
        joined = joined * (PEAK_CEILING / peak)
    return joined, target_sr


__all__ = [
    "DEFAULT_CROSSFADE_MS", "DEFAULT_MAX_TRIM_MS", "ensure_float_audio",
    "ensure_mono", "normalize_segment_gain", "resample_audio",
    "stitch_segments", "trim_boundary_silence",
]
