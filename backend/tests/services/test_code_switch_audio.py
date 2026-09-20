"""Stitching two engines' output into one stream.

Covers the shapes the child engines really produce: 24 kHz + 24 kHz (both XTTS
variants), 24 kHz + 22.05 kHz (a differently-rated engine in the English slot),
1-D and (1, n) tensors, empty and very short spans, and both ends of the
crossfade range.
"""
from __future__ import annotations

import math

import pytest
import torch

from services.code_switch_audio import (
    DEFAULT_CROSSFADE_MS, ensure_float_audio, ensure_mono,
    normalize_segment_gain, resample_audio, stitch_segments,
    trim_boundary_silence,
)


def _tone(seconds: float, sample_rate: int, freq: float = 220.0, amp: float = 0.5,
          dims: int = 2):
    n = int(seconds * sample_rate)
    t = torch.arange(n, dtype=torch.float32) / sample_rate
    wave = amp * torch.sin(2 * math.pi * freq * t)
    return wave if dims == 1 else wave.unsqueeze(0)


# ── shape / dtype normalization ───────────────────────────────────────────

@pytest.mark.parametrize("shape", [(100,), (1, 100), (2, 100), (1, 1, 100)])
def test_ensure_mono_always_returns_one_channel_two_dims(shape):
    out = ensure_mono(torch.rand(*shape))
    assert out.ndim == 2 and out.shape[0] == 1 and out.shape[-1] == 100


def test_ensure_mono_averages_extra_channels():
    stereo = torch.stack([torch.full((4,), 1.0), torch.full((4,), -1.0)])
    assert torch.allclose(ensure_mono(stereo), torch.zeros(1, 4))


def test_ensure_float_audio_scales_int_pcm_and_leaves_float_alone():
    pcm = torch.tensor([0, 32767, -32768], dtype=torch.int16)
    out = ensure_float_audio(pcm)
    assert out.dtype == torch.float32
    assert out.max() <= 1.0 and out.min() >= -1.0
    quiet = torch.full((8,), 0.01)
    assert torch.equal(ensure_float_audio(quiet), quiet)


# ── resampling ────────────────────────────────────────────────────────────

def test_resample_is_a_no_op_at_the_same_rate():
    audio = _tone(0.1, 24_000)
    assert torch.equal(resample_audio(audio, 24_000, 24_000), audio)


def test_resample_lands_on_the_expected_length():
    audio = _tone(0.5, 22_050)
    out = resample_audio(audio, 22_050, 24_000)
    assert out.shape[0] == 1
    assert abs(out.shape[-1] - round(audio.shape[-1] * 24_000 / 22_050)) <= 2


def test_resample_rejects_a_nonsense_rate():
    with pytest.raises(ValueError):
        resample_audio(_tone(0.1, 24_000), 0, 24_000)


# ── trimming / gain ───────────────────────────────────────────────────────

def test_trim_removes_padding_but_only_up_to_the_budget():
    sr = 24_000
    pad = torch.zeros(1, sr)  # a full second of silence, budget is 120 ms
    body = _tone(0.2, sr)
    trimmed = trim_boundary_silence(torch.cat([pad, body, pad], dim=-1), sr)
    expected = body.shape[-1] + 2 * (sr - int(sr * 120 / 1000))
    assert trimmed.shape[-1] == expected


def test_trim_leaves_an_all_silent_span_alone():
    silence = torch.zeros(1, 480)
    assert torch.equal(trim_boundary_silence(silence, 24_000), silence)


def test_gain_alignment_is_clamped_in_both_directions():
    quiet = torch.full((1, 100), 0.001)
    loud = torch.full((1, 100), 0.9)
    assert float(normalize_segment_gain(quiet, 1.0).abs().max()) == pytest.approx(0.002)
    assert float(normalize_segment_gain(loud, 0.001).abs().max()) == pytest.approx(0.45)


def test_gain_alignment_never_divides_by_a_silent_span():
    silence = torch.zeros(1, 100)
    assert torch.equal(normalize_segment_gain(silence, 0.2), silence)


# ── stitching ─────────────────────────────────────────────────────────────

def test_two_matched_24k_spans_join_at_24k():
    a, b = _tone(0.3, 24_000), _tone(0.3, 24_000, freq=330.0)
    out, sr = stitch_segments([(a, 24_000), (b, 24_000)])
    assert sr == 24_000
    assert out.ndim == 2 and out.shape[0] == 1
    overlap = int(24_000 * DEFAULT_CROSSFADE_MS / 1000)
    # Defaults on, so the boundary trim may shave a couple of near-zero samples
    # off a sine that starts at zero crossing.
    assert abs(out.shape[-1] - (a.shape[-1] + b.shape[-1] - overlap)) <= 8


def test_mismatched_rates_resolve_upward_to_the_higher_one():
    out, sr = stitch_segments(
        [(_tone(0.2, 24_000), 24_000), (_tone(0.2, 22_050), 22_050)],
        trim_silence=False, normalize_gain=False,
    )
    assert sr == 24_000
    expected = int(0.2 * 24_000) * 2 - int(24_000 * DEFAULT_CROSSFADE_MS / 1000)
    assert abs(out.shape[-1] - expected) <= 4


def test_one_dimensional_input_is_accepted():
    out, sr = stitch_segments([
        (_tone(0.1, 24_000, dims=1), 24_000),
        (_tone(0.1, 24_000, dims=1), 24_000),
    ])
    assert out.ndim == 2 and out.shape[0] == 1 and sr == 24_000


def test_empty_spans_are_dropped_not_crashed_on():
    body = _tone(0.2, 24_000)
    out, sr = stitch_segments([
        (torch.zeros(1, 0), 24_000), (body, 24_000), (torch.zeros(1, 0), 24_000),
    ], trim_silence=False, normalize_gain=False)
    assert sr == 24_000
    assert out.shape[-1] == body.shape[-1]


def test_no_spans_at_all_returns_silence_not_an_exception():
    out, sr = stitch_segments([])
    assert out.numel() >= 1 and sr == 24_000


def test_very_short_spans_survive_a_crossfade_longer_than_they_are():
    tiny = _tone(0.002, 24_000)  # 48 samples, crossfade wants 720
    out, sr = stitch_segments([(tiny, 24_000), (tiny, 24_000)],
                              trim_silence=False, normalize_gain=False)
    assert sr == 24_000
    assert 0 < out.shape[-1] <= 2 * tiny.shape[-1]


def test_zero_crossfade_is_a_hard_concatenation():
    a, b = _tone(0.1, 24_000), _tone(0.1, 24_000, freq=440.0)
    out, _ = stitch_segments([(a, 24_000), (b, 24_000)], crossfade_ms=0,
                             trim_silence=False, normalize_gain=False)
    assert out.shape[-1] == a.shape[-1] + b.shape[-1]


def test_thirty_ms_crossfade_overlaps_exactly_thirty_ms():
    a, b = _tone(0.4, 24_000), _tone(0.4, 24_000)
    out, _ = stitch_segments([(a, 24_000), (b, 24_000)], crossfade_ms=30,
                             trim_silence=False, normalize_gain=False)
    assert out.shape[-1] == a.shape[-1] + b.shape[-1] - int(24_000 * 30 / 1000)


def test_gain_alignment_removes_the_volume_step_between_engines():
    loud = _tone(0.3, 24_000, amp=0.8)
    quiet = _tone(0.3, 24_000, amp=0.2)
    out, _ = stitch_segments([(loud, 24_000), (quiet, 24_000)],
                             crossfade_ms=0, trim_silence=False)
    half = out.shape[-1] // 2
    left = float(out[..., :half].abs().mean())
    right = float(out[..., half:].abs().mean())
    assert max(left, right) / min(left, right) < 2.2  # was 4x before alignment


def test_the_stitched_take_never_clips():
    hot = _tone(0.2, 24_000, amp=0.98)
    out, _ = stitch_segments([(hot, 24_000), (_tone(0.2, 24_000, amp=0.2), 24_000)])
    assert float(out.abs().max()) <= 0.99 + 1e-6


def test_an_explicit_target_rate_wins_over_the_children():
    out, sr = stitch_segments([(_tone(0.2, 24_000), 24_000)],
                              target_sample_rate=16_000, trim_silence=False)
    assert sr == 16_000
    assert abs(out.shape[-1] - int(0.2 * 16_000)) <= 2
