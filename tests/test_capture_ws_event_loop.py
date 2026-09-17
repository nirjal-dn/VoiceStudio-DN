"""Dictation must not freeze the backend while it turns the audio buffer into a
WAV: legacy WebM sessions run ffmpeg on the whole buffer every ~2 s."""
import asyncio
import time

import pytest

from api.routers import capture_ws


async def _max_loop_gap(coro, tick=0.02):
    gaps = []
    done = asyncio.Event()

    async def ticker():
        last = time.monotonic()
        while not done.is_set():
            await asyncio.sleep(tick)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    try:
        result = await coro
    finally:
        done.set()
        await task
    return result, max(gaps)


def _slow(seconds):
    def convert(*args, **kwargs):
        time.sleep(seconds)  # stands in for ffmpeg decoding a long buffer
        return None

    return convert


@pytest.mark.parametrize("pcm_sr", [None, 16000])
def test_partial_transcription_keeps_the_event_loop_responsive(monkeypatch, pcm_sr):
    monkeypatch.setattr(capture_ws, "_chunks_to_wav", _slow(0.5))
    monkeypatch.setattr(capture_ws, "_pcm16_to_wav", _slow(0.5))
    text, gap = asyncio.run(_max_loop_gap(capture_ws._transcribe_buffer([b"\x00" * 3200] * 50, pcm_sr=pcm_sr)))
    assert text == ""
    assert gap < 0.25, f"event loop stalled {gap:.2f}s during buffer conversion"


@pytest.mark.parametrize("pcm_sr", [None, 16000])
def test_final_transcription_keeps_the_event_loop_responsive(monkeypatch, pcm_sr):
    monkeypatch.setattr(capture_ws, "_chunks_to_wav", _slow(0.5))
    monkeypatch.setattr(capture_ws, "_pcm16_to_wav", _slow(0.5))
    result, gap = asyncio.run(_max_loop_gap(capture_ws._transcribe_buffer_full([b"\x00" * 3200] * 50, pcm_sr=pcm_sr)))
    assert result["text"] == ""
    assert gap < 0.25, f"event loop stalled {gap:.2f}s during buffer conversion"


def test_conversion_output_is_unchanged(tmp_path):
    """The offloaded conversion writes the same WAV as before."""
    import wave

    pcm = (b"\x10\x00\xf0\xff" * 4000)
    path = capture_ws._pcm16_to_wav(pcm, 16000)
    try:
        with wave.open(path, "rb") as wf:
            assert (wf.getnchannels(), wf.getsampwidth(), wf.getframerate()) == (1, 2, 16000)
            assert wf.readframes(wf.getnframes()) == pcm
    finally:
        import os

        os.unlink(path)
