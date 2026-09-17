"""Indic Parler-TTS engine: registry, availability, sentence splitting (no model)."""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import torch

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


# ── sidecar: text handling and synthesis with a fake Parler model ───────────

@pytest.mark.parametrize("text", ["", "   ", "।", "? ! .", "\n\t"])
def test_nothing_speakable_splits_to_no_sentences(text):
    assert sidecar.sentences(text) == []


def test_long_nepali_paragraph_splits_on_every_danda():
    paragraph = (
        "नेपालमा मनसुन सामान्यतया असार महिनादेखि सुरु हुन्छ। "
        "यस वर्ष भने केही ढिलो भित्रिएको मौसम पूर्वानुमान महाशाखाले जनाएको छ। "
        "किसानहरूले धान रोपाइँको तयारी गरिरहेका छन्॥ "
    ) * 20
    parts = sidecar.sentences(paragraph)
    assert len(parts) == 60
    assert all(p.endswith(("।", "॥")) for p in parts)


class _Batch:
    def __init__(self, text):
        self.input_ids = torch.tensor([[len(text)]])
        self.attention_mask = torch.ones(1, 1)

    def to(self, device):
        return self


class _FakeParler:
    def __init__(self):
        self.config = type("C", (), {"sampling_rate": 44100})()
        self.prompts = []

    def generate(self, input_ids, attention_mask, prompt_input_ids, prompt_attention_mask):
        self.prompts.append(int(prompt_input_ids[0, 0]))
        return torch.full((1, 4410), 0.25)


def _install(monkeypatch):
    model = _FakeParler()
    descriptions = []

    def desc_tok(text, return_tensors=None):
        descriptions.append(text)
        return _Batch(text)

    monkeypatch.setattr(sidecar, "_model", (model, lambda t, return_tensors=None: _Batch(t), desc_tok, "cpu"))
    return model, descriptions


def _audio_frame(out):
    out.seek(0)
    frames = []
    while (frame := sidecar._recv(out)) is not None:
        frames.append(frame)
    return frames[-1]


def test_synthesis_renders_each_sentence_with_gaps(monkeypatch):
    model, descriptions = _install(monkeypatch)
    out = io.BytesIO()
    sidecar._handle_synthesize({"text": "नमस्ते। म अमृता हुँ। तपाईंलाई कस्तो छ?"}, out)
    frame = _audio_frame(out)
    assert len(model.prompts) == 3
    gap = int(sidecar._GAP_S * 44100)
    assert frame["sample_rate"] == 44100
    assert frame["n_samples"] == 3 * 4410 + 2 * gap
    pcm = np.frombuffer(base64.b64decode(frame["audio_pcm_b64"]), dtype=np.int16)
    assert pcm.size == frame["n_samples"]
    assert descriptions == [sidecar.DEFAULT_DESCRIPTION]


def test_instruct_overrides_the_default_voice_description(monkeypatch):
    _, descriptions = _install(monkeypatch)
    sidecar._handle_synthesize(
        {"text": "नमस्ते।", "instruct": "  Suresh speaks slowly in a deep voice.  "}, io.BytesIO()
    )
    assert descriptions == ["Suresh speaks slowly in a deep voice."]


@pytest.mark.parametrize("msg", [{}, {"text": ""}, {"text": None}, {"text": ["नमस्ते"]}, {"text": " । "}])
def test_invalid_text_fails_before_loading_the_model(monkeypatch, msg):
    monkeypatch.setattr(sidecar, "_load", lambda stdout: pytest.fail("model loaded"))
    with pytest.raises(ValueError):
        sidecar._handle_synthesize(msg, io.BytesIO())


def test_frames_round_trip_and_reject_oversize():
    import struct

    buf = io.BytesIO()
    sidecar._send(buf, {"op": "synthesize", "text": "श्रीमान् क्षत्रिय"})
    buf.seek(0)
    assert sidecar._recv(buf)["text"] == "श्रीमान् क्षत्रिय"
    with pytest.raises(IOError, match="too large"):
        sidecar._recv(io.BytesIO(struct.pack("!I", sidecar.MAX_FRAME_BYTES + 1)))
