"""POST /generate with Nepali input, end to end through the real router and
middleware stack with a fake engine (no model, CPU only)."""
import importlib
import unicodedata

import pytest
import torch

DANDA = "।"


def _tts():
    return importlib.import_module("services.tts_backend")


def _fake_engine(engine_id, fail_on=None):
    class _Engine(_tts().TTSBackend):
        id = engine_id
        display_name = "Fake Nepali Engine (test)"
        calls: list = []

        @property
        def sample_rate(self):
            return 24000

        @property
        def supported_languages(self):
            return ["multi"]

        @classmethod
        def is_available(cls):
            return True, "ready"

        def generate(self, text, **kw):
            type(self).calls.append((text, kw))
            if fail_on and fail_on in text:
                raise RuntimeError("synthesis failed inside the engine")
            return torch.zeros(1, 4800)

    return _Engine


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from main import app

    return TestClient(app, client=("127.0.0.1", 50000))


@pytest.fixture()
def engine(monkeypatch):
    fake = _fake_engine("fake-nepali-engine")
    monkeypatch.setitem(_tts()._REGISTRY, "fake-nepali-engine", fake)
    return fake


def _post(client, text, **extra):
    return client.post("/generate", data={
        "text": text, "engine": "fake-nepali-engine", "language": "Nepali", **extra,
    })


@pytest.mark.parametrize("text", [
    "नमस्ते, मेरो नाम सीता श्रेष्ठ हो।",
    "क्षत्रिय ज्ञान श्रीमान् द्वारा संस्कृत स्वास्थ्य।",
    "मेरो Gmail account मा OTP आएन, कृपया IT टिमलाई खबर गर्नुहोस्।",
])
def test_nepali_text_reaches_the_engine_unchanged(client, engine, text):
    res = _post(client, text)
    assert res.status_code == 200, res.text
    assert [t for t, _ in engine.calls] == [text]
    assert engine.calls[0][1]["language"] == "Nepali"
    assert res.headers["content-type"].startswith("audio/")


@pytest.mark.parametrize(("text", "spoken"), [
    ("मेरो Gmail account मा 2 वटा OTP आए।", "मेरो Gmail account मा दुई वटा OTP आए।"),
    (
        "वि.सं. २०८१-०४-१५ गते, रु. १,५०,००० भुक्तानी भयो!",
        "वि.सं. दुई हजार एकासी साउन पन्ध्र गते, रु. एक लाख पचास हजार भुक्तानी भयो!",
    ),
    ("बैठक 10:30 बजे, 50% उपस्थिति", "बैठक दश बजेर तीस मिनेट, पचास प्रतिशत उपस्थिति"),
])
def test_nepali_numbers_dates_and_times_are_spoken_for_every_engine(client, engine, monkeypatch, text, spoken):
    monkeypatch.setenv("OMNIVOICE_TEXT_NORMALIZATION", "1")
    res = _post(client, text)
    assert res.status_code == 200, res.text
    assert [t for t, _ in engine.calls] == [spoken]


def test_decomposed_devanagari_is_composed_before_synthesis(client, engine):
    decomposed = "ऩया"  # न + nukta, which NFC composes to U+0929
    assert unicodedata.normalize("NFC", decomposed) != decomposed
    res = _post(client, decomposed)
    assert res.status_code == 200, res.text
    assert engine.calls[0][0] == unicodedata.normalize("NFC", decomposed)


@pytest.mark.parametrize("blank", ["   ", "\n\t ", "　"])
def test_whitespace_only_text_is_rejected_before_the_engine(client, engine, blank):
    res = _post(client, blank)
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()
    assert engine.calls == []


def test_missing_text_is_a_validation_error(client, engine):
    res = client.post("/generate", data={"engine": "fake-nepali-engine"})
    assert res.status_code == 422
    assert engine.calls == []


def test_long_nepali_text_is_chunked_at_danda_without_losing_words(client, engine):
    sentence = (
        "काठमाडौं महानगरपालिकाले आगामी आर्थिक वर्षका लागि नयाँ बजेट सार्वजनिक गरेको छ। "
        "बजेटमा शिक्षा, स्वास्थ्य र पूर्वाधार विकासलाई प्राथमिकता दिइएको छ। "
    )
    text = (sentence * 12).strip()
    res = _post(client, text, max_chunk_chars="300")
    assert res.status_code == 200, res.text
    chunks = [t for t, _ in engine.calls]
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    assert all(c.endswith(DANDA) for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_engine_failure_is_reported_not_swallowed(client, monkeypatch):
    fake = _fake_engine("fake-nepali-failing", fail_on="असफल")
    monkeypatch.setitem(_tts()._REGISTRY, "fake-nepali-failing", fake)
    res = client.post("/generate", data={"text": "यो असफल हुनेछ।", "engine": "fake-nepali-failing"})
    assert res.status_code >= 500
    assert "synthesis failed inside the engine" in res.text


def test_legacy_oshara_alias_routes_to_xtts_nepali_and_reports_missing_venv(client, tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIVOICE_XTTS_NEPALI_DIR", str(tmp_path))
    res = client.post("/generate", data={"text": "नमस्ते।", "engine": "oshara-xtts-v2"})
    assert res.status_code == 400
    assert "xtts-nepali" in res.text and "venv" in res.text.lower()


def test_unknown_engine_is_a_client_error(client):
    res = client.post("/generate", data={"text": "नमस्ते।", "engine": "no-such-engine"})
    assert 400 <= res.status_code < 500


# ── seeds reach engines that seed themselves (sidecars) ─────────────────────


def test_seed_is_forwarded_to_engines_that_accept_it(client, monkeypatch):
    seeded = _fake_engine("fake-nepali-seeded")
    seeded.accepts_seed = True
    plain = _fake_engine("fake-nepali-plain")
    monkeypatch.setitem(_tts()._REGISTRY, seeded.id, seeded)
    monkeypatch.setitem(_tts()._REGISTRY, plain.id, plain)

    assert client.post("/generate", data={"text": "नमस्ते।", "engine": seeded.id, "seed": "42"}).status_code == 200
    assert client.post("/generate", data={"text": "नमस्ते।", "engine": plain.id, "seed": "42"}).status_code == 200
    assert seeded.calls[-1][1].get("seed") == 42
    assert "seed" not in plain.calls[-1][1]


def test_chunked_nepali_text_gets_a_distinct_seed_per_chunk(client, monkeypatch):
    seeded = _fake_engine("fake-nepali-seeded-chunks")
    seeded.accepts_seed = True
    monkeypatch.setitem(_tts()._REGISTRY, seeded.id, seeded)
    text = ("काठमाडौं उपत्यकामा आज बिहानदेखि हल्का वर्षा भइरहेको छ। " * 12).strip()
    res = client.post("/generate", data={
        "text": text, "engine": seeded.id, "seed": "7", "max_chunk_chars": "200",
    })
    assert res.status_code == 200, res.text
    seeds = [kw.get("seed") for _, kw in seeded.calls]
    assert len(seeds) > 1 and seeds == list(range(7, 7 + len(seeds)))


def test_nepali_engines_declare_their_capabilities():
    from engines.indic_parler import IndicParlerBackend
    from engines.xtts_nepali import XttsNepaliBackend

    assert XttsNepaliBackend.uses_ref_text is False and XttsNepaliBackend.accepts_seed is True
    assert XttsNepaliBackend.default_language == "ne"
    assert IndicParlerBackend.supports_voice_design is True
    assert IndicParlerBackend.uses_ref_text is False and IndicParlerBackend.default_language == "ne"
    rows = {r["id"]: r for r in _tts().list_backends()}
    assert rows["indic-parler-tts"]["default_language"] == "ne"
    assert rows["xtts-nepali"]["default_language"] == "ne"


# ── response body is the saved take (no second encode) ──────────────────────


def test_response_body_is_byte_identical_to_the_saved_take(client, engine):
    import os

    from api.routers import generation

    res = _post(client, "नमस्ते साथी।")
    assert res.status_code == 200, res.text
    saved = os.path.join(generation.OUTPUTS_DIR, res.headers["X-Audio-Path"])
    with open(saved, "rb") as fh:
        assert res.content == fh.read()
    assert res.headers["Content-Length"] == str(len(res.content))


def test_saved_take_bytes_falls_back_to_encoding_when_the_file_is_gone(tmp_path, monkeypatch):
    import io

    from api.routers import generation
    from services.audio_io import _safe_torchaudio_save

    monkeypatch.setattr(generation, "OUTPUTS_DIR", str(tmp_path))
    audio = torch.linspace(-0.5, 0.5, 2400).unsqueeze(0)
    expected = io.BytesIO()
    _safe_torchaudio_save(expected, audio, 24000, format="wav")
    assert generation._saved_take_bytes("pruned.wav", audio, 24000) == expected.getvalue()
