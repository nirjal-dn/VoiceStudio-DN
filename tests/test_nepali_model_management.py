"""Nepali models are managed like every other model: catalogued, pinned to a
reviewed revision, preflighted before any download, and loaded once."""
import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from services import hf_revisions

ROOT = Path(__file__).resolve().parents[1]
NEPALI_REPOS = {
    "ai4bharat/indic-conformer-600m-multilingual": ("ASR", ["indic-conformer"]),
    "Oshara/xtts-v2-nepali": ("TTS", ["xtts-nepali"]),
    "ai4bharat/indic-parler-tts": ("TTS", ["indic-parler-tts"]),
}


def _load_sidecar(rel):
    spec = importlib.util.spec_from_file_location(f"sidecar_{rel.replace('/', '_')}", ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    saved = os.environ.get("TORCHAUDIO_USE_TORCHCODEC")
    spec.loader.exec_module(module)
    if saved is None:
        os.environ.pop("TORCHAUDIO_USE_TORCHCODEC", None)
    return module


# ── catalogue + pins ────────────────────────────────────────────────────────


def test_nepali_models_are_in_the_catalogue_with_their_engines():
    catalog = {m["repo_id"]: m for m in yaml.safe_load((ROOT / "backend/config/models.yaml").read_text())["models"]}
    for repo, (role, engines) in NEPALI_REPOS.items():
        assert catalog[repo]["role"] == role and catalog[repo]["engines"] == engines
        assert catalog[repo]["size_gb"] > 0 and not catalog[repo].get("required")
    assert catalog["Oshara/xtts-v2-nepali"]["allow_patterns"] == ["epoch-20/*"]


def test_engine_revisions_match_the_reviewed_pins():
    from engines import indic_conformer

    xtts = _load_sidecar("backend/engines/xtts_nepali/main.py")
    parler = _load_sidecar("backend/engines/indic_parler/main.py")
    assert indic_conformer.REVISION == hf_revisions.revision_for(indic_conformer.REPO_ID)
    assert xtts._REVISION == hf_revisions.revision_for(xtts._REPO_ID)
    assert parler.REVISION == hf_revisions.revision_for(parler.REPO_ID)


# ── missing-model preflight (no silent multi-GB download) ───────────────────


@pytest.fixture
def indic_pinned(monkeypatch):
    from services import asr_backend

    asr_backend._INSTALLED_REPO_MEMO.clear()
    monkeypatch.setattr(asr_backend, "_asr_backend_pinned", lambda: True)
    monkeypatch.setattr(asr_backend, "active_backend_id", lambda: "indic-conformer")
    yield asr_backend
    asr_backend._INSTALLED_REPO_MEMO.clear()


@pytest.mark.parametrize("purpose", ["transcribe", "dictation"])
def test_uninstalled_indic_conformer_answers_with_its_own_download(indic_pinned, purpose):
    from api.routers.setup import models as setup_models

    with patch.object(setup_models, "is_cached", return_value=False):
        payload = indic_pinned.asr_model_missing_error(purpose=purpose)
    assert payload["error"] == "asr_model_missing"
    assert payload["missing_repo_id"] == "ai4bharat/indic-conformer-600m-multilingual"
    assert payload["recommended"]["repo_id"] == "ai4bharat/indic-conformer-600m-multilingual"


@pytest.mark.parametrize("purpose", ["transcribe", "dictation"])
def test_installed_indic_conformer_passes_the_preflight(indic_pinned, purpose):
    from api.routers.setup import models as setup_models

    with patch.object(setup_models, "is_cached", return_value=True), \
         patch.object(setup_models, "cache_is_complete", return_value=True):
        assert indic_pinned.asr_model_missing_error(purpose=purpose) is None


def test_dictation_socket_refuses_before_downloading(indic_pinned, monkeypatch):
    """The whole-recording dictation path is preflighted too."""
    from fastapi.testclient import TestClient

    from api.routers.setup import models as setup_models
    from main import app

    monkeypatch.setattr(indic_pinned, "pinned_whole_recording_engine", lambda: "indic-conformer")
    with patch.object(setup_models, "is_cached", return_value=False):
        client = TestClient(app, client=("127.0.0.1", 50000))
        with client.websocket_connect("/ws/transcribe") as ws:
            frame = ws.receive_json()
    assert frame.get("error") == "asr_model_missing" or frame.get("type") == "error"
    assert "indic-conformer" in str(frame)


# ── XTTS model loading (fake coqui-tts; CPU) ────────────────────────────────


@pytest.fixture
def fake_coqui(monkeypatch, tmp_path):
    calls = {"snapshot": [], "loaded": 0, "device": []}

    class XttsConfig:
        languages = ["en", "hi", "ne"]

        def load_json(self, path):
            assert path.endswith("config.json")

    class _Tokenizer:
        char_limits = {"hi": 150}

        def preprocess_text(self, txt, lang):
            if lang == "ne":
                raise NotImplementedError("no Nepali cleaners")
            return txt

    class Xtts:
        @classmethod
        def init_from_config(cls, config):
            inst = cls()
            inst.tokenizer = _Tokenizer()
            return inst

        def load_checkpoint(self, config, checkpoint_dir, use_deepspeed):
            calls["loaded"] += 1
            assert use_deepspeed is False

        def to(self, device):
            calls["device"].append(device)
            return self

    modules = {
        "TTS": types.ModuleType("TTS"),
        "TTS.tts": types.ModuleType("TTS.tts"),
        "TTS.tts.configs": types.ModuleType("TTS.tts.configs"),
        "TTS.tts.configs.xtts_config": types.ModuleType("TTS.tts.configs.xtts_config"),
        "TTS.tts.models": types.ModuleType("TTS.tts.models"),
        "TTS.tts.models.xtts": types.ModuleType("TTS.tts.models.xtts"),
    }
    modules["TTS.tts.configs.xtts_config"].XttsConfig = XttsConfig
    modules["TTS.tts.models.xtts"].Xtts = Xtts
    modules["TTS.tts.models"].xtts = modules["TTS.tts.models.xtts"]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    import huggingface_hub

    def snapshot(repo, **kw):
        calls["snapshot"].append((repo, kw))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.delenv("OMNIVOICE_XTTS_NEPALI_MODEL_DIR", raising=False)
    monkeypatch.setenv("OMNIVOICE_XTTS_NEPALI_DEVICE", "cpu")
    return calls


def test_xtts_model_loads_pinned_checkpoint_once_on_cpu(fake_coqui):
    import io

    xtts = _load_sidecar("backend/engines/xtts_nepali/main.py")
    first = xtts._load_model(io.BytesIO())
    assert xtts._load_model(io.BytesIO()) is first  # reused, not reloaded
    assert fake_coqui["loaded"] == 1 and fake_coqui["device"] == ["cpu"]
    (repo, kw), = fake_coqui["snapshot"]
    assert repo == "Oshara/xtts-v2-nepali"
    assert kw["revision"] == hf_revisions.revision_for(repo)
    assert kw["allow_patterns"] == ["epoch-20/*"]
    # Nepali goes through the Hindi cleaners instead of raising.
    assert first.tokenizer.preprocess_text("नमस्ते", "ne") == "नमस्ते"
    assert first.tokenizer.char_limits["ne"] == 150


def test_xtts_local_model_dir_skips_the_download(fake_coqui, monkeypatch, tmp_path):
    import io

    monkeypatch.setenv("OMNIVOICE_XTTS_NEPALI_MODEL_DIR", str(tmp_path))
    xtts = _load_sidecar("backend/engines/xtts_nepali/main.py")
    xtts._load_model(io.BytesIO())
    assert fake_coqui["snapshot"] == []


def test_xtts_download_failure_propagates_without_caching_a_model(fake_coqui, monkeypatch):
    import io

    import huggingface_hub

    def offline(*a, **k):
        raise ConnectionError("offline")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", offline)
    xtts = _load_sidecar("backend/engines/xtts_nepali/main.py")
    with pytest.raises(ConnectionError):
        xtts._load_model(io.BytesIO())
    assert xtts._MODEL is None
