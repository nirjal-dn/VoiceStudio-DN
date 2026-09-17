"""scripts/bootstrap_models.py installs required models into the cache the app
reads, pinned and pattern-filtered, and fails loudly."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "bootstrap_models.py"


@pytest.fixture
def bootstrap(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("bootstrap_models_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr("core.user_env.load_into_environ", lambda *a, **k: False)
    for var in ("HF_HOME", "HF_HUB_CACHE", "TORCH_HOME"):
        monkeypatch.delenv(var, raising=False)
    return module


def _fake_hub(monkeypatch, fail=()):
    calls = []

    def snapshot_download(repo_id, **kw):
        import os

        calls.append((repo_id, kw, os.environ.get("HF_HUB_CACHE")))
        if repo_id in fail:
            raise ConnectionError("offline")
        return "/tmp/snapshot"

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return calls


def test_downloads_required_models_pinned_into_the_configured_cache(bootstrap, monkeypatch, tmp_path):
    from services import hf_revisions

    monkeypatch.setenv("OMNIVOICE_CACHE_DIR", str(tmp_path / "models"))
    calls = _fake_hub(monkeypatch)
    assert bootstrap.main() == 0
    required = [m["repo_id"] for m in bootstrap.required_models()]
    assert [c[0] for c in calls] == required and required
    for repo_id, kw, cache in calls:
        assert kw["revision"] == hf_revisions.revision_for(repo_id)
        assert cache == str(tmp_path / "models")  # env set before the download


def test_required_models_are_read_with_their_allow_patterns(bootstrap, monkeypatch, tmp_path):
    catalog = tmp_path / "models.yaml"
    catalog.write_text(
        "models:\n"
        "  - repo_id: Oshara/xtts-v2-nepali\n    required: true\n    allow_patterns: ['epoch-20/*']\n"
        "  - repo_id: ai4bharat/indic-parler-tts\n",
        encoding="utf-8",
    )
    models = bootstrap.required_models(catalog)
    assert models == [{"repo_id": "Oshara/xtts-v2-nepali", "required": True, "allow_patterns": ["epoch-20/*"]}]

    monkeypatch.setattr(bootstrap, "required_models", lambda path=None: models)
    calls = _fake_hub(monkeypatch)
    assert bootstrap.main() == 0
    assert calls[0][1]["allow_patterns"] == ["epoch-20/*"]


def test_a_failed_download_exits_nonzero_after_trying_the_rest(bootstrap, monkeypatch):
    monkeypatch.setattr(bootstrap, "required_models", lambda path=None: [
        {"repo_id": "k2-fsa/OmniVoice", "required": True},
        {"repo_id": "Oshara/xtts-v2-nepali", "required": True},
    ])
    calls = _fake_hub(monkeypatch, fail={"k2-fsa/OmniVoice"})
    assert bootstrap.main() == 1
    assert [c[0] for c in calls] == ["k2-fsa/OmniVoice", "Oshara/xtts-v2-nepali"]
