"""
Bootstrap required models into the local Hugging Face cache.

Run this after cloning to download models marked `required: true` in
`config/models.yaml` so ASR/TTS/LLM paths work out-of-the-box.

Usage:
    source .venv/bin/activate
    python scripts/bootstrap_models.py

It respects existing HF cache env vars and uses `services.hf_revisions.revision_for`
so the project pins the correct reviewed revision where available.
"""

from __future__ import annotations

import os
import sys
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bootstrap_models")

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_YAML = REPO_ROOT / "backend" / "config" / "models.yaml"

try:
    from services.hf_revisions import revision_for
except Exception:
    # If running outside backend package path, add backend to sys.path
    _backend = str(REPO_ROOT / "backend")
    if _backend not in sys.path:
        sys.path.insert(0, _backend)
    from services.hf_revisions import revision_for

try:
    from huggingface_hub import snapshot_download
except Exception as e:
    log.error("Please install huggingface_hub into your environment: %s", e)
    raise


def load_models_yaml(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("models", []) if data else []


def ensure_cache_env():
    # honor OMNIVOICE_CACHE_DIR if set; otherwise use HF defaults
    cache = os.environ.get("OMNIVOICE_CACHE_DIR")
    if cache:
        os.environ.setdefault("HF_HOME", cache)
        os.environ.setdefault("HF_HUB_CACHE", cache)
        os.environ.setdefault("TORCH_HOME", cache)


def main():
    ensure_cache_env()
    models = load_models_yaml(MODELS_YAML)
    required = [m for m in models if m.get("required")]
    if not required:
        log.info("No required models declared in %s", MODELS_YAML)
        return

    log.info("Found %d required models", len(required))
    for spec in required:
        repo_id = spec.get("repo_id")
        if not repo_id:
            continue
        try:
            rev = revision_for(repo_id)
        except Exception:
            rev = None
        log.info("Installing %s (revision=%s)...", repo_id, rev or "main")
        try:
            snapshot_download(repo_id, revision=rev or None)
            log.info("Installed %s", repo_id)
        except Exception as e:
            log.exception("Failed to install %s: %s", repo_id, e)
            log.warning("You can install it later via Model Catalogue in the app or rerun this script.")


if __name__ == "__main__":
    main()
