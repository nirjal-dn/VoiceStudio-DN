"""
Bootstrap required models into the local Hugging Face cache.

Run this after cloning to download models marked `required: true` in
`backend/config/models.yaml`, into the same cache the app reads (it honours
the project `.env`, the in-app Settings env file and `OMNIVOICE_CACHE_DIR`).

Usage:
    source .venv/bin/activate
    python scripts/bootstrap_models.py

Downloads are pinned to the reviewed revisions in `services/hf_revisions.py`
and respect each model's `allow_patterns`. Exits 1 if any download failed.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND = REPO_ROOT / "backend"
MODELS_YAML = BACKEND / "config" / "models.yaml"

log = logging.getLogger("bootstrap_models")


def configure_environment() -> None:
    """Same env sources and cache routing as backend/main.py, applied before
    huggingface_hub is imported (it fixes its cache path at import)."""
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))
    try:
        import dotenv

        dotenv.load_dotenv(REPO_ROOT / ".env", override=False)
    except ImportError:
        pass
    from core.user_env import load_into_environ
    from core.config import apply_cache_dir_env

    load_into_environ()
    apply_cache_dir_env()


def required_models(path: Path = MODELS_YAML) -> list[dict]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return [m for m in data.get("models", []) if m.get("required") and m.get("repo_id")]


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    configure_environment()
    from huggingface_hub import snapshot_download
    from services.hf_revisions import revision_for

    models = required_models()
    if not models:
        log.info("No required models declared in %s", MODELS_YAML)
        return 0
    log.info("Found %d required models (cache: %s)", len(models), os.environ.get("HF_HUB_CACHE", "default"))
    failed = []
    for spec in models:
        repo_id = spec["repo_id"]
        rev = revision_for(repo_id)
        log.info("Installing %s (revision=%s)...", repo_id, rev)
        try:
            snapshot_download(repo_id, revision=rev, allow_patterns=spec.get("allow_patterns"))
            log.info("Installed %s", repo_id)
        except Exception:  # noqa: BLE001 — report every model, then fail the run
            log.exception("Failed to install %s", repo_id)
            failed.append(repo_id)
    if failed:
        log.error("Failed: %s. Retry, or install from Model Catalogue in the app.", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
