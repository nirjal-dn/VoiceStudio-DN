"""indic-parler-tts: AI4Bharat Indic Parler-TTS in a sidecar with its own venv.

Runs ai4bharat/indic-parler-tts (Apache-2.0; 21 languages including Nepali).
The voice is chosen by a text description rather than a reference clip, so the
engine takes the request's ``instruct`` as that description, defaulting to the
Nepali speaker "Amrita" (main.py). parler-tts pins transformers 4.46.1 while
the app pins >=5.3, hence the separate venv:
``$OMNIVOICE_INDIC_PARLER_DIR/.venv``, default ``~/.omnivoice/engines/indic-parler/.venv``::

    uv venv --python 3.11 ~/.omnivoice/engines/indic-parler/.venv
    uv pip install --python ~/.omnivoice/engines/indic-parler/.venv/bin/python \\
        --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu torchaudio==2.8.0+cpu
    uv pip install --python ~/.omnivoice/engines/indic-parler/.venv/bin/python \\
        git+https://github.com/huggingface/parler-tts.git

The model is gated on Hugging Face (accept its terms, then set a token); the
~3.8 GB download happens on first use.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

from services.subprocess_backend import SubprocessBackend

_DIR_ENV_VAR = "OMNIVOICE_INDIC_PARLER_DIR"


class IndicParlerBackend(SubprocessBackend):
    """Indic Parler-TTS in a killable sidecar with its own venv."""

    id = "indic-parler-tts"
    display_name = "Indic Parler-TTS (21 langs incl. Nepali, voice by description)"
    _DEFAULT_SAMPLE_RATE = 44_100
    gpu_compat: tuple[str, ...] = ("cuda", "cpu")
    supports_cloning = False

    @classmethod
    def venv_python(cls) -> Path:
        root = Path(os.environ.get(_DIR_ENV_VAR) or Path.home() / ".omnivoice" / "engines" / "indic-parler")
        return root / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    @classmethod
    def sidecar_script(cls) -> Path:
        return Path(__file__).resolve().parent / "main.py"

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        py = cls.venv_python()
        if not py.is_file():
            return False, (
                f"Indic Parler-TTS venv not found at {py.parent.parent}. Create it with "
                "parler-tts (see backend/engines/indic_parler/__init__.py), or point "
                f"{_DIR_ENV_VAR} at a directory containing that .venv."
            )
        return True, "ready"

    @property
    def recv_timeout_s(self) -> float:
        # The sidecar heartbeats through download, load and each sentence.
        try:
            v = float(os.environ.get("OMNIVOICE_INDIC_PARLER_RECV_TIMEOUT_S", "600"))
        except (TypeError, ValueError):
            return 600.0
        return max(30.0, v) if math.isfinite(v) else 600.0

    @property
    def sample_rate(self) -> int:
        return self._DEFAULT_SAMPLE_RATE

    @property
    def supported_languages(self) -> list[str]:
        return ["multi"]


__all__ = ["IndicParlerBackend"]
