"""xtts-nepali: Coqui XTTS v2 with the Oshara Nepali fine-tune, in a sidecar.

Local try-out engine for ``Oshara/xtts-v2-nepali`` (XTTS v2 fine-tuned to add
Nepali, 24 kHz, zero-shot cloning from a reference clip). Not an upstream
candidate as-is: the weights inherit the Coqui Public Model License, which is
non-commercial and so fails bar #2 of docs/engine-acceptance.md.

Runs in its own venv because coqui-tts pins transformers 4.57 while the app
pins >=5.3 (same dedicated-venv shape as dots-tts / moss-tts-v15). The venv is
``$OMNIVOICE_XTTS_NEPALI_DIR/.venv``, defaulting to
``~/.omnivoice/engines/xtts-nepali/.venv``::

    uv venv --python 3.13 ~/.omnivoice/engines/xtts-nepali/.venv
    uv pip install --python ~/.omnivoice/engines/xtts-nepali/.venv/bin/python \\
        --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu torchaudio==2.8.0+cpu
    uv pip install --python ~/.omnivoice/engines/xtts-nepali/.venv/bin/python \\
        coqui-tts==0.27.5 transformers==4.57.6

(Use the cu128 index instead of cpu on an NVIDIA host.) Weights (~1.9 GB)
download from HuggingFace on the first generate. Opt in from the Engines panel
or with ``OMNIVOICE_TTS_BACKEND=xtts-nepali``. Sidecar knobs live in main.py.
"""
from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path

from services.subprocess_backend import SubprocessBackend

logger = logging.getLogger("omnivoice.engines.xtts_nepali")

_DIR_ENV_VAR = "OMNIVOICE_XTTS_NEPALI_DIR"


class XttsNepaliBackend(SubprocessBackend):
    """XTTS v2 (Oshara Nepali fine-tune) in a killable sidecar with its own venv."""

    id = "xtts-nepali"
    display_name = "XTTS v2 Nepali (Oshara fine-tune, CPML non-commercial)"
    _DEFAULT_SAMPLE_RATE = 24_000
    gpu_compat: tuple[str, ...] = ("cuda", "cpu")
    supports_cloning = True
    # Language the UI selects instead of "Auto" while this engine is active.
    default_language = "ne"

    @classmethod
    def engine_dir(cls) -> Path:
        return Path(
            os.environ.get(_DIR_ENV_VAR)
            or Path.home() / ".omnivoice" / "engines" / "xtts-nepali"
        )

    @classmethod
    def venv_python(cls) -> Path:
        venv = cls.engine_dir() / ".venv"
        if sys.platform == "win32":
            return venv / "Scripts" / "python.exe"
        return venv / "bin" / "python"

    @classmethod
    def sidecar_script(cls) -> Path:
        return Path(__file__).resolve().parent / "main.py"

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        py = cls.venv_python()
        if not py.is_file():
            return False, (
                f"XTTS venv not found at {py.parent.parent}. Create it with "
                "coqui-tts==0.27.5 + transformers==4.57.6 (see "
                "backend/engines/xtts_nepali/__init__.py), or point "
                f"{_DIR_ENV_VAR} at a directory containing that .venv."
            )
        return True, "ready (non-commercial CPML weights)"

    @property
    def recv_timeout_s(self) -> float:
        # The sidecar heartbeats every few seconds through the weight download,
        # the model load and each sentence, so this only bounds a silent wedge.
        try:
            v = float(os.environ.get("OMNIVOICE_XTTS_NEPALI_RECV_TIMEOUT_S", "600"))
        except (ValueError, TypeError):
            return 600.0
        if not math.isfinite(v):
            return 600.0
        return max(30.0, v)

    @property
    def sample_rate(self) -> int:
        return self._DEFAULT_SAMPLE_RATE

    @property
    def supported_languages(self) -> list[str]:
        # Nepali plus the 17 base XTTS v2 languages, selected by ``language``.
        return ["multi"]


__all__ = ["XttsNepaliBackend"]
