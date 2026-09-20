"""xtts-en: base Coqui XTTS v2 in the xtts-nepali sidecar, for English spans.

The smallest English adapter that pairs with ``xtts-nepali``: same sidecar
script, same venv, same wire protocol — only the checkpoint differs
(``coqui/XTTS-v2`` instead of ``Oshara/xtts-v2-nepali``). That is deliberate.
The Nepali engine IS a fine-tune of this checkpoint, so when the code-switch
service (:mod:`services.code_switch_tts`) hands both engines the same reference
clip, the two halves of a mixed Nepali/English sentence stay recognisably the
same speaker in a way two unrelated models never manage.

It is a normal engine as well as a code-switch child: select it in Settings, or
with ``OMNIVOICE_TTS_BACKEND=xtts-en``, for English-only cloning.

**Install:** it reuses the ``xtts-nepali`` venv, so if that engine works this
one does too — see docs/engines/xtts-en.md. Point ``OMNIVOICE_XTTS_EN_DIR`` at
a different directory to give it a venv of its own. Weights (~1.9 GB) download
from Hugging Face on the first generate. The Coqui Public Model License is
non-commercial, same as the Nepali fine-tune.

Sidecar knobs are the ``OMNIVOICE_XTTS_NEPALI_*`` set with an ``EN`` infix:
``OMNIVOICE_XTTS_EN_DEVICE``, ``_MODEL_DIR``, ``_SPEAKER``, ``_TEMPERATURE``,
``_REPETITION_PENALTY``, ``_TOP_K``, ``_TOP_P``, ``_RECV_TIMEOUT_S``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from engines.xtts_nepali import XttsNepaliBackend
from services.subprocess_backend import recv_timeout_from_env

logger = logging.getLogger("omnivoice.engines.xtts_en")

_DIR_ENV_VAR = "OMNIVOICE_XTTS_EN_DIR"


class XttsEnglishBackend(XttsNepaliBackend):
    """Base XTTS v2 (English + 16 other languages) in its own sidecar process."""

    id = "xtts-en"
    display_name = "XTTS v2 (Coqui base, CPML non-commercial)"
    default_language = "en"

    @classmethod
    def engine_dir(cls) -> Path:
        # Defaults to the xtts-nepali venv: coqui-tts is already installed
        # there and the two engines pin the same dependencies, so a user who
        # set up the Nepali engine gets this one for free.
        override = os.environ.get(_DIR_ENV_VAR)
        return Path(override) if override else XttsNepaliBackend.engine_dir()

    def sidecar_env(self) -> dict:
        # Per-process, NOT inherited: both engines are resident at once during
        # a code-switched render, so the variant cannot live in the parent env.
        return {"OMNIVOICE_XTTS_VARIANT": "en"}

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        py = cls.venv_python()
        if not py.is_file():
            return False, (
                f"XTTS venv not found at {py.parent.parent}. Create it with "
                "coqui-tts==0.27.5 + transformers==4.57.6 (see "
                "docs/engines/xtts-en.md) — the xtts-nepali venv works as-is — "
                f"or point {_DIR_ENV_VAR} at a directory containing that .venv."
            )
        return True, "ready (non-commercial CPML weights)"

    @property
    def recv_timeout_s(self) -> float:
        return recv_timeout_from_env("OMNIVOICE_XTTS_EN_RECV_TIMEOUT_S", 600.0)


__all__ = ["XttsEnglishBackend"]
