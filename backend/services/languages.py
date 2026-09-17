"""Language registry: one place for what the app needs to know per language.

Adding a language means one ``Language`` entry here instead of scattered
``if language == "…"`` branches. Consumers resolve whatever the caller sent —
a display name from ``frontend/src/languages.json`` ("Nepali"), an ISO 639-1 /
639-3 code ("ne", "npi"), or a BCP-47 tag ("ne-NP") — through :func:`resolve`.

Only languages with behaviour beyond the defaults are registered; an
unregistered language simply gets the generic pipeline.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class Language:
    code: str                       # ISO 639-1 (or 639-3 when there is none)
    name: str                       # English display name, as in languages.json
    native_name: str
    aliases: tuple[str, ...] = ()   # other spellings/codes callers send
    #: Unicode block the language is written in, for script checks.
    script: Optional[str] = None
    script_range: Optional[tuple[int, int]] = None
    #: FLORES-200 code for NLLB translation.
    flores: Optional[str] = None
    #: Text → speakable text before TTS (numbers, dates, …). Must be idempotent.
    tts_normalizer: Optional[Callable[[str], str]] = field(default=None, compare=False)


def _nepali_normalizer(text: str) -> str:
    from services.nepali_text import normalize_numbers

    return normalize_numbers(text)


LANGUAGES: tuple[Language, ...] = (
    Language(
        code="ne",
        name="Nepali",
        native_name="नेपाली",
        aliases=("npi", "nep", "ne-np", "nepali (nepal)", "नेपाली"),
        script="Devanagari",
        script_range=(0x0900, 0x097F),
        flores="npi_Deva",
        tts_normalizer=_nepali_normalizer,
    ),
)

_BY_KEY: dict[str, Language] = {}
for _lang in LANGUAGES:
    for _key in (_lang.code, _lang.name, _lang.native_name, *_lang.aliases):
        _BY_KEY[_key.strip().lower()] = _lang


def resolve(language: object) -> Optional[Language]:
    """The registered language for a display name / code / BCP-47 tag, or None."""
    if not language:
        return None
    key = str(language).strip().lower().replace("_", "-")
    if not key or key == "auto":
        return None
    return _BY_KEY.get(key) or _BY_KEY.get(key.split("-")[0])


def iso_code(language: object) -> Optional[str]:
    """Best-effort ISO code for any language value: registered languages map to
    their code; any of the UI's 646 display names map through OmniVoice's
    name table ("Hindi" → "hi"); other 2–3 letter codes / tags pass through."""
    lang = resolve(language)
    if lang:
        return lang.code
    if not language:
        return None
    raw = str(language).strip().lower()
    if not raw or raw == "auto":
        return None
    try:
        from omnivoice.utils.lang_map import LANG_NAME_TO_ID
    except Exception:  # noqa: BLE001 — name table unavailable: codes still work
        LANG_NAME_TO_ID = {}
    named = LANG_NAME_TO_ID.get(raw)
    if named:
        return resolve(named).code if resolve(named) else named
    key = raw.replace("_", "-").split("-")[0]
    return key if key.isalpha() and 2 <= len(key) <= 3 else None


def script_ratio(text: str, language: object) -> Optional[float]:
    """Fraction of letters in ``text`` written in the language's script, or
    None when the language has no registered script or the text no letters."""
    lang = resolve(language)
    if not lang or not lang.script_range:
        return None
    lo, hi = lang.script_range
    # Vowel signs and viramas are marks, not letters, but they carry most of
    # an abugida's writing — count them so Devanagari is not under-weighted.
    letters = [c for c in text if c.isalpha() or unicodedata.category(c) in ("Mn", "Mc")]
    if not letters:
        return None
    return sum(lo <= ord(c) <= hi for c in letters) / len(letters)


def text_matches_language(text: str, language: object, threshold: float = 0.25) -> bool:
    """False only when the language has a known script and the text is written
    almost entirely outside it (a Latin or romanized transcript for a Nepali
    voice). Code-mixed text with some native script still matches."""
    ratio = script_ratio(text, language)
    return ratio is None or ratio >= threshold


def normalize_for_speech(text: str, language: object) -> str:
    """Apply the language's TTS normalizer, if it has one."""
    lang = resolve(language)
    return lang.tts_normalizer(text) if lang and lang.tts_normalizer else text
