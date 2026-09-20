"""Dual-TTS orchestration for mixed Nepali/English text.

One sentence, two engines: Devanagari spans go to the configured Nepali engine
(``xtts-nepali`` by default), Latin spans to the configured English engine
(``xtts-en``), both cloning the SAME reference clip, and the spans are stitched
back into one stream. The Nepali fine-tune reads Latin script badly and base
XTTS reads Devanagari not at all, so routing is the whole fix — no model
change, no retraining.

**This is not a** :class:`~services.tts_backend.TTSBackend`. Child engines are
resolved through :func:`~services.tts_backend.get_engine_instance_for`, which
is the cached-access-by-id API; the machine's globally active backend
(``active_backend_id`` / ``get_active_tts_backend``) is never read and never
changed, so a code-switched render cannot leak into the next single-language
one.

Sequential by design (see :attr:`CodeSwitchConfig.allow_parallel`): two
resident models already cost VRAM, and running them concurrently on one device
buys latency at the price of the OOM class this app spends most of its memory
discipline avoiding.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack
from dataclasses import dataclass
from typing import List, Optional, Tuple

from services.language_segmenter import (
    ENGLISH, NEPALI, LanguageSegment, is_mixed, segment_languages,
)

logger = logging.getLogger("omnivoice.code_switch")

#: Engine ids the feature defaults to. Both are XTTS v2 — the Nepali fine-tune
#: and its base model — which is what makes one reference clip sound like one
#: speaker across the boundary.
DEFAULT_NEPALI_ENGINE = "xtts-nepali"
DEFAULT_ENGLISH_ENGINE = "xtts-en"

_OFF = {"0", "false", "no", "off", ""}


class CodeSwitchError(ValueError):
    """A code-switched render cannot run as configured.

    Subclasses ``ValueError`` so the ``/generate`` route's existing
    ValueError→400 mapping turns it into an actionable client error rather
    than an opaque 500 — the same contract as
    :class:`services.tts_backend.TTSInputError`.
    """


def _flag(key: str, env: str, default: bool) -> bool:
    from core import prefs

    raw = prefs.resolve(key, env=env, default="1" if default else "0")
    return str(raw).strip().lower() not in _OFF


def _text_pref(key: str, env: str, default: str) -> str:
    from core import prefs

    return str(prefs.resolve(key, env=env, default=default) or default).strip()


def _int_pref(key: str, env: str, default: int) -> int:
    from core import prefs

    try:
        return int(str(prefs.resolve(key, env=env, default=default)).strip())
    except (TypeError, ValueError):
        return default


def code_switch_enabled() -> bool:
    """Whether mixed-language routing may run at all (default on)."""
    return _flag("code_switch_enabled", "OMNIVOICE_CODE_SWITCH_ENABLED", True)


def should_code_switch(text: str) -> bool:
    """``auto`` mode's decision: only genuinely mixed text is code-switched.

    Pure Nepali and pure English both keep the unchanged single-engine path.
    """
    return code_switch_enabled() and is_mixed(text)


@dataclass
class CodeSwitchConfig:
    """Resolved settings for one code-switched render.

    Env var wins over the prefs store wins over these defaults — the same
    precedence :func:`core.prefs.resolve` gives every other setting, so there
    is no second settings mechanism here.
    """

    nepali_engine: str
    english_engine: str
    crossfade_ms: int = 30
    normalize_gain: bool = True
    normalize_sample_rate: bool = True
    trim_boundary_silence: bool = True
    allow_parallel: bool = False

    @classmethod
    def from_prefs(cls) -> "CodeSwitchConfig":
        return cls(
            nepali_engine=_text_pref(
                "code_switch_ne_backend", "OMNIVOICE_CODE_SWITCH_NE_BACKEND",
                DEFAULT_NEPALI_ENGINE),
            english_engine=_text_pref(
                "code_switch_en_backend", "OMNIVOICE_CODE_SWITCH_EN_BACKEND",
                DEFAULT_ENGLISH_ENGINE),
            crossfade_ms=_int_pref(
                "code_switch_crossfade_ms", "OMNIVOICE_CODE_SWITCH_CROSSFADE_MS", 30),
            normalize_gain=_flag(
                "code_switch_normalize_gain", "OMNIVOICE_CODE_SWITCH_NORMALIZE_GAIN", True),
            trim_boundary_silence=_flag(
                "code_switch_trim_silence", "OMNIVOICE_CODE_SWITCH_TRIM_SILENCE", True),
            allow_parallel=_flag(
                "code_switch_parallel", "OMNIVOICE_CODE_SWITCH_PARALLEL", False),
        )


class CodeSwitchTTSService:
    """Route each language span to its own engine and stitch the result."""

    #: Human-readable names used in the errors a user has to act on.
    _LABELS = {NEPALI: "Nepali", ENGLISH: "English"}

    def __init__(self, config: Optional[CodeSwitchConfig] = None) -> None:
        self.config = config or CodeSwitchConfig.from_prefs()
        self._engines: dict[str, object] = {}
        #: Rate the stitched take comes back at. Known only once the child
        #: engines are resolved, so callers read it after validate()/generate().
        self.target_sample_rate: Optional[int] = None

    # ── child backends ────────────────────────────────────────────────────

    def engine_id_for(self, language: str) -> str:
        return (self.config.nepali_engine if language == NEPALI
                else self.config.english_engine)

    def _resolve(self, language: str):
        """Cached child engine for *language*, validated before it is used.

        Everything that can make a render fail late is checked here: an id the
        registry does not know, an engine whose environment is not installed,
        an engine that cannot clone (both spans must be the same speaker), one
        that declares it cannot speak the language, and one whose sample rate
        is unusable.
        """
        if language in self._engines:
            return self._engines[language]

        from services.tts_backend import (
            canonical_engine_id, get_backend_class, get_engine_instance_for,
        )

        label = self._LABELS.get(language, language)
        engine_id = canonical_engine_id(self.engine_id_for(language))
        if not engine_id:
            raise CodeSwitchError(
                f"Mixed-language synthesis requires {label} TTS engine, but no "
                f"{label} backend is configured. Set it in Settings or with "
                f"OMNIVOICE_CODE_SWITCH_{'NE' if language == NEPALI else 'EN'}_BACKEND."
            )
        try:
            backend_cls = get_backend_class(engine_id)
        except ValueError as exc:
            raise CodeSwitchError(
                f"Mixed-language synthesis requires {label} TTS engine, but the "
                f"configured {label} backend {engine_id!r} is not a known engine "
                f"id. See GET /engines/tts for the list."
            ) from exc

        try:
            available, reason = backend_cls.is_available()
        except Exception as exc:  # noqa: BLE001 — a probe must not 500 the route
            available, reason = False, f"{type(exc).__name__}: {exc}"
        if not available:
            raise CodeSwitchError(
                f"Mixed-language synthesis requires {label} TTS engine, but the "
                f"configured {label} backend ({engine_id}) is unavailable: {reason}"
            )
        if not getattr(backend_cls, "supports_cloning", True):
            raise CodeSwitchError(
                f"The configured {label} backend ({engine_id}) cannot clone a "
                "reference voice, so the two halves of a mixed-language take "
                "would be different speakers. Choose a cloning-capable engine."
            )

        instance = get_engine_instance_for(engine_id)

        languages = list(getattr(instance, "supported_languages", ["multi"]) or [])
        if languages and "multi" not in languages:
            codes = {str(code).split("-")[0].lower() for code in languages}
            if language not in codes:
                raise CodeSwitchError(
                    f"The configured {label} backend ({engine_id}) does not list "
                    f"{language!r} among its languages ({', '.join(languages)})."
                )

        try:
            sample_rate = int(instance.sample_rate)
        except Exception as exc:  # noqa: BLE001
            raise CodeSwitchError(
                f"The configured {label} backend ({engine_id}) does not report a "
                "usable sample rate."
            ) from exc
        if sample_rate <= 0:
            raise CodeSwitchError(
                f"The configured {label} backend ({engine_id}) reports an unusable "
                f"sample rate ({sample_rate})."
            )

        self._engines[language] = instance
        return instance

    def validate(self, languages=(NEPALI, ENGLISH)) -> None:
        """Resolve and check every child backend before any audio is rendered.

        Called by the route ahead of dispatch so a misconfiguration is a fast,
        actionable 400 instead of half a take and a 500.
        """
        rates = []
        for language in languages:
            rates.append(int(self._resolve(language).sample_rate))
        if rates:
            # Highest rate wins: meeting a 22.05 kHz engine by downsampling the
            # 24 kHz one would throw away quality on every Nepali span.
            self.target_sample_rate = max(rates)

    # ── rendering ─────────────────────────────────────────────────────────

    def _plan(self, text: str, language_mode: str) -> List[LanguageSegment]:
        """Spans to render, honouring an explicit single-language override."""
        mode = (language_mode or "auto").strip().lower()
        if mode in (NEPALI, "nepali"):
            return [LanguageSegment(text, NEPALI, 0, len(text), 1.0, "forced")]
        if mode in (ENGLISH, "english"):
            return [LanguageSegment(text, ENGLISH, 0, len(text), 1.0, "forced")]
        return segment_languages(text)

    @staticmethod
    def _normalized(segment: LanguageSegment) -> str:
        """Run one span through the shared TTS text normalizer, in ITS language.

        Same gate every other pipeline uses (``normalize_for_tts`` honours the
        user's normalization pref), applied per span so a number inside a
        Nepali clause is spoken in Nepali and one inside an English clause in
        English. Falls back to the raw span if normalization raises — dropping
        the user's text is never the better outcome.
        """
        from services.text_normalization import normalize_for_tts

        try:
            return normalize_for_tts(segment.text, segment.language) or segment.text
        except Exception:  # noqa: BLE001
            logger.debug("normalization skipped for a %s span", segment.language,
                         exc_info=True)
            return segment.text

    def generate(self, text: str, *, ref_audio: Optional[str] = None,
                 ref_text: Optional[str] = None, speed: float = 1.0,
                 language_mode: str = "auto") -> Tuple["object", int]:
        """Render *text* across the child engines. Returns ``(audio, sample_rate)``.

        The same ``ref_audio`` and ``ref_text`` go to both engines — each keeps
        its own model-specific conditioning cache; no latent is shared between
        two different models, which would not mean anything.
        """
        from services.code_switch_audio import stitch_segments
        from services.tts_backend import engine_in_use

        segments = [s for s in self._plan(text, language_mode) if s.text.strip()]
        if not segments:
            from services.tts_backend import TTSInputError

            raise TTSInputError("Text has nothing speakable.")

        languages = [s.language for s in segments]
        self.validate(dict.fromkeys(languages))

        logger.info(
            "code-switch detected segments=%d languages=%s ne_backend=%s en_backend=%s",
            len(segments), ",".join(languages),
            self.engine_id_for(NEPALI), self.engine_id_for(ENGLISH),
        )

        def _render(segment: LanguageSegment):
            engine = self._resolve(segment.language)
            # Held for the length of THIS span's synthesis: the single-active-
            # engine sweep must not unload an engine mid-render.
            with engine_in_use(engine):
                audio = engine.generate(
                    self._normalized(segment),
                    ref_audio=ref_audio, ref_text=ref_text,
                    language=segment.language, speed=speed, duration=None,
                )
            return audio, int(engine.sample_rate)

        # Both engines are held for the whole request, so the one that already
        # rendered stays resident while the other works — no reload per span.
        with ExitStack() as stack:
            for language in dict.fromkeys(languages):
                stack.enter_context(engine_in_use(self._resolve(language)))
            if self.config.allow_parallel and len(segments) > 1:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=2) as pool:
                    rendered = list(pool.map(_render, segments))
            else:
                rendered = [_render(segment) for segment in segments]

        audio, sample_rate = stitch_segments(
            rendered,
            target_sample_rate=self.target_sample_rate,
            crossfade_ms=self.config.crossfade_ms,
            normalize_gain=self.config.normalize_gain,
            normalize_sample_rate=self.config.normalize_sample_rate,
            trim_silence=self.config.trim_boundary_silence,
            texts=[s.text for s in segments],
        )
        self.target_sample_rate = sample_rate
        return audio, sample_rate

    @property
    def applies_own_mastering(self) -> bool:
        """True only when EVERY child engine masters its own output.

        The stitched take then skips the shared mastering chain, exactly as a
        single studio-grade engine's output does.
        """
        return bool(self._engines) and all(
            getattr(engine, "applies_own_mastering", False)
            for engine in self._engines.values()
        )


__all__ = [
    "CodeSwitchConfig", "CodeSwitchError", "CodeSwitchTTSService",
    "DEFAULT_ENGLISH_ENGINE", "DEFAULT_NEPALI_ENGINE", "code_switch_enabled",
    "should_code_switch",
]
