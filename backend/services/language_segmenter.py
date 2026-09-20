"""Deterministic Nepali / English span detection for mixed-language TTS.

The Nepali TTS engine (``xtts-nepali``) reads Devanagari well and Latin script
badly; a base-XTTS English engine is the reverse. Splitting one sentence into
per-language spans lets each engine render the part it is good at
(:mod:`services.code_switch_tts` does the routing).

Deterministic on purpose — no model, no network, no per-call state. The three
signals, in priority order:

A. **Devanagari script** — decisive for Nepali.
B. **An English / technical vocabulary** (:data:`ENGLISH_VOCABULARY`) — a
   signal, not a whitelist: unknown Latin words still classify as English.
C. **A Romanized-Nepali lexicon** (:data:`ROMANIZED_NEPALI`) — a first-version
   deterministic signal only (see the ceiling note on that constant).

then a contextual pass (D) that gives numbers, punctuation, URLs, emails and
technical identifiers the language of the span they sit in, and a merge pass
(E) that joins neighbouring tokens of the same language so one language run is
one TTS call rather than one call per word.

Both vocabularies are plain module-level frozensets, meant to be edited.

The segmenter never rewrites, drops or reorders text: for any input,
``"".join(s.text for s in segment_languages(text)) == text`` whenever the text
contains anything speakable (guarded by a test).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

#: Language tags a segment can carry.
NEPALI = "ne"
ENGLISH = "en"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class LanguageSegment:
    """One contiguous run of ``text`` in a single language.

    ``start``/``end`` index into the ORIGINAL string, and consecutive segments
    are adjacent (``segments[i].end == segments[i + 1].start``), so the input
    can always be reassembled from the output.
    """

    text: str
    language: str
    start: int
    end: int
    confidence: float
    reason: str


# ── Vocabularies (editable) ────────────────────────────────────────────────

#: English / technical terms that keep their English pronunciation inside a
#: Nepali sentence. One signal among several — an unlisted Latin-script word
#: still classifies as English via ``latin_default``. Extend freely; entries
#: are matched lowercase, after punctuation is stripped.
ENGLISH_VOCABULARY: frozenset[str] = frozenset({
    # Spec list
    "project", "meeting", "server", "database", "model", "training", "gpu",
    "cpu", "api", "deployment", "voice", "studio", "audio", "video", "clone",
    "design", "transcript", "system", "complete", "working", "successfully",
    # Everyday office / tech vocabulary that shows up in the same sentences
    "email", "file", "files", "folder", "laptop", "mobile", "internet",
    "online", "offline", "software", "hardware", "update", "upload",
    "download", "install", "backup", "restart", "migration", "deploy",
    "release", "report", "presentation", "schedule", "manager", "team",
    "office", "client", "customer", "budget", "data", "network", "login",
    "password", "account", "dashboard", "feature", "bug", "test", "testing",
    "review", "approve", "cancel", "confirm", "download", "link", "message",
    "call", "camera", "screen", "battery", "charger", "browser", "app",
})

#: Romanized Nepali written in Latin script. First-version signal only, as the
#: spec asks — the list is short, matching is exact-token, and there is no
#: attempt at morphology.
#:
#: ponytail: a Romanized-Nepali span is routed to the Nepali engine, which
#: reads Devanagari — it will approximate Latin input rather than pronounce it
#: properly. Upgrade path when this matters: transliterate Latin→Devanagari
#: before handing the span to the Nepali engine.
ROMANIZED_NEPALI: frozenset[str] = frozenset({
    # Spec list
    "ma", "malai", "aja", "bholi", "garchu", "garne", "janchu", "cha", "chha",
    "ra", "tara",
    # Common closed-class words, same shape (pronouns, postpositions, copulas)
    "mero", "hamro", "timro", "tapai", "tapailai", "hami", "hamilai", "timi",
    "timilai", "usle", "uslai", "yo", "tyo", "yesto", "tyesto", "kaha", "kina",
    "kasari", "kati", "hijo", "abha", "ahile", "pachi", "aghi", "sanga",
    "bata", "lagi", "garera", "gareko", "garyo", "bhayo", "bhaneko", "chhu",
    "chhau", "chhan", "chan", "thiyo", "hunchha", "huncha", "hola", "pani",
    "matra", "dherai", "ramro", "naramro", "thulo", "sano", "gari", "gaye",
})

# ── Token shapes ───────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"\S+")
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ꣠-ꣿ]")
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_LATIN_RE = re.compile(r"[A-Za-z]")
_URL_RE = re.compile(r"^(?:[a-z][a-z0-9+.\-]*://|www\.)\S+$", re.IGNORECASE)
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
#: file.py, snake_case, some/path, v1.2.3 — carried with whatever span they
#: sit in rather than pronounced as their own language.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9]+(?:[._\-/\\][A-Za-z0-9]+)+$")

#: Stripped from a token's edges before it is looked up. Devanagari danda and
#: the usual Latin/typographic punctuation; combining marks are letters here
#: and are deliberately NOT stripped.
_EDGE_PUNCT = "".join((
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
    "।॥",          # danda, double danda
    "‘’“”–—…«»",
))


def _core(token: str) -> str:
    """The lookup form of a token: edge punctuation removed, lowercased."""
    return token.strip(_EDGE_PUNCT).lower()


def _classify_token(token: str) -> tuple[str, float, str]:
    """(language, confidence, reason) for one whitespace-delimited token."""
    core = _core(token)

    # A. Devanagari is decisive, even mixed into a Latin token.
    if _DEVANAGARI_RE.search(token):
        return NEPALI, 0.99, "devanagari"

    if not core:
        return UNKNOWN, 0.0, "punctuation"

    # D (inputs). Things that must ride along with a neighbouring span rather
    # than form one: web/technical shapes and text with no letters at all.
    if _URL_RE.match(core):
        return UNKNOWN, 0.0, "url"
    if _EMAIL_RE.match(core):
        return UNKNOWN, 0.0, "email"
    if not _LETTER_RE.search(core):
        return UNKNOWN, 0.0, "numeric"
    if _IDENTIFIER_RE.match(core):
        return UNKNOWN, 0.0, "identifier"

    # B. English / technical vocabulary.
    if core in ENGLISH_VOCABULARY:
        return ENGLISH, 0.9, "english_vocabulary"

    # C. Romanized Nepali.
    if core in ROMANIZED_NEPALI:
        return NEPALI, 0.8, "romanized_nepali"

    # Any other Latin-script word reads as English. Never the other way round:
    # the Nepali engine cannot pronounce Latin script at all, so an unknown
    # Latin word is strictly safer on the English engine.
    if _LATIN_RE.search(core):
        return ENGLISH, 0.5, "latin_default"

    return UNKNOWN, 0.0, "unclassified"


def _resolve_unknowns(tokens: list[dict]) -> None:
    """D. Give every context-dependent token the language of its span.

    Previous token first (a number keeps the language of the clause that
    introduced it), then the next one, then English — which is where digits,
    URLs and identifiers belong when nothing else is known.
    """
    for index, tok in enumerate(tokens):
        if tok["language"] != UNKNOWN:
            continue
        inherited = None
        for prev in range(index - 1, -1, -1):
            if tokens[prev]["language"] != UNKNOWN:
                inherited = tokens[prev]["language"]
                break
        if inherited is None:
            for nxt in range(index + 1, len(tokens)):
                if tokens[nxt]["language"] != UNKNOWN:
                    inherited = tokens[nxt]["language"]
                    break
        if inherited is None:
            tok["language"], tok["reason"] = ENGLISH, f"{tok['reason']}+default"
        else:
            tok["language"], tok["reason"] = inherited, f"{tok['reason']}+inherited"


def _merge_reasons(reasons: Iterable[str]) -> str:
    unique = list(dict.fromkeys(reasons))
    return "+".join(unique[:3]) if len(unique) <= 3 else "+".join(unique[:3] + ["..."])


def segment_languages(text: str) -> List[LanguageSegment]:
    """Split *text* into contiguous single-language segments.

    Returns ``[]`` when there is nothing speakable (empty / whitespace-only);
    otherwise the segments tile the whole string, whitespace included, so
    nothing the user typed is dropped or reordered.
    """
    if not text or not text.strip():
        return []

    tokens: list[dict] = []
    for match in _TOKEN_RE.finditer(text):
        language, confidence, reason = _classify_token(match.group())
        tokens.append({
            "start": match.start(), "end": match.end(), "language": language,
            "confidence": confidence, "reason": reason,
        })
    if not tokens:
        return []

    _resolve_unknowns(tokens)

    # E. One segment per language run — never one per word.
    runs: list[list[dict]] = [[tokens[0]]]
    for tok in tokens[1:]:
        if tok["language"] == runs[-1][-1]["language"]:
            runs[-1].append(tok)
        else:
            runs.append([tok])

    segments: List[LanguageSegment] = []
    for index, run in enumerate(runs):
        # The first segment absorbs any leading whitespace and every segment
        # runs up to the next one's first token, so concatenating the segment
        # texts reproduces the input exactly.
        start = 0 if index == 0 else run[0]["start"]
        end = len(text) if index == len(runs) - 1 else runs[index + 1][0]["start"]
        segments.append(LanguageSegment(
            text=text[start:end],
            language=run[0]["language"],
            start=start,
            end=end,
            confidence=min(t["confidence"] for t in run),
            reason=_merge_reasons(t["reason"] for t in run),
        ))
    return segments


def language_profile(text: str) -> dict:
    """Segment counts per language — the cheap summary the router logs."""
    segments = segment_languages(text)
    counts: dict[str, int] = {}
    for segment in segments:
        counts[segment.language] = counts.get(segment.language, 0) + 1
    return {
        "segments": len(segments),
        "languages": [s.language for s in segments],
        "counts": counts,
    }


def is_mixed(text: str) -> bool:
    """True when *text* holds at least one Nepali AND one English segment."""
    languages = {s.language for s in segment_languages(text)}
    return NEPALI in languages and ENGLISH in languages


__all__ = [
    "ENGLISH", "ENGLISH_VOCABULARY", "LanguageSegment", "NEPALI",
    "ROMANIZED_NEPALI", "UNKNOWN", "is_mixed", "language_profile",
    "segment_languages",
]
