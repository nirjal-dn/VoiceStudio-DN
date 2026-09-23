"""Spoken-number → digit inverse text normalization for the code-switch ASR.

Whisper transcribes numbers as *words* as often as digits, and in whichever
language was spoken: English number words in Latin, Nepali number words in
Devanagari. The ``whisper-ne-en`` engine wants digits, in the script of the
language the number was spoken in — and that language is known *here*, from the
number word itself, never guessed from the surrounding sentence::

    "nine thousand eight hundred forty-five"  -> "9845"    (English -> ASCII)
    "सन्तानब्बे, एकचालिस"                       -> "९७, ४१"   (Nepali  -> Devanagari)

English number words become **ASCII** digits; Nepali number words become
**Devanagari** digits. Bare digits Whisper already emitted (``9845``) are left
untouched — the original spoken language is no longer recoverable from the
transcript, so guessing it from neighbouring script would be wrong (an English
amount inside a Nepali sentence must stay ``9845``, not become ``९८४५``).

Scope / known ceilings:
  * Cardinals only (no ordinals/fractions). Compositional up through करोड /
    billion via the shared accumulate.
  * Nepali spelling varies by source and by what Whisper emits; :func:`_norm_ne`
    folds the common axes (ी↔ि, ू↔ु, dropped ँ, न्न↔न sandhi) so e.g.
    ``सन्तानब्बे`` and ``सन्तान्नब्बे`` both resolve to 97. It is not a full
    fuzzy matcher — an unrecognised spelling is simply left as words.
  * ``छ`` is both "six" and the copula "is", so a *standalone* ``छ`` is NOT
    converted (converting the verb everywhere would wreck normal Nepali). It
    still counts inside a multi-word number.
    # ponytail: homograph guard is छ-only; add others here if they surface.
"""
from __future__ import annotations

import re
import unicodedata

# ── English ─────────────────────────────────────────────────────────────────

_EN_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90,
}
_EN_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000,
              "billion": 1_000_000_000}
_EN_ALL = {**_EN_UNITS, **_EN_SCALES}

# One number word, then any run of more joined by spaces/hyphens and an optional
# "and" ("one hundred and five"). Longest-first alternation so "seventeen" wins
# over "seven". \b keeps "hundreds"/"oney" from matching.
_EN_WORD = "|".join(sorted(map(re.escape, _EN_ALL), key=len, reverse=True))
_EN_SPAN = re.compile(
    r"\b(?:%s)(?:[\s\-]+(?:and[\s\-]+)?(?:%s))*\b" % (_EN_WORD, _EN_WORD),
    re.IGNORECASE,
)


# ── Nepali ──────────────────────────────────────────────────────────────────

_NE_UNITS_RAW = {
    "शून्य": 0, "एक": 1, "दुई": 2, "तीन": 3, "चार": 4, "पाँच": 5, "छ": 6,
    "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "एघार": 11, "बाह्र": 12, "तेह्र": 13,
    "चौध": 14, "पन्ध्र": 15, "सोह्र": 16, "सत्र": 17, "अठार": 18, "उन्नाइस": 19,
    "बीस": 20, "एक्काइस": 21, "बाइस": 22, "तेइस": 23, "चौबीस": 24, "पच्चीस": 25,
    "छब्बीस": 26, "सत्ताइस": 27, "अठ्ठाइस": 28, "उनन्तीस": 29, "तीस": 30,
    "एकत्तीस": 31, "बत्तीस": 32, "तेत्तीस": 33, "चौँतीस": 34, "पैँतीस": 35,
    "छत्तीस": 36, "सैँतीस": 37, "अठतीस": 38, "उनन्चालीस": 39, "चालीस": 40,
    "एकचालीस": 41, "बयालीस": 42, "त्रिचालीस": 43, "चवालीस": 44, "पैँतालीस": 45,
    "छयालीस": 46, "सतचालीस": 47, "अठचालीस": 48, "उनन्चास": 49, "पचास": 50,
    "एकाउन्न": 51, "बाउन्न": 52, "त्रिपन्न": 53, "चउन्न": 54, "पचपन्न": 55,
    "छपन्न": 56, "सन्ताउन्न": 57, "अन्ठाउन्न": 58, "उनन्साठी": 59, "साठी": 60,
    "एकसट्ठी": 61, "बयसट्ठी": 62, "त्रिसट्ठी": 63, "चौसट्ठी": 64, "पैँसट्ठी": 65,
    "छयसट्ठी": 66, "सतसट्ठी": 67, "अठसट्ठी": 68, "उनन्सत्तरी": 69, "सत्तरी": 70,
    "एकहत्तर": 71, "बहत्तर": 72, "त्रिहत्तर": 73, "चौहत्तर": 74, "पचहत्तर": 75,
    "छयहत्तर": 76, "सतहत्तर": 77, "अठहत्तर": 78, "उनासी": 79, "असी": 80,
    "एकासी": 81, "बयासी": 82, "त्रियासी": 83, "चौरासी": 84, "पचासी": 85,
    "छयासी": 86, "सतासी": 87, "अठासी": 88, "उनान्नब्बे": 89, "नब्बे": 90,
    "एकान्नब्बे": 91, "बयान्नब्बे": 92, "त्रियान्नब्बे": 93, "चौरान्नब्बे": 94,
    "पन्चान्नब्बे": 95, "छयान्नब्बे": 96, "सन्तान्नब्बे": 97, "अन्ठान्नब्बे": 98,
    "उनान्सय": 99,
    # Common spelling alternates the fold below does not cover.
    "दश": 10, "शुन्य": 0,
}
_NE_SCALES_RAW = {"सय": 100, "हजार": 1000, "लाख": 100_000, "करोड": 10_000_000}


def _norm_ne(word: str) -> str:
    """Fold the Nepali spelling axes that vary between sources and Whisper output
    so one table entry matches every common rendering (see module docstring)."""
    w = unicodedata.normalize("NFC", word)
    return (w.replace("ँ", "").replace("ी", "ि").replace("ू", "ु")
             .replace("न्न", "न"))


_NE_UNITS = {_norm_ne(k): v for k, v in _NE_UNITS_RAW.items()}
_NE_SCALES = {_norm_ne(k): v for k, v in _NE_SCALES_RAW.items()}
_NE_ALL = {**_NE_UNITS, **_NE_SCALES}
#: Homographs too risky to convert as a lone word (छ = "six" and the copula "is").
_NE_AMBIGUOUS = {_norm_ne("छ")}

#: A maximal run of Devanagari letters = one "word". Excludes the digits ०-९
#: (U+0966-096F), the dandas । ॥ (U+0964-0965) and the abbreviation sign ॰
#: (U+0970) so a number touching punctuation still matches (``पाँच।`` → ``५।``).
_NE_WORD_RE = re.compile(r"[ऀ-ॣॱ-ॿ]+")

#: ASCII → Devanagari digit map. A Nepali number *word* is rendered straight to
#: Devanagari numerals — the spoken language is known, so no context guess.
_DEVANAGARI_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")


def _accumulate(values) -> int:
    """Fold a left-to-right sequence of (kind, value) number tokens into an int.
    ``kind`` is 'u' (unit/tens, additive), 'h' (hundred, multiplies the current
    group) or 's' (thousand+, flushes the current group). Shared by both langs
    ("two thousand eighty-one" and "दुई हजार एकासी" fold identically)."""
    result = current = 0
    for kind, val in values:
        if kind == "u":
            current += val
        elif kind == "h":
            current = (current or 1) * val
        else:  # 's' — thousand / million / लाख / करोड
            result += (current or 1) * val
            current = 0
    return result + current


def _classify_en(word: str):
    w = word.lower()
    if w in _EN_UNITS:
        return ("u", _EN_UNITS[w])
    if w == "hundred":
        return ("h", 100)
    if w in _EN_SCALES:
        return ("s", _EN_SCALES[w])
    return None


def english_words_to_digits(text: str) -> str:
    """Replace English cardinal number words with ASCII digits.

    ``"nine thousand eight hundred forty-five"`` → ``"9845"``. Non-number text,
    existing digits, and Devanagari are untouched."""
    def repl(m):
        toks = [t for t in re.split(r"[\s\-]+", m.group()) if t and t.lower() != "and"]
        vals = [_classify_en(t) for t in toks]
        if any(v is None for v in vals):
            return m.group()  # a non-number word slipped in — leave verbatim
        return str(_accumulate(vals))

    return _EN_SPAN.sub(repl, text)


def _classify_ne(norm_word: str):
    if norm_word in _NE_UNITS:
        return ("u", _NE_UNITS[norm_word])
    if norm_word == _norm_ne("सय"):
        return ("h", 100)
    if norm_word in _NE_SCALES:
        return ("s", _NE_SCALES[norm_word])
    return None


def nepali_words_to_digits(text: str) -> str:
    """Replace Nepali (Devanagari) cardinal number words with Devanagari digits.

    ``"सन्तानब्बे, एकचालिस"`` → ``"९७, ४१"`` (the comma breaks the two numbers
    apart — only whitespace-separated words join into one number). A Nepali
    number word is unambiguously Nepali, so it goes straight to Devanagari
    numerals — no guess from surrounding script, and bare digits already in the
    text are never touched.

    ``छ`` ("six" *and* the copula "is") is only treated as a number when the very
    next word is a scale (``छ सय`` = 600); anywhere else — standalone, trailing a
    number (``पच्चीस छ`` = "twenty-five is"), or before a plain word — it stays a
    word. That keeps the far-more-common verb from being turned into ``6``."""
    words = list(_NE_WORD_RE.finditer(text))

    def ws_gap(a: int, b: int) -> bool:
        """True when words a→b are separated by whitespace only (so they belong
        to one number); a comma or any other char ends the number."""
        return not text[words[a].end():words[b].start()].strip()

    def number_at(k: int):
        """Classification of word k as a number token, honouring the ``छ``
        homograph rule, or None when it is not a number in this position."""
        nw = _norm_ne(words[k].group())
        cls = _classify_ne(nw)
        if cls is None:
            return None
        if nw in _NE_AMBIGUOUS:
            nxt_is_scale = (
                k + 1 < len(words) and ws_gap(k, k + 1)
                and _norm_ne(words[k + 1].group()) in _NE_SCALES
            )
            return cls if nxt_is_scale else None
        return cls

    repls = []  # (start, end, digits)
    i = 0
    while i < len(words):
        cls = number_at(i)
        if cls is None:
            i += 1
            continue
        # Extend the span across whitespace-only gaps of further number words.
        span = [cls]
        j = i + 1
        while j < len(words) and ws_gap(j - 1, j):
            nxt = number_at(j)
            if nxt is None:
                break
            span.append(nxt)
            j += 1
        digits = str(_accumulate(span)).translate(_DEVANAGARI_DIGITS)
        repls.append((words[i].start(), words[j - 1].end(), digits))
        i = j

    if not repls:
        return text
    out, last = [], 0
    for start, end, digits in repls:
        out.append(text[last:start])
        out.append(digits)
        last = end
    out.append(text[last:])
    return "".join(out)


def to_digits(text: str) -> str:
    """Nepali (→ Devanagari digits) then English (→ ASCII digits) spoken-number
    conversion. The scripts are disjoint, so the passes never touch each other's
    output; bare digits already in the text are preserved. The ASR entry point."""
    return english_words_to_digits(nepali_words_to_digits(text))
