"""Nepali text normalization for speech: numbers, dates, times, phones, acronyms.

Stdlib-only on purpose: it runs in the app (``services.text_normalization``,
every TTS engine) and is loaded by file path inside the xtts-nepali sidecar,
which lives in its own venv.

Rules are conservative and idempotent — every rewrite removes the digits /
capital runs it matched, so a second pass is a no-op. Latin words in mixed
Nepali/English text are left alone (English spelling is not phonetic; users
override with the pronunciation dictionary).
"""
from __future__ import annotations

import re

NUMBER_WORDS = (
    "शून्य", "एक", "दुई", "तीन", "चार", "पाँच", "छ", "सात", "आठ", "नौ",
    "दश", "एघार", "बाह्र", "तेह्र", "चौध", "पन्ध्र", "सोह्र", "सत्र", "अठार",
    "उन्नाइस",
)
_TENS = ("", "", "बीस", "तीस", "चालीस", "पचास", "साठी", "सत्तरी", "असी", "नब्बे")
_20_TO_99 = {
    21: "एक्काइस", 22: "बाइस", 23: "तेइस", 24: "चौबीस", 25: "पच्चीस",
    26: "छब्बीस", 27: "सत्ताइस", 28: "अठ्ठाइस", 29: "उनन्तीस",
    31: "एकतीस", 32: "बत्तीस", 33: "तेत्तीस", 34: "चौँतीस", 35: "पैँतीस",
    36: "छत्तीस", 37: "सैँतीस", 38: "अठतीस", 39: "उनन्चालीस",
    41: "एकचालीस", 42: "बयालीस", 43: "त्रिचालीस", 44: "चवालीस", 45: "पैँतालीस",
    46: "छयालीस", 47: "सतचालीस", 48: "अठचालीस", 49: "उनन्चास",
    51: "एकाउन्न", 52: "बाउन्न", 53: "त्रिपन्न", 54: "चउन्न", 55: "पचपन्न",
    56: "छपन्न", 57: "सन्ताउन्न", 58: "अन्ठाउन्न", 59: "उनन्साठी",
    61: "एकसट्ठी", 62: "बयसट्ठी", 63: "त्रिसट्ठी", 64: "चौंसट्ठी", 65: "पैंसट्ठी",
    66: "छयसट्ठी", 67: "सतसट्ठी", 68: "अठसट्ठी", 69: "उनन्सत्तरी",
    71: "एकहत्तर", 72: "बहत्तर", 73: "त्रिहत्तर", 74: "चौहत्तर", 75: "पचहत्तर",
    76: "छयहत्तर", 77: "सतहत्तर", 78: "अठहत्तर", 79: "उनासी",
    81: "एकासी", 82: "बयासी", 83: "त्रियासी", 84: "चौरासी", 85: "पचासी",
    86: "छयासी", 87: "सतासी", 88: "अठासी", 89: "उनान्नब्बे",
    91: "एकान्नब्बे", 92: "बयानब्बे", 93: "त्रियान्नब्बे", 94: "चौरान्नब्बे",
    95: "पन्चानब्बे", 96: "छयान्नब्बे", 97: "सन्तानब्बे", 98: "अन्ठान्नब्बे",
    99: "उनान्सय",
}
#: Gregorian months as spoken in Nepali.
AD_MONTHS = (
    "", "जनवरी", "फेब्रुअरी", "मार्च", "अप्रिल", "मे", "जुन", "जुलाई", "अगस्ट",
    "सेप्टेम्बर", "अक्टोबर", "नोभेम्बर", "डिसेम्बर",
)
#: Bikram Sambat months (the calendar most Nepali dates are written in).
BS_MONTHS = (
    "", "बैशाख", "जेठ", "असार", "साउन", "भदौ", "असोज", "कात्तिक", "मंसिर",
    "पुस", "माघ", "फागुन", "चैत",
)
#: A 4-digit year in this range is Bikram Sambat (AD 1993–2143 ≈ BS 2050–2200).
_BS_YEARS = range(2050, 2201)

_ACRONYM_LETTERS = {
    "A": "ए", "B": "बी", "C": "सी", "D": "डी", "E": "ई", "F": "एफ",
    "G": "जी", "H": "एच", "I": "आई", "J": "जे", "K": "के", "L": "एल",
    "M": "एम", "N": "एन", "O": "ओ", "P": "पी", "Q": "क्यू", "R": "आर",
    "S": "एस", "T": "टी", "U": "यू", "V": "भी", "W": "डब्ल्यू", "X": "एक्स",
    "Y": "वाई", "Z": "जेड",
}

# Nepal phone formats: +977 international, 98/97/96 mobile, 01 Kathmandu
# landline. Matched before dates/integers so a phone is never one cardinal.
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+977(?:[\s-]?\d){10}|(?:98|97|96)(?:[\s-]?\d){8}|"
    r"01(?:[\s-]?\d){6,7})(?!\w)"
)
# Digit grouping with commas, Western (1,000,000) or lakh style (1,50,000). The
# last group must be three digits, so a decimal comma such as "3,14" is not.
# Number boundaries: Nepali attaches suffixes straight to digits ("10गते",
# "५जना"), so a Devanagari letter may follow. Digits touching a Latin letter or
# another number ("COVID19", "mp3", "1st", "v2.0.1", a ratio "3:2" or an
# invalid time) belong to a code and are left alone.
_S = r"(?<![A-Za-z\d:])(?<![A-Za-z\d][.,])"
_E = r"(?![A-Za-z\d]|:\d)"
_DEVANAGARI_LETTER = re.compile(r"[\u0904-\u0939\u093d\u0950\u0958-\u0961\u0972-\u097f]")
_GROUPED_INT_RE = re.compile(_S + r"\d{1,3}(?:,\d{2,3})*,\d{3}" + _E)
# A trailing "गते" / "बजे" already in the text is absorbed, not doubled.
_YMD_RE = re.compile(_S + r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})" + _E + r"(\s*गते)?")
_DMY_RE = re.compile(_S + r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})" + _E + r"(\s*गते)?")
_TIME_RE = re.compile(_S + r"(\d{1,2}):(\d{2})" + _E + r"(\s*बजे)?")
_PERCENT_RE = re.compile(r"(\d)\s?%")
_DECIMAL_RE = re.compile(_S + r"\d+[.,]\d+" + _E)
_INTEGER_RE = re.compile(_S + r"\d+" + _E)
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,8}\b")

#: Longest integer read as a cardinal; longer runs (IDs, account numbers) are
#: read digit by digit.
_MAX_CARDINAL_DIGITS = 13


def _under_hundred(value: int) -> str:
    if value < 20:
        return NUMBER_WORDS[value]
    if value in _20_TO_99:
        return _20_TO_99[value]
    return _TENS[value // 10] + (f" {NUMBER_WORDS[value % 10]}" if value % 10 else "")


_SCALES = ((10**11, "खर्ब"), (10**9, "अर्ब"), (10**7, "करोड"), (10**5, "लाख"), (1000, "हजार"))


def number_words(value: int) -> str:
    """Cardinal number in Nepali words (Indian grouping: हजार, लाख, करोड, अर्ब, खर्ब)."""
    if value < 0:
        return f"माइनस {number_words(-value)}"
    if value < 100:
        return _under_hundred(value)
    for scale, word in _SCALES:
        if value >= scale:
            head, rest = divmod(value, scale)
            return f"{number_words(head)} {word}" + (f" {number_words(rest)}" if rest else "")
    head, rest = divmod(value, 100)
    return f"{NUMBER_WORDS[head]} सय" + (f" {_under_hundred(rest)}" if rest else "")


def digit_words(digits: str) -> str:
    return " ".join(NUMBER_WORDS[int(d)] for d in digits)


def _date(year: int, month: int, day: int, m: re.Match) -> str:
    if not 1 <= month <= 12 or not 1 <= day <= 32:
        return m.group(0)
    if year in _BS_YEARS:
        return f"{number_words(year)} {BS_MONTHS[month]} {number_words(day)} गते"
    return f"{number_words(day)} {AD_MONTHS[month]} {number_words(year)}" + (m.group(4) or "")


def normalize_numbers(text: str) -> str:
    """Rewrite phones, dates, times, percents, decimals and integers as Nepali
    words. Accepts ASCII and Devanagari digits alike."""
    if not text:
        return text

    def phone(m: re.Match) -> str:
        prefix = "प्लस " if m.group(0).lstrip().startswith("+") else ""
        return prefix + digit_words(re.sub(r"\D", "", m.group(0)))

    def ymd(m: re.Match) -> str:
        return _date(int(m.group(1)), int(m.group(2)), int(m.group(3)), m)

    def dmy(m: re.Match) -> str:
        year = int(m.group(3))
        return _date(year + 2000 if year < 100 else year, int(m.group(2)), int(m.group(1)), m)

    def clock(m: re.Match) -> str:
        hour, minute = int(m.group(1)), int(m.group(2))
        if hour > 24 or minute > 59:
            return m.group(0)
        spoken = f"{number_words(hour)} बजे"
        return spoken if minute == 0 else f"{number_words(hour)} बजेर {number_words(minute)} मिनेट"

    def spaced(m: re.Match, words: str) -> str:
        # "५जना" → "पाँच जना": keep the suffix a separate spoken word.
        nxt = m.string[m.end():m.end() + 1]
        return f"{words} " if nxt and _DEVANAGARI_LETTER.match(nxt) else words

    def decimal(m: re.Match) -> str:
        whole, fraction = re.split(r"[.,]", m.group(0), maxsplit=1)
        return spaced(m, f"{number_words(int(whole))} दशमलव {digit_words(fraction)}")

    def integer(m: re.Match) -> str:
        raw = m.group(0)
        if len(raw) > _MAX_CARDINAL_DIGITS or (len(raw) > 1 and int(raw[0]) == 0):
            return spaced(m, digit_words(raw))  # IDs and leading-zero codes ("007")
        return spaced(m, number_words(int(raw)))

    text = _GROUPED_INT_RE.sub(lambda m: m.group(0).replace(",", ""), text)
    text = _PHONE_RE.sub(phone, text)
    text = _YMD_RE.sub(ymd, text)
    text = _DMY_RE.sub(dmy, text)
    text = _TIME_RE.sub(clock, text)
    text = _PERCENT_RE.sub(r"\1 प्रतिशत", text)
    text = _DECIMAL_RE.sub(decimal, text)
    return _INTEGER_RE.sub(integer, text)


def spell_acronyms(text: str) -> str:
    """Spell short all-capital Latin acronyms with Nepali letter names
    ("NEA" → "एन ई ए"). For engines that cannot read Latin script."""
    return _ACRONYM_RE.sub(lambda m: " ".join(_ACRONYM_LETTERS[ch] for ch in m.group(0)), text)
