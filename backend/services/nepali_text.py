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

# ── currency ──────────────────────────────────────────────────────────────
# A number is only spoken as MONEY when the text itself carries a currency
# marker. A bare number stays a bare number ("100 किताब" → "एक सय किताब",
# never "एक सय रुपैयाँ"), which is the whole point: the marker is the signal,
# not the digits.
#
# ``रु`` already survived because it is Devanagari. The Latin spellings did
# not: "Rs" reached the transliterator and came out "र्स", and "NPR" was
# spelled letter by letter — so exactly the markers a user is most likely to
# type were the ones that broke.
#: The Devanagari marker every other spelling is rewritten to. It is NOT
#: expanded to "रुपैयाँ": the existing normalizer deliberately leaves रु. in
#: place (it is in the sidecar's _NE_ABBREVIATIONS so its period does not end
#: a sentence), and six tests pin that. Only the spellings that BREAK are
#: converted.
_CURRENCY_MARKER = "रु."
_PAISA_WORD = "पैसा"
#: Latin and symbol currency markers. These are the ones that failed: "Rs"
#: survived normalization, reached the transliterator and came out "र्स", and
#: "NPR" was spelled letter by letter — so the spellings a user is most likely
#: to type were exactly the broken ones. The Devanagari रु / रू already work
#: and are matched only so an amount's DECIMAL can be read as paisa.
_LATIN_CURRENCY_RE = re.compile(
    r"(?<![A-Za-z\u0900-\u097f])(?:NPR|INR|Rs\.|Rs|₹)\s*"
    r"(?=\d)",
    re.IGNORECASE,
)
#: An amount whose fraction should be read as paisa rather than digits.
_PAISA_RE = re.compile(
    r"((?<![A-Za-z\u0900-\u097f])(?:रू\.|रु\.|रू|रु)\s*)"
    r"(\d+(?:,\d{2,3})*)\.(\d{2})(?![\d])"
)

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
        """``15/05/2082`` → day/month/year, the convention in Nepal.

        A first field above 12 cannot be a month, so ``05/15/2026`` is read as
        the US month/day order instead of being left with its slashes spoken
        aloud. A genuinely ambiguous pair ("01/02") stays day-first — guessing
        the other way would be just as wrong and less expected here.
        """
        first, second = int(m.group(1)), int(m.group(2))
        year = int(m.group(3))
        year = year + 2000 if year < 100 else year
        if second > 12 and first <= 12:
            day, month = second, first  # month/day/year
        else:
            day, month = first, second  # day/month/year
        return _date(year, month, day, m)

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

    def paisa(m: re.Match) -> str:
        """``रु 45.50`` → ``रु. पैँतालीस पचास पैसा``.

        Two decimal places after a currency marker are paisa, not digits:
        "पैँतालीस दशमलव पाँच शून्य" is how a measurement is read, not money.
        Exactly two — one ("रु 1.5") is ambiguous, so it falls through to the
        ordinary decimal pass rather than being silently called 50 paisa.
        """
        marker, whole, fraction = m.group(1), m.group(2), m.group(3)
        rupees = number_words(int(whole.replace(",", "")))
        cents = int(fraction)
        if not cents:
            return f"{marker}{rupees}"
        return f"{marker}{rupees} {number_words(cents)} {_PAISA_WORD}"

    def decimal(m: re.Match) -> str:
        whole, fraction = re.split(r"[.,]", m.group(0), maxsplit=1)
        return spaced(m, f"{number_words(int(whole))} दशमलव {digit_words(fraction)}")

    def integer(m: re.Match) -> str:
        raw = m.group(0)
        if len(raw) > _MAX_CARDINAL_DIGITS or (len(raw) > 1 and int(raw[0]) == 0):
            return spaced(m, digit_words(raw))  # IDs and leading-zero codes ("007")
        return spaced(m, number_words(int(raw)))

    # Latin markers first, rewritten to the Devanagari one the rest of the
    # pipeline (and the sidecar's abbreviation list) already understands.
    text = _LATIN_CURRENCY_RE.sub(_CURRENCY_MARKER + " ", text)
    # Then paisa, BEFORE _DECIMAL_RE claims the fraction as bare digits.
    text = _PAISA_RE.sub(paisa, text)
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


# ── Latin → Devanagari transliteration ────────────────────────────────────
# The xtts-nepali checkpoint reads Devanagari. Latin words are NOT rejected by
# the Hindi cleaner this engine routes through — they tokenize fine ("email" →
# 3 tokens) — but those tokens carry Devanagari phonetics and the Nepali
# fine-tune saw almost no Latin in training, so the model GUESSES at them.
# That guess is what an English word inside a Nepali sentence sounds like.
# Rewriting Latin to Devanagari hands the model tokens it actually knows, and
# keeps the whole sentence on ONE engine — hence one voice.
#
# Precedence, most specific first:
#   1. SPOKEN_ACRONYMS — said as a word ("OK" → ओके, not ओ के)
#   2. ALL-CAPS        — the writer's own signal for an initialism
#   3. LOANWORDS       — conventional spellings no rule can derive
#   4. PROPER_NOUNS    — names, places, brands
#   5. _translit_word  — phonetic fallback
#
# Stdlib-only, like the rest of this file: it is loaded BY PATH into the
# sidecar's own venv (engines/xtts_nepali/main.py::_load_nepali_text), so it
# can never import a transliteration package.

#: Everyday English loanwords with settled Nepali spellings. Rule-based
#: transliteration mangles these (office → *अफ्फिसे), which is why they are a
#: table. Nepali, not Hindi, vowels: अफिस not ऑफिस, डाक्टर not डॉक्टर.
LOANWORDS: dict[str, str] = {
    # Work / office
    "office": "अफिस", "meeting": "मिटिङ", "manager": "म्यानेजर",
    "boss": "बस", "staff": "स्टाफ", "team": "टिम", "project": "प्रोजेक्ट",
    "client": "क्लाइन्ट", "customer": "कस्टमर", "company": "कम्पनी",
    "business": "बिजनेस", "job": "जब", "work": "वर्क", "salary": "सलरी",
    "budget": "बजेट", "report": "रिपोर्ट", "presentation": "प्रेजेन्टेसन",
    "schedule": "सेड्युल", "deadline": "डेडलाइन", "training": "ट्रेनिङ",
    "interview": "इन्टरभ्यु", "contract": "कन्ट्र्याक्ट", "branch": "ब्रान्च",
    "department": "डिपार्टमेन्ट", "shift": "सिफ्ट", "leave": "बिदा",
    # Technology
    "computer": "कम्प्युटर", "laptop": "ल्याप्टप", "mobile": "मोबाइल",
    "phone": "फोन", "internet": "इन्टरनेट", "online": "अनलाइन",
    "offline": "अफलाइन", "email": "इमेल", "software": "सफ्टवेयर",
    "hardware": "हार्डवेयर", "website": "वेबसाइट", "browser": "ब्राउजर",
    "download": "डाउनलोड", "upload": "अपलोड", "install": "इन्स्टल",
    "update": "अपडेट", "backup": "ब्याकअप", "restart": "रिस्टार्ट",
    "password": "पासवर्ड", "login": "लगिन", "account": "एकाउन्ट",
    "server": "सर्भर", "database": "डाटाबेस", "network": "नेटवर्क",
    "screen": "स्क्रिन", "camera": "क्यामेरा", "battery": "ब्याट्री",
    "charger": "चार्जर", "cable": "केबल", "printer": "प्रिन्टर",
    "keyboard": "किबोर्ड", "mouse": "माउस", "file": "फाइल",
    "folder": "फोल्डर", "data": "डाटा", "message": "मेसेज",
    "video": "भिडियो", "audio": "अडियो", "photo": "फोटो",
    "app": "एप", "link": "लिंक", "click": "क्लिक", "search": "सर्च",
    "system": "सिस्टम", "program": "प्रोग्राम", "code": "कोड",
    "model": "मोडेल", "test": "टेस्ट", "bug": "बग", "feature": "फिचर",
    # Money / commerce
    "bank": "बैंक", "money": "मनी", "cash": "क्यास", "card": "कार्ड",
    "loan": "लोन", "interest": "इन्टरेस्ट", "payment": "पेमेन्ट",
    "bill": "बिल", "price": "प्राइस", "discount": "डिस्काउन्ट",
    "market": "मार्केट", "shop": "सप", "shopping": "सपिङ",
    "order": "अर्डर", "delivery": "डेलिभरी", "service": "सर्भिस",
    "ticket": "टिकट", "booking": "बुकिङ", "tax": "ट्याक्स",
    # Education
    "school": "स्कुल", "college": "कलेज", "university": "युनिभर्सिटी",
    "student": "स्टुडेन्ट", "teacher": "टिचर", "class": "क्लास",
    "exam": "परीक्षा", "result": "रिजल्ट", "subject": "सब्जेक्ट",
    "book": "बुक", "notebook": "नोटबुक", "library": "लाइब्रेरी",
    "course": "कोर्स", "degree": "डिग्री", "study": "स्टडी",
    "admission": "भर्ना", "scholarship": "छात्रवृत्ति", "fee": "शुल्क",
    # Health
    "doctor": "डाक्टर", "nurse": "नर्स", "medicine": "मेडिसिन",
    "clinic": "क्लिनिक", "health": "हेल्थ", "hospital": "अस्पताल",
    "operation": "अपरेसन", "blood": "ब्लड", "mask": "मास्क",
    "vaccine": "भ्याक्सिन", "fever": "ज्वरो",
    # Transport
    "bus": "बस", "taxi": "ट्याक्सी", "car": "कार", "bike": "बाइक",
    "cycle": "साइकल", "train": "ट्रेन", "flight": "फ्लाइट",
    "airport": "एयरपोर्ट", "station": "स्टेसन", "driver": "ड्राइभर",
    "road": "रोड", "traffic": "ट्राफिक", "parking": "पार्किङ",
    # Daily life
    "hotel": "होटल", "restaurant": "रेस्टुरेन्ट", "room": "रुम",
    "kitchen": "किचन", "table": "टेबल", "chair": "चेयर",
    "tv": "टिभी", "radio": "रेडियो", "news": "न्युज", "paper": "पेपर",
    "refrigerator": "रेफ्रिजेरेटर", "fridge": "फ्रिज", "fan": "पंखा",
    "heater": "हिटर", "oven": "ओभन", "party": "पार्टी", "plan": "प्लान",
    "time": "टाइम", "date": "डेट", "weekend": "विकेन्ड",
    "holiday": "बिदा", "birthday": "बर्थडे", "gift": "गिफ्ट",
    "form": "फारम",
    # Government / civic
    "government": "सरकार", "minister": "मन्त्री", "police": "प्रहरी",
    "court": "अदालत", "license": "लाइसेन्स", "passport": "राहदानी",
    "visa": "भिसा", "citizenship": "नागरिकता", "election": "निर्वाचन",
    "policy": "नीति",
    # Common adjectives / verbs in mixed speech
    "ok": "ओके", "fine": "फाइन", "good": "गुड", "best": "बेस्ट",
    "new": "न्यु", "old": "ओल्ड", "full": "फुल", "free": "फ्रि",
    "ready": "रेडी", "busy": "बिजी", "sorry": "सरी", "thanks": "थ्यांक्स",
    "please": "प्लिज", "welcome": "वेलकम", "start": "स्टार्ट",
    "stop": "स्टप", "close": "क्लोज", "open": "ओपन", "send": "सेन्ड",
    "check": "चेक", "call": "कल", "meet": "मिट", "join": "जोइन",
    "share": "सेयर", "save": "सेभ", "cancel": "क्यान्सल",
    "confirm": "कन्फर्म", "approve": "अप्रुभ", "review": "रिभ्यु",
    "complete": "कम्प्लिट", "final": "फाइनल", "total": "टोटल",
    "problem": "प्रब्लम", "solution": "सोलुसन", "idea": "आइडिया",
    "reason": "रिजन", "change": "चेन्ज", "level": "लेभल",
    "size": "साइज", "number": "नम्बर", "list": "लिस्ट", "group": "ग्रुप",
    "member": "मेम्बर", "friend": "फ्रेन्ड", "family": "परिवार",
    "home": "होम", "city": "सिटी", "country": "देश", "world": "वर्ल्ड",
    "people": "पिपल", "person": "पर्सन", "life": "लाइफ",
    "power": "पावर", "energy": "एनर्जी", "water": "वाटर",
    "light": "लाइट", "air": "एयर", "gas": "ग्यास",
}

#: Acronyms pronounced as a WORD. Without these, spell_acronyms' letter-by-
#: letter rule turns "OK" into "ओ के" and "NASA" into "एन ए एस ए".
SPOKEN_ACRONYMS: dict[str, str] = {
    "OK": "ओके", "NASA": "नासा", "UNESCO": "युनेस्को", "UNICEF": "युनिसेफ",
    "SAARC": "सार्क", "COVID": "कोभिड", "AIDS": "एड्स", "RADAR": "रडार",
    "LASER": "लेजर", "SIM": "सिम", "ATM": "एटिएम", "PIN": "पिन",
    "WIFI": "वाइफाइ", "SEE": "एसइइ", "NEB": "नेब",
}

#: Names, places and brands the fallback mangles. Unbounded in general; these
#: are the ones that actually recur in Nepali speech.
PROPER_NOUNS: dict[str, str] = {
    # Places
    "nepal": "नेपाल", "kathmandu": "काठमाडौं", "pokhara": "पोखरा",
    "lalitpur": "ललितपुर", "bhaktapur": "भक्तपुर", "chitwan": "चितवन",
    "biratnagar": "विराटनगर", "birgunj": "वीरगञ्ज", "dharan": "धरान",
    "butwal": "बुटवल", "nepalgunj": "नेपालगञ्ज", "janakpur": "जनकपुर",
    "everest": "सगरमाथा", "india": "भारत", "china": "चीन",
    "america": "अमेरिका", "australia": "अस्ट्रेलिया", "japan": "जापान",
    "korea": "कोरिया", "london": "लन्डन", "dubai": "दुबई",
    # Brands
    "google": "गुगल", "facebook": "फेसबुक", "youtube": "युट्युब",
    "whatsapp": "ह्वाट्सएप", "instagram": "इन्स्टाग्राम", "viber": "भाइबर",
    "microsoft": "माइक्रोसफ्ट", "apple": "एप्पल", "android": "एन्ड्रोइड",
    "windows": "विन्डोज", "gmail": "जिमेल", "zoom": "जुम",
    # Given names — the fallback maps every 'a' to the inherent vowel, which
    # is right for English ("stand" → स्टन्ड) and wrong here ("Rajesh" → रजेश).
    "ram": "राम", "shyam": "श्याम", "sita": "सीता", "gita": "गीता",
    "hari": "हरि", "krishna": "कृष्ण", "bishnu": "विष्णु",
    "suman": "सुमन", "sumit": "सुमित", "nirjal": "निर्जल",
    "john": "जोन", "michael": "माइकल", "david": "डेभिड", "sarah": "सारा",
    "rajesh": "राजेश", "prakash": "प्रकाश", "bishal": "विशाल",
    "subash": "सुबास", "nabin": "नवीन", "dinesh": "दिनेश",
    "bikash": "विकास", "manish": "मनीष", "anita": "अनिता",
    "sunita": "सुनिता", "kamala": "कमला", "sarita": "सरिता",
    "binod": "विनोद", "deepak": "दीपक", "sandeep": "सन्दीप",
    "santosh": "सन्तोष", "raju": "राजु", "gopal": "गोपाल",
    # Surnames
    "shah": "शाह", "singh": "सिंह", "sharma": "शर्मा", "thapa": "थापा",
    "shrestha": "श्रेष्ठ", "gurung": "गुरुङ", "magar": "मगर",
    "tamang": "तामाङ", "maharjan": "महर्जन", "adhikari": "अधिकारी",
    "poudel": "पौडेल", "bhattarai": "भट्टराई", "karki": "कार्की",
    "rai": "राई", "limbu": "लिम्बु", "chhetri": "क्षेत्री",
}


#: Multi-letter graphemes, longest first — the matcher is greedy, so order
#: matters. English spelling is not phonetic, so this is an approximation with
#: a known ceiling, used only for words no table covers.
_DIGRAPHS: tuple[tuple[str, str], ...] = (
    ("tion", "सन"), ("sion", "सन"), ("ough", "अफ"), ("augh", "अफ"),
    ("tch", "च"), ("dge", "ज"), ("igh", "ाइ"), ("ing", "िङ"),
    ("ch", "च"), ("sh", "श"), ("th", "थ"), ("ph", "फ"), ("gh", "ग"),
    ("ck", "क"), ("ng", "ङ"), ("qu", "क्व"), ("wh", "व"),
    ("ee", "ी"), ("ea", "ी"), ("oo", "ु"), ("ou", "ाउ"), ("ow", "ाउ"),
    ("ai", "ै"), ("ay", "े"), ("oi", "ोइ"), ("oy", "ोइ"),
    ("au", "ौ"), ("aw", "ौ"), ("ie", "ी"), ("ei", "े"),
)

#: Single consonants. Devanagari consonants carry an inherent 'a', which the
#: vowel logic below strips (halanta) or replaces (matra).
_CONSONANTS: dict[str, str] = {
    "b": "ब", "c": "क", "d": "ड", "f": "फ", "g": "ग", "h": "ह",
    "j": "ज", "k": "क", "l": "ल", "m": "म", "n": "न", "p": "प",
    "q": "क", "r": "र", "s": "स", "t": "ट", "v": "भ", "w": "व",
    "x": "क्स", "y": "य", "z": "ज",
}

#: Vowels as MATRAS (combining signs, after a consonant).
_VOWEL_MATRAS: dict[str, str] = {"a": "", "e": "े", "i": "ि", "o": "ो", "u": "ु"}

#: Vowels as INDEPENDENT letters (word-initial, or after another vowel).
_VOWEL_LETTERS: dict[str, str] = {
    "a": "अ", "e": "ए", "i": "इ", "o": "ओ", "u": "उ",
}

_HALANTA = "्"
_MATRA_CHARS = "ािीुूेैोौ"


def _translit_word(word: str) -> str:
    """Phonetic Latin→Devanagari for a word no table covers.

    Deliberately approximate: English orthography does not map cleanly to any
    phonetic script ("though" / "through" / "tough" share four letters and
    three sounds). The goal is a pronounceable Devanagari form the Nepali
    checkpoint can voice in the SAME speaker as the rest of the sentence — not
    a faithful English accent. A word that must be exact belongs in LOANWORDS
    or in the user's pronunciation dictionary.
    """
    lowered = word.lower()
    # Silent terminal 'e' ("name", "close"): English lengthens the preceding
    # vowel and drops this one; keeping it produced "नमे" for "name". Not for
    # 2-3 letter words ("the", "see") or after a vowel ("free"), where it is
    # voiced.
    if len(lowered) > 3 and lowered.endswith("e") and lowered[-2] not in "aeiou":
        lowered = lowered[:-1]
    out: list[str] = []
    i = 0
    at_start = True
    while i < len(lowered):
        for source, target in _DIGRAPHS:
            if lowered.startswith(source, i):
                # A matra needs a consonant to attach to; at a word start there
                # is none, so fall through to the single-letter path, which
                # emits an independent vowel instead.
                if target and target[0] in _MATRA_CHARS and at_start:
                    break
                out.append(target)
                i += len(source)
                at_start = False
                break
        else:
            ch = lowered[i]
            if ch in _CONSONANTS:
                out.append(_CONSONANTS[ch])
                nxt = lowered[i + 1] if i + 1 < len(lowered) else ""
                if nxt not in _VOWEL_MATRAS:
                    out.append(_HALANTA)  # suppress the inherent 'a'
                at_start = False
            elif ch in _VOWEL_LETTERS:
                if at_start:
                    out.append(_VOWEL_LETTERS[ch])
                else:
                    if out and out[-1] == _HALANTA:
                        out.pop()
                    out.append(_VOWEL_MATRAS[ch])
                at_start = False
            else:
                out.append(ch)  # digits, punctuation — untouched
            i += 1
    result = "".join(out)
    # A word-final halanta reads as a clipped consonant; Nepali prefers the
    # inherent vowel there ("नेट्" → "नेट").
    if result.endswith(_HALANTA):
        result = result[:-1]
    return result


#: Latin word, optionally carrying internal apostrophes ("don't").
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")

#: Bracketed spans are markup or a user pronunciation override — same shape as
#: chunked_tts._BRACKET_TAG_RE. Never transliterated: "[pause 300ms]" must stay
#: parseable, and "[[office]]" is the user's explicit final say.
_BRACKET_SPAN_RE = re.compile(r"\[[^\][\n]{0,128}\]")

_ALL_CAPS_RE = re.compile(r"^[A-Z]{2,8}$")


def _word_to_devanagari(word: str) -> str:
    """One Latin token → Devanagari, most specific rule first."""
    upper = word.upper()
    if upper in SPOKEN_ACRONYMS:
        return SPOKEN_ACRONYMS[upper]
    # ALL-CAPS is the writer's own signal that this is an initialism, and it
    # beats the tables: "USA" reads "यू एस ए" while "usa" would hit
    # PROPER_NOUNS. Both are right for their own spelling.
    if _ALL_CAPS_RE.match(word):
        # "NEA" → "एन ई ए", matching spell_acronyms so the two normalizer
        # paths (TRANSLITERATE on/off) never disagree about an acronym.
        return " ".join(_ACRONYM_LETTERS[ch] for ch in word)
    lowered = word.lower()
    if lowered in LOANWORDS:
        return LOANWORDS[lowered]
    if lowered in PROPER_NOUNS:
        return PROPER_NOUNS[lowered]
    return _translit_word(word)


def transliterate_latin(text: str) -> str:
    """Rewrite every Latin word in *text* as Devanagari.

    For engines whose checkpoint reads Devanagari (xtts-nepali). Leaves
    bracketed markup and ``[[…]]`` pronunciation overrides untouched, and is
    idempotent: the output has no Latin left, so a second pass is a no-op.
    """
    spans = [(m.start(), m.end()) for m in _BRACKET_SPAN_RE.finditer(text)]

    def protected(start: int, end: int) -> bool:
        return any(s <= start and end <= e for s, e in spans)

    def replace(m: re.Match) -> str:
        if protected(m.start(), m.end()):
            return m.group(0)
        return _word_to_devanagari(m.group(0))

    return _LATIN_WORD_RE.sub(replace, text)
