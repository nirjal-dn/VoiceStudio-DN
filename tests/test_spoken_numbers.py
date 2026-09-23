"""Spoken-number → digit inverse normalization (``services.spoken_numbers``).

Pure string logic, no model/weights. English number words → ASCII digits,
Nepali number words → Devanagari digits, each decided from the number word
itself; bare digits already in the text are preserved. These pin the converter
contract the code-switch ASR relies on (end-to-end in ``test_code_switch_asr.py``).
"""
from __future__ import annotations

import pytest

from services import spoken_numbers as sn
from services.spoken_numbers import _NE_UNITS_RAW


# ── English words → digits ───────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("nine thousand eight hundred forty-five", "9845"),   # the spec example
    ("forty-five", "45"),
    ("one hundred and five", "105"),
    ("twenty five", "25"),
    ("two thousand eighty-one", "2081"),
    ("one million", "1000000"),
    ("zero", "0"),
    # Context words untouched; only the number span converts.
    ("I have twenty-five apples and three hundred rupees",
     "I have 25 apples and 300 rupees"),
    # Not a number → verbatim (no stray conversion of ordinary words).
    ("hello world", "hello world"),
    ("hundreds of people", "hundreds of people"),   # "hundreds" ≠ "hundred"
    ("", ""),
])
def test_english_words_to_digits(text, expected):
    assert sn.english_words_to_digits(text) == expected


# ── Nepali words → digits ────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    # Spec example — user spelling differs from the source table (सन्तान्नब्बे /
    # एकचालीस) yet both resolve via the spelling fold. The comma keeps them two
    # separate numbers.
    ("सन्तानब्बे, एकचालिस", "९७, ४१"),
    ("एकचालीस", "४१"),
    ("नब्बे", "९०"),
    ("एक सय", "१००"),
    ("एक सय पच्चीस", "१२५"),
    ("एक सय एकचालीस", "१४१"),
    ("दुई हजार एकासी", "२०८१"),
    ("छ सय", "६००"),                       # छ IS a number right before a scale
    ("मलाई एक सय रुपैयाँ चाहियो", "मलाई १०० रुपैयाँ चाहियो"),
    ("पाँच जना", "५ जना"),
    ("", ""),
])
def test_nepali_words_to_digits(text, expected):
    # Nepali number words are unambiguously Nepali → Devanagari digits directly.
    assert sn.nepali_words_to_digits(text) == expected


@pytest.mark.parametrize("text", [
    "मेरो नाम राम छ",                       # छ = copula "is", standalone
    "उमेर पच्चीस छ",                        # trailing छ must NOT join पच्चीस
    "यो राम्रो छ",
])
def test_standalone_chha_is_not_converted(text):
    # छ ("is") is the common Nepali copula; only "छ <scale>" (छ सय) is a number.
    out = sn.nepali_words_to_digits(text)
    assert "छ" in out and "6" not in out and "६" not in out


def test_trailing_chha_does_not_inflate_preceding_number():
    # Regression: "पच्चीस छ" (twenty-five is) must be "२५ छ", never 25+6=31.
    assert sn.nepali_words_to_digits("उमेर पच्चीस छ") == "उमेर २५ छ"


def test_full_nepali_table_roundtrips():
    # Every 0–99 word in the table converts back to its integer, in Devanagari (छ
    # excluded — it only counts before a scale). Guards a bad table edit/collision.
    to_dev = str.maketrans("0123456789", "०१२३४५६७८९")
    for word, n in _NE_UNITS_RAW.items():
        if n == 6:
            continue
        assert sn.nepali_words_to_digits(f"{word} वटा") == f"{str(n).translate(to_dev)} वटा"


def test_spelling_variants_fold_to_same_number():
    # The fold unifies the axes that vary between sources and Whisper output.
    assert sn.nepali_words_to_digits("सन्तान्नब्बे") == "९७"   # source spelling
    assert sn.nepali_words_to_digits("सन्तानब्बे") == "९७"     # user spelling
    assert sn.nepali_words_to_digits("चालीस") == "४०"          # long ी
    assert sn.nepali_words_to_digits("चालिस") == "४०"          # short ि
    assert sn.nepali_words_to_digits("दस") == "१०"
    assert sn.nepali_words_to_digits("दश") == "१०"             # स/श alternate


# ── Combined entry point (both scripts, disjoint) ────────────────────────────

def test_to_digits_handles_both_languages_in_one_string():
    # English → ASCII, Nepali → Devanagari, in the same sentence.
    assert sn.to_digits("मलाई एक सय रुपैयाँ र five dollars चाहियो") == \
        "मलाई १०० रुपैयाँ र 5 dollars चाहियो"


def test_to_digits_leaves_existing_digits_untouched():
    assert sn.to_digits("मैले 2081-05-15 मा 3 PM meeting राखेँ") == \
        "मैले 2081-05-15 मा 3 PM meeting राखेँ"
