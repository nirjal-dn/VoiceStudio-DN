"""Latin → Devanagari transliteration for the xtts-nepali checkpoint.

The Oshara fine-tune reads Devanagari. Latin words reached its Hindi cleaner
route, which has no rules for them, so English came out as whatever the model
guessed. Routing those spans to a second (English) engine instead — the
code-switch pipeline — produced a take that sounded like TWO DIFFERENT PEOPLE,
because the fine-tune and the base checkpoint no longer share a speaker
embedding. Rewriting Latin to Devanagari keeps one engine, hence one voice.

Stdlib-only: services/nepali_text.py is loaded BY PATH into the sidecar's own
venv, so it can never import a transliteration package.
"""
from __future__ import annotations

import pytest

from services.nepali_text import (
    LOANWORDS, PROPER_NOUNS, SPOKEN_ACRONYMS, transliterate_latin,
)

DEVANAGARI = range(0x0900, 0x0980)


def _has_latin(text: str) -> bool:
    return any("a" <= c.lower() <= "z" for c in text)


# ── the point of the feature ──────────────────────────────────────────────

def test_a_mixed_sentence_comes_back_with_no_latin_left():
    """The whole reason this exists: one engine can only speak one script."""
    out = transliterate_latin("म आज office गएर manager सँग meeting गर्छु।")
    assert not _has_latin(out)
    assert out == "म आज अफिस गएर म्यानेजर सँग मिटिङ गर्छु।"


@pytest.mark.parametrize("word", ["office", "email", "computer", "doctor", "bank"])
def test_every_common_loanword_uses_its_conventional_spelling(word):
    """Rule-based transliteration mangles these (office → *अफ्फिसे), which is
    why they are a table rather than a rule."""
    assert transliterate_latin(word) == LOANWORDS[word]


def test_an_unlisted_word_still_loses_its_latin():
    """Decision (a): unknown words get the phonetic fallback rather than
    staying Latin — a single Latin word would otherwise be the one fragment
    the Nepali checkpoint cannot voice."""
    out = transliterate_latin("यो refrigerator हो")
    assert not _has_latin(out)


# ── acronyms ──────────────────────────────────────────────────────────────

def test_a_spelled_acronym_is_spelled_letter_by_letter():
    assert transliterate_latin("NEA ले भन्यो") == "एन ई ए ले भन्यो"


def test_an_acronym_said_as_a_word_is_not_spelled_out():
    """spell_acronyms rendered "OK" as "ओ के" — two letters, not a word."""
    assert transliterate_latin("OK भयो") == "ओके भयो"
    assert transliterate_latin("NASA") == "नासा"


# ── proper nouns ──────────────────────────────────────────────────────────

def test_nepali_names_written_in_latin_become_devanagari():
    assert transliterate_latin("Ram र Sita") == "राम र सीता"


def test_place_names_use_their_nepali_form():
    assert transliterate_latin("Kathmandu") == PROPER_NOUNS["kathmandu"]


# ── the user's final say (decision 4) ─────────────────────────────────────

def test_a_pronunciation_override_is_never_transliterated():
    """`[[…]]` is the user's explicit per-occurrence override. The audited
    ordering in services/text_normalization.py keeps the dictionary ABOVE the
    normalizer; transliteration must not quietly invert that."""
    assert transliterate_latin("म [[office]] गएँ") == "म [[office]] गएँ"


def test_markup_stays_parseable():
    """"[pause 300ms]" rewritten to Devanagari stops being a directive and
    starts being spoken aloud."""
    assert transliterate_latin("NEA [pause 300ms] ले") == "एन ई ए [pause 300ms] ले"


# ── contract ──────────────────────────────────────────────────────────────

def test_transliteration_is_idempotent():
    """Every pipeline normalizes more than once (parent, then sidecar); a
    second pass must be a no-op."""
    once = transliterate_latin("म office गएर meeting गरेँ")
    assert transliterate_latin(once) == once


def test_devanagari_input_is_untouched():
    text = "म आज कार्यालय जान्छु।"
    assert transliterate_latin(text) == text


def test_digits_survive_for_the_number_normalizer():
    """normalize_numbers runs BEFORE this; anything numeric it left behind
    (an ID, a version) must not be mangled into letters here."""
    assert "2026" in transliterate_latin("सन् 2026 मा")


def test_every_table_value_is_actually_devanagari():
    """A Latin value in the tables would defeat the whole feature, silently."""
    for table in (LOANWORDS, PROPER_NOUNS, SPOKEN_ACRONYMS):
        for key, value in table.items():
            assert not _has_latin(value), f"{key} → {value}"


def test_no_table_key_is_unreachable():
    """Keys are matched against whitespace-delimited tokens, so a key holding
    a space or an underscore can never fire (caught 8 such entries in review)."""
    for table in (LOANWORDS, PROPER_NOUNS):
        for key in table:
            assert key.isalpha(), f"{key!r} can never match a word token"


# ── reported by ear, 2026-09-21 ───────────────────────────────────────────
# Three words sounded wrong in the first end-to-end listen. Each had a
# different cause, so each gets a different fix — and a test, because a table
# entry is one careless edit away from regressing silently.

def test_laptop_does_not_grow_an_extra_syllable():
    """Without the halanta ("ल्यापटप") the inherent vowel on प makes it
    "lyaa-pa-ta-pa" — four syllables instead of two."""
    assert transliterate_latin("laptop") == "ल्याप्टप"


def test_refrigerator_is_tabled_rather_than_guessed():
    """The rule fallback produced "रेफ्रिगेरटोर". Long, low-frequency words
    are exactly where phonetic rules fail worst — table them."""
    assert transliterate_latin("refrigerator") == "रेफ्रिजेरेटर"
    assert transliterate_latin("fridge") == "फ्रिज"


def test_a_silent_h_in_a_name_is_not_voiced():
    """"John" → "जोह्न" sounded the h. The same fault hit every name of that
    shape (Shah → शह, Singh → सिङह, Sarah → सरह), so the fix is the class,
    not the one word."""
    assert transliterate_latin("John") == "जोन"
    assert transliterate_latin("Shah") == "शाह"
    assert transliterate_latin("Singh") == "सिंह"
    assert transliterate_latin("Sarah") == "सारा"


def test_common_nepali_names_keep_their_long_vowels():
    """The fallback maps every 'a' to the inherent vowel, which is right for
    English ("plan" → प्लन) and wrong for Nepali names ("Rajesh" → रजेश).
    One letter cannot be both, so the names are tabled and the rule is left
    alone — changing it would break the English words it currently gets right.
    """
    assert transliterate_latin("Rajesh") == "राजेश"
    assert transliterate_latin("Prakash") == "प्रकाश"
    # ...and an UNTABLED English word keeps the short vowel the rule gives it
    # ("plan" would not prove this — it is a loanword, so it never reaches the
    # fallback at all).
    assert transliterate_latin("stand") == "स्टन्ड"
