"""Amounts, dates and plain numbers in Nepali TTS normalization.

The governing rule, stated by the user: *say an amount only when the text
carries a currency marker* — रु / Rs / रुपैयाँ / NPR. A bare number is a bare
number. "100 किताब" is a hundred books, not a hundred rupees, and a normalizer
that cannot tell the difference is worse than one that does nothing.
"""
from __future__ import annotations

import pytest

from services.nepali_text import normalize_numbers


# ── the rule: no marker, no money ─────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "मैले 100 किताब किनेँ",
    "म 25 वर्षको",
    "कोठा नम्बर 305",
    "सन् 2026",
    "50 जना आए",
])
def test_a_bare_number_is_never_spoken_as_an_amount(text):
    out = normalize_numbers(text)
    assert "रुपैयाँ" not in out, f"invented a currency in {out!r}"
    assert "पैसा" not in out


# ── markers that used to break ────────────────────────────────────────────

@pytest.mark.parametrize(("text", "expected"), [
    ("Rs 1500", "रु. एक हजार पाँच सय"),
    ("Rs. 500", "रु. पाँच सय"),
    ("NPR 5000", "रु. पाँच हजार"),
    ("₹ 250", "रु. दुई सय पचास"),
])
def test_a_latin_marker_becomes_the_devanagari_one(text, expected):
    """The Devanagari markers already worked and are left exactly as they are
    (रु. is in the sidecar's _NE_ABBREVIATIONS so its period does not end a
    sentence). The LATIN ones did not: "Rs" survived normalization, reached
    the transliterator and came out "र्स", and "NPR" was spelled letter by
    letter — so the spellings a user is most likely to type were exactly the
    broken ones."""
    assert normalize_numbers(text) == expected


@pytest.mark.parametrize("text", ["रु. 1000", "रू. 2,50,000", "रु 500"])
def test_a_devanagari_marker_is_left_where_it_is(text):
    """Six existing tests pin this; the fix must not quietly restyle it."""
    assert normalize_numbers(text).startswith(text[:2])


# ── paisa ─────────────────────────────────────────────────────────────────

def test_two_decimal_places_are_paisa_not_digits():
    """"पैँतालीस दशमलव पाँच शून्य" is how you read a measurement, not money."""
    assert normalize_numbers("रु 45.50") == "रु पैँतालीस पचास पैसा"
    assert normalize_numbers("रु 1200.75") == "रु एक हजार दुई सय पचहत्तर पैसा"


def test_zero_paisa_is_left_unsaid():
    assert normalize_numbers("रु 100.00") == "रु एक सय"


def test_a_non_paisa_fraction_is_not_given_a_denomination():
    """One decimal place is not paisa. Reading it as one would silently turn
    "Rs 1.5" into 50 paisa; a plain decimal is honest about the ambiguity."""
    assert normalize_numbers("Rs 1.5") == "रु. एक दशमलव पाँच"


# ── dates ─────────────────────────────────────────────────────────────────

def test_day_first_is_the_default():
    assert normalize_numbers("15/05/2082") == "दुई हजार बयासी भदौ पन्ध्र गते"


def test_a_first_field_over_twelve_is_read_month_first():
    """"05/15/2026" has no valid day-first reading. It used to fall through
    unparsed, so the SLASHES were spoken aloud."""
    assert normalize_numbers("05/15/2026") == "पन्ध्र मे दुई हजार छब्बीस"


def test_an_impossible_date_is_left_alone():
    """13/13 is not a date in either order. Leaving it beats inventing one."""
    assert "/" in normalize_numbers("13/13/2026")


def test_the_bikram_sambat_range_still_selects_the_nepali_calendar():
    """2050–2200 is BS; outside it the date is Gregorian."""
    assert "भदौ" in normalize_numbers("2082-05-15")
    assert "जनवरी" in normalize_numbers("2026-01-30")


# ── what must not regress ─────────────────────────────────────────────────

@pytest.mark.parametrize(("text", "expected"), [
    ("7:30 बजे", "सात बजेर तीस मिनेट"),
    ("50%", "पचास प्रतिशत"),
    ("3.14", "तीन दशमलव एक चार"),
    ("1,00,000", "एक लाख"),
])
def test_the_other_number_forms_are_unchanged(text, expected):
    assert normalize_numbers(text) == expected


def test_a_phone_number_is_still_read_digit_by_digit():
    assert normalize_numbers("9801234567") == (
        "नौ आठ शून्य एक दुई तीन चार पाँच छ सात")
