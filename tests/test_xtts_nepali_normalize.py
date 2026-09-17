"""Nepali text normalization in the xtts-nepali sidecar (stdlib-only module).

Loaded by path: the sidecar runs in its own venv and is not importable as a
package from the app environment.
"""
import importlib.util
import os
from pathlib import Path

import pytest

_MAIN = Path(__file__).resolve().parents[1] / "backend" / "engines" / "xtts_nepali" / "main.py"


@pytest.fixture(scope="module")
def xtts():
    saved = os.environ.get("TORCHAUDIO_USE_TORCHCODEC")
    spec = importlib.util.spec_from_file_location("xtts_nepali_sidecar", _MAIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    if saved is None:
        os.environ.pop("TORCHAUDIO_USE_TORCHCODEC", None)
    else:
        os.environ["TORCHAUDIO_USE_TORCHCODEC"] = saved


@pytest.mark.parametrize(
    ("raw", "spoken"),
    [
        ("1,000", "एक हजार"),
        ("1,500", "एक हजार पाँच सय"),
        ("1,50,000", "एक लाख पचास हजार"),
        ("10,00,000", "दश लाख"),
        ("1,000,000", "दश लाख"),
        ("१,५०,०००", "एक लाख पचास हजार"),
    ],
)
def test_digit_grouping_commas_are_not_decimals(xtts, raw, spoken):
    out = xtts._normalize_nepali_text(f"रु. {raw} तिर्नुहोस्।")
    assert out == f"रु. {spoken} तिर्नुहोस्।"
    assert "दशमलव" not in out


@pytest.mark.parametrize(
    ("raw", "spoken"),
    [
        ("3.14", "तीन दशमलव एक चार"),
        ("3,14", "तीन दशमलव एक चार"),
        ("1,000.5", "एक हजार दशमलव पाँच"),
    ],
)
def test_decimals_still_read_as_decimals(xtts, raw, spoken):
    assert xtts._normalize_nepali_text(raw) == spoken


def test_plain_numbers_and_phones_unchanged(xtts):
    assert xtts._normalize_nepali_text("२०८१ साल") == "दुई हजार एकासी साल"
    assert xtts._normalize_nepali_text("9801234567") == "नौ आठ शून्य एक दुई तीन चार पाँच छ सात"
    assert xtts._normalize_nepali_text("USA मा") == "यू एस ए मा"


# ── Realistic Nepali inputs ─────────────────────────────────────────────────


@pytest.mark.parametrize(("raw", "spoken"), [
    # Dates (ISO and day/month/year)
    ("2024-08-15 मा बैठक बस्छ।", "पन्ध्र अगस्ट दुई हजार चौबीस मा बैठक बस्छ।"),
    ("15/08/2024 देखि लागू", "पन्ध्र अगस्ट दुई हजार चौबीस देखि लागू"),
    # Nepali (Devanagari) numerals
    ("२५ जना विद्यार्थी", "पच्चीस जना विद्यार्थी"),
    ("कक्षा १० को नतिजा", "कक्षा दश को नतिजा"),
    # English numerals in a Nepali sentence
    ("मसँग 3 वटा किताब छन्।", "मसँग तीन वटा किताब छन्।"),
    ("जनसंख्या 30,00,000 भन्दा बढी", "जनसंख्या तीस लाख भन्दा बढी"),
    # Large numbers (lakh / crore)
    ("2,50,00,000", "दुई करोड पचास लाख"),
    # Phone formats
    ("+977 9812345678", "प्लस नौ सात सात नौ आठ एक दुई तीन चार पाँच छ सात आठ"),
    ("01-4412345", "शून्य एक चार चार एक दुई तीन चार पाँच"),
])
def test_realistic_numbers_dates_and_phones(xtts, raw, spoken):
    assert xtts._normalize_nepali_text(raw) == spoken


@pytest.mark.parametrize("text", [
    "राम बहादुर थापा र सीता श्रेष्ठ काठमाडौंमा बस्छन्।",  # proper nouns
    "क्षत्रिय ज्ञान श्रीमान् द्वारा संस्कृत स्वास्थ्य",  # conjuncts
    "के तपाईं भोलि आउनुहुन्छ? हजुर, आउँछु।",  # punctuation
    "Kathmandu University मा admission खुल्यो।",  # mixed, lowercase English
])
def test_text_without_numbers_or_acronyms_is_unchanged(xtts, text):
    assert xtts._normalize_nepali_text(text) == text


def test_mixed_text_spells_acronyms_and_reads_numbers(xtts):
    out = xtts._normalize_nepali_text("NEA ले 2 घण्टा लोडसेडिङ घोषणा गर्‍यो।")
    assert out == "एन ई ए ले दुई घण्टा लोडसेडिङ घोषणा गर्‍यो।"
    assert "‍" in out  # the ZWJ in गर्‍यो survives normalization


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_blank_text_is_left_blank(xtts, blank):
    assert xtts._normalize_nepali_text(blank) == blank


def test_nepali_number_words_cover_every_value_below_a_hundred(xtts):
    words = [xtts._ne_number(n) for n in range(100)]
    assert len(set(words)) == 100  # a distinct Nepali word for each
    assert all(w and not any(ch.isdigit() for ch in w) for w in words)
