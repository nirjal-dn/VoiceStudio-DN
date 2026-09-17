"""Nepali as a registered language: registry resolution, script checks, and the
speech normalizer every TTS engine receives through normalize_for_tts."""
import pytest

from services import languages, nepali_text
from services.text_normalization import normalize_for_tts, normalize_text


# ── registry ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", ["Nepali", "nepali", "ne", "NE", "npi", "nep", "ne-NP", "ne_NP", "नेपाली"])
def test_every_spelling_of_nepali_resolves(value):
    lang = languages.resolve(value)
    assert lang is not None and lang.code == "ne"
    assert lang.flores == "npi_Deva" and lang.script == "Devanagari"
    assert languages.iso_code(value) == "ne"


@pytest.mark.parametrize("value", [None, "", "Auto", "auto", "English", "hi", "xx-YY"])
def test_other_values_are_not_nepali(value):
    assert languages.resolve(value) is None


@pytest.mark.parametrize(("value", "code"), [
    ("English", "en"), ("Hindi", "hi"), ("Maithili", "mai"), ("en", "en"), ("pt-BR", "pt"),
    ("Auto", None), ("", None), ("Klingon Galaxy", None),
])
def test_iso_code_passthrough_for_unregistered_languages(value, code):
    assert languages.iso_code(value) == code


@pytest.mark.parametrize(("text", "matches"), [
    ("मेरो नाम सीता श्रेष्ठ हो।", True),
    ("मेरो Gmail account मा OTP आएन।", True),       # mostly Devanagari letters
    ("Hello, my name is Sita Shrestha.", False),    # Latin transcript
    ("mero naam sita ho", False),                   # romanized Nepali
    ("१२३ ४५६ !!", True),                           # no letters: no evidence
])
def test_script_check_for_nepali(text, matches):
    assert languages.text_matches_language(text, "Nepali") is matches


def test_script_check_is_permissive_for_unregistered_languages():
    assert languages.text_matches_language("नमस्ते", "English")
    assert languages.text_matches_language("hello", None)


# ── Nepali speech normalizer ────────────────────────────────────────────────


@pytest.mark.parametrize(("raw", "spoken"), [
    # Bikram Sambat dates (year 2050–2200) use Nepali months
    ("वि.सं. २०८१-०१-०१ मा नयाँ वर्ष", "वि.सं. दुई हजार एकासी बैशाख एक गते मा नयाँ वर्ष"),
    ("2081/04/10 गते", "दुई हजार एकासी साउन दश गते"),
    ("2081-12-30गते", "दुई हजार एकासी चैत तीस गते"),
    # Gregorian dates keep Gregorian months
    ("2024-08-15", "पन्ध्र अगस्ट दुई हजार चौबीस"),
    ("25/12/2024 मा क्रिसमस", "पच्चीस डिसेम्बर दुई हजार चौबीस मा क्रिसमस"),
    # Times
    ("बिहान 7:05 बजे", "बिहान सात बजेर पाँच मिनेट"),
    ("१०:३० मा बैठक", "दश बजेर तीस मिनेट मा बैठक"),
    ("9:00", "नौ बजे"),
    # Percentages and decimals
    ("50% छुट", "पचास प्रतिशत छुट"),
    ("ब्याजदर 7.5% भयो", "ब्याजदर सात दशमलव पाँच प्रतिशत भयो"),
    # Large amounts in Nepali grouping
    ("रु. 10,00,00,000", "रु. दश करोड"),
    ("1,00,00,00,000", "एक अर्ब"),
    # Suffixes attached to numbers stay separate words
    ("५जना विद्यार्थी", "पाँच जना विद्यार्थी"),
    ("२०८१सालमा", "दुई हजार एकासी सालमा"),
    # Leading-zero codes and long IDs are read digit by digit
    ("कोड 007", "कोड शून्य शून्य सात"),
    ("खाता 12345678901234", "खाता एक दुई तीन चार पाँच छ सात आठ नौ शून्य एक दुई तीन चार"),
])
def test_normalize_numbers(raw, spoken):
    assert nepali_text.normalize_numbers(raw) == spoken


@pytest.mark.parametrize("text", [
    "COVID19 खोप", "mp3 फाइल", "1st position", "v2.0.1 संस्करण", "3:2 अनुपात", "13:99",
    "राम बहादुर थापा", "Kathmandu University", "", "   \n",
])
def test_codes_names_and_blank_text_are_left_alone(text):
    assert nepali_text.normalize_numbers(text) == text


@pytest.mark.parametrize("text", [
    "वि.सं. २०८१-०५-१५ गते बिहान 10:30 बजे रु. 1,50,000 भुक्तानी, 50% छुट, फोन 9812345678।",
    "मूल्य 3.14 र 2024-08-15, कोड 007",
])
def test_normalizer_is_idempotent_and_removes_digits(text):
    once = nepali_text.normalize_numbers(text)
    assert nepali_text.normalize_numbers(once) == once
    assert not any(ch.isdigit() for ch in once)


def test_number_words_scales():
    assert nepali_text.number_words(0) == "शून्य"
    assert nepali_text.number_words(100) == "एक सय"
    assert nepali_text.number_words(1_05_000) == "एक लाख पाँच हजार"
    assert nepali_text.number_words(2_00_00_00_00_000) == "दुई खर्ब"
    assert nepali_text.number_words(-5) == "माइनस पाँच"


# ── integration: every engine gets it through normalize_for_tts ─────────────


@pytest.mark.parametrize("language", ["Nepali", "ne", "ne-NP"])
def test_normalize_for_tts_speaks_nepali_numbers(language, monkeypatch):
    monkeypatch.setenv("OMNIVOICE_TEXT_NORMALIZATION", "1")
    out = normalize_for_tts("आज 2081-05-15 गते 3 जना आए।", language)
    assert out == "आज दुई हजार एकासी भदौ पन्ध्र गते तीन जना आए।"


def test_nepali_normalization_leaves_markup_and_pronunciation_overrides(monkeypatch):
    monkeypatch.setenv("OMNIVOICE_TEXT_NORMALIZATION", "1")
    out = normalize_for_tts("[pause 300ms] 5 जना [[2081|दुई हजार एकासी]] आए।", "Nepali")
    assert out.startswith("[pause 300ms] पाँच जना")
    assert "[[2081|दुई हजार एकासी]]" in out


def test_other_languages_are_unchanged_by_the_nepali_rules():
    assert normalize_text("Room 101 on 2024-08-15", "Hindi") == "Room 101 on 2024-08-15"
    assert normalize_text("५जना", None) == "५जना"


def test_normalization_can_be_disabled(monkeypatch):
    monkeypatch.setenv("OMNIVOICE_TEXT_NORMALIZATION", "0")
    assert normalize_for_tts("3 जना", "Nepali") == "3 जना"


def test_normalize_for_tts_never_raises(monkeypatch):
    monkeypatch.setenv("OMNIVOICE_TEXT_NORMALIZATION", "1")

    def boom(text):
        raise RuntimeError("bad rule")

    # Patch the module text_normalization resolves at call time: a full-suite
    # run can re-import services.* and leave this file's alias stale.
    import sys

    import services.nepali_text  # noqa: F401 — ensure it is in sys.modules

    monkeypatch.setattr(sys.modules["services.nepali_text"], "normalize_numbers", boom)
    assert normalize_for_tts("3 जना", "Nepali") == "3 जना"


def test_xtts_sidecar_uses_the_shared_normalizer():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "backend" / "engines" / "xtts_nepali" / "main.py"
    spec = importlib.util.spec_from_file_location("xtts_sidecar_shared", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._nepali.normalize_numbers("50%") == nepali_text.normalize_numbers("50%")
    assert module._normalize_nepali_text("NEA 5%") == "एन ई ए पाँच प्रतिशत"
