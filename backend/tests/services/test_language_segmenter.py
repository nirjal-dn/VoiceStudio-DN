"""Nepali/English span detection — the ten cases the routing layer depends on.

Every assertion here is about a decision :mod:`services.code_switch_tts` makes
downstream: which engine renders which words, and whether any of the user's
text was lost on the way.
"""
from __future__ import annotations

import pytest

from services.language_segmenter import (
    ENGLISH, ENGLISH_VOCABULARY, NEPALI, ROMANIZED_NEPALI, is_mixed,
    language_profile, segment_languages,
)


def _langs(text: str) -> list[str]:
    return [s.language for s in segment_languages(text)]


def _texts(text: str) -> list[str]:
    return [s.text for s in segment_languages(text)]


# ── 1-2: single-language text is one segment, on the right engine ──────────

def test_pure_nepali_is_one_nepali_segment():
    text = "म आज कार्यालय जान्छु।"
    segments = segment_languages(text)
    assert [s.language for s in segments] == [NEPALI]
    assert segments[0].text == text
    assert segments[0].reason == "devanagari"
    assert not is_mixed(text)


def test_pure_english_is_one_english_segment():
    text = "I am going to the office today."
    segments = segment_languages(text)
    assert [s.language for s in segments] == [ENGLISH]
    assert segments[0].text == text
    assert not is_mixed(text)


# ── 3-6: mixed text alternates, and Devanagari never lands on English ──────

def test_one_english_word_inside_nepali_splits_three_ways():
    text = "म आज office जान्छु।"
    assert _langs(text) == [NEPALI, ENGLISH, NEPALI]
    assert _texts(text) == ["म आज ", "office ", "जान्छु।"]
    assert is_mixed(text)


def test_several_english_words_alternate_without_swallowing_devanagari():
    """The spec sketches this as NE→EN→NE→EN→NE; the text actually holds seven
    language runs (``गएर`` and ``सँग`` each sit between two English words).
    Collapsing to five would hand Devanagari to the English engine, which is
    the exact mispronunciation this feature exists to remove — so the shape
    that is pinned is the alternation, with every Devanagari run on Nepali."""
    text = "म आज office गएर manager सँग meeting गर्छु।"
    languages = _langs(text)
    assert languages == [NEPALI, ENGLISH, NEPALI, ENGLISH, NEPALI, ENGLISH, NEPALI]
    assert all(
        segment.language == NEPALI
        for segment in segment_languages(text)
        if any("ऀ" <= ch <= "ॿ" for ch in segment.text)
    )


def test_technical_english_inside_nepali_clause_is_mixed():
    text = "हामीले server restart गरेर database migration complete गर्यौं।"
    assert _langs(text) == [NEPALI, ENGLISH, NEPALI, ENGLISH, NEPALI]
    assert _texts(text)[1] == "server restart "
    assert _texts(text)[3] == "database migration complete "


def test_acronyms_are_english_and_nepali_connectives_are_not():
    text = "GPU र CPU दुवै working छन्।"
    languages = _langs(text)
    assert is_mixed(text)
    assert languages[0] == ENGLISH  # GPU
    assert languages[1] == NEPALI   # र
    assert set(languages) == {ENGLISH, NEPALI}


# ── 7: numbers ride along, they do not become their own span ──────────────

def test_numbers_do_not_become_a_standalone_segment():
    text = "हामीले 25 files upload गरेका छौं।"
    segments = segment_languages(text)
    assert [s.language for s in segments] == [NEPALI, ENGLISH, NEPALI]
    # The digits stay inside the clause that introduced them.
    assert "25" in segments[0].text
    assert not any(s.text.strip() == "25" for s in segments)


@pytest.mark.parametrize("text", [
    "मैले 2026-01-05 मा file पठाएँ।",
    "मेरो email ram@example.com हो।",
    "config.yaml मा setting बदल्नुहोस्।",
    "https://example.com हेर्नुहोस् please।",
])
def test_dates_emails_urls_and_identifiers_never_stand_alone(text):
    segments = segment_languages(text)
    assert all(len(s.text.split()) >= 1 for s in segments)
    # None of these tokens is a segment of its own.
    for token in ("2026-01-05", "ram@example.com", "config.yaml", "https://example.com"):
        assert not any(s.text.strip() == token for s in segments)


# ── 8-9: Romanized Nepali ─────────────────────────────────────────────────

def test_romanized_nepali_is_mostly_nepali():
    text = "ma aja office janchu"
    segments = segment_languages(text)
    assert [s.language for s in segments] == [NEPALI, ENGLISH, NEPALI]
    nepali_words = sum(len(s.text.split()) for s in segments if s.language == NEPALI)
    assert nepali_words > sum(len(s.text.split()) for s in segments if s.language == ENGLISH)


def test_romanized_nepali_alternates_with_english():
    text = "ma aja office janchu because meeting cha"
    assert _langs(text) == [NEPALI, ENGLISH, NEPALI, ENGLISH, NEPALI]


def test_unknown_latin_words_go_to_english_never_to_the_nepali_engine():
    """The Nepali engine cannot pronounce Latin script, so an unlisted Latin
    word is strictly safer on the English one."""
    segments = segment_languages("म आज quarterly reconciliation गर्छु।")
    assert [s.language for s in segments] == [NEPALI, ENGLISH, NEPALI]


# ── 10: nothing is ever lost ──────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "म आज कार्यालय जान्छु।",
    "I am going to the office today.",
    "म आज very happy छु।",
    "  म आज office गएर manager सँग meeting गर्छु।  ",
    "हामीले 25 files upload गरेका छौं।",
    "ma aja office janchu because meeting cha",
    "GPU र CPU दुवै working छन्।",
    "“Quoted” — म भन्छु… ok?!",
    "line one\nline two मा English",
    "\tम\t\tआज\toffice\t",
])
def test_original_text_is_reproduced_exactly(text):
    segments = segment_languages(text)
    assert "".join(s.text for s in segments) == text
    # Segments tile the string with no gaps and no overlaps.
    assert segments[0].start == 0
    assert segments[-1].end == len(text)
    for left, right in zip(segments, segments[1:]):
        assert left.end == right.start
    for segment in segments:
        assert text[segment.start:segment.end] == segment.text


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_nothing_speakable_yields_no_segments(text):
    assert segment_languages(text) == []
    assert not is_mixed(text)


def test_adjacent_same_language_tokens_are_one_segment_not_one_per_word():
    segments = segment_languages("म आज कार्यालय गएर काम गरेँ र घर फर्कें।")
    assert len(segments) == 1


def test_language_profile_summarizes_without_the_text():
    profile = language_profile("म आज office जान्छु।")
    assert profile == {
        "segments": 3,
        "languages": [NEPALI, ENGLISH, NEPALI],
        "counts": {NEPALI: 2, ENGLISH: 1},
    }


def test_the_two_lexicons_do_not_disagree_with_each_other():
    assert not (ENGLISH_VOCABULARY & ROMANIZED_NEPALI)
    assert all(word == word.lower() for word in ENGLISH_VOCABULARY | ROMANIZED_NEPALI)
