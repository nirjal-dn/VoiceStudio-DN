"""Nepali / Devanagari regression net for the shared text pipeline.

Covers the engine-agnostic pieces every Nepali TTS request passes through:
the long-text chunker (services/chunked_tts.py) and dictation text polish
(services/text_polish.py). Engine-specific normalization lives in
tests/test_xtts_nepali_normalize.py.
"""
import unicodedata

import pytest

from services.chunked_tts import split_text_into_chunks
from services.text_polish import polish_text

DANDA = "।"
VIRAMA = "्"

# Realistic news-style Nepali (conjuncts, chandrabindu, nukta-free).
NEWS = (
    "काठमाडौं महानगरपालिकाले आगामी आर्थिक वर्षका लागि नयाँ बजेट सार्वजनिक गरेको छ। "
    "बजेटमा शिक्षा, स्वास्थ्य र पूर्वाधार विकासलाई प्राथमिकता दिइएको प्रमुख प्रशासकीय "
    "अधिकृतले जानकारी दिनुभयो। स्थानीय जनप्रतिनिधिहरूले विद्यालयका भौतिक संरचना "
    "सुधार्न विशेष कार्यक्रम ल्याउने प्रतिबद्धता व्यक्त गरे। "
)
MIXED = (
    "मेरो Gmail account मा OTP आएन, कृपया IT support टिमलाई खबर गर्नुहोस्। "
    "Zoom meeting भोलि बिहान 10:30 बजे Teams बाट सुरु हुनेछ। "
)
CONJUNCTS = "क्षत्रिय ज्ञान श्रीमान् द्वारा संस्कृत स्वास्थ्य राष्ट्रिय विद्यार्थी शृङ्खला"


def _no_split_syllables(chunks):
    """No chunk may start with a combining mark or follow a trailing virama."""
    for prev, nxt in zip(chunks, chunks[1:]):
        assert unicodedata.category(nxt[0]) not in ("Mn", "Mc"), (prev[-5:], nxt[:5])
        assert not prev.endswith(VIRAMA), (prev[-5:], nxt[:5])


def _covers(chunks, text):
    assert "".join(c.replace(" ", "") for c in chunks) == text.replace(" ", "")


# ── chunker ─────────────────────────────────────────────────────────────────


def test_short_nepali_sentence_is_one_chunk_unchanged():
    text = "नमस्ते, मेरो नाम सीता श्रेष्ठ हो।"
    assert split_text_into_chunks(text, 800) == [text]


def test_conjunct_heavy_text_is_not_altered():
    assert split_text_into_chunks(CONJUNCTS, 800) == [CONJUNCTS]


@pytest.mark.parametrize("max_chars", [120, 250, 400, 800])
def test_long_news_text_splits_only_at_danda(max_chars):
    text = (NEWS * 8).strip()
    chunks = split_text_into_chunks(text, max_chars)
    assert len(chunks) > 1
    assert all(len(c) <= max_chars for c in chunks)
    assert all(c.endswith(DANDA) for c in chunks[:-1]), [c[-10:] for c in chunks]
    _covers(chunks, text)


def test_mixed_nepali_english_keeps_latin_words_and_times_whole():
    text = (MIXED * 10).strip()
    chunks = split_text_into_chunks(text, 200)
    _covers(chunks, text)
    for c in chunks[:-1]:
        assert c.endswith(DANDA)
    assert all("10:3" not in c or "10:30" in c for c in chunks)


def test_very_long_text_without_spaces_never_splits_a_syllable():
    """A pasted run with no spaces or punctuation forces hard cuts; a cut must
    never separate a vowel sign or a conjunct's second consonant."""
    word = "संयुक्ताक्षरहरूक्षत्रज्ञश्रीद्वाराप्रतिष्ठान"
    text = word * 60
    for max_chars in range(8, 400, 7):
        chunks = split_text_into_chunks(text, max_chars)
        assert "".join(chunks) == text
        assert all(len(c) <= max_chars for c in chunks)
        _no_split_syllables(chunks)


def test_nepali_numerals_and_punctuation_survive_chunking():
    text = ("वि.सं. २०८१ साउन १५ गते, मूल्य रु. १,५०,००० भयो! के यो सही हो? " * 30).strip()
    chunks = split_text_into_chunks(text, 300)
    _covers(chunks, text)
    assert all(len(c) <= 300 for c in chunks)


@pytest.mark.parametrize("blank", ["", "   ", "\n\t  \n"])
def test_empty_or_whitespace_text_yields_no_chunks(blank):
    assert split_text_into_chunks(blank, 800) == []


def test_punctuation_only_nepali_tail_is_folded_not_sent_alone():
    text = ("यो वाक्य लामो छ " * 60).strip() + " ।।"
    chunks = split_text_into_chunks(text, 400)
    assert all(any(ch.isalnum() for ch in c) for c in chunks)


# ── dictation polish ────────────────────────────────────────────────────────


@pytest.mark.parametrize(("raw", "polished"), [
    ("मेरो नाम राम हो", "मेरो नाम राम हो।"),
    ("मेरो नाम राम हो।", "मेरो नाम राम हो।"),
    ("के तपाईं आउनुहुन्छ?", "के तपाईं आउनुहुन्छ?"),
    ("  धन्यवाद   सबैलाई  ", "धन्यवाद सबैलाई।"),
    ("रु. १५००", "रु. १५००।"),
])
def test_polish_terminates_devanagari_with_danda(raw, polished):
    assert polish_text(raw) == polished


def test_polish_mixed_text_ending_in_english_uses_latin_stop():
    assert polish_text("मेरो Gmail account") == "मेरो Gmail account."


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_polish_blank_is_empty(blank):
    assert polish_text(blank) == ""


def test_polish_is_idempotent_for_nepali():
    once = polish_text(NEWS)
    assert polish_text(once) == once
    assert not once.endswith(DANDA + DANDA)
