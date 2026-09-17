"""Devanagari sentence boundaries in the long-text chunker.

Nepali/Hindi sentences end with the danda (U+0964) or double danda (U+0965),
usually with no space before the next word's break. The chunker must split
there instead of mid-sentence at a comma or the last space.
"""
from services.chunked_tts import split_text_into_chunks

DANDA = "।"
DOUBLE_DANDA = "॥"


def _nepali_text(ender: str) -> str:
    first = ("यो एक लामो नेपाली वाक्य हो जसमा धेरै शब्दहरू छन् " * 12).strip() + ender
    second = ("अर्को वाक्य, पनि लामो छ " * 30).strip() + ender
    return f"{first} {second}"


def test_long_nepali_text_splits_at_danda():
    text = _nepali_text(DANDA)
    chunks = split_text_into_chunks(text, 800)
    assert len(chunks) >= 2
    assert chunks[0].endswith(DANDA)
    assert "".join(c.replace(" ", "") for c in chunks) == text.replace(" ", "")


def test_long_nepali_text_splits_at_double_danda():
    chunks = split_text_into_chunks(_nepali_text(DOUBLE_DANDA), 800)
    assert chunks[0].endswith(DOUBLE_DANDA)


def test_short_nepali_text_is_one_chunk():
    text = "नमस्ते। तपाईंलाई कस्तो छ?"
    assert split_text_into_chunks(text, 800) == [text]
