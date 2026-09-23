"""Code-switch + punctuation restoration (``services.codeswitch_restore``).

No model/weights here: the llama.cpp GGUF call is monkeypatched to canned
outputs, so these pin the *contract* around it — gating, the Devanagari-
preservation guard, output cleanup, and the pass-through-on-anything path.
The strict-rule behaviour (Latinize only English words, keep Nepali, add
punctuation, don't translate/paraphrase) is enforced by the guard, which is
what these tests exercise across the required cases: Nepali-only, English
code-switch, punctuation, numbers, names, and mixed sentences.
"""
from __future__ import annotations

import pytest

from services import codeswitch_restore as cs


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setenv("OMNIVOICE_CODESWITCH_RESTORE", "1")


def _fake_llm(monkeypatch, output: str):
    """Make restore_codeswitch() return ``output`` without a model."""
    monkeypatch.setattr(cs, "restore_codeswitch", lambda text, **kw: cs._strip_wrapping(output))


# ── pure helpers ─────────────────────────────────────────────────────────────

def test_has_devanagari():
    assert cs._has_devanagari("मेरो account")
    assert not cs._has_devanagari("my account 123")
    assert not cs._has_devanagari("")


def test_devanagari_stream_keeps_only_devanagari_in_order():
    assert cs._devanagari_stream("मेरो account को") == "मेरोको"


def test_subsequence_allows_postposition_split():
    # "अकाउन्टको" → "account को": only DROPS Devanagari, "को" stays in order.
    assert cs._devanagari_preserved("अकाउन्टको ब्यालेन्स", "account को balance") is True


def test_guard_rejects_full_translation():
    # Nepali fully translated to English → all Devanagari dropped → rejected.
    assert cs._devanagari_preserved("मेरो खाता", "my account") is False


def test_guard_rejects_invented_nepali():
    # A Nepali word not present in the source (paraphrase / hallucination).
    assert cs._devanagari_preserved("मेरो account", "मेरो नयाँ account") is False


def test_guard_rejects_reordered_nepali():
    assert cs._devanagari_preserved("पहिलो दोस्रो", "दोस्रो पहिलो") is False


def test_guard_accepts_unchanged_nepali_with_punctuation():
    assert cs._devanagari_preserved("नमस्ते मेरो नाम", "नमस्ते, मेरो नाम।") is True


def test_strip_wrapping_peels_fence_quotes_and_preamble():
    assert cs._strip_wrapping("```\nमेरो account\n```") == "मेरो account"
    assert cs._strip_wrapping('"मेरो account।"') == "मेरो account।"
    assert cs._strip_wrapping("Output:\nमेरो account।") == "मेरो account।"


# ── orchestration: maybe_restore_codeswitch ──────────────────────────────────

def test_disabled_is_passthrough(monkeypatch):
    monkeypatch.delenv("OMNIVOICE_CODESWITCH_RESTORE", raising=False)
    _fake_llm(monkeypatch, "account को")
    assert cs.maybe_restore_codeswitch("अकाउन्ट को") is None


def test_no_devanagari_skips_llm(monkeypatch):
    # Pure-Latin input: nothing phonetically-Devanagari to restore; LLM not called.
    def _boom(*a, **k):
        raise AssertionError("LLM must not run on non-Devanagari input")
    monkeypatch.setattr(cs, "restore_codeswitch", _boom)
    assert cs.maybe_restore_codeswitch("hello world") is None


def test_english_codeswitch_latinized(monkeypatch):
    src = "मेरो अकाउन्टको ब्यालेन्स चेक गरिदिनुहोस्"
    _fake_llm(monkeypatch, "मेरो account को balance check गरिदिनुहोस्।")
    out = cs.maybe_restore_codeswitch(src)
    assert out == "मेरो account को balance check गरिदिनुहोस्।"


def test_punctuation_added_question(monkeypatch):
    src = "नमस्ते मेरो अकाउन्टको ब्यालेन्स कति छ"
    _fake_llm(monkeypatch, "नमस्ते, मेरो account को balance कति छ?")
    out = cs.maybe_restore_codeswitch(src)
    assert out.endswith("?") and "account" in out and "नमस्ते" in out


def test_nepali_only_keeps_devanagari(monkeypatch):
    # No English words — only punctuation should be added, all Devanagari kept.
    src = "नमस्ते मेरो नाम राम हो"
    _fake_llm(monkeypatch, "नमस्ते, मेरो नाम राम हो।")
    out = cs.maybe_restore_codeswitch(src)
    assert out == "नमस्ते, मेरो नाम राम हो।"


def test_nepali_only_translation_attempt_rejected(monkeypatch):
    # If the model wrongly translates a Nepali-only line, the guard rejects it
    # and the raw transcript stands (None → caller keeps `text`).
    src = "मेरो नाम राम हो"
    _fake_llm(monkeypatch, "My name is Ram.")
    assert cs.maybe_restore_codeswitch(src) is None


def test_numbers_preserved(monkeypatch):
    # Numbers (ASCII digits) must not change; Nepali stays; English Latinized.
    src = "मेरो अकाउन्टमा 5000 रुपैयाँ छ"
    _fake_llm(monkeypatch, "मेरो account मा 5000 रुपैयाँ छ।")
    out = cs.maybe_restore_codeswitch(src)
    assert "5000" in out and "रुपैयाँ" in out and "account" in out


def test_name_in_devanagari_preserved(monkeypatch):
    src = "काठमाडौंमा मेरो मिटिङ छ"
    _fake_llm(monkeypatch, "काठमाडौंमा मेरो meeting छ।")
    out = cs.maybe_restore_codeswitch(src)
    assert "काठमाडौं" in out and "meeting" in out


def test_mixed_sentence(monkeypatch):
    src = "आज मैले नयाँ फोन किन्ने प्लान गरेको छु"
    # "फोन"→phone, "प्लान"→plan; Nepali (आज मैले नयाँ किन्ने गरेको छु) kept.
    _fake_llm(monkeypatch, "आज मैले नयाँ phone किन्ने plan गरेको छु।")
    out = cs.maybe_restore_codeswitch(src)
    assert "phone" in out and "plan" in out and "नयाँ" in out


def test_paraphrase_rejected_end_to_end(monkeypatch):
    src = "मेरो अकाउन्ट खोल"
    # Model paraphrases — introduces a Nepali word ("कृपया") not in source.
    _fake_llm(monkeypatch, "कृपया मेरो account खोल्नुहोस्।")
    assert cs.maybe_restore_codeswitch(src) is None


def test_llm_failure_is_passthrough(monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("model exploded")
    monkeypatch.setattr(cs, "restore_codeswitch", _raise)
    assert cs.maybe_restore_codeswitch("मेरो account") is None


def test_identical_output_returns_none(monkeypatch):
    # Model returned the input unchanged → nothing to surface.
    src = "मेरो नाम"
    _fake_llm(monkeypatch, "मेरो नाम")
    assert cs.maybe_restore_codeswitch(src) is None


@pytest.mark.asyncio
async def test_async_wrapper(monkeypatch):
    src = "मेरो अकाउन्ट"
    _fake_llm(monkeypatch, "मेरो account।")
    out = await cs.maybe_restore_codeswitch_async(src)
    assert out == "मेरो account।"
