"""Code-switch + punctuation restoration for Devanagari ASR transcripts.

IndicConformer transcribes Nepali speech to *pure Devanagari*, so English words
the speaker actually said come back spelled phonetically in Devanagari
("अकाउन्ट" for "account", "ब्यालेन्स" for "balance"). This module runs the raw
transcript through a small, local, CPU-friendly instruction LLM (llama.cpp /
GGUF) whose ONLY job is to:

  * rewrite clearly-English words that were transcribed phonetically in
    Devanagari back to English Latin script, and
  * add sentence punctuation.

It must NOT translate Nepali, correct grammar, paraphrase, or add/remove words.
Example:  ``मेरो अकाउन्टको ब्यालेन्स चेक गरिदिनुहोस्``
      →   ``मेरो account को balance check गरिदिनुहोस्।``

Design mirrors :mod:`services.refinement` (the sibling dictation-LLM layer):

* **Opt-in, default off** — enabled only via ``OMNIVOICE_CODESWITCH_RESTORE``.
  With it off (the default on every OS) this is an identical no-op pass-through,
  so the cross-platform default-behaviour parity rule holds. The heavy
  ``llama-cpp-python`` dependency is a soft import (extra: ``codeswitch``); a
  missing package or model is a clean pass-through, never an error.
* **Loaded once, reused** — the GGUF model is a lock-guarded module singleton.
* **Deterministic** — temperature 0, fixed seed.
* **Best-effort** — never raises; any failure returns None and the raw
  (Devanagari) transcript stands.
* **Guard-railed** — the LLM output is accepted only if its Devanagari
  codepoints are a subsequence of the input's (see :func:`_devanagari_preserved`).
  That single invariant lets the model Latinize words and split postpositions
  ("अकाउन्टको" → "account को") while making translation, paraphrase, reordering,
  or invention of Nepali text fail closed.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time

logger = logging.getLogger("omnivoice.codeswitch")

# Default model: Qwen2.5-1.5B-Instruct. Chosen for a 16 GB CPU-only host —
# strong instruction-following AND multilingual (incl. Devanagari) at ~1.0 GB
# RAM resident at Q4_K_M, sub-2s on a short dictation utterance. Override the
# repo/file for a bigger model (e.g. Qwen2.5-3B-Instruct-GGUF) or point
# OMNIVOICE_CODESWITCH_MODEL_PATH at a local .gguf to skip the download.
_DEFAULT_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
_DEFAULT_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

# Hard wall-clock budget (s) for one restoration call. Restoration is
# best-effort; a slow/stuck model must never delay a dictation final beyond
# this — the raw transcript ships instead. Env-tunable for slower hardware.
_DEFAULT_TIMEOUT_S = 20.0

# Detection: any Devanagari codepoint (block U+0900–U+097F).
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
# Guard stream: Devanagari *content* only — excludes the danda / double danda
# (U+0964–U+0965), which the model legitimately ADDS as punctuation and which
# must therefore not count against the input-subsequence check.
_DEVANAGARI_CONTENT = re.compile(r"[ऀ-ॣ०-ॿ]")

_SYSTEM_PROMPT = """You transform a Nepali speech-to-text transcript. The transcript is written entirely in Devanagari, but the speaker mixed in English words that were transcribed phonetically in Devanagari. You are a transliteration filter, NOT an assistant and NOT a translator.

Do EXACTLY two things:
1. Rewrite ONLY the words that are clearly English (transcribed phonetically in Devanagari) back into normal English Latin spelling. Example: "अकाउन्ट" is the English word "account"; "ब्यालेन्स" is "balance"; "चेक" is "check".
2. Add appropriate punctuation (commas, and a final . ? or ।).

Strict rules:
- Keep every genuine Nepali word in Devanagari, unchanged and in the same order.
- Do NOT translate Nepali into English.
- Do NOT correct grammar, rephrase, summarize, or reorder anything.
- Do NOT add, remove, or invent words.
- Do NOT change numbers, names, or other content except to Latinize a clearly English word.
- A Nepali postposition attached to an English word may be separated (e.g. "अकाउन्टको" → "account को").
- Make the smallest possible change.
- Output ONLY the transformed transcript on a single line — no quotes, no explanation, no preamble."""

# Few-shot as real chat turns: small models echo examples embedded in the
# system prompt, but honour them as prior turns (same lesson refinement.py
# learned). These are the task's canonical cases.
_EXAMPLES: list[tuple[str, str]] = [
    ("मेरो अकाउन्टको ब्यालेन्स चेक गरिदिनुहोस्",
     "मेरो account को balance check गरिदिनुहोस्।"),
    ("नमस्ते मेरो अकाउन्टको ब्यालेन्स कति छ",
     "नमस्ते, मेरो account को balance कति छ?"),
]

_model = None
_model_lock = threading.Lock()
_load_failed = False  # sticky: don't retry a broken load every utterance


def _enabled() -> bool:
    return os.environ.get("OMNIVOICE_CODESWITCH_RESTORE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _timeout_s() -> float:
    try:
        v = float(os.environ.get("OMNIVOICE_CODESWITCH_TIMEOUT_S", ""))
        if v > 0:
            return v
    except (TypeError, ValueError):
        pass
    return _DEFAULT_TIMEOUT_S


def _has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI.search(text or ""))


def _devanagari_stream(text: str) -> str:
    """Devanagari *content* codepoints of ``text``, in order (danda excluded so
    added punctuation doesn't count against the guard)."""
    return "".join(_DEVANAGARI_CONTENT.findall(text or ""))


def _is_subsequence(needle: str, haystack: str) -> bool:
    it = iter(haystack)
    return all(ch in it for ch in needle)


def _devanagari_preserved(src: str, out: str) -> bool:
    """The anti-hallucination / anti-translation guard. Accept ``out`` only if:

    1. its Devanagari chars are a subsequence of ``src``'s (in order) — so
       paraphrase, reordering, or invented Nepali (which add or reorder
       Devanagari) fail closed, while Latinizing a word or splitting a
       postposition ("अकाउन्टको" → "account को", which only DROPS Devanagari)
       passes; and
    2. it still retains some Devanagari when ``src`` had it — a full Nepali→
       English translation drops all Devanagari and would trivially satisfy (1)
       (∅ is a subsequence of anything), so it is rejected here.

    ponytail: a *partial* translation (drops some but not all Nepali words) can
    still slip past — it's a subsequence and retains some Devanagari. The strict
    prompt + few-shot are the first defence; tighten with a per-token
    alignment check only if that failure mode is observed in practice.
    """
    src_dev = _devanagari_stream(src)
    out_dev = _devanagari_stream(out)
    if src_dev and not out_dev:
        return False
    return _is_subsequence(out_dev, src_dev)


def _model_path() -> str | None:
    """Resolve the local .gguf path, downloading from HF on first use. Returns
    None (clean pass-through) if the runtime/model can't be made available."""
    explicit = os.environ.get("OMNIVOICE_CODESWITCH_MODEL_PATH", "").strip()
    if explicit:
        return explicit if os.path.exists(explicit) else None
    repo = os.environ.get("OMNIVOICE_CODESWITCH_MODEL_REPO", _DEFAULT_REPO).strip()
    fname = os.environ.get("OMNIVOICE_CODESWITCH_MODEL_FILE", _DEFAULT_FILE).strip()
    # Pin the reviewed commit for the curated repo (supply-chain: branch names
    # are mutable). A user-supplied repo has no pinned SHA — follow its default.
    revision = None
    try:
        from services.hf_revisions import CURATED_REVISIONS
        revision = CURATED_REVISIONS.get(repo)
    except Exception:  # noqa: BLE001
        pass
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo, filename=fname, revision=revision)
    except Exception as e:  # noqa: BLE001 — missing model/network is not an error
        logger.warning("code-switch model unavailable (%s) — passing through", e)
        return None


def _get_model():
    """Load the GGUF model once; reuse it forever. None on any failure."""
    global _model, _load_failed
    if _model is not None:
        return _model
    if _load_failed:
        return None
    with _model_lock:
        if _model is not None:
            return _model
        if _load_failed:
            return None
        try:
            from llama_cpp import Llama
        except Exception as e:  # noqa: BLE001 — extra not installed
            logger.warning(
                "llama-cpp-python not installed (%s) — code-switch restoration off. "
                "Install with: uv sync --extra codeswitch", e,
            )
            _load_failed = True
            return None
        path = _model_path()
        if not path:
            _load_failed = True
            return None
        try:
            n_threads = os.cpu_count() or 4
            _model = Llama(
                model_path=path,
                n_ctx=2048,          # dictation utterances are short
                n_threads=n_threads,
                seed=0,              # deterministic
                verbose=False,
            )
            logger.info("code-switch model loaded (%s, %d threads)", path, n_threads)
            return _model
        except Exception as e:  # noqa: BLE001
            logger.warning("code-switch model load failed (%s) — passing through", e)
            _load_failed = True
            return None


def _strip_wrapping(text: str) -> str:
    """Defensive: peel a code fence / surrounding quotes / one-line preamble a
    small model may add despite the prompt. Keep the last non-empty line so a
    leading 'Output:' line is dropped but the transcript survives."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[^\n]*\n?|\n?```$", "", t).strip()
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if lines:
        t = lines[-1].strip()
    if len(t) >= 2 and t[0] in "\"'“‘" and t[-1] in "\"'”’":
        t = t[1:-1].strip()
    return t


def restore_codeswitch(text: str, *, timeout_s: float | None = None) -> str:
    """Run one restoration pass. Raises on model/inference failure — callers
    use :func:`maybe_restore_codeswitch` for the swallow-into-pass-through path.

    ``timeout_s`` bounds the llama.cpp generation via its own deadline hook so a
    stuck decode can't outlive the budget."""
    model = _get_model()
    if model is None:
        raise RuntimeError("code-switch model unavailable")
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for src, dst in _EXAMPLES:
        messages.append({"role": "user", "content": src})
        messages.append({"role": "assistant", "content": dst})
    messages.append({"role": "user", "content": text})

    deadline = time.monotonic() + (timeout_s if timeout_s is not None else _timeout_s())
    res = model.create_chat_completion(
        messages=messages,
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
        seed=0,
        stopping_criteria=_deadline_stop(deadline),
    )
    return _strip_wrapping(res["choices"][0]["message"]["content"])


def _deadline_stop(deadline: float):
    """A llama.cpp StoppingCriteriaList that halts decoding past ``deadline``.
    Returns None if the helper types aren't importable (older llama-cpp)."""
    try:
        from llama_cpp import StoppingCriteriaList
    except Exception:  # noqa: BLE001
        return None

    def _stop(input_ids, logits) -> bool:  # noqa: ARG001
        return time.monotonic() >= deadline

    return StoppingCriteriaList([_stop])


def maybe_restore_codeswitch(text: str, *, timeout_s: float | None = None) -> str | None:
    """Best-effort code-switch + punctuation restoration for a Devanagari final.

    Returns the restored transcript, or None when restoration is off, the text
    has no Devanagari, the runtime/model is unavailable, the guard rejects the
    output, or anything fails — the raw transcript always stands. Never raises.
    Blocking (CPU inference); async callers use :func:`maybe_restore_codeswitch_async`.
    """
    if not text or not text.strip() or not _enabled():
        return None
    if not _has_devanagari(text):
        return None  # nothing phonetically-Devanagari to restore
    try:
        out = restore_codeswitch(text, timeout_s=timeout_s)
    except Exception as e:  # noqa: BLE001 — pass-through is the contract
        logger.warning("code-switch restoration skipped: %s", e)
        return None
    if not out or out == text.strip():
        return None
    if not _devanagari_preserved(text, out):
        logger.warning(
            "code-switch restoration rejected (Devanagari not preserved) — "
            "keeping raw transcript",
        )
        return None
    return out


async def maybe_restore_codeswitch_async(
    text: str, *, timeout_s: float | None = None
) -> str | None:
    """Async, hard-time-bounded restoration for the live-dictation final path.
    Runs :func:`maybe_restore_codeswitch` off-thread under the budget so a slow
    model can never block the ``final`` send. None on timeout/failure."""
    if not text or not text.strip() or not _enabled() or not _has_devanagari(text):
        return None
    budget = timeout_s if timeout_s is not None else _timeout_s()
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(maybe_restore_codeswitch, text, timeout_s=budget),
            timeout=budget + 2.0,  # generation self-bounds; this is the backstop
        )
    except (asyncio.TimeoutError, Exception) as e:  # noqa: BLE001
        logger.warning("code-switch restoration failed/timed out: %s", e)
        return None


# ── Full normalization (indic-conformer-qwen engine) ─────────────────────────
# The strict restore path above is deliberately minimal (Latinize English + add
# punctuation) and its subsequence guard FORBIDS spelling/number edits. The
# `indic-conformer-qwen` engine wants MORE — correct obvious transcription slips
# and normalize spoken numbers too — so it uses this parallel path with a
# broader prompt and a relaxed sanity guard. The model loader/singleton is
# shared; only the prompt and guard differ. Always-on: the ENGINE opting in is
# the switch, not an env var.

_NORMALIZE_SYSTEM_PROMPT = """You normalize a Nepali speech-to-text transcript. It is written entirely in Devanagari; the speaker mixed in English words that were transcribed phonetically in Devanagari, and the recognizer added no punctuation. You are a transcript normalizer, NOT an assistant and NOT a translator.

Do these things, and ONLY these:
1. Rewrite words that are clearly English (transcribed phonetically in Devanagari) back into normal English Latin spelling. Example: "अकाउन्ट" → "account", "ब्यालेन्स" → "balance", "चेक" → "check".
2. Correct obvious spelling/transcription slips in Nepali words (a dropped or duplicated matra, a split word), only when the intended word is unambiguous.
3. Convert spoken number expressions to digits when the meaning is clear: Nepali number words → Devanagari numerals, English number words → Western digits. Leave numbers already written as digits unchanged.
4. Add punctuation (commas, and a final . ? or ।), sentence spacing, and normal capitalization of Latin words.

Strict rules:
- Keep every genuine Nepali word in Devanagari; keep every genuine English word in Latin.
- Do NOT translate Nepali into English or English into Nepali.
- Do NOT paraphrase, summarize, reorder, add, or remove content words.
- Preserve the original meaning and wording. Make the smallest change that reads correctly.
- Output ONLY the normalized transcript on a single line — no quotes, no explanation, no preamble."""

_NORMALIZE_EXAMPLES: list[tuple[str, str]] = [
    ("मेरो अकाउन्टको ब्यालेन्स चेक गरिदिनुहोस्",
     "मेरो account को balance check गरिदिनुहोस्।"),
    ("नमस्ते मेरो अकाउन्टको ब्यालेन्स कति छ",
     "नमस्ते, मेरो account को balance कति छ?"),
    ("मेरो अकाउन्टमा पाँच हजार रुपैयाँ छ",
     "मेरो account मा ५००० रुपैयाँ छ।"),
]

# Relaxed sanity bounds for the normalized output (vs the strict subsequence
# guard). Spelling and number edits legitimately add/drop Devanagari, so the
# subsequence check can't apply here.
# ponytail: heuristic ceiling — a length window + a Devanagari-retention floor.
# It stops the two failure modes that matter (runaway generation, and wholesale
# translation/romanization to English script — the recent "Nepali written in
# Latin" class, which drops Devanagari retention to ~0). The floor is low (0.25)
# on purpose: short code-switch-heavy utterances ("मेरो account", "phone गर") are
# legitimately <50% Devanagari once the English is Latinized, so a higher floor
# would false-reject them. A partial mistranslation can still slip; per-token
# alignment is the upgrade path if that's ever observed.
_NORMALIZE_MIN_RATIO = 0.4
_NORMALIZE_MAX_RATIO = 2.5
_NORMALIZE_MIN_DEVANAGARI_RETENTION = 0.25


def _normalization_sane(src: str, out: str) -> bool:
    """Accept ``out`` as a normalization of ``src`` only if it stays within a
    length window and keeps most of the input's Devanagari content (so a full
    translation to English script fails closed)."""
    src, out = (src or "").strip(), (out or "").strip()
    if src and not out:
        return False
    if not src:
        return False
    ratio = len(out) / len(src)
    if ratio < _NORMALIZE_MIN_RATIO or ratio > _NORMALIZE_MAX_RATIO:
        return False
    src_dev = _devanagari_stream(src)
    if src_dev:
        retained = len(_devanagari_stream(out)) / len(src_dev)
        if retained < _NORMALIZE_MIN_DEVANAGARI_RETENTION:
            return False
    return True


def normalize_transcript(text: str, *, timeout_s: float | None = None) -> str:
    """Run one full-normalization pass (Latinize English + fix obvious spelling +
    spoken-number digits + punctuation). Raises on model/inference failure;
    callers use :func:`maybe_normalize_transcript` for the pass-through path."""
    model = _get_model()
    if model is None:
        raise RuntimeError("normalization model unavailable")
    messages = [{"role": "system", "content": _NORMALIZE_SYSTEM_PROMPT}]
    for src, dst in _NORMALIZE_EXAMPLES:
        messages.append({"role": "user", "content": src})
        messages.append({"role": "assistant", "content": dst})
    messages.append({"role": "user", "content": text})

    deadline = time.monotonic() + (timeout_s if timeout_s is not None else _timeout_s())
    res = model.create_chat_completion(
        messages=messages,
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
        seed=0,
        stopping_criteria=_deadline_stop(deadline),
    )
    return _strip_wrapping(res["choices"][0]["message"]["content"])


def maybe_normalize_transcript(text: str, *, timeout_s: float | None = None) -> str | None:
    """Best-effort full normalization for a Devanagari final. Returns the
    normalized transcript, or None when the text has no Devanagari, the model is
    unavailable, the sanity guard rejects the output, or anything fails — the
    raw transcript always stands. Unlike :func:`maybe_restore_codeswitch` this is
    NOT env-gated: the `indic-conformer-qwen` engine is the opt-in. Never raises.
    """
    if not text or not text.strip():
        return None
    if not _has_devanagari(text):
        return None  # nothing phonetically-Devanagari to normalize
    try:
        out = normalize_transcript(text, timeout_s=timeout_s)
    except Exception as e:  # noqa: BLE001 — pass-through is the contract
        logger.warning("transcript normalization skipped: %s", e)
        return None
    if not out or out == text.strip():
        return None
    if not _normalization_sane(text, out):
        logger.warning(
            "transcript normalization rejected (out of sanity bounds) — "
            "keeping raw transcript",
        )
        return None
    return out


async def maybe_normalize_transcript_async(
    text: str, *, timeout_s: float | None = None
) -> str | None:
    """Async, hard-time-bounded full normalization. Runs
    :func:`maybe_normalize_transcript` off-thread under the budget. None on
    timeout/failure."""
    if not text or not text.strip() or not _has_devanagari(text):
        return None
    budget = timeout_s if timeout_s is not None else _timeout_s()
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(maybe_normalize_transcript, text, timeout_s=budget),
            timeout=budget + 2.0,
        )
    except (asyncio.TimeoutError, Exception) as e:  # noqa: BLE001
        logger.warning("transcript normalization failed/timed out: %s", e)
        return None
