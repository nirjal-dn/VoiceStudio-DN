# Whisper large-v3 — Nepali + English code-switch (single-pass)

Engine id: `whisper-ne-en` · speech-to-text · CUDA · CPU · 16 kHz

Shown in **Transcription Model Selection** as **Whisper large-v3 — Nepali+English
code-switch (single-pass)**. For per-segment routing between IndicConformer and
Whisper, see **[Nepali + English (Auto)](ne-en-router.md)** instead.

Transcribes **code-switched Nepali–English speech** — a single sentence that
mixes both languages, like:

> नमस्ते, मेरो account को balance कति छ?

in one pass, keeping Nepali in **Devanagari** and English in **Latin**. It is a
thin specialisation of [Faster-Whisper](faster-whisper.md) (same CTranslate2
`large-v3` weights, same device/OOM fallback), so it needs **no extra install**
and appears in the ASR list once faster-whisper is present (it is, by default).

Enable it in **Model Catalogue → transcription**, or pin it:

```bash
export OMNIVOICE_ASR_BACKEND=whisper-ne-en
```

## How it decodes (the forced-Nepali recipe)

Whisper saw far more Hindi than Nepali, and the two share Devanagari and much
phonology — so plain auto-detect rarely scores `ne` on top for Nepali speech,
and picking the higher of `ne`/`en` lands on **English** for any clip where
English is acoustically prominent. Decoding that clip as `en` then **romanises
or drops the Nepali** — the "Nepali is never detected, English works fine"
failure. Decoding as `ne`, by contrast, keeps the English words in Latin **and**
the Nepali in Devanagari, and does no harm to clean English audio.

So the engine does not detect at all — it **forces `language="ne"`** for every
recording. This is the recipe validated in the reference notebook
(`STT_openai_Whisper`):

1. **`language="ne"`, forced.** Nepali in Devanagari, English kept in Latin, no
   drift to Hindi. Works for pure Nepali, pure English, and mixed input alike.
2. **`task="transcribe"`, always.** Never `translate` — neither language is
   rendered into the other.
3. **Deterministic decoding.** `temperature=0.0` with beam search
   (`beam_size=best_of=5`) for stable, reproducible dictation.
4. **Built-in Silero VAD** (`vad_filter=True`, `min_silence_duration_ms=500`)
   drops non-speech, and `condition_on_previous_text=False` stops one window's
   bias bleeding into the next. The result is **one continuous transcript**
   (top-level `text`), never split by language or script.
5. **Whole-recording single segment.** Whisper's VAD splits speech into
   per-utterance segments internally (that is how it drops silence), but they
   are merged back before returning — a dictated recording comes out as **one
   segment spanning start→stop**, not a timestamped list of utterances.
   Consumers that need per-utterance timings (e.g. dubbing lip-sync) use the
   plain [Faster-Whisper](faster-whisper.md) engine instead.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `OMNIVOICE_CS_LANGUAGE` | `ne` | Whisper decode language. Set `en` to pin English, or any Whisper language code. |
| `OMNIVOICE_CS_BEAM_SIZE` | `5` | Beam size / `best_of`. Lower (e.g. `1`) is faster on CPU at some accuracy cost. |
| `OMNIVOICE_CS_ASR_MODEL` | `Systran/faster-whisper-large-v3` | The CTranslate2 large-v3 repo to load. |

## Limits

Large-v3 code-switches well but is not perfect: a rare English word inside a
Nepali matrix can still be transliterated into Devanagari, and vice-versa. For
production-grade intra-word switching, fine-tune large-v3 on a Nepali–English
code-switched corpus and point `OMNIVOICE_CS_ASR_MODEL` at the converted result
— no code change needed. Pure-Nepali audio where you do **not** need English is
still best served by [IndicConformer](indic-conformer.md).
