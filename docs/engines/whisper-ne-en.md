# Whisper large-v3 — Nepali + English code-switch (single-pass)

Engine id: `whisper-ne-en` · speech-to-text · CUDA · CPU · 16 kHz

Shown in **Transcription Model Selection** as **Whisper large-v3 — Nepali+English
code-switch (single-pass)**.

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

### Turbo variant (`whisper-ne-en-turbo`)

`whisper-ne-en-turbo` runs the **exact same recipe** on the Whisper large-v3
**Turbo** weights (`deepdml/faster-whisper-large-v3-turbo-ct2`, ~1.6 GB, 0.8B
params) — ~5× faster for quicker dictation at a small accuracy cost. Same forced-
`ne` decode, mixed-script priming, Devanagari-digit mapping, punctuation, and
English/Nepali separation; only the model differs. Pin it the same way:

```bash
export OMNIVOICE_ASR_BACKEND=whisper-ne-en-turbo
```

Every `OMNIVOICE_CS_*` variable below applies to both engines
(`OMNIVOICE_CS_ASR_MODEL` overrides the pinned model on either).

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
3. **Temperature-fallback decoding.** Beam search (`beam_size=best_of=5`) with
   Whisper's standard fallback ladder (`temperature=[0.0, 0.2, … 1.0]`): the
   greedy `0.0` pass is kept for clean audio, but a window that fails the quality
   gates (`compression_ratio_threshold=2.4`, `log_prob_threshold=-1.0`) — the
   dominant Nepali failure mode, a repetition loop or low-confidence output — is
   retried at a higher temperature instead of emitting garbage. This is the main
   Nepali-accuracy knob; set `OMNIVOICE_CS_TEMPERATURE=0.0` to force greedy-only.
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
6. **Spoken number _words_ become digits, in the spoken language's script.**
   Whisper writes numbers as *words* as often as digits, in whichever language
   was spoken — and that language is known from the word itself, so the script
   is decided there, never guessed from the surrounding sentence: English number
   words → Western digits (`nine thousand eight hundred forty-five` → `9845`),
   Nepali number words → Devanagari numerals (`सन्तानब्बे, एकचालिस` → `९७, ४१`).
   A number Whisper already wrote as **bare digits** is left untouched — its
   spoken language is no longer recoverable, so an English amount inside a Nepali
   sentence stays `9845`, never `९८४५`. The `छ` homograph ("six" / the copula
   "is") is only read as `6` right before a scale word (`छ सय` → `600`), so
   ordinary Nepali is left intact. Toggle with `OMNIVOICE_CS_SPOKEN_NUMBERS`.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `OMNIVOICE_CS_LANGUAGE` | `ne` | Whisper decode language. Set `en` to pin English, or any Whisper language code. |
| `OMNIVOICE_CS_BEAM_SIZE` | `5` | Beam size / `best_of`. Lower (e.g. `1`) is faster on CPU at some accuracy cost. |
| `OMNIVOICE_CS_TEMPERATURE` | ladder `0.0,0.2,…,1.0` | Decode temperature. A single float (e.g. `0.0`) forces greedy-only; a comma list customises the fallback ladder. |
| `OMNIVOICE_CS_REPETITION_PENALTY` | `1.1` | Soft penalty (>1.0) against the decode repetition loop that both duplicates the tail (“the last few words repeat”) and, via its overshooting timestamp, truncates long files. Lower to `1.0` to disable; raise (e.g. `1.3`) for stubborn loops. |
| `OMNIVOICE_CS_NO_REPEAT_NGRAM` | `0` (off) | Hard block: forbid any repeated N-token run. Off by default because it can clip genuine Nepali reduplication (`बिस्तारै बिस्तारै`); set `3` only if the soft penalty above leaves loops. |
| `OMNIVOICE_CS_HALLUCINATION_SILENCE_S` | `2.0` | Skip silent gaps longer than this (seconds), where Whisper tends to hallucinate/loop. Applied only on the word-timestamped paths (dub / file), the only place faster-whisper can locate gaps; `0` disables. |
| `OMNIVOICE_CS_SPOKEN_NUMBERS` | `1` (on) | Convert spoken number *words* to digits, each in the script of the language it was spoken in: English words → Western digits (`nine thousand eight hundred forty-five` → `9845`), Nepali words → Devanagari numerals (`सन्तानब्बे` → `९७`). Cardinals only, decided from the number word itself — bare digits Whisper already emitted are preserved, never guessed from surrounding script. A lone/trailing `छ` (also the copula "is") stays a word — it counts as `6` only right before a scale (`छ सय` → `600`). Set `0` to keep Whisper's number words verbatim. |
| `OMNIVOICE_CS_ASR_MODEL` | `Systran/faster-whisper-large-v3` | The CTranslate2 large-v3 repo to load. |

## Limits

Large-v3 code-switches well but is not perfect: a rare English word inside a
Nepali matrix can still be transliterated into Devanagari, and vice-versa. For
production-grade intra-word switching, fine-tune large-v3 on a Nepali–English
code-switched corpus and point `OMNIVOICE_CS_ASR_MODEL` at the converted result
— no code change needed. Pure-Nepali audio where you do **not** need English is
still best served by [IndicConformer](indic-conformer.md).

On a **long file**, if the transcript stops early or the last few words repeat,
that is Whisper's decode repetition loop (a looped segment's overshooting
timestamp makes the decoder seek past the rest of the audio). The default
`OMNIVOICE_CS_REPETITION_PENALTY=1.1` suppresses it; raise it, or enable
`OMNIVOICE_CS_NO_REPEAT_NGRAM=3`, if a specific clip still loops. Note this is a
*decoding* fix, not a speed one: large-v3 on CPU is slow, and a genuinely long
file can exceed the whole-file transcribe guard (`OMNIVOICE_ASR_TRANSCRIBE_TIMEOUT_S`,
default 300 s) — raise that for very long single files, or pick a smaller/Turbo
model.
