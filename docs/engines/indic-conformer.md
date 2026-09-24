# IndicConformer 600M

Engine id: `indic-conformer` · speech-to-text · in-process on CPU · 16 kHz

Runs [`ai4bharat/indic-conformer-600m-multilingual`](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
(MIT). It transcribes 22 Indian languages, including Nepali, in the language's native script
(Devanagari for Nepali), with word timestamps.

## Setup

The model is gated on Hugging Face. To download it:

1. Sign in on the model page and accept its terms.
2. Give VoiceStudio a Hugging Face read token, in any of these ways:
   - **Settings → API Keys**
   - `HF_TOKEN` in the project `.env` (loaded at startup, and ignored by git)
   - `hf auth login`

Then install **IndicConformer 600M** from **Model Catalogue** (~2.4 GB, pinned to a reviewed
revision). Until the weights are installed, transcription and dictation answer with the same
"download this model" prompt as the other speech engines instead of starting a download on their
own. It needs no extra packages: onnxruntime and torch are already in the app environment.

## Use

In **Model Catalogue**, open the ASR tab and choose **Use** on the IndicConformer row. You can
also set `OMNIVOICE_ASR_BACKEND=indic-conformer`.

The language comes from the request when it names one of the supported languages: the Dubbing
source language, the `language` field of `POST /transcribe` and `/v1/audio/transcriptions`, and the
voice's language when a clone reference is transcribed. A request for a language the model does not
cover (for example English) fails with a message naming the supported languages, instead of being
transcribed with the Nepali vocabulary. Otherwise the default below applies.

| Env var | Default | Effect |
|---|---|---|
| `OMNIVOICE_INDIC_CONFORMER_LANG` | `ne` | Default language code: `as bn brx doi gu hi kn kok ks mai ml mni mr ne or pa sa sat sd ta te ur` |

## Dictation

While IndicConformer is the selected engine, Dictation uses it, even if a Sherpa dictation model is
also chosen. Each recording is transcribed after you press stop; there are no partial results while
you speak. Recordings up to 90 s are transcribed in one pass; longer audio is cut at its quietest
moments into passes of at most 90 s, with word timestamps kept on the full timeline.

## Script

For Devanagari languages (Nepali by default), every character outside the Devanagari block is
removed from the output, so no Latin, Arabic or Hangul text can appear. The model's `|` sentence
mark is written as `।`, and a finished dictation ends with `।` instead of a Latin period.

## IndicConformer + Qwen normalization (`indic-conformer-qwen`)

A companion engine runs IndicConformer, then **automatically** normalizes the raw
Devanagari transcript with a small local LLM — no manual step. The pipeline is
`Audio → IndicConformer → Qwen normalization → final transcript`. It fixes the
first two limitations below in one pass:

- **Latinizes code-switched English** — `अकाउन्ट` → `account`, `ब्यालेन्स` → `balance`.
- **Corrects obvious transcription slips** in Nepali words.
- **Digitizes spoken numbers** — Nepali number words → Devanagari numerals,
  English number words → Western digits.
- **Adds punctuation, spacing, and Latin capitalization.**

Genuine Nepali stays in Devanagari; nothing is translated. Normalization is
**best-effort**: if the LLM is unavailable or its output fails the sanity guard
(runaway length, or dropping most of the input's Devanagari — the anti-translation
check), the raw IndicConformer transcript stands. The raw text is kept in the
result's `raw_text` and logged for debugging.

**Setup**: needs the IndicConformer weights (above) **plus** the Qwen normalizer
and its runtime — install the `codeswitch` extra (`uv sync --extra codeswitch`,
which brings `llama-cpp-python`) and the **Qwen2.5-1.5B Instruct Q4_K_M** GGUF
(~1 GB, downloaded on first use, pinned to a reviewed revision). Until both are
present the engine reports itself unavailable with the install hint; plain
`indic-conformer` keeps working without them. Reuses the same normalization
module as the opt-in [codeswitch-restore](codeswitch-restore.md) layer, so its
`OMNIVOICE_CODESWITCH_MODEL_REPO` / `_FILE` / `_PATH` and
`OMNIVOICE_CODESWITCH_TIMEOUT_S` overrides apply here too.

Select **IndicConformer + Qwen normalization** in **Model Catalogue → ASR**, or
`OMNIVOICE_ASR_BACKEND=indic-conformer-qwen`.

## Limitations

- Punctuation inside sentences (`?`, `,`) is not produced. *(The
  `indic-conformer-qwen` engine above adds it.)*
- English words spoken inside Nepali come out in Devanagari (for example, "insurance" as इन्सुरेन्स).
  *(The `indic-conformer-qwen` engine above Latinizes them.)*
- Runs on CPU only. On an i7-9700 it transcribes about 5× faster than real time and uses ~3 GB of RAM,
  plus roughly 0.4 GB for a 30 s pass and 1 GB for a 120 s pass (measured); passes are capped at 90 s.
- Uploads at other sample rates are resampled to 16 kHz with an anti-aliasing filter; stereo is mixed
  to mono.
