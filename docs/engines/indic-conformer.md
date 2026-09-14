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

The ~2.4 GB download happens on the first transcription. It needs no extra packages:
onnxruntime and torch are already in the app environment.

## Use

In **Model Catalogue**, open the ASR tab and choose **Use** on the IndicConformer row. You can
also set `OMNIVOICE_ASR_BACKEND=indic-conformer`.

| Env var | Default | Effect |
|---|---|---|
| `OMNIVOICE_INDIC_CONFORMER_LANG` | `ne` | Language code: `as bn brx doi gu hi kn kok ks mai ml mni mr ne or pa sa sat sd ta te ur` |

## Dictation

While IndicConformer is the selected engine, Dictation uses it, even if a Sherpa dictation model is
also chosen. Each recording is transcribed once, as one complete file, after you press stop. There
are no partial results while you speak, no cut at pauses, and no chunking of long recordings.

## Script

For Devanagari languages (Nepali by default), every character outside the Devanagari block is
removed from the output, so no Latin, Arabic or Hangul text can appear. The model's `|` sentence
mark is written as `।`, and a finished dictation ends with `।` instead of a Latin period.

## Limitations

- Punctuation inside sentences (`?`, `,`) is not produced.
- English words spoken inside Nepali come out in Devanagari (for example, "insurance" as इन्सुरेन्स).
- Runs on CPU only. On an i7-9700 it transcribes about 5× faster than real time and uses ~3 GB of RAM.
  Because the whole recording is encoded in one pass, memory grows with recording length.
