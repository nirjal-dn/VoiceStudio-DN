# Indic Parler-TTS

Engine id: `indic-parler-tts` · subprocess sidecar with its own venv · CPU or CUDA · 44.1 kHz

Runs [`ai4bharat/indic-parler-tts`](https://huggingface.co/ai4bharat/indic-parler-tts) (Apache-2.0).
It speaks 21 languages, including Nepali. The voice comes from a text description, not a
reference clip, so this engine does no voice cloning.

## Setup

parler-tts pins `transformers` 4.46.1 and the app pins 5.3+, so the engine needs its own venv.
Its default location is `~/.omnivoice/engines/indic-parler/.venv`. To use another directory,
set `OMNIVOICE_INDIC_PARLER_DIR` to a directory that contains `.venv`.

```bash
D=~/.omnivoice/engines/indic-parler
uv venv --python 3.11 "$D/.venv"
uv pip install --python "$D/.venv/bin/python" \
  --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu torchaudio==2.8.0+cpu
uv pip install --python "$D/.venv/bin/python" git+https://github.com/huggingface/parler-tts.git
```

On Windows, the interpreter is at `.venv\Scripts\python.exe`.

The model is gated. Accept its terms on the model page, then give VoiceStudio a Hugging Face
token (Settings → API Keys, `HF_TOKEN`, or `hf auth login`). The ~3.8 GB download happens on
the first generate.

## Use

Select **Indic Parler-TTS** in the Engines panel (<kbd>Ctrl</kbd>/<kbd>Cmd</kbd>+<kbd>E</kbd>), or set
`OMNIVOICE_TTS_BACKEND=indic-parler-tts`. Write the text in Devanagari.

The voice description is the request's instruction (the **By design** tab). With no description,
the engine uses the recommended Nepali speaker:
"Amrita speaks with a clear voice at a moderate pace. The recording is of very high quality, with no background noise."
Name a speaker and describe pitch, pace, emotion and recording quality to change the voice.

| Env var | Default | Effect |
|---|---|---|
| `OMNIVOICE_INDIC_PARLER_DESCRIPTION` | Amrita, as above | Description used when a request gives none |
| `OMNIVOICE_INDIC_PARLER_DEVICE` | auto | `cpu` or `cuda` |
| `OMNIVOICE_INDIC_PARLER_RECV_TIMEOUT_S` | `600` | Seconds without a sidecar frame before a hung sidecar is killed |

## Limitations

- No voice cloning.
- Text is rendered one sentence at a time, split at `।`, `?`, `!` and `.`.
- The model is large: about 4 GB of RAM, and slow on CPU.
