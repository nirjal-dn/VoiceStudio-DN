# XTTS v2 Nepali (Oshara fine-tune)

Engine id: `xtts-nepali` · subprocess sidecar with its own venv · CPU or CUDA · 24 kHz

The legacy request id `oshara-xtts-v2` is accepted as an alias and normalized to
`xtts-nepali`; new clients should use the canonical id.

Runs [`Oshara/xtts-v2-nepali`](https://huggingface.co/Oshara/xtts-v2-nepali), a fine-tune of
Coqui XTTS v2 that adds Nepali (`ne`), with zero-shot voice cloning from a reference clip.
The 17 base XTTS v2 languages remain selectable.

> [!WARNING]
> **Non-commercial weights.** The model inherits the
> [Coqui Public Model License](https://coqui.ai/cpml). This is a local try-out engine;
> it does not meet the licence bar in [engine-acceptance.md](../engine-acceptance.md).

## Setup

coqui-tts pins `transformers` 4.57 and the app pins 5.3+, so the engine needs its own venv.
Its default location is `~/.omnivoice/engines/xtts-nepali/.venv`. To use another directory,
set `OMNIVOICE_XTTS_NEPALI_DIR` to a directory that contains `.venv`.

```bash
D=~/.omnivoice/engines/xtts-nepali
uv venv --python 3.13 "$D/.venv"
# CPU wheels; on an NVIDIA host use https://download.pytorch.org/whl/cu128 and +cu128
uv pip install --python "$D/.venv/bin/python" \
  --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu torchaudio==2.8.0+cpu
uv pip install --python "$D/.venv/bin/python" coqui-tts==0.27.5 transformers==4.57.6
```

On Windows, the interpreter is at `.venv\Scripts\python.exe`.

The ~1.9 GB checkpoint is listed in **Model Catalogue** (XTTS v2 Nepali weights) and pinned to a
reviewed revision. If it is not installed, it downloads from Hugging Face on the first generate and
loads from the cache after that.

Before Nepali synthesis, numbers, dates (Bikram Sambat and Gregorian), clock times, percentages and
phone numbers are converted to Nepali words (see [Nepali in VoiceStudio](../nepali.md)); this happens
for every engine when the language is Nepali. This engine additionally spells short all-capital
acronyms as Nepali letter names (for example, `API` becomes `ए पी आई`), because the Hindi tokenizer
cannot read Latin capitals. For ordinary English words, use the pronunciation
dictionary or an inline override when a specific Nepali pronunciation is needed;
automatic transliteration is intentionally not guessed because English spelling is
not phonetic.

Nepal phone numbers are read digit-by-digit. Supported forms include local mobile
numbers (`9801234567`, `9801 234 567`, `980-123-4567`), international numbers
(`+977 9801234567`), and common Kathmandu landlines (`01-4XXXXXX`). The
international prefix is spoken as `प्लस नौ सात सात`.

## Use

Select **XTTS v2 Nepali** in the Engines panel (<kbd>Ctrl</kbd>/<kbd>Cmd</kbd>+<kbd>E</kbd>),
or set `OMNIVOICE_TTS_BACKEND=xtts-nepali`. Choose Nepali, or leave the language on Auto
(Auto defaults to Nepali), and pick a voice to clone. A clean reference clip of 6 s or longer
clones best. Without a clip, the engine uses a built-in XTTS speaker. XTTS clones from the audio
alone, so VoiceStudio does not transcribe the reference clip for this engine. The **By design** tab's
voice description is not supported: XTTS needs a reference voice. The request seed is honoured, so
the same seed and text reproduce the same take.

## Settings

| Env var | Default | Effect |
|---|---|---|
| `OMNIVOICE_XTTS_NEPALI_CHECKPOINT` | `epoch-20` | Better default for natural Nepali prosody; `epoch-10` matches the standalone Oshara script |
| `OMNIVOICE_XTTS_NEPALI_ROUTE` | `hi` | `hi` sends Nepali as Hindi. `ne` applies the Hindi text cleaners and adds the `[ne]` prefix |
| `OMNIVOICE_XTTS_NEPALI_MODEL_DIR` | — | Local checkpoint folder. Setting it skips the download |
| `OMNIVOICE_XTTS_NEPALI_DEVICE` | auto | `cpu` or `cuda` |
| `OMNIVOICE_XTTS_NEPALI_SPEAKER` | first built-in | Built-in speaker used when no reference clip is given |
| `OMNIVOICE_XTTS_NEPALI_TEMPERATURE` | `0.7` | Higher is more expressive, lower is steadier |
| `OMNIVOICE_XTTS_NEPALI_REPETITION_PENALTY` | `10.0` | Higher values discourage repeated sounds but flatten intonation |
| `OMNIVOICE_XTTS_NEPALI_TOP_K` | `50` | Sampling pool size |
| `OMNIVOICE_XTTS_NEPALI_TOP_P` | `0.85` | Nucleus sampling threshold |
| `OMNIVOICE_XTTS_NEPALI_RECV_TIMEOUT_S` | `600` | Seconds without a sidecar frame before a hung sidecar is killed |

## Limitations

- Stock coqui-tts has no Nepali text cleaners, and the vocab has no `[ne]` token. The `ne` route
  speaks a stray "ne" syllable before the text, so the default route is `hi`.
- The defaults were tuned for a natural, conversational tone. Pitch variation (the model card's
  expressiveness measure) rose from 2.42 to 3.74 semitones at the same 3% round-trip character
  error rate as the old `hi` / 0.7 / 10.0 settings. Natural speech is about 4.95.
- The reference clip matters most for tone: XTTS copies the reference's delivery. Use 10–15 s of
  relaxed, conversational Nepali speech, not a short or read-aloud clip.
- Text is split into sentences at the danda (`।`, `॥`), `?`, `!` and `.`, with a 0.2 s gap between
  pieces. A period after a Nepali abbreviation (`रु.`, `डा.`, `नं.`, `वि.सं.`) does not end a sentence.
  A sentence longer than the model's token limit is split at word boundaries, and a single over-long
  word at syllable boundaries (a conjunct or vowel sign is never cut off).
- English words inside Nepali text are left as written (English spelling is not phonetic). Use the
  pronunciation dictionary to give a word a Nepali spelling.
- Pitch is flatter than natural speech (see the model card).
- CPU is slow. On an i7-9700, a warm engine took about 50–80 s to render 10.5 s of audio
  (5–8× real time). The first generate adds roughly 30 s to load the model.
