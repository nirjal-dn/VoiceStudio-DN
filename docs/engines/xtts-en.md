# XTTS v2 (Coqui base)

Engine id: `xtts-en` · subprocess sidecar, shares the `xtts-nepali` venv · CPU or CUDA · 24 kHz

Runs the base [`coqui/XTTS-v2`](https://huggingface.co/coqui/XTTS-v2) checkpoint with zero-shot
voice cloning from a reference clip. Seventeen languages, English first among them.

Its reason for existing is [Nepali + English mixed speech](../nepali.md#nepali--english-mixed-speech):
[`xtts-nepali`](xtts-nepali.md) is a fine-tune of *this* checkpoint, so when the code-switch
service hands both engines the same reference clip, the English half and the Nepali half of one
sentence still sound like the same speaker. It also works on its own as an English cloning engine.

> [!WARNING]
> **Non-commercial weights.** The model carries the
> [Coqui Public Model License](https://coqui.ai/cpml), the same as the Nepali fine-tune. This is a
> local try-out engine; it does not meet the licence bar in [engine-acceptance.md](../engine-acceptance.md).

## Setup

It runs in the **same venv as `xtts-nepali`** (identical dependency pins), so if that engine works,
this one already does — nothing extra to install. Follow
[the xtts-nepali setup](xtts-nepali.md#setup) if you have not yet.

To give it a venv of its own, set `OMNIVOICE_XTTS_EN_DIR` to a directory containing `.venv`.

The ~1.9 GB checkpoint is listed in **Model Catalogue** (XTTS v2 base weights) and pinned to a
reviewed revision. If it is not installed, it downloads from Hugging Face on the first generate and
loads from the cache after that. It is a separate download from the Nepali fine-tune.

## Use

Select **XTTS v2 (Coqui base)** in the Engines panel (<kbd>Ctrl</kbd>/<kbd>Cmd</kbd>+<kbd>E</kbd>),
or set `OMNIVOICE_TTS_BACKEND=xtts-en`. Leave the language on Auto (Auto means English here) or pick
one of the 17 supported languages, and choose a voice to clone. A clean reference clip of 6 s or
longer clones best; without one the engine uses a built-in XTTS speaker. XTTS clones from the audio
alone, so VoiceStudio does not transcribe the reference clip. The **By design** tab's voice
description is not supported: XTTS needs a reference voice. The request seed is honoured.

For mixed Nepali/English text you do not select this engine directly — set the TTS mode to
**Nepali + English mixed** (or leave it on **Automatic**) and VoiceStudio routes each span. See
[Nepali in VoiceStudio](../nepali.md#nepali--english-mixed-speech).

## Settings

Same knobs as the Nepali engine, with an `EN` infix so the two sidecars are tuned independently.

| Env var | Default | Effect |
|---|---|---|
| `OMNIVOICE_XTTS_EN_DIR` | the `xtts-nepali` engine dir | Directory containing the engine's `.venv` |
| `OMNIVOICE_XTTS_EN_MODEL_DIR` | — | Local checkpoint folder. Setting it skips the download |
| `OMNIVOICE_XTTS_EN_DEVICE` | auto | `cpu` or `cuda` |
| `OMNIVOICE_XTTS_EN_SPEAKER` | first built-in | Built-in speaker used when no reference clip is given |
| `OMNIVOICE_XTTS_EN_TEMPERATURE` | `0.7` | Higher is more expressive, lower is steadier |
| `OMNIVOICE_XTTS_EN_REPETITION_PENALTY` | `10.0` | Higher values discourage repeated sounds but flatten intonation |
| `OMNIVOICE_XTTS_EN_TOP_K` | `50` | Sampling pool size |
| `OMNIVOICE_XTTS_EN_TOP_P` | `0.85` | Nucleus sampling threshold |
| `OMNIVOICE_XTTS_EN_RECV_TIMEOUT_S` | `600` | Seconds without a sidecar frame before a hung sidecar is killed |

## Limitations

- Nepali is **not** among its languages — that is what `xtts-nepali` is for. A request that asks
  this engine for `ne` is refused rather than mispronounced.
- The reference clip matters most for tone: XTTS copies the reference's delivery. Use 10–15 s of
  relaxed, conversational speech, not a short or read-aloud clip.
- During a mixed render both this engine and `xtts-nepali` stay resident, so plan for two XTTS
  models in memory rather than one.
- CPU is slow, the same order as the Nepali engine: roughly 5–8× real time on an i7-class CPU, plus
  about 30 s to load the model on the first generate.
