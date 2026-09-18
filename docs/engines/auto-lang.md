# Automatic language routing (`auto-lang`)

Engine id: `auto-lang` · ASR **and** TTS · no extra install

Two thin router engines that pick the right model per language automatically, so
you can dictate and synthesize Nepali and English in the same conversation
without switching engines by hand. Pin them once and forget the picker.

## What it does

**ASR — `auto-lang`** (Settings → ASR, or `OMNIVOICE_ASR_BACKEND=auto-lang`)

Each recording is transcribed once by the host's best Whisper engine, which
detects the spoken language for free. When the detected language is Nepali, the
same audio is re-run through [IndicConformer](indic-conformer.md) and its
Devanagari-native transcript is returned instead. English (and every other
non-Indic language) keeps Whisper's single-pass result — only the Nepali path
pays for a second transcription.

Because IndicConformer transcribes a whole recording (no streaming partials),
the router is a whole-recording engine: it transcribes when you stop speaking,
exactly like a pinned IndicConformer. Each recording is detected independently,
so switching languages mid-conversation just works.

**TTS — `auto-lang`** (Settings → Voice, or `OMNIVOICE_TTS_BACKEND=auto-lang`)

Each line is routed by its script: Devanagari text → the Nepali engine
([`xtts-nepali`](xtts-nepali.md), Oshara/XTTS); everything else → the default
engine (`omnivoice`). Routing is per call, so a mixed batch sends each line to
the right engine. A line that mixes Devanagari and Latin goes to the Nepali
engine, which already normalizes mixed text.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `OMNIVOICE_AUTO_ASR_INDIC_LANGS` | `ne` | Comma list of Whisper-detected ISO-639-1 codes that route to IndicConformer (it serves 22 Indic languages, e.g. `ne,hi,mr`). |
| `OMNIVOICE_AUTO_TTS_NEPALI` | `xtts-nepali` | Engine id for Devanagari text. |
| `OMNIVOICE_AUTO_TTS_DEFAULT` | `omnivoice` | Engine id for everything else. |

## Model management

Both routers keep their delegate engines warm for low-latency switching. On a
memory-tight host, pin a single engine instead — two warm models (Whisper +
IndicConformer, or OmniVoice + XTTS) may be too heavy to co-reside.

Availability degrades gracefully: the routers report ready as long as the
default/English engine is available, and fall back to it when the Nepali engine
is missing or errors, so you never lose a recording or a line.
