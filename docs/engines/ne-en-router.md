# Nepali + English (Auto)

Engine id: `ne-en-router` · speech-to-text · CUDA · CPU · 16 kHz

Transcribes a recording that mixes **Nepali and English** by splitting it into
speech segments, deciding the language of each, and sending each to the model
that transcribes that language best — **IndicConformer** for Nepali (native
Devanagari) and **Whisper large-v2** for English (Latin) — then merging the
pieces back in chronological order. No manual model switching.

Pick **Nepali + English (Auto)** in the transcription engine switch, or pin it:

```bash
export OMNIVOICE_ASR_BACKEND=ne-en-router
```

## Pipeline

```
Audio
  ↓
Whisper large-v2  →  VAD/segmentation + English transcript (one pass)
  ↓
per-segment language ID   (constrained to ne / en only)
  ↓
hysteresis                (don't flip on uncertain segments)
  ↓
┌───────────────┬──────────────────┐
│ Nepali        │ English          │
│ IndicConformer│ Whisper large-v2 │
└───────────────┴──────────────────┘
  ↓
timestamp merge  →  one transcript
```

1. **Segmentation + English ASR** — Whisper large-v2 transcribes the whole clip
   once with its built-in Silero VAD. That single pass provides the speech
   segments (with word timings) *and* the English transcript, so no separate
   segmenter is needed.
2. **Language ID** — each segment's audio is classified with the Whisper
   encoder's language head, **restricted to `ne` and `en`** (every other
   language is dropped and the two renormalised), so speech is never routed
   elsewhere. A segment (2–15 s) gives the classifier far more acoustic context
   than a single word would.
3. **Hysteresis** — a segment only *switches* the running language when its
   confidence clears `OMNIVOICE_NEEN_LID_MARGIN` (default 0.60); an uncertain
   segment inherits the previous language. This prevents the router flapping
   `ne ↔ en` on low-confidence predictions.
4. **Route & merge** — consecutive same-language segments are grouped into spans;
   English spans keep Whisper's text, Nepali spans are re-transcribed by
   IndicConformer, and every span is merged back in start-time order.

Each segment logs which model handled it, e.g.
`ne-en-router [0.00–2.10s] lang=ne (p=0.87) → IndicConformer`.

## When to use this vs. `whisper-ne-en`

| | `ne-en-router` (this) | [`whisper-ne-en`](whisper-ne-en.md) |
| --- | --- | --- |
| Strategy | per-segment routing to two models | one forced-Nepali Whisper pass |
| Best at | languages that alternate between utterances (pause-separated) | code-switching *within* one sentence (loanwords, no pauses) |
| Nepali output | IndicConformer (Devanagari-native) | Whisper Devanagari |

VAD splits on **silence, not language**, so English loanwords spoken mid-sentence
with no pause (`मेरो account को balance`) stay inside a Nepali segment and are
handled by IndicConformer. For heavy intra-sentence code-switching, prefer
`whisper-ne-en`.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `OMNIVOICE_NEEN_WHISPER_MODEL` | `Systran/faster-whisper-large-v2` | Whisper model for the English path, segmentation and LID. Point at an already-downloaded model (e.g. large-v3) to avoid a second download. |
| `OMNIVOICE_NEEN_LID_MARGIN` | `0.60` | Minimum ne/en confidence needed to switch languages between consecutive segments. Higher = fewer switches. |

## Model management

Availability requires only faster-whisper (the English/segmentation/LID path).
IndicConformer is optional: if it is missing or errors on a span, that span keeps
Whisper's transcription — a recording is never lost. Both models stay warm for
low-latency use; on a memory-tight host pin a single engine instead.
