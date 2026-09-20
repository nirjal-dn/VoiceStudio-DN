# Nepali in VoiceStudio

Nepali (`ne`) is a registered language (`backend/services/languages.py`): every module resolves
"Nepali", `ne`, `npi`, `nep` and `ne-NP` to the same language and gets the same behaviour.

## Engines

| Task | Engine | Notes |
|---|---|---|
| Voice clone (TTS) | [XTTS v2 Nepali](engines/xtts-nepali.md) | Clones from a 10–15 s reference clip; non-commercial weights |
| English inside Nepali (TTS) | [XTTS v2 base](engines/xtts-en.md) | The English half of mixed speech; shares the Nepali engine's venv |
| Voice design (TTS) | [Indic Parler-TTS](engines/indic-parler-tts.md) | Voice from a text description |
| Speech-to-text, dictation | [IndicConformer](engines/indic-conformer.md) | Devanagari output, word timestamps, CPU |
| Translation (Dub) | NLLB (`npi_Deva`), Google, DeepL, LLM | LLM output is rejected unless it is in Devanagari |

All three models are in **Model Catalogue** and pinned to reviewed revisions. XTTS and Parler also
need their own venvs (see each engine page). While a Nepali engine is active, the language picker
switches from Auto to Nepali; switching to an engine without a default language switches it back.

## Text for speech

When the request language is Nepali, text is normalized before any engine sees it
(`backend/services/nepali_text.py`). Devanagari and ASCII digits are both handled.

| Written | Spoken |
|---|---|
| `रु. 1,50,000` | रु. एक लाख पचास हजार |
| `10,00,00,000` | दश करोड |
| `3.5` | तीन दशमलव पाँच |
| `50%` | पचास प्रतिशत |
| `२०८१-०४-१५ गते` (year 2050–2200: Bikram Sambat) | दुई हजार एकासी साउन पन्ध्र गते |
| `2024-08-15`, `15/08/2024` | पन्ध्र अगस्ट दुई हजार चौबीस |
| `10:30 बजे` | दश बजेर तीस मिनेट |
| `९८४१२३४५६७`, `+977 9812345678`, `01-4412345` | read digit by digit |
| `५जना` | पाँच जना |
| `कोड 007`, 14+ digit IDs | read digit by digit |

Left unchanged on purpose: English words and names, codes that mix letters and digits (`COVID19`,
`mp3`, `1st`, `v2.0.1`), ratios and invalid times (`3:2`, `13:99`). Use the pronunciation
dictionary for words that need a specific reading. Normalization can be turned off with
`OMNIVOICE_TEXT_NORMALIZATION=0`.

Long text is split at the danda (`।`, `॥`) before `?`/`!`/`.`, commas and spaces; a forced cut
never separates a vowel sign or conjunct from its consonant. Whitespace-only text is rejected with
`400 Text is empty`.

## Nepali + English mixed speech

Everyday Nepali is full of English: `म आज office गएर manager सँग meeting गर्छु।` The Nepali engine
reads Devanagari well and Latin script badly, so VoiceStudio splits the text by language and renders
each span on the engine that suits it — Devanagari on [`xtts-nepali`](engines/xtts-nepali.md), Latin
on [`xtts-en`](engines/xtts-en.md) — with the **same reference clip** for both, then joins the spans
with a short crossfade. Both engines are XTTS v2 (one is a fine-tune of the other), which is what
keeps the two halves sounding like the same speaker.

The **TTS mode** control next to the language picker chooses how this runs:

| Mode | Behaviour |
|---|---|
| **Automatic** (default) | Mixed Nepali/English text is split; pure Nepali and pure English are rendered by one engine exactly as before |
| **Single language** | Never splits — the historical behaviour, one engine for the whole text |
| **Nepali + English mixed** | Always splits, and fails with a clear error if the English engine is unavailable |

In **Automatic**, if the English engine is not installed the request still produces a take from the
single configured engine, and the response says so in the `X-OmniVoice-Code-Switch` header rather
than pretending the English words were routed.

Mixed rendering is a local feature: a render sent to another machine
(see [remote GPU](remote-gpu.md)) uses that worker's single engine.

| Env var | Default | Effect |
|---|---|---|
| `OMNIVOICE_CODE_SWITCH_ENABLED` | `1` | Master switch. `0` makes every request single-engine |
| `OMNIVOICE_CODE_SWITCH_NE_BACKEND` | `xtts-nepali` | Engine id for Devanagari spans |
| `OMNIVOICE_CODE_SWITCH_EN_BACKEND` | `xtts-en` | Engine id for Latin spans |
| `OMNIVOICE_CODE_SWITCH_CROSSFADE_MS` | `30` | Crossfade between spans, in milliseconds |
| `OMNIVOICE_CODE_SWITCH_NORMALIZE_GAIN` | `1` | Clamped gain alignment, so neither engine is louder |
| `OMNIVOICE_CODE_SWITCH_TRIM_SILENCE` | `1` | Trims up to 120 ms of each engine's own padding at a boundary |
| `OMNIVOICE_CODE_SWITCH_PARALLEL` | `0` | Render spans concurrently. Off by default: two models on one device is how hosts run out of memory |

Language detection is deterministic and lives in `backend/services/language_segmenter.py`: Devanagari
is Nepali, an editable English/technical vocabulary and any other Latin word is English, an editable
Romanized-Nepali lexicon (`ma`, `aja`, `garchu`, …) is Nepali, and numbers, dates, URLs, emails and
identifiers keep the language of the span around them instead of becoming spans of their own.

## Speech recognition

The request's language is passed to engines that take one (IndicConformer); auto-detecting
engines (Whisper) are unaffected. When a clone reference clip is transcribed automatically, the
transcript is saved to the voice only if it is written in the voice language's script, so an
English or romanized mis-hearing of a Nepali clip is used once and never stored.

## Known limitations

- No Nepali UI translation (the interface languages are listed in Settings).
- English words inside Nepali are not transliterated; ordinals (`१०औँ`) are read as number + suffix.
- Romanized Nepali (`ma aja janchu`) is *detected* as Nepali but still reaches a Devanagari-reading
  engine, so it is approximated rather than pronounced properly. Write Nepali in Devanagari.
- Mixed rendering does not match speech rate or pitch across the boundary yet; it routes, levels and
  joins. `[pause Nms]` markers and long-text chunking do not apply to a mixed take.
- Bikram Sambat detection is by year (2050–2200); other calendar years are read as Gregorian.
- IndicConformer produces no commas or question marks; each pause-delimited sentence ends with `।`.
