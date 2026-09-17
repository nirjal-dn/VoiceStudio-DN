# Nepali in VoiceStudio

Nepali (`ne`) is a registered language (`backend/services/languages.py`): every module resolves
"Nepali", `ne`, `npi`, `nep` and `ne-NP` to the same language and gets the same behaviour.

## Engines

| Task | Engine | Notes |
|---|---|---|
| Voice clone (TTS) | [XTTS v2 Nepali](engines/xtts-nepali.md) | Clones from a 10–15 s reference clip; non-commercial weights |
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

## Speech recognition

The request's language is passed to engines that take one (IndicConformer); auto-detecting
engines (Whisper) are unaffected. When a clone reference clip is transcribed automatically, the
transcript is saved to the voice only if it is written in the voice language's script, so an
English or romanized mis-hearing of a Nepali clip is used once and never stored.

## Known limitations

- No Nepali UI translation (the interface languages are listed in Settings).
- English words inside Nepali are not transliterated; ordinals (`१०औँ`) are read as number + suffix.
- Bikram Sambat detection is by year (2050–2200); other calendar years are read as Gregorian.
- IndicConformer produces no commas or question marks; each pause-delimited sentence ends with `।`.
