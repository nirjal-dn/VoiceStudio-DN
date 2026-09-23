# Code-switch restoration (Devanagari → code-switched transcript)

A post-processing layer — **not** an ASR engine. IndicConformer transcribes
Nepali speech to *pure Devanagari*, so English words the speaker actually said
come back spelled phonetically in Devanagari (`अकाउन्ट` for "account",
`ब्यालेन्स` for "balance"). This layer runs the finished transcript through a
small, local, CPU-only instruction LLM that does **exactly two things**:

1. rewrites clearly-English words (transcribed phonetically in Devanagari) back
   to normal English Latin spelling, and
2. adds sentence punctuation.

```
Audio → IndicConformer → Devanagari transcript → LLM → code-switched transcript + punctuation
```

| Input (IndicConformer)                     | Output (restored)                          |
|--------------------------------------------|--------------------------------------------|
| `मेरो अकाउन्टको ब्यालेन्स चेक गरिदिनुहोस्` | `मेरो account को balance check गरिदिनुहोस्।` |
| `नमस्ते मेरो अकाउन्टको ब्यालेन्स कति छ`     | `नमस्ते, मेरो account को balance कति छ?`     |

It is **not** a translator or grammar corrector. Nepali words stay in
Devanagari and in order; it never translates Nepali→English, never rephrases,
and makes the smallest possible change.

## Enabling it

Opt-in and **off by default** on every platform (so the default-behaviour
parity rule holds — with it off this is a no-op pass-through everywhere).

```bash
uv sync --extra codeswitch          # installs llama-cpp-python (CPU build)
export OMNIVOICE_CODESWITCH_RESTORE=1
```

On first use it downloads the model (~1 GB) to the Hugging Face cache. The
restored text is surfaced as the transcript's `refined_text` (the raw
Devanagari `text` and per-segment timings are left untouched), so clients that
already read `refined_text ?? text` pick it up with no change.

## Model

Default: **Qwen2.5-1.5B-Instruct**, GGUF `Q4_K_M`. Picked for a 16 GB CPU-only
host — strong instruction-following *and* multilingual (incl. Devanagari) at a
tiny footprint. Generation is deterministic (temperature 0, fixed seed) and the
model is loaded once and reused for the process lifetime.

| Metric (16 GB, CPU-only, e.g. i7-9700) | Estimate                       |
|----------------------------------------|--------------------------------|
| Resident RAM (Q4_K_M, `n_ctx=2048`)    | ~1.0–1.4 GB                     |
| First-call load                        | ~1–3 s (then cached in-process)|
| Latency, one dictation utterance       | ~0.5–2 s                       |
| Model download (first use)             | ~1 GB                          |

A bigger model trades latency for accuracy — Qwen2.5-3B-Instruct-GGUF is a drop-in.

## Environment variables

| Variable                            | Default                              | Purpose |
|-------------------------------------|--------------------------------------|---------|
| `OMNIVOICE_CODESWITCH_RESTORE`      | *(off)*                              | `1`/`true` to enable. |
| `OMNIVOICE_CODESWITCH_MODEL_PATH`   | *(unset)*                            | Absolute path to a local `.gguf` (skips the download). |
| `OMNIVOICE_CODESWITCH_MODEL_REPO`   | `Qwen/Qwen2.5-1.5B-Instruct-GGUF`    | HF repo to download from. |
| `OMNIVOICE_CODESWITCH_MODEL_FILE`   | `qwen2.5-1.5b-instruct-q4_k_m.gguf`  | GGUF file within the repo. |
| `OMNIVOICE_CODESWITCH_TIMEOUT_S`    | `20`                                 | Hard wall-clock budget per call. |

## Safety guard

The LLM output is accepted only if its Devanagari **content** codepoints are a
subsequence of the input's (the danda `।`/`॥` it adds as punctuation is
excluded), and it must still retain some Devanagari when the input had it.
Latinizing a word or splitting a postposition (`अकाउन्टको` → `account को`) only
*drops* Devanagari, so it passes; a translation, paraphrase, reordering, or
invented Nepali word introduces or reorders Devanagari (or drops all of it) and
is rejected — the raw transcript stands. This is best-effort defence-in-depth:
a *partial* translation can still slip past the guard, so the strict prompt and
few-shot examples remain the first line of defence.

Any failure — package not installed, model missing, timeout, guard rejection —
falls back to the raw transcript. It never raises and never blocks the final.
