#!/usr/bin/env python
"""LoRA fine-tune Whisper (large-v3 by default) for Nepali↔English code-switch.

Targets the exact "English comes out in Devanagari" problem: because the labels
keep English in Latin (and Nepali in Devanagari), the model learns to STOP
transliterating embedded English. That only generalises with enough data — a
couple of rows just memorises them (see REALITY CHECK). For an immediate,
no-training nudge, the whisper-ne-en engine already primes decoding with a
mixed-script prompt (OMNIVOICE_CS_PROMPT).

    uv pip install peft                       # only missing dep
    uv run python scripts/finetune_whisper_ne_en.py \
        --data "/home/edigitalnepal/Downloads/data.csv" \
        --out  ./whisper-ne-en-lora

CSV format (header required): ``audio_path,transcription``. audio_path may point
at any media ffmpeg reads (mp4/m4a/wav); the transcript is the code-switched
Devanagari+Latin text as spoken.

────────────────────────────────────────────────────────────────────────────
REALITY CHECK — read before trusting the output
────────────────────────────────────────────────────────────────────────────
* Two rows is NOT a fine-tuning set. A 1.5B-param model trained on 2 clips
  memorises them and forgets the rest (catastrophic overfitting). This script
  is correct; the *data* is the limit. For real ne-en code-switch gains you need
  hours of labelled audio (thousands of clips), split train/eval.
* large-v2 full/LoRA training wants a GPU. On a CPU-only box, smoke-test the
  pipeline with ``--model openai/whisper-small`` (or ``whisper-tiny``) first —
  it exercises every line below in minutes instead of hours.
The knobs that matter for "tune a little" without nuking the model: few epochs,
small LoRA rank, low LR. Defaults below are deliberately gentle.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

SAMPLE_RATE = 16_000


# ── audio: decode any media → 16 kHz mono float32 via the bundled ffmpeg ─────
def _ffmpeg() -> str:
    """The imageio-ffmpeg binary (a project dep), so mp4/AAC decode works even
    when no system ffmpeg is on PATH — the same reason the app bundles it."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"  # fall back to PATH


def decode_audio(path: str) -> np.ndarray:
    cmd = [_ffmpeg(), "-nostdin", "-threads", "0", "-i", path,
           "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
           "-ar", str(SAMPLE_RATE), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


# ── data: read + clean the CSV (strips CRLF, stray quotes/apostrophes) ───────
def load_rows(csv_path: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            raw = (r.get("audio_path") or "").strip().strip("'\"").strip()
            text = (r.get("transcription") or "").strip()
            if not raw or not text:
                continue
            if not Path(raw).is_file():
                print(f"  ! skip (missing file): {raw}", file=sys.stderr)
                continue
            rows.append((raw, text))
    if not rows:
        sys.exit(f"No usable rows in {csv_path} (need columns audio_path,transcription).")
    return rows


class SpeechDataset(torch.utils.data.Dataset):
    """Pre-extracts log-mel features + label ids so training is pure GEMM."""

    def __init__(self, rows, processor):
        self.items = []
        for audio_path, text in rows:
            audio = decode_audio(audio_path)
            feats = processor.feature_extractor(
                audio, sampling_rate=SAMPLE_RATE
            ).input_features[0]
            labels = processor.tokenizer(text).input_ids
            self.items.append({"input_features": feats, "labels": labels})
            print(f"  + {Path(audio_path).name}: {len(audio)/SAMPLE_RATE:.1f}s, "
                  f"{len(labels)} label tokens")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


@dataclass
class Collator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features],
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)
        # Drop the BOS the tokenizer prepends; the model adds it back.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="CSV: audio_path,transcription")
    ap.add_argument("--out", default="./whisper-ne-en-lora")
    ap.add_argument("--model", default="openai/whisper-large-v3",
                    help="base model; use openai/whisper-small for a CPU smoke test")
    ap.add_argument("--epochs", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=1e-3, help="LoRA LR (1e-3 typical)")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--language", default="nepali",
                    help="Whisper label language; 'nepali' keeps Devanagari, "
                         "English words stay Latin (code-switch friendly)")
    ap.add_argument("--no-lora", action="store_true", help="full fine-tune (GPU only)")
    args = ap.parse_args()

    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration, WhisperProcessor)

    cuda = torch.cuda.is_available()
    if "large" in args.model and not cuda:
        print("WARNING: large-v2 on CPU will be extremely slow / may OOM. "
              "Use --model openai/whisper-small to smoke-test the pipeline.",
              file=sys.stderr)

    print(f"Loading {args.model} …")
    processor = WhisperProcessor.from_pretrained(
        args.model, language=args.language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    # Standard Whisper-FT config: let labels carry the language/task prefix.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # required with gradient checkpointing
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    if not args.no_lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            sys.exit("peft not installed — run: uv pip install peft "
                     "(or pass --no-lora for a full fine-tune, GPU only).")
        model = get_peft_model(model, LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_rank * 2,
            target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none"))
        model.print_trainable_parameters()
        model.enable_input_require_grads()  # so grads flow with checkpointing

    print("Preparing dataset …")
    rows = load_rows(args.data)
    dataset = SpeechDataset(rows, processor)

    targs = Seq2SeqTrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        fp16=cuda,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,  # our dataset yields dicts (needed with peft)
        label_names=["labels"],
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=dataset,
        data_collator=Collator(processor, model.config.decoder_start_token_id),
        processing_class=processor.feature_extractor,
    )
    print("Training …")
    trainer.train()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)          # LoRA adapter (or full weights)
    processor.save_pretrained(args.out)
    print(f"Saved to {args.out}")

    # ── sanity: transcribe the training clips with the tuned model ──────────
    print("\n── inference on training clips (overfit check) ──")
    model.eval()
    dev = "cuda" if cuda else "cpu"
    model.to(dev)
    for audio_path, ref in rows:
        feats = processor.feature_extractor(
            decode_audio(audio_path), sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(dev)
        with torch.no_grad():
            ids = model.generate(feats, language=args.language,
                                 task="transcribe", max_new_tokens=225)
        hyp = processor.batch_decode(ids, skip_special_tokens=True)[0]
        print(f"\n{Path(audio_path).name}\n  ref: {ref}\n  hyp: {hyp.strip()}")


if __name__ == "__main__":
    main()
