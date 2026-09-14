"""indic-conformer: AI4Bharat IndicConformer 600M multilingual ASR.

Runs ai4bharat/indic-conformer-600m-multilingual (MIT, 22 Indian languages
including Nepali) in-process on CPU. The model is ONNX plus a TorchScript
feature extractor, and both runtimes are already in the app environment, so
this skips the model card's ``trust_remote_code`` transformers wrapper and
drives the graphs directly (greedy CTC, which also yields word timestamps).

Each recording is transcribed in one pass over the whole file, never in
chunks. Output is in the language's native script; for Devanagari languages
(Nepali by default) anything outside the Devanagari block is removed, so a
stray token can never surface as Latin, Arabic or Hangul text.

The model is gated on Hugging Face: accept its terms on the model page, then
provide a token through Settings -> API Keys, ``HF_TOKEN`` (the project ``.env``
is loaded at startup), or ``hf auth login``. The ~2.4 GB download happens on
first use.

Settings:
    OMNIVOICE_INDIC_CONFORMER_LANG     language code (default ``ne``)
"""
from __future__ import annotations

import logging
import math
import os
import re
import threading

from services.asr_backend import ASRBackend, _load_audio_16k_mono_f32

logger = logging.getLogger("omnivoice.asr.indic_conformer")

REPO_ID = "ai4bharat/indic-conformer-600m-multilingual"
REVISION = "e9b71b369c048e2c6b634d4c131061c34e441179"
LANGUAGES = (
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml",
    "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur",
)
#: Languages written in Devanagari, whose output is kept strictly Devanagari.
_DEVANAGARI_LANGS = frozenset({"brx", "doi", "hi", "kok", "mai", "mr", "ne", "sa"})
_NOT_DEVANAGARI = re.compile(r"[^ऀ-ॿ]+")

_SAMPLE_RATE = 16_000
_BLANK_ID = 256
#: Encoder frame hop: 10 ms features, 8x subsampling.
_FRAME_S = 0.08
#: A pause this long starts a new segment.
_SEGMENT_GAP_S = 0.8

_runtime = None
_runtime_lock = threading.Lock()
#: One transcription at a time: each holds several hundred MB of activations.
_infer_lock = threading.Lock()


def _language() -> str:
    lang = os.environ.get("OMNIVOICE_INDIC_CONFORMER_LANG", "ne").strip().lower()
    if lang not in LANGUAGES:
        raise ValueError(
            f"OMNIVOICE_INDIC_CONFORMER_LANG={lang!r} is not supported; "
            f"use one of: {', '.join(LANGUAGES)}."
        )
    return lang


def clean_word(text: str, lang: str) -> str:
    """Drop the vocabulary's ``<unk>`` token; for Devanagari languages turn the
    ASCII ``|`` sentence mark into ``।`` and remove every non-Devanagari character."""
    text = text.replace("<unk>", "")
    if lang in _DEVANAGARI_LANGS:
        text = _NOT_DEVANAGARI.sub("", text.replace("|", "।"))
    return text


class _Runtime:
    """The loaded graphs plus vocabularies."""

    def __init__(self, root: str) -> None:
        import json

        import onnxruntime as ort
        import torch

        assets = os.path.join(root, "assets")
        self.preprocessor = torch.jit.load(os.path.join(assets, "preprocessor.ts"), map_location="cpu")
        providers = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(os.path.join(assets, "encoder.onnx"), providers=providers)
        self.ctc = ort.InferenceSession(os.path.join(assets, "ctc_decoder.onnx"), providers=providers)
        with open(os.path.join(assets, "vocab.json"), encoding="utf-8") as f:
            self.vocab = json.load(f)
        with open(os.path.join(assets, "language_masks.json"), encoding="utf-8") as f:
            self.masks = json.load(f)

    def words(self, audio, lang: str) -> list[tuple[str, float, float, float]]:
        """Greedy CTC words for 16 kHz mono float32 audio, as
        (text, start_s, end_s, probability)."""
        import torch

        wav = torch.from_numpy(audio).unsqueeze(0)
        with torch.inference_mode():
            feats, length = self.preprocessor(
                input_signal=wav, length=torch.tensor([wav.shape[-1]])
            )
        enc, enc_len = self.encoder.run(
            ["outputs", "encoded_lengths"],
            {"audio_signal": feats.numpy(), "length": length.numpy()},
        )
        logits = self.ctc.run(["logprobs"], {"encoder_output": enc})[0][0, : int(enc_len[0])]
        best, path = torch.from_numpy(logits[:, self.masks[lang]]).log_softmax(-1).max(-1)
        vocab = self.vocab[lang]

        words: list[list] = []  # [text, start, end, logprob_sum, frames]
        prev = _BLANK_ID
        for f, (tok, lp) in enumerate(zip(path.tolist(), best.tolist())):
            if tok == _BLANK_ID:
                prev = tok
                continue
            end = (f + 1) * _FRAME_S
            if tok == prev:  # one token held across frames
                words[-1][2] = end
                words[-1][3] += lp
                words[-1][4] += 1
                continue
            prev = tok
            piece = vocab[tok]
            if piece.startswith("▁") or not words:
                words.append([piece.lstrip("▁"), f * _FRAME_S, end, lp, 1])
            else:
                words[-1][0] += piece
                words[-1][2] = end
                words[-1][3] += lp
                words[-1][4] += 1
        out = []
        for text, start, stop, lp_sum, n in words:
            text = clean_word(text, lang)
            if text:
                out.append((text, start, stop, math.exp(lp_sum / n)))
        return out


def _load() -> _Runtime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            from huggingface_hub import snapshot_download

            try:
                root = snapshot_download(REPO_ID, revision=REVISION, local_files_only=True)
            except Exception:
                try:
                    root = snapshot_download(REPO_ID, revision=REVISION)
                except Exception as exc:
                    if type(exc).__name__ in {"GatedRepoError", "RepositoryNotFoundError"}:
                        raise RuntimeError(
                            f"{REPO_ID} is gated. Accept its terms at "
                            f"https://huggingface.co/{REPO_ID} and set a Hugging Face "
                            "token (Settings -> API Keys, HF_TOKEN, or `hf auth login`)."
                        ) from exc
                    raise
            logger.info("loading IndicConformer from %s", root)
            _runtime = _Runtime(root)
        return _runtime


def _segments(words, word_timestamps: bool) -> list[dict]:
    groups: list[list] = []
    for w in words:
        if not groups or w[1] - groups[-1][-1][2] >= _SEGMENT_GAP_S:
            groups.append([])
        groups[-1].append(w)
    return [
        {
            "text": " ".join(w[0] for w in group),
            "start": round(group[0][1], 3),
            "end": round(group[-1][2], 3),
            "words": [
                {"word": f" {w[0]}", "start": round(w[1], 3), "end": round(w[2], 3), "probability": round(w[3], 3)}
                for w in group
            ] if word_timestamps else [],
        }
        for group in groups
    ]


class IndicConformerBackend(ASRBackend):
    id = "indic-conformer"
    display_name = "IndicConformer 600M (22 Indic languages)"
    gpu_compat = ("cpu",)
    # ~0.2x real time on CPU once loaded: fast enough for dictation.
    serves_capture = True
    # Dictation sends the whole recording once it stops; never partials or chunks.
    whole_recording = True
    # Language the UI selects instead of "Auto" while this engine is active.
    default_language = os.environ.get("OMNIVOICE_INDIC_CONFORMER_LANG", "ne").strip().lower()

    @property
    def _model(self):
        return _runtime

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import onnxruntime  # noqa: F401
            import torch  # noqa: F401
        except Exception as exc:
            return False, f"onnxruntime and torch are required ({exc})."
        try:
            _language()
        except ValueError as exc:
            return False, str(exc)
        from huggingface_hub import try_to_load_from_cache

        if isinstance(try_to_load_from_cache(REPO_ID, "assets/encoder.onnx", revision=REVISION), str):
            return True, "ready (CPU)"
        return True, (
            "ready (CPU); first use downloads the gated model (~2.4 GB): accept its "
            f"terms at https://huggingface.co/{REPO_ID} and set a Hugging Face token"
        )

    def ensure_loaded(self) -> None:
        _load()

    def transcribe(self, audio_path: str, *, word_timestamps: bool = True) -> dict:
        lang = _language()
        audio, _sr = _load_audio_16k_mono_f32(audio_path)
        words = []
        if len(audio) >= _SAMPLE_RATE // 10:
            runtime = _load()
            # ponytail: one pass over the whole recording as required; encoder
            # memory grows with length, so very long files need a bigger box.
            with _infer_lock:
                words = runtime.words(audio, lang)
        segments = _segments(words, word_timestamps)
        return {
            "text": " ".join(s["text"] for s in segments),
            "segments": segments,
            "chunks": [{"text": s["text"], "timestamp": (s["start"], s["end"])} for s in segments],
            "language": lang,
        }

    def unload(self) -> None:
        global _runtime
        with _runtime_lock:
            _runtime = None


__all__ = ["IndicConformerBackend", "LANGUAGES", "REPO_ID", "clean_word"]
