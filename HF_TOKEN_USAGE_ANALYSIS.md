# HF_TOKEN Usage: IndicConformer & Indic Parler-TTS

## Overview
HF_TOKEN (Hugging Face access token) is used in VoiceStudio to access **gated models** from Hugging Face:
- **IndicConformer**: `ai4bharat/indic-conformer-600m-multilingual` (ASR - Speech-to-Text)
- **Indic Parler-TTS**: `ai4bharat/indic-parler-tts` (TTS - Text-to-Speech)

Both models are **gated** on Hugging Face, meaning users must:
1. Accept the model's terms on the HuggingFace model page
2. Provide a valid HF token with read access

---

## Token Resolution Pipeline: 3-Source Cascade

**Location**: [`backend/services/token_resolver.py`](backend/services/token_resolver.py)

VoiceStudio resolves the HF token with this priority order (first valid source wins):

```
Priority 1 (App)    → settings_store.get_hf_token()  [encrypted in SQLite]
Priority 2 (Env)    → os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
Priority 3 (HF CLI) → ~/.cache/huggingface/token    [written by `hf auth login`]
```

### Validation
Each candidate token is validated using `huggingface_hub.whoami(token=...)` to verify it's live.
- Results are cached for 300 seconds to avoid hammering the HF API
- Invalid/missing tokens automatically fall through to the next source
- Cache is invalidated when user clicks "Test now" in Settings or when a 401 error occurs mid-job

---

## IndicConformer (ASR) - In-Process Model

### Architecture
- **Type**: In-process on CPU (no subprocess sidecar)
- **Model**: ONNX + TorchScript (greedy CTC decoder)
- **Size**: ~2.4 GB (gated, requires HF_TOKEN)
- **Languages**: 22 Indian languages (22 including Nepali)

### Token Usage Points

#### 1. **Model Download** (First Use)
**Location**: [`backend/engines/indic_conformer/__init__.py`](backend/engines/indic_conformer/__init__.py#L1-L24)

```python
REPO_ID = "ai4bharat/indic-conformer-600m-multilingual"
REVISION = "e9b71b369c048e2c6b634d4c131061c34e441179"
```

The model is downloaded via `huggingface_hub` library which **automatically uses** `HF_TOKEN` from:
- `os.environ.get("HF_TOKEN")`
- Local HuggingFace CLI token file
- Huggingface_hub internal token resolution

The app doesn't explicitly pass HF_TOKEN to download this model; `huggingface_hub` handles token resolution internally.

#### 2. **Settings Panel Display**
**Location**: [`backend/api/routers/settings.py`](backend/api/routers/settings.py) → `/api/hf-token/state`

Shows whether HF token is set and allows user to:
- Save a token
- Test token validity
- Clear token

#### 3. **Model Availability Check**
**Location**: [`backend/services/asr_backend.py`](backend/services/asr_backend.py#L2578-L2707)

```python
def _indic_conformer():
    """Lazy: engines.indic_conformer imports this module for ASRBackend."""
    from engines.indic_conformer import IndicConformerBackend
    return IndicConformerBackend
```

When IndicConformer is selected, the backend checks if the gated model can be accessed.

---

## Indic Parler-TTS (TTS) - Subprocess Sidecar Model

### Architecture
- **Type**: Subprocess sidecar (separate Python venv)
- **Model**: Transformer-based conditional generation (parler-tts)
- **Size**: ~3.8 GB (gated, requires HF_TOKEN)
- **Voice**: Description-based (no reference clip needed)
- **Sidecar Location**: `~/.omnivoice/engines/indic-parler/.venv`

### Token Injection Path

#### 1. **Build Engine Environment**
**Location**: [`backend/services/engine_env.py#build_engine_env()`](backend/services/engine_env.py#L300-L345)

```python
def build_engine_env(
    *,
    base_env: Optional[dict] = None,
    inject_hf_token: bool = True,
) -> dict:
    """Build the environment dict to pass to an engine subprocess launcher."""
    env = dict(base_env if base_env is not None else os.environ)

    if inject_hf_token:
        try:
            from services import token_resolver
            
            resolved = token_resolver.resolve()
            if resolved and resolved.token:
                env["HF_TOKEN"] = resolved.token
                env["YOUR_HF_TOKEN"] = resolved.token  # For SoniTranslate compatibility
        except Exception:
            logger.exception("build_engine_env: token resolver failed (non-fatal)")
    
    return env
```

**Key Points**:
- Token is resolved via the 3-source cascade
- Injected as both `HF_TOKEN` and `YOUR_HF_TOKEN` environment variables
- Passed to subprocess via `env` parameter

#### 2. **Subprocess Spawn**
**Location**: [`backend/services/subprocess_backend.py#_spawn()`](backend/services/subprocess_backend.py#L481-L545)

```python
def _spawn(self) -> None:
    """Launch the sidecar if not already running."""
    
    # Env forwarding contract (Locked Decision D5):
    # - Inherit the parent's full env via os.environ.copy().
    # - The parent's env already carries HF_TOKEN (injected by the
    #   Phase 1 AUTH-04 launch sites that call token_resolver.resolve())
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    kwargs = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "env": env,  # ← HF_TOKEN passed here
        "bufsize": 0,
    }
    
    self._proc = spawn_owned([python_path, script_path], **kwargs)
```

#### 3. **Sidecar Script Model Loading**
**Location**: [`backend/engines/indic_parler/main.py#_load()`](backend/engines/indic_parler/main.py#L82-L103)

```python
REPO_ID = "ai4bharat/indic-parler-tts"
REVISION = "7b527af5ee8ed1f9a28d80b19703ed9bb8ba10ca"

def _load(stdout):
    global _model
    if _model is None:
        with _Heartbeat(stdout, "loading_model"):
            import torch
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
            
            device = os.environ.get("OMNIVOICE_INDIC_PARLER_DEVICE") or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # HF_TOKEN from environment is used automatically by huggingface_hub
            model = ParlerTTSForConditionalGeneration.from_pretrained(
                REPO_ID, revision=REVISION
            ).to(device)
            
            tokenizer = AutoTokenizer.from_pretrained(REPO_ID, revision=REVISION)
            description_tokenizer = AutoTokenizer.from_pretrained(
                model.config.text_encoder._name_or_path
            )
            _model = (model, tokenizer, description_tokenizer, device)
    return _model
```

**The Flow**:
1. Sidecar script inherits `HF_TOKEN` from parent process environment
2. When `parler_tts` and `transformers` load the gated model via `.from_pretrained()`, 
   the `huggingface_hub` library automatically reads `HF_TOKEN` from `os.environ`
3. Model downloads to cache directory (respects `HF_HUB_CACHE` if set)

---

## Complete Token Flow Diagram

```
USER INPUT (Settings → API Keys)
        ↓
[Encrypted SQLite Settings Store]
        ↓
token_resolver.py:resolve()
  ├─ _read_app() → settings_store.get_hf_token()
  ├─ _read_env() → os.environ.get("HF_TOKEN")
  └─ _read_hf_cli() → ~/.cache/huggingface/token
        ↓
Validation: huggingface_hub.whoami(token=...)
        ↓
        ├─ IndicConformer (In-Process)
        │   └─ huggingface_hub auto-reads HF_TOKEN from os.environ
        │       └─ Calls: ai4bharat/indic-conformer-600m-multilingual.from_pretrained()
        │
        └─ Indic Parler-TTS (Subprocess)
            ├─ engine_env.build_engine_env()
            │   └─ token_resolver.resolve() → HF_TOKEN, YOUR_HF_TOKEN
            ├─ subprocess_backend._spawn()
            │   └─ env = os.environ.copy() + HF_TOKEN
            └─ Sidecar script inherits HF_TOKEN
                └─ parler_tts.from_pretrained()
                    └─ huggingface_hub reads os.environ["HF_TOKEN"]
                        └─ Downloads: ai4bharat/indic-parler-tts
```

---

## Key Files and Locations

| Component | File | Purpose |
|-----------|------|---------|
| **Token Resolver** | [`backend/services/token_resolver.py`](backend/services/token_resolver.py) | 3-source cascade, validation, caching |
| **Engine Env Builder** | [`backend/services/engine_env.py`](backend/services/engine_env.py) | Injects HF_TOKEN into subprocess environment |
| **Subprocess Backend** | [`backend/services/subprocess_backend.py`](backend/services/subprocess_backend.py) | Spawns sidecars with environment |
| **IndicConformer Engine** | [`backend/engines/indic_conformer/__init__.py`](backend/engines/indic_conformer/__init__.py) | In-process ASR model |
| **Indic Parler Engine** | [`backend/engines/indic_parler/__init__.py`](backend/engines/indic_parler/__init__.py) | Subprocess TTS backend |
| **Indic Parler Sidecar** | [`backend/engines/indic_parler/main.py`](backend/engines/indic_parler/main.py) | Actual model loading/generation |
| **Settings API** | [`backend/api/routers/settings.py`](backend/api/routers/settings.py) | Save/test/clear HF token |
| **System Routes** | [`backend/api/routers/system.py`](backend/api/routers/system.py) | Diagnostics (checks for HF token presence) |

---

## Settings Panel Integration

**Location**: [`frontend/src/components/settings/ApiKeysPanel.jsx`](frontend/src/components/settings/ApiKeysPanel.jsx)

Shows:
- Three token sources (App/Env/HF CLI) with set/unset indicator
- Masked preview (`hf_…3jw`)
- "Test now" button that calls `/api/settings/hf-token/state?fresh=1`
- Active badge on highest-priority valid source

**Endpoints**:
- `GET /api/settings/hf-token/state` — Check token state (validate optional)
- `POST /api/settings/hf-token` — Save token to app settings
- `DELETE /api/settings/hf-token` — Clear token from app settings

---

## Security & Privacy

1. **Redaction**: HF_TOKEN is redacted from logs via [`backend/core/logging_filter.py`](backend/core/logging_filter.py)
   - Pattern: `hf_.*?[a-z0-9]{20}` is redacted to `***`

2. **Subprocess Stderr**: Drained via same logging filter (threat T-02-03)

3. **Encryption at Rest**: Token is encrypted in SQLite using Fernet symmetric AEAD
   - Key derived per-install from machine ID
   - Falls back to env/CLI source if moved across machines

4. **No Cloud Logging**: Tokens never sent to external services (local-first design)

---

## Troubleshooting

### Token Not Found
```
Error: "HF_TOKEN not set"
```
Check cascade order:
1. Settings → API Keys (saved in app?)
2. `echo $HF_TOKEN` (in environment?)
3. `cat ~/.cache/huggingface/token` (HF CLI token file?)

### 401 Unauthorized
Token exists but invalid → may need re-authentication at huggingface.co

### Model Download Fails
1. Verify token has **read** scope (not write-only)
2. Accept model's terms on HuggingFace model page
3. Check internet connectivity
4. Verify cache directory writable (`~/.cache/huggingface` or custom `HF_HUB_CACHE`)

---

## Environment Variables

| Variable | Used By | Purpose |
|----------|---------|---------|
| `HF_TOKEN` | Both engines | Primary token source |
| `HUGGING_FACE_HUB_TOKEN` | Both engines | Legacy fallback |
| `YOUR_HF_TOKEN` | Indic Parler sidecar | SoniTranslate compatibility |
| `HF_TOKEN_PATH` | token_resolver | Custom HF CLI token file location |
| `HF_HUB_CACHE` | huggingface_hub | Custom model cache directory |
| `HF_ENDPOINT` | huggingface_hub | Custom HF API endpoint (enterprise) |
| `OMNIVOICE_INDIC_CONFORMER_LANG` | IndicConformer | Default language (default: `ne`) |
| `OMNIVOICE_INDIC_PARLER_DIR` | Indic Parler | Custom sidecar venv location |
| `OMNIVOICE_INDIC_PARLER_DEVICE` | Indic Parler sidecar | `cpu` or `cuda` |

---

## References

- [IndicConformer Setup Docs](docs/engines/indic-conformer.md)
- [Indic Parler-TTS Setup Docs](docs/engines/indic-parler-tts.md)
- [HuggingFace Token Setup](docs/setup/huggingface-token.md)
- [Backend Services Structure](docs/STRUCTURE.md)
