# ask -- Architecture & Technical Design

This document captures every design decision, trade-off, known issue, and implementation detail in the `ask` project. It is intended as a comprehensive reference for future development.

---

## 1. What ask Does

`ask` is a CLI tool that runs multiple AI models locally on Apple Silicon Macs. The user types a question and `ask` automatically routes it to the right specialist model -- coding, vision, speech-to-text, or general chat -- all running offline via Apple's MLX framework. Only one model is loaded at a time to stay within memory limits.

Two interfaces: one-shot (`ask "question"`) and interactive REPL (`ask -i`).

---

## 2. Target Hardware

- macOS only, Apple Silicon only (M1/M2/M3/M4)
- Minimum 8GB unified memory; tested primarily on 16GB M4 MacBook Air
- All models use 4-bit quantization (`Q4`) -- approximately 0.55GB per billion parameters
- The 16GB tier uses ~1.8GB for the classifier + up to ~5.9GB for the largest specialist (VLM), totaling ~7.7GB peak, leaving headroom for macOS

---

## 3. High-Level Architecture

```
User Input
    |
    v
[cli.py] -- Parses args, detects subcommands (models vs query)
    |
    v
[router.py] -- Classifies query via Phi-3.5-mini LLM
    |            (skipped if --image, --audio, or --model flag is set)
    |
    +-- role = "general" | "code" | "vision" | "audio"
    |
    v
[models.py] -- Loads the specialist model for that role (lazy, cached)
    |            Unloads previous model first if different role
    |
    v
Response printed to stdout
```

### Key Invariant

**At most two models are in memory at once**: the classifier (Phi-3.5-mini, ~1.8GB) and one specialist. The classifier stays resident across queries in interactive mode. When the specialist role changes, the previous specialist is explicitly unloaded (`gc.collect()` + `mx.metal.clear_cache()`).

When explicit flags (`--image`, `--audio`, `--model`) are used, the classifier is never loaded at all.

---

## 4. Module-by-Module Design

### 4.1 `cli.py` -- Entry Point

**Two parser paths.** The main `argparse` parser handles queries, but `ask models ...` is intercepted before argparse via `sys.argv[1] == "models"` check. This is necessary because argparse's `nargs="*"` for the query positional would swallow `models` as part of the query text.

**One-shot mode.** `ask "question"` parses args, loads config, routes, generates, prints, exits. The model loads on first use and the process terminates after one response.

**Interactive mode.** `ask -i` enters a REPL loop. The classifier stays in memory between queries. If the user switches roles (e.g., from a general question to a code question), the previous specialist is unloaded and the new one loaded. REPL supports `/image <path> <prompt>` and `/audio <path>` prefix commands.

**Warning suppression.** Transformers and model configs emit many noisy warnings (`rope_parameters`, deprecated calls, etc.). These are suppressed at module import time via `warnings.filterwarnings` and the `TRANSFORMERS_NO_ADVISORY_WARNINGS` env var.

### 4.2 `router.py` -- Query Classification

**Why LLM classification?** Considered keyword/regex matching and embedding-based classification. Chose LLM because it handles ambiguous queries well ("explain this error in my code" -> CODE, "what causes rain" -> GENERAL) without maintaining fragile rule sets.

**Classifier model: Phi-3.5-mini-instruct (3.8B, 4-bit).** Chosen for speed (~41 tok/s on M4) and tiny memory footprint (~1.8GB). The classification prompt asks for a single category keyword and limits generation to 5 tokens. The response is parsed by scanning for known category keywords (GENERAL, CODE, VISION, AUDIO) in the uppercased output.

**Fallback.** If the classifier output doesn't contain any known keyword, defaults to `"general"`.

**Routing priority (in `route_query`).** Explicit `--model` flag overrides everything. Then `--image` implies vision, `--audio` implies audio. Only if none of these are set does the LLM classifier run.

**Classifier caching.** The classifier model and tokenizer are stored in module-level globals (`_classifier_model`, `_classifier_tokenizer`). In interactive mode, this means the classifier loads once and stays resident.

### 4.3 `models.py` -- Model Loading & Generation

**Single-model cache.** Module-level globals `_current_role`, `_current_model`, `_current_processor`, `_current_type` track the currently loaded specialist. `_ensure_model()` loads a model only if the requested role differs from the cached one.

**Unloading.** `_unload_current()` sets all globals to `None`, calls `gc.collect()`, then `mx.metal.clear_cache()`. The Metal cache clear is important -- without it, GPU memory from the previous model lingers.

**Three generation paths:**

1. **`generate_text()`** -- For `mlx_lm` models (general, code). Uses the tokenizer's `apply_chat_template()` to format prompts correctly for each model family. Appends ` /no_think` to the user content for Qwen3 models (but not Qwen2.5-Coder) to suppress their chain-of-thought thinking mode.

2. **`generate_vision()`** -- For `mlx_vlm` models. Uses `mlx_vlm.prompt_utils.apply_chat_template()` which is a different function from the tokenizer's method. Takes image paths as input. Returns `result.text` from the `GenerationResult` object that `mlx_vlm.generate()` returns (different return type than `mlx_lm.generate()` which returns a string directly).

3. **`transcribe_audio()`** -- For `mlx_qwen3_asr` models. Runs as a **subprocess** (`python -m mlx_qwen3_asr ...`) rather than in-process. This is because mlx-qwen3-asr has its own model loading and audio processing pipeline that doesn't expose a clean Python API for integration. The `--stdout-only` flag suppresses its progress bars. Timeout is 10 minutes.

**`run_query()` dispatch.** This is the main function called by `cli.py`. It checks `model_type` and `role` to decide which generation path to use. For vision models called without an image (user asks a text question that got routed to vision), it falls back to text-only generation through the VLM.

**Output cleaning.** `_clean_response()` strips:
- `<think>...</think>` blocks -- Qwen3 models have a "thinking mode" where they emit chain-of-thought reasoning inside these tags. Even with `/no_think`, partial thinking blocks sometimes appear.
- Special tokens: `<|im_end|>`, `<|im_start|>`, `<|endoftext|>`, `<|end|>`, `</s>`, `<s>` -- these leak through when the model generates its own stop tokens before the tokenizer's stopping criteria kicks in.

**Stderr suppression during model loading.** Both `router.py` and `models.py` redirect `sys.stderr` to `io.StringIO()` and set `logging.disable(logging.WARNING)` during model loading. This catches noisy config messages from `transformers` model configs (particularly Phi-3.5-mini's `rope_parameters` warnings and various deprecation notices). The original stderr is restored in a `finally` block.

### 4.4 `config.py` -- Configuration

**Config location.** `~/.ask/config.yaml`. Created by `setup.sh`, read at startup.

**Config format.** YAML with two top-level keys:
```yaml
classifier:
  model: <huggingface-model-id>
  type: mlx_lm
  description: <human-readable string>

roles:
  general: { model: ..., type: ..., description: ... }
  code:    { model: ..., type: ..., description: ... }
  vision:  { model: ..., type: ..., description: ... }
  audio:   { model: ..., type: ..., description: ... }
```

**Tier profiles.** Stored as package data in `src/ask/model_profiles/`. Three tiers: `tier_8gb.yaml`, `tier_16gb.yaml`, `tier_32gb.yaml`. Setup picks the right one based on detected RAM.

**`PROFILES_DIR` path.** Resolves to `Path(__file__).resolve().parent / "model_profiles"`. This works both in development (running from the repo) and after `pip install` (profiles are installed as package data alongside the Python files). An earlier version used `parent.parent.parent` to navigate to the repo root, which broke after installation.

### 4.5 `hardware.py` -- Hardware Detection

Uses `sysctl` for chip name and RAM, `system_profiler SPDisplaysDataType` for GPU core count. Tier selection:
- 32GB+ -> `32gb`
- 16GB+ -> `16gb`
- 8GB+  -> `8gb`
- <8GB  -> `unsupported` (exits with error)

### 4.6 `model_manager.py` -- Model Management CLI

Provides `ask models list|add|remove|download` subcommands. `add` writes a new entry to `~/.ask/config.yaml`. `remove` reads the tier profile for the current hardware and restores the default model for that role. `download` calls `huggingface_hub.snapshot_download()` to pre-fetch weights.

`_save_config()` preserves the comment header lines (hardware metadata written by `setup.sh`) when rewriting the YAML file.

---

## 5. Model Choices & Rationale

### Classifier: Phi-3.5-mini-instruct (3.8B, 4-bit)

- ~41 tok/s on M4, ~1.8GB VRAM
- Only generates 5 tokens per classification, so latency is <200ms
- Good instruction following for structured classification tasks
- Small enough to stay co-resident with any specialist model

### General: Qwen3-8B (4-bit)

- ~21 tok/s on M4, ~4.7GB
- Strong reasoning and knowledge across domains
- Has a "thinking mode" (`<think>` tags) that we suppress with `/no_think`

### Code: Qwen2.5-Coder-7B-Instruct (4-bit)

- ~20 tok/s on M4, ~4.4GB
- 88% on HumanEval -- best coding performance in its size class at time of selection
- Does NOT have thinking mode (Qwen2.5 family, not Qwen3), so `/no_think` is not appended

### Vision: Qwen2.5-VL-7B-Instruct (4-bit)

- ~23 tok/s on M4, ~5.9GB (the largest specialist)
- Handles image understanding, OCR, diagram interpretation
- Uses `mlx_vlm` framework which has a different API than `mlx_lm`

### Audio: Qwen3-ASR-0.6B (16GB tier) / Qwen3-ASR-1.7B (32GB tier)

- 0.6B: RTF 0.73x (real-time capable), ~0.6GB
- 1.7B: RTF 2.88x (slower but more accurate), ~2-3GB
- Speech-to-text only (no TTS, no audio understanding)
- Run as subprocess via `mlx-qwen3-asr` CLI

### 8GB Tier Compromise

On 8GB machines, both general and code roles use Phi-3.5-mini (the classifier model). This avoids loading a second model and keeps peak memory under 6GB. The vision model (5.9GB) is tight but workable since it's never co-loaded with the classifier.

---

## 6. Known Issues & Workarounds

### 6.1 transformers v4 vs v5 Conflict

**Problem.** `mlx-lm>=0.30` declares `transformers>=5.0.0`. But Qwen2.5-VL model configs reference the processor class `Qwen2VLImageProcessor`, which was renamed/reorganized in transformers v5. Loading the VLM with transformers v5 fails with `Unrecognized image processor`.

**Workaround.** `setup.sh` installs the package normally (which pulls transformers v5 via mlx-lm), then force-downgrades:
```bash
pip install "transformers>=4.45.0,<5.0"
```
This produces a pip incompatibility warning (`mlx-lm 0.31.1 requires transformers>=5.0.0, but you have transformers 4.57.6 which is incompatible`) but all functionality works correctly.

**Why not pin in pyproject.toml?** Pinning `transformers<5` in the package dependencies causes pip to resolve to an older `mlx-vlm==0.3.9` (which is compatible with transformers v4), but that version requires PyTorch -- a massive unnecessary dependency that defeats the purpose of using MLX.

**Verified working combination:** `mlx-lm==0.31.1` + `mlx-vlm==0.4.1` + `transformers==4.57.6`

**Long-term fix.** This will resolve itself when either (a) the Qwen2.5-VL model configs are updated for transformers v5, or (b) mlx-vlm adds native support for the new processor class names.

### 6.2 Qwen3 Thinking Mode Leakage

**Problem.** Qwen3 models have a thinking mode where they emit `<think>reasoning</think>` before their actual response. This produces long, unwanted chain-of-thought text.

**Workaround.** Two layers:
1. Append ` /no_think` to the prompt content for Qwen3 models (detected by checking if `"qwen3"` is in the model name). This instructs the model to skip thinking mode via its chat template.
2. `_clean_response()` strips any `<think>...</think>` blocks that slip through anyway.

**The `/no_think` is NOT added for Qwen2.5-Coder** because it's a Qwen2.5 model (no thinking mode), and the literal text `/no_think` would be treated as part of the user's query, causing bizarre output like code that references `/no_think` as a file path.

### 6.3 Special Token Leakage

**Problem.** Models sometimes generate their own stop tokens (`<|im_end|>`, `<|endoftext|>`, etc.) as literal text before the tokenizer's stopping criteria activates.

**Workaround.** `_clean_response()` strips a hardcoded list of common special tokens via string replacement.

### 6.4 Noisy Model Loading Output

**Problem.** Loading models via `mlx_lm.load()` and `mlx_vlm.load()` triggers various warnings and info messages from the transformers library (rope_parameters warnings, deprecation notices, config validation messages). These clutter the user-facing output.

**Workaround.** During model loading, stderr is temporarily redirected to `io.StringIO()` and `logging.disable(logging.WARNING)` is set. Both are restored in a `finally` block. Additionally, `warnings.filterwarnings("ignore", ...)` is set at module import time in `cli.py` for the most common patterns.

### 6.5 setup.sh Overwrites Config

Running `./setup.sh` regenerates `~/.ask/config.yaml` from the tier profile, losing any custom model swaps made via `ask models add`. Users who want to preserve customizations must back up their config before re-running setup.

---

## 7. Configuration & File Layout

### Runtime State (per-user)

```
~/.ask/
  config.yaml    # Active configuration (roles -> models mapping)
  venv/          # Python virtual environment with all dependencies
```

### Repository

```
ask/
  setup.sh                           # One-command install script
  pyproject.toml                     # Package metadata, deps, entry point
  .gitignore
  README.md
  ARCHITECTURE.md                    # This document
  src/ask/
    __init__.py                      # Package init, __version__
    cli.py                           # Argument parsing, REPL, subcommand dispatch
    config.py                        # YAML config load/save, tier profile loading
    hardware.py                      # Apple Silicon hardware detection
    router.py                        # LLM-based query classification
    models.py                        # Model loading, caching, generation dispatch
    model_manager.py                 # `ask models` subcommand handlers
    model_profiles/
      tier_8gb.yaml                  # Default models for 8GB machines
      tier_16gb.yaml                 # Default models for 16GB machines
      tier_32gb.yaml                 # Default models for 32GB+ machines
```

### Package Data

`model_profiles/*.yaml` is declared as package data in `pyproject.toml`:
```toml
[tool.setuptools.package-data]
ask = ["model_profiles/*.yaml"]
```
This ensures profiles are installed alongside the Python code, so `config.py` can find them via `Path(__file__).resolve().parent / "model_profiles"` both in development and after `pip install`.

---

## 8. Dependency Graph

```
ask-cli
  ├── mlx >= 0.18.0           # Apple's ML framework (Metal acceleration)
  ├── mlx-lm >= 0.20.0        # Text LLM loading/generation on MLX
  ├── mlx-vlm >= 0.1.0        # Vision-language model support on MLX
  ├── mlx-qwen3-asr >= 0.3.0  # Qwen3 speech-to-text on MLX
  ├── pyyaml >= 6.0            # Config file parsing
  ├── rich >= 13.0             # (available but not heavily used yet)
  └── huggingface-hub >= 0.20  # Model downloading
```

**Transitive dependency of note:** `transformers` is pulled by mlx-lm (declares >=5.0) and mlx-vlm. We force it to 4.45-4.x range post-install (see Section 6.1).

**System dependencies (installed by setup.sh via Homebrew):**
- Python 3 (>= 3.10)
- ffmpeg (required by mlx-qwen3-asr for audio decoding)

---

## 9. Setup Flow (setup.sh)

1. **Platform check.** Verify macOS + arm64. Exit with error otherwise.
2. **System deps.** Install Homebrew, Python 3, ffmpeg via `brew install` if missing.
3. **Hardware detection.** Read chip, RAM, GPU cores via `sysctl` and `system_profiler`. Select tier (8gb/16gb/32gb).
4. **Venv creation.** Delete existing `~/.ask/venv/` if present. Create fresh with `python3 -m venv`.
5. **Package install.** `pip install .` from the repo directory. This installs the `ask-cli` package and all its dependencies.
6. **transformers downgrade.** `pip install "transformers>=4.45.0,<5.0"` to work around VLM compatibility (see Section 6.1).
7. **Config write.** Copy the appropriate tier profile to `~/.ask/config.yaml` with hardware metadata comments.
8. **Classifier download.** `huggingface_hub.snapshot_download()` for Phi-3.5-mini (~2GB). This is the only model downloaded during setup; specialists are lazy-loaded on first use.
9. **Shell alias.** Add `alias ask='~/.ask/venv/bin/ask'` to `.zshrc`, `.bashrc`, or `.profile`. Updates existing alias if present.
10. **Smoke test.** Run `ask --version` to verify the installation.

---

## 10. Data Flow for a Query

### One-shot: `ask "What is a closure?"`

```
1. cli.py:main()
   - sys.argv[1] != "models", so use main parser
   - args.query = ["What", "is", "a", "closure?"]
   - query = "What is a closure?"

2. cli.py:_handle_query()
   - Calls router.route_query(config, query)

3. router.py:route_query()
   - No force_role, no image_path, no audio_path
   - Falls through to classify_query()

4. router.py:classify_query()
   - _load_classifier() loads Phi-3.5-mini (if not already loaded)
   - Formats CLASSIFICATION_PROMPT with query
   - Generates up to 5 tokens
   - Response: "GENERAL" -> returns "general"

5. cli.py:_handle_query() (continued)
   - Prints "[general]" to stderr
   - get_model_for_role(config, "general") -> {model: "Qwen3-8B-4bit", type: "mlx_lm", ...}
   - Calls models.run_query("general", model_cfg, query)

6. models.py:run_query()
   - model_type is "mlx_lm", role is "general"
   - Falls through to generate_text()

7. models.py:generate_text()
   - _ensure_model() unloads classifier's specialist slot (if any), loads Qwen3-8B
   - Model name contains "qwen3" and not "coder" -> appends " /no_think"
   - Formats via apply_chat_template()
   - mlx_lm.generate() produces raw text
   - _clean_response() strips any <think> blocks and special tokens
   - Returns clean response

8. cli.py:_handle_query() (continued)
   - Prints response to stdout
   - Process exits
```

### Interactive: `ask -i` followed by `/image photo.png describe this`

```
1. cli.py:main() -> _run_repl()
2. User types: "/image photo.png describe this"
3. Detected as /image command:
   - image_path = "photo.png"
   - query = "describe this"
4. _handle_query() called with image_path set
5. router.route_query() sees image_path -> returns "vision" (no classifier needed)
6. models.run_query() -> generate_vision()
7. _ensure_model() loads Qwen2.5-VL-7B (unloads any previous specialist)
8. mlx_vlm prompt formatting with image
9. Generation, cleanup, print
```

---

## 11. Model Type Dispatch Table

| `type` field in config | Framework | Loading | Generation | Notes |
|---|---|---|---|---|
| `mlx_lm` | mlx-lm | `mlx_lm.load()` returns `(model, tokenizer)` | `mlx_lm.generate()` returns `str` | Used for classifier, general, code |
| `mlx_vlm` | mlx-vlm | `mlx_vlm.load()` returns `(model, processor)` | `mlx_vlm.generate()` returns `GenerationResult` with `.text` | Different prompt formatting via `apply_chat_template` from `mlx_vlm.prompt_utils` |
| `mlx_qwen3_asr` | mlx-qwen3-asr | Not preloaded (subprocess) | `python -m mlx_qwen3_asr <file> --model <name> --stdout-only` | Model loading handled by the subprocess |

---

## 12. Design Decisions & Alternatives Considered

### Why local models, not API calls?
Privacy, offline capability, zero ongoing cost. The target user is a developer on an Apple Silicon Mac who wants a quick `ask` command without API keys or internet dependency.

### Why one model at a time?
On 16GB machines, two 7B models (~10GB) plus macOS overhead (~4GB) would exceed available memory and trigger swap. The classifier is small enough (~1.8GB) to co-reside with any specialist.

### Why Phi-3.5-mini as classifier instead of a smaller model?
Tried smaller options, but classification accuracy dropped on ambiguous queries. Phi-3.5-mini at 4-bit is fast enough (~200ms per classification) and accurate enough to avoid misrouting.

### Why subprocess for ASR?
`mlx-qwen3-asr` has its own model loading, audio preprocessing (via ffmpeg), and chunked decoding pipeline. There's no clean Python API to call it in-process. The subprocess approach also naturally isolates its memory usage.

### Why not use argparse subparsers for `ask models`?
argparse subparsers don't mix well with `nargs="*"` positional arguments. If `models` were declared as a subparser, then `ask "tell me about models"` would try to invoke the `models` subcommand. The `sys.argv[1] == "models"` check is simple and unambiguous -- the word "models" as the first argument is always a subcommand, never a query.

### Why YAML config instead of JSON or TOML?
YAML supports comments (JSON doesn't), and the config is simple enough that TOML's advantages don't matter. Comments let us embed hardware metadata and human-readable descriptions in the config file.

### Why alias instead of symlink for PATH?
Symlinks into a venv's `bin/` directory can break if the venv is recreated at a different path. An alias in `.zshrc` is easy to update and survives venv recreation. The downside is that `ask` only works in interactive shells (not in scripts), but that's acceptable for a CLI tool meant for human use.

---

## 13. Performance Characteristics (16GB M4)

| Model | Role | Throughput | Memory | First-load time |
|---|---|---|---|---|
| Phi-3.5-mini-instruct-4bit | classifier | ~41 tok/s | ~1.8GB | ~3s |
| Qwen3-8B-4bit | general | ~21 tok/s | ~4.7GB | ~5s |
| Qwen2.5-Coder-7B-Instruct-4bit | code | ~20 tok/s | ~4.4GB | ~5s |
| Qwen2.5-VL-7B-Instruct-4bit | vision | ~23 tok/s | ~5.9GB | ~7s |
| Qwen3-ASR-0.6B | audio | RTF 0.73x | ~0.6GB | ~2s |

First-load time includes HuggingFace cache lookup, weight deserialization, and Metal shader compilation. Subsequent loads in the same session are faster due to OS-level file caching.

---

## 14. Security Considerations

- No network calls at inference time. All models run locally.
- HuggingFace model downloads use HTTPS. Models are cached at `~/.cache/huggingface/`.
- `setup.sh` runs `curl` to install Homebrew (standard Homebrew installation method).
- Audio transcription uses `subprocess.run()` with a hardcoded command structure (`sys.executable`, `-m`, `mlx_qwen3_asr`). The audio path is passed as an argument, not interpolated into a shell string, so there's no command injection risk.
- Config YAML is loaded with `yaml.safe_load()`, not `yaml.load()`.

---

## 15. Future Considerations

These are areas that may need attention but have no planned timeline:

- **Streaming output.** Currently waits for full generation before printing. Both `mlx_lm` and `mlx_vlm` support streaming, which would improve perceived latency.
- **Conversation history.** Interactive mode is stateless -- each query starts fresh. Adding context/history would require managing token budgets and memory.
- **Custom roles.** Users might want roles beyond the four built-in ones (e.g., "translation", "summarization"). The config format supports this but the router's classification prompt and `VALID_ROLES` dict are hardcoded to four categories.
- **The transformers v4/v5 conflict** (Section 6.1) will need to be revisited as upstream libraries evolve.
- **Testing.** No automated tests exist. The project was validated through manual end-to-end testing of all four model types.
- **`rich` library.** Listed as a dependency but not used in the codebase yet. Originally intended for formatted output.
