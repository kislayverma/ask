# ask

A local AI assistant for Apple Silicon Macs. Type a question, and `ask` automatically routes it to the right model -- coding, vision, speech-to-text, or general chat -- all running offline on your hardware via [MLX](https://github.com/ml-explore/mlx).

## Quick Start

```bash
git clone https://github.com/kislayverma/ask.git
cd ask
./setup.sh
```

The setup script will:
1. Install system dependencies (Python 3, ffmpeg) via Homebrew if needed
2. Detect your hardware and select the best models for your RAM
3. Create a virtual environment and install everything
4. Download the classifier model (~2GB)
5. Add the `ask` command to your shell

## Usage

```bash
# Ask anything -- auto-routed to the right model
ask "What is a closure in JavaScript?"

# Force a specific model
ask --model code "Write a Python binary search function"

# Analyze an image
ask --image screenshot.png "What error is shown here?"

# Transcribe audio
ask --audio meeting.wav

# Interactive mode (model stays warm between queries)
ask -i
```

### Interactive Mode Commands

In interactive mode (`ask -i`), you can:

```
ask> What is quantum entanglement?          # auto-routed
ask> /image photo.png What's in this?       # vision
ask> /audio recording.wav                   # transcribe
ask> quit                                   # exit
```

## How It Works

```
Your query
    |
    v
[Router] -- Phi-3.5-mini classifies your query
    |
    +---> GENERAL  --> Qwen3 8B
    +---> CODE     --> Qwen2.5-Coder 7B
    +---> VISION   --> Qwen2.5-VL 7B
    +---> AUDIO    --> Qwen3-ASR 0.6B
```

The router uses **Phi-3.5-mini** (3.8B, 4-bit) as a fast classifier. It reads your query and decides which specialist model to invoke. Only one model is loaded at a time to fit comfortably in memory.

Explicit flags (`--image`, `--audio`, `--model`) bypass the classifier entirely.

## Model Tiers

The setup script auto-detects your RAM and picks the best models:

| RAM | Tier | General | Coding | Vision | Audio |
|-----|------|---------|--------|--------|-------|
| 8GB | `8gb` | Phi-3.5-mini 3.8B | Phi-3.5-mini 3.8B | Qwen2.5-VL 7B | Qwen3-ASR 0.6B |
| 16GB | `16gb` | Qwen3 8B | Qwen2.5-Coder 7B | Qwen2.5-VL 7B | Qwen3-ASR 0.6B |
| 32GB+ | `32gb` | Qwen3 8B | Qwen2.5-Coder 7B | Qwen2.5-VL 7B | Qwen3-ASR 1.7B |

Models download on first use (~4-6GB each). Only the classifier (~2GB) downloads during setup.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- 8GB+ unified memory
- ~20GB free disk space (for all models)

## Custom Models

You can swap any role to a different model using the `ask models` command:

```bash
# See what's currently configured
ask models list

# Swap the general model to Llama 3.2
ask models add general mlx-community/Llama-3.2-3B-Instruct-4bit --type mlx_lm

# Pre-download so first use is instant
ask models download general

# Reset back to the default for your hardware tier
ask models remove general
```

The `--type` flag tells ask which framework to use:

| Type | Use for |
|------|---------|
| `mlx_lm` | Text-only LLMs (general chat, coding) |
| `mlx_vlm` | Vision-language models (image understanding) |
| `mlx_qwen3_asr` | Qwen3 speech-to-text models |

Any model from the [mlx-community](https://huggingface.co/mlx-community) HuggingFace organization works. Use 4-bit quantized models (names ending in `-4bit`) to stay within memory limits.

Config lives at `~/.ask/config.yaml`. You can also edit it directly, or re-run `./setup.sh` to regenerate it from scratch.

## Project Structure

```
ask/
├── setup.sh                    # One-command setup
├── pyproject.toml              # Package definition
└── src/ask/
    ├── cli.py                  # CLI entry point (one-shot + REPL)
    ├── router.py               # LLM-based query classifier
    ├── models.py               # Model loader/cache manager
    ├── model_manager.py        # Model add/remove/download commands
    ├── config.py               # Config file handling
    ├── hardware.py             # Hardware detection
    └── model_profiles/         # Default model selections per RAM tier
        ├── tier_8gb.yaml
        ├── tier_16gb.yaml
        └── tier_32gb.yaml
```

## License

MIT
