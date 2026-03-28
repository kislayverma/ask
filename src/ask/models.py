"""Model manager: load, cache, and run specialist models on demand.

Only one specialist model is kept in memory at a time. When a different
role is requested, the previous model is unloaded first.
"""

import gc
import re
import subprocess
import sys
import warnings
from pathlib import Path

import mlx.core as mx

# Module-level cache: the currently loaded specialist model.
_current_role: str | None = None
_current_model = None
_current_processor = None  # tokenizer for mlx_lm, processor for mlx_vlm
_current_type: str | None = None


def _clean_response(text: str) -> str:
    """Clean model output by removing special tokens and thinking blocks.

    Handles:
    - Qwen3's <think>...</think> blocks (chain-of-thought that shouldn't be shown)
    - Special tokens like <|im_end|>, <|endoftext|>, etc.
    """
    # Remove <think>...</think> blocks (Qwen3 thinking mode)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove common special tokens
    special_tokens = [
        "<|im_end|>", "<|im_start|>", "<|endoftext|>",
        "<|end|>", "</s>", "<s>",
    ]
    for token in special_tokens:
        text = text.replace(token, "")

    return text.strip()


def _unload_current():
    """Unload the currently cached specialist model to free memory."""
    global _current_role, _current_model, _current_processor, _current_type
    _current_role = None
    _current_model = None
    _current_processor = None
    _current_type = None
    gc.collect()
    mx.metal.clear_cache()


def _ensure_model(role: str, model_cfg: dict):
    """Load the model for `role` if it's not already cached."""
    global _current_role, _current_model, _current_processor, _current_type

    if _current_role == role:
        return  # Already loaded

    if _current_role is not None:
        _unload_current()

    model_name = model_cfg["model"]
    model_type = model_cfg["type"]

    print(f"  Loading {role} model ({model_name})...", file=sys.stderr, end="", flush=True)

    import io
    import logging
    logging.disable(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _orig_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            if model_type == "mlx_lm":
                from mlx_lm import load
                _current_model, _current_processor = load(model_name)

            elif model_type == "mlx_vlm":
                from mlx_vlm import load
                _current_model, _current_processor = load(model_name)

            else:
                # For mlx_qwen3_asr, we don't preload -- it's invoked as a subprocess.
                _current_model = "subprocess"
                _current_processor = None
        finally:
            sys.stderr = _orig_stderr
            logging.disable(logging.NOTSET)

    _current_role = role
    _current_type = model_type
    print(" done.", file=sys.stderr)


def generate_text(role: str, model_cfg: dict, prompt: str, max_tokens: int = 1024) -> str:
    """Generate a text response using an mlx_lm model.

    Uses the tokenizer's chat template to properly format the prompt,
    which handles Qwen3's thinking mode and other model-specific formatting.
    """
    _ensure_model(role, model_cfg)

    from mlx_lm import generate

    # Build a chat-formatted prompt using the tokenizer's template.
    # Adding /no_think suppresses Qwen3's chain-of-thought output.
    # Only add it for models that support thinking mode (Qwen3).
    model_name = model_cfg.get("model", "").lower()
    content = prompt
    if "qwen3" in model_name and "coder" not in model_name:
        content = prompt + " /no_think"

    messages = [
        {"role": "user", "content": content},
    ]

    if hasattr(_current_processor, "apply_chat_template"):
        formatted = _current_processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
    else:
        # Fallback for tokenizers without chat template
        formatted = prompt

    raw = generate(
        _current_model,
        _current_processor,
        prompt=formatted,
        max_tokens=max_tokens,
        verbose=False,
    )
    return _clean_response(raw)


def generate_vision(
    role: str, model_cfg: dict, prompt: str, image_path: str, max_tokens: int = 1024
) -> str:
    """Generate a response from an image + text prompt using an mlx_vlm model."""
    _ensure_model(role, model_cfg)

    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    config = load_config(model_cfg["model"])
    formatted_prompt = apply_chat_template(
        _current_processor,
        config,
        prompt,
        images=[image_path],
        num_images=1,
    )
    result = generate(
        _current_model,
        _current_processor,
        formatted_prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    # mlx_vlm.generate returns a GenerationResult object
    raw = result.text if hasattr(result, "text") else str(result)
    return _clean_response(raw)


def transcribe_audio(model_cfg: dict, audio_path: str) -> str:
    """Transcribe an audio file using mlx-qwen3-asr as a subprocess.

    We use subprocess here because mlx-qwen3-asr manages its own model
    loading and audio processing pipeline.
    """
    model_name = model_cfg["model"]
    cmd = [
        sys.executable, "-m", "mlx_qwen3_asr",
        audio_path,
        "--model", model_name,
        "--stdout-only",
    ]

    print(f"  Transcribing with {model_name}...", file=sys.stderr, flush=True)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  ASR error: {result.stderr.strip()}", file=sys.stderr)
            return f"[Transcription failed: {result.stderr.strip()}]"
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[Transcription timed out after 10 minutes]"


def run_query(
    role: str,
    model_cfg: dict,
    query: str,
    image_path: str | None = None,
    audio_path: str | None = None,
    max_tokens: int = 1024,
) -> str:
    """Run a query against the appropriate model based on role.

    This is the main dispatch function that cli.py calls after routing.
    """
    model_type = model_cfg["type"]

    if role == "audio" or model_type == "mlx_qwen3_asr":
        if not audio_path:
            return "[Error: audio role requires an audio file path]"
        return transcribe_audio(model_cfg, audio_path)

    if role == "vision" or model_type == "mlx_vlm":
        if not image_path:
            # Fall back to text-only with the VLM
            _ensure_model(role, model_cfg)
            from mlx_vlm import generate
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config

            config = load_config(model_cfg["model"])
            formatted_prompt = apply_chat_template(
                _current_processor, config, query, num_images=0,
            )
            result = generate(
                _current_model, _current_processor, formatted_prompt,
                max_tokens=max_tokens, verbose=False,
            )
            raw = result.text if hasattr(result, "text") else str(result)
            return _clean_response(raw)
        return generate_vision(role, model_cfg, query, image_path, max_tokens)

    # Default: text generation with mlx_lm
    return generate_text(role, model_cfg, query, max_tokens)
