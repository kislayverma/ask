"""Route user queries to the appropriate specialist model using an LLM classifier."""

import sys
import warnings

from ask.config import get_classifier_model

# The classifier model and tokenizer are cached at module level
# so repeated classifications don't reload the model.
_classifier_model = None
_classifier_tokenizer = None

CLASSIFICATION_PROMPT = """\
You are a query classifier. Given a user query, classify it into exactly one category.
Reply with ONLY the category name, nothing else.

Categories:
- CODE: programming, debugging, code review, writing functions, fixing bugs, software questions
- VISION: analyzing images, describing photos, OCR, reading screenshots, visual questions
- AUDIO: transcribing audio, speech-to-text, listening to recordings
- GENERAL: everything else (chat, reasoning, knowledge, math, writing, explanations)

User query: {query}

Category:"""

VALID_ROLES = {"GENERAL": "general", "CODE": "code", "VISION": "vision", "AUDIO": "audio"}


def _load_classifier(config: dict):
    """Load the classifier model (Phi-3.5-mini) into memory."""
    global _classifier_model, _classifier_tokenizer

    if _classifier_model is not None:
        return

    classifier_cfg = get_classifier_model(config)
    model_name = classifier_cfg["model"]

    print(f"  Loading classifier ({model_name})...", file=sys.stderr, end="", flush=True)

    import logging
    logging.disable(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Temporarily suppress stderr to catch noisy model config messages
        import io
        _orig_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            from mlx_lm import load
            _classifier_model, _classifier_tokenizer = load(model_name)
        finally:
            sys.stderr = _orig_stderr
            logging.disable(logging.NOTSET)

    print(" done.", file=sys.stderr)


def classify_query(config: dict, query: str) -> str:
    """Classify a query into a role: 'general', 'code', 'vision', or 'audio'.

    Uses the classifier LLM (Phi-3.5-mini) to determine which specialist
    model should handle the query.

    Returns one of: 'general', 'code', 'vision', 'audio'.
    """
    _load_classifier(config)

    from mlx_lm import generate

    prompt = CLASSIFICATION_PROMPT.format(query=query)
    response = generate(
        _classifier_model,
        _classifier_tokenizer,
        prompt=prompt,
        max_tokens=5,
        verbose=False,
    )

    # Parse the response -- extract the first valid category
    response_upper = response.strip().upper()
    for keyword, role in VALID_ROLES.items():
        if keyword in response_upper:
            return role

    # Default to general if classification is unclear
    return "general"


def route_query(
    config: dict,
    query: str,
    image_path: str | None = None,
    audio_path: str | None = None,
    force_role: str | None = None,
) -> str:
    """Determine which role should handle this query.

    Priority:
    1. Explicit --model flag (force_role)
    2. --image flag or image file detected -> vision
    3. --audio flag or audio file detected -> audio
    4. LLM classification of the text query

    Returns one of: 'general', 'code', 'vision', 'audio'.
    """
    # 1. Explicit override
    if force_role:
        if force_role in VALID_ROLES.values():
            return force_role
        print(
            f"Warning: unknown role '{force_role}', falling back to classifier.",
            file=sys.stderr,
        )

    # 2. Image provided
    if image_path:
        return "vision"

    # 3. Audio provided
    if audio_path:
        return "audio"

    # 4. LLM classification
    return classify_query(config, query)
