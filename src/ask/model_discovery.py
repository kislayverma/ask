"""Discover the best MLX models for each role by querying HuggingFace.

Called during setup to recommend specialist models based on the user's
hardware. Falls back to hardcoded tier profiles if the API is unreachable.

Usage (from setup.sh):
    python -m ask.model_discovery \
        --ram 16 --tier 16gb --chip "Apple M4" --gpu-cores 10 \
        --config ~/.ask/config.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml

from ask.config import PROFILES_DIR, ASK_HOME

# ─── Constants ────────────────────────────────────────────────────────────────

# Fixed classifier -- updating this is a release-level decision.
CLASSIFIER = {
    "model": "mlx-community/Phi-3.5-mini-instruct-4bit",
    "type": "mlx_lm",
    "description": "Phi-3.5-mini 3.8B (4-bit) -- fast classifier, ~1.8GB",
}
CLASSIFIER_SIZE_GB = 1.8

# macOS base memory overhead (conservative estimate).
OS_OVERHEAD_GB = 4.0

# How many candidates to show the user per role when they want to override.
MAX_ALTERNATIVES = 3

# Quantization suffixes in order of preference (smallest first for tight budgets).
QUANT_SUFFIXES = ["-4bit", "-3bit", "-8bit", "-bf16", "-fp16"]

# ─── Role search definitions ─────────────────────────────────────────────────

# Each role defines how to search HuggingFace for candidate models.
#   author:       HuggingFace org to search in
#   pipeline_tag: the HF pipeline tag to filter on
#   name_must:    substrings that MUST appear in the model ID (any one match)
#   name_must_not: substrings that must NOT appear in the model ID
#   type:         the ask framework type (mlx_lm, mlx_vlm, mlx_qwen3_asr)
#   fallback_authors: additional authors to search if mlx-community has nothing

ROLE_SEARCH = {
    "general": {
        "author": "mlx-community",
        "pipeline_tag": "text-generation",
        "name_must": [],  # no specific name requirement
        "name_must_not": ["coder", "code-", "codestral"],
        "type": "mlx_lm",
    },
    "code": {
        "author": "mlx-community",
        "pipeline_tag": "text-generation",
        "name_must": ["coder", "code-", "codestral"],
        "name_must_not": [],
        "type": "mlx_lm",
    },
    "vision": {
        "author": "mlx-community",
        "pipeline_tag": "image-text-to-text",
        "name_must": [],
        "name_must_not": [],
        "type": "mlx_vlm",
    },
    "audio": {
        "author": "mlx-community",
        "pipeline_tag": "automatic-speech-recognition",
        # mlx_qwen3_asr ONLY supports Qwen3-ASR models. Other ASR architectures
        # (Parakeet, Whisper, etc.) will fail with parameter mismatch errors.
        "name_must": ["qwen3-asr", "qwen3_asr"],
        "name_must_not": [],
        "type": "mlx_qwen3_asr",
        "fallback_authors": ["Qwen"],
        "fallback_name_must": ["Qwen3-ASR", "qwen3-asr"],
    },
}


# ─── HuggingFace API queries ─────────────────────────────────────────────────

def _query_models(author, pipeline_tag, limit=50):
    """Query HuggingFace for models matching the given criteria.

    Returns a list of ModelInfo objects sorted by downloads (descending).
    Returns an empty list if the API is unreachable.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return []

    api = HfApi()
    try:
        models = list(api.list_models(
            author=author,
            pipeline_tag=pipeline_tag,
            sort="downloads",
            limit=limit,
        ))
        return models
    except Exception:
        return []


def _estimate_size_gb(model_info):
    """Estimate model size in GB from HuggingFace metadata.

    Tries safetensors metadata first, then falls back to sibling file sizes.
    Returns None if size cannot be determined.
    """
    # Try safetensors parameter count -> rough size estimate
    if hasattr(model_info, "safetensors") and model_info.safetensors:
        st = model_info.safetensors
        # safetensors has a 'total' field (total bytes) on some models,
        # or parameter_count dict like {"F16": 8000000000}
        if hasattr(st, "total") and st.total:
            return st.total / (1024 ** 3)
        if hasattr(st, "parameter_count") and st.parameter_count:
            total_params = sum(st.parameter_count.values())
            # For 4-bit quant: ~0.55GB per billion params
            # For 8-bit: ~1.1GB per billion params
            # Default to 4-bit estimate since we prefer 4-bit models
            model_id_lower = model_info.id.lower()
            if "8bit" in model_id_lower or "8-bit" in model_id_lower:
                return total_params * 1.1 / 1e9
            elif "bf16" in model_id_lower or "fp16" in model_id_lower:
                return total_params * 2.0 / 1e9
            else:
                return total_params * 0.55 / 1e9

    # Try summing sibling file sizes
    if hasattr(model_info, "siblings") and model_info.siblings:
        total = 0
        for f in model_info.siblings:
            if hasattr(f, "size") and f.size:
                total += f.size
        if total > 0:
            return total / (1024 ** 3)

    # Last resort: parse parameter count from model name
    return _estimate_size_from_name(model_info.id)


def _estimate_size_from_name(model_id):
    """Crude size estimate from model name patterns like '8B-4bit', '1.7B'.

    Returns estimated GB or None.
    """
    import re
    model_id_lower = model_id.lower()

    # Match patterns like "8b", "7b", "3.8b", "0.6b", "1.7b", "70b"
    match = re.search(r"[\-_](\d+(?:\.\d+)?)b[\-_]", model_id_lower)
    if not match:
        match = re.search(r"[\-_](\d+(?:\.\d+)?)b$", model_id_lower)
    if not match:
        # Try without separators: "Qwen3-8B" -> find "8b"
        match = re.search(r"(\d+(?:\.\d+)?)b", model_id_lower)

    if not match:
        return None

    params_b = float(match.group(1))

    # Estimate based on quantization
    if "4bit" in model_id_lower or "4-bit" in model_id_lower:
        return params_b * 0.55
    elif "3bit" in model_id_lower:
        return params_b * 0.45
    elif "8bit" in model_id_lower or "8-bit" in model_id_lower:
        return params_b * 1.1
    elif "bf16" in model_id_lower or "fp16" in model_id_lower:
        return params_b * 2.0
    else:
        # Default to 4-bit assumption for mlx-community models
        return params_b * 0.55


def _is_quantized_4bit(model_id):
    """Check if a model is 4-bit quantized (preferred for memory efficiency)."""
    lower = model_id.lower()
    return "4bit" in lower or "4-bit" in lower


def _matches_name_filter(model_id, must_have, must_not_have):
    """Check if model_id matches the name filters for a role."""
    lower = model_id.lower()

    if must_not_have:
        for term in must_not_have:
            if term.lower() in lower:
                return False

    if must_have:
        return any(term.lower() in lower for term in must_have)

    return True


# ─── Candidate discovery per role ─────────────────────────────────────────────

def discover_candidates(role, budget_gb, tier):
    """Find candidate models for a role that fit within the memory budget.

    Returns a list of dicts: [{"model": str, "type": str, "size_gb": float,
                               "downloads": int, "description": str}, ...]
    Sorted by downloads descending. Empty list if nothing found.
    """
    search = ROLE_SEARCH[role]
    candidates = []

    # Primary search
    models = _query_models(search["author"], search["pipeline_tag"])

    # For audio, also try fallback authors if primary yields nothing usable
    if not models and "fallback_authors" in search:
        for author in search["fallback_authors"]:
            models = _query_models(author, search["pipeline_tag"])
            if models:
                break

    # If primary search found nothing with pipeline_tag for audio,
    # try fallback authors with name-based matching
    if not models and "fallback_authors" in search:
        for author in search["fallback_authors"]:
            all_models = _query_models(author, pipeline_tag=None)
            name_filters = search.get("fallback_name_must", [])
            models = [m for m in all_models
                      if any(n.lower() in m.id.lower() for n in name_filters)]
            if models:
                break

    for m in models:
        if not _matches_name_filter(m.id, search["name_must"], search["name_must_not"]):
            continue

        size = _estimate_size_gb(m)
        if size is None:
            continue

        if size > budget_gb:
            continue

        # For 8GB tier, strongly prefer 4-bit models
        if tier == "8gb" and not _is_quantized_4bit(m.id):
            continue

        # For 16GB tier, prefer 4-bit but allow 8-bit if they fit
        # (4-bit models will naturally rank higher since more fit)

        downloads = getattr(m, "downloads", 0) or 0
        description = f"{m.id.split('/')[-1]} -- ~{size:.1f}GB, {downloads:,} downloads"

        candidates.append({
            "model": m.id,
            "type": search["type"],
            "size_gb": round(size, 1),
            "downloads": downloads,
            "description": description,
        })

    # Sort by downloads descending
    candidates.sort(key=lambda c: c["downloads"], reverse=True)
    return candidates


# ─── Memory budget ────────────────────────────────────────────────────────────

def compute_budget(ram_gb):
    """Compute the maximum specialist model size in GB.

    Budget = total_RAM - classifier - OS_overhead
    The classifier stays resident alongside the specialist (in interactive mode).
    """
    budget = ram_gb - CLASSIFIER_SIZE_GB - OS_OVERHEAD_GB
    return max(budget, 0)


# ─── Tier profile fallback ────────────────────────────────────────────────────

def _load_fallback(tier):
    """Load the hardcoded tier profile as a fallback."""
    profile_path = PROFILES_DIR / f"tier_{tier}.yaml"
    if not profile_path.exists():
        # Shouldn't happen, but degrade gracefully
        profile_path = PROFILES_DIR / "tier_16gb.yaml"
    with open(profile_path) as f:
        return yaml.safe_load(f)


# ─── Interactive UI ───────────────────────────────────────────────────────────

# ANSI colors (matching setup.sh style)
_BLUE = "\033[0;34m"
_GREEN = "\033[0;32m"
_YELLOW = "\033[0;33m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_NC = "\033[0m"


def _print_header(ram_gb, tier, budget_gb):
    """Print the discovery header."""
    print()
    print(f"{_BOLD}Model Discovery{_NC}")
    print(f"  RAM: {ram_gb}GB | Tier: {tier} | Specialist budget: {budget_gb:.1f}GB")
    print()


def _print_recommendations(recommendations):
    """Print the recommended model for each role."""
    print(f"{_BOLD}Recommended models:{_NC}")
    print()
    for role, rec in recommendations.items():
        source = rec.get("_source", "discovered")
        tag = f" {_DIM}(default){_NC}" if source == "fallback" else ""
        print(f"  {_BLUE}{role:8s}{_NC}  {rec['model']}")
        print(f"           {_DIM}{rec['description']}{_NC}{tag}")
    print()


def _ask_accept_defaults(recommendations):
    """Ask user to accept defaults or override individual roles.

    Returns the final selections dict (role -> model config).
    """
    _print_recommendations(recommendations)

    try:
        answer = input(f"Accept these models? [{_GREEN}Y{_NC}/n/role name to change] ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return recommendations

    if not answer or answer.lower() in ("y", "yes"):
        return recommendations

    # User wants to change something
    if answer.lower() in ("n", "no"):
        # Let them change each role
        for role in recommendations:
            recommendations = _override_role(role, recommendations)
        return recommendations

    # User typed a specific role name
    role = answer.lower()
    if role in recommendations:
        recommendations = _override_role(role, recommendations)
        # Ask again for further changes
        return _ask_accept_defaults(recommendations)
    else:
        print(f"  Unknown role '{answer}'. Available: {', '.join(recommendations.keys())}")
        return _ask_accept_defaults(recommendations)


def _override_role(role, recommendations):
    """Show alternatives for a role and let the user pick."""
    alternatives = recommendations[role].get("_alternatives", [])
    if not alternatives:
        print(f"  {_YELLOW}No alternatives found for {role}. Keeping current selection.{_NC}")
        return recommendations

    print()
    print(f"  {_BOLD}Alternatives for {role}:{_NC}")
    current = recommendations[role]["model"]
    for i, alt in enumerate(alternatives, 1):
        marker = f" {_GREEN}<-- current{_NC}" if alt["model"] == current else ""
        print(f"    {i}. {alt['model']}")
        print(f"       {_DIM}{alt['description']}{_NC}{marker}")
    print(f"    0. Keep current ({current})")
    print()

    try:
        choice = input(f"  Pick [0-{len(alternatives)}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return recommendations

    if not choice or choice == "0":
        return recommendations

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(alternatives):
            picked = alternatives[idx]
            recommendations[role] = {
                "model": picked["model"],
                "type": picked["type"],
                "description": picked["description"],
                "_alternatives": alternatives,
                "_source": "user",
            }
            print(f"  {_GREEN}Set {role} -> {picked['model']}{_NC}")
    except ValueError:
        print(f"  Invalid choice. Keeping current.")

    return recommendations


# ─── Main discovery flow ──────────────────────────────────────────────────────

def run_discovery(ram_gb, tier, chip, gpu_cores, config_path):
    """Main entry point: discover models, get user confirmation, write config.

    Returns the final config dict.
    """
    budget_gb = compute_budget(ram_gb)
    fallback = _load_fallback(tier)
    roles = ["general", "code", "vision", "audio"]

    _print_header(ram_gb, tier, budget_gb)
    print(f"{_BLUE}[info]{_NC}  Searching HuggingFace for the latest MLX models...")
    print()

    recommendations = {}
    api_available = False

    for role in roles:
        candidates = discover_candidates(role, budget_gb, tier)

        if candidates:
            api_available = True
            top = candidates[0]
            recommendations[role] = {
                "model": top["model"],
                "type": top["type"],
                "description": top["description"],
                "_alternatives": candidates[:MAX_ALTERNATIVES],
                "_source": "discovered",
            }
        else:
            # Fall back to tier profile
            fb = fallback["roles"][role]
            recommendations[role] = {
                "model": fb["model"],
                "type": fb["type"],
                "description": fb["description"],
                "_alternatives": [],
                "_source": "fallback",
            }

    if not api_available:
        print(f"  {_YELLOW}Could not reach HuggingFace API. Using default models for {tier} tier.{_NC}")
        print()

    # Let user confirm or override
    final = _ask_accept_defaults(recommendations)

    # Build config dict
    config = {
        "classifier": dict(CLASSIFIER),
        "roles": {},
    }

    for role in roles:
        sel = final[role]
        config["roles"][role] = {
            "model": sel["model"],
            "type": sel["type"],
            "description": sel["description"],
        }

    # Write config file
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"# ask configuration -- generated by setup.sh\n"
        f"# Hardware: {chip}, {ram_gb}GB RAM, {gpu_cores} GPU cores\n"
        f"# Tier: {tier}\n"
        f"# Re-run setup.sh to regenerate this file.\n\n"
    )

    with open(config_file, "w") as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"{_GREEN}[ok]{_NC}    Config written to {config_file}")
    return config


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Discover optimal MLX models for this machine."
    )
    parser.add_argument("--ram", type=int, required=True, help="Total RAM in GB")
    parser.add_argument("--tier", required=True, help="Hardware tier (8gb/16gb/32gb)")
    parser.add_argument("--chip", default="unknown", help="Chip name")
    parser.add_argument("--gpu-cores", default="unknown", help="GPU core count")
    parser.add_argument("--config", required=True, help="Path to write config.yaml")

    args = parser.parse_args()
    run_discovery(args.ram, args.tier, args.chip, args.gpu_cores, args.config)


if __name__ == "__main__":
    main()
