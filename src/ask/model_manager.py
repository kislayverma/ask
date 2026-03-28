"""Manage model configuration: list, add, remove, download models."""

import sys

import yaml

from ask.config import CONFIG_PATH, load_config, load_tier_profile
from ask.hardware import detect_hardware

VALID_TYPES = {"mlx_lm", "mlx_vlm", "mlx_qwen3_asr"}
VALID_ROLES = {"general", "code", "vision", "audio"}


def _save_config(config: dict) -> None:
    """Write config back to ~/.ask/config.yaml, preserving comments at the top."""
    # Read existing file to preserve header comments
    header_lines = []
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            for line in f:
                if line.startswith("#"):
                    header_lines.append(line)
                else:
                    break

    with open(CONFIG_PATH, "w") as f:
        for line in header_lines:
            f.write(line)
        if header_lines:
            f.write("\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def list_models() -> None:
    """Print the current model configuration."""
    config = load_config()

    # Classifier
    clf = config.get("classifier", {})
    print(f"  classifier: {clf.get('model', '?')}")
    print(f"              type={clf.get('type', '?')}")
    print()

    # Roles
    roles = config.get("roles", {})
    for role_name in ("general", "code", "vision", "audio"):
        if role_name not in roles:
            print(f"  {role_name:10s}   (not configured)")
            continue
        r = roles[role_name]
        print(f"  {role_name:10s}   {r.get('model', '?')}")
        desc = r.get("description", "")
        if desc:
            print(f"              {desc}")
        print(f"              type={r.get('type', '?')}")
        print()


def add_model(role: str, model_id: str, model_type: str) -> None:
    """Set a role to use a specific model.

    Args:
        role: One of general, code, vision, audio.
        model_id: HuggingFace model ID (e.g. mlx-community/Qwen3-8B-4bit).
        model_type: One of mlx_lm, mlx_vlm, mlx_qwen3_asr.
    """
    if role not in VALID_ROLES:
        print(f"Error: invalid role '{role}'. Choose from: {', '.join(sorted(VALID_ROLES))}", file=sys.stderr)
        sys.exit(1)

    if model_type not in VALID_TYPES:
        print(f"Error: invalid type '{model_type}'. Choose from: {', '.join(sorted(VALID_TYPES))}", file=sys.stderr)
        sys.exit(1)

    config = load_config()
    config.setdefault("roles", {})[role] = {
        "model": model_id,
        "type": model_type,
        "description": f"Custom model (set via ask models add)",
    }
    _save_config(config)
    print(f"  {role} -> {model_id} (type={model_type})")
    print(f"\n  Config updated at {CONFIG_PATH}")
    print(f"  The model will download on first use. Run 'ask models download {role}' to pre-download.")


def remove_model(role: str) -> None:
    """Reset a role back to the default from the hardware tier profile."""
    if role not in VALID_ROLES:
        print(f"Error: invalid role '{role}'. Choose from: {', '.join(sorted(VALID_ROLES))}", file=sys.stderr)
        sys.exit(1)

    hw = detect_hardware()
    tier = hw["tier"]
    if tier == "unsupported":
        print("Error: could not detect hardware tier to determine defaults.", file=sys.stderr)
        sys.exit(1)

    profile = load_tier_profile(tier)
    default_model = profile.get("roles", {}).get(role)
    if not default_model:
        print(f"Error: no default model for role '{role}' in tier '{tier}'.", file=sys.stderr)
        sys.exit(1)

    config = load_config()
    config.setdefault("roles", {})[role] = default_model
    _save_config(config)
    print(f"  {role} -> {default_model['model']} (restored to {tier} default)")
    print(f"  Config updated at {CONFIG_PATH}")


def download_model(role: str) -> None:
    """Pre-download the model for a role so first use is instant."""
    config = load_config()
    roles = config.get("roles", {})

    if role not in roles:
        print(f"Error: role '{role}' not configured. Run 'ask models list' to see current config.", file=sys.stderr)
        sys.exit(1)

    model_cfg = roles[role]
    model_id = model_cfg["model"]
    model_type = model_cfg["type"]

    print(f"  Downloading {model_id}...")

    from huggingface_hub import snapshot_download
    snapshot_download(model_id)

    print(f"  Done. {role} model is ready for offline use.")
