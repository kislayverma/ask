"""Load and manage the ask configuration stored at ~/.ask/config.yaml."""

import os
import sys
from pathlib import Path

import yaml


ASK_HOME = Path.home() / ".ask"
CONFIG_PATH = ASK_HOME / "config.yaml"
PROFILES_DIR = Path(__file__).resolve().parent / "model_profiles"


def load_config() -> dict:
    """Load the resolved config from ~/.ask/config.yaml.

    Exits with a helpful message if the config doesn't exist (setup not run).
    """
    if not CONFIG_PATH.exists():
        print(
            "Error: ask is not set up yet.\n"
            "Run the setup script first:\n\n"
            "  cd /path/to/ask && ./setup.sh\n",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_tier_profile(tier: str) -> dict:
    """Load a model tier profile from model_profiles/<tier>.yaml."""
    profile_path = PROFILES_DIR / f"tier_{tier}.yaml"
    if not profile_path.exists():
        print(f"Error: unknown tier '{tier}'. No profile at {profile_path}", file=sys.stderr)
        sys.exit(1)

    with open(profile_path) as f:
        return yaml.safe_load(f)


def write_config(config: dict) -> None:
    """Write the resolved config to ~/.ask/config.yaml."""
    ASK_HOME.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_model_for_role(config: dict, role: str) -> dict:
    """Return the model config dict for a given role (general, code, vision, audio)."""
    roles = config.get("roles", {})
    if role not in roles:
        print(f"Error: unknown role '{role}'. Available: {list(roles.keys())}", file=sys.stderr)
        sys.exit(1)
    return roles[role]


def get_classifier_model(config: dict) -> dict:
    """Return the classifier model config dict."""
    return config.get("classifier", {})
