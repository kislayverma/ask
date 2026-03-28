"""CLI entry point for the ask tool.

Usage:
    ask "What is a closure in JavaScript?"           # one-shot, auto-routed
    ask --model code "Write a binary search"         # one-shot, force coding model
    ask --image photo.png "What's in this image?"    # one-shot, vision model
    ask --audio meeting.wav                          # one-shot, transcribe audio
    ask -i                                           # interactive REPL mode
    ask models list                                  # show current model config
    ask models add <role> <model-id> --type mlx_lm   # swap in a custom model
    ask models remove <role>                         # reset role to tier default
    ask models download <role>                       # pre-download a model
"""

import argparse
import os
import sys
import warnings

# Suppress noisy warnings from transformers/model configs before any imports
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*rope_parameters.*")
warnings.filterwarnings("ignore", message=".*Calling.*deprecated.*")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from ask import __version__
from ask.config import get_model_for_role, load_config
from ask.models import run_query
from ask.router import route_query


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ask",
        description="Local AI assistant that auto-routes queries to the right model.",
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="The question or prompt to send to the AI.",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Start an interactive REPL session.",
    )
    parser.add_argument(
        "--image",
        metavar="PATH",
        help="Path to an image file. Forces the vision model.",
    )
    parser.add_argument(
        "--audio",
        metavar="PATH",
        help="Path to an audio file. Forces the ASR model.",
    )
    parser.add_argument(
        "--model",
        metavar="ROLE",
        choices=["general", "code", "vision", "audio"],
        help="Force a specific model role (general, code, vision, audio).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate (default: 1024).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"ask {__version__}",
    )
    return parser


def _build_models_parser() -> argparse.ArgumentParser:
    """Build the argument parser for 'ask models' subcommands."""
    parser = argparse.ArgumentParser(
        prog="ask models",
        description="Manage model configuration.",
    )
    sub = parser.add_subparsers(dest="action")

    # ask models list
    sub.add_parser("list", help="Show current model configuration.")

    # ask models add <role> <model_id> --type <type>
    add_p = sub.add_parser("add", help="Set a role to use a specific model.")
    add_p.add_argument("role", choices=["general", "code", "vision", "audio"])
    add_p.add_argument("model_id", help="HuggingFace model ID (e.g. mlx-community/Qwen3-8B-4bit)")
    add_p.add_argument(
        "--type", required=True, dest="model_type",
        choices=["mlx_lm", "mlx_vlm", "mlx_qwen3_asr"],
        help="Model framework type.",
    )

    # ask models remove <role>
    rm_p = sub.add_parser("remove", help="Reset a role to the default model for your hardware tier.")
    rm_p.add_argument("role", choices=["general", "code", "vision", "audio"])

    # ask models download <role>
    dl_p = sub.add_parser("download", help="Pre-download the model for a role.")
    dl_p.add_argument("role", choices=["general", "code", "vision", "audio"])

    return parser


def _handle_query(
    config: dict,
    query: str,
    image_path: str | None = None,
    audio_path: str | None = None,
    force_role: str | None = None,
    max_tokens: int = 1024,
) -> None:
    """Route a query and print the response."""
    # Determine which model to use
    role = route_query(config, query, image_path=image_path, audio_path=audio_path, force_role=force_role)
    print(f"  [{role}]", file=sys.stderr)

    # Get model config for this role
    model_cfg = get_model_for_role(config, role)

    # Generate response
    response = run_query(
        role,
        model_cfg,
        query,
        image_path=image_path,
        audio_path=audio_path,
        max_tokens=max_tokens,
    )

    print(response)


def _run_repl(config: dict, max_tokens: int = 1024) -> None:
    """Run an interactive REPL loop."""
    print("ask interactive mode (type 'quit' or Ctrl-D to exit)", file=sys.stderr)
    print("Tip: prefix with /image <path> or /audio <path> for multimodal queries\n", file=sys.stderr)

    while True:
        try:
            user_input = input("ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.", file=sys.stderr)
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.", file=sys.stderr)
            break

        # Parse REPL-specific commands
        image_path = None
        audio_path = None
        query = user_input

        if user_input.startswith("/image "):
            parts = user_input[7:].strip().split(maxsplit=1)
            if len(parts) >= 1:
                image_path = parts[0]
                query = parts[1] if len(parts) > 1 else "Describe this image."

        elif user_input.startswith("/audio "):
            parts = user_input[7:].strip().split(maxsplit=1)
            if len(parts) >= 1:
                audio_path = parts[0]
                query = ""

        try:
            _handle_query(
                config,
                query,
                image_path=image_path,
                audio_path=audio_path,
                max_tokens=max_tokens,
            )
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)

        print()  # blank line between responses


def _handle_models_command(argv: list[str]) -> None:
    """Handle 'ask models ...' subcommands."""
    from ask.model_manager import add_model, download_model, list_models, remove_model

    parser = _build_models_parser()
    args = parser.parse_args(argv)

    if args.action is None:
        parser.print_help()
        sys.exit(1)

    if args.action == "list":
        list_models()
    elif args.action == "add":
        add_model(args.role, args.model_id, args.model_type)
    elif args.action == "remove":
        remove_model(args.role)
    elif args.action == "download":
        download_model(args.role)


def main():
    """Main entry point for the ask CLI."""
    # Intercept 'ask models ...' before the main parser sees it.
    # This avoids 'models' being consumed as part of a query string.
    if len(sys.argv) >= 2 and sys.argv[1] == "models":
        _handle_models_command(sys.argv[2:])
        return

    parser = _build_parser()
    args = parser.parse_args()

    # Load config (exits with helpful message if setup not run)
    config = load_config()

    if args.interactive:
        _run_repl(config, max_tokens=args.max_tokens)
        return

    # One-shot mode: need a query (or --audio)
    query = " ".join(args.query) if args.query else ""

    if not query and not args.audio:
        parser.print_help()
        sys.exit(1)

    _handle_query(
        config,
        query,
        image_path=args.image,
        audio_path=args.audio,
        force_role=args.model,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
