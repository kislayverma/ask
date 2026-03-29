#!/usr/bin/env bash
#
# setup.sh -- One-command setup for the ask CLI tool.
#
# What it does:
#   1. Checks for Apple Silicon Mac
#   2. Installs system dependencies (Homebrew, Python 3, ffmpeg) if missing
#   3. Detects hardware (chip, RAM) and selects the best model tier
#   4. Creates a Python virtual environment at ~/.ask/venv/
#   5. Installs the ask package and all dependencies
#   6. Downloads the classifier model (Phi-3.5-mini, ~2GB)
#   7. Writes config to ~/.ask/config.yaml
#   8. Adds ask to your PATH
#   9. Runs a smoke test
#
# Usage:
#   git clone https://github.com/kislayverma/ask.git
#   cd ask
#   ./setup.sh
#

set -euo pipefail

# ─── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'  # No Color

info()    { echo -e "${BLUE}[info]${NC}  $*"; }
success() { echo -e "${GREEN}[ok]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC}  $*"; }
fail()    { echo -e "${RED}[fail]${NC}  $*"; exit 1; }

ASK_HOME="$HOME/.ask"
VENV_DIR="$ASK_HOME/venv"
CONFIG_FILE="$ASK_HOME/config.yaml"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─── Step 0: Check platform ───────────────────────────────────────────────────
echo ""
echo -e "${BOLD}ask setup${NC}"
echo "========="
echo ""

ARCH=$(uname -m)
OS=$(uname -s)

if [[ "$OS" != "Darwin" ]]; then
    fail "ask currently requires macOS with Apple Silicon. Detected: $OS"
fi

if [[ "$ARCH" != "arm64" ]]; then
    fail "ask requires Apple Silicon (arm64). Detected: $ARCH"
fi

success "macOS Apple Silicon detected"

# ─── Step 1: Install system dependencies ──────────────────────────────────────
info "Checking system dependencies..."

# Homebrew
if ! command -v brew &>/dev/null; then
    info "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    eval "$(/opt/homebrew/bin/brew shellenv)"
    success "Homebrew installed"
else
    success "Homebrew found"
fi

# Python 3
if ! command -v python3 &>/dev/null; then
    info "Installing Python 3..."
    brew install python3
    success "Python 3 installed"
else
    success "Python 3 found ($(python3 --version))"
fi

# ffmpeg (required by mlx-qwen3-asr)
if ! command -v ffmpeg &>/dev/null; then
    info "Installing ffmpeg..."
    brew install ffmpeg
    success "ffmpeg installed"
else
    success "ffmpeg found"
fi

echo ""

# ─── Step 2: Detect hardware ─────────────────────────────────────────────────
info "Detecting hardware..."

CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
RAM_GB=$((RAM_BYTES / 1073741824))

# Determine GPU cores
GPU_CORES=$(system_profiler SPDisplaysDataType 2>/dev/null | grep "Total Number of Cores" | head -1 | awk -F: '{print $2}' | tr -d ' ')
GPU_CORES=${GPU_CORES:-"unknown"}

echo "  Chip:      $CHIP"
echo "  RAM:       ${RAM_GB} GB"
echo "  GPU cores: $GPU_CORES"

# Select tier
if [[ $RAM_GB -ge 32 ]]; then
    TIER="32gb"
elif [[ $RAM_GB -ge 16 ]]; then
    TIER="16gb"
elif [[ $RAM_GB -ge 8 ]]; then
    TIER="8gb"
else
    fail "Minimum 8GB RAM required. Detected: ${RAM_GB}GB"
fi

success "Selected model tier: ${TIER}"
echo ""

# ─── Step 3: Create virtual environment ──────────────────────────────────────
info "Setting up virtual environment at $VENV_DIR..."

mkdir -p "$ASK_HOME"

if [[ -d "$VENV_DIR" ]]; then
    warn "Existing venv found. Removing and recreating..."
    rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"
success "Virtual environment created"

# ─── Step 4: Install the ask package ─────────────────────────────────────────
info "Installing ask and dependencies (this may take a few minutes)..."

"$VENV_DIR/bin/pip" install --upgrade pip --quiet
"$VENV_DIR/bin/pip" install "$SCRIPT_DIR" --quiet 2>&1 | tail -1

# Workaround: mlx-lm declares transformers>=5.0, but Qwen2.5-VL model configs
# use processor class names (Qwen2VLImageProcessor) that only exist in
# transformers v4. Force downgrade after install so both text and vision work.
info "Pinning transformers compatibility (v4.x for VLM support)..."
"$VENV_DIR/bin/pip" install "transformers>=4.45.0,<5.0" --quiet 2>&1 | tail -1

success "ask package installed"
echo ""

# ─── Step 5: Discover and select models ──────────────────────────────────────
info "Discovering the best models for your hardware..."
echo ""

# Run model discovery: queries HuggingFace for the latest MLX models,
# recommends candidates that fit in memory, and lets the user confirm
# or override. Falls back to hardcoded tier profiles if offline.
if ! "$VENV_DIR/bin/python3" -m ask.model_discovery \
    --ram "$RAM_GB" \
    --tier "$TIER" \
    --chip "$CHIP" \
    --gpu-cores "$GPU_CORES" \
    --config "$CONFIG_FILE"; then
    warn "Model discovery failed. Falling back to default tier profile..."
    PROFILE_FILE="$SCRIPT_DIR/src/ask/model_profiles/tier_${TIER}.yaml"
    if [[ ! -f "$PROFILE_FILE" ]]; then
        fail "Model profile not found: $PROFILE_FILE"
    fi
    cat > "$CONFIG_FILE" <<EOF
# ask configuration -- generated by setup.sh
# Hardware: $CHIP, ${RAM_GB}GB RAM, ${GPU_CORES} GPU cores
# Tier: $TIER
# Re-run setup.sh to regenerate this file.

$(cat "$PROFILE_FILE")
EOF
    success "Config written (defaults)"
fi

echo ""

# ─── Step 6: Download classifier model ───────────────────────────────────────
CLASSIFIER_MODEL=$(grep -A1 "^classifier:" "$CONFIG_FILE" | grep "model:" | awk '{print $2}')

info "Downloading classifier model ($CLASSIFIER_MODEL)..."
info "This is a one-time ~2GB download. Specialist models download on first use."
echo ""

"$VENV_DIR/bin/python3" -c "
from huggingface_hub import snapshot_download
snapshot_download('$CLASSIFIER_MODEL')
print('Download complete.')
"

success "Classifier model ready"
echo ""

# ─── Step 7: Add to PATH ─────────────────────────────────────────────────────
info "Setting up the 'ask' command..."

ASK_BIN="$VENV_DIR/bin/ask"

# Detect shell config file
SHELL_NAME=$(basename "$SHELL")
if [[ "$SHELL_NAME" == "zsh" ]]; then
    SHELL_RC="$HOME/.zshrc"
elif [[ "$SHELL_NAME" == "bash" ]]; then
    SHELL_RC="$HOME/.bashrc"
else
    SHELL_RC="$HOME/.profile"
fi

# Add alias if not already present
if ! grep -q "alias ask=" "$SHELL_RC" 2>/dev/null; then
    echo "" >> "$SHELL_RC"
    echo "# ask -- local AI assistant (added by ask/setup.sh)" >> "$SHELL_RC"
    echo "alias ask='$ASK_BIN'" >> "$SHELL_RC"
    success "Added 'ask' alias to $SHELL_RC"
    warn "Run 'source $SHELL_RC' or open a new terminal to use it."
else
    # Update the existing alias
    if [[ "$OS" == "Darwin" ]]; then
        sed -i '' "s|alias ask=.*|alias ask='$ASK_BIN'|" "$SHELL_RC"
    else
        sed -i "s|alias ask=.*|alias ask='$ASK_BIN'|" "$SHELL_RC"
    fi
    success "Updated existing 'ask' alias in $SHELL_RC"
fi

echo ""

# ─── Step 8: Smoke test ──────────────────────────────────────────────────────
info "Running smoke test..."

if "$ASK_BIN" --version &>/dev/null; then
    VERSION=$("$ASK_BIN" --version)
    success "Smoke test passed ($VERSION)"
else
    warn "Smoke test failed, but installation may still work. Try: ask --version"
fi

echo ""

# ─── Done ─────────────────────────────────────────────────────────────────────
echo -e "${BOLD}${GREEN}Setup complete!${NC}"
echo ""
echo "  Config:   $CONFIG_FILE"
echo "  Venv:     $VENV_DIR"
echo "  Tier:     $TIER ($CHIP, ${RAM_GB}GB)"
echo ""
echo "Usage:"
echo "  ask \"What is a closure in JavaScript?\"        # auto-routed"
echo "  ask --model code \"Write a binary search\"      # force coding model"
echo "  ask --image photo.png \"What's in this?\"       # vision model"
echo "  ask --audio meeting.wav                        # transcribe audio"
echo "  ask -i                                         # interactive mode"
echo ""
echo "Specialist models will download on first use (~4-6GB each)."
echo ""
