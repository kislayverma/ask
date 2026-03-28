"""Detect Apple Silicon hardware and select the appropriate model tier."""

import platform
import subprocess
import sys


def _run(cmd: list[str]) -> str:
    """Run a command and return stripped stdout, or empty string on failure."""
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def detect_hardware() -> dict:
    """Return a dict describing the current machine's hardware.

    Keys: chip, ram_gb, gpu_cores, os, arch, tier
    """
    info: dict = {
        "os": platform.system(),
        "arch": platform.machine(),
        "chip": "unknown",
        "ram_gb": 0,
        "gpu_cores": 0,
        "tier": "unsupported",
    }

    if info["os"] != "Darwin" or info["arch"] != "arm64":
        return info

    # Chip name (e.g. "Apple M4")
    info["chip"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "unknown"

    # Total RAM in bytes → GB
    raw_mem = _run(["sysctl", "-n", "hw.memsize"])
    if raw_mem.isdigit():
        info["ram_gb"] = int(raw_mem) // (1024 ** 3)

    # GPU core count
    sp_output = _run(["system_profiler", "SPDisplaysDataType"])
    for line in sp_output.splitlines():
        if "Total Number of Cores" in line:
            parts = line.split(":")
            if len(parts) == 2 and parts[1].strip().isdigit():
                info["gpu_cores"] = int(parts[1].strip())
            break

    # Select tier based on RAM
    ram = info["ram_gb"]
    if ram >= 32:
        info["tier"] = "32gb"
    elif ram >= 16:
        info["tier"] = "16gb"
    elif ram >= 8:
        info["tier"] = "8gb"
    else:
        info["tier"] = "unsupported"

    return info


def print_hardware_summary(info: dict) -> None:
    """Print a human-readable hardware summary to stderr."""
    print(f"  Chip:      {info['chip']}", file=sys.stderr)
    print(f"  RAM:       {info['ram_gb']} GB", file=sys.stderr)
    print(f"  GPU cores: {info['gpu_cores']}", file=sys.stderr)
    print(f"  Tier:      {info['tier']}", file=sys.stderr)
