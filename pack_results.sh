#!/usr/bin/env bash
# ==============================================================================
# One-Click Result Packaging Script for TF-DualMethyl
#
# Usage (from project root or model_plus/):
#   bash pack_results.sh                # Refresh any missing tables/plots & pack into .zip
#   bash pack_results.sh --only-pack    # Immediately pack existing results/figures into .zip
#   bash pack_results.sh --light-only   # Only pack lightweight zip (no .pth weights, ~15-30 MB)
#   bash pack_results.sh --all-weights  # Include .pth weights for all 9 models
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate 2>/dev/null || source ../.venv/bin/activate 2>/dev/null || true

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

export PYTHONUNBUFFERED=1
"$PYTHON_BIN" -u "$SCRIPT_DIR/pack_results.py" "$@"

