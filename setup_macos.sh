#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

RECREATE_VENV=0
SETUP_ONLY=0
RUN_APP=1
CAMERA_INDEX=""
FRAME_WIDTH=640
FRAME_HEIGHT=480
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recreate-venv)
      RECREATE_VENV=1
      shift
      ;;
    --setup-only)
      SETUP_ONLY=1
      RUN_APP=0
      shift
      ;;
    --run-app)
      RUN_APP=1
      shift
      ;;
    --camera-index)
      CAMERA_INDEX="$2"
      shift 2
      ;;
    --frame-width)
      FRAME_WIDTH="$2"
      shift 2
      ;;
    --frame-height)
      FRAME_HEIGHT="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[Setup] Python not found: $PYTHON_BIN"
  echo "[Setup] Install Python 3.12 (recommended) using Homebrew: brew install python@3.12"
  exit 1
fi

PY_MINOR="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PY_MINOR" != "3.10" && "$PY_MINOR" != "3.11" && "$PY_MINOR" != "3.12" ]]; then
  echo "[Setup] Unsupported Python version: $PY_MINOR"
  echo "[Setup] Use Python 3.10-3.12. Recommended: 3.12"
  exit 1
fi

if [[ $RECREATE_VENV -eq 1 && -d ".venv" ]]; then
  rm -rf .venv
fi

if [[ ! -f ".venv/bin/python" ]]; then
  echo "[Setup] Creating virtual environment with $PYTHON_BIN..."
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate

echo "[Setup] Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "[Setup] Environment is ready."

if [[ $SETUP_ONLY -eq 1 ]]; then
  echo "[Setup] Setup-only mode complete."
  echo "[Setup] Run: source .venv/bin/activate && python body_framing_guidance/main.py"
  exit 0
fi

if [[ $RUN_APP -eq 1 ]]; then
  APP_ARGS=("body_framing_guidance/main.py" "--frame-width" "$FRAME_WIDTH" "--frame-height" "$FRAME_HEIGHT")
  if [[ -n "$CAMERA_INDEX" ]]; then
    APP_ARGS+=("--camera-index" "$CAMERA_INDEX")
  fi

  echo "[Setup] Starting app..."
  python "${APP_ARGS[@]}"
fi
